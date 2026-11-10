#!/usr/bin/env python3
"""LoopDetector 误报率/检出能力测试与调参工具。

对给定文本文件模拟 LLM 流式输出的死循环场景，检验 LoopDetector 的
误报率与检出能力，用于 theta、loop_length、repeat_num 调参。对每个
循环串长度组运行 trials 次试验，每次试验：

- 正常阶段：从文件中随机取截断点（保证截断点前至少有 cycle-len 个
  字符），将截断点之前的全部内容按块送入检测器，全程 check() 不应
  报警（统计误报次数与全程最高得分）；
- 循环阶段：取截断点前 cycle-len 个字符作为循环串，原样反复送入
  检测器（不做任何改写），统计 check() 报警所需轮数（上限
  max-cycle-rounds 轮，超过记为未检出）。

随机数由 seed+文件路径+循环串长度+试验序号派生，同一命令重复运行
结果一致。

用法：
    python3 test/demo_loop_detect.py FILE... [--loop-length 400] \
        [--theta 0.80] [--repeat-num 3] [--cycle-len 200] \
        [--cycle-set exact|all] [--trials 30] [--seed N] \
        [--skip-high-repeat] \
        [--max-cycle-rounds 10] [--max-file-chars N] [--json-summary PATH]

FILE 可为文件或目录（目录递归收集其中所有文件）。含 NUL 字节的二进制
文件跳过并计入跳过统计；长度不足 2*cycle-len 的文件跳过（--cycle-set
=all 时按组判断，全部组均过短才计为文件级跳过）；超过 max-file-chars
的大文件只读前缀以控制耗时。启用 --skip-high-repeat 时，包含长度不
小于 loop-length 的重复子串的文件也会跳过。

--cycle-set 选择循环串长度集合：exact 仅使用 --cycle-len 指定的长度
（默认）；all 枚举 {loop-length//3, loop-length//2, loop-length*0.7,
loop-length-10, loop-length-1, loop-length}（去掉小于 2 的项并去重升序）逐组运行，
输出与 JSON 汇总按循环串长度分组。

退出码：无失败（正常阶段零误报且循环阶段均在轮数上限内报警）为 0，
否则为 1。
"""

import argparse
import json
import os
import random
import statistics
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aitoolman.loop_detect import LoopDetector

_FEED_CHUNK = 4096          # 正常阶段逐块送入的块大小（字符）
_MAX_LOOP_ROUNDS = 10       # 循环阶段报警轮数默认上限，超过记为未检出


def _read_text_prefix(path: str,
                      max_chars: int) -> Tuple[Optional[str], Optional[str]]:
    """读取文件文本前缀（utf-8 容错解码）。

    Returns:
        (text, None) 读取成功，超过 max_chars 只读前缀；
        (None, reason) 跳过，reason 为 "binary" 或 "unreadable: ..."。
    """
    try:
        with open(path, 'rb') as f:
            head = f.read(8192)
    except OSError as e:
        return None, "unreadable: %s" % e
    if b'\x00' in head:
        return None, "binary"
    blocks: List[str] = []
    remaining = max_chars
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        while remaining > 0:
            block = f.read(min(4096, remaining))
            if not block:
                break
            blocks.append(block)
            remaining -= len(block)
    return ''.join(blocks), None


def _collect_files(paths: List[str]) -> List[str]:
    """展开输入路径：文件直接收集，目录递归收集其中所有文件（排序保证顺序稳定）"""
    result: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            for root, dirs, names in os.walk(path):
                dirs.sort()
                for name in sorted(names):
                    result.append(os.path.join(root, name))
        else:
            result.append(path)
    return result


def _has_repeated_substring(text: str, length: int) -> bool:
    """判断文本中是否存在长度不小于 length 的重复子串。

    使用双滚动哈希降低碰撞概率。length <= 0 或文本过短时返回 False。
    """
    if length <= 0 or len(text) < length + 1:
        return False
    mod1 = 1000000007
    mod2 = 1000000009
    base = 911382323
    power1 = 1
    power2 = 1
    for _ in range(length - 1):
        power1 = (power1 * base) % mod1
        power2 = (power2 * base) % mod2
    h1 = 0
    h2 = 0
    for i in range(length):
        c = ord(text[i])
        h1 = (h1 * base + c) % mod1
        h2 = (h2 * base + c) % mod2
    seen = {(h1, h2)}
    for i in range(length, len(text)):
        out_c = ord(text[i - length])
        in_c = ord(text[i])
        h1 = (h1 - out_c * power1) % mod1
        h1 = (h1 * base + in_c) % mod1
        h2 = (h2 - out_c * power2) % mod2
        h2 = (h2 * base + in_c) % mod2
        key = (h1, h2)
        if key in seen:
            return True
        seen.add(key)
    return False


def _cycle_lengths(args: argparse.Namespace) -> List[int]:
    """根据 --cycle-set 计算待测试的循环串长度列表（去重升序）。

    exact 模式仅返回 --cycle-len；all 模式枚举
    {loop-length//3, loop-length//2, loop-length*0.7, loop-length-10,
    loop-length-1, loop-length}，去掉小于 2 的项并去重升序。
    """
    if args.cycle_set == "exact":
        return [args.cycle_len]
    loop = args.loop_length
    candidates = {loop // 3, loop // 2, int(loop * 0.7),
                  loop - 10, loop - 1, loop}
    return sorted(c for c in candidates if c >= 2)


def _run_trial(text: str, cycle_len: int, args: argparse.Namespace,
               rnd: random.Random) -> Tuple[bool, float, Optional[int]]:
    """运行单次试验。

    Returns:
        (正常阶段是否误报, 正常阶段最高得分, 循环阶段报警轮数，未报警为 None)
    """
    # 截断点：随机选取（保证截断点前至少有 cycle_len 个字符），
    # 正常阶段送入截断点之前的全部内容
    cut = rnd.randint(cycle_len, len(text))
    prefix = text[:cut]

    detector = LoopDetector(args.loop_length, args.theta, args.repeat_num)
    false_alarm = False
    max_score = 0.0
    for i in range(0, len(prefix), _FEED_CHUNK):
        detector.update(prefix[i:i + _FEED_CHUNK])
        if detector.check():
            false_alarm = True
        score = detector.score()
        if score > max_score:
            max_score = score

    # 循环阶段：截断点前 cycle_len 个字符作为循环串，原样反复送入
    cycle = text[cut - cycle_len:cut]
    alarm_round = None
    for round_no in range(1, args.max_cycle_rounds + 1):
        detector.update(cycle)
        if detector.check():
            alarm_round = round_no
            break
    return false_alarm, max_score, alarm_round


def _fmt_rounds(rounds: List[int]) -> str:
    """报警轮数列表格式化为 min/median/max"""
    if not rounds:
        return "n/a"
    return "min=%g median=%g max=%g" % (
        min(rounds), statistics.median(rounds), max(rounds))


def test_file(path: str, args: argparse.Namespace) -> Tuple[str, List[str], dict]:
    """对单个文件按循环串长度分组运行全部试验。

    Returns:
        (status, 汇总行列表, 结构化摘要)，status 为 "pass"/"fail"/"skipped"。
    """
    summary = {
        "path": path,
        "status": "skipped",
        "skip_reason": None,
        "cycles": [],
    }
    text, skip_reason = _read_text_prefix(path, args.max_file_chars)
    if text is None:
        summary["skip_reason"] = skip_reason
        return "skipped", ["  skipped (%s)" % skip_reason], summary
    if args.skip_high_repeat:
        repeat_len = args.loop_length
        if _has_repeated_substring(text, repeat_len):
            reason = ("high-repeat: found repeated substring length >= %d"
                      % repeat_len)
            summary["skip_reason"] = reason
            return "skipped", ["  skipped (%s)" % reason], summary

    lines: List[str] = []
    all_ok = True
    any_tested = False
    for cycle_len in _cycle_lengths(args):
        if len(text) < 2 * cycle_len:
            reason = ("too short: %d chars < 2*cycle-len=%d"
                      % (len(text), 2 * cycle_len))
            summary["cycles"].append({
                "cycle_len": cycle_len,
                "status": "skipped",
                "skip_reason": reason,
                "false_alarms": None,
                "max_score": None,
                "alarm_rounds": None,
                "undetected": None,
                "trials": None,
            })
            lines.append("  cycle_len=%d skipped (%s)" % (cycle_len, reason))
            continue
        any_tested = True
        false_alarms = 0
        max_score = 0.0
        alarm_rounds: List[int] = []
        undetected = 0
        for i in range(args.trials):
            rnd = random.Random("%s|%s|%d|%d"
                                % (args.seed, path, cycle_len, i))
            false_alarm, score, alarm_round = _run_trial(
                text, cycle_len, args, rnd)
            if false_alarm:
                false_alarms += 1
            if score > max_score:
                max_score = score
            if alarm_round is None:
                undetected += 1
            else:
                alarm_rounds.append(alarm_round)

        ok = false_alarms == 0 and undetected == 0
        if not ok:
            all_ok = False
        summary["cycles"].append({
            "cycle_len": cycle_len,
            "status": "pass" if ok else "fail",
            "skip_reason": None,
            "false_alarms": false_alarms,
            "max_score": max_score,
            "alarm_rounds": alarm_rounds,
            "undetected": undetected,
            "trials": args.trials,
        })
        lines.extend([
            "  cycle_len=%d" % cycle_len,
            "    trials=%d" % args.trials,
            "    normal: false_alarms=%d max_score=%.4f"
            % (false_alarms, max_score),
            "    loop: alarm_rounds %s undetected=%d/%d"
            % (_fmt_rounds(alarm_rounds), undetected, args.trials),
            "    result: %s" % ("PASS" if ok else "FAIL"),
        ])

    if not any_tested:
        reasons = "; ".join(c["skip_reason"] for c in summary["cycles"]
                            if c["skip_reason"])
        summary["skip_reason"] = reasons
        return "skipped", ["  skipped (%s)" % reasons], summary

    status = "pass" if all_ok else "fail"
    summary["status"] = status
    lines.append("  result: %s" % ("PASS" if all_ok else "FAIL"))
    return status, lines, summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="LoopDetector 误报率/检出能力测试与调参工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("files", nargs="+", metavar="FILE",
                        help="待测试的文本文件或目录路径（目录递归收集）")
    parser.add_argument("--loop-length", type=int, default=400,
                        help="检测器窗口参数，即预期能检测到的循环长度（字符）")
    parser.add_argument("--theta", type=float, default=0.80,
                        help="报警阈值（与 aitoolman.loop_detect 默认值一致）")
    parser.add_argument("--repeat-num", type=int, default=3,
                        help="报警所需连续热窗口次数（与"
                             " aitoolman.loop_detect 默认值一致）")
    parser.add_argument("--cycle-len", type=int, default=200,
                        help="循环串长度（字符），--cycle-set=all 时忽略")
    parser.add_argument("--cycle-set", choices=("exact", "all"),
                        default="exact",
                        help="循环串长度集合：exact 仅用 --cycle-len；"
                             "all 枚举 loop-length//3、loop-length//2、"
                             "loop-length*0.7、loop-length-10、"
                             "loop-length-1、loop-length"
                             "（去重升序，跳过小于 2 的项）")
    parser.add_argument("--trials", type=int, default=30,
                        help="每文件每循环串长度的试验次数")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子（与文件路径、循环串长度、试验序号"
                             "共同派生随机数）")
    parser.add_argument("--max-file-chars", type=int, default=1000000,
                        help="单文件读取的最大字符数（大文件只读前缀）")
    parser.add_argument("--skip-high-repeat", action="store_true",
                        help="跳过包含长度 >= loop-length 重复子串的文件")
    parser.add_argument("--max-cycle-rounds", type=int,
                        default=_MAX_LOOP_ROUNDS,
                        help="循环阶段最大轮数")
    parser.add_argument("--json-summary", default=None, metavar="PATH",
                        help="将逐文件（按循环串长度分组）与整体汇总写入"
                             " JSON 文件")
    parser.add_argument("--quiet", action="store_true",
                        help="只输出失败与跳过文件及最终汇总")
    args = parser.parse_args(argv)

    all_ok = True
    tested = 0
    skipped = 0
    summaries = []
    for index, path in enumerate(_collect_files(args.files), 1):
        status, lines, summary = test_file(path, args)
        summaries.append(summary)
        if status == "skipped":
            skipped += 1
        else:
            tested += 1
            if status == "fail":
                all_ok = False
        if not (args.quiet and status == "pass"):
            print("== %s ==" % path)
            for line in lines:
                print(line)
        elif index % 10 == 0:
            print("progress: %d files" % index, flush=True)

    overall = "PASS" if all_ok else "FAIL"
    print("== summary ==")
    print("tested=%d skipped=%d" % (tested, skipped))
    print("overall: %s" % overall)

    if args.json_summary:
        data = {
            "files": summaries,
            "summary": {
                "tested": tested,
                "skipped": skipped,
                "overall": overall,
            },
        }
        with open(args.json_summary, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
