#!/usr/bin/env python3
"""LoopDetector 误报率/检出能力测试与调参工具。

对给定文本文件模拟 LLM 流式输出的死循环场景，检验 LoopDetector 的
误报率与检出能力，用于 theta、loop_length 调参。每个文件运行 trials
次试验，每次试验：

- 正常阶段：在文件前半部随机取截断点，将截断点之前的内容按块送入
  检测器，全程 check() 不应报警（统计误报次数与全程最高得分）；
- 循环阶段：取截断点前 cycle-len 个字符作为循环串，反复送入检测器，
  每轮随机改写循环串开头 0~mutate 个字符（改写字符取自固定可打印
  字符集），统计 check() 报警所需轮数（上限 100 轮，超过记为未检出）。

随机数由 seed+文件路径+试验序号派生，同一命令重复运行结果一致。

用法：
    python3 test/demo_loop_detect.py FILE... [--loop-length 400] \
        [--theta 0.9912] [--cycle-len 200] [--mutate 10] [--trials 30] \
        [--seed N]

FILE 可为文件或目录（目录递归收集其中所有文件）。含 NUL 字节的二进制
文件与长度不足 2*cycle-len 字符的文件跳过并计入跳过统计；超过
max-file-chars 的大文件只读前缀以控制耗时。

退出码：无失败（正常阶段零误报且循环阶段均在轮数上限内报警）为 0，
否则为 1。
"""

import argparse
import os
import random
import statistics
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aitoolman.loop_detect import LoopDetector

_FEED_CHUNK = 4096          # 正常阶段逐块送入的块大小（字符）
_MAX_LOOP_ROUNDS = 100      # 循环阶段报警轮数上限，超过记为未检出
_MUTATE_CHARS = ("abcdefghijklmnopqrstuvwxyz"
                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                 "0123456789")  # 循环串开头改写用的固定可打印字符集


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


def _mutate_head(cycle: str, k: int, rnd: random.Random) -> str:
    """将循环串开头 k 个字符随机改写为固定字符集中的字符"""
    if k <= 0:
        return cycle
    return ''.join(rnd.choice(_MUTATE_CHARS) for _ in range(k)) + cycle[k:]


def _run_trial(text: str, args: argparse.Namespace,
               rnd: random.Random) -> Tuple[bool, float, Optional[int]]:
    """运行单次试验。

    Returns:
        (正常阶段是否误报, 正常阶段最高得分, 循环阶段报警轮数，未报警为 None)
    """
    # 截断点：文件前半部随机选取（保证截断点前至少有 cycle-len 个字符）
    cut = rnd.randint(args.cycle_len, len(text) // 2)
    prefix = text[:cut]

    detector = LoopDetector(args.loop_length, args.theta)
    false_alarm = False
    max_score = 0.0
    for i in range(0, len(prefix), _FEED_CHUNK):
        detector.update(prefix[i:i + _FEED_CHUNK])
        if detector.check():
            false_alarm = True
        score = detector.score()
        if score > max_score:
            max_score = score

    # 循环阶段：截断点前 cycle-len 个字符作为循环串，反复送入
    cycle = text[cut - args.cycle_len:cut]
    alarm_round = None
    for round_no in range(1, _MAX_LOOP_ROUNDS + 1):
        k = rnd.randint(0, args.mutate)
        detector.update(_mutate_head(cycle, k, rnd))
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


def test_file(path: str, args: argparse.Namespace) -> Tuple[str, List[str]]:
    """对单个文件运行全部试验。

    Returns:
        (status, 汇总行列表)，status 为 "pass"/"fail"/"skipped"。
    """
    text, skip_reason = _read_text_prefix(path, args.max_file_chars)
    if text is None:
        return "skipped", ["  skipped (%s)" % skip_reason]
    if len(text) < 2 * args.cycle_len:
        return "skipped", [
            "  skipped (too short: %d chars < 2*cycle-len=%d)"
            % (len(text), 2 * args.cycle_len)]

    false_alarms = 0
    max_score = 0.0
    alarm_rounds: List[int] = []
    undetected = 0
    for i in range(args.trials):
        rnd = random.Random("%s|%s|%d" % (args.seed, path, i))
        false_alarm, score, alarm_round = _run_trial(text, args, rnd)
        if false_alarm:
            false_alarms += 1
        if score > max_score:
            max_score = score
        if alarm_round is None:
            undetected += 1
        else:
            alarm_rounds.append(alarm_round)

    ok = false_alarms == 0 and undetected == 0
    lines = [
        "  trials=%d" % args.trials,
        "  normal: false_alarms=%d max_score=%.4f" % (false_alarms, max_score),
        "  loop: alarm_rounds %s undetected=%d/%d"
        % (_fmt_rounds(alarm_rounds), undetected, args.trials),
        "  result: %s" % ("PASS" if ok else "FAIL"),
    ]
    return ("pass" if ok else "fail"), lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="LoopDetector 误报率/检出能力测试与调参工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("files", nargs="+", metavar="FILE",
                        help="待测试的文本文件或目录路径（目录递归收集）")
    parser.add_argument("--loop-length", type=int, default=400,
                        help="检测器循环窗口长度（bigram 数）")
    parser.add_argument("--theta", type=float, default=0.9912,
                        help="报警阈值")
    parser.add_argument("--cycle-len", type=int, default=200,
                        help="循环串长度（字符）")
    parser.add_argument("--mutate", type=int, default=10,
                        help="每轮随机改写循环串开头的最大字符数"
                             "（0~mutate，改写字符取自 ASCII 字母与数字）")
    parser.add_argument("--trials", type=int, default=30,
                        help="每文件试验次数")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子（与文件路径、试验序号共同派生随机数）")
    parser.add_argument("--max-file-chars", type=int, default=1000000,
                        help="单文件读取的最大字符数（大文件只读前缀）")
    parser.add_argument("--quiet", action="store_true",
                        help="只输出失败与跳过文件及最终汇总")
    args = parser.parse_args(argv)

    all_ok = True
    tested = 0
    skipped = 0
    for path in _collect_files(args.files):
        status, lines = test_file(path, args)
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

    print("== summary ==")
    print("tested=%d skipped=%d" % (tested, skipped))
    print("overall: %s" % ("PASS" if all_ok else "FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
