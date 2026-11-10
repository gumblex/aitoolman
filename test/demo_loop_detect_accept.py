#!/usr/bin/env python3
"""LoopDetector 验收脚本。

对文档与 SQLite 测试语料做端到端验收：

- 正常阶段零误报；
- 注入循环后在限定轮数内检出；
- 循环长度默认覆盖 [loop_length/4, loop_length] 区间，包含非整数
  实际循环长度（如 133、266、387 等）。

默认验收 docs/quick_start.md 与 docs/quick_start.en.md（30 次试验、
20 轮上限）。指定 --sqlite 目录后，对目录内文本文件逐文件验收
（1 次试验、单文件最多读取 --max-file-chars 字符），自动跳过二进制、
过短以及包含长度不小于 loop_length 重复子串的文件。可用 --max-files
与 --sample-offset 对文件列表分批采样，控制单次运行时长。

用法：
    python3 test/demo_loop_detect_accept.py
    python3 test/demo_loop_detect_accept.py --sqlite ../sqlite/test
    python3 test/demo_loop_detect_accept.py --sqlite ../sqlite/test \
        --max-files 200 --sample-offset 0

退出码：全部通过为 0，否则为 1。
"""

import argparse
import json
import os
import random
import statistics
import sys
from typing import List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
for path in (_ROOT, _HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from demo_loop_detect import (  # noqa: E402
    _collect_files,
    _has_repeated_substring,
    _read_text_prefix,
    _run_trial,
)

_DEFAULT_DOCS = ('docs/quick_start.md', 'docs/quick_start.en.md')


def _default_cycle_lens(loop_length: int) -> List[int]:
    """默认循环长度集合：覆盖 [loop_length/4, loop_length] 区间的多个
    比例取整结果，去重升序。"""
    loop = loop_length
    candidates = {
        max(2, loop // 4),
        max(2, loop // 3),
        int(loop * 0.4),
        loop // 2,
        int(loop * 0.6),
        2 * loop // 3,
        3 * loop // 4,
        int(loop * 0.8),
        loop - 13,
        loop - 1,
        loop,
    }
    return sorted(c for c in candidates if 2 <= c <= loop)


def _parse_cycle_lens(raw: Optional[str], loop_length: int) -> List[int]:
    """解析 --cycle-lens（逗号分隔），为空时返回默认集合。"""
    if not raw:
        return _default_cycle_lens(loop_length)
    values = set()
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        values.add(int(part))
    values = sorted(v for v in values if v >= 2)
    if not values:
        raise ValueError("cycle-lens 至少需要一个不小于 2 的整数")
    return values


def _make_run_args(args: argparse.Namespace, trials: int) -> argparse.Namespace:
    """构造 _run_trial 所需的参数对象。"""
    return argparse.Namespace(
        loop_length=args.loop_length,
        theta=args.theta,
        repeat_num=args.repeat_num,
        max_cycle_rounds=args.max_cycle_rounds,
        max_file_chars=args.max_file_chars,
        skip_high_repeat=not args.no_skip_high_repeat,
        trials=trials,
        seed=args.seed,
        cycle_set="exact",
        cycle_len=0,
        quiet=args.quiet,
    )


def _fmt_rounds(rounds: List[int]) -> str:
    if not rounds:
        return "n/a"
    return "min=%g median=%g max=%g" % (
        min(rounds), statistics.median(rounds), max(rounds))


def _accept_file(path: str, cycle_lens: List[int], args: argparse.Namespace,
                 trials: int, check_high_repeat: bool,
                 ) -> Tuple[str, List[str], dict]:
    """验收单个文件。

    Returns:
        (status, 输出行, 结构化摘要)，status 为 pass/skip/fail。
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
    if check_high_repeat and not args.no_skip_high_repeat:
        if _has_repeated_substring(text, args.loop_length):
            reason = ("high-repeat: found repeated substring length >= %d"
                      % args.loop_length)
            summary["skip_reason"] = reason
            return "skipped", ["  skipped (%s)" % reason], summary

    run_args = _make_run_args(args, trials)
    lines: List[str] = []
    all_ok = True
    tested_cycles = 0
    for cycle_len in cycle_lens:
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
        tested_cycles += 1
        false_alarms = 0
        max_score = 0.0
        alarm_rounds: List[int] = []
        undetected = 0
        for i in range(trials):
            rnd = random.Random("%s|%s|%d|%d"
                                % (args.seed, path, cycle_len, i))
            false_alarm, score, alarm_round = _run_trial(
                text, cycle_len, run_args, rnd)
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
            "trials": trials,
        })
        lines.extend([
            "  cycle_len=%d" % cycle_len,
            "    trials=%d" % trials,
            "    normal: false_alarms=%d max_score=%.4f"
            % (false_alarms, max_score),
            "    loop: alarm_rounds %s undetected=%d/%d"
            % (_fmt_rounds(alarm_rounds), undetected, trials),
            "    result: %s" % ("PASS" if ok else "FAIL"),
        ])

    if tested_cycles == 0:
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
        description="LoopDetector 验收脚本：文档与 SQLite 语料零误报、"
                    "限定轮数内检出",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("paths", nargs="*", metavar="FILE",
                        help="待验收的文档或文本文件；为空时验收 "
                             "docs/quick_start.md 与 docs/quick_start.en.md")
    parser.add_argument("--sqlite", action="append", default=[],
                        metavar="DIR",
                        help="SQLite 测试目录（可多次指定），目录递归收集"
                             "其中文本文件逐文件验收")
    parser.add_argument("--loop-length", type=int, default=400,
                        help="检测器窗口参数，即预期能检测到的循环长度"
                             "（字符）")
    parser.add_argument("--theta", type=float, default=0.80,
                        help="报警阈值（与 aitoolman.loop_detect 默认值一致）")
    parser.add_argument("--repeat-num", type=int, default=3,
                        help="报警所需连续热窗口次数（与"
                             " aitoolman.loop_detect 默认值一致）")
    parser.add_argument("--cycle-lens", default=None,
                        help="逗号分隔的循环长度列表；默认由多个比例取整"
                             "覆盖 [loop_length/4, loop_length] 区间")
    parser.add_argument("--docs-trials", type=int, default=30,
                        help="文档模式每循环长度的试验次数")
    parser.add_argument("--sqlite-trials", type=int, default=1,
                        help="SQLite 模式每文件每循环长度的试验次数")
    parser.add_argument("--max-cycle-rounds", type=int, default=20,
                        help="循环阶段报警轮数上限")
    parser.add_argument("--max-file-chars", type=int, default=20000,
                        help="单文件读取的最大字符数")
    parser.add_argument("--max-files", type=int, default=None,
                        help="SQLite 模式下最多验收的文件数（分批采样）")
    parser.add_argument("--sample-offset", type=int, default=0,
                        help="SQLite 模式下文件列表的起始偏移（分批采样）")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子（与路径、循环长度、试验序号共同"
                             "派生随机数，保证可复现）")
    parser.add_argument("--no-skip-high-repeat", action="store_true",
                        help="不跳过包含长度 >= loop_length 重复子串的"
                             "SQLite 文件")
    parser.add_argument("--json-summary", default=None, metavar="PATH",
                        help="将逐文件与整体汇总写入 JSON 文件")
    parser.add_argument("--quiet", action="store_true",
                        help="只输出失败与跳过文件及最终汇总")
    args = parser.parse_args(argv)

    try:
        cycle_lens = _parse_cycle_lens(args.cycle_lens, args.loop_length)
    except ValueError as e:
        parser.error(str(e))
        return 2

    doc_paths = list(args.paths) if args.paths else list(_DEFAULT_DOCS)

    docs = []
    docs_ok = True
    docs_tested = 0
    docs_failed = 0
    for path in doc_paths:
        status, lines, summary = _accept_file(
            path, cycle_lens, args, args.docs_trials,
            check_high_repeat=False)
        docs.append(summary)
        if status == "skipped":
            continue
        docs_tested += 1
        if status == "fail":
            docs_ok = False
            docs_failed += 1
        if not (args.quiet and status == "pass"):
            print("== %s ==" % path)
            for line in lines:
                print(line)

    sqlite = []
    sqlite_ok = True
    sqlite_tested = 0
    sqlite_skipped = 0
    sqlite_failed = 0
    for directory in args.sqlite:
        files = _collect_files([directory])
        files = files[args.sample_offset:]
        if args.max_files is not None:
            files = files[:args.max_files]
        for path in files:
            status, lines, summary = _accept_file(
                path, cycle_lens, args, args.sqlite_trials,
                check_high_repeat=True)
            sqlite.append(summary)
            if status == "skipped":
                sqlite_skipped += 1
            else:
                sqlite_tested += 1
                if status == "fail":
                    sqlite_ok = False
                    sqlite_failed += 1
            if not (args.quiet and status == "pass"):
                print("== %s ==" % path)
                for line in lines:
                    print(line)

    overall_ok = docs_ok and sqlite_ok
    print("== summary ==")
    print("docs_tested=%d docs_failed=%d"
          % (docs_tested, docs_failed))
    print("sqlite_tested=%d sqlite_skipped=%d sqlite_failed=%d"
          % (sqlite_tested, sqlite_skipped, sqlite_failed))
    print("overall: %s" % ("PASS" if overall_ok else "FAIL"))

    if args.json_summary:
        data = {
            "docs": docs,
            "sqlite": sqlite,
            "summary": {
                "docs_tested": docs_tested,
                "docs_failed": docs_failed,
                "sqlite_tested": sqlite_tested,
                "sqlite_skipped": sqlite_skipped,
                "sqlite_failed": sqlite_failed,
                "overall": "PASS" if overall_ok else "FAIL",
            },
        }
        with open(args.json_summary, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
