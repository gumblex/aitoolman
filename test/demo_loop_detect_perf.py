"""性能探针：测量周期校验与模板型不报警路径的耗时。"""
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aitoolman.loop_detect import LoopDetector


def _template_grams(length=800):
    flags = ('', '-', '+', '0', ' ', '#')
    blocks = []
    for i in range(40):
        blocks.append(
            '    do_test printf-1.%d.%d {\n'
            '      sqlite3_mprintf_int {abc: (%%%s%dd) (%%%s%dx) :xyz}\\\n'
            '           999999999 999999999 999999999\n'
            '    } {abc: (999999999) (3b9ac9ff) (7346544777) :xyz}  ;# %d\n'
            % (i // 10, i % 10,
               flags[i % 6], 5 + i % 3,
               flags[(i // 6) % 6], 6 + (i // 3) % 3,
               i))
    text = ''.join(blocks)
    return list(LoopDetector._iter_ngrams(text))[:length]


def _periodic_grams(length=800, period=200):
    base = [chr(0x4e00 + i) * 2 for i in range(period)]
    return [base[i % period] for i in range(length)]


def _measure(fn, repeat=1):
    start = time.perf_counter()
    for _ in range(repeat):
        r = fn()
    elapsed = time.perf_counter() - start
    print("elapsed=%.6fs repeat=%d result=%s" % (elapsed, repeat, r), flush=True)
    return elapsed


def test_probe():
    from collections import deque
    n = 800
    periodic_list = _periodic_grams(n, 200)
    template_list = _template_grams(n)
    periodic_deque = deque(periodic_list)
    template_deque = deque(template_list)

    print("\n[periodic list]", flush=True)
    _measure(lambda: LoopDetector._has_period(periodic_list), repeat=3)
    print("[periodic deque]", flush=True)
    _measure(lambda: LoopDetector._has_period(periodic_deque), repeat=3)
    print("[template list]", flush=True)
    _measure(lambda: LoopDetector._has_period(template_list), repeat=3)
    print("[template deque]", flush=True)
    _measure(lambda: LoopDetector._has_period(template_deque), repeat=3)

    # 模板型文本 update：统计 _has_period 调用次数与总耗时
    flags = ('', '-', '+', '0', ' ', '#')
    blocks = []
    for i in range(40):
        blocks.append(
            '    do_test printf-1.%d.%d {\n'
            '      sqlite3_mprintf_int {abc: (%%%s%dd) (%%%s%dx) :xyz}\\\n'
            '           999999999 999999999 999999999\n'
            '    } {abc: (999999999) (3b9ac9ff) (7346544777) :xyz}  ;# %d\n'
            % (i // 10, i % 10,
               flags[i % 6], 5 + i % 3,
               flags[(i // 6) % 6], 6 + (i // 3) % 3,
               i))
    text = ''.join(blocks)
    det = LoopDetector(loop_length=400)
    calls = [0]
    durations = [0.0]
    orig = LoopDetector._has_period

    def wrapped(q):
        calls[0] += 1
        t0 = time.perf_counter()
        r = orig(q)
        durations[0] += time.perf_counter() - t0
        return r

    LoopDetector._has_period = staticmethod(wrapped)
    try:
        start = time.perf_counter()
        for pos in range(0, len(text), 256):
            det.update(text[pos:pos + 256])
        total = time.perf_counter() - start
    finally:
        LoopDetector._has_period = staticmethod(orig)

    print("\n[template update] chars=%d total=%.6fs period_calls=%d period_time=%.6fs"
          % (len(text), total, calls[0], durations[0]), flush=True)
    print("[template update] score=%.4f alarmed=%s" % (
        det.score(), det.has_alarmed()), flush=True)

    # 正常流式文本 update 吞吐（较小规模，避免超时）
    normal = (
        "在本项目中，流式输出会以增量片段的形式到达，每次 update 可能只包含"
        "一个字符或一小段文本。检测器需要在窗口未装满时快速缓存，并在装满后"
        "对每个新的 n-gram 做常数时间的增量统计更新。The quick brown fox "
        "jumps over the lazy dog. 0123456789 abcdefghij.\n"
    ) * 200
    det2 = LoopDetector(loop_length=400)
    start = time.perf_counter()
    for pos in range(0, len(normal), 64):
        det2.update(normal[pos:pos + 64])
    normal_time = time.perf_counter() - start
    print("\n[normal update] chars=%d time=%.6fs chars/s=%.0f" % (
        len(normal), normal_time, len(normal) / normal_time), flush=True)

if __name__ == "__main__":
    test_probe()
