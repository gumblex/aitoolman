"""扫描 LoopDetector 相似度阈值，权衡循环检出能力与正常文本开销。

覆盖 loop_length 300/400/500 与六种循环串长度比例在 10 轮内的检出情况；
同时测量 docs/quick_start.md 全文更新耗时、误报与最高得分，以及模板型
文本、随机文本的误报情况。
"""

import os
import sys
import random
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aitoolman.loop_detect import LoopDetector


def pattern(length):
    """生成长度为 length 且最小周期为 length 的宽字符周期串。"""
    return ''.join(chr(0x4e00 + i) for i in range(length))


def detection_failures(theta):
    failures = []
    for loop_length in (300, 400, 500):
        cycle_lengths = sorted({
            loop_length // 3,
            loop_length // 2,
            int(loop_length * 0.7),
            loop_length - 10,
            loop_length - 1,
            loop_length,
        })
        for cycle_len in cycle_lengths:
            detector = LoopDetector(loop_length=loop_length, theta=theta)
            p = pattern(cycle_len)
            alarmed = False
            for _ in range(10):
                detector.update(p)
                if detector.check() or detector.has_alarmed():
                    alarmed = True
                    break
            if not alarmed:
                failures.append((loop_length, cycle_len))
    return failures


def read_text(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def run_normal(text, theta, loop_length=400):
    detector = LoopDetector(loop_length=loop_length, theta=theta)
    start = time.perf_counter()
    max_score = 0.0
    for pos in range(0, len(text), 4096):
        score = detector.update(text[pos:pos + 4096])
        max_score = max(max_score, score)
    elapsed = time.perf_counter() - start
    return (detector.has_alarmed(), max_score, elapsed,
            len(text) / elapsed if elapsed else 0.0)


def template_text():
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
    return ''.join(blocks)


def run_text_alarm(text, theta, loop_length=400):
    detector = LoopDetector(loop_length=loop_length, theta=theta)
    max_score = 0.0
    for pos in range(0, len(text), 4096):
        max_score = max(max_score, detector.update(text[pos:pos + 4096]))
    return detector.has_alarmed(), max_score


def run_random(theta, loop_length=400):
    rnd = random.Random(20240901)
    alphabet = '天地你我他,。ab cd01XY'
    detector = LoopDetector(loop_length=loop_length, theta=theta)
    max_score = 0.0
    for _ in range(20000):
        max_score = max(max_score, detector.update(rnd.choice(alphabet)))
    return detector.has_alarmed(), max_score


def test_scan():
    text = read_text('docs/quick_start.md')
    template = template_text()
    print('text_len=%d template_len=%d' % (len(text), len(template)),
          flush=True)
    thetas = [round(0.78 + 0.01 * i, 2) for i in range(19)]
    for theta in thetas:
        failures = detection_failures(theta)
        alarmed, max_score, elapsed, cps = run_normal(text, theta)
        t_alarm, t_score = run_text_alarm(template, theta)
        r_alarm, r_score = run_random(theta)
        print(
            'theta=%.2f det_fails=%d%s normal_alarm=%s max_score=%.4f '
            'elapsed=%.3fs cps=%.0f template_alarm=%s t_score=%.4f '
            'random_alarm=%s r_score=%.4f'
            % (theta, len(failures),
               (' ' + repr(failures)) if failures else '',
               alarmed, max_score, elapsed, cps,
               t_alarm, t_score, r_alarm, r_score),
            flush=True)

if __name__ == "__main__":
    test_scan()
