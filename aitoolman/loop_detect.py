import math
from collections import Counter, deque
from itertools import islice


class LoopDetector:
    """LLM 流式输出死循环检测器（bigram 词频 + 余弦相似度）。

    原理：维护一个 2*window 个 bigram 的滑动窗口，前一半记为 A、后一半记为 B，
    分别统计频次，以两段频次向量的余弦相似度作为得分。模型陷入死循环时，
    当前内容与一个窗口之前的内容几乎一致，得分会持续逼近 1。
    得分 > theta 的窗口称为“热”窗口，连续 3 个热窗口即判定死循环。

    接口：
        update(s)  输入任意长度文本，内部切成 2 字符 bigram 批量更新，返回当前得分
        check()    连续 3 个完整窗口得分超过阈值时返回 True（报警）
    """

    __slots__ = ("window", "theta", "_t2", "q", "cntA", "cntB",
                 "dot", "normA", "normB", "warm", "_score", "_hot", "_last")

    def __init__(self, loop_length: int = 400, theta: float = 0.9912):
        if loop_length < 2:
            raise ValueError("loop_length must be >= 2")
        self.window = loop_length//2  # 半个窗口长度（A、B 各含 loop_length//2 个 bigram）
        self.theta = theta          # 报警阈值
        self._t2 = theta * theta    # 阈值平方：热路径用平方比较，免开方
        self.q = deque()            # 窗口内的 bigram 序列（满时长度 loop_length）
        self.cntA = Counter()       # A 段频次表
        self.cntB = Counter()       # B 段频次表
        self.dot = 0                # Σ cntA[t] * cntB[t]
        self.normA = 0              # Σ cntA[t]^2
        self.normB = 0              # Σ cntB[t]^2
        self.warm = False           # 窗口是否已装满
        self._score = 0.0           # 最近一个完整窗口的得分
        self._hot = 0               # 连续超过阈值的窗口数
        self._last = ""             # 上一段文本末字符，用于跨 chunk 拼 bigram

    # ----------------------------------------------------------------------
    def update(self, s: str) -> float:
        """送入一段文本（任意长度），切成 2 字符 bigram 批量更新滑动窗口，
        返回当前得分；窗口未装满前返回 0.0。"""
        last = self._last
        if last:
            s = last + s                      # 跨 chunk 桥接出 1 个 bigram
        if len(s) < 2:
            if s:
                self._last = s                # 不足 2 字符，留到下一段再拼
            return self._score
        self._last = s[-1]

        W = self.window
        q = self.q
        # 纯 C 级地依次产出 s 的全部 2 字符 bigram（惰性，无中间大列表）
        it = map(str.__add__, s, s[1:])

        if not self.warm:
            target = W + W
            q.extend(islice(it, target - len(q)))         # 先把窗口灌满
            if len(q) != target:
                return self._score                        # 本段仍不足以装满

            # 装满后初始化 A/B 频次、范数与点积（Counter 计数走 C 实现）
            cntA = self.cntA = Counter(islice(q, W))
            cntB = self.cntB = Counter(islice(q, W, target))
            normA = sum(a * a for a in cntA.values())
            normB = sum(b * b for b in cntB.values())
            dot = 0
            for t, b in cntB.items():
                a = cntA.get(t, 0)
                if a:
                    dot += a * b
            self.normA = normA
            self.normB = normB
            self.dot = dot
            self.warm = True
            score = self._score = dot / math.sqrt(normA * normB)
            self._hot = 1 if score > self.theta else 0

        # ---------------- 热路径：每个 bigram 仅 O(1) 增量更新 ----------------
        cntA = self.cntA
        cntB = self.cntB
        cAget = cntA.get
        cBget = cntB.get
        dot = self.dot
        normA = self.normA
        normB = self.normB
        t2 = self._t2
        hot = self._hot
        mid_idx = W - 1           # popleft 之后，原 B 段首元素位于下标 W-1
        popleft = q.popleft
        append = q.append

        for t in it:
            old = popleft()       # 滑出 A 段最左的 bigram
            mid = q[mid_idx]      # 原 B 段第一个 bigram（即将移入 A 段）

            # 从 A 段删除 old
            a = cntA[old]
            dot -= cBget(old, 0)
            normA -= a + a - 1
            if a == 1:
                del cntA[old]
            else:
                cntA[old] = a - 1

            # 从 B 段删除 mid，同时把它计入 A 段
            b = cntB[mid]
            a2 = cAget(mid, 0)
            dot += b - 1 - a2     # Δdot = cntB[mid](删后) - cntA[mid]
            normB -= b + b - 1
            if b == 1:
                del cntB[mid]
            else:
                cntB[mid] = b - 1
            normA += a2 + a2 + 1
            cntA[mid] = a2 + 1

            # 新 bigram t 落入 B 段
            append(t)
            b2 = cBget(t, 0)
            dot += cAget(t, 0)
            normB += b2 + b2 + 1
            cntB[t] = b2 + 1

            # 平方比较代替开方：dot^2 > theta^2*normA*normB 等价于得分超过阈值
            if dot * dot > t2 * normA * normB:
                hot += 1
            else:
                hot = 0

        self.dot = dot
        self.normA = normA
        self.normB = normB
        self._hot = hot
        if normA and normB:
            score = dot / math.sqrt(normA * normB)
        else:
            score = 0.0
        self._score = score
        return score

    # ----------------------------------------------------------------------
    def check(self) -> bool:
        """连续 3 个完整窗口的得分都超过阈值时返回 True（报警）。"""
        return self._hot >= 3

    def score(self) -> float:
        """当前得分（最近一个完整窗口的 A/B 余弦相似度）。"""
        dot = self.dot
        if dot and self.normA and self.normB:
            return dot / math.sqrt(self.normA * self.normB)
        return 0.0


if __name__ == "__main__":
    import random

    rnd = random.Random(7)
    d = LoopDetector()
    for _ in range(2000):
        d.update(rnd.choice("天地你我他,。"))       # 逐字符送入正常文本（测试跨块桥接）
    print("normal stream -> score=%.3f alarm=%s" % (d.score(), d.check()))

    d = LoopDetector()
    for _ in range(400):
        d.update("d = LoopDetector(window=32, theta=0.85)")                        # 周期性重复文本
    print("looping stream -> score=%.3f alarm=%s" % (d.score(), d.check()))
