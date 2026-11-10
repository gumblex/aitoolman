import math
from collections import Counter, deque
from itertools import accumulate, islice


class LoopDetector:
    """LLM 流式输出死循环检测器（混合 n-gram 词频 + 余弦相似度）。

    n-gram 粒度由起始字符决定：码点 < 0x800 的字符取 3 字符，其余（含
    宽字符与类别边界）取 2 字符。每个字符位置恰产出一个 n-gram。

    loop_length 表示预期能检测到的循环长度（单位为字符）。检测器为
    A、B 两段各维护 loop_length 个 n-gram，完整滑动窗口包含
    2*loop_length 个 n-gram，分别统计两段频次，以两段频次向量的余弦
    相似度作为得分。模型陷入死循环时，当前内容与一段窗口之前的内容
    几乎一致，得分会持续逼近 1。得分 > theta 的窗口称为“热”窗口，
    连续 repeat_num 个热窗口即判定死循环。

    流式输出中出现 A、B 两窗口均未见过的新 n-gram 时视为循环被打断，
    连续热窗口计数清零（滑动窗口与频次状态不受影响）。

    报警前置周期校验：频次向量相似但序列不重复的文本（如结构雷同的
    模板段落）得分也可能持续超过阈值，但其窗口内不存在不超过窗口半
    长的重复周期；真实循环的窗口序列以循环周期为位移逐位置一致。连
    续热窗口数达到 repeat_num 时先校验窗口序列的最小周期，未通过则
    不计报警并重新累计连续热窗口计数。

    接口：
        update(s)      输入任意长度文本，返回当前得分
        check()        连续 repeat_num 个完整窗口得分超过阈值时返回 True（报警）
        has_alarmed()  本次检测过程中是否曾报警（粘性标记）
        score()        当前得分
    """

    _PERIOD_COOL = 2

    __slots__ = ("window", "theta", "_t2", "repeat_num", "q", "cntA", "cntB",
                 "dot", "normA", "normB", "warm", "_score", "_hot", "_alarmed",
                 "_pending", "_period_cool")

    def __init__(self, loop_length: int = 400, theta: float = 0.80,
                 repeat_num: int = 3):
        """初始化检测器。

        Args:
            loop_length: 预期能检测到的循环长度（字符），必须 >= 2。
            theta: 热窗口的余弦相似度阈值。
            repeat_num: 判定报警所需的连续热窗口次数，必须 >= 1。
        """
        if loop_length < 2:
            raise ValueError("loop_length must be >= 2")
        if repeat_num < 1:
            raise ValueError("repeat_num must be >= 1")
        self.window = loop_length  # A、B 各含 loop_length 个 n-gram，完整队列为 2*loop_length 个
        self.theta = theta          # 报警阈值
        self.repeat_num = repeat_num  # 连续热窗口报警次数
        self._t2 = theta * theta    # 阈值平方：热路径用平方比较，免开方
        self.q = deque()            # 窗口内的 n-gram 序列（满时长度 2*loop_length）
        self.cntA = Counter()       # A 段频次表
        self.cntB = Counter()       # B 段频次表
        self.dot = 0                # Σ cntA[t] * cntB[t]
        self.normA = 0              # Σ cntA[t]^2
        self.normB = 0              # Σ cntB[t]^2
        self.warm = False           # 窗口是否已装满
        self._score = 0.0           # 最近一个完整窗口的得分
        self._hot = 0               # 连续超过阈值的窗口数
        self._alarmed = False       # 本次检测是否曾报警（粘性）
        self._pending = ""          # 跨 chunk 桥接的未处理尾部字符（至多 2 个）
        self._period_cool = 0       # 周期校验冷却计数：>0 时跳过校验并递减

    # ----------------------------------------------------------------------
    @staticmethod
    def _iter_ngrams(s: str):
        """按混合 n-gram 规则惰性切分文本。

        每个字符位置产出一个 n-gram：起始字符码点 < 0x800 取 3 字符，
        否则取 2 字符；遇到剩余字符不足以构成完整 n-gram 的位置即停止
        （其后位置必然也不完整），剩余字符由调用方桥接到下一段。
        """
        i = 0
        n = len(s)
        while i < n:
            if ord(s[i]) < 0x800:
                if i + 3 > n:
                    break
                yield s[i:i + 3]
            else:
                if i + 2 > n:
                    break
                yield s[i:i + 2]
            i += 1

    @staticmethod
    def _has_period(parts) -> bool:
        """检查窗口序列是否存在不超过窗口半长的边界周期（自对齐）。

        将 n-gram 序列拼为字符流并记录各元素的起始偏移，元素级周期 p 对应
        字符偏移 start 满足 s[start:] == s[:len(s)-start]；用内建
        str.startswith(s[:len(s)-start], start) 借助 C 实现逐段比较，
        p <= n // 2 说明序列存在半窗内的周期，即报警所需的逐位置自对齐。
        真实循环的窗口来自长度不超过窗口半长的循环串，必然满足；随机与
        模板型文本的最小周期通常大于半窗，从而抑制误报。
        """
        n = len(parts)
        if n < 4:
            return True
        s = "".join(parts)
        size = len(s)
        offsets = list(accumulate(map(len, parts), initial=0))
        first = s[0]
        for p in range(1, n // 2 + 1):
            start = offsets[p]
            if s[start] == first and s.startswith(s[:size - start], start):
                return True
        return False

    # ----------------------------------------------------------------------
    def update(self, s: str) -> float:
        """送入一段文本（任意长度），按混合 n-gram 规则批量更新滑动窗口，
        返回当前得分；窗口未装满前返回 0.0。
        出现 A、B 两窗口均未见过的新 n-gram 时清零连续热窗口计数；
        内部连续热窗口达到 repeat_num 且窗口序列存在不超过半窗的周期时
        置位报警标记，周期校验未通过则重新累计连续热窗口计数。"""
        if self._pending:
            s = self._pending + s          # 跨 chunk 拼接上一段未处理的尾部
        self._pending = ""
        if len(s) < 2:
            if s:
                self._pending = s    # 不足 2 字符，留到下一段再拼
            return self._score

        it = self._iter_ngrams(s)
        produced = 0                 # 已产出 n-gram 数（每个产出对应一个起始字符位置）

        if not self.warm:
            target = 2 * self.window
            filled = len(self.q)
            self.q.extend(islice(it, target - filled))   # 先把窗口灌满
            produced = len(self.q) - filled
            if len(self.q) != target:
                self._pending = s[produced:]        # 本段仍不足以装满
                return self._score                  # 本段仍不足以装满

            # 装满后初始化 A/B 频次、范数与点积（Counter 计数走 C 实现）
            self.cntA = Counter(islice(self.q, self.window))
            self.cntB = Counter(islice(self.q, self.window, target))
            self.normA = sum(a * a for a in self.cntA.values())
            self.normB = sum(b * b for b in self.cntB.values())
            self.dot = 0
            for t, b in self.cntB.items():
                a = self.cntA.get(t, 0)
                if a:
                    self.dot += a * b
            self.warm = True
            self._score = self.dot / math.sqrt(self.normA * self.normB)
            self._hot = 1 if self._score > self.theta else 0
            if self._hot >= self.repeat_num:
                if self._period_cool > 0:
                    self._period_cool -= 1
                    self._hot = self.repeat_num - 1
                elif self._has_period(self.q):
                    self._alarmed = True
                else:
                    self._hot = 0
                    self._period_cool = max(self._PERIOD_COOL, self.repeat_num)

        # ---------------- 热路径：每个 n-gram 仅 O(1) 增量更新 ----------------
        consumed = 0
        for t in it:
            consumed += 1
            # A、B 两窗口均未见过的新 n-gram 说明循环被打断，清零连续热窗口计数
            seen_new = False
            if t not in self.cntA and t not in self.cntB:
                self._hot = 0
                self._period_cool = 0
                seen_new = True

            old = self.q.popleft()       # 滑出 A 段最左的 n-gram
            mid = self.q[self.window - 1]      # 原 B 段第一个 n-gram（即将移入 A 段）

            # 从 A 段删除 old
            a = self.cntA[old]
            self.dot -= self.cntB.get(old, 0)
            self.normA -= a + a - 1
            if a == 1:
                del self.cntA[old]
            else:
                self.cntA[old] = a - 1

            # 从 B 段删除 mid，同时把它计入 A 段
            b = self.cntB[mid]
            a2 = self.cntA.get(mid, 0)
            self.dot += b - 1 - a2     # Δdot = cntB[mid](删后) - cntA[mid]
            self.normB -= b + b - 1
            if b == 1:
                del self.cntB[mid]
            else:
                self.cntB[mid] = b - 1
            self.normA += a2 + a2 + 1
            self.cntA[mid] = a2 + 1

            # 新 n-gram t 落入 B 段
            self.q.append(t)
            b2 = self.cntB.get(t, 0)
            self.dot += self.cntA.get(t, 0)
            self.normB += b2 + b2 + 1
            self.cntB[t] = b2 + 1

            # 平方比较代替开方：dot^2 > theta^2*normA*normB 等价于得分超过阈值
            if not seen_new and self.dot * self.dot > self._t2 * self.normA * self.normB:
                self._hot += 1
                if self._hot >= self.repeat_num:
                    if self._period_cool > 0:
                        self._period_cool -= 1
                        self._hot = self.repeat_num - 1
                    elif self._has_period(self.q):
                        self._alarmed = True
                    else:
                        self._hot = 0   # 窗口序列未自对齐：抑制本轮报警并重新累计
                        self._period_cool = max(self._PERIOD_COOL, self.repeat_num)
            else:
                self._hot = 0

        self._pending = s[produced + consumed:]
        if self.normA and self.normB:
            self._score = self.dot / math.sqrt(self.normA * self.normB)
        else:
            self._score = 0.0
        return self._score

    # ----------------------------------------------------------------------
    def check(self) -> bool:
        """连续 repeat_num 个完整窗口的得分都超过阈值、且窗口序列存在
        不超过半窗的周期时返回 True（报警）。"""
        return self._hot >= self.repeat_num

    def has_alarmed(self) -> bool:
        """本次检测过程中是否曾出现连续热窗口报警（粘性标记）。"""
        return self._alarmed

    def score(self) -> float:
        """当前得分（最近一个完整窗口的 A/B 余弦相似度）。"""
        if self.dot and self.normA and self.normB:
            return self.dot / math.sqrt(self.normA * self.normB)
        return 0.0


if __name__ == "__main__":
    import random

    rnd = random.Random(7)
    d = LoopDetector()
    for _ in range(2000):
        d.update(rnd.choice("天地你我他,。ab cd01XY"))   # 逐字符送入正常文本（测试跨块桥接）
    print("normal stream -> score=%.3f alarm=%s" % (d.score(), d.check()))

    d = LoopDetector()
    for _ in range(400):
        d.update("d = LoopDetector(window=32, theta=0.85)")   # 周期性重复文本（ASCII 3-gram）
    print("looping stream -> score=%.3f alarm=%s" % (d.score(), d.check()))
