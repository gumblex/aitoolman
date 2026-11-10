"""LoopDetector 检测器单元测试与 provider 流式死循环检测集成测试。

检测器部分覆盖混合 n-gram 切分规则、构造校验、窗口未满行为、
repeat_num 连续热窗口阈值、报警前置周期门控、新 n-gram 清零连续
热计数、正常/周期文本报警行为、报警粘性、任意分块一致性与得分取值
范围；provider 集成部分基于项目内置 mock HTTP 上游（mock-loop 场景）
驱动真实流式路径，验证 loop_detect_length 的检测中断、默认关闭与
两级配置优先级。
"""

import asyncio
import random
import unittest

from aitoolman.channel import Channel
from aitoolman.loop_detect import LoopDetector
from aitoolman.model import FinishReason, LLMProviderRequest, Message
from aitoolman.provider import LLMProviderManager

try:
    from test.mock_http_server import start_mock_http_server
    MOCK_SERVER_AVAILABLE = True
except ImportError:
    start_mock_http_server = None
    MOCK_SERVER_AVAILABLE = False

CLIENT_TOKEN = 'loop-test-token'


class TestLoopDetector(unittest.TestCase):
    """LoopDetector 检测器行为测试。"""

    CYCLE = 'abcdefghij0123456789'
    MIXED_TEXT = '对于码点<0x800的字符'
    RANDOM_ALPHABET = '天地你我他,。ab cd01XY'

    def test_construction(self):
        """参数构造"""
        detector = LoopDetector(loop_length=100, theta=0.9, repeat_num=2)
        self.assertEqual(detector.window, 100)
        self.assertEqual(detector.theta, 0.9)
        self.assertEqual(detector.repeat_num, 2)

    def test_default_params(self):
        """默认参数：theta=0.80、repeat_num=3、window=400"""
        detector = LoopDetector()
        self.assertEqual(detector.window, 400)
        self.assertAlmostEqual(detector.theta, 0.80)
        self.assertEqual(detector.repeat_num, 3)

    def test_invalid_loop_length(self):
        """loop_length 小于 2 时构造抛出 ValueError"""
        for length in (0, 1, -400):
            with self.assertRaises(ValueError):
                LoopDetector(loop_length=length)

    def test_invalid_repeat_num(self):
        """repeat_num 小于 1 时构造抛出 ValueError"""
        for num in (0, -1):
            with self.assertRaises(ValueError):
                LoopDetector(loop_length=200, repeat_num=num)

    def test_mixed_ngram_split_rules(self):
        """混合 n-gram 切分规则逐项断言：起始字符码点 <0x800 取 3 字符，
        否则取 2 字符；每个字符位置恰产出一个 n-gram。"""
        grams = list(LoopDetector._iter_ngrams(self.MIXED_TEXT))
        self.assertEqual(grams, [
            '对于', '于码', '码点', '点<', '<0x', '0x8',
            'x80', '800', '00的', '0的字', '的字', '字符',
        ])
        # 窄字符起始取 3 字符，可跨宽字符边界；每个字符位置各产出一个 n-gram
        self.assertEqual(list(LoopDetector._iter_ngrams('a中b')),
                         ['a中b', '中b'])
        self.assertEqual(list(LoopDetector._iter_ngrams('ab<0x8中')),
                         ['ab<', 'b<0', '<0x', '0x8', 'x8中'])
        # 纯 ASCII 与纯宽字符
        self.assertEqual(list(LoopDetector._iter_ngrams('abcd')),
                         ['abc', 'bcd'])
        self.assertEqual(list(LoopDetector._iter_ngrams('天地你我')),
                         ['天地', '地你', '你我'])
        # 剩余字符不足时不产出残缺 n-gram
        self.assertEqual(list(LoopDetector._iter_ngrams('中')), [])
        self.assertEqual(list(LoopDetector._iter_ngrams('a')), [])

    def _pattern(self, length):
        """生成长度为 length 且最小周期为 length 的宽字符周期串。"""
        return ''.join(chr(0x4e00 + i) for i in range(length))

    def test_window_semantics_loop_length_400(self):
        """loop_length=400 时指定周期长度均应在有限字符内报警。"""
        for cycle_len in (400, 399, 390, 200, 133):
            detector = LoopDetector(loop_length=400, theta=0.97)
            pattern = self._pattern(cycle_len)
            total = 0
            for _ in range(100):
                detector.update(pattern)
                total += len(pattern)
                if detector.has_alarmed():
                    break
            self.assertTrue(
                detector.has_alarmed(),
                f"cycle_len={cycle_len}, total={total}")

    def test_loop_length_semantics_default_params(self):
        """默认参数下 loop_length 新语义：预期循环长度附近的周期均应在
        10 倍周期长度的字符预算内报警。"""
        for loop_length, cycle_len in ((400, 390), (400, 133), (200, 190)):
            detector = LoopDetector(loop_length=loop_length)
            pattern = self._pattern(cycle_len)
            total = 0
            for _ in range(10):
                detector.update(pattern)
                total += len(pattern)
                if detector.has_alarmed():
                    break
            self.assertTrue(
                detector.has_alarmed(),
                f"loop_length={loop_length}, cycle_len={cycle_len}")
            self.assertLessEqual(
                total, 10 * cycle_len,
                f"loop_length={loop_length}, cycle_len={cycle_len}")

    def test_repeat_num_one_alarms_on_first_hot(self):
        """repeat_num=1：窗口装满后的首个热窗口即报警。"""
        detector = LoopDetector(loop_length=200, repeat_num=1)
        pattern = self._pattern(200)
        detector.update(pattern * 2)   # 400 字符仅产出 399 个宽字符 2-gram，窗口未装满
        self.assertFalse(detector.check())
        detector.update(pattern)       # 续送一个周期装满窗口：A、B 段频次一致得分 1.0，单热即报警
        self.assertTrue(detector.check())
        self.assertTrue(detector.has_alarmed())

    def test_repeat_num_five_requires_consecutive_hot(self):
        """repeat_num=5：需累计 5 个连续热窗口才报警。"""
        detector = LoopDetector(loop_length=200, repeat_num=5)
        pattern = self._pattern(200)
        detector.update(pattern * 2)   # 窗口装满：1 个热窗口
        self.assertFalse(detector.check())
        detector.update(pattern[:4])   # 再积累 3 个热窗口：共 4 个
        self.assertFalse(detector.check())
        detector.update(pattern[4:5])  # 经桥接再积累 1 个：共 5 个
        self.assertTrue(detector.check())
        self.assertTrue(detector.has_alarmed())

    def test_period_check_cooldown_regression(self):
        """周期校验冷却回归：模板型文本触发多次校验失败后，真实周期
        文本仍能在 10 轮内报警。"""
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
        detector = LoopDetector()
        for pos in range(0, len(text), 4096):
            detector.update(text[pos:pos + 4096])
        self.assertFalse(detector.check())

        pattern = self._pattern(400)
        for _ in range(10):
            detector.update(pattern)
            if detector.check():
                break
        self.assertTrue(detector.has_alarmed())

    def test_non_integer_cycle_lengths_alarm(self):
        """非整除比例的周期串（137、211、317）在默认参数下 10 轮内报警"""
        for cycle_len in (137, 211, 317):
            detector = LoopDetector()
            pattern = self._pattern(cycle_len)
            alarmed = False
            for _ in range(10):
                detector.update(pattern)
                if detector.check() or detector.has_alarmed():
                    alarmed = True
                    break
            self.assertTrue(
                alarmed, f"cycle_len={cycle_len}")

    def test_template_like_text_no_alarm(self):
        """结构雷同但内容渐变的模板型长文本（模拟重复模板结构的代码
        段落）不触发报警：各窗口得分可能持续超过阈值，但窗口序列不
        存在自对齐，报警前置校验应抑制报警。"""
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
        detector = LoopDetector()
        max_score = 0.0
        for pos in range(0, len(text), 4096):
            max_score = max(max_score, detector.update(text[pos:pos + 4096]))
        self.assertFalse(detector.has_alarmed())
        self.assertFalse(detector.check())
        # 得分确实曾超过阈值，保证用例真实覆盖报警门控路径
        self.assertGreater(max_score, detector.theta)

    def test_dispersed_matches_no_alarm(self):
        """高匹配率但匹配分散的文本不触发报警：滑动窗口内绝大多数
        n-gram 相同，得分超过阈值，但独特 n-gram 的分布使窗口序列不
        存在不超过半窗的周期。"""
        blocks = []
        for i in range(60):
            blocks.append('甲' * 24 + chr(0x4e00 + i))
        text = ''.join(blocks)
        detector = LoopDetector(loop_length=200, repeat_num=1)
        max_score = 0.0
        for pos in range(0, len(text), 100):
            score = detector.update(text[pos:pos + 100])
            max_score = max(max_score, score)
            self.assertFalse(detector.check())
            self.assertFalse(detector.has_alarmed())
        self.assertGreater(max_score, detector.theta)

    def test_cycle_lengths_alarm_within_ten_rounds(self):
        """多种 loop_length 与循环串长度组合下，循环文本在不超过 10
        轮重复送入内触发报警。"""
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
                detector = LoopDetector(loop_length=loop_length)
                pattern = self._pattern(cycle_len)
                alarmed = False
                for _ in range(10):
                    detector.update(pattern)
                    if detector.check() or detector.has_alarmed():
                        alarmed = True
                        break
                self.assertTrue(
                    alarmed,
                    f"loop_length={loop_length}, cycle_len={cycle_len}")

    def test_cycle_after_prefix_alarms_within_budget(self):
        """正常前缀后进入循环内容：门控引入的报警延迟不得突破检出预算"""
        rnd = random.Random(2024)
        prefix = ''.join(rnd.choice(self.RANDOM_ALPHABET)
                         for _ in range(5000))
        detector = LoopDetector()
        detector.update(prefix)
        self.assertFalse(detector.has_alarmed())
        pattern = self._pattern(400)
        total = 0
        for _ in range(10):
            detector.update(pattern)
            total += len(pattern)
            if detector.check():
                break
        self.assertTrue(detector.has_alarmed())
        self.assertLessEqual(total, 10 * 400)

    def test_new_gram_resets_hot_count(self):
        """出现 A、B 两窗口均未见过的新 n-gram 时清零连续热窗口计数：
        循环被新内容打断后报警被推迟，循环恢复后仍能检出。"""
        cycle = self._pattern(200)
        detector = LoopDetector(loop_length=200, repeat_num=3)
        # 窗口装满：得分 1.0，连续热窗口 1 个，未达 repeat_num
        detector.update(cycle * 2)
        self.assertFalse(detector.check())
        # 插入全新片段：若无清零机制，插入期间连续热计数会累积到
        # repeat_num 而报警
        detector.update('全新片段甲乙丙丁')
        self.assertFalse(detector.check())
        self.assertFalse(detector.has_alarmed())
        # 循环恢复后重新累积连续热窗口，在有限字符内报警
        for _ in range(10):
            detector.update(cycle)
            if detector.check():
                break
        self.assertTrue(detector.has_alarmed())

    def test_has_alarmed_sticky(self):
        """大块输入内部出现连续热窗口后，报警标记保持为真。"""
        detector = LoopDetector(loop_length=200)
        detector.update(self.CYCLE * 100)
        self.assertTrue(detector.has_alarmed())
        detector.update('天地你我他,。' * 1000)
        self.assertTrue(detector.has_alarmed())

    def test_unfilled_window_no_alarm(self):
        """窗口未装满时不报警且得分为 0"""
        detector = LoopDetector(loop_length=400)
        for _ in range(10):  # 10*20=200 字符，不足完整窗口 2*loop_length=800 n-gram
            score = detector.update(self.CYCLE)
            self.assertEqual(score, 0.0)
            self.assertEqual(detector.score(), 0.0)
            self.assertFalse(detector.check())

    def test_random_text_no_alarm(self):
        """正常随机文本长流不触发报警"""
        rnd = random.Random(20240901)
        detector = LoopDetector()
        for _ in range(20000):
            detector.update(rnd.choice(self.RANDOM_ALPHABET))
        self.assertFalse(detector.check())
        self.assertFalse(detector.has_alarmed())
        self.assertLess(detector.score(), detector.theta)

    def test_periodic_text_alarms_within_budget(self):
        """周期重复文本在有限字符内触发报警"""
        detector = LoopDetector(loop_length=200)
        total = 0
        for _ in range(60):
            detector.update(self.CYCLE)
            total += len(self.CYCLE)
            if detector.check():
                break
        self.assertTrue(detector.check())
        self.assertLessEqual(total, 1200)

    def test_periodic_text_char_by_char_alarms(self):
        """逐字符送入的周期文本同样在有限字符内报警"""
        detector = LoopDetector(loop_length=200)
        total = 0
        for _ in range(60):
            for ch in self.CYCLE:
                detector.update(ch)
                total += 1
            if detector.check():
                break
        self.assertTrue(detector.check())
        self.assertLessEqual(total, 1200)

    def test_chunked_update_consistency(self):
        """任意分块方式送入与一次性送入的最终得分一致"""
        rnd = random.Random(777)
        text = ''.join(rnd.choice('天地你我他,。') for _ in range(3000))
        reference = LoopDetector(loop_length=200)
        reference.update(text)
        expected = reference.score()
        self.assertGreater(expected, 0.0)
        for size in (1, 3, 7, 512):
            detector = LoopDetector(loop_length=200)
            for i in range(0, len(text), size):
                detector.update(text[i:i + size])
            self.assertAlmostEqual(detector.score(), expected, places=12)

    def test_chunked_update_mixed_text_consistency(self):
        """含窄/宽字符的文本任意分块送入与一次性送入的最终得分一致
        （同时覆盖 2 字符与 3 字符 n-gram 的跨块桥接）"""
        text = self.MIXED_TEXT * 80
        reference = LoopDetector(loop_length=200)
        reference.update(text)
        expected = reference.score()
        self.assertGreater(expected, 0.0)
        for size in (1, 2, 3, 5, 7, 512):
            detector = LoopDetector(loop_length=200)
            for i in range(0, len(text), size):
                detector.update(text[i:i + size])
            self.assertAlmostEqual(detector.score(), expected, places=12)

    def test_update_and_score_values(self):
        """update 返回当前得分，与 score() 一致且取值在 [0, 1]"""
        detector = LoopDetector(loop_length=200)
        score = detector.update('abc')
        self.assertIsInstance(score, float)
        self.assertEqual(score, 0.0)

        score = detector.update(self.CYCLE * 30)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertAlmostEqual(detector.score(), score, places=12)
        self.assertIsInstance(detector.check(), bool)


@unittest.skipIf(not MOCK_SERVER_AVAILABLE,
                 "mock HTTP server dependencies are not available")
class TestProviderLoopDetection(unittest.IsolatedAsyncioTestCase):
    """provider 流式死循环检测集成测试（内置 mock HTTP 上游）。"""

    LOOP_TEXT = 'abcdefghij0123456789' * 3
    LOOP_TAIL = LOOP_TEXT[-20:]           # mock-loop 默认循环串
    CLIENT_TOKEN = 'loop-test-token'

    @classmethod
    def setUpClass(cls):
        cls.base_url, cls._stop_server = start_mock_http_server(
            cls.CLIENT_TOKEN)

    @classmethod
    def tearDownClass(cls):
        cls._stop_server()

    # ---------------------------- 辅助 ----------------------------

    def _make_config(self, default_extra=None):
        """构造 provider 配置，四个模型分别对应四类配置场景"""
        url = self.base_url + '/v1/chat/completions'
        auth = {'Authorization': 'Bearer ' + self.CLIENT_TOKEN}
        loop_headers = {**auth, 'X-Mock-Loop-Repeats': '300'}
        default = {
            'timeout': 30,
            'max_retries': 1,
            'parallel': 4,
            'api_type': 'openai',
        }
        if default_extra:
            default.update(default_extra)
        return {
            'default': default,
            'api': {
                # 模型级配置 loop_detect_length=400：开启检测
                'loop-on-model': {
                    'url': url, 'model': 'mock-loop', 'parallel': 1,
                    'loop_detect_length': 400,
                    'headers': dict(loop_headers),
                },
                # 不做任何配置：默认 loop_detect_length=0，检测关闭
                'loop-off-model': {
                    'url': url, 'model': 'mock-loop', 'parallel': 1,
                    'headers': dict(auth),
                },
                # 无模型级配置，仅继承全局默认
                'loop-inherit-model': {
                    'url': url, 'model': 'mock-loop', 'parallel': 1,
                    'headers': dict(loop_headers),
                },
                # 模型级显式 0，覆盖全局默认的 400
                'loop-override-model': {
                    'url': url, 'model': 'mock-loop', 'parallel': 1,
                    'loop_detect_length': 0,
                    'headers': dict(auth),
                },
            },
            'model_tag': {},
        }

    async def _drive_stream(self, config, model_name):
        """以给定配置发起流式请求，返回 (response, 通道事件列表)"""
        manager = LLMProviderManager(config)
        request = LLMProviderRequest(
            client_id='test', context_id='test',
            request_id='loop-test-' + model_name,
            model_name=model_name,
            messages=[Message.from_content(self.LOOP_TEXT, role='user')],
            stream=True,
            output_channel=Channel(),
            response=asyncio.get_running_loop().create_future())
        await manager._request_wrapper(request)
        response = await asyncio.wait_for(request.response, timeout=30)
        events = []
        while True:
            try:
                event = await asyncio.wait_for(
                    request.output_channel.read(), timeout=30)
            except (EOFError, asyncio.TimeoutError):
                break
            events.append(event)
            if event.topic == 'response' and event.data is None:
                break
        return response, events

    def _assert_channel_finalized(self, events):
        """通道以 response 结束标记收尾，且已生成内容写入通道"""
        self.assertGreater(len(events), 1)
        self.assertEqual(events[-1].topic, 'response')
        self.assertIsNone(events[-1].data)
        streamed = ''.join(
            event.data for event in events
            if event.topic == 'response' and event.data)
        self.assertGreater(len(streamed), 0)

    # ------------------------- 集成用例 -------------------------

    async def test_stream_loop_detected_with_model_config(self):
        """模型级 loop_detect_length>=2：循环流报警中断且已生成内容保留"""
        response, events = await self._drive_stream(
            self._make_config(), 'loop-on-model')

        self.assertEqual(response.finish_reason,
                         FinishReason.error_format.value)
        self.assertIn('Loop detected', response.error_text)
        self.assertTrue(response.response_text.startswith(self.LOOP_TEXT))
        self.assertGreater(len(response.response_text), len(self.LOOP_TEXT))
        self._assert_channel_finalized(events)
        self.assertTrue(any(event.topic == 'reasoning' and event.data is None
                            for event in events))

    async def test_stream_loop_detection_disabled_by_default(self):
        """未配置 loop_detect_length：同一循环流正常完成"""
        response, events = await self._drive_stream(
            self._make_config(), 'loop-off-model')

        full_text = self.LOOP_TEXT + self.LOOP_TAIL * 64
        self.assertEqual(response.finish_reason, 'stop')
        self.assertFalse(response.error_text)
        self.assertEqual(response.response_text, full_text)
        self._assert_channel_finalized(events)

    async def test_stream_loop_detected_with_default_config(self):
        """仅全局默认配置 loop_detect_length：流式请求全局生效并报警"""
        response, events = await self._drive_stream(
            self._make_config(default_extra={'loop_detect_length': 400}),
            'loop-inherit-model')

        self.assertEqual(response.finish_reason,
                         FinishReason.error_format.value)
        self.assertIn('Loop detected', response.error_text)
        self.assertTrue(response.response_text.startswith(self.LOOP_TEXT))
        self._assert_channel_finalized(events)

    async def test_model_config_overrides_default(self):
        """模型级显式 0 覆盖全局默认：检测关闭且正常完成"""
        response, events = await self._drive_stream(
            self._make_config(default_extra={'loop_detect_length': 400}),
            'loop-override-model')

        full_text = self.LOOP_TEXT + self.LOOP_TAIL * 64
        self.assertEqual(response.finish_reason, 'stop')
        self.assertFalse(response.error_text)
        self.assertEqual(response.response_text, full_text)
        self._assert_channel_finalized(events)


if __name__ == '__main__':
    unittest.main()
