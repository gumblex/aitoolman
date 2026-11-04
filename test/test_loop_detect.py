"""LoopDetector 检测器单元测试与 provider 流式死循环检测集成测试。

检测器部分覆盖构造校验、窗口未满行为、正常/周期/变异文本的报警行为、
分块一致性与得分取值范围；provider 集成部分基于项目内置 mock HTTP
上游（mock-loop 场景）驱动真实流式路径，验证 loop_detect_length 的
检测中断、默认关闭与两级配置优先级。
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
    MUTATE_ALPHABET = 'XYZxyz1234'

    def test_construction(self):
        """参数构造"""
        detector = LoopDetector(loop_length=100, theta=0.9)
        self.assertEqual(detector.window, 50)
        self.assertEqual(detector.theta, 0.9)

    def test_invalid_loop_length(self):
        """loop_length 小于 2 时构造抛出 ValueError"""
        for length in (0, 1, -400):
            with self.assertRaises(ValueError):
                LoopDetector(loop_length=length)

    def test_unfilled_window_no_alarm(self):
        """窗口未装满时不报警且得分为 0"""
        detector = LoopDetector(loop_length=400)
        for _ in range(10):  # 10*20=200 字符，不足 loop_length=400
            score = detector.update(self.CYCLE)
            self.assertEqual(score, 0.0)
            self.assertEqual(detector.score(), 0.0)
            self.assertFalse(detector.check())

    def test_random_text_no_alarm(self):
        """正常随机文本长流不触发报警"""
        rnd = random.Random(20240901)
        detector = LoopDetector()
        for _ in range(20000):
            detector.update(rnd.choice('天地你我他,。'))
        self.assertFalse(detector.check())
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

    def test_mutated_cycle_detected(self):
        """开头随机改写 0~10 字符的循环串仍被检出"""
        rnd = random.Random(3)
        cycle_text = self.CYCLE * 10
        detector = LoopDetector(loop_length=200)
        detected = False
        for _ in range(60):
            k = rnd.randint(0, 10)
            head = ''.join(
                rnd.choice(self.MUTATE_ALPHABET) for _ in range(k))
            detector.update(head + cycle_text[k:])
            if detector.check():
                detected = True
                break
        self.assertTrue(detected)

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
