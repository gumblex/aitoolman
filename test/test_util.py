import unittest
from aitoolman.util import calculate_rank_weights, estimate_text_tokens
from aitoolman.model import Message


class TestCalculateRankWeights(unittest.TestCase):
    def test_calculate_rank_weights_basic(self):
        """基本排名权重计算，验证权重之和约为1，越靠前权重越高"""
        items = ['model-a', 'model-b', 'model-c']
        result = calculate_rank_weights(items)
        self.assertEqual(len(result), 3)
        self.assertSetEqual(set(result.keys()), {'model-a', 'model-b', 'model-c'})
        # 权重之和约为1
        self.assertAlmostEqual(sum(result.values()), 1.0, places=9)
        # 越靠前权重越高
        self.assertGreater(result['model-a'], result['model-b'])
        self.assertGreater(result['model-b'], result['model-c'])

    def test_calculate_rank_weights_empty(self):
        """空列表返回空字典"""
        result = calculate_rank_weights([])
        self.assertEqual(result, {})

    def test_calculate_rank_weights_single(self):
        """单元素列表权重验证"""
        result = calculate_rank_weights(['model-a'])
        self.assertEqual(result, {'model-a': 1.0})

    def test_calculate_rank_weights_custom_ratio(self):
        """自定义 rank_adjust_ratio 参数"""
        items = ['a', 'b', 'c']
        result = calculate_rank_weights(items, rank_adjust_ratio=0.5)
        # 权重之和约为1
        self.assertAlmostEqual(sum(result.values()), 1.0, places=9)
        # 越靠前权重越高
        self.assertGreater(result['a'], result['b'])
        self.assertGreater(result['b'], result['c'])


class TestEstimateTextTokens(unittest.TestCase):
    def test_estimate_text_tokens_basic(self):
        """基本文本Token估算"""
        messages = [Message.from_content("Hello world", role="user")]
        result = estimate_text_tokens(messages)
        # "Hello world" = 11 bytes, 11/4.0 = 2.75 -> ceil = 3
        self.assertGreaterEqual(result, 3)

    def test_estimate_text_tokens_empty(self):
        """无内容消息返回0"""
        messages = [Message.from_content("", role="user")]
        result = estimate_text_tokens(messages)
        self.assertEqual(result, 3)

    def test_estimate_text_tokens_custom_ratio(self):
        """自定义 bytes_per_token 参数"""
        messages = [Message.from_content("Hello world", role="user")]
        result = estimate_text_tokens(messages, bytes_per_token=2.0)
        self.assertGreaterEqual(result, 6)

    def test_estimate_text_tokens_multibyte(self):
        """包含中文等多字节字符的估算"""
        messages = [Message.from_content("你好世界", role="user")]
        result = estimate_text_tokens(messages)
        self.assertGreaterEqual(result, 3)


if __name__ == '__main__':
    unittest.main()
