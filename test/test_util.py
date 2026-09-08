import unittest
from aitoolman.util import calculate_rank_weights, calc_message_length, estimate_text_token, MessageLength
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


class TestCalcMessageLengthAndEstimateTextToken(unittest.TestCase):
    def test_text_basic(self):
        """基本文本长度计算与Token估算"""
        messages = [Message.from_content("Hello world", role="user")]
        length = calc_message_length(messages)
        # "user" (4 bytes) + "Hello world" (11 bytes) = 15 bytes
        self.assertEqual(length.text_bytes, 15)
        self.assertEqual(length.message_num, 1)
        self.assertEqual(length.fragment_num, 2)  # role, content
        self.assertEqual(length.image_num, 0)
        self.assertEqual(length.video_num, 0)

        # ceil(15/4.0) = 4, (1+2) = 3. Total = 7
        result = estimate_text_token(length, 4.0, 1300, 50000)
        self.assertEqual(result, 7)

    def test_text_empty(self):
        """无内容消息Token估算"""
        messages = [Message.from_content("", role="user")]
        length = calc_message_length(messages)
        # "user" (4 bytes), content is "" so not counted as fragment
        self.assertEqual(length.text_bytes, 4)
        self.assertEqual(length.message_num, 1)
        self.assertEqual(length.fragment_num, 1)  # only role
        # ceil(4/4) = 1, (1+1) = 2. Total = 3
        result = estimate_text_token(length, 4.0, 1300, 50000)
        self.assertEqual(result, 3)

    def test_text_custom_bytes_per_token(self):
        """自定义 bytes_per_token 参数"""
        messages = [Message.from_content("Hello world", role="user")]
        length = calc_message_length(messages)
        # ceil(15/2.0) = 8, (1+2) = 3. Total = 11
        result = estimate_text_token(length, bytes_per_token=2.0)
        self.assertEqual(result, 11)

    def test_text_multibyte(self):
        """包含中文等多字节字符的估算"""
        messages = [Message.from_content("你好世界", role="user")]
        length = calc_message_length(messages)
        # "user" (4) + "你好世界" (12) = 16 bytes
        self.assertEqual(length.text_bytes, 16)
        # ceil(16/4) = 4, (1+2) = 3. Total = 7
        result = estimate_text_token(length, 4.0, 1300, 50000)
        self.assertEqual(result, 7)

    def test_multimedia_token_estimation(self):
        """包含图片和视频的Token估算"""
        length = MessageLength(
            text_bytes=10,
            message_num=1,
            fragment_num=2,
            image_num=2,
            video_num=1
        )
        # text: ceil(10/4)=3
        # special: 1+2=3
        # image: 2 * 1300 = 2600
        # video: 1 * 50000 = 50000
        # total = 3 + 3 + 2600 + 50000 = 52606
        result = estimate_text_token(length, 4.0, 1300, 50000)
        self.assertEqual(result, 52606)


if __name__ == '__main__':
    unittest.main()
