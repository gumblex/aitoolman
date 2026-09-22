import json
import unittest

import msgspec

from aitoolman.util import (
    calculate_rank_weights, calc_message_length, estimate_text_token,
    decode_json, encode_message, MessageLength
)
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


class TestDecodeJsonEncodeMessage(unittest.TestCase):
    """decode_json / encode_message（msgspec 实现）行为测试"""

    def test_decode_json_str(self):
        """str 输入解码为 Python 对象"""
        self.assertEqual(decode_json('{"a": 1, "b": [true, null]}'),
                         {"a": 1, "b": [True, None]})

    def test_decode_json_bytes(self):
        """bytes 输入解码为 Python 对象"""
        self.assertEqual(decode_json(b'{"a": 1}'), {"a": 1})

    def test_decode_json_nested_structure(self):
        """嵌套结构往返保持一致"""
        obj = {"list": [1, 2.5, None, True], "nested": {"x": {"y": "z"}}}
        self.assertEqual(decode_json(encode_message(obj)), obj)

    def test_decode_json_non_ascii(self):
        """非 ASCII 内容解码正确"""
        self.assertEqual(decode_json('{"msg": "你好世界"}'), {"msg": "你好世界"})

    def test_decode_json_invalid_raises(self):
        """非法 JSON 抛出 msgspec.DecodeError"""
        with self.assertRaises(msgspec.DecodeError):
            decode_json('{"a": }')

    def test_encode_message_returns_bytes(self):
        """encode_message 返回 bytes 类型"""
        result = encode_message({"a": 1})
        self.assertIsInstance(result, bytes)
        self.assertEqual(decode_json(result), {"a": 1})

    def test_encode_message_named_tuple_to_array(self):
        """NamedTuple 编码为 JSON 数组"""
        length = MessageLength(text_bytes=1, message_num=2, fragment_num=3,
                               image_num=4, video_num=5)
        result = encode_message(length)
        self.assertEqual(json.loads(result), [1, 2, 3, 4, 5])

    def test_encode_message_non_ascii_utf8(self):
        """非 ASCII 字符按 UTF-8 编码（不做转义）"""
        result = encode_message({"msg": "你好"})
        self.assertIn("你好".encode('utf-8'), result)
        self.assertEqual(decode_json(result), {"msg": "你好"})

    def test_roundtrip_none_and_empty_containers(self):
        """None 与空容器往返一致"""
        obj = {"a": None, "b": [], "c": {}}
        self.assertEqual(decode_json(encode_message(obj)), obj)


if __name__ == '__main__':
    unittest.main()
