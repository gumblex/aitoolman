import math
import json
import base64
import socket
import mimetypes
try:
    import tomllib
except ImportError:
    import tomli as tomllib
try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False
import ksuid
from typing import List, Dict, Union, NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Message


def get_id():
    return str(ksuid.Ksuid())


def get_host_id(hostname=None):
    host_part = hostname or socket.gethostname()
    return host_part + "_" + str(ksuid.Ksuid())[-6:]


def make_insert(d):
    keys, values = zip(*d.items())
    return ', '.join(keys), ', '.join('?' * len(values)), values


def make_insert_auto(d):
    keys, values = zip(*d.items())
    converted_values = []
    for value in values:
        if (value is None or isinstance(value, str)
                or isinstance(value, int) or isinstance(value, float)):
            converted_values.append(value)
        else:
            converted_values.append(json.dumps(value, ensure_ascii=False))
    return ', '.join(keys), ', '.join('?' * len(values)), converted_values


def load_config(filename):
    with open(filename, 'rb') as f:
        return tomllib.load(f)


def load_config_str(s: str):
    return tomllib.loads(s)


def generate_data_url(data, content_type):
    return 'data:%s;base64,%s' % (
        content_type,
        base64.b64encode(data).decode('utf-8')
    )


def get_mime_type(filename):
    return mimetypes.guess_type(filename)[0]


def decode_json(s):
    if ORJSON_AVAILABLE:
        return orjson.loads(s)
    if isinstance(s, bytes):
        return json.loads(s.decode('utf-8'))
    return json.loads(s)


def encode_message(obj) -> bytes:
    if ORJSON_AVAILABLE:
        return orjson.dumps(obj)
    return json.dumps(obj, ensure_ascii=False).encode('utf-8')


def calculate_rank_weights(items: List[Union[str, List[str]]], rank_adjust_ratio: float = 0.25) -> Dict[str, float]:
    """按排名计算权重

    公式：权重 = ((2*choice_num + choice_rank*(adjust_ratio - 1) + (adjust_ratio - 1)*(choice_rank + 1))/(choice_num^2*(adjust_ratio + 1)))

    Args:
        items: 模型名列表，顺序即优先级（越靠前权重越高）。支持嵌套列表：嵌套列表中的模型共享该位置的排名权重，随机轮流使用。
        rank_adjust_ratio: 权重调整参数，默认0.25

    Returns:
        Dict[模型名, 权重]
    """
    if not items:
        return {}
    choice_num = len(items)
    result = {}
    for choice_rank, item in enumerate(items):
        weight = (
            (2.0 * choice_num + choice_rank * (rank_adjust_ratio - 1) +
                (rank_adjust_ratio - 1) * (choice_rank + 1)) /
            (choice_num ** 2 * (rank_adjust_ratio + 1))
        )
        if isinstance(item, list):
            for sub_item in item:
                result[sub_item] = weight
        else:
            result[item] = weight
    return result


class MessageLength(NamedTuple):
    """消息长度统计信息"""
    text_bytes: int       # 文本内容总字节数
    message_num: int      # 消息数量
    fragment_num: int     # 文本片段数量（有内容的字段数）
    image_num: int        # 图片数量
    video_num: int        # 视频数量


def calc_message_length(messages: 'List[Message]') -> MessageLength:
    """计算消息列表的长度信息

    统计所有消息的文本内容总字节数、消息数量、文本片段数量、图片数量、视频数量

    Args:
        messages: Message对象列表

    Returns:
        MessageLength统计结果
    """
    total_bytes = 0
    message_num = 0
    fragment_num = 0
    image_num = 0
    video_num = 0

    for msg in messages:
        message_num += 1
        for key in ('role', 'content', 'reasoning_content', 'tool_call_id'):
            value = getattr(msg, key)
            if value:
                total_bytes += len(value.encode('utf-8'))
                fragment_num += 1
        # 统计多媒体内容
        if msg.media_content:
            for media in msg.media_content:
                if media.media_type == "image":
                    image_num += 1
                elif media.media_type == "video":
                    video_num += 1

    return MessageLength(
        text_bytes=total_bytes,
        message_num=message_num,
        fragment_num=fragment_num,
        image_num=image_num,
        video_num=video_num
    )


def estimate_text_token(
    length: MessageLength,
    bytes_per_token: float = 4.0,
    token_per_image: int = 1300,
    token_per_video: int = 50000
) -> int:
    """基于消息长度信息估算输入Token数

    计算公式：text_bytes / bytes_per_token 向上取整 + (message_num + fragment_num) + image_num * token_per_image + video_num * token_per_video

    Args:
        length: 消息长度统计信息
        bytes_per_token: 单Token对应的UTF-8字节数估算值
        token_per_image: 单张图片估算Token数
        token_per_video: 单个视频估算Token数

    Returns:
        估算的Token数
    """
    text_tokens = math.ceil(length.text_bytes / bytes_per_token)
    special_tokens = length.message_num + length.fragment_num
    media_tokens = length.image_num * token_per_image + length.video_num * token_per_video
    return text_tokens + special_tokens + media_tokens
