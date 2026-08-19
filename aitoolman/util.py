import math
import json
import base64
import socket
import mimetypes
try:
    import tomllib
except ImportError:
    import tomli as tomllib
import ksuid
from typing import List, Dict, Union



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


def encode_message(obj):
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


def estimate_text_tokens(messages, bytes_per_token: float = 4.0) -> int:
    """估算消息列表的总输入Token数

    统计所有消息的文本内容总字节数，除以bytes_per_token向上取整

    Args:
        messages: Message对象列表
        bytes_per_token: 单Token对应的UTF-8字节数估算值

    Returns:
        估算的Token数（向上取整）
    """
    total_bytes = 0
    special_token = 0
    for msg in messages:
        for key in ('role', 'content', 'reasoning_content', 'tool_call_id'):
            value = getattr(msg, key)
            if value:
                total_bytes += len(value.encode('utf-8'))
                special_token += 1
        special_token += 1
    return math.ceil(total_bytes / bytes_per_token) + special_token
