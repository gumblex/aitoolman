import re
from typing import Optional, Dict

import xmltodict
import json_repair
from bs4 import BeautifulSoup


def parse_json(s):
    return json_repair.loads(s, skip_json_loads=True)


def parse_xml(s: str, root: str) -> Optional[Dict]:
    """
    从LLM输出的文本中解析XML

    Args:
        s: 包含XML的文本字符串
        root: 期望的XML根标签名

    Returns:
        解析后的字典，如果无法解析则返回None
    """
    try:
        # 1. 使用正则从头搜索开始标签，从尾搜索结束标签
        # 构建开始标签的正则表达式（可能包含属性）
        start_pattern = rf'<{root}(\s+[^>]*)?>'
        start_match = re.search(start_pattern, s, re.DOTALL)

        if not start_match:
            return None

        # 找到开始标签的位置
        start_pos = start_match.start()

        # 从字符串末尾向前搜索结束标签
        end_tag = f'</{root}>'
        end_pos = s.rfind(end_tag)

        if end_pos == -1:
            return None

        # 计算结束标签的结束位置
        end_pos += len(end_tag)

        # 提取完整的XML结构
        xml_str = s[start_pos:end_pos]

        # 2. 使用bs4修复XML结构
        # 使用xml解析器来修复XML
        soup = BeautifulSoup(xml_str, 'xml')

        # 获取修复后的XML字符串
        fixed_xml = str(soup)

        # 3. 用xmltodict解析XML为Dict
        # 设置disable_entities=True以避免XXE攻击
        # 不处理命名空间，原样输出
        result = xmltodict.parse(
            fixed_xml,
            process_namespaces=False,
            disable_entities=True
        )

        return result

    except Exception as e:
        # 如果解析过程中出现任何异常，返回None
        return None


DEFAULT_PROCESSORS = {
    "builtin.parse_json": parse_json,
}
