import re
from typing import Optional, Dict

import xmltodict
import json_repair
from bs4 import BeautifulSoup


def parse_json(s):
    return json_repair.loads(s, skip_json_loads=True)


def get_xml_tag_content(s: str, root: str, with_tag: bool = False) -> Optional[str]:
    """
    从LLM输出的文本中找到 <root> 标签内的内容，忽略属性和一切其他XML格式。

    Args:
        s: 包含XML的文本字符串
        root: 期望的XML根标签名
        with_tag: 是否包含根标签

    Returns:
        解析后的文本
    """
    start_pattern = rf'<{root}(\s+[^>]*)?>'
    start_match = re.search(start_pattern, s, re.DOTALL)

    if not start_match:
        return None

    start_tag_end = start_match.end()
    start_pos = start_match.start() if with_tag else start_tag_end

    # 从字符串末尾向前搜索结束标签
    end_tag = f'</{root}>'
    end_pos = s.rfind(end_tag)
    # including end_pos == -1
    if end_pos < start_tag_end:
        return s[start_pos:]
    if with_tag:
        end_pos += len(end_tag)
    return s[start_pos:end_pos]


def parse_xml(s: str, root: str) -> Optional[Dict]:
    """
    从LLM输出的文本中解析XML

    Args:
        s: 包含XML的文本字符串
        root: 期望的XML根标签名

    Returns:
        解析后的字典，如果无法解析则返回None
    """
    xml_str = get_xml_tag_content(s, root, with_tag=True)
    if not xml_str:
        return None
    try:
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
    # get_xml_tag_content, parse_xml 需要用 functools.partial
}
