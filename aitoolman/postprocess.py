import re
import bisect
from typing import Optional, Dict

import xmltodict
import json_repair
from bs4 import BeautifulSoup


def parse_json(s):
    return json_repair.loads(s, skip_json_loads=True)


CDATA_START = '<![CDATA['
CDATA_END = ']]>'


def get_xml_tag_content(s: str, root: str, with_tag: bool = False, cdata: bool = False) -> Optional[str]:
    """
    从LLM输出的文本中找到 <root> 标签内的内容，忽略属性和一切其他XML格式。

    Args:
        s: 包含XML的文本字符串
        root: 期望的XML根标签名
        with_tag: 是否包含根标签
        cdata: 是否识别 <![CDATA[...]]> 标签

    Returns:
        解析后的文本
    """
    start_pattern = rf'<{root}(\s+[^>]*)?>'
    if cdata:
        start_pattern += r'<!\[CDATA\['
    start_match = re.search(start_pattern, s, re.DOTALL)

    if not start_match:
        return None

    start_tag_end = start_match.end()
    start_pos = start_match.start() if with_tag else start_tag_end

    # 从字符串末尾向前搜索结束标签
    end_tag = f'</{root}>'
    if cdata:
        end_tag = CDATA_END + end_tag
    end_pos = s.rfind(end_tag)
    # including end_pos == -1
    if end_pos < start_tag_end:
        return s[start_pos:]
    if with_tag:
        end_pos += len(end_tag)
    return s[start_pos:end_pos]


def parse_xml(s: str, root: str, force_list=None) -> Optional[Dict]:
    """
    从LLM输出的文本中解析XML

    Args:
        s: 包含XML的文本字符串
        root: 期望的XML根标签名
        force_list: 用于 xmltodict，将标签作为列表

    Returns:
        解析后的字典，如果无法解析则返回None
    """
    xml_str = get_xml_tag_content(s, root, with_tag=True)
    if not xml_str:
        return None
    try:
        return xmltodict.parse(
            xml_str,
            process_namespaces=False,
            disable_entities=True,
            force_list=force_list
        )
    except Exception:
        if xml_str.count(CDATA_START) > 1:
            xml_str = escape_nested_cdata(xml_str)
        try:
            soup = BeautifulSoup(xml_str, 'xml')
            fixed_xml = str(soup)
            return xmltodict.parse(
                fixed_xml,
                process_namespaces=False,
                disable_entities=True,
                force_list=force_list
            )
        except Exception as e:
            # 如果解析过程中出现任何异常，返回None
            return None


def escape_nested_cdata(s: str) -> str:
    """
    修复XML中，有嵌套CDATA块的问题。

    规则：
    1. 只有当嵌套的CDATA块有对应的结束标记时，才进行转义。
       即：`<![CDATA[外层<![CDATA[内层]]>外层]]>` 会被转义为 `<![CDATA[外层<![CDATA[内层]]]]><![CDATA[>外层]]>`。
       但：`<![CDATA[外层<![CDATA[内层]]>` 不会被转义，因为内层的 `]]>` 实际上关闭了外层。
    2. 已经转义过的 `]]]]><![CDATA[` 序列会被视为普通文本，不会被再次转义。
    """
    ESCAPED_CDATA_END = ']]]]><![CDATA[>'

    # 1. 标记化
    # 将字符串分解为标记：TEXT, START, END
    # 注意：ESCAPED_CDATA_END 包含 CDATA_END，所以必须优先匹配
    tokens = []
    n = len(s)
    i = 0

    while i < n:
        if s.startswith(ESCAPED_CDATA_END, i):
            # 视为普通文本，不作为结构性的结束标记
            tokens.append(('TEXT', ESCAPED_CDATA_END))
            i += len(ESCAPED_CDATA_END)
        elif s.startswith(CDATA_START, i):
            tokens.append(('START', CDATA_START))
            i += len(CDATA_START)
        elif s.startswith(CDATA_END, i):
            tokens.append(('END', CDATA_END))
            i += len(CDATA_END)
        else:
            # 普通文本
            j = i
            while j < n:
                if s.startswith(ESCAPED_CDATA_END, j) or \
                        s.startswith(CDATA_START, j) or \
                        s.startswith(CDATA_END, j):
                    break
                j += 1
            tokens.append(('TEXT', s[i:j]))
            i = j

    # 2. 预处理
    # 收集所有 END 标记的索引，用于快速查找
    end_indices = [idx for idx, (t, _) in enumerate(tokens) if t == 'END']

    # 3. 重建字符串
    res = []
    stack = 0  # 当前嵌套深度

    for idx, (t, v) in enumerate(tokens):
        if t == 'START':
            if stack == 0:
                # 顶层 CDATA 开始
                res.append(v)
                stack += 1
            else:
                # 在 CDATA 内部遇到新的开始标记
                # 检查是否有足够的结束标记来支持嵌套
                # 我们需要至少 2 个结束标记：一个关闭内层，一个关闭外层
                # 查找当前索引之后有多少个结束标记
                next_end_pos = bisect.bisect_right(end_indices, idx)
                remaining_ends = len(end_indices) - next_end_pos

                if remaining_ends >= 2:
                    # 确认是有效的嵌套
                    res.append(v)
                    stack += 1
                else:
                    # 不是有效的嵌套，视为普通文本
                    res.append(v)

        elif t == 'END':
            if stack > 0:
                stack -= 1
                if stack == 0:
                    # 回到顶层，使用原始结束标记
                    res.append(CDATA_END)
                else:
                    # 仍在嵌套中，使用转义后的结束标记
                    res.append(ESCAPED_CDATA_END)
            else:
                # 不在 CDATA 中，视为普通文本
                res.append(v)

        elif t == 'TEXT':
            res.append(v)

    return ''.join(res)


DEFAULT_PROCESSORS = {
    "builtin.parse_json": parse_json,
    # get_xml_tag_content, parse_xml 需要用 functools.partial
}
