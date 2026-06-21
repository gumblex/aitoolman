import json_repair
from fix_llm_xml import find_xml_document, parse_xml, get_xml_tag_text

get_xml_tag_content = find_xml_document


def parse_json(s):
    return json_repair.loads(s, skip_json_loads=True)


DEFAULT_PROCESSORS = {
    "builtin.parse_json": parse_json,
    # get_xml_tag_content, parse_xml 需要用 functools.partial
}
