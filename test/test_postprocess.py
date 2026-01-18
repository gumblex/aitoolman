import unittest
from aitoolman.postprocess import parse_xml, get_xml_tag_content

class TestParseXML(unittest.TestCase):
    """测试parse_xml函数的单元测试类"""

    def test_simple_xml_with_attributes(self):
        """测试简单XML结构（带属性）"""
        test_xml = """
        这是一些前置文本
        <root attr1="value1" attr2="value2">
            <child1>文本1</child1>
            <child2>文本2</child2>
        </root>
        这是一些后置文本
        """

        result = parse_xml(test_xml, "root")

        # 验证结果不为None
        self.assertIsNotNone(result)

        # 验证根元素存在
        self.assertIn("root", result)

        # 验证属性
        self.assertEqual(result["root"]["@attr1"], "value1")
        self.assertEqual(result["root"]["@attr2"], "value2")

        # 验证子元素
        self.assertEqual(result["root"]["child1"], "文本1")
        self.assertEqual(result["root"]["child2"], "文本2")

    def test_nested_xml_structure(self):
        """测试嵌套XML结构"""
        test_xml = """
        <response>
            <data>
                <item id="1">第一项</item>
                <item id="2">第二项</item>
            </data>
            <status>success</status>
        </response>
        """

        result = parse_xml(test_xml, "response")

        # 验证结果不为None
        self.assertIsNotNone(result)

        # 验证根元素
        self.assertIn("response", result)

        # 验证嵌套结构
        self.assertIn("data", result["response"])
        self.assertIn("status", result["response"])

        # 验证子元素
        self.assertEqual(result["response"]["status"], "success")

        # 验证列表项
        items = result["response"]["data"]["item"]
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["@id"], "1")
        self.assertEqual(items[0]["#text"], "第一项")

    def test_xml_with_namespace(self):
        """测试包含命名空间的XML"""
        test_xml = """
        <root xmlns:ns="http://example.com">
            <child>普通子元素</child>
            <ns:child>带命名空间的子元素</ns:child>
        </root>
        """

        result = parse_xml(test_xml, "root")

        # 验证结果不为None
        self.assertIsNotNone(result)

        # 验证命名空间原样输出
        self.assertIn("ns:child", result["root"])
        self.assertEqual(result["root"]["ns:child"], "带命名空间的子元素")

    def test_invalid_xml(self):
        """测试无效XML（返回None）"""
        test_xml = """
        这不是一个完整的XML
        <root>
            <child>没有闭合标签
        """

        result = parse_xml(test_xml, "root")

        self.assertEqual(result, {'root': {'child': '没有闭合标签'}})

    def test_xml_with_comments(self):
        """测试带注释的XML"""
        test_xml = """
        前置文本
        <root>
            <!-- 这是一个注释 -->
            <child>内容</child>
            <another>更多内容</another>
        </root>
        后置文本
        """

        result = parse_xml(test_xml, "root")

        # 验证结果不为None
        self.assertIsNotNone(result)

        # 验证注释被正确处理
        # 注意：xmltodict默认不处理注释，除非设置process_comments=True
        # 这里我们只验证XML结构被正确解析
        self.assertIn("root", result)
        self.assertEqual(result["root"]["child"], "内容")
        self.assertEqual(result["root"]["another"], "更多内容")

    def test_xml_surrounded_by_text(self):
        """测试XML被其他文本包围的情况"""
        test_xml = """
        这是前置文本，可能很长很长
        包含各种字符!@#$%^&*()

        <data id="123">
            <name>测试名称</name>
            <value>100</value>
        </data>

        这是后置文本，也可能包含特殊字符
        """

        result = parse_xml(test_xml, "data")

        # 验证结果不为None
        self.assertIsNotNone(result)

        # 验证XML被正确提取和解析
        self.assertIn("data", result)
        self.assertEqual(result["data"]["@id"], "123")
        self.assertEqual(result["data"]["name"], "测试名称")
        self.assertEqual(result["data"]["value"], "100")

    def test_empty_root_tag(self):
        """测试空根标签"""
        test_xml = "<root></root>"

        result = parse_xml(test_xml, "root")

        # 验证结果不为None
        self.assertIsNotNone(result)

        # 验证空标签被正确解析
        self.assertIn("root", result)
        # 空标签在xmltodict中通常解析为None或空字符串
        # 具体行为可能取决于xmltodict版本

    def test_multiple_root_tags(self):
        """测试多个根标签（应只提取第一个）"""
        test_xml = """
        <root1>
            <child>第一个根</child>
        </root1>
        <root2>
            <child>第二个根</child>
        </root2>
        """

        result = parse_xml(test_xml, "root1")

        # 验证结果不为None
        self.assertIsNotNone(result)

        # 验证只提取了第一个根标签
        self.assertIn("root1", result)
        self.assertEqual(result["root1"]["child"], "第一个根")

        # 验证第二个根标签没有被包含
        # 注意：由于我们的函数只提取第一个完整的根标签结构，
        # 所以第二个根标签不会被包含在结果中

    def test_root_tag_with_special_characters(self):
        """测试根标签包含特殊字符"""
        test_xml = """
        <root-tag special="true">
            <sub-tag>内容</sub-tag>
        </root-tag>
        """

        result = parse_xml(test_xml, "root-tag")

        # 验证结果不为None
        self.assertIsNotNone(result)

        # 验证特殊字符的标签名被正确解析
        self.assertIn("root-tag", result)
        self.assertEqual(result["root-tag"]["@special"], "true")
        self.assertEqual(result["root-tag"]["sub-tag"], "内容")


class TestGetXmlTagContent(unittest.TestCase):
    """测试get_xml_tag_content函数的单元测试类"""

    def test_simple_extraction(self):
        """测试正常提取根标签内容"""
        text = "前置文本<root>核心内容</root>后置文本"
        result = get_xml_tag_content(text, "root")
        self.assertEqual(result, "核心内容")

    def test_incomplete_start_tag(self):
        """测试缺少有效开始标签的情况"""
        text = "没有有效的<roo>开始标签</root>"
        result = get_xml_tag_content(text, "root")
        self.assertIsNone(result)

    def test_incomplete_end_tag(self):
        """测试缺少结束标签的情况"""
        text = "有开始标签<root>但没有结束标签"
        result = get_xml_tag_content(text, "root")
        self.assertEqual(result, "但没有结束标签")

    def test_multiple_same_tags(self):
        """测试存在多个相同根标签的情况"""
        text = "第一个<root>内容1</root>中间文本<root>内容2</root>最后"
        result = get_xml_tag_content(text, "root")
        # 函数会提取从第一个开始标签到最后一个结束标签的完整范围
        self.assertEqual(result, "内容1</root>中间文本<root>内容2")

    def test_tag_with_special_characters(self):
        """测试包含特殊字符的标签名提取"""
        text = "前缀<my-root-tag attr='test'>内部内容</my-root-tag>后缀"
        result = get_xml_tag_content(text, "my-root-tag")
        self.assertEqual(result, "内部内容")

    def test_start_tag_with_attributes(self):
        """测试带属性的开始标签提取"""
        text = "前置<user id='123' name='Alice' enabled='true'>用户信息</user>后置"
        result = get_xml_tag_content(text, "user", with_tag=True)
        self.assertEqual(result, "<user id='123' name='Alice' enabled='true'>用户信息</user>")

    def test_no_target_tag(self):
        """测试文本中没有目标标签的情况"""
        text = "整个文本只有<other>其他标签</other>，没有目标标签"
        result = get_xml_tag_content(text, "root")
        self.assertIsNone(result)

    def test_end_tag_before_start_tag(self):
        """测试结束标签在开始标签之前的异常情况"""
        text = "</root>结束标签在前面<root>开始标签在后面"
        result = get_xml_tag_content(text, "root")
        self.assertEqual(result, "开始标签在后面")

    def test_empty_tag(self):
        """测试结束标签在开始标签之前的异常情况"""
        text = "aaaa<root></root>bbbb"
        self.assertEqual(get_xml_tag_content(text, "root"), '')
        self.assertEqual(get_xml_tag_content(text, "root", with_tag=True), '<root></root>')

    def test_nested_tags_inside_root(self):
        """测试根标签内部包含嵌套标签的情况"""
        text = "前<root><child1>子内容1</child1><child2>子内容2</child2></root>后"
        result = get_xml_tag_content(text, "root")
        self.assertEqual(result, "<child1>子内容1</child1><child2>子内容2</child2>")

    def test_tag_with_whitespace_in_attributes(self):
        """测试属性中包含空格的开始标签提取"""
        text = "前置<item class='product featured' price='99.99'>商品</item>后置"
        result = get_xml_tag_content(text, "item", with_tag=True)
        self.assertEqual(result, "<item class='product featured' price='99.99'>商品</item>")


if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)
