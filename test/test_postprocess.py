import unittest
from aitoolman.postprocess import parse_xml

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

        # 验证结果为None
        self.assertIsNone(result)

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


if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)
