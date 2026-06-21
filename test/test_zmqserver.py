import unittest
from aitoolman.zmqserver import LLMZmqServer


def _make_server(auth_token=None, manage_token=None):
    """创建测试用服务器实例，可设置两种token"""
    server = object.__new__(LLMZmqServer)
    server.zmq_auth_token = auth_token
    server.zmq_manage_token = manage_token
    return server


class TestCheckPermission(unittest.TestCase):
    """测试权限校验逻辑"""

    def test_manage_permission_none_manage_token(self):
        """管理token为None时，任何人都没有管理权限"""
        server = _make_server(manage_token=None)
        # 任何token都不应有管理权限
        self.assertFalse(server._check_permission('any_token', manage=True))
        self.assertFalse(server._check_permission('', manage=True))
        self.assertFalse(server._check_permission('valid_token', manage=True))

    def test_manage_permission_valid_token(self):
        """正确管理token通过校验"""
        server = _make_server(manage_token='manage_secret')
        self.assertTrue(server._check_permission('manage_secret', manage=True))

    def test_manage_permission_invalid_token(self):
        """错误管理token被拒绝"""
        server = _make_server(manage_token='manage_secret')
        self.assertFalse(server._check_permission('wrong_token', manage=True))

    def test_manage_permission_empty_token_match(self):
        """空管理token匹配空字符串"""
        server = _make_server(manage_token='')
        self.assertTrue(server._check_permission('', manage=True))
        self.assertFalse(server._check_permission('some_token', manage=True))

    def test_normal_permission_no_auth_token(self):
        """普通token为None时，任何token都有普通权限"""
        server = _make_server(auth_token=None)
        # 即使管理token为None，只要auth_token为None就有普通权限
        self.assertTrue(server._check_permission('any_token', manage=False))
        self.assertTrue(server._check_permission('', manage=False))

        # 设置管理token后，普通权限仍然可用
        server_with_manage = _make_server(auth_token=None, manage_token='manage_secret')
        self.assertTrue(server_with_manage._check_permission('any_token', manage=False))
        self.assertTrue(server_with_manage._check_permission('', manage=False))

    def test_normal_permission_with_auth_token(self):
        """普通token不为None时，只有匹配的token有普通权限"""
        server = _make_server(auth_token='auth_secret')
        self.assertTrue(server._check_permission('auth_secret', manage=False))
        self.assertFalse(server._check_permission('wrong_token', manage=False))
        self.assertFalse(server._check_permission('', manage=False))

    def test_normal_permission_with_both_tokens(self):
        """同时设置auth和manage token时，两种token都有普通权限"""
        server = _make_server(auth_token='auth_secret', manage_token='manage_secret')
        # auth_token有普通权限
        self.assertTrue(server._check_permission('auth_secret', manage=False))
        # manage_token也有普通权限
        self.assertTrue(server._check_permission('manage_secret', manage=False))
        # 其他token没有普通权限
        self.assertFalse(server._check_permission('other_token', manage=False))

    def test_normal_permission_empty_auth_token(self):
        """普通token为空字符串时，只有空token有普通权限"""
        server = _make_server(auth_token='')
        self.assertTrue(server._check_permission('', manage=False))
        self.assertFalse(server._check_permission('any_token', manage=False))

        # 同时设置manage_token
        server_with_manage = _make_server(auth_token='', manage_token='manage_secret')
        self.assertTrue(server_with_manage._check_permission('', manage=False))
        self.assertTrue(server_with_manage._check_permission('manage_secret', manage=False))
        self.assertFalse(server_with_manage._check_permission('other_token', manage=False))


if __name__ == '__main__':
    unittest.main()
