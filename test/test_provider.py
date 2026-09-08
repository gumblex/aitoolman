import json
import unittest
from unittest import mock

from aitoolman.sse import ServerSentEvent

from aitoolman.provider import OpenAICompatibleFormat, LLMProviderManager
from aitoolman.model import LLMProviderRequest, LLMProviderResponse, LLMNoAvailableModelError, Message


# OK
class TestOpenAIFormat(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model_config = {
            "model": "gpt-3.5-turbo",
            "url": "https://api.openai.com/v1/chat/completions",
            "body_options": {"temperature": 0.7}
        }
        self.format_strategy = OpenAICompatibleFormat(self.model_config)

    def test_make_request_body_basic(self):
        request = LLMProviderRequest(
            client_id="test_client",
            context_id="test_context",
            request_id="test_request",
            model_name="gpt-3.5-turbo",
            messages=[Message.from_content("Hello", role="user")],
            stream=False
        )

        body = self.format_strategy.make_request_body(request)

        self.assertEqual(body["model"], "gpt-3.5-turbo")
        self.assertEqual(body["messages"],
                         [{"role": "user", "content": "Hello"}])
        self.assertEqual(body["stream"], False)
        self.assertEqual(body["temperature"], 0.7)

    def test_make_request_body_with_tools(self):
        request = LLMProviderRequest(
            client_id="test_client",
            context_id="test_context",
            request_id="test_request",
            model_name="gpt-3.5-turbo",
            messages=[Message.from_content("Hello", role="user")],
            tools={"test": {"description": "test", "param": {}}},
            stream=False
        )

        body = self.format_strategy.make_request_body(request)
        self.assertIn("tools", body)

    def test_parse_batch_response_success(self):
        response = LLMProviderResponse(
            client_id="test_client",
            context_id="test_context",
            request_id="test_request",
            model_name="gpt-3.5-turbo",
            stream=False
        )

        response_data = {
            "choices": [{
                "message": {
                    "content": "Hello, world!",
                    "reasoning_content": "Thinking...",
                    "tool_calls": [{"id": "1"}]
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }

        self.format_strategy.parse_batch_response(response, response_data)

        self.assertEqual(response.response_text, "Hello, world!")
        self.assertEqual(response.response_reasoning, "Thinking...")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.prompt_tokens, 10)
        self.assertEqual(response.completion_tokens, 5)

    def test_parse_batch_response_no_choices(self):
        response = LLMProviderResponse(
            client_id="test_client",
            context_id="test_context",
            request_id="test_request",
            model_name="gpt-3.5-turbo",
            stream=False
        )

        response_data = {}

        with self.assertRaises(ValueError):
            self.format_strategy.parse_batch_response(response, response_data)

    def test_parse_stream_chunk_data_done(self):
        response = LLMProviderResponse(
            client_id="test_client",
            context_id="test_context",
            request_id="test_request",
            model_name="gpt-3.5-turbo",
            stream=True
        )

        event = self.format_strategy.parse_stream_event(
            response, ServerSentEvent(data=b"[DONE]"))
        self.assertTrue(event.is_end)

    def test_parse_stream_chunk_valid_data(self):
        response = LLMProviderResponse(
            client_id="test_client",
            context_id="test_context",
            request_id="test_request",
            model_name="gpt-3.5-turbo",
            stream=True
        )

        chunk_data = {
            "choices": [{
                "delta": {"content": "Hello", "reasoning_content": "Think"},
                "finish_reason": None
            }]
        }

        event = self.format_strategy.parse_stream_event(
            response, ServerSentEvent(data=json.dumps(chunk_data)))
        self.assertFalse(event.is_end)
        self.assertEqual(event.content, "Hello")
        self.assertEqual(event.reasoning, "Think")


def make_test_config():
    return {
        'default': {
            'timeout': 30,
            'max_retries': 2,
            'parallel': 1,
            'api_type': 'openai',
        },
        'api': {
            'model-a': {
                'type': 'openai',
                'model': 'model-a',
                'url': 'http://localhost:8000/v1/chat/completions',
                'parallel': 2,
            },
            'model-b': {
                'type': 'openai',
                'model': 'model-b',
                'url': 'http://localhost:8001/v1/chat/completions',
                'parallel': 4,
                'max_input_tokens': 1000,
                'bytes_per_token': 4.0,
            },
            'model-c': {
                'type': 'openai',
                'model': 'model-c',
                'url': 'http://localhost:8002/v1/chat/completions',
                'enable': False,
            },
        },
        'model_tag': {
            'fast': ['model-a', 'model-b'],
            'cheap': ['model-b'],
        },
        'model_alias': {
            'alias_a': 'model-a',
        },
    }


class TestResolveModel(unittest.TestCase):
    def test_resolve_model_exact_match(self):
        """精确匹配优先，传入真实模型名直接返回"""
        manager = LLMProviderManager(make_test_config())
        result = manager.resolve_model(['model-a'])
        self.assertEqual(result, ['model-a'])

    def test_resolve_model_tag_match(self):
        """标签匹配，返回权重最高的模型"""
        manager = LLMProviderManager(make_test_config())
        result = manager.resolve_model(['fast'])
        # model-a is first in 'fast' tag, so it has highest weight
        self.assertEqual(result[0], 'model-a')

    def test_resolve_model_multiple_tags(self):
        """多标签交集匹配"""
        manager = LLMProviderManager(make_test_config())
        # 'fast' has model-a, model-b; 'cheap' has model-b
        # intersection is model-b
        result = manager.resolve_model(['fast', 'cheap'])
        self.assertEqual(result[0], 'model-b')

    def test_resolve_model_disabled(self):
        """禁用模型不参与路由"""
        manager = LLMProviderManager(make_test_config())
        with self.assertRaises(LLMNoAvailableModelError):
            manager.resolve_model(['model-c'])

    def test_resolve_model_token_filter(self):
        """Token限制过滤"""
        manager = LLMProviderManager(make_test_config())
        # model-b has max_input_tokens=1000, bytes_per_token=4.0
        # Create messages that exceed 1000 tokens
        long_text = "a" * 5000  # 5000 bytes / 4.0 = 1250 tokens > 1000
        messages = [Message.from_content(long_text, role="user")]
        # 'fast' tag has model-a (no limit) and model-b (1000 token limit)
        # model-b should be filtered out, leaving model-a
        result = manager.resolve_model(['fast'], messages)
        self.assertEqual(result[0], 'model-a')

    def test_resolve_model_no_available(self):
        """无可用模型抛出 LLMNoAvailableModelError"""
        manager = LLMProviderManager(make_test_config())
        with self.assertRaises(LLMNoAvailableModelError):
            manager.resolve_model(['nonexistent_tag'])

    def test_resolve_model_alias_compat(self):
        """原 model_alias 自动转为标签"""
        manager = LLMProviderManager(make_test_config())
        result = manager.resolve_model(['alias_a'])
        self.assertEqual(result[0], 'model-a')


class TestListModels(unittest.TestCase):
    def test_list_models_all(self):
        """无tag返回所有可用模型"""
        manager = LLMProviderManager(make_test_config())
        models = manager.list_models()
        names = [m.name for m in models]
        self.assertIn('model-a', names)
        self.assertIn('model-b', names)
        self.assertNotIn('model-c', names)  # disabled

    def test_list_models_by_tag(self):
        """按tag过滤"""
        manager = LLMProviderManager(make_test_config())
        models = manager.list_models('fast')
        names = [m.name for m in models]
        self.assertIn('model-a', names)
        self.assertIn('model-b', names)
        # Should be sorted by weight descending, model-a first
        self.assertEqual(names[0], 'model-a')

    def test_list_models_exclude_disabled(self):
        """排除禁用模型"""
        manager = LLMProviderManager(make_test_config())
        models = manager.list_models()
        names = [m.name for m in models]
        self.assertNotIn('model-c', names)


class TestConfigUpdate(unittest.TestCase):
    def test_update_config(self):
        """全量配置更新"""
        manager = LLMProviderManager(make_test_config())
        new_config = make_test_config()
        new_config['api']['model-a']['parallel'] = 8
        manager.update_config(new_config)
        models = manager.list_models()
        model_a = [m for m in models if m.name == 'model-a'][0]
        self.assertEqual(model_a.parallel, 8)

    def test_update_api_config(self):
        """单模型配置更新"""
        test_config = make_test_config()
        manager = LLMProviderManager(test_config)
        updated = test_config['api']['model-a'].copy()
        updated.update({'parallel': 10})
        manager.update_api_config('model-a', updated)
        models = manager.list_models()
        model_a = [m for m in models if m.name == 'model-a'][0]
        self.assertEqual(model_a.parallel, 10)

    def test_change_api_status(self):
        """启用/禁用模型"""
        manager = LLMProviderManager(make_test_config())
        # model-c is initially disabled
        self.assertIn('model-c', manager.disabled_models)
        # Enable model-c
        manager.change_api_status('model-c', True)
        self.assertNotIn('model-c', manager.disabled_models)
        models = manager.list_models()
        names = [m.name for m in models]
        self.assertIn('model-c', names)
        # Disable model-a
        manager.change_api_status('model-a', False)
        self.assertIn('model-a', manager.disabled_models)
        models = manager.list_models()
        names = [m.name for m in models]
        self.assertNotIn('model-a', names)

    def test_update_model_tag(self):
        """更新标签"""
        manager = LLMProviderManager(make_test_config())
        manager.update_model_tag('fast', ['model-b', 'model-a'])
        result = manager.resolve_model(['fast'])
        # Now model-b is first, so it has highest weight
        self.assertEqual(result, ['model-b', 'model-a'])


class TestCalculateTagWeights(unittest.TestCase):
    def test_calculate_tag_weights(self):
        """验证权重预计算结果"""
        manager = LLMProviderManager(make_test_config())
        # 'fast' tag: model-a (rank 0), model-b (rank 1)
        fast_weights = manager.model_tag_weights['fast']
        self.assertIn('model-a', fast_weights)
        self.assertIn('model-b', fast_weights)
        # model-a should have higher weight than model-b
        self.assertGreater(fast_weights['model-a'], fast_weights['model-b'])
        # Weights should sum approximately to 1
        self.assertAlmostEqual(sum(fast_weights.values()), 1.0, places=9)
        # 'cheap' tag: model-b only
        cheap_weights = manager.model_tag_weights['cheap']
        self.assertEqual(cheap_weights, {'model-b': 1.0})
        # alias_a tag (from model_alias)
        alias_weights = manager.model_tag_weights.get('alias_a')
        self.assertEqual(alias_weights, {'model-a': 1.0})



def make_nested_test_config():
    """包含嵌套模型列表的测试配置"""
    return {
        'default': {
            'timeout': 30,
            'max_retries': 2,
            'parallel': 1,
            'api_type': 'openai',
        },
        'api': {
            'model-a': {
                'type': 'openai',
                'model': 'model-a',
                'url': 'http://localhost:8000/v1/chat/completions',
                'parallel': 2,
            },
            'model-b': {
                'type': 'openai',
                'model': 'model-b',
                'url': 'http://localhost:8001/v1/chat/completions',
                'parallel': 4,
                'max_input_tokens': 1000,
                'bytes_per_token': 4.0,
            },
            'model-d': {
                'type': 'openai',
                'model': 'model-d',
                'url': 'http://localhost:8003/v1/chat/completions',
                'parallel': 2,
            },
            'model-e': {
                'type': 'openai',
                'model': 'model-e',
                'url': 'http://localhost:8004/v1/chat/completions',
                'parallel': 2,
                'max_input_tokens': 1000,
                'bytes_per_token': 4.0,
            },
            'model-f1': {
                'type': 'openai',
                'model': 'model-f1',
                'url': 'http://localhost:8004/v1/chat/completions',
                'parallel': 2,
                'max_input_tokens': 1000,
                'bytes_per_token': 4.0,
            },
            'model-f2': {
                'type': 'openai',
                'model': 'model-f2',
                'url': 'http://localhost:8004/v1/chat/completions',
                'parallel': 2,
                'max_input_tokens': 1000,
                'bytes_per_token': 4.0,
            },
            'model-f3': {
                'type': 'openai',
                'model': 'model-f3',
                'url': 'http://localhost:8004/v1/chat/completions',
                'parallel': 2,
                'max_input_tokens': 1000,
                'bytes_per_token': 4.0,
            },
        },
        'model_tag': {
            'low_cost': ['model-a', 'model-b', ['model-d', 'model-e']],
            'fast': ['model-a', 'model-b'],
            'cheap': ['model-b', ['model-d', 'model-e']],
            'same': [['model-f1', 'model-f2', 'model-f3']],
        },
        'model_alias': {},
    }


class TestNestedModelList(unittest.TestCase):
    """嵌套模型列表功能测试"""

    def test_nested_tag_weights(self):
        """嵌套组内模型共享排名权重"""
        manager = LLMProviderManager(make_nested_test_config())
        weights = manager.model_tag_weights['low_cost']
        # model-d 和 model-e 共享位置2的权重，应该相等
        self.assertEqual(weights['model-d'], weights['model-e'])
        # model-a (位置0) 权重最高
        self.assertGreater(weights['model-a'], weights['model-b'])
        # model-b (位置1) 权重高于 model-d/e (位置2)
        self.assertGreater(weights['model-b'], weights['model-d'])
        self.assertGreaterEqual(sum(weights.values()), 1.0)

    def test_nested_resolve_model(self):
        """嵌套标签解析返回所有可用模型"""
        manager = LLMProviderManager(make_nested_test_config())
        result = manager.resolve_model(['low_cost'])
        # model-a 权重最高，排第一
        self.assertEqual(result[0], 'model-a')
        # model-d 和 model-e 都应在结果中
        self.assertIn('model-d', result)
        self.assertIn('model-e', result)

    def test_nested_multiple_tags(self):
        """多标签交集匹配嵌套列表"""
        manager = LLMProviderManager(make_nested_test_config())
        # 'low_cost' 有 model-a, model-b, model-d, model-e
        # 'cheap' 有 model-b, model-d, model-e
        # 交集为 model-b, model-d, model-e
        result = manager.resolve_model(['low_cost', 'cheap'])
        self.assertIn('model-b', result)
        self.assertIn('model-d', result)
        self.assertIn('model-e', result)
        self.assertNotIn('model-a', result)

    def test_nested_token_filter(self):
        """Token限制过滤嵌套组模型"""
        manager = LLMProviderManager(make_nested_test_config())
        # model-b 和 model-e 有 max_input_tokens=1000, bytes_per_token=4.0
        long_text = "a" * 5000  # 5000 bytes / 4.0 = 1250 tokens > 1000
        messages = [Message.from_content(long_text, role="user")]
        result = manager.resolve_model(['low_cost'], messages)
        # model-b 和 model-e 被过滤，剩 model-a 和 model-d
        self.assertIn('model-a', result)
        self.assertIn('model-d', result)
        self.assertNotIn('model-b', result)
        self.assertNotIn('model-e', result)

    def test_nested_mixed_list(self):
        """平面与嵌套列表混用兼容"""
        manager = LLMProviderManager(make_nested_test_config())
        # 'fast' 标签为纯平面列表，行为不变
        result_fast = manager.resolve_model(['fast'])
        self.assertEqual(result_fast[0], 'model-a')
        # 'cheap' 标签包含嵌套列表
        result_cheap = manager.resolve_model(['cheap'])
        self.assertEqual(result_cheap[0], 'model-b')

    def test_nested_random_selection(self):
        """同权重模型随机轮流选择"""
        manager = LLMProviderManager(make_nested_test_config())
        # 'cheap' 标签: model-b (位置0, 权重高), [model-d, model-e] (位置1, 共享权重)
        # model-d 和 model-e 权重相同，排序应随机
        results = set()
        for _ in range(20):
            result = manager.resolve_model(['cheap'])
            if len(result) >= 3:
                results.add((result[1], result[2]))
        # 应至少观察到两种不同的顺序（随机性验证）
        self.assertGreater(len(results), 1)

    def test_nested_random_selection_same_group(self):
        """同权重模型随机轮流选择"""
        manager = LLMProviderManager(make_nested_test_config())
        results = set()
        for _ in range(100):
            result = manager.resolve_model(['same'])
            results.add(tuple(result))
        self.assertEqual(len(results), 6)

    def test_nested_disabled_model(self):
        """嵌套组中禁用模型不参与路由"""
        config = make_nested_test_config()
        config['api']['model-d']['enable'] = False
        manager = LLMProviderManager(config)
        result = manager.resolve_model(['low_cost'])
        self.assertNotIn('model-d', result)
        self.assertIn('model-e', result)

    def test_backward_compatibility(self):
        """向后兼容：原有平面列表配置仍正常工作"""
        manager = LLMProviderManager(make_test_config())
        # 原有 fast 标签行为不变
        result = manager.resolve_model(['fast'])
        self.assertEqual(result[0], 'model-a')
        # 原有权重计算不变
        fast_weights = manager.model_tag_weights['fast']
        self.assertGreater(fast_weights['model-a'], fast_weights['model-b'])

    def test_nested_resolve_model_load_balancing(self):
        """同级模型按并发量升序排序"""
        manager = LLMProviderManager(make_nested_test_config())
        # model-f1/f2/f3 在 'same' 标签中共享权重，模拟不同并发负载
        queue_map = {'model-f1': 3, 'model-f2': 1, 'model-f3': 2}
        with mock.patch.object(manager, 'get_queue_length', side_effect=lambda name: queue_map.get(name, 0)):
            result = manager.resolve_model(['same'], context_id='test-context')
        # 并发量少的模型排在前面：f2(1) -> f3(2) -> f1(3)
        self.assertEqual(result, ['model-f2', 'model-f3', 'model-f1'])

    def test_nested_resolve_model_stable_with_context_id(self):
        """相同 context_id 输出稳定"""
        manager = LLMProviderManager(make_nested_test_config())
        results = set()
        for _ in range(10):
            result = manager.resolve_model(['same'], context_id='stable-ctx')
            results.add(tuple(result))
        self.assertEqual(len(results), 1)

    def test_nested_resolve_model_different_context_id(self):
        """不同 context_id 输出多样"""
        manager = LLMProviderManager(make_nested_test_config())
        results = set()
        for i in range(20):
            result = manager.resolve_model(['same'], context_id='ctx-%d' % i)
            results.add(tuple(result))
        self.assertGreater(len(results), 1)


class TestTagRankSyntax(unittest.TestCase):
    """tag:rank 语法功能测试"""

    def test_tag_rank_no_rank_suffix_unchanged(self):
        """不使用 :rank 后缀的标签行为完全不变，rank=1 等同于不使用 rank"""
        manager = LLMProviderManager(make_nested_test_config())
        result_no_rank = manager.resolve_model(['low_cost'], context_id='test-ctx')
        result_rank_1 = manager.resolve_model(['low_cost:1'], context_id='test-ctx')
        self.assertEqual(result_no_rank, result_rank_1)

    def test_tag_rank_rank_2_moves_first_to_end(self):
        """rank=2 将第1项移到末尾，第2项变为最高优先"""
        manager = LLMProviderManager(make_nested_test_config())
        # low_cost: [model-a, model-b, [model-d, model-e]]
        # rank=2 后顺序: [model-b, [model-d, model-e], model-a]
        result = manager.resolve_model(['low_cost:2'], context_id='test-ctx')
        # model-b 现在是最高优先
        self.assertEqual(result[0], 'model-b')
        # model-a 移到末尾
        self.assertEqual(result[-1], 'model-a')

    def test_tag_rank_rank_3_moves_first_two_to_end(self):
        """rank=3 将前2项移到末尾，第3项变为最高优先"""
        manager = LLMProviderManager(make_nested_test_config())
        # low_cost: [model-a, model-b, [model-d, model-e]]
        # rank=3 后顺序: [[model-d, model-e], model-a, model-b]
        # model-d 和 model-e 共享位置0的权重（最高）
        result = manager.resolve_model(['low_cost:3'], context_id='fixed-ctx')
        # model-a 和 model-b 权重低于 model-d/model-e
        self.assertNotIn(result[0], ['model-a', 'model-b'])
        # 第一位应是嵌套组内的模型
        self.assertIn(result[0], ['model-d', 'model-e'])

    def test_tag_rank_overflow_modulo(self):
        """超出范围的 rank 自动取模循环（4个模型传 rank=6 等同于 rank=2）"""
        manager = LLMProviderManager(make_nested_test_config())
        # low_cost 有3个元素
        # rank=4 等同于 rank=1 (shift = 3 % 3 = 0)
        result_rank_1 = manager.resolve_model(['low_cost:1'], context_id='test-ctx')
        result_rank_4 = manager.resolve_model(['low_cost:4'], context_id='test-ctx')
        self.assertEqual(result_rank_1, result_rank_4)
        # rank=5 等同于 rank=2 (shift = 4 % 3 = 1)
        result_rank_2 = manager.resolve_model(['low_cost:2'], context_id='test-ctx')
        result_rank_5 = manager.resolve_model(['low_cost:5'], context_id='test-ctx')
        self.assertEqual(result_rank_2, result_rank_5)

    def test_tag_rank_nested_list_as_single_element(self):
        """嵌套列表视为单个元素整体移位"""
        manager = LLMProviderManager(make_nested_test_config())
        # low_cost: [model-a, model-b, [model-d, model-e]]
        # rank=3 后: [[model-d, model-e], model-a, model-b]
        # 嵌套组 [model-d, model-e] 整体移到第一位
        result = manager.resolve_model(['low_cost:3'], context_id='nested-test')
        # 第一位应是 model-d 或 model-e（嵌套组内的模型）
        self.assertIn(result[0], ['model-d', 'model-e'])
        # model-a 和 model-b 应在后面
        self.assertIn('model-a', result)
        self.assertIn('model-b', result)

    def test_tag_rank_multiple_tags_independent(self):
        """混合使用多个标签，各标签独立调整"""
        manager = LLMProviderManager(make_nested_test_config())
        # low_cost:2 → [model-b, [model-d, model-e], model-a]
        # fast (无rank) → [model-a, model-b]
        # 交集为 model-a, model-b
        result = manager.resolve_model(['low_cost:2', 'fast'], context_id='test-ctx')
        self.assertIn('model-a', result)
        self.assertIn('model-b', result)
        # model-b 在 low_cost:2 中权重最高，在 fast 中权重第二
        # model-a 在 low_cost:2 中权重最低，在 fast 中权重最高
        # 两者权重相近，但 model-b 在 low_cost:2 的提升应使其排名靠前
        self.assertEqual(result[0], 'model-b')

    def test_tag_rank_multiple_tags_both_adjusted(self):
        """多个标签都使用 rank 调整，各自独立计算"""
        manager = LLMProviderManager(make_nested_test_config())
        # low_cost:2 → [model-b, [model-d, model-e], model-a]
        # fast:2 → [model-b, model-a]
        # 交集为 model-a, model-b
        # model-b 在两个标签中都是最高优先
        result = manager.resolve_model(['low_cost:2', 'fast:2'], context_id='test-ctx')
        self.assertEqual(result[0], 'model-b')

    def test_tag_rank_invalid_rank_fallback(self):
        """rank 部分非整数时，整个字符串作为标签名查找"""
        manager = LLMProviderManager(make_nested_test_config())
        # "low_cost:abc" 中 abc 无法转为整数
        # 回退为查找名为 "low_cost:abc" 的标签/模型，不存在则抛出异常
        with self.assertRaises(LLMNoAvailableModelError):
            manager.resolve_model(['low_cost:abc'])

    def test_tag_rank_weights_verification(self):
        """验证 rank 调整后的权重计算正确"""
        manager = LLMProviderManager(make_nested_test_config())
        # 原始 low_cost 权重: model-a 最高
        original_weights = manager.model_tag_weights['low_cost']
        self.assertGreater(original_weights['model-a'], original_weights['model-b'])

        # rank=2 后: model-b 应成为最高权重
        tag_matched = manager._resolve_tag_weights(['low_cost:2'])
        adjusted_weights = tag_matched['low_cost']
        self.assertGreater(adjusted_weights['model-b'], adjusted_weights['model-a'])
        # model-a 移到末尾，权重最低
        self.assertLess(adjusted_weights['model-a'], adjusted_weights['model-b'])

    def test_tag_rank_single_element_tag(self):
        """单元素标签使用 rank 不影响结果"""
        manager = LLMProviderManager(make_nested_test_config())
        # same 标签只有1个嵌套元素 [[model-f1, model-f2, model-f3]]
        # rank=2 时，由于只有1个元素，shift=0，不调整
        result_rank_1 = manager.resolve_model(['same:1'], context_id='test-ctx')
        result_rank_2 = manager.resolve_model(['same:2'], context_id='test-ctx')
        # 两者结果应相同（都包含所有模型）
        self.assertEqual(set(result_rank_1), set(result_rank_2))

    def test_tag_rank_with_direct_model_name(self):
        """tag:rank 语法不适用于直接模型名，rank 被忽略"""
        manager = LLMProviderManager(make_nested_test_config())
        # model-a:2 会被解析为 tag_name="model-a", rank=2
        # 但 model-a 不是标签，是直接模型名，rank 被忽略
        # 精确匹配优先，直接返回 model-a
        result = manager.resolve_model(['model-a:2'])
        self.assertEqual(result, ['model-a'])

    def test_tag_rank_fast_tag(self):
        """测试 fast 标签（纯平面列表）的 rank 调整"""
        manager = LLMProviderManager(make_nested_test_config())
        # fast: [model-a, model-b]
        # rank=2 后: [model-b, model-a]
        result = manager.resolve_model(['fast:2'], context_id='test-ctx')
        self.assertEqual(result[0], 'model-b')
        self.assertEqual(result[1], 'model-a')

    def test_tag_rank_does_not_modify_original_config(self):
        """rank 调整不影响原始配置"""
        manager = LLMProviderManager(make_nested_test_config())
        original_models = list(manager.model_tag['low_cost'])
        # 使用 rank 调整
        manager.resolve_model(['low_cost:2'])
        manager.resolve_model(['low_cost:3'])
        # 原始配置不应改变
        self.assertEqual(manager.model_tag['low_cost'], original_models)
        # 原始权重也不应改变
        original_weights = manager.model_tag_weights['low_cost']
        self.assertGreater(original_weights['model-a'], original_weights['model-b'])

    def test_tag_rank_with_token_filter(self):
        """rank 调整与 Token 限制过滤组合使用"""
        manager = LLMProviderManager(make_nested_test_config())
        # low_cost:2 → [model-b, [model-d, model-e], model-a]
        # model-b 和 model-e 有 max_input_tokens=1000
        long_text = "a" * 5000  # 1250 tokens > 1000
        messages = [Message.from_content(long_text, role="user")]
        result = manager.resolve_model(['low_cost:2'], messages, context_id='test-ctx')
        # model-b 和 model-e 被过滤
        self.assertNotIn('model-b', result)
        self.assertNotIn('model-e', result)
        # model-d (位置1) 和 model-a (位置2) 保留
        # model-d 权重更高（位置1 vs 位置2）
        self.assertEqual(result[0], 'model-d')

    def test_tag_rank_large_rank_value(self):
        """较大的 rank 值正确取模循环"""
        manager = LLMProviderManager(make_nested_test_config())
        # low_cost 有3个元素
        # rank=10: shift = 9 % 3 = 0，等同于 rank=1
        result_rank_1 = manager.resolve_model(['low_cost:1'], context_id='test-ctx')
        result_rank_10 = manager.resolve_model(['low_cost:10'], context_id='test-ctx')
        self.assertEqual(result_rank_1, result_rank_10)
        # rank=11: shift = 10 % 3 = 1，等同于 rank=2
        result_rank_2 = manager.resolve_model(['low_cost:2'], context_id='test-ctx')
        result_rank_11 = manager.resolve_model(['low_cost:11'], context_id='test-ctx')
        self.assertEqual(result_rank_2, result_rank_11)

    def test_tag_rank_zero_rank(self):
        """rank=0 等同于不调整（shift = -1 % N = N-1）"""
        manager = LLMProviderManager(make_nested_test_config())
        # low_cost 有3个元素
        # rank=0: shift = (0-1) % 3 = 2，等同于 rank=3
        result_rank_0 = manager.resolve_model(['low_cost:0'], context_id='test-ctx')
        result_rank_3 = manager.resolve_model(['low_cost:3'], context_id='test-ctx')
        self.assertEqual(result_rank_0, result_rank_3)

    def test_tag_rank_negative_rank(self):
        """负数 rank 正确取模循环"""
        manager = LLMProviderManager(make_nested_test_config())
        # low_cost 有3个元素
        # rank=-1: shift = (-1-1) % 3 = -2 % 3 = 1，等同于 rank=2
        result_rank_neg1 = manager.resolve_model(['low_cost:-1'], context_id='test-ctx')
        result_rank_2 = manager.resolve_model(['low_cost:2'], context_id='test-ctx')
        self.assertEqual(result_rank_neg1, result_rank_2)

    def test_tag_rank_disabled_model_in_shifted_position(self):
        """rank 调整后禁用模型仍被过滤"""
        config = make_nested_test_config()
        # 禁用 model-b（在 low_cost 中位置1）
        config['api']['model-b']['enable'] = False
        manager = LLMProviderManager(config)
        # low_cost:2 → [model-b(disabled), [model-d, model-e], model-a]
        # model-b 被过滤后剩 model-a, model-d, model-e
        result = manager.resolve_model(['low_cost:2'], context_id='test-ctx')
        self.assertNotIn('model-b', result)
        # model-d/model-e 在位置1，model-a 在位置2
        # model-d/model-e 权重高于 model-a
        self.assertIn(result[0], ['model-d', 'model-e'])


if __name__ == '__main__':
    unittest.main()
