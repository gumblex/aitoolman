import json
import asyncio
import logging
import unittest
from unittest import mock

from httpx_sse import ServerSentEvent

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
            response, ServerSentEvent(data="[DONE]"))
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


if __name__ == '__main__':
    unittest.main()
