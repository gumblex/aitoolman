import unittest
import msgspec.json
from unittest import mock

from aitoolman.sse import ServerSentEvent
from aitoolman.provider import OpenAICompatibleFormat, AnthropicFormat, LLMProviderManager
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
            response, ServerSentEvent(data=msgspec.json.encode(chunk_data)))
        self.assertFalse(event.is_end)
        self.assertEqual(event.content, "Hello")
        self.assertEqual(event.reasoning, "Think")

    def _make_response(self, stream):
        return LLMProviderResponse(
            client_id="test_client",
            context_id="test_context",
            request_id="test_request",
            model_name="gpt-3.5-turbo",
            stream=stream
        )

    def test_parse_batch_response_meta(self):
        """批式 response_meta：除 choices 外字段原样整体存入"""
        response = self._make_response(False)
        response_data = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-3.5-turbo",
            "choices": [{
                "message": {"content": "Hi"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            "custom_field": {"x": 1}
        }
        self.format_strategy.parse_batch_response(response, response_data)
        self.assertEqual(response.response_meta, {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-3.5-turbo",
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            "custom_field": {"x": 1}
        })
        self.assertNotIn("choices", response.response_meta)

    def test_parse_stream_usage_to_tokens_and_meta(self):
        """流式 usage 写入 token 统计并按 Struct 字段并入 response_meta"""
        response = self._make_response(True)
        chunk = {
            "id": "c1",
            "choices": [{"delta": {}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        self.format_strategy.parse_stream_event(
            response, ServerSentEvent(data=msgspec.json.encode(chunk)))
        self.format_strategy.parse_stream_event(
            response, ServerSentEvent(data=b"[DONE]"))
        self.assertEqual(response.prompt_tokens, 10)
        self.assertEqual(response.completion_tokens, 5)
        self.assertEqual(response.response_meta.get("usage"),
                         {"prompt_tokens": 10, "completion_tokens": 5})

    def test_parse_stream_missing_and_null_fields(self):
        """字段缺失/null 的 chunk（含仅有 id 的 tool_calls）可正常解析"""
        response = self._make_response(True)
        chunk = {
            "choices": [{
                "delta": {"tool_calls": [{"id": "1"}]},
                "finish_reason": None
            }]
        }
        event = self.format_strategy.parse_stream_event(
            response, ServerSentEvent(data=msgspec.json.encode(chunk)))
        self.assertFalse(event.is_end)
        self.assertEqual(event.content, "")

        end = self.format_strategy.parse_stream_event(
            response, ServerSentEvent(data=b"[DONE]"))
        self.assertTrue(end.is_end)
        self.assertEqual(len(end.tool_calls), 1)
        self.assertEqual(end.tool_calls[0].id, "1")

    def test_parse_stream_unknown_fields_from_last_raw_event(self):
        """仅 [DONE] 前最后一帧的未知字段并入 response_meta"""
        response = self._make_response(True)
        chunk1 = {
            "id": "c1",
            "choices": [{"delta": {"content": "hello"}}],
            "custom_field": {"x": 1}
        }
        chunk2 = {"id": "c1", "choices": [{"delta": {"content": " world"}}]}
        self.format_strategy.parse_stream_event(
            response, ServerSentEvent(data=msgspec.json.encode(chunk1)))
        self.format_strategy.parse_stream_event(
            response, ServerSentEvent(data=msgspec.json.encode(chunk2)))
        self.format_strategy.parse_stream_event(
            response, ServerSentEvent(data=b"[DONE]"))
        # 未知字段取自末帧 chunk2；chunk1 的 custom_field 不并入
        self.assertNotIn("custom_field", response.response_meta)
        self.assertEqual(response.response_meta.get("id"), "c1")

    def test_parse_stream_unknown_fields_on_last_chunk(self):
        """末帧自身携带未知字段时并入 response_meta"""
        response = self._make_response(True)
        chunk = {
            "id": "c1",
            "choices": [{"delta": {"content": "hi"}}],
            "custom_field": {"x": 1}
        }
        self.format_strategy.parse_stream_event(
            response, ServerSentEvent(data=msgspec.json.encode(chunk)))
        self.format_strategy.parse_stream_event(
            response, ServerSentEvent(data=b"[DONE]"))
        self.assertEqual(response.response_meta.get("custom_field"), {"x": 1})

    def test_parse_stream_invalid_chunk_returns_empty_event(self):
        """非法 JSON chunk 记录告警并返回空事件，不中断流"""
        response = self._make_response(True)
        event = self.format_strategy.parse_stream_event(
            response, ServerSentEvent(data=b"{invalid json"))
        self.assertFalse(event.is_end)
        self.assertEqual(event.content, "")
        self.assertEqual(event.reasoning, "")
        self.assertEqual(event.tool_calls, [])


class TestAnthropicFormat(unittest.TestCase):
    def setUp(self):
        self.model_config = {
            "model": "claude-3",
            "url": "https://api.anthropic.com/v1/messages"
        }
        self.format_strategy = AnthropicFormat(self.model_config)

    def _make_response(self, stream):
        return LLMProviderResponse(
            client_id="test_client",
            context_id="test_context",
            request_id="test_request",
            model_name="claude-3",
            stream=stream
        )

    def test_parse_batch_response_meta(self):
        """批式 response_meta：除 content 外字段原样整体存入"""
        response = self._make_response(False)
        response_data = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3",
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 7}
        }
        self.format_strategy.parse_batch_response(response, response_data)
        self.assertEqual(response.response_text, "Hello")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.prompt_tokens, 5)
        self.assertEqual(response.completion_tokens, 7)
        self.assertEqual(response.response_meta, {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 7}
        })
        self.assertNotIn("content", response.response_meta)

    def test_parse_batch_response_tool_use(self):
        """批式 tool_use block 转换为 ToolCall，finish_reason=tool_calls"""
        response = self._make_response(False)
        response_data = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3",
            "content": [
                {"type": "text", "text": "calling"},
                {"type": "tool_use", "id": "toolu-1", "name": "add_task",
                 "input": {"content": "todo"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 7}
        }
        self.format_strategy.parse_batch_response(response, response_data)
        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(len(response.response_tool_calls), 1)
        tool = response.response_tool_calls[0]
        self.assertEqual(tool.id, "toolu-1")
        self.assertEqual(tool.name, "add_task")
        self.assertEqual(tool.arguments, {"content": "todo"})

    def test_parse_stream_meta_from_message_events(self):
        """流式从 message_start/message_delta 收集 meta，message_stop 触发写入"""
        response = self._make_response(True)

        message_start = {
            "type": "message_start",
            "message": {
                "id": "msg_1", "type": "message", "role": "assistant",
                "model": "claude-3", "content": [],
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 1}
            }
        }
        event = self.format_strategy.parse_stream_event(
            response, ServerSentEvent(event="message_start",
                                      data=msgspec.json.encode(message_start)))
        self.assertFalse(event.is_end)
        self.assertEqual(response.prompt_tokens, 10)
        self.assertEqual(response.completion_tokens, 1)

        delta_event = {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": "Hi"}
        }
        event = self.format_strategy.parse_stream_event(
            response, ServerSentEvent(event="content_block_delta",
                                      data=msgspec.json.encode(delta_event)))
        self.assertEqual(event.content, "Hi")

        # message_delta 带未知字段 x_extra，验证末帧未知字段并入
        message_delta = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"input_tokens": 10, "output_tokens": 7},
            "x_extra": "v"
        }
        self.format_strategy.parse_stream_event(
            response, ServerSentEvent(event="message_delta",
                                      data=msgspec.json.encode(message_delta)))
        self.assertEqual(response.prompt_tokens, 10)
        self.assertEqual(response.completion_tokens, 7)

        end = self.format_strategy.parse_stream_event(
            response, ServerSentEvent(event="message_stop",
                                      data=msgspec.json.encode({"type": "message_stop"})))
        self.assertTrue(end.is_end)
        self.assertEqual(response.response_meta, {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3",
            "usage": {"input_tokens": 10, "output_tokens": 7},
            "stop_reason": "end_turn",
            "x_extra": "v"
        })

    def test_parse_stream_invalid_chunk_returns_empty_event(self):
        """非法 JSON chunk 记录告警并返回空事件"""
        response = self._make_response(True)
        event = self.format_strategy.parse_stream_event(
            response, ServerSentEvent(data=b"{invalid json"))
        self.assertFalse(event.is_end)
        self.assertEqual(event.content, "")


def _names(result):
    return [name for name, _ in result]


def make_test_config():
    return {
        'default': {
            'timeout': 30,
            'max_retries': 2,
            'parallel': 1,
            'api_type': 'openai',
            'headers': {'X-Default': 'default'},
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
            'provider-1': {
                'type': 'openai',
                'model': 'provider-1',
                'url': 'http://localhost:8010/v1/chat/completions',
                'parallel': 2,
            },
            'provider-2': {
                'type': 'openai',
                'model': 'provider-2',
                'url': 'http://localhost:8011/v1/chat/completions',
                'parallel': 2,
                'max_input_tokens': 1000,
                'bytes_per_token': 4.0,
            },
        },
        'model_tag': {
            'fast': ['model-a', 'model-b'],
            'cheap': ['model-b'],
            'deep': ['deepseek'],
        },
        'model_provider': {
            'deepseek': ['provider-2', 'provider-1'],
        },
    }


class TestResolveModel(unittest.TestCase):
    def setUp(self):
        self.manager = LLMProviderManager(make_test_config())

    def test_exact_api_model(self):
        """精确匹配真实模型名直接返回单元素评分列表"""
        self.assertEqual(self.manager.resolve_model('model-a'),
                         [('model-a', 1.0)])

    def test_tag_returns_scored_tuples(self):
        """标签匹配返回 (提供商, 模型层评分) 元组列表"""
        result = self.manager.resolve_model('fast', context_id='test-ctx')
        self.assertEqual(_names(result)[0], 'model-a')
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[1], float)

    def test_multiple_tags_intersection(self):
        """多标签按交集匹配，模型层评分写入元组"""
        result = self.manager.resolve_model('fast,cheap', context_id='test-ctx')
        self.assertEqual(_names(result), ['model-b'])

    def test_disabled_model_raises(self):
        """禁用模型不参与路由"""
        with self.assertRaises(LLMNoAvailableModelError):
            self.manager.resolve_model('model-c')

    def test_unknown_tag_raises(self):
        """未配置的标签直接报错，不做回退"""
        with self.assertRaises(LLMNoAvailableModelError):
            self.manager.resolve_model('nonexistent_tag')

    def test_token_filter(self):
        """Token 限制过滤 api 模型"""
        long_text = "a" * 5000
        messages = [Message.from_content(long_text, role="user")]
        result = self.manager.resolve_model('fast', messages, context_id='test-ctx')
        self.assertEqual(_names(result), ['model-a'])

    def test_comma_unknown_item_raises(self):
        """逗号分隔的项不自动去空格，未知项直接报错"""
        with self.assertRaises(LLMNoAvailableModelError):
            self.manager.resolve_model('fast, cheap')
        with self.assertRaises(LLMNoAvailableModelError):
            self.manager.resolve_model(' model-a')

    def test_global_rank_rotation(self):
        """全局 :N 后缀对最终提供商列表循环旋转"""
        base = _names(self.manager.resolve_model('fast', context_id='test-ctx'))
        self.assertEqual(base, ['model-a', 'model-b'])
        rotated = _names(self.manager.resolve_model('fast:2', context_id='test-ctx'))
        self.assertEqual(rotated, ['model-b', 'model-a'])

    def test_model_rank_accumulates(self):
        """model_rank 与 :N 的旋转值相加"""
        base = _names(self.manager.resolve_model('fast', context_id='test-ctx'))
        self.assertEqual(
            _names(self.manager.resolve_model(
                'fast', context_id='test-ctx', model_rank=1)),
            _names(self.manager.resolve_model('fast:2', context_id='test-ctx')))
        self.assertEqual(
            _names(self.manager.resolve_model(
                'fast', context_id='test-ctx', model_rank=2)),
            base)

    def test_invalid_global_rank_raises(self):
        """非法 :N 后缀直接报语法错误"""
        with self.assertRaises(LLMNoAvailableModelError):
            self.manager.resolve_model('fast:abc')
        with self.assertRaises(LLMNoAvailableModelError):
            self.manager.resolve_model('fast:')

    def test_slash_is_not_special(self):
        """废除 /N 语法，含斜杠的整串按普通标签名处理"""
        with self.assertRaises(LLMNoAvailableModelError):
            self.manager.resolve_model('fast/2')

    def test_literal_tag_with_slash(self):
        """配置中存在含斜杠的标签名时按字面匹配"""
        config = make_test_config()
        config['model_tag']['tag/abc'] = ['model-a']
        manager = LLMProviderManager(config)
        self.assertEqual(manager.resolve_model('tag/abc'), [('model-a', 0.0)])

    def test_logical_model_providers(self):
        """逻辑模型名展开为排序后的提供商列表"""
        queue_map = {'provider-1': 1, 'provider-2': 2}
        with mock.patch.object(self.manager, 'get_queue_length',
                               side_effect=lambda n: queue_map.get(n, 0)):
            result = self.manager.resolve_model('deepseek', context_id='test-ctx')
        self.assertEqual(_names(result), ['provider-1', 'provider-2'])
        self.assertTrue(all(score == 1.0 for _, score in result))

    def test_logical_model_in_tag(self):
        """标签可引用逻辑模型名并展开提供商"""
        queue_map = {'provider-1': 1, 'provider-2': 2}
        with mock.patch.object(self.manager, 'get_queue_length',
                               side_effect=lambda n: queue_map.get(n, 0)):
            result = self.manager.resolve_model('deep', context_id='test-ctx')
        self.assertEqual(_names(result), ['provider-1', 'provider-2'])

    def test_logical_model_token_filter(self):
        """逻辑模型按各提供商 Token 限制过滤"""
        long_text = "a" * 5000
        messages = [Message.from_content(long_text, role="user")]
        with mock.patch.object(self.manager, 'get_queue_length', return_value=0):
            result = self.manager.resolve_model('deepseek', messages)
        self.assertEqual(_names(result), ['provider-1'])

    def test_logical_model_all_providers_disabled(self):
        """逻辑模型的提供商全部禁用时不可用"""
        config = make_test_config()
        config['api']['provider-1']['enable'] = False
        config['api']['provider-2']['enable'] = False
        manager = LLMProviderManager(config)
        with self.assertRaises(LLMNoAvailableModelError):
            manager.resolve_model('deepseek')

    def test_rank_by_model_rotates_models(self):
        """rank_by_model 决定先旋转模型层还是先展开提供商"""
        config = make_test_config()
        config['model_tag']['combo'] = ['deepseek', 'model-a']
        manager = LLMProviderManager(config)
        queue_map = {'provider-1': 1, 'provider-2': 2, 'model-a': 0}
        with mock.patch.object(manager, 'get_queue_length',
                               side_effect=lambda n: queue_map.get(n, 0)):
            by_provider = manager.resolve_model('combo:2', context_id='test-ctx')
            by_model = manager.resolve_model(
                'combo:2', context_id='test-ctx', rank_by_model=True)
        self.assertEqual(_names(by_provider),
                         ['provider-2', 'model-a', 'provider-1'])
        self.assertEqual(_names(by_model),
                         ['model-a', 'provider-1', 'provider-2'])


class TestListModels(unittest.TestCase):
    def test_list_models_all(self):
        """无 tag 返回所有未禁用 api 模型"""
        manager = LLMProviderManager(make_test_config())
        names = [m.name for m in manager.list_models()]
        self.assertIn('model-a', names)
        self.assertIn('provider-1', names)
        self.assertIn('provider-2', names)
        self.assertNotIn('model-c', names)

    def test_list_models_by_tag(self):
        """按标签返回提供商信息，按权重降序"""
        manager = LLMProviderManager(make_test_config())
        names = [m.name for m in manager.list_models('fast')]
        self.assertEqual(names[0], 'model-a')

    def test_list_models_by_logical_model(self):
        """逻辑模型返回其全部提供商"""
        manager = LLMProviderManager(make_test_config())
        queue_map = {'provider-1': 1, 'provider-2': 2}
        with mock.patch.object(manager, 'get_queue_length',
                               side_effect=lambda n: queue_map.get(n, 0)):
            models = manager.list_models('deepseek')
        self.assertEqual([m.name for m in models],
                         ['provider-1', 'provider-2'])

    def test_provider_model_info_shares_tag_weight(self):
        """提供商的 ModelInfo.tags 通过逻辑模型反查标签权重"""
        manager = LLMProviderManager(make_test_config())
        by_name = {m.name: m for m in manager.list_models('deep')}
        self.assertIn('provider-1', by_name)
        self.assertIn('deep', by_name['provider-1'].tags)


class TestListTags(unittest.TestCase):
    def test_list_tags_basic(self):
        """返回标签及逻辑模型名条目"""
        manager = LLMProviderManager(make_test_config())
        tags = manager.list_tags()
        self.assertEqual(set(tags.keys()), {'fast', 'cheap', 'deep', 'deepseek'})
        self.assertEqual([m.name for m in tags['fast']], ['model-a', 'model-b'])
        self.assertEqual([m.name for m in tags['cheap']], ['model-b'])
        self.assertEqual([m.name for m in tags['deep']],
                         ['provider-2', 'provider-1'])
        self.assertEqual([m.name for m in tags['deepseek']],
                         ['provider-2', 'provider-1'])

    def test_list_tags_flattens_nested(self):
        """嵌套模型列表被展平"""
        manager = LLMProviderManager(make_nested_test_config())
        tags = manager.list_tags()
        self.assertEqual([m.name for m in tags['low_cost']],
                         ['model-a', 'model-b', 'model-d', 'model-e'])
        self.assertEqual([m.name for m in tags['cheap']],
                         ['model-b', 'model-d', 'model-e'])
        self.assertEqual([m.name for m in tags['same']],
                         ['model-f1', 'model-f2', 'model-f3'])

    def test_list_tags_exclude_disabled(self):
        """排除禁用模型"""
        config = make_test_config()
        config['model_tag']['fast'].append('model-c')
        manager = LLMProviderManager(config)
        tags = manager.list_tags()
        self.assertEqual([m.name for m in tags['fast']], ['model-a', 'model-b'])

    def test_list_tags_exclude_disabled_provider(self):
        """逻辑模型的全禁用提供商被排除"""
        config = make_test_config()
        config['api']['provider-2']['enable'] = False
        manager = LLMProviderManager(config)
        tags = manager.list_tags()
        self.assertEqual([m.name for m in tags['deep']], ['provider-1'])
        self.assertEqual([m.name for m in tags['deepseek']], ['provider-1'])

    def test_list_tags_filters_unconfigured(self):
        """标签中未在 api 配置的模型名被过滤"""
        manager = LLMProviderManager(make_test_config())
        manager.model_tag['fast'] = ['model-a', 'model-x']
        tags = manager.list_tags()
        self.assertEqual([m.name for m in tags['fast']], ['model-a'])

    def test_list_tags_keeps_empty_tag(self):
        """标签下模型全部禁用时，标签保留且值为空列表"""
        config = make_test_config()
        config['api']['model-b']['enable'] = False
        manager = LLMProviderManager(config)
        tags = manager.list_tags()
        self.assertIn('cheap', tags)
        self.assertEqual(tags['cheap'], [])
        self.assertEqual([m.name for m in tags['fast']], ['model-a'])


class TestModelProviderConfig(unittest.TestCase):
    def test_conflict_with_api_key(self):
        config = make_test_config()
        config['model_provider']['model-a'] = ['provider-1']
        with self.assertRaises(ValueError):
            LLMProviderManager(config)

    def test_conflict_with_tag(self):
        config = make_test_config()
        config['model_provider']['fast'] = ['provider-1']
        with self.assertRaises(ValueError):
            LLMProviderManager(config)

    def test_invalid_values(self):
        for value in ([], 'provider-1', ['provider-x'], [1]):
            with self.subTest(value=value):
                config = make_test_config()
                config['model_provider']['bad'] = value
                with self.assertRaises(ValueError):
                    LLMProviderManager(config)

    def test_tag_must_reference_known_model(self):
        config = make_test_config()
        config['model_tag']['bad'] = ['unknown']
        with self.assertRaises(ValueError):
            LLMProviderManager(config)


class TestUpdateModelProvider(unittest.TestCase):
    def setUp(self):
        self.manager = LLMProviderManager(make_test_config())

    def test_update_model_provider(self):
        """更新逻辑模型的提供商列表"""
        self.manager.update_model_provider('deepseek', ['provider-1'])
        with mock.patch.object(self.manager, 'get_queue_length', return_value=0):
            result = self.manager.resolve_model('deepseek')
        self.assertEqual(_names(result), ['provider-1'])

    def test_update_invalid_values(self):
        with self.assertRaises(ValueError):
            self.manager.update_model_provider('model-a', ['provider-1'])
        with self.assertRaises(ValueError):
            self.manager.update_model_provider('fast', ['provider-1'])
        with self.assertRaises(ValueError):
            self.manager.update_model_provider('deepseek', [])
        with self.assertRaises(ValueError):
            self.manager.update_model_provider('deepseek', [1])
        with self.assertRaises(ValueError):
            self.manager.update_model_provider('deepseek', ['provider-x'])

    def test_update_api_config_logical_name_rejected(self):
        with self.assertRaises(ValueError):
            self.manager.update_api_config('deepseek', {'url': 'x', 'model': 'x'})

    def test_update_model_tag_logical_name_rejected(self):
        with self.assertRaises(ValueError):
            self.manager.update_model_tag('deepseek', ['model-a'])


class TestUserAgentAndHeaders(unittest.TestCase):
    def setUp(self):
        self.manager = LLMProviderManager(make_test_config())

    def test_user_agent(self):
        from aitoolman import VERSION
        self.assertEqual(
            self.manager.http_client.headers.get('user-agent'),
            f'aitoolman/{VERSION}')

    def test_http_client_has_only_user_agent(self):
        self.assertNotIn('x-default', self.manager.http_client.headers)

    def test_merge_headers(self):
        merged = self.manager._merge_headers({'X-Model': 'model'})
        self.assertEqual(merged['X-Default'], 'default')
        self.assertEqual(merged['X-Model'], 'model')

    def test_model_headers_override_default(self):
        merged = self.manager._merge_headers({'X-Default': 'model'})
        self.assertEqual(merged['X-Default'], 'model')


class TestCalculateTagWeights(unittest.TestCase):
    def test_calculate_tag_weights(self):
        """验证权重预计算结果"""
        manager = LLMProviderManager(make_test_config())
        fast_weights = manager.model_tag_weights['fast']
        self.assertGreater(fast_weights['model-a'], fast_weights['model-b'])
        self.assertAlmostEqual(sum(fast_weights.values()), 1.0, places=9)
        self.assertEqual(manager.model_tag_weights['cheap'], {'model-b': 1.0})
        self.assertEqual(manager.model_tag_weights['deep'], {'deepseek': 1.0})


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
    }


class TestNestedModelList(unittest.TestCase):
    """嵌套模型列表功能测试"""

    def test_nested_tag_weights(self):
        """嵌套组内模型共享排名权重"""
        manager = LLMProviderManager(make_nested_test_config())
        weights = manager.model_tag_weights['low_cost']
        self.assertEqual(weights['model-d'], weights['model-e'])
        self.assertGreater(weights['model-a'], weights['model-b'])
        self.assertGreater(weights['model-b'], weights['model-d'])
        self.assertGreaterEqual(sum(weights.values()), 1.0)

    def test_nested_resolve_model(self):
        """嵌套标签解析返回所有可用模型"""
        manager = LLMProviderManager(make_nested_test_config())
        names = _names(manager.resolve_model('low_cost'))
        self.assertEqual(names[0], 'model-a')
        self.assertIn('model-d', names)
        self.assertIn('model-e', names)

    def test_nested_multiple_tags(self):
        """多标签交集匹配嵌套列表"""
        manager = LLMProviderManager(make_nested_test_config())
        names = _names(manager.resolve_model('low_cost,cheap'))
        self.assertIn('model-b', names)
        self.assertIn('model-d', names)
        self.assertIn('model-e', names)
        self.assertNotIn('model-a', names)

    def test_nested_token_filter(self):
        """Token 限制过滤嵌套组模型"""
        manager = LLMProviderManager(make_nested_test_config())
        long_text = "a" * 5000
        messages = [Message.from_content(long_text, role="user")]
        names = _names(manager.resolve_model('low_cost', messages))
        self.assertIn('model-a', names)
        self.assertIn('model-d', names)
        self.assertNotIn('model-b', names)
        self.assertNotIn('model-e', names)

    def test_nested_mixed_list(self):
        """平面与嵌套列表混用兼容"""
        manager = LLMProviderManager(make_nested_test_config())
        self.assertEqual(_names(manager.resolve_model('fast'))[0], 'model-a')
        self.assertEqual(_names(manager.resolve_model('cheap'))[0], 'model-b')

    def test_nested_random_selection(self):
        """同权重模型随机轮流选择"""
        manager = LLMProviderManager(make_nested_test_config())
        results = set()
        for _ in range(20):
            names = _names(manager.resolve_model('cheap'))
            if len(names) >= 3:
                results.add((names[1], names[2]))
        self.assertGreater(len(results), 1)

    def test_nested_random_selection_same_group(self):
        """同权重模型随机轮流选择"""
        manager = LLMProviderManager(make_nested_test_config())
        results = set()
        for _ in range(100):
            results.add(tuple(_names(manager.resolve_model('same'))))
        self.assertEqual(len(results), 6)

    def test_nested_disabled_model(self):
        """嵌套组中禁用模型不参与路由"""
        config = make_nested_test_config()
        config['api']['model-d']['enable'] = False
        manager = LLMProviderManager(config)
        names = _names(manager.resolve_model('low_cost'))
        self.assertNotIn('model-d', names)
        self.assertIn('model-e', names)

    def test_backward_compatibility(self):
        """向后兼容：原有平面列表配置仍正常工作"""
        manager = LLMProviderManager(make_nested_test_config())
        self.assertEqual(_names(manager.resolve_model('fast'))[0], 'model-a')
        fast_weights = manager.model_tag_weights['fast']
        self.assertGreater(fast_weights['model-a'], fast_weights['model-b'])

    def test_nested_resolve_model_load_balancing(self):
        """同级模型按并发量升序排序"""
        manager = LLMProviderManager(make_nested_test_config())
        queue_map = {'model-f1': 3, 'model-f2': 1, 'model-f3': 2}
        with mock.patch.object(manager, 'get_queue_length',
                               side_effect=lambda name: queue_map.get(name, 0)):
            names = _names(manager.resolve_model('same', context_id='test-context'))
        self.assertEqual(names, ['model-f2', 'model-f3', 'model-f1'])

    def test_nested_resolve_model_stable_with_context_id(self):
        """相同 context_id 输出稳定"""
        manager = LLMProviderManager(make_nested_test_config())
        results = set()
        for _ in range(10):
            results.add(tuple(_names(
                manager.resolve_model('same', context_id='stable-ctx'))))
        self.assertEqual(len(results), 1)

    def test_nested_resolve_model_different_context_id(self):
        """不同 context_id 输出多样"""
        manager = LLMProviderManager(make_nested_test_config())
        results = set()
        for i in range(20):
            results.add(tuple(_names(
                manager.resolve_model('same', context_id='ctx-%d' % i))))
        self.assertGreater(len(results), 1)


class TestGlobalRank(unittest.TestCase):
    """全局 :N 旋转与 model_rank 测试"""

    def setUp(self):
        self.manager = LLMProviderManager(make_nested_test_config())

    def test_no_suffix_equals_rank_1(self):
        self.assertEqual(
            _names(self.manager.resolve_model('low_cost', context_id='test-ctx')),
            _names(self.manager.resolve_model('low_cost:1', context_id='test-ctx')))

    def test_rank_2_rotates_expanded_models(self):
        names = _names(self.manager.resolve_model('low_cost:2', context_id='test-ctx'))
        self.assertEqual(names[0], 'model-b')
        self.assertEqual(names[-1], 'model-a')

    def test_rank_period_is_expanded_length(self):
        # low_cost 展开 4 个模型
        self.assertEqual(
            _names(self.manager.resolve_model('low_cost:5', context_id='test-ctx')),
            _names(self.manager.resolve_model('low_cost:1', context_id='test-ctx')))
        self.assertEqual(
            _names(self.manager.resolve_model('low_cost:6', context_id='test-ctx')),
            _names(self.manager.resolve_model('low_cost:2', context_id='test-ctx')))

    def test_model_rank_accumulates(self):
        self.assertEqual(
            _names(self.manager.resolve_model(
                'low_cost', context_id='test-ctx', model_rank=1)),
            _names(self.manager.resolve_model('low_cost:2', context_id='test-ctx')))

    def test_zero_and_negative_rank(self):
        self.assertEqual(
            _names(self.manager.resolve_model('low_cost:0', context_id='test-ctx')),
            _names(self.manager.resolve_model('low_cost:4', context_id='test-ctx')))
        self.assertEqual(
            _names(self.manager.resolve_model('low_cost:-1', context_id='test-ctx')),
            _names(self.manager.resolve_model('low_cost:3', context_id='test-ctx')))

    def test_multiple_tags_with_global_rank(self):
        names = _names(self.manager.resolve_model(
            'low_cost,fast:2', context_id='test-ctx'))
        self.assertEqual(names[0], 'model-b')

    def test_invalid_rank_raises(self):
        with self.assertRaises(LLMNoAvailableModelError):
            self.manager.resolve_model('low_cost:abc')

    def test_rank_does_not_modify_config(self):
        original = list(self.manager.model_tag['low_cost'])
        self.manager.resolve_model('low_cost:2')
        self.manager.resolve_model('low_cost:3')
        self.assertEqual(self.manager.model_tag['low_cost'], original)

    def test_rank_with_token_filter(self):
        long_text = "a" * 5000
        messages = [Message.from_content(long_text, role="user")]
        names = _names(self.manager.resolve_model('low_cost:2', messages))
        self.assertNotIn('model-b', names)
        self.assertNotIn('model-e', names)
        self.assertEqual(names[0], 'model-d')


class TestConfigUpdate(unittest.TestCase):
    def test_update_config(self):
        """全量配置更新"""
        manager = LLMProviderManager(make_test_config())
        new_config = make_test_config()
        new_config['api']['model-a']['parallel'] = 8
        manager.update_config(new_config)
        model_a = [m for m in manager.list_models() if m.name == 'model-a'][0]
        self.assertEqual(model_a.parallel, 8)

    def test_update_api_config(self):
        """单模型配置更新"""
        test_config = make_test_config()
        manager = LLMProviderManager(test_config)
        updated = test_config['api']['model-a'].copy()
        updated.update({'parallel': 10})
        manager.update_api_config('model-a', updated)
        model_a = [m for m in manager.list_models() if m.name == 'model-a'][0]
        self.assertEqual(model_a.parallel, 10)

    def test_change_api_status(self):
        """启用/禁用模型"""
        manager = LLMProviderManager(make_test_config())
        self.assertIn('model-c', manager.disabled_models)
        manager.change_api_status('model-c', True)
        self.assertNotIn('model-c', manager.disabled_models)
        self.assertIn('model-c', [m.name for m in manager.list_models()])
        manager.change_api_status('model-a', False)
        self.assertIn('model-a', manager.disabled_models)
        self.assertNotIn('model-a', [m.name for m in manager.list_models()])

    def test_update_model_tag(self):
        """更新标签"""
        manager = LLMProviderManager(make_test_config())
        manager.update_model_tag('fast', ['model-b', 'model-a'])
        self.assertEqual(_names(manager.resolve_model('fast')),
                         ['model-b', 'model-a'])


if __name__ == '__main__':
    unittest.main()
