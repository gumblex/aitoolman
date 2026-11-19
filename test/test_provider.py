import unittest
import msgspec.json
from unittest import mock

from aitoolman.sse import ServerSentEvent
from aitoolman.provider import OpenAICompatibleFormat, AnthropicFormat, ResponsesFormat, LLMProviderManager
from aitoolman.model import LLMProviderRequest, LLMProviderResponse, LLMNoAvailableModelError, MediaContent, Message


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
        # 无累积快照时回退为最小结束消息
        self.assertEqual(event.response_message,
                         [{"role": "assistant", "content": ""}])

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

    def test_parse_stream_tool_calls_index_merging(self):
        """tool_calls 交错增量按 index 定位合并进 message 快照"""
        response = self._make_response(True)
        chunks = [
            {"choices": [{"index": 0, "delta": {
                "role": "assistant",
                "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                "function": {"name": "add", "arguments": ""}}]}}]},
            {"choices": [{"index": 0, "delta": {"content": "hi"}}]},
            {"choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": '{"a"'}}]}}]},
            {"choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": ': 1}'}}]}}]},
        ]
        for chunk in chunks:
            self.format_strategy.parse_stream_event(
                response, ServerSentEvent(data=msgspec.json.encode(chunk)))

        message = self.format_strategy._choice_messages[0]
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"], "hi")
        self.assertEqual(message["tool_calls"][0]["id"], "call_1")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "add")
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"], '{"a": 1}')

    def test_parse_stream_unknown_delta_fields_merge_into_snapshot(self):
        """delta 未知字段经通用合并自然并入 message 快照"""
        response = self._make_response(True)
        chunk = {"choices": [{"index": 0, "delta": {
            "content": "hi", "custom_field": {"x": 1}}}]}
        self.format_strategy.parse_stream_event(
            response, ServerSentEvent(data=msgspec.json.encode(chunk)))
        self.assertEqual(
            self.format_strategy._choice_messages[0]["custom_field"], {"x": 1})

    def test_parse_stream_done_end_message_structure(self):
        """[DONE] 结束消息为协议真实 assistant message 结构"""
        response = self._make_response(True)
        chunks = [
            {"choices": [{"delta": {"role": "assistant", "content": "Hello",
                                    "reasoning_content": "Think"}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": "add", "arguments": '{"a": 1}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        for chunk in chunks:
            self.format_strategy.parse_stream_event(
                response, ServerSentEvent(data=msgspec.json.encode(chunk)))
        end = self.format_strategy.parse_stream_event(
            response, ServerSentEvent(data=b"[DONE]"))

        self.assertTrue(end.is_end)
        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(len(end.tool_calls), 1)
        self.assertEqual(end.tool_calls[0].id, "call_1")
        self.assertEqual(end.tool_calls[0].name, "add")
        self.assertEqual(end.tool_calls[0].arguments, {"a": 1})
        self.assertEqual(len(end.response_message), 1)
        raw = end.response_message[0]
        self.assertEqual(raw["role"], "assistant")
        self.assertEqual(raw["content"], "Hello")
        self.assertEqual(raw["reasoning_content"], "Think")
        self.assertEqual(raw["tool_calls"][0]["function"]["arguments"], '{"a": 1}')


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

    def test_parse_stream_full_block_lifecycle(self):
        """流式完整块生命周期：text/thinking/tool_use 交错累积，结束消息为完整 message"""
        response = self._make_response(True)

        message_start = {
            "type": "message_start",
            "message": {
                "id": "msg_2", "type": "message", "role": "assistant",
                "model": "claude-3", "content": [],
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 8, "output_tokens": 1}
            }
        }
        self.format_strategy.parse_stream_event(
            response, ServerSentEvent(event="message_start",
                                      data=msgspec.json.encode(message_start)))

        # block 0: text
        self.format_strategy.parse_stream_event(
            response, ServerSentEvent(event="content_block_start",
                                      data=msgspec.json.encode({
                                          "type": "content_block_start", "index": 0,
                                          "content_block": {"type": "text"}})))
        event = self.format_strategy.parse_stream_event(
            response, ServerSentEvent(event="content_block_delta",
                                      data=msgspec.json.encode({
                                          "type": "content_block_delta", "index": 0,
                                          "delta": {"type": "text_delta",
                                                    "text": "Hi"}})))
        self.assertEqual(event.content, "Hi")
        self.format_strategy.parse_stream_event(
            response, ServerSentEvent(event="content_block_stop",
                                      data=msgspec.json.encode({
                                          "type": "content_block_stop", "index": 0})))

        # block 1: thinking + signature
        self.format_strategy.parse_stream_event(
            response, ServerSentEvent(event="content_block_start",
                                      data=msgspec.json.encode({
                                          "type": "content_block_start", "index": 1,
                                          "content_block": {"type": "thinking",
                                                            "thinking": ""}})))
        event = self.format_strategy.parse_stream_event(
            response, ServerSentEvent(event="content_block_delta",
                                      data=msgspec.json.encode({
                                          "type": "content_block_delta", "index": 1,
                                          "delta": {"type": "thinking_delta",
                                                    "thinking": "why"}})))
        self.assertEqual(event.reasoning, "why")
        self.format_strategy.parse_stream_event(
            response, ServerSentEvent(event="content_block_delta",
                                      data=msgspec.json.encode({
                                          "type": "content_block_delta", "index": 1,
                                          "delta": {"type": "signature_delta",
                                                    "signature": "sig=="}})))
        self.format_strategy.parse_stream_event(
            response, ServerSentEvent(event="content_block_stop",
                                      data=msgspec.json.encode({
                                          "type": "content_block_stop", "index": 1})))

        # block 2: tool_use 分段 partial_json
        self.format_strategy.parse_stream_event(
            response, ServerSentEvent(event="content_block_start",
                                      data=msgspec.json.encode({
                                          "type": "content_block_start", "index": 2,
                                          "content_block": {"type": "tool_use",
                                                            "id": "toolu-9",
                                                            "name": "add_task",
                                                            "input": {}}})))
        self.format_strategy.parse_stream_event(
            response, ServerSentEvent(event="content_block_delta",
                                      data=msgspec.json.encode({
                                          "type": "content_block_delta", "index": 2,
                                          "delta": {"type": "input_json_delta",
                                                    "partial_json": '{"cont'}})))
        self.format_strategy.parse_stream_event(
            response, ServerSentEvent(event="content_block_delta",
                                      data=msgspec.json.encode({
                                          "type": "content_block_delta", "index": 2,
                                          "delta": {"type": "input_json_delta",
                                                    "partial_json": 'ent": "a"}'}})))
        self.format_strategy.parse_stream_event(
            response, ServerSentEvent(event="content_block_stop",
                                      data=msgspec.json.encode({
                                          "type": "content_block_stop", "index": 2})))

        # message_delta: stop_reason 与 usage 覆盖
        self.format_strategy.parse_stream_event(
            response, ServerSentEvent(event="message_delta",
                                      data=msgspec.json.encode({
                                          "type": "message_delta",
                                          "delta": {"stop_reason": "tool_use",
                                                    "stop_sequence": None},
                                          "usage": {"input_tokens": 8,
                                                    "output_tokens": 12}})))

        end = self.format_strategy.parse_stream_event(
            response, ServerSentEvent(event="message_stop",
                                      data=msgspec.json.encode({"type": "message_stop"})))
        self.assertTrue(end.is_end)
        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(len(end.tool_calls), 1)
        self.assertEqual(end.tool_calls[0].id, "toolu-9")
        self.assertEqual(end.tool_calls[0].name, "add_task")
        self.assertEqual(end.tool_calls[0].arguments, {"content": "a"})
        self.assertEqual(end.response_message, [{
            "id": "msg_2",
            "type": "message",
            "role": "assistant",
            "model": "claude-3",
            "content": [
                {"type": "text", "text": "Hi"},
                {"type": "thinking", "thinking": "why", "signature": "sig=="},
                {"type": "tool_use", "id": "toolu-9", "name": "add_task",
                 "input": {"content": "a"}}
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 8, "output_tokens": 12}
        }])
        self.assertNotIn("content", response.response_meta)
        self.assertEqual(response.response_meta, {
            "id": "msg_2",
            "type": "message",
            "role": "assistant",
            "model": "claude-3",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 8, "output_tokens": 12}
        })
        self.assertEqual(response.completion_tokens, 12)

        # 内部状态已清空
        self.assertIsNone(self.format_strategy._snapshot)
        self.assertEqual(self.format_strategy._blocks, {})
        self.assertEqual(self.format_strategy._partial_json, {})

    def test_parse_stream_invalid_chunk_returns_empty_event(self):
        """非法 JSON chunk 记录告警并返回空事件"""
        response = self._make_response(True)
        event = self.format_strategy.parse_stream_event(
            response, ServerSentEvent(data=b"{invalid json"))
        self.assertFalse(event.is_end)
        self.assertEqual(event.content, "")


class TestResponsesFormat(unittest.IsolatedAsyncioTestCase):
    """OpenAI Responses API 格式策略测试"""

    def setUp(self):
        self.model_config = {
            "model": "gpt-5",
            "url": "https://api.openai.com/v1/responses",
            "body_options": {"temperature": 0.7}
        }
        self.format_strategy = ResponsesFormat(self.model_config)

    def _make_response(self, stream):
        return LLMProviderResponse(
            client_id="test_client",
            context_id="test_context",
            request_id="test_request",
            model_name="gpt-5",
            stream=stream
        )

    def test_make_request_body_system_to_instructions(self):
        """system 消息合并为 instructions，options 覆盖 body_options"""
        request = LLMProviderRequest(
            client_id="test_client",
            context_id="test_context",
            request_id="test_request",
            model_name="gpt-5",
            messages=[
                Message(role="system", content="system prompt"),
                Message.from_content("hello", role="user"),
            ],
            options={"temperature": 0.2},
            stream=False
        )

        body = self.format_strategy.make_request_body(request)

        self.assertEqual(body["model"], "gpt-5")
        self.assertEqual(body["instructions"], "system prompt")
        self.assertEqual(body["input"],
                         [{"role": "user", "content": "hello"}])
        self.assertEqual(body["stream"], False)
        self.assertEqual(body["temperature"], 0.2)

    def test_make_request_body_with_tools(self):
        """工具定义为平铺结构（无 function 嵌套层）"""
        request = LLMProviderRequest(
            client_id="test_client",
            context_id="test_context",
            request_id="test_request",
            model_name="gpt-5",
            messages=[Message.from_content("hello", role="user")],
            tools={"add": {"description": "加法", "param": {}}},
            stream=False
        )

        body = self.format_strategy.make_request_body(request)

        self.assertEqual(body["tools"][0]["type"], "function")
        self.assertEqual(body["tools"][0]["name"], "add")
        self.assertEqual(body["tools"][0]["description"], "加法")
        self.assertNotIn("function", body["tools"][0])

    def test_serialize_message_raw_value_passthrough(self):
        """reasoning 原始项原样回传，保留 encrypted_content"""
        raw = {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "encrypted_content": "cipher"
        }
        message = Message(role="assistant", content=None, raw_value=raw)
        self.assertEqual(self.format_strategy.serialize_message(message), raw)

    def test_serialize_message_tool_result(self):
        """tool 消息转为 function_call_output 输入项"""
        message = Message(role="tool", content="42", tool_call_id="call_1")
        self.assertEqual(self.format_strategy.serialize_message(message), {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "42"
        })

    def test_serialize_message_function_call_expand(self):
        """assistant 工具调用消息展开为 function_call 输入项列表"""
        message = Message(role="assistant", content=None, raw_value={
            "tool_calls": [{
                "type": "function_call",
                "call_id": "call_1",
                "name": "add",
                "arguments": '{"a": 1}'
            }]
        })
        self.assertEqual(self.format_strategy.serialize_message(message), [{
            "type": "function_call",
            "call_id": "call_1",
            "name": "add",
            "arguments": '{"a": 1}'
        }])

    def test_serialize_message_media_to_input_image(self):
        """媒体内容转为 input_image 输入项"""
        message = Message(
            role="user",
            content="看图",
            media_content=[MediaContent(
                media_type="image",
                raw_value={
                    "type": "input_image",
                    "image_url": "data:image/png;base64,xxx"
                }
            )]
        )
        self.assertEqual(self.format_strategy.serialize_message(message), {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "看图"},
                {"type": "input_image", "image_url": "data:image/png;base64,xxx"}
            ]
        })

    def test_parse_batch_response_output_items(self):
        """按 output 顺序输出 reasoning/message/function_call 三项"""
        response = self._make_response(False)
        response_data = {
            "id": "resp_1",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "想"}],
                    "encrypted_content": "cipher"
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "你好"}]
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "add",
                    "arguments": '{"a": 1}'
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "input_tokens_details": {"cached_tokens": 3}
            }
        }

        self.format_strategy.parse_batch_response(response, response_data)

        self.assertEqual(response.provider_response_id, "resp_1")
        self.assertEqual(response.response_text, "你好")
        self.assertEqual(response.response_reasoning, "想")
        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(response.prompt_tokens, 10)
        self.assertEqual(response.completion_tokens, 5)
        self.assertEqual(response.cached_tokens, 3)
        self.assertNotIn("output", response.response_meta)
        self.assertEqual(len(response.response_tool_calls), 1)
        self.assertEqual(response.response_tool_calls[0].id, "call_1")
        self.assertEqual(response.response_tool_calls[0].name, "add")
        self.assertEqual(response.response_tool_calls[0].arguments, {"a": 1})
        self.assertEqual(len(response.response_message), 3)
        self.assertEqual(
            response.response_message[0].raw_value["type"], "reasoning")
        self.assertEqual(response.response_message[1].content, "你好")
        self.assertEqual(
            response.response_message[2].raw_value["call_id"], "call_1")

    def test_parse_response_message_output_items(self):
        """message 项提取 output_text 文本，reasoning/function_call 项 raw_value 承载原始项"""
        response = self._make_response(False)
        message_item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "你好"}]
        }
        reasoning_item = {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "想"}]
        }
        function_call_item = {
            "type": "function_call",
            "call_id": "call_1",
            "name": "add",
            "arguments": '{"a": 1}'
        }

        message = self.format_strategy.parse_response_message(response, message_item)
        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.content, "你好")
        self.assertEqual(message.raw_value, message_item)

        reasoning = self.format_strategy.parse_response_message(response, reasoning_item)
        self.assertIsNone(reasoning.content)
        self.assertEqual(reasoning.raw_value, reasoning_item)

        function_call = self.format_strategy.parse_response_message(
            response, function_call_item)
        self.assertEqual(function_call.raw_value, function_call_item)

        converted = self.format_strategy.parse_response_message(response, {
            "role": "assistant", "content": "Hi"})
        self.assertEqual(converted.content, "Hi")
        self.assertEqual(
            converted.raw_value, {"role": "assistant", "content": "Hi"})

    def test_parse_stream_response_created(self):
        """response.created 记录提供商响应 ID"""
        response = self._make_response(True)
        event = self.format_strategy.parse_stream_event(
            response,
            ServerSentEvent(data=msgspec.json.encode({
                "type": "response.created",
                "response": {"id": "resp_1"}
            })))
        self.assertFalse(event.is_end)
        self.assertEqual(response.provider_response_id, "resp_1")

    def test_parse_stream_output_text_and_reasoning(self):
        """文本/推理增量累积到缓冲并按事件输出"""
        response = self._make_response(True)
        self.format_strategy.parse_stream_event(
            response,
            ServerSentEvent(data=msgspec.json.encode({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message", "role": "assistant", "content": []}
            })))
        event = self.format_strategy.parse_stream_event(
            response,
            ServerSentEvent(data=msgspec.json.encode({
                "type": "response.output_text.delta",
                "output_index": 0,
                "delta": "Hello"
            })))
        self.assertEqual(event.content, "Hello")
        self.assertEqual(self.format_strategy.stream_message_text[0], "Hello")

        event = self.format_strategy.parse_stream_event(
            response,
            ServerSentEvent(data=msgspec.json.encode({
                "type": "response.reasoning_text.delta",
                "delta": "think"
            })))
        self.assertEqual(event.reasoning, "think")

    def test_parse_stream_reasoning_raw_and_summary(self):
        """reasoning_raw_text/reasoning_summary_text 增量累积，done 修正完整文本"""
        response = self._make_response(True)
        event = self.format_strategy.parse_stream_event(
            response,
            ServerSentEvent(data=msgspec.json.encode({
                "type": "response.reasoning_raw_text.delta",
                "output_index": 1,
                "delta": "raw"
            })))
        self.assertEqual(event.reasoning, "raw")
        self.assertEqual(self.format_strategy.stream_reasoning_raw[1], "raw")

        self.format_strategy.parse_stream_event(
            response,
            ServerSentEvent(data=msgspec.json.encode({
                "type": "response.reasoning_summary_text.delta",
                "output_index": 1,
                "delta": "sum"
            })))
        self.assertEqual(self.format_strategy.stream_reasoning_summary[1], "sum")

        event = self.format_strategy.parse_stream_event(
            response,
            ServerSentEvent(data=msgspec.json.encode({
                "type": "response.reasoning_raw_text.done",
                "output_index": 1,
                "text": "RAW"
            })))
        self.assertFalse(event.is_end)
        self.assertEqual(self.format_strategy.stream_reasoning_raw[1], "RAW")

    def test_parse_stream_function_call_arguments(self):
        """function_call 参数增量累积，done 幂等覆盖最终值"""
        response = self._make_response(True)
        self.format_strategy.parse_stream_event(
            response,
            ServerSentEvent(data=msgspec.json.encode({
                "type": "response.output_item.added",
                "output_index": 2,
                "item": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "add",
                    "arguments": ""
                }
            })))
        self.format_strategy.parse_stream_event(
            response,
            ServerSentEvent(data=msgspec.json.encode({
                "type": "response.function_call_arguments.delta",
                "output_index": 2,
                "delta": '{"a": 1}'
            })))
        self.assertEqual(
            self.format_strategy.stream_tool_index[2]["arguments"], '{"a": 1}')

        self.format_strategy.parse_stream_event(
            response,
            ServerSentEvent(data=msgspec.json.encode({
                "type": "response.function_call_arguments.done",
                "output_index": 2,
                "arguments": '{"a": 2}'
            })))
        self.assertEqual(
            self.format_strategy.stream_tool_index[2]["arguments"], '{"a": 2}')
        self.assertEqual(list(self.format_strategy.stream_tool_index), [2])

    def test_parse_stream_end_event_with_output(self):
        """response.completed 携带完整 output 时返回按序消息列表并清空缓冲"""
        response = self._make_response(True)
        self.format_strategy.parse_stream_event(
            response,
            ServerSentEvent(data=msgspec.json.encode({
                "type": "response.output_text.delta",
                "output_index": 0,
                "delta": "Hello"
            })))
        event = self.format_strategy.parse_stream_event(
            response,
            ServerSentEvent(data=msgspec.json.encode({
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "output": [{
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello"}]
                    }],
                    "usage": {"input_tokens": 10, "output_tokens": 5}
                }
            })))
        self.assertTrue(event.is_end)
        self.assertEqual(response.provider_response_id, "resp_1")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.prompt_tokens, 10)
        self.assertEqual(response.completion_tokens, 5)
        self.assertEqual(len(event.response_message), 1)
        self.assertEqual(event.response_message[0]["content"],
                         [{"type": "output_text", "text": "Hello"}])
        self.assertEqual(self.format_strategy.stream_message_text, {})

    def test_parse_stream_end_event_fallback_to_stream_state(self):
        """结束载荷无 output 时回退到流式累积项构造消息列表"""
        response = self._make_response(True)
        events = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "reasoning", "id": "rs_1", "summary": []}
            },
            {
                "type": "response.reasoning_raw_text.delta",
                "output_index": 0,
                "delta": "raw"
            },
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {"type": "message", "role": "assistant", "content": []}
            },
            {
                "type": "response.output_text.delta",
                "output_index": 1,
                "delta": "Hi"
            }
        ]
        for data in events:
            self.format_strategy.parse_stream_event(
                response, ServerSentEvent(data=msgspec.json.encode(data)))

        event = self.format_strategy.parse_stream_event(
            response,
            ServerSentEvent(data=msgspec.json.encode({
                "type": "response.completed",
                "response": {"id": "resp_1", "status": "completed"}
            })))
        self.assertTrue(event.is_end)
        self.assertEqual(len(event.response_message), 2)
        self.assertEqual(event.response_message[0]["type"], "reasoning")
        self.assertEqual(event.response_message[0]["content"],
                         [{"type": "reasoning_text", "text": "raw"}])
        self.assertEqual(event.response_message[1]["content"],
                         [{"type": "output_text", "text": "Hi"}])

    def test_parse_stream_invalid_event_returns_empty(self):
        """非法 JSON、空行与未知事件返回空事件，不中断流"""
        response = self._make_response(True)
        for data in (b"{invalid json", b"   ",
                     msgspec.json.encode({"type": "response.unknown"})):
            event = self.format_strategy.parse_stream_event(
                response, ServerSentEvent(data=data))
            self.assertFalse(event.is_end)
            self.assertEqual(event.content, "")
            self.assertIsNone(event.response_message)


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
        """rank_by_model 决定 model_rank 旋转模型层还是提供商列表"""
        config = make_test_config()
        config['model_tag']['combo'] = ['deepseek', 'model-a']
        manager = LLMProviderManager(config)
        queue_map = {'provider-1': 1, 'provider-2': 2, 'model-a': 0}
        with mock.patch.object(manager, 'get_queue_length',
                               side_effect=lambda n: queue_map.get(n, 0)):
            by_provider = manager.resolve_model(
                'combo', context_id='test-ctx', model_rank=1)
            by_model = manager.resolve_model(
                'combo', context_id='test-ctx', model_rank=1,
                rank_by_model=True)
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


class TestGlobalRankModelLayer(unittest.TestCase):
    """:N 恒定按模型层旋转测试（标签混合逻辑模型与 api 模型）"""

    def setUp(self):
        config = make_test_config()
        config['model_tag']['combo'] = ['deepseek', 'model-a']
        self.manager = LLMProviderManager(config)
        self.queue_map = {'provider-1': 1, 'provider-2': 2, 'model-a': 0}

    def _resolve(self, tags, **kwargs):
        with mock.patch.object(self.manager, 'get_queue_length',
                               side_effect=lambda n: self.queue_map.get(n, 0)):
            return _names(self.manager.resolve_model(
                tags, context_id='test-ctx', **kwargs))

    def test_no_suffix_default_unchanged(self):
        """无 :N 后缀时行为不变，model_rank 仅旋转提供商列表"""
        self.assertEqual(self._resolve('combo'),
                         ['provider-1', 'provider-2', 'model-a'])
        self.assertEqual(self._resolve('combo', model_rank=1),
                         ['provider-2', 'model-a', 'provider-1'])

    def test_global_rank_rotates_model_layer(self):
        """:N 按模型层旋转后再展开提供商"""
        self.assertEqual(self._resolve('combo:2'),
                         ['model-a', 'provider-1', 'provider-2'])

    def test_global_rank_ignores_rank_by_model(self):
        """:N 的旋转结果不受 rank_by_model 影响"""
        self.assertEqual(
            self._resolve('combo:2'),
            self._resolve('combo:2', rank_by_model=True))

    def test_rank_by_model_true_stacks_model_rank(self):
        """rank_by_model=True 时 model_rank 与 :N 在模型层叠加"""
        self.assertEqual(
            self._resolve('combo:2', model_rank=1, rank_by_model=True),
            ['provider-1', 'provider-2', 'model-a'])

    def test_rank_by_model_false_rank_only_providers(self):
        """rank_by_model=False 时 model_rank 仅旋转展开后的提供商列表"""
        self.assertEqual(
            self._resolve('combo:2', model_rank=1),
            ['provider-1', 'provider-2', 'model-a'])

    def test_wraparound(self):
        """:N 按模型层周期取模回绕"""
        self.assertEqual(self._resolve('combo:3'), self._resolve('combo:1'))
        self.assertEqual(self._resolve('combo:5'), self._resolve('combo:1'))


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
