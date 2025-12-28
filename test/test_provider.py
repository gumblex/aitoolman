import json
import asyncio
import logging
import unittest

from httpx_sse import ServerSentEvent

from aitoolman.provider import OpenAICompatibleFormat
from aitoolman.model import LLMRequest, LLMResponse, Message


# OK
class TestOpenAIFormat(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model_config = {
            "model": "gpt-3.5-turbo",
            "url": "https://api.openai.com/v1/chat/completions",
            "default_options": {"temperature": 0.7}
        }
        self.format_strategy = OpenAICompatibleFormat(self.model_config)

    def test_make_request_body_basic(self):
        request = LLMRequest(
            client_id="test_client",
            context_id="test_context",
            request_id="test_request",
            model_name="gpt-3.5-turbo",
            messages=[Message("Hello", role="user")],
            stream=False
        )

        body = self.format_strategy.make_request_body(request)

        self.assertEqual(body["model"], "gpt-3.5-turbo")
        self.assertEqual(body["messages"],
                         [{"role": "user", "content": "Hello"}])
        self.assertEqual(body["stream"], False)
        self.assertEqual(body["temperature"], 0.7)

    def test_make_request_body_with_tools(self):
        request = LLMRequest(
            client_id="test_client",
            context_id="test_context",
            request_id="test_request",
            model_name="gpt-3.5-turbo",
            messages=[Message("Hello", role="user")],
            tools={"test": {"description": "test", "param": {}}},
            stream=False
        )

        body = self.format_strategy.make_request_body(request)
        self.assertIn("tools", body)

    def test_parse_batch_response_success(self):
        response = LLMResponse(
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
        response = LLMResponse(
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
        response = LLMResponse(
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
        response = LLMResponse(
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

