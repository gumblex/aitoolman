"""Anthropic Messages 格式策略实现与对应 msgspec 结构定义"""
import json
import logging
from typing import Optional, Dict, List, Any

import msgspec

from .. import sse
from .. import util
from ..model import FinishReason, Message, ToolCall
from .common import (
    LLMFormatStrategy,
    ProviderStreamEvent,
    map_finish_reason,
    serialize_tool_parameters,
    struct_meta_merge,
)

logger = logging.getLogger(__name__)


class AnthropicUsage(msgspec.Struct):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class AnthropicMessage(msgspec.Struct):
    id: Optional[str] = None
    type: Optional[str] = None
    role: Optional[str] = None
    model: Optional[str] = None
    content: Optional[Any] = None
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Optional[AnthropicUsage] = None


class AnthropicContentBlock(msgspec.Struct):
    type: str
    id: Optional[str] = None
    name: Optional[str] = None
    text: Optional[str] = None
    thinking: Optional[str] = None
    input: Optional[Any] = None


class AnthropicDelta(msgspec.Struct):
    type: Optional[str] = None
    text: Optional[str] = None
    thinking: Optional[str] = None
    partial_json: Optional[str] = None
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None


class AnthropicStreamEvent(msgspec.Struct):
    type: str
    index: Optional[int] = None
    message: Optional[AnthropicMessage] = None
    content_block: Optional[AnthropicContentBlock] = None
    delta: Optional[AnthropicDelta] = None
    usage: Optional[AnthropicUsage] = None


# response_meta 已知字段：按 Struct 字段顺序合并非 None 值（同名后者覆盖）
ANTHROPIC_META_FIELDS = ('id', 'type', 'role', 'model', 'stop_reason', 'stop_sequence', 'usage')
ANTHROPIC_KNOWN_FIELDS = ('type', 'index', 'message', 'content_block', 'delta', 'usage')


class AnthropicFormat(LLMFormatStrategy):
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)

        self.current_tool_index = -1
        self.stream_meta_buffer: Dict[str, Any] = {}
        self.stream_last_raw_event: Optional[sse.ServerSentEvent] = None
        self._stream_chunk_decoder = msgspec.json.Decoder(type=AnthropicStreamEvent)

    def serialize_tool_description(
            self, tools_configs: Dict[str, Dict[str, Any]]
    ) -> List[Dict]:
        """Convert tool configurations to Anthropic's tool format"""
        tools = []
        for func_name, func_config in tools_configs.items():
            tool = {
                "name": func_name,
                "description": func_config.get("description", ""),
                # 参数结构解析使用公共函数
                "input_schema": serialize_tool_parameters(func_config.get("param", {}))
            }
            tools.append(tool)
        return tools

    def parse_tool_calls(self, tool_calls: List[Dict]) -> List[ToolCall]:
        """Parse Anthropic tool calls into ToolCall objects"""
        result = []
        for raw_tool in tool_calls:
            tool_id = raw_tool.get("id")
            name = raw_tool.get("name", '')
            input_data = raw_tool.get("input", {})
            arguments_text = json.dumps(input_data) if input_data else ""
            result.append(ToolCall(
                id=tool_id,
                type="tool_use",
                name=name,
                arguments_text=arguments_text,
                arguments=input_data
            ))
        return result

    def serialize_message(self, message: Message) -> Dict[str, Any]:
        """Serialize a Message to Anthropic's message format"""
        if message.raw_value is not None:
            return message.raw_value

        # Handle tool results (role=tool)
        if message.role == "tool":
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id or "",
                    "content": message.content or ""
                }]
            }

        result: Dict[str, Any] = {"role": message.role}
        content_blocks = []

        # Add text content
        if message.content:
            content_blocks.append({
                "type": "text",
                "text": message.content
            })

        # Add image content (base64 data URLs only)
        for media in (message.media_content or ()):
            url = None
            if media.url:
                url = media.url
            elif media.data and media.mime_type:
                url = util.generate_data_url(
                    media.data,
                    media.mime_type
                )
            elif media.filename:
                mime_type = util.get_mime_type(media.filename)
                with open(media.filename, 'rb') as f:
                    data = f.read()
                url = util.generate_data_url(data, mime_type)

            if url and url.startswith("data:"):
                base64_data = url.split(",", 1)[1] if "," in url else url
                media_type = media.mime_type or "image/jpeg"

                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data
                    }
                })

        if content_blocks:
            result["content"] = content_blocks

        return result

    def make_request_body(self, request) -> Dict[str, Any]:
        """Build Anthropic API request body"""
        body = {
            "model": self.model_config["model"],
            "messages": [],
            "stream": request.stream
        }
        body.update({**self.model_config.get("body_options", {}), **request.options})
        body.setdefault("max_tokens", 20000)

        # Separate system messages and tool results
        system_messages = []
        regular_messages = []
        tool_result_messages = []

        for msg in request.messages:
            if msg.role == "system":
                system_messages.append(msg)
            elif msg.role == "tool":
                tool_result_messages.append(msg)
            else:
                regular_messages.append(msg)

        # Add system messages as top-level system parameter
        if system_messages:
            system_content = []
            for sys_msg in system_messages:
                if sys_msg.content:
                    system_content.append({
                        "type": "text",
                        "text": sys_msg.content
                    })
            if system_content:
                body["system"] = system_content

        # Serialize all messages
        serialized_messages = []
        for msg in regular_messages:
            serialized_messages.append(self.serialize_message(msg))
        for tool_msg in tool_result_messages:
            serialized_messages.append(self.serialize_message(tool_msg))

        body["messages"] = serialized_messages

        # Handle tools
        if request.tools and self.model_config.get("abilities", {}).get("tool"):
            body["tools"] = self.serialize_tool_description(request.tools)

        logger.debug("[%s] request body: %s", request.request_id, body)
        return body

    @classmethod
    def parse_response_message(cls, response, message: Dict[str, Any]) -> Message:
        return Message(
            role="assistant",
            content=response.response_text,
            reasoning_content=response.response_reasoning,
            raw_value=message
        )

    def parse_batch_response(self, response,
                             response_data: Dict[str, Any]) -> None:
        """Parse non-streaming response"""
        # response_meta：除 content 外的字段原样整体存入
        response.response_meta = {k: v for k, v in response_data.items() if k != 'content'}
        response.response_text = ""
        response.response_reasoning = ""

        content_blocks = response_data.get("content") or []
        tool_calls = []

        for block in content_blocks:
            block_type = block.get("type")
            if block_type == "text":
                response.response_text += block.get("text", "")
            elif block_type == "thinking":
                response.response_reasoning += block.get("thinking", "")
            elif block_type == "tool_use":
                tool_calls.append(block)

        if tool_calls:
            response.response_tool_calls = self.parse_tool_calls(tool_calls)
            response.finish_reason = FinishReason.tool_calls.value
        else:
            # Map Anthropic stop_reason to FinishReason
            stop_reason = response_data.get("stop_reason", "unknown")
            mapped_reason = map_finish_reason(stop_reason)
            if mapped_reason:
                response.finish_reason = mapped_reason
            elif stop_reason in FinishReason:
                response.finish_reason = stop_reason
            else:
                logger.warning(
                    "[%s: Anthropic] Unknown stop_reason: %s",
                    response.request_id, stop_reason)
                response.finish_reason = FinishReason.error_request.value
                response.error_text = 'stop_reason: %s' % stop_reason

        response.provider_response_id = response_data.get("id")
        response.response_message = [self.parse_response_message(response, response_data)]
        # Parse token usage（含缓存命中token）
        usage = response_data.get("usage", {})
        response.prompt_tokens = usage.get("input_tokens")
        response.completion_tokens = usage.get("output_tokens")
        response.cached_tokens = usage.get("cache_read_input_tokens")

    def parse_stream_event(self, response,
                           event: sse.ServerSentEvent) -> ProviderStreamEvent:
        """Parse streaming response chunk"""
        logger.debug('[%s] stream event: %s', response.request_id, event)
        # Handle event type
        event_type = event.event
        line_bytes: bytes = event.data.strip()
        if event_type == "error":
            response.finish_reason = FinishReason.error_request.value
            response.error_text = line_bytes.decode('utf-8')
            logger.error("[%s: Anthropic] Stream error: %s",
                         response.request_id, response.error_text)
            return ProviderStreamEvent(True, "", "", [], None)

        # Handle stream end
        if line_bytes == b"[DONE]" or event_type == "message_stop":
            final_tool_calls = self.parse_tool_calls(self.stream_tool_buffers)
            response_message = self.build_stream_response_message()

            # 末帧原始事件中的未知字段并入后写入 response_meta 并清空缓冲
            if self.stream_last_raw_event is not None:
                raw = util.decode_json(self.stream_last_raw_event.data)
                for key, value in raw.items():
                    if key not in ANTHROPIC_KNOWN_FIELDS:
                        self.stream_meta_buffer[key] = value
                self.stream_last_raw_event = None
            response.response_meta = self.stream_meta_buffer

            # Clear buffers
            self.clear_stream_buffers()
            self.stream_meta_buffer = {}
            self.current_tool_index = -1

            if not response.finish_reason:
                response.finish_reason = FinishReason.stop.value

            return ProviderStreamEvent(True, "", "", final_tool_calls, [response_message])

        if not line_bytes:
            return ProviderStreamEvent.empty()

        try:
            chunk_data = self._stream_chunk_decoder.decode(line_bytes)
        except (msgspec.DecodeError, msgspec.ValidationError):
            logger.warning("[%s: Anthropic] Invalid JSON chunk: %s",
                           response.request_id, line_bytes)
            return ProviderStreamEvent.empty()

        # 缓存末帧原始事件（流结束后还原未知字段）
        self.stream_last_raw_event = event

        # message_start: 按已知字段顺序合并 message 中的 meta（content 除外）
        if chunk_data.type == "message_start" and chunk_data.message is not None:
            struct_meta_merge(chunk_data.message, ANTHROPIC_META_FIELDS, self.stream_meta_buffer)
            if chunk_data.message.id is not None:
                response.provider_response_id = chunk_data.message.id
            if chunk_data.message.usage is not None:
                response.prompt_tokens = chunk_data.message.usage.input_tokens
                response.completion_tokens = chunk_data.message.usage.output_tokens
                if chunk_data.message.usage.cache_read_input_tokens is not None:
                    response.cached_tokens = chunk_data.message.usage.cache_read_input_tokens

        # Handle token usage (顶层 usage，如 message_delta 事件)
        usage = chunk_data.usage
        if usage is not None:
            response.prompt_tokens = usage.input_tokens
            response.completion_tokens = usage.output_tokens
            if usage.cache_read_input_tokens is not None:
                response.cached_tokens = usage.cache_read_input_tokens
            self.stream_meta_buffer['usage'] = {
                k: v for k, v in msgspec.structs.asdict(usage).items() if v is not None}

        # message_delta: 合并 delta 中的 stop_reason/stop_sequence
        if chunk_data.type == "message_delta" and chunk_data.delta is not None:
            for meta_key in ('stop_reason', 'stop_sequence'):
                meta_value = getattr(chunk_data.delta, meta_key, None)
                if meta_value is not None:
                    self.stream_meta_buffer[meta_key] = meta_value

        # Handle content_block (start of new content block)
        if chunk_data.content_block is not None:
            block = chunk_data.content_block
            block_type = block.type

            if block_type == "tool_use":
                # New tool call started
                self.current_tool_index += 1
                self.stream_tool_buffers.append({
                    "id": block.id,
                    "name": block.name,
                    "input": {}
                })

        # Handle delta (content updates)
        delta_text = ""
        delta_reasoning = ""

        if chunk_data.delta is not None:
            delta = chunk_data.delta
            delta_type = delta.type

            if delta_type == "text_delta":
                text = delta.text or ''
                delta_text = text
                self.stream_content_buffer += text

            elif delta_type == "thinking_delta":
                thinking = delta.thinking or ''
                delta_reasoning = thinking
                self.stream_reasoning_buffer += thinking

            elif delta_type == "input_json_delta":
                # Update current tool call's input
                if self.current_tool_index >= 0 and self.stream_tool_buffers:
                    partial_json = delta.partial_json or ''
                    current_tool = self.stream_tool_buffers[
                        self.current_tool_index]
                    # Accumulate partial JSON (simplified approach)
                    current_input = current_tool.get("input", {})
                    if isinstance(current_input, dict) and not current_input:
                        # Start with empty dict, will be filled by partial_json
                        current_tool["input"] = partial_json
                    elif isinstance(current_input, str):
                        # Append to existing partial JSON string
                        current_tool["input"] = current_input + partial_json

        return ProviderStreamEvent(
            False,
            delta_text,
            delta_reasoning,
            [],  # Tool calls returned at end
            None
        )
