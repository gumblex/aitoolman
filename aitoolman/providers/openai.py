"""OpenAI Chat Completions 格式策略实现与对应 msgspec 结构定义"""
import logging
from typing import Optional, Dict, List, Any, Union

import msgspec

from .. import postprocess
from .. import sse
from .. import util
from ..model import FinishReason, Message, ToolCall
from .common import (
    LLMFormatStrategy,
    ProviderStreamEvent,
    map_finish_reason,
    serialize_tool_parameters,
)

logger = logging.getLogger(__name__)


class OpenAIPromptTokensDetails(msgspec.Struct):
    cached_tokens: Optional[int] = None


class OpenAIUsage(msgspec.Struct):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    prompt_tokens_details: Optional[OpenAIPromptTokensDetails] = None


class OpenAIChoice(msgspec.Struct):
    delta: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    index: Optional[int] = None


class OpenAIChunk(msgspec.Struct):
    id: Optional[str] = None
    choices: Optional[List[OpenAIChoice]] = None
    usage: Optional[OpenAIUsage] = None


# 每帧重复携带的标识性字段
_OVERWRITE_KEYS = ("id", "role")


def _accumulate_delta(acc: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并 delta 到 acc（对齐官方流式叠加语义）"""
    for key, delta_value in delta.items():
        acc_value = acc.get(key)
        is_delta_list = isinstance(delta_value, list)
        has_indexed_items = is_delta_list and any(
            isinstance(item, dict) and "index" in item
            for item in delta_value
        )

        if acc_value is None:
            if has_indexed_items:
                # 初始化为空列表，向下落入按 index 合并分支
                acc_value = acc[key] = []
            else:
                acc[key] = delta_value
                continue
        elif key == "index" or key == "type" or (key in _OVERWRITE_KEYS and delta_value):
            # index/type 与每帧重复携带的标识性字段（role/id）直接覆盖而非拼接
            acc[key] = delta_value
            continue

        if isinstance(acc_value, (str, int, float)) and isinstance(delta_value, (str, int, float)):
            acc[key] += delta_value
        elif isinstance(acc_value, dict) and isinstance(delta_value, dict):
            # 递归合并就地修改 acc_value 并返回其自身，无需重复赋值 acc[key]
            _accumulate_delta(acc_value, delta_value)
        elif isinstance(acc_value, list) and is_delta_list:
            if all(isinstance(item, (str, int, float)) for item in acc_value) and (
                    acc_value or not has_indexed_items):
                acc[key] += delta_value
            else:
                # acc_value 与 acc[key] 为同一列表对象，复用局部引用减少重复下标
                for entry in delta_value:
                    if not isinstance(entry, dict):
                        raise TypeError("Unexpected type: %s" % type(entry).__name__)
                    try:
                        acc_value[entry["index"]] = _accumulate_delta(
                            acc_value[entry["index"]], entry)
                    except IndexError:
                        acc_value.insert(entry["index"], entry)
    return acc


class OpenAICompatibleFormat(LLMFormatStrategy):

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)

        self.stream_last_raw_event: Optional[sse.ServerSentEvent] = None
        # choice index → message 快照（delta 经通用递归合并累积）
        self._choice_messages: Dict[int, dict] = {}
        self._stream_chunk_decoder = msgspec.json.Decoder(type=OpenAIChunk)

    def serialize_tool_description(self, tools_configs: Dict[str, Dict[str, Any]]) -> List[Dict]:
        tools = []
        for func_name, func_config in tools_configs.items():
            # 构建function核心配置（参数结构解析使用公共函数）
            function = {
                "name": func_name,
                "description": func_config.get("description", ""),
                "parameters": serialize_tool_parameters(func_config.get("param", {}))
            }
            # 构建完整的tool对象（固定type为function，与配置一致）
            tool = {
                "type": func_config.get("type", "function"),
                "function": function
            }
            tools.append(tool)
        return tools

    def parse_tool_calls(self, tool_calls: List[Dict]) -> List[ToolCall]:
        result = []
        for raw_tool in tool_calls:
            # 提取工具调用基础字段
            tool_id = raw_tool.get("id")
            tool_type = raw_tool.get("type", "function")
            function = raw_tool.get("function", {})
            name = function.get("name", '')
            arguments_text = function.get("arguments", "")
            arguments = postprocess.parse_json(arguments_text)

            # 构造ToolCall对象
            result.append(ToolCall(
                id=tool_id,
                type=tool_type,
                name=name,
                arguments_text=arguments_text,
                arguments=arguments
            ))
        return result

    def serialize_message(self, message: Message) -> Dict[str, Any]:
        """将Message转为对应提供商的消息体"""

        # 如果存在原始值，直接返回
        if message.raw_value is not None:
            return message.raw_value

        result: Dict[str, Any] = {"role": message.role}

        if message.media_content:
            # 多媒体内容
            content_list = []

            # 添加文本部分（如果有）
            if message.content:
                content_list.append({
                    "type": "text",
                    "text": message.content
                })

            for media in message.media_content:
                if media.raw_value:
                    content_list.append(media.raw_value)
                else:
                    # 添加多媒体部分
                    media_item = {}
                    # 确定URL或数据
                    url = None
                    if media.url:
                        url = media.url
                    elif media.data and media.mime_type:
                        # 生成data URL
                        url = util.generate_data_url(
                            media.data,
                            media.mime_type
                        )
                    elif media.filename:
                        # 从文件读取并生成data URL
                        mime_type = util.get_mime_type(media.filename)
                        with open(media.filename, 'rb') as f:
                            data = f.read()
                        url = util.generate_data_url(data, mime_type)
                    if not url:
                        raise ValueError("No media content")

                    # 根据媒体类型构建不同的结构
                    if media.media_type in ("image", "video"):
                        url_type = media.media_type + "_url"
                        media_item["type"] = url_type
                        image_url_obj = {"url": url}
                        if media.options:
                            image_url_obj.update(media.options)
                        media_item[url_type] = image_url_obj

                    content_list.append(media_item)
            result["content"] = content_list
        elif message.content is not None:
            result["content"] = message.content

        if message.reasoning_content:
            result["reasoning_content"] = message.reasoning_content
        if message.tool_call_id:
            result["tool_call_id"] = message.tool_call_id
        return result

    def make_request_body(self, request) -> Dict[str, Any]:
        """构建OpenAI风格的请求体（model/messages/stream/tools等）"""
        body = {
            "model": self.model_config["model"],
            "messages": [self.serialize_message(m) for m in request.messages],
            "stream": request.stream
        }
        body.update(
            {**self.model_config.get("body_options", {}), **request.options})
        # 工具调用支持
        if request.tools:
            try:
                body["tools"] = self.serialize_tool_description(request.tools)
            except Exception as ex:
                raise ValueError("Tool description format invalid (%s: %s)" % (type(ex).__name__, ex))
        logger.debug("[%s] request body: %s", request.request_id, body)
        return body

    @classmethod
    def parse_response_message(cls, response, message: Dict[str, Any]) -> Message:
        return Message(
            role="assistant",
            content=message.get("content", ""),
            reasoning_content=message.get("reasoning_content", ""),
            raw_value=message
        )

    def parse_batch_response(self, response, response_data: Dict[str, Any]) -> None:
        """解析OpenAI非流式响应（含工具调用转换）"""
        # response_meta：除 choices 外的字段原样整体存入
        response.response_meta = {k: v for k, v in response_data.items() if k != 'choices'}
        choices = response_data.get("choices", [])
        if not choices:
            raise ValueError("No choices in response")

        choice = choices[0]
        message = choice["message"]

        response.response_text = message.get("content", "")
        response.response_reasoning = message.get("reasoning_content", "")
        response.response_tool_calls = self.parse_tool_calls(message.get("tool_calls") or [])
        response.finish_reason = choice.get("finish_reason", FinishReason.stop.value)

        # Token统计（含缓存命中token）
        usage = response_data.get("usage", {})
        response.prompt_tokens = usage.get("prompt_tokens")
        response.completion_tokens = usage.get("completion_tokens")
        response.cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        response.provider_response_id = response_data.get("id")
        response.response_message = [self.parse_response_message(response, message)]

    def parse_stream_event(self, response, event: sse.ServerSentEvent) -> ProviderStreamEvent:
        """解析OpenAI流式响应（含工具调用增量累积）"""
        # 1. 处理空行
        line_bytes: bytes = event.data.strip()
        if not line_bytes:
            return ProviderStreamEvent.empty()

        # 2. 处理流式结束标记（统一返回工具调用）
        if line_bytes == b"[DONE]":
            # 取最小 index 的 message 快照构造协议真实结束消息
            if self._choice_messages:
                response_message = self._choice_messages[min(self._choice_messages)]
                if not response_message.get("role"):
                    response_message["role"] = "assistant"
                if response_message.get("content") is None:
                    response_message["content"] = ""
            else:
                response_message = {"role": "assistant", "content": ""}
            # 转换累积的工具调用为ToolCall列表
            final_tool_calls = self.parse_tool_calls(response_message.get("tool_calls") or [])

            # 末帧原始事件中的未知字段并入后写入 response_meta
            if self.stream_last_raw_event is not None:
                response_meta = util.decode_json(self.stream_last_raw_event.data)
                if 'choices' in response_meta:
                    del response_meta['choices']
                self.stream_last_raw_event = None
                response.response_meta = response_meta
            # 清空内部快照状态
            self._choice_messages = {}
            # 设置结束标记并返回完整工具调用
            if not response.finish_reason:
                response.finish_reason = FinishReason.stop.value
            return ProviderStreamEvent(True, "", "", final_tool_calls, [response_message])

        # 3. 用 Struct 解析JSON格式的chunk
        try:
            chunk_data = self._stream_chunk_decoder.decode(line_bytes)
        except (msgspec.DecodeError, msgspec.ValidationError):
            logger.warning("[%s: OpenAI] Invalid JSON chunk: %s", response.request_id, line_bytes)
            return ProviderStreamEvent.empty()

        # 首个非空 chunk id 作为提供商响应 ID
        if response.provider_response_id is None and chunk_data.id:
            response.provider_response_id = chunk_data.id

        # 缓存末帧原始事件（流结束后还原未知字段）
        self.stream_last_raw_event = event

        logger.debug('[%s] chunk_data: %s', response.request_id, chunk_data)

        # 4. 处理Token统计（含缓存命中token）
        if chunk_data.usage:
            response.prompt_tokens = chunk_data.usage.prompt_tokens
            response.completion_tokens = chunk_data.usage.completion_tokens
            if chunk_data.usage.prompt_tokens_details is not None:
                response.cached_tokens = chunk_data.usage.prompt_tokens_details.cached_tokens

        # 5. choices 全量按 index 累积 message 快照
        choices = chunk_data.choices or []
        if not choices:
            return ProviderStreamEvent.empty()
        for choice in choices:
            index = choice.index if choice.index is not None else 0
            # _accumulate_delta 仅就地修改 acc、不修改 delta，无需冗余浅拷贝
            self._choice_messages[index] = _accumulate_delta(
                self._choice_messages.get(index) or {}, choice.delta or {})

        # 首个 choice 的 delta 提取增量内容
        choice = choices[0]
        delta: Dict[str, Any] = choice.delta or {}

        # 6. 处理finish_reason
        if choice.finish_reason:
            response.finish_reason = map_finish_reason(choice.finish_reason) or choice.finish_reason

        content_delta = delta.get('content') or ''
        reasoning_delta = delta.get('reasoning_content') or ''

        # 7. 返回当前chunk的内容（工具调用暂不返回）
        return ProviderStreamEvent(
            False,
            content_delta,
            reasoning_delta,
            [], None
        )


# Responses API response.status → 统一 FinishReason 值映射
RESPONSES_STATUS_MAP: Dict[str, str] = {
    "completed": "stop",
    "incomplete": "length",
    "failed": "error: request",
    "cancelled": "cancelled",
}


class ResponsesFormat(LLMFormatStrategy):
    """OpenAI Responses API 格式策略（兼容 DeepSeek Responses 实现）"""

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        # output_index → 流式工具调用缓冲项映射（function_call 增量累积）
        self.stream_tool_index: Dict[Any, Dict[str, Any]] = {}
        # output_index → 输出项副本（reasoning / message / function_call）
        self.stream_output_items: Dict[Any, Dict[str, Any]] = {}
        # output_index → reasoning 原文/摘要文本累积
        self.stream_reasoning_raw: Dict[Any, str] = {}
        self.stream_reasoning_summary: Dict[Any, str] = {}
        # output_index → message 文本累积
        self.stream_message_text: Dict[Any, str] = {}

    def _reset_stream_state(self) -> None:
        """清空流式输出项状态"""
        self.stream_tool_index = {}
        self.stream_output_items = {}
        self.stream_reasoning_raw = {}
        self.stream_reasoning_summary = {}
        self.stream_message_text = {}

    def serialize_tool_description(self, tools_configs: Dict[str, Dict[str, Any]]) -> List[Dict]:
        """Responses 工具定义为平铺结构（无 function 嵌套层）"""
        tools = []
        for func_name, func_config in tools_configs.items():
            tools.append({
                "type": "function",
                "name": func_name,
                "description": func_config.get("description", ""),
                "parameters": serialize_tool_parameters(func_config.get("param", {}))
            })
        return tools

    def parse_tool_calls(self, tool_calls: List[Dict]) -> List[ToolCall]:
        """解析 function_call 输出项（含流式缓冲项）为 ToolCall 列表"""
        result = []
        for raw_tool in tool_calls:
            arguments_text = raw_tool.get("arguments", "")
            result.append(ToolCall(
                id=raw_tool.get("call_id") or raw_tool.get("id"),
                type=raw_tool.get("type", "function"),
                name=raw_tool.get("name", ''),
                arguments_text=arguments_text,
                arguments=postprocess.parse_json(arguments_text)
            ))
        return result

    def serialize_message(self, message: Message) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """将Message转为Responses输入项；assistant工具调用消息展开为输入项列表"""
        # 工具结果 → function_call_output 输入项
        if message.role == "tool":
            return {
                "type": "function_call_output",
                "call_id": message.tool_call_id or "",
                "output": message.content or ""
            }

        if message.raw_value is not None:
            raw = message.raw_value
            tool_calls = raw.get("tool_calls")
            if tool_calls:
                # assistant 工具调用消息展开为 function_call 输入项列表
                items: List[Dict[str, Any]] = []
                if raw.get("content"):
                    items.append({"role": "assistant", "content": raw["content"]})
                for tool_call in tool_calls:
                    if tool_call.get("type") == "function_call" or "call_id" in tool_call:
                        items.append({
                            "type": "function_call",
                            "call_id": tool_call.get("call_id") or tool_call.get("id") or "",
                            "name": tool_call.get("name") or "",
                            "arguments": tool_call.get("arguments") or ""
                        })
                    else:
                        # 兼容 Chat Completions 风格的工具调用结构
                        function = tool_call.get("function") or {}
                        items.append({
                            "type": "function_call",
                            "call_id": tool_call.get("id") or "",
                            "name": function.get("name") or "",
                            "arguments": function.get("arguments") or ""
                        })
                return items
            return raw

        result: Dict[str, Any] = {"role": message.role}

        if message.media_content:
            content_list = []
            if message.content:
                content_list.append({
                    "type": "input_text",
                    "text": message.content
                })
            for media in message.media_content:
                if media.raw_value:
                    content_list.append(media.raw_value)
                    continue
                url = None
                if media.url:
                    url = media.url
                elif media.data and media.mime_type:
                    url = util.generate_data_url(media.data, media.mime_type)
                elif media.filename:
                    mime_type = util.get_mime_type(media.filename)
                    with open(media.filename, 'rb') as f:
                        data = f.read()
                    url = util.generate_data_url(data, mime_type)
                if not url:
                    raise ValueError("No media content")
                media_item: Dict[str, Any] = {"type": "input_image", "image_url": url}
                if media.options:
                    media_item.update(media.options)
                content_list.append(media_item)
            result["content"] = content_list
        elif message.content is not None:
            result["content"] = message.content
        return result

    def make_request_body(self, request) -> Dict[str, Any]:
        """构建Responses API请求体（system消息合并为顶层instructions参数）"""
        input_items: List[Dict[str, Any]] = []
        instructions_parts: List[str] = []
        for m in request.messages:
            if m.role == "system" and m.raw_value is None:
                if m.content:
                    instructions_parts.append(m.content)
                continue
            serialized = self.serialize_message(m)
            if isinstance(serialized, list):
                input_items.extend(serialized)
            else:
                input_items.append(serialized)
        body: Dict[str, Any] = {
            "model": self.model_config["model"],
            "input": input_items,
            "stream": request.stream
        }
        if instructions_parts:
            body["instructions"] = "\n\n".join(instructions_parts)
        body.update(
            {**self.model_config.get("body_options", {}), **request.options})
        # 工具调用支持
        if request.tools:
            try:
                body["tools"] = self.serialize_tool_description(request.tools)
            except Exception as ex:
                raise ValueError("Tool description format invalid (%s: %s)" % (type(ex).__name__, ex))
        logger.debug("[%s] request body: %s", request.request_id, body)
        return body

    def parse_response_message(self, response, message: Dict[str, Any]) -> Message:
        """将Responses输出项转为Message（raw_value承载原始项，message项提取output_text文本）"""
        item_type = message.get("type")
        if item_type in ("reasoning", "function_call"):
            return Message(role="assistant", content=None, raw_value=message)
        if item_type == "message":
            content = message.get("content")
            if isinstance(content, str):
                text = content
            else:
                text = "".join(
                    part.get("text") or ""
                    for part in (content or [])
                    if part.get("type") == "output_text"
                )
            return Message(role="assistant", content=text, raw_value=message)
        # 流式结束构造的 {"role": "assistant", "content": ...} 文本形态
        return Message(
            role=message.get("role", "assistant"),
            content=message.get("content"),
            raw_value=message
        )

    def parse_batch_response(self, response, response_data: Dict[str, Any]) -> None:
        """解析Responses API非流式响应（遍历output数组按type分发）"""
        response.response_meta = {k: v for k, v in response_data.items() if k != 'output'}

        text_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_call_items: List[Dict[str, Any]] = []
        output_items: List[Dict[str, Any]] = list(response_data.get("output") or [])
        for item in output_items:
            item_type = item.get("type", "message")
            if item_type == "message":
                for part in item.get("content") or []:
                    if part.get("type") == "output_text":
                        text_parts.append(part.get("text") or "")
            elif item_type == "reasoning":
                for part in (item.get("summary") or []) + (item.get("content") or []):
                    if part.get("type") in ("summary_text", "reasoning_text"):
                        reasoning_parts.append(part.get("text") or "")
            elif item_type == "function_call":
                tool_call_items.append(item)

        response.response_text = "".join(text_parts)
        response.response_reasoning = "".join(reasoning_parts)
        response.response_tool_calls = self.parse_tool_calls(tool_call_items)

        status = response_data.get("status")
        if status:
            response.finish_reason = RESPONSES_STATUS_MAP.get(status) or status
        # 含 function_call 输出时对齐 Chat Completions 语义，标记 tool_calls
        if tool_call_items and response.finish_reason == FinishReason.stop.value:
            response.finish_reason = FinishReason.tool_calls.value

        # Token统计（含缓存命中token）
        usage = response_data.get("usage") or {}
        response.prompt_tokens = usage.get("input_tokens")
        response.completion_tokens = usage.get("output_tokens")
        response.cached_tokens = (usage.get("input_tokens_details") or {}).get("cached_tokens")

        # 按输出顺序构造可回传上下文的输出项消息
        response.provider_response_id = response_data.get("id")
        response.response_message = [
            self.parse_response_message(response, item)
            for item in output_items
            if item.get("type", "message") in ("message", "reasoning", "function_call")
        ]

    @staticmethod
    def _sort_output_index(output_index: Any) -> Any:
        """output_index 排序键（兼容非整数索引）"""
        if isinstance(output_index, int):
            return (0, output_index)
        return (1, str(output_index))

    def _build_output_messages(self, output_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将按序排列的输出项转换为可回传上下文的原始消息列表"""
        messages: List[Dict[str, Any]] = []
        for item in output_items:
            item_type = item.get("type", "message")
            if item_type == "reasoning":
                messages.append(item)
            elif item_type == "message":
                # 保留原始 item 全部字段原样回传（content 已含完整输出文本）
                messages.append(item)
            elif item_type == "function_call":
                messages.append({
                    "type": "function_call",
                    "call_id": item.get("call_id") or "",
                    "name": item.get("name") or "",
                    "arguments": item.get("arguments") or ""
                })
        return messages

    def _build_stream_messages(self) -> List[Dict[str, Any]]:
        """按 output_index 升序把流式累积的输出项转换为可回传上下文的消息列表"""
        messages: List[Dict[str, Any]] = []
        for output_index in sorted(self.stream_output_items, key=self._sort_output_index):
            item = dict(self.stream_output_items[output_index])
            item_type = item.get("type", "message")
            if item_type == "reasoning":
                raw_text = self.stream_reasoning_raw.get(output_index)
                summary_text = self.stream_reasoning_summary.get(output_index)
                if raw_text:
                    item["content"] = [{"type": "reasoning_text", "text": raw_text}]
                if summary_text:
                    item["summary"] = [{"type": "summary_text", "text": summary_text}]
                messages.append(item)
            elif item_type == "message":
                # 保留原始 item 全部字段，仅以累积文本按 output_text 结构覆盖 content
                item["content"] = [{
                    "type": "output_text",
                    "text": self.stream_message_text.get(output_index, "")
                }]
                messages.append(item)
            elif item_type == "function_call":
                tool_item = self.stream_tool_index.get(output_index) or item
                messages.append({
                    "type": "function_call",
                    "call_id": tool_item.get("call_id") or "",
                    "name": tool_item.get("name") or "",
                    "arguments": tool_item.get("arguments") or ""
                })
        return messages

    def _handle_end_event(self, response, event_type: str, data: Dict[str, Any]) -> ProviderStreamEvent:
        """处理response.completed/incomplete/failed结束事件（携带完整response载荷）"""
        final_tool_calls = self.parse_tool_calls([
            self.stream_tool_index[index]
            for index in sorted(self.stream_tool_index, key=self._sort_output_index)
        ])

        response_data = data.get("response") or {}
        response.response_meta = {k: v for k, v in response_data.items() if k != 'output'}
        if response.provider_response_id is None:
            response.provider_response_id = response_data.get("id")

        status = response_data.get("status") or {
            "response.completed": "completed",
            "response.incomplete": "incomplete",
            "response.failed": "failed"
        }[event_type]
        response.finish_reason = RESPONSES_STATUS_MAP.get(status) or status
        # 含 function_call 输出时对齐 Chat Completions 语义，标记 tool_calls
        if final_tool_calls and response.finish_reason == FinishReason.stop.value:
            response.finish_reason = FinishReason.tool_calls.value

        usage = response_data.get("usage") or {}
        response.prompt_tokens = usage.get("input_tokens")
        response.completion_tokens = usage.get("output_tokens")
        response.cached_tokens = (usage.get("input_tokens_details") or {}).get("cached_tokens")

        # 优先使用结束载荷中的完整输出项，缺失时回退流式累积项
        output_items = response_data.get("output") or []
        if output_items:
            response_messages = self._build_output_messages(output_items)
        else:
            response_messages = self._build_stream_messages()

        # 清空流式状态
        self._reset_stream_state()
        return ProviderStreamEvent(True, "", "", final_tool_calls, response_messages)

    def parse_stream_event(self, response, event: sse.ServerSentEvent) -> ProviderStreamEvent:
        """解析Responses API流式事件（细粒度事件流，以结束事件为流终止条件）"""
        # 1. 处理空行
        line_bytes: bytes = event.data.strip()
        if not line_bytes:
            return ProviderStreamEvent.empty()

        # 2. 解析事件JSON载荷（事件类型在data的type字段）
        try:
            data = util.decode_json(line_bytes)
            if not isinstance(data, dict):
                raise ValueError("event data is not a JSON object")
        except Exception:
            logger.warning("[%s: Responses] Invalid JSON event: %s", response.request_id, line_bytes)
            return ProviderStreamEvent.empty()

        logger.debug('[%s] chunk_data: %s', response.request_id, data)

        event_type = data.get("type", "")

        # 记录提供商响应 ID
        if event_type == "response.created":
            created_response = data.get("response") or {}
            if response.provider_response_id is None:
                response.provider_response_id = created_response.get("id")
            return ProviderStreamEvent.empty()

        # 3. 结束事件（无[DONE]标记，以此作为流终止条件）
        if event_type in ("response.completed", "response.incomplete", "response.failed"):
            return self._handle_end_event(response, event_type, data)

        # 4. 文本/推理增量
        if event_type == "response.output_text.delta":
            content_delta = data.get("delta") or ""
            output_index = data.get("output_index")
            if content_delta:
                self.stream_message_text[output_index] = (
                    self.stream_message_text.get(output_index, "") + content_delta
                )
            return ProviderStreamEvent(False, content_delta, "", [], None)

        if event_type == "response.reasoning_text.delta":
            reasoning_delta = data.get("delta") or ""
            return ProviderStreamEvent(False, "", reasoning_delta, [], None)

        # 思维链原文/摘要增量同样作为 reasoning 文本输出
        if event_type in ("response.reasoning_raw_text.delta", "response.reasoning_summary_text.delta"):
            reasoning_delta = data.get("delta") or ""
            output_index = data.get("output_index")
            if reasoning_delta:
                target = (
                    self.stream_reasoning_raw
                    if event_type == "response.reasoning_raw_text.delta"
                    else self.stream_reasoning_summary
                )
                target[output_index] = target.get(output_index, "") + reasoning_delta
            return ProviderStreamEvent(False, "", reasoning_delta, [], None)

        if event_type in ("response.reasoning_raw_text.done", "response.reasoning_summary_text.done"):
            output_index = data.get("output_index")
            text = data.get("text")
            if text is not None:
                target = (
                    self.stream_reasoning_raw
                    if event_type == "response.reasoning_raw_text.done"
                    else self.stream_reasoning_summary
                )
                target[output_index] = text
            return ProviderStreamEvent.empty()

        # 5. 输出项登记：function_call 累积工具调用，reasoning/message 保留原始项
        if event_type == "response.output_item.added":
            item = data.get("item") or {}
            output_index = data.get("output_index")
            item_type = item.get("type")
            if item_type == "function_call":
                tool_item = {
                    "type": "function_call",
                    "call_id": item.get("call_id") or "",
                    "name": item.get("name") or "",
                    "arguments": item.get("arguments") or ""
                }
                self.stream_tool_index[output_index] = tool_item
            if item_type in ("function_call", "reasoning", "message"):
                self.stream_output_items[output_index] = dict(item)
            return ProviderStreamEvent.empty()

        # 6. 工具调用参数增量累积
        if event_type == "response.function_call_arguments.delta":
            tool_item = self.stream_tool_index.get(data.get("output_index"))
            if tool_item is not None:
                tool_item["arguments"] += data.get("delta") or ""
            return ProviderStreamEvent.empty()

        # 7. 工具调用参数最终值（幂等覆盖，避免与delta重复拼接）
        if event_type == "response.function_call_arguments.done":
            tool_item = self.stream_tool_index.get(data.get("output_index"))
            if tool_item is not None and data.get("arguments") is not None:
                tool_item["arguments"] = data["arguments"]
            return ProviderStreamEvent.empty()

        # 8. 输出项完成（function_call时幂等覆盖最终值，reasoning/message 保留完整项）
        if event_type == "response.output_item.done":
            item = data.get("item") or {}
            output_index = data.get("output_index")
            item_type = item.get("type")
            if item_type == "function_call":
                tool_item = self.stream_tool_index.get(output_index)
                if tool_item is not None:
                    if item.get("call_id"):
                        tool_item["call_id"] = item["call_id"]
                    if item.get("name"):
                        tool_item["name"] = item["name"]
                    if item.get("arguments") is not None:
                        tool_item["arguments"] = item["arguments"]
            if item_type in ("function_call", "reasoning", "message"):
                self.stream_output_items[output_index] = dict(item)
            return ProviderStreamEvent.empty()

        # 其余事件（response.created/in_progress、content_part.*等）忽略
        return ProviderStreamEvent.empty()
