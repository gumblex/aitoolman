import abc
import json
import time
import asyncio
import logging
import typing
from typing import Optional, Dict, Any, List, Callable

from . import postprocess, util
from .model import LLMRequest, LLMResponse, FinishReason, Message, ToolCall
from .resmanager import ResourceManager

import httpx
import httpx_sse


# logger = logging.getLogger(__name__)


def _get_retry_after(response: httpx.Response) -> Optional[int]:
    """解析HTTP响应中的Retry-After头，返回重试时间（秒）"""
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None and retry_after.isdigit():
        return int(retry_after)
    return None


class StreamEvent(typing.NamedTuple):
    is_end: bool
    content: str
    reasoning: str
    tool_calls: Any
    response_message: Any

    @classmethod
    def empty(cls, is_end=False):
        return cls(is_end, '', '', [], None)


class RequestTask(typing.NamedTuple):
    request: LLMRequest
    task: asyncio.Task


class LLMFormatStrategy(abc.ABC):
    """LLM请求/响应格式策略抽象类"""
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config  # 模型配置（包含api_type、url、headers等）

    @abc.abstractmethod
    def serialize_tool_description(self, tools_configs: Dict[str, Dict[str, Any]]) -> List[Dict]:
        """
        转换工具配置格式为Provider格式。配置格式：

        [tools.add_task]
        type = "function"
        description = "添加日程"
        param.datetime.type = "string"
        param.datetime.description = "日期时间，如 2025-12-31 12:34:56"
        param.datetime.required = false
        param.content.type = "string"
        param.content.description = "待办事项"
        param.content.required = true
        """
        pass

    @abc.abstractmethod
    def parse_tool_calls(self, tool_calls: List[Dict]) -> List[ToolCall]:
        """解析大模型返回的工具调用"""
        pass

    @abc.abstractmethod
    def serialize_message(self, message: Message) -> Dict[str, Any]:
        """将Message转为对应提供商的消息体"""
        pass

    @abc.abstractmethod
    def make_request_body(self, request: LLMRequest) -> Dict[str, Any]:
        """构建符合Provider要求的请求体"""
        pass

    @abc.abstractmethod
    def parse_batch_response(self, response: LLMResponse, response_data: Dict[str, Any]) -> None:
        """解析非流式响应，填充LLMResponse"""
        pass

    @abc.abstractmethod
    def parse_stream_event(self, response: LLMResponse, event: httpx_sse.ServerSentEvent) -> StreamEvent:
        """
        解析流式响应的单个chunk
        返回值：(是否结束, 内容片段, 推理片段, 工具调用列表)
        """
        pass


class OpenAICompatibleFormat(LLMFormatStrategy):
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        self.stream_content_buffer = ''
        self.stream_reasoning_buffer = ''
        self.stream_tool_buffers = []

    def serialize_tool_description(self, tools_configs: Dict[str, Dict[str, Any]]) -> List[Dict]:
        tools = []
        for func_name, func_config in tools_configs.items():
            # 构建function核心配置
            function = {
                "name": func_name,
                "description": func_config.get("description", ""),
                "parameters": {
                    "type": "object",  # 参数结构固定为object类型（符合JSON Schema要求）
                    "properties": {},
                    "required": []
                }
            }
            # 解析所有参数配置
            param_configs = func_config.get("param", {})
            for param_name, param_info in param_configs.items():
                param = {"type": param_info.get("type", "string")}
                desc = param_info.get("description", "")
                if desc:
                    param["description"] = desc
                function["parameters"]["properties"][param_name] = param
                if param_info.get("required"):
                    function["parameters"]["required"].append(param_name)
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
            name = function.get("name")
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

        result = {"role": message.role}

        if message.media_content is not None:
            # 多媒体内容
            content_list = []

            # 添加文本部分（如果有）
            if message.content:
                content_list.append({
                    "type": "text",
                    "text": message.content
                })

            if message.media_content.raw_value:
                content_list.append(message.media_content.raw_value)
            else:
                # 添加多媒体部分
                media_item = {}
                # 确定URL或数据
                url = None
                if message.media_content.url:
                    url = message.media_content.url
                elif message.media_content.data and message.media_content.mime_type:
                    # 生成data URL
                    url = util.generate_data_url(
                        message.media_content.data,
                        message.media_content.mime_type
                    )
                elif message.media_content.filename:
                    # 从文件读取并生成data URL
                    mime_type = util.get_mime_type(message.media_content.filename)
                    with open(message.media_content.filename, 'rb') as f:
                        data = f.read()
                    url = util.generate_data_url(data, mime_type)
                if not url:
                    raise ValueError("No media content")

                # 根据媒体类型构建不同的结构
                if message.media_content.media_type in ("image", "video"):
                    url_type = message.media_content.media_type + "_url"
                    media_item["type"] = url_type
                    image_url_obj = {"url": url}
                    if message.media_content.options:
                        image_url_obj.update(message.media_content.options)
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

    def make_request_body(self, request: LLMRequest) -> Dict[str, Any]:
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
        self.logger.debug("[%s] request body: %s", request.request_id, body)
        return body

    def parse_batch_response(self, response: LLMResponse, response_data: Dict[str, Any]) -> None:
        """解析OpenAI非流式响应（含工具调用转换）"""
        choices = response_data.get("choices", [])
        if not choices:
            raise ValueError("No choices in response")

        choice = choices[0]
        message = choice["message"]

        response.response_text = message.get("content", "")
        response.response_reasoning = message.get("reasoning_content", "")
        response.response_tool_calls = self.parse_tool_calls(message.get("tool_calls", []))
        response.finish_reason = choice.get("finish_reason", FinishReason.stop.value)

        # Token统计
        usage = response_data.get("usage", {})
        response.prompt_tokens = usage.get("prompt_tokens")
        response.completion_tokens = usage.get("completion_tokens")
        response.response_message = message

    def parse_stream_event(self, response: LLMResponse, event: httpx_sse.ServerSentEvent) -> StreamEvent:
        """解析OpenAI流式响应（含工具调用增量累积）"""
        # 1. 处理空行
        line = event.data.strip()
        if not line:
            return StreamEvent.empty()

        # 2. 处理流式结束标记（统一返回工具调用）
        if line == "[DONE]":
            # 转换累积的工具调用为ToolCall列表
            final_tool_calls = self.parse_tool_calls(self.stream_tool_buffers)
            response_message = {
                "content": self.stream_content_buffer,
                "tool_calls": self.stream_tool_buffers
            }
            if self.stream_reasoning_buffer:
                response_message["reasoning_content"] = self.stream_reasoning_buffer

            # 清空缓冲区
            self.stream_tool_buffers.clear()
            self.stream_content_buffer = ""
            self.stream_reasoning_buffer = ""
            # 设置结束标记并返回完整工具调用
            if not response.finish_reason:
                response.finish_reason = FinishReason.stop.value
            return StreamEvent(True, "", "", final_tool_calls, response_message)

        # 3. 解析JSON格式的chunk
        try:
            chunk_data = json.loads(line)
        except json.JSONDecodeError:
            self.logger.warning("[%s: OpenAI] Invalid JSON chunk: %s", response.request_id, line)
            return StreamEvent.empty()

        self.logger.debug('[%s] chunk_data: %s', response.request_id, chunk_data)

        # 4. 处理Token统计
        usage = chunk_data.get("usage")
        if usage:
            response.prompt_tokens = usage.get("prompt_tokens")
            response.completion_tokens = usage.get("completion_tokens")

        # 5. 提取delta内容
        choices = chunk_data.get("choices", [])
        if not choices:
            return StreamEvent.empty()
        choice = choices[0]
        delta = choice.get("delta", {})

        # 6. 处理finish_reason
        if choice.get("finish_reason"):
            response.finish_reason = choice["finish_reason"]

        # 7. 增量累积工具调用内容
        for tool_delta in delta.get("tool_calls", []):
            tool_id = tool_delta.get("id")
            if tool_id:
                tool_item = tool_delta.copy()
                self.stream_tool_buffers.append(tool_item)
                continue
            elif not self.stream_tool_buffers:
                self.logger.warning("[%s] Function delta without id: %s", response.request_id, delta)
                continue

            tool_item = self.stream_tool_buffers[-1]
            fn_delta = tool_delta.get("function", {})
            fn_item = tool_item.get("function", {})
            tool_item['function'] = fn_item
            for key, value in fn_delta.items():
                fn_item[key] = fn_item.get(key, '') + value

        content_delta = delta.get("content", "")
        reasoning_delta = delta.get("reasoning_content", "")

        if content_delta:
            self.stream_content_buffer += content_delta
        if reasoning_delta:
            self.stream_reasoning_buffer += reasoning_delta

        # 8. 返回当前chunk的内容（工具调用暂不返回）
        return StreamEvent(
            False,
            content_delta,
            reasoning_delta,
            [], None
        )


class AnthropicFormat(LLMFormatStrategy):
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        self.stream_content_buffer = ''
        self.stream_reasoning_buffer = ''
        self.stream_tool_buffers = []
        self.current_tool_index = -1

    def serialize_tool_description(
            self, tools_configs: Dict[str, Dict[str, Any]]
    ) -> List[Dict]:
        """Convert tool configurations to Anthropic's tool format"""
        tools = []
        for func_name, func_config in tools_configs.items():
            tool = {
                "name": func_name,
                "description": func_config.get("description", ""),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            param_configs = func_config.get("param", {})
            for param_name, param_info in param_configs.items():
                param = {"type": param_info.get("type", "string")}
                desc = param_info.get("description", "")
                if desc:
                    param["description"] = desc
                tool["input_schema"]["properties"][param_name] = param
                if param_info.get("required"):
                    tool["input_schema"]["required"].append(param_name)
            tools.append(tool)
        return tools

    def parse_tool_calls(self, tool_calls: List[Dict]) -> List[ToolCall]:
        """Parse Anthropic tool calls into ToolCall objects"""
        result = []
        for raw_tool in tool_calls:
            tool_id = raw_tool.get("id")
            name = raw_tool.get("name")
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

        result = {"role": message.role}
        content_blocks = []

        # Add text content
        if message.content:
            content_blocks.append({
                "type": "text",
                "text": message.content
            })

        # Add image content (base64 data URLs only)
        if message.media_content:
            url = None
            if message.media_content.url:
                url = message.media_content.url
            elif message.media_content.data and message.media_content.mime_type:
                url = util.generate_data_url(
                    message.media_content.data,
                    message.media_content.mime_type
                )
            elif message.media_content.filename:
                mime_type = util.get_mime_type(message.media_content.filename)
                with open(message.media_content.filename, 'rb') as f:
                    data = f.read()
                url = util.generate_data_url(data, mime_type)

            if url and url.startswith("data:"):
                base64_data = url.split(",", 1)[1] if "," in url else url
                media_type = message.media_content.mime_type or "image/jpeg"

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

    def make_request_body(self, request: LLMRequest) -> Dict[str, Any]:
        """Build Anthropic API request body"""
        body = {
            "model": self.model_config["model"],
            "messages": [],
            "stream": request.stream
        }
        body.update(request.options)
        body.setdefault("max_tokens", 64000)

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

        self.logger.debug("[%s] request body: %s", request.request_id, body)
        return body

    def parse_batch_response(self, response: LLMResponse,
                             response_data: Dict[str, Any]) -> None:
        """Parse non-streaming response"""
        response.response_message = response_data
        response.response_text = ""
        response.response_reasoning = ""

        content_blocks = response_data.get("content", [])
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
            if stop_reason == "end_turn":
                response.finish_reason = FinishReason.stop.value
            elif stop_reason == "max_tokens":
                response.finish_reason = FinishReason.length.value
            elif stop_reason == "tool_use":
                response.finish_reason = FinishReason.tool_calls.value
            else:
                response.finish_reason = stop_reason

        # Parse token usage
        usage = response_data.get("usage", {})
        response.prompt_tokens = usage.get("input_tokens")
        response.completion_tokens = usage.get("output_tokens")

    def parse_stream_event(self, response: LLMResponse,
                           event: httpx_sse.ServerSentEvent) -> StreamEvent:
        """Parse streaming response chunk"""
        self.logger.debug('[%s] stream event: %s', response.request_id, event)
        # Handle event type
        event_type = event.event
        line = event.data.strip()
        if event_type == "error":
            response.finish_reason = FinishReason.error_request.value
            response.error_text = line
            self.logger.error("[%s: Anthropic] Stream error: %s",
                         response.request_id, line)
            return StreamEvent(True, "", "", [], None)

        # Handle stream end
        if line == "[DONE]" or event_type == "message_stop":
            final_tool_calls = self.parse_tool_calls(self.stream_tool_buffers)
            response_message = {
                "content": self.stream_content_buffer,
                "tool_calls": self.stream_tool_buffers
            }
            if self.stream_reasoning_buffer:
                response_message[
                    "reasoning_content"] = self.stream_reasoning_buffer

            # Clear buffers
            self.stream_tool_buffers = []
            self.stream_content_buffer = ""
            self.stream_reasoning_buffer = ""
            self.current_tool_index = -1

            if not response.finish_reason:
                response.finish_reason = FinishReason.stop.value

            return StreamEvent(True, "", "", final_tool_calls, response_message)

        if not line:
            return StreamEvent.empty()

        try:
            chunk_data = json.loads(line)
        except json.JSONDecodeError:
            self.logger.warning("[%s: Anthropic] Invalid JSON chunk: %s",
                           response.request_id, line)
            return StreamEvent.empty()

        # Handle token usage
        usage = chunk_data.get("usage")
        if usage:
            response.prompt_tokens = usage.get("input_tokens")
            response.completion_tokens = usage.get("output_tokens")

        # Handle content_block (start of new content block)
        if "content_block" in chunk_data:
            block = chunk_data["content_block"]
            block_type = block.get("type")

            if block_type == "tool_use":
                # New tool call started
                self.current_tool_index += 1
                self.stream_tool_buffers.append({
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": {}
                })

        # Handle delta (content updates)
        delta_text = ""
        delta_reasoning = ""

        if "delta" in chunk_data:
            delta = chunk_data["delta"]
            delta_type = delta.get("type")

            if delta_type == "text_delta":
                text = delta.get("text", "")
                delta_text = text
                self.stream_content_buffer += text

            elif delta_type == "thinking_delta":
                thinking = delta.get("thinking", "")
                delta_reasoning = thinking
                self.stream_reasoning_buffer += thinking

            elif delta_type == "input_json_delta":
                # Update current tool call's input
                if self.current_tool_index >= 0 and self.stream_tool_buffers:
                    partial_json = delta.get("partial_json", "")
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

        return StreamEvent(
            False,
            delta_text,
            delta_reasoning,
            [],  # Tool calls returned at end
            None
        )


class LLMProviderManager:
    """LLM提供商管理器，负责API调用、资源管理与请求重试"""
    format_strategies = {
        "openai": OpenAICompatibleFormat,
        "anthropic": AnthropicFormat,
    }

    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.default_config = config['default']
        self.api_config = config['api']
        self.model_alias = config.get('model_alias', {})
        for key in self.api_config.keys():
            if key not in self.model_alias:
                self.model_alias[key] = key

        self.timeout = self.default_config['timeout']
        self.max_retries = self.default_config['max_retries']
        self.default_parallel = self.default_config['parallel']
        self.retry_duration = self.default_config.get('retry_duration', 0.5)
        self.retry_factor = self.default_config.get('retry_factor', 1.5)

        self.http_client = httpx.AsyncClient(
            timeout=self.timeout, http2=True,
            headers=self.default_config.get('headers', {})
        )

        # 资源管理器（控制模型并行度）
        self.resource_manager = ResourceManager(self._parse_model_capacities())

        # 活跃请求列表（用于取消操作）
        self.active_requests: Dict[str, RequestTask] = {}

    def _parse_model_capacities(self) -> Dict[str, int]:
        """从配置中解析各模型的并行处理能力"""
        capacities = {}
        for model_name, model_config in self.api_config.items():
            capacities[model_name] = model_config.get("parallel", self.default_parallel)
        return capacities

    async def initialize(self):
        """初始化ZeroMQ sockets和HTTP客户端"""
        await self.http_client.__aenter__()

    async def _end_request_with_error(
        self, request: LLMRequest, response: LLMResponse, error_text: str, finish_reason: FinishReason
    ):
        response.error_text = error_text
        response.finish_reason = finish_reason.value
        is_fragment = request.stream
        if is_fragment:
            if request.reasoning_channel:
                await request.reasoning_channel.write_fragment("", end=True)
            if request.response_channel:
                await request.response_channel.write_fragment("", end=True)
        else:
            if request.reasoning_channel:
                await request.reasoning_channel.write_message(None)
            if request.response_channel:
                await request.response_channel.write_message(None)

    async def _async_post_with_retry(self, request: LLMRequest, url: str, **kwargs) -> Dict:
        """带重试的HTTP POST请求（处理限流、超时等异常）"""
        retries = 0
        while True:
            retry_after = None
            status_code = None

            try:
                response = await self.http_client.post(url, **kwargs)
                status_code = response.status_code
                response.raise_for_status()  # 抛出HTTP错误（如4xx/5xx）
                return response.json()

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                # 仅重试特定状态码：408（超时）、429（限流）、5xx（服务器错误）
                should_retry = status_code in (408, 429) or 500 <= status_code < 600
                if not should_retry or retries >= self.max_retries:
                    raise e
                retry_after = _get_retry_after(e.response)

            except (httpx.ReadTimeout, httpx.ReadError, httpx.ConnectError) as e:
                if retries >= self.max_retries:
                    raise e

            # 计算重试间隔（指数退避）
            if retry_after is None:
                retry_after = self.retry_duration * (self.retry_factor ** retries)

            self.logger.warning(
                "[%s] Retry (%d/%d) after %.1fs, status code: %s",
                request.request_id, retries + 1, self.max_retries, retry_after, status_code
            )
            await asyncio.sleep(retry_after)
            retries += 1

    async def _handle_batch_request(
            self, request: LLMRequest, response: LLMResponse,
            model_config: Dict, request_body: Dict,
            format_strategy: LLMFormatStrategy
    ):
        """处理非流式请求（完整响应）"""
        start_time = time.monotonic()
        try:
            # 发送带重试的HTTP请求
            response_data = await self._async_post_with_retry(
                request=request,
                url=model_config["url"],
                json=request_body,
                headers=model_config.get("headers", {}),
                timeout=model_config.get("timeout", self.timeout)
            )
        except Exception as e:
            await self._end_request_with_error(
                request, response, f"{type(e).__name__}: {str(e)}", FinishReason.error_request
            )
            return

        # 时间统计
        total_duration = time.monotonic() - start_time
        response.total_response_time = total_duration
        response.time_to_first_token = total_duration  # 非流式请求无部分响应，首次响应时间等于总时间

        try:
            format_strategy.parse_batch_response(response, response_data)
        except Exception as e:
            await self._end_request_with_error(
                request, response,
                f"Failed to parse batch response: {str(e)}",
                FinishReason.error_format
            )
            return

        # 发送完整响应到Channel
        if request.reasoning_channel and response.response_reasoning:
            await request.reasoning_channel.write_message(response.response_reasoning)
        if request.response_channel:
            await request.response_channel.write_message(response.response_text)

    async def _handle_stream_request(
            self, request: LLMRequest, response: LLMResponse,
            model_config: Dict, request_body: Dict,
            format_strategy: LLMFormatStrategy
    ):
        """处理流式请求（分块响应）"""
        start_time = time.monotonic()
        first_token_received = False
        thinking_end = False

        try:
            # 发送流式HTTP请求
            async with self.http_client.stream(
                method="POST",
                url=model_config["url"],
                json=request_body,
                headers=model_config.get("headers", {}),
                timeout=model_config.get("timeout", self.timeout)
            ) as http_response:
                http_response.raise_for_status()

                event_source = httpx_sse.EventSource(http_response)
                # 逐行解析流式响应（SSE格式：data: ...）
                async for sse_event in event_source.aiter_sse():
                    if request.is_cancelled:
                        break
                    event = format_strategy.parse_stream_event(response, sse_event)

                    # 记录首次响应时间（TTFT）
                    if not first_token_received and (event.content or event.reasoning):
                        response.time_to_first_token = time.monotonic() - start_time
                        first_token_received = True

                    # 累积响应内容到LLMResponse
                    response.response_text += event.content
                    response.response_reasoning += event.reasoning
                    if event.tool_calls:
                        response.response_tool_calls = (response.response_tool_calls or []) + event.tool_calls

                    if event.response_message:
                        response.response_message = event.response_message

                    # 发送分块内容到Channel
                    if request.reasoning_channel and event.reasoning:
                        await request.reasoning_channel.write_fragment(event.reasoning)
                    if not thinking_end and event.content:
                        if request.reasoning_channel:
                            await request.reasoning_channel.write_fragment("", end=True)
                        thinking_end = True
                    elif thinking_end and event.reasoning:
                        # reasoning after content
                        thinking_end = False
                    if request.response_channel and event.content:
                        await request.response_channel.write_fragment(event.content)
                    if event.is_end:
                        break

            # 流式响应结束，记录总时间
            response.total_response_time = time.monotonic() - start_time

            # 发送结束标记到Channel
            if request.reasoning_channel and not thinking_end:
                await request.reasoning_channel.write_fragment("", end=True)
            if request.response_channel:
                await request.response_channel.write_fragment("", end=True)

        except Exception as e:
            await self._end_request_with_error(
                request, response,
                f"{type(e).__name__}: {str(e)}",
                FinishReason.error_request
            )

    async def _call_llm_api(self, request: LLMRequest, response: LLMResponse):
        """核心API调用入口，返回完整的LLM响应对象"""

        # 校验模型配置
        real_model_name = self.model_alias.get(request.model_name)
        model_config = self.api_config.get(real_model_name, {}).copy()
        if not real_model_name or not model_config:
            await self._end_request_with_error(
                request, response,
                f"Model not found: {request.model_name}",
                FinishReason.error_request
            )
            return
        request.model_name = real_model_name
        response.model_name = real_model_name

        api_type = model_config.get("api_type", self.default_config.get('api_type', 'openai'))
        if api_type not in self.format_strategies:
            await self._end_request_with_error(
                request, response,
                "Unsupported API type: %s" % api_type,
                FinishReason.error_request
            )
            self.logger.error("[%s] Unsupported API type: %s", request.request_id, api_type)
            return

        body_options = self.default_config.get('body_options')
        if body_options:
            model_options = body_options.copy()
            model_options.update(model_config.get('body_options', {}))
            model_config['body_options'] = model_options

        format_strategy = self.format_strategies[api_type](model_config)

        # 构建请求体
        try:
            request_body = format_strategy.make_request_body(request)
        except Exception as e:
            await self._end_request_with_error(
                request, response,
                f"Failed to build request body: {str(e)}",
                FinishReason.error_request
            )
            return

        # logger.debug("Request body: %s", request_body)

        # 计算排队时间与队列长度
        queue_start = time.monotonic()
        response.queue_length = self.resource_manager.get_queue_length(request.model_name)

        # 获取资源锁（控制模型并行度）
        async with self.resource_manager.acquire(request.model_name, request.request_id):
            response.queue_time = time.monotonic() - queue_start

            # 根据请求类型处理
            if request.stream:
                await self._handle_stream_request(
                    request, response, model_config, request_body, format_strategy)
            else:
                await self._handle_batch_request(
                    request, response, model_config, request_body, format_strategy)
            if response.finish_reason.startswith('error') or response.finish_reason in ('cancelled', 'unknown'):
                self.logger.warning(
                    "[%s] Request error (%s): %s",
                    request.request_id, response.finish_reason, response.error_text)
            else:
                self.logger.info(
                    "[%s] Request end (%s). %s/%s tokens, queue %.1fs/first %.1fs/total %.1fs",
                    request.request_id, response.finish_reason,
                    response.prompt_tokens, response.completion_tokens,
                    response.queue_time or 0, response.time_to_first_token or 0,
                    response.total_response_time or 0
                )

    async def _request_wrapper(self, request: LLMRequest, callback=None) -> LLMResponse:
        # 初始化响应对象
        response = LLMResponse(
            client_id=request.client_id,
            context_id=request.context_id or "",
            request_id=request.request_id,
            model_name=request.model_name,
            stream=request.stream
        )
        response.start_time = time.time()

        try:
            await self._call_llm_api(request, response)
        except asyncio.CancelledError:
            self.logger.info("[%s] Request cancelled.", request.request_id)
            await self._end_request_with_error(
                request, response,
                "cancelled",
                FinishReason.cancelled
            )
        except Exception as e:
            self.logger.exception("[%s] API call error", request.request_id)
            await self._end_request_with_error(
                request, response,
                f"API call error: {type(e).__name__} - {str(e)}",
                FinishReason.error
            )
        finally:
            try:
                del self.active_requests[request.request_id]
            except KeyError:
                pass
        request.response.set_result(response)
        if callback:
            await callback(request)
        return response

    def process_request(
            self, request: LLMRequest,
            callback: Optional[Callable[[LLMRequest], typing.Coroutine]] = None
    ) -> RequestTask:
        """处理请求"""
        task = asyncio.create_task(self._request_wrapper(request, callback))
        rt = RequestTask(request, task)
        self.active_requests[request.request_id] = rt
        return rt

    async def cancel_request(self, request_id: str):
        """取消指定请求"""
        rt = self.active_requests.get(request_id)
        if not rt:
            self.logger.warning("[%s] Request not found to cancel", request_id)
            return
        rt.request.is_cancelled = True
        rt.task.cancel()
        try:
            await rt.task
        except asyncio.CancelledError:
            self.logger.info("[%s] Cancelled", request_id)
        finally:
            try:
                del self.active_requests[request_id]
            except KeyError:
                pass

    async def cancel_all_requests(self, client_id: str, context_id: Optional[str] = None):
        """取消指定客户端/上下文的所有请求"""
        cancelled_count = 0
        total_count = 0

        # 遍历所有活跃请求，匹配client_id与context_id
        for request_id, (request, task) in list(self.active_requests.items()):
            if request.client_id == client_id and (not context_id or request.context_id == context_id):
                total_count += 1
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    cancelled_count += 1
                del self.active_requests[request_id]

        self.logger.info(
            "[Client: %s, Context: %s] Cancelled %d/%d requests",
            client_id, context_id, cancelled_count, total_count)

    async def cleanup(self):
        """清理资源"""
        for rt in self.active_requests.values():
            rt.request.is_cancelled = True
            rt.task.cancel()
        self.active_requests.clear()
        await self.http_client.aclose()
