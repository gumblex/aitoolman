import re
import json
import inspect
from typing import Dict, List, Any, Callable, Optional, Union, NamedTuple, Tuple

import aitoolman
from aitoolman import MessageRole


class AuditEvent(NamedTuple):
    context_id: str
    event_type: str
    kwargs: Dict[str, Any]


def run_response_function(
    request: aitoolman.LLMProviderRequest, fn: Callable[[str], Union[str, List[aitoolman.ToolCall]]]
) -> aitoolman.LLMProviderResponse:
    try:
        response_content = fn(request.messages[-1].content)
    except aitoolman.LLMLengthLimitError:
        return aitoolman.LLMProviderResponse(
            client_id=request.client_id,
            context_id=request.context_id,
            request_id=request.request_id,
            model_name=request.model_name,
            stream=False,
            finish_reason=aitoolman.FinishReason.length.value,
            response_text='',
            response_message=None
        )
    except aitoolman.LLMContentFilterError as ex:
        return aitoolman.LLMProviderResponse(
            client_id=request.client_id,
            context_id=request.context_id,
            request_id=request.request_id,
            model_name=request.model_name,
            stream=False,
            finish_reason=aitoolman.FinishReason.content_filter.value,
            response_text=str(ex),
            response_message=aitoolman.Message(role='assistant', content=str(ex))
        )
    except Exception as ex:
        return aitoolman.LLMProviderResponse(
            client_id=request.client_id,
            context_id=request.context_id,
            request_id=request.request_id,
            model_name=request.model_name,
            stream=False,
            finish_reason=aitoolman.FinishReason.error_request.value,
            error_text=str(ex)
        )
    if isinstance(response_content, str):
        return aitoolman.LLMProviderResponse(
            client_id=request.client_id,
            context_id=request.context_id,
            request_id=request.request_id,
            model_name=request.model_name,
            stream=False,
            finish_reason=aitoolman.FinishReason.stop.value,
            response_text=response_content,
            response_message=aitoolman.Message(
                role='assistant', content=response_content,
                raw_value={"content": response_content}
            )
        )
    else:
        tool_calls_with_id = [tc._replace(id=aitoolman.get_id()) for tc in response_content]
        return aitoolman.LLMProviderResponse(
            client_id=request.client_id,
            context_id=request.context_id,
            request_id=request.request_id,
            model_name=request.model_name,
            stream=False,
            finish_reason=aitoolman.FinishReason.tool_calls.value,
            response_tool_calls=tool_calls_with_id,
            response_message=aitoolman.Message(
                role='assistant', content='',
                raw_value={"tool_calls": [tc._asdict() for tc in tool_calls_with_id]}
            )
        )


def make_simple_response(
    request: aitoolman.LLMProviderRequest,
    response_content: Union[str, List[aitoolman.ToolCall]]
) -> aitoolman.LLMProviderResponse:
    if isinstance(response_content, str):
        return aitoolman.LLMProviderResponse(
            client_id=request.client_id,
            context_id=request.context_id,
            request_id=request.request_id,
            model_name=request.model_name,
            stream=False,
            finish_reason=aitoolman.FinishReason.stop.value,
            response_text=response_content,
            response_message=aitoolman.Message(
                role='assistant', content=response_content,
                raw_value={"content": response_content}
            )
        )
    else:
        return aitoolman.LLMProviderResponse(
            client_id=request.client_id,
            context_id=request.context_id,
            request_id=request.request_id,
            model_name=request.model_name,
            stream=False,
            finish_reason=aitoolman.FinishReason.tool_calls.value,
            response_tool_calls=response_content,
            response_message=aitoolman.Message(
                role='assistant', content='',
                raw_value={"tool_calls": [tc._asdict() for tc in response_content]}
            )
        )


def make_tool_call_response(request: aitoolman.LLMProviderRequest):
    tool_results = []
    tool_call_map = {}
    for msg in request.messages:
        if msg.role == 'assistant' and msg.raw_value.get('tool_calls'):
            for tc in msg.raw_value['tool_calls']:
                tool_call_map[tc['id']] = tc
    for msg in request.messages:
        if msg.role == 'tool' and msg.tool_call_id in tool_call_map:
            tc = tool_call_map[msg.tool_call_id]
            tool_results.append({
                "id": tc['id'],
                "arguments": tc['arguments'],
                "result": msg.content
            })
    response_text = json.dumps(tool_results, ensure_ascii=False)
    return aitoolman.LLMProviderResponse(
        client_id=request.client_id,
        context_id=request.context_id,
        request_id=request.request_id,
        model_name=request.model_name,
        stream=False,
        finish_reason=aitoolman.FinishReason.stop.value,
        response_text=response_text,
        response_message=aitoolman.Message(
            role='assistant', content=response_text,
            raw_value={"content": response_text}
        )
    )


def default_response(request: aitoolman.LLMProviderRequest) -> aitoolman.LLMProviderResponse:
    return make_simple_response(request, request.messages[-1].content)


class LLMResponseGenerator:
    def __init__(self, response_fn = None):
        self.response_fn: Dict[str, Callable[[str], Union[str, List[aitoolman.ToolCall]]]] = response_fn or {}

    def __call__(self, request: aitoolman.LLMProviderRequest) -> aitoolman.LLMProviderResponse:
        last_msg = request.messages[-1]
        if last_msg.role == MessageRole.user.value:
            for match_str, fn in self.response_fn.items():
                if not match_str or re.search(match_str, last_msg.content):
                    return run_response_function(request, fn)
            return aitoolman.LLMProviderResponse(
                client_id=request.client_id,
                context_id=request.context_id,
                request_id=request.request_id,
                model_name=request.model_name,
                stream=False,
                finish_reason=aitoolman.FinishReason.error_request.value,
                error_text="no matching function",
                response_message=None
            )
        elif last_msg.role == MessageRole.tool.value:
            tool_results = []
            tool_call_map = {}
            for msg in request.messages:
                if msg.role == 'assistant' and msg.raw_value.get('tool_calls'):
                    for tc in msg.raw_value['tool_calls']:
                        tool_call_map[tc['id']] = tc
            for msg in request.messages:
                if msg.role == 'tool' and msg.tool_call_id in tool_call_map:
                    tc = tool_call_map[msg.tool_call_id]
                    tool_results.append({
                        "id": tc['id'],
                        "arguments": tc['arguments'],
                        "result": msg.content
                    })
            response_text = json.dumps(tool_results, ensure_ascii=False)
            return aitoolman.LLMProviderResponse(
                client_id=request.client_id,
                context_id=request.context_id,
                request_id=request.request_id,
                model_name=request.model_name,
                stream=False,
                finish_reason=aitoolman.FinishReason.stop.value,
                response_text=response_text,
                response_message=aitoolman.Message(
                    role='assistant', content=response_text,
                    raw_value={"content": response_text}
                )
            )
        return aitoolman.LLMProviderResponse(
            client_id=request.client_id,
            context_id=request.context_id,
            request_id=request.request_id,
            model_name=request.model_name,
            stream=False,
            finish_reason=aitoolman.FinishReason.error_request.value,
            error_text="invalid message role",
            response_message=None
        )


class MockLLMClient(aitoolman.LLMClient):
    """模拟LLM客户端，用于测试，不实际调用远程API"""
    def __init__(self, response_generator: Optional[Callable[[aitoolman.LLMProviderRequest], aitoolman.LLMProviderResponse]] = None):
        super().__init__()
        self.requests: List[aitoolman.LLMProviderRequest] = []
        self.events: List[AuditEvent] = []
        self.response_generator: Callable[
            [aitoolman.LLMProviderRequest], aitoolman.LLMProviderResponse
        ] = response_generator or default_response

    def set_response_fn_on_text(self, fn_map: Dict[str, Callable[[str], Union[str, List[aitoolman.ToolCall]]]]):
        self.response_generator = LLMResponseGenerator(fn_map)

    async def request(
        self,
        model_name: str,
        messages: List[aitoolman.Message],
        tools: Dict[str, Dict[str, Any]] = None,
        options: Dict[str, Any] = None,
        stream: bool = False,
        context_id: str = None,
        output_channel: Any = None,
        reasoning_channel: Any = None
    ) -> aitoolman.LLMProviderRequest:
        request = self.make_request(
            model_name, messages, tools, options, stream,
            context_id, output_channel, reasoning_channel
        )
        self.requests.append(request)
        response = self.response_generator(request)
        if inspect.isawaitable(response):
            response = await response
        request.response.set_result(response)
        return request

    async def cancel(self, request_id: str):
        pass

    async def audit_event(self, context_id: str, event_type: str, **kwargs):
        self.events.append(AuditEvent(context_id, event_type, kwargs))


class MockTextChannelCollector(aitoolman.ChannelCollector):
    def __init__(self, channels):
        super().__init__(channels)
        self.events_start: List[str] = []
        self.events_read: List[Tuple[str, Any]] = []
        self.events_end: List[str] = []
        self.events_eof: List[str] = []

    async def on_channel_start(self, channel_name: str):
        self.events_start.append(channel_name)

    async def on_channel_read(self, channel_name: str, message):
        self.events_read.append((channel_name, message))

    async def on_channel_end(self, channel_name: str):
        self.events_end.append(channel_name)

    async def on_channel_eof(self, channel_name: str):
        self.events_eof.append(channel_name)

