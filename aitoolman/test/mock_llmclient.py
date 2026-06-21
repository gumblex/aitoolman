import re
import json
import inspect
from typing import Dict, List, Any, Callable, Optional, Union, NamedTuple

from .. import model as _model
from .. import client as _client


class AuditEvent(NamedTuple):
    context_id: str
    event_type: str
    kwargs: Dict[str, Any]


def run_response_function(
    request: _model.LLMProviderRequest, fn: Callable[[str], Union[str, List[_model.ToolCall]]]
) -> _model.LLMProviderResponse:
    try:
        response_content = fn(request.messages[-1].content)
    except _model.LLMLengthLimitError:
        return _model.LLMProviderResponse(
            client_id=request.client_id,
            context_id=request.context_id,
            request_id=request.request_id,
            model_name=request.model_name,
            stream=False,
            finish_reason=_model.FinishReason.length.value,
            response_text='',
            response_message=None
        )
    except _model.LLMContentFilterError as ex:
        return _model.LLMProviderResponse(
            client_id=request.client_id,
            context_id=request.context_id,
            request_id=request.request_id,
            model_name=request.model_name,
            stream=False,
            finish_reason=_model.FinishReason.content_filter.value,
            response_text=str(ex),
            response_message=_model.Message(role='assistant', content=str(ex))
        )
    except Exception as ex:
        return _model.LLMProviderResponse(
            client_id=request.client_id,
            context_id=request.context_id,
            request_id=request.request_id,
            model_name=request.model_name,
            stream=False,
            finish_reason=_model.FinishReason.error_request.value,
            error_text=str(ex)
        )
    if isinstance(response_content, str):
        return _model.LLMProviderResponse(
            client_id=request.client_id,
            context_id=request.context_id,
            request_id=request.request_id,
            model_name=request.model_name,
            stream=False,
            finish_reason=_model.FinishReason.stop.value,
            response_text=response_content,
            response_message=_model.Message(
                role='assistant', content=response_content,
                raw_value={"content": response_content}
            )
        )
    else:
        tool_calls_with_id = [tc._replace(id=_model.get_id()) for tc in response_content]
        return _model.LLMProviderResponse(
            client_id=request.client_id,
            context_id=request.context_id,
            request_id=request.request_id,
            model_name=request.model_name,
            stream=False,
            finish_reason=_model.FinishReason.tool_calls.value,
            response_tool_calls=tool_calls_with_id,
            response_message=_model.Message(
                role='assistant', content='',
                raw_value={"tool_calls": [tc._asdict() for tc in tool_calls_with_id]}
            )
        )


def make_simple_response(
    request: _model.LLMProviderRequest,
    response_content: Union[str, List[_model.ToolCall]]
) -> _model.LLMProviderResponse:
    if isinstance(response_content, str):
        return _model.LLMProviderResponse(
            client_id=request.client_id,
            context_id=request.context_id,
            request_id=request.request_id,
            model_name=request.model_name,
            stream=False,
            finish_reason=_model.FinishReason.stop.value,
            response_text=response_content,
            response_message=_model.Message(
                role='assistant', content=response_content,
                raw_value={"content": response_content}
            )
        )
    else:
        return _model.LLMProviderResponse(
            client_id=request.client_id,
            context_id=request.context_id,
            request_id=request.request_id,
            model_name=request.model_name,
            stream=False,
            finish_reason=_model.FinishReason.tool_calls.value,
            response_tool_calls=response_content,
            response_message=_model.Message(
                role='assistant', content='',
                raw_value={"tool_calls": [tc._asdict() for tc in response_content]}
            )
        )


def make_tool_call_response(request: _model.LLMProviderRequest):
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
    return _model.LLMProviderResponse(
        client_id=request.client_id,
        context_id=request.context_id,
        request_id=request.request_id,
        model_name=request.model_name,
        stream=False,
        finish_reason=_model.FinishReason.stop.value,
        response_text=response_text,
        response_message=_model.Message(
            role='assistant', content=response_text,
            raw_value={"content": response_text}
        )
    )


def default_response(request: _model.LLMProviderRequest) -> _model.LLMProviderResponse:
    return make_simple_response(request, request.messages[-1].content)


class LLMResponseGenerator:
    def __init__(self, response_fn = None):
        self.response_fn: Dict[str, Callable[[str], Union[str, List[_model.ToolCall]]]] = response_fn or {}

    def __call__(self, request: _model.LLMProviderRequest) -> _model.LLMProviderResponse:
        last_msg = request.messages[-1]
        if last_msg.role == _model.MessageRole.user.value:
            for match_str, fn in self.response_fn.items():
                if not match_str or re.search(match_str, last_msg.content):
                    return run_response_function(request, fn)
            return _model.LLMProviderResponse(
                client_id=request.client_id,
                context_id=request.context_id,
                request_id=request.request_id,
                model_name=request.model_name,
                stream=False,
                finish_reason=_model.FinishReason.error_request.value,
                error_text="no matching function",
                response_message=None
            )
        elif last_msg.role == _model.MessageRole.tool.value:
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
            return _model.LLMProviderResponse(
                client_id=request.client_id,
                context_id=request.context_id,
                request_id=request.request_id,
                model_name=request.model_name,
                stream=False,
                finish_reason=_model.FinishReason.stop.value,
                response_text=response_text,
                response_message=_model.Message(
                    role='assistant', content=response_text,
                    raw_value={"content": response_text}
                )
            )
        return _model.LLMProviderResponse(
            client_id=request.client_id,
            context_id=request.context_id,
            request_id=request.request_id,
            model_name=request.model_name,
            stream=False,
            finish_reason=_model.FinishReason.error_request.value,
            error_text="invalid message role",
            response_message=None
        )


class MockLLMClient(_client.LLMClient):
    """模拟LLM客户端，用于测试，不实际调用远程API"""
    def __init__(self, response_generator: Optional[Callable[[_model.LLMProviderRequest], _model.LLMProviderResponse]] = None):
        super().__init__()
        self.requests: List[_model.LLMProviderRequest] = []
        self.events: List[AuditEvent] = []
        self.response_generator: Callable[
            [_model.LLMProviderRequest], _model.LLMProviderResponse
        ] = response_generator or default_response

    def set_response_fn_on_text(self, fn_map: Dict[str, Callable[[str], Union[str, List[_model.ToolCall]]]]):
        self.response_generator = LLMResponseGenerator(fn_map)

    async def request(
        self,
        model_name: str,
        messages: List[_model.Message],
        tools: Dict[str, Dict[str, Any]] = None,
        options: Dict[str, Any] = None,
        stream: bool = False,
        context_id: str = None,
        output_channel: Any = None
    ) -> _model.LLMProviderRequest:
        request = self.make_request(
            model_name, messages, tools, options, stream,
            context_id, output_channel
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

    async def list_models(self, tag: Optional[str] = None):
        return []

    async def resolve_model(self, tags: Union[str, List[str]], messages = None) -> str:
        return 'test_model'

    async def update_config(self, new_config: Dict[str, Any]):
        pass

    async def update_api_config(self, model_name: str, model_config: Dict[str, Any]):
        pass

    async def change_api_status(self, model_name: str, enabled: bool):
        pass

    async def update_model_tag(self, tag: str, models: List[str]):
        pass
