import json
import inspect
from typing import Dict, List, Any, Callable, Optional, Union, NamedTuple

from aitoolman import LLMClient, LLMProviderRequest, LLMProviderResponse, Message


class AuditEvent(NamedTuple):
    context_id: str
    event_type: str
    kwargs: Dict[str, Any]


def default_llm_response(request: LLMProviderRequest) -> LLMProviderResponse:
    response_text = json.dumps({
        'result': request.messages[-1].content
    }, ensure_ascii=False)
    return LLMProviderResponse(
        client_id=request.client_id,
        context_id=request.context_id,
        request_id=request.request_id,
        model_name=request.model_name,
        stream=False,
        finish_reason='stop',
        response_text=response_text,
        response_message={"content": response_text}
    )


class MockLLMClient(LLMClient):
    """模拟LLM客户端，用于测试，不实际调用远程API"""
    def __init__(self, response_generator: Optional[Callable[[
        LLMProviderRequest
    ], LLMProviderResponse]] = None):
        super().__init__()
        self.requests: List[LLMProviderRequest] = []
        self.events: List[AuditEvent] = []
        self.response_generator: Callable[[LLMProviderRequest], LLMProviderResponse] = response_generator or default_llm_response

    async def request(
        self,
        model_name: str,
        messages: List[Message],
        tools: Dict[str, Dict[str, Any]] = None,
        options: Dict[str, Any] = None,
        stream: bool = False,
        context_id: str = None,
        output_channel: Any = None,
        reasoning_channel: Any = None
    ) -> LLMProviderRequest:
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
