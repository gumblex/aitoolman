from __future__ import annotations
import abc
import logging

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

import jinja2

from . import channel as _channel
from .model import LLMModuleResult, Message, MediaContent, FinishReason

if TYPE_CHECKING:
    from .app import LLMApplication


@dataclass
class ModuleConfig:
    """模块配置"""
    name: str
    model: str
    templates: Dict[str, str] = field(default_factory=dict)
    tools: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stream: bool = False
    output_channel: Optional[_channel.TextFragmentOutput] = None
    reasoning_channel: Optional[_channel.TextFragmentOutput] = None
    post_processor: Optional[str] = None
    save_context: bool = False
    options: Dict[str, Any] = field(default_factory=dict)


class LLMModule(abc.ABC):
    """LLM模块基类"""

    def __init__(self, app: 'LLMApplication'):
        self.app: 'LLMApplication' = app

    @abc.abstractmethod
    async def __call__(self, *, _media: Optional[MediaContent] = None, **kwargs) -> LLMModuleResult:
        ...


class DefaultLLMModule(LLMModule):
    """默认LLM模块实现"""

    def __init__(self, app: 'LLMApplication', config: ModuleConfig):
        super().__init__(app)
        self.config = config
        self.context_messages: List[Message] = []
        self.templates: Dict[str, jinja2.Template] = {}
        for name, template in config.templates.items():
            self.templates[name] = self.app.jinja_env.from_string(template)
        self.logger = logging.getLogger("LLMModule[%s]" % self.config.name)

    def render_messages(self, kwargs, media_content: Optional[MediaContent] = None) -> List[Message]:
        result = []
        if self.context_messages:
            result.extend(self.context_messages)
        elif 'system' in self.templates:
            result.append(Message(
                self.render_template('system', **kwargs),
                role='system'
            ))
        result.append(Message(
            self.render_template('user', **kwargs),
            role='user', media_content=media_content
        ))
        return result

    def render_template(self, template_name: str, **kwargs) -> str:
        """渲染模板"""
        all_vars = {**self.app.vars, **kwargs}
        template = self.app.jinja_env.from_string(self.config.templates[template_name])
        return template.render(**all_vars)

    async def __call__(self, *, _media: Optional[MediaContent] = None, **kwargs) -> LLMModuleResult:
        """执行模块调用"""
        messages = self.render_messages(kwargs, media_content=_media)
        request = await self.app.client.request(
            model_name=self.config.model,
            messages=messages,
            tools=self.config.tools,
            options=self.config.options,
            stream=self.config.stream,
            context_id=self.app.context_id,
            response_channel=self.config.output_channel,
            reasoning_channel=self.config.reasoning_channel
        )
        response = await request.response
        result = LLMModuleResult.from_response(response)
        result.request_params = kwargs
        result.request_messages = messages

        if self.config.post_processor:
            try:
                result.data = self.app.processors[self.config.post_processor](result.response_text)
            except Exception:
                self.logger.exception("Post-process failed: %s", kwargs)
                result.status = FinishReason.error_format

        if result.status in (FinishReason.stop, FinishReason.tool_calls):
            self.context_messages = messages.copy()
            self.context_messages.append(Message(result.response_message))

        return result
