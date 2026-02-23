import logging
import inspect
import functools
from typing import Any, Dict, Optional, Callable, List, Union, NamedTuple

import jinja2

from . import util
from . import model as _model
from . import postprocess
from . import client as _client
from . import channel as _channel


logger = logging.getLogger(__name__)


class ModuleConfig(NamedTuple):
    """模块配置"""
    name: str
    description: str
    model: str
    templates: Dict[str, str] = {}
    tools: Dict[str, Dict[str, Any]] = {}
    stream: bool = False
    output_channel: Optional[_channel.TextFragmentChannel] = None
    reasoning_channel: Optional[_channel.TextFragmentChannel] = None
    post_processor: Optional[str] = None
    options: Dict[str, Any] = {}


class ConfigTemplateLoader(jinja2.BaseLoader):
    """自定义Jinja2模板加载器，支持全局和模块模板交叉引用"""

    def __init__(self, config_dict: Dict[str, Any]):
        self.global_templates = config_dict.get('template', {})
        self.modules = config_dict.get('module', {})

    def get_source(self, environment, template):
        """
        获取模板源码
        template 可以是:
        - 'header' -> 全局模板
        - 'module/task_planner/user' -> 模块task_planner的user模板
        """
        if template in self.global_templates:
            source = self.global_templates[template]
            return source, template, lambda: True
        elif template.startswith('module/'):
            # 模块模板: module/{module_name}/{template_name}
            parts = template.split('/')
            module_name = parts[1]
            template_name = parts[2]
            module_config = self.modules.get(module_name, {})
            if template_name in module_config.get('template', {}):
                source = module_config['template'][template_name]
                return source, template, lambda: True
        raise jinja2.TemplateNotFound(template)

    def __contains__(self, item):
        if item in self.global_templates:
            return True
        elif item.startswith('module/'):
            parts = item.split('/')
            module_name = parts[1]
            template_name = parts[2]
            return template_name in self.modules.get(module_name, {}).get('template', {})
        return False

    def list_templates(self):
        """列出所有可用模板（调试用）"""
        templates = list(self.global_templates.keys())
        for module_name, module_config in self.modules.items():
            for template_name in module_config.get('template', {}).keys():
                templates.append(f"module/{module_name}/{template_name}")
        return templates


class _LLMModule(NamedTuple):
    app: 'LLMApplication'
    module_name: str

    async def __call__(
            self, *,
            _model_name: Optional[str] = None,
            _context_messages: Optional[List[_model.Message]] = None,
            _media_content: Optional[_model.MediaContent] = None,
            **kwargs
    ) -> _model.LLMModuleResult:
        return await self.app.call(_model.LLMModuleRequest(
            module_name=self.module_name,
            template_params=kwargs,
            model_name=_model_name,
            context_messages=_context_messages,
            media_content=_media_content
        ))


class LLMApplication:
    """LLM应用上下文"""

    def __init__(
        self,
        client: _client.LLMClient,
        config_dict: Optional[Dict[str, Any]] = None,
        processors: Optional[Dict[str, Callable[[str], Any]]] = None,
        channels: Optional[Dict[str, _channel.TextFragmentChannel]] = None,
        context_id: Optional[str] = None
    ):
        self.client: _client.LLMClient = client
        self.context_id: str = context_id or util.get_id()
        self.vars: Dict[str, Any] = {}
        self.channels: Dict[str, _channel.Channel] = {}
        self.processors: Dict[str, Callable[[str], Any]] = postprocess.DEFAULT_PROCESSORS.copy()

        # 加载全局工具定义
        self.global_tools: Dict[str, Any] = {}

        # 配置初始化
        self.config = (config_dict or {}).copy()
        self.config.setdefault('module', {})
        self.config.setdefault('template', {})
        self.config.setdefault('tools', {})

        # 加载全局工具
        self.global_tools = self.config.get('tools', {})

        # 初始化Jinja2环境，使用自定义loader
        self.template_loader = ConfigTemplateLoader(config_dict)
        self.jinja_env: jinja2.Environment = jinja2.Environment(
            loader=self.template_loader,
            autoescape=False
        )

        if processors:
            self.processors.update(processors)

        if channels:
            self.channels.update(channels)
        if 'stdin' not in self.channels:
            self.channels['stdin'] = _channel.Channel()
        if 'stdout' not in self.channels:
            self.channels['stdout'] = _channel.TextFragmentChannel()
        if 'reasoning' not in self.channels:
            self.channels['reasoning'] = _channel.TextFragmentChannel()

        self.module_default: ModuleConfig = self._parse_module_config(
            '', self.config.get('module_default', {}))
        self.module_configs: Dict[str, ModuleConfig] = {}
        for module_name in self.config['module'].keys():
            self.module_configs[module_name] = self.load_module_config(module_name)

    def _parse_module_config(self, module_name, config) -> ModuleConfig:
        # 处理工具配置（支持全局工具引用）
        tools_config = config.get('tools', {})
        resolved_tools = {}

        for tool_name, tool_config in tools_config.items():
            # 如果工具配置为空dict，表示引用全局工具
            if isinstance(tool_config, dict) and not tool_config:
                if tool_name in self.global_tools:
                    resolved_tools[tool_name] = self.global_tools[tool_name]
                else:
                    raise ValueError(f"Module '{module_name}' referenced undefined global tool '{tool_name}'.")
            else:
                # 使用模块自定义配置（覆盖全局配置）
                resolved_tools[tool_name] = tool_config

        # 解析通道配置
        channel_name = config.get('output_channel')
        output_channel = self.channels[channel_name] if channel_name else None
        channel_name = config.get('reasoning_channel')
        reasoning_channel = self.channels[channel_name] if channel_name else None

        # 创建模块配置对象
        return ModuleConfig(
            name=module_name,
            description=config.get('description', ''),
            model=config.get('model', ''),
            templates=config.get('template', {}),
            tools=resolved_tools,
            stream=config.get('stream', False),
            output_channel=output_channel,
            reasoning_channel=reasoning_channel,
            post_processor=config.get('post_processor'),
            options=config.get('options', {})
        )

    def load_module_config(self, module_name) -> ModuleConfig:
        # raise KeyError
        module_config = self.config['module'][module_name]
        config = self.config.get('module_default', {}).copy()
        config.update(module_config)
        return self._parse_module_config(module_name, config)

    def __getitem__(self, name: str) -> _LLMModule:
        """通过下标访问的方式调用模块"""
        if name in self.module_configs:
            return _LLMModule(self, name)
        raise KeyError(f"No LLM module named '{name}'")

    def add_processor(self, name: str, processor: Callable):
        """添加后处理器"""
        self.processors[name] = processor

    def get_processor(self, name: str) -> Optional[Callable]:
        """获取后处理器"""
        return self.processors.get(name)

    def has_template(self, template_name: str) -> bool:
        return template_name in self.template_loader

    def render_template(self, template_name: str, **kwargs) -> str:
        """渲染命名模板"""
        all_vars = {**self.vars, **kwargs}
        return self.jinja_env.get_template(template_name).render(**all_vars)

    def add_channel(self, name: str, channel: _channel.TextFragmentChannel):
        """添加自定义通道"""
        self.channels[name] = channel

    def add_module(self, module_config: ModuleConfig):
        """添加LLM模块"""
        self.module_configs[module_config.name] = module_config

    async def audit_event(self, event_type: str, **kwargs):
        """触发用户自定义审计事件"""
        await self.client.audit_event(self.context_id, event_type, **kwargs)

    def render_direct_request(
            self, module_request: Union[_model.LLMModuleRequest, _model.LLMDirectRequest]
    ) -> _model.LLMDirectRequest:
        if isinstance(module_request, _model.LLMDirectRequest):
            return module_request
        messages = []
        if module_request.context_messages:
            messages.extend(module_request.context_messages)
        elif self.has_template("module/%s/system" % module_request.module_name):
            messages.append(_model.Message.from_content(self.render_template(
                "module/%s/system" % module_request.module_name,
                **module_request.template_params
            ), role='system'))
        messages.append(_model.Message.from_content(self.render_template(
            "module/%s/user" % module_request.module_name,
            **module_request.template_params
        ), role='user', media_content=module_request.media_content))
        config = self.module_configs[module_request.module_name]
        return _model.LLMDirectRequest(
            model_name=(module_request.model_name or config.model),
            messages=messages,
            tools=(module_request.tools if module_request.tools is not None else config.tools),
            options=(module_request.options if module_request.options is not None else config.options),
            stream=(module_request.stream if module_request.stream is not None else config.stream),
            output_channel=(module_request.output_channel if module_request.output_channel is not None else config.output_channel),
            reasoning_channel=(module_request.reasoning_channel if module_request.reasoning_channel is not None else config.reasoning_channel)
        )

    async def _send_request(self, direct_request: _model.LLMDirectRequest) -> _model.LLMModuleResult:
        """直接发送LLM请求"""
        output_channel = (
            self.channels[direct_request.output_channel]
            if isinstance(direct_request.output_channel, str)
            else direct_request.output_channel
        )
        reasoning_channel = (
            self.channels[direct_request.reasoning_channel]
            if isinstance(direct_request.reasoning_channel, str)
            else direct_request.reasoning_channel
        )
        request = await self.client.request(
            model_name=direct_request.model_name,
            messages=direct_request.messages,
            tools=direct_request.tools,
            options=direct_request.options,
            stream=direct_request.stream,
            context_id=self.context_id,
            output_channel=output_channel,
            reasoning_channel=reasoning_channel
        )
        response = await request.response
        return _model.LLMModuleResult.from_response(direct_request, response)

    async def _call_module(self, module_request: _model.LLMModuleRequest) -> _model.LLMModuleResult:
        direct_req = self.render_direct_request(module_request)
        result = await self._send_request(direct_req)
        result.module_name = module_request.module_name
        result.request_params = module_request.template_params

        config = self.module_configs[module_request.module_name]
        if config.post_processor and result.status == _model.FinishReason.stop:
            post_processor = self.processors[config.post_processor]
            try:
                data = post_processor(result.response_text)
                if inspect.isawaitable(data):
                    result.data = await data
                else:
                    result.data = data
            except Exception:
                logger.exception("Post-process failed: %s", module_request)
                result.status = _model.FinishReason.error_format
        return result

    async def call(self, request: Union[_model.LLMModuleRequest, _model.LLMDirectRequest]) -> _model.LLMModuleResult:
        if isinstance(request, _model.LLMModuleRequest):
            return await self._call_module(request)
        else:
            return await self._send_request(request)

    @classmethod
    def factory(
            cls,
            client: _client.LLMClient,
            config_dict: Optional[Dict[str, Any]] = None,
            processors: Optional[Dict[str, Callable[[str], Any]]] = None,
            channels: Optional[Dict[str, _channel.TextFragmentChannel]] = None,
    ) -> Callable[..., 'LLMApplication']:
        """创建应用工厂函数"""
        return functools.partial(
            cls,
            client=client,
            config_dict=config_dict,
            processors=processors,
            channels=channels
        )

