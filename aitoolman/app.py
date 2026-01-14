import functools
import logging
from typing import Any, Dict, Optional, Callable

import jinja2

from . import util
from . import postprocess
from . import client as _client
from . import channel as _channel
from .module import LLMModule, ModuleConfig, DefaultLLMModule


logger = logging.getLogger(__name__)


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

    def list_templates(self):
        """列出所有可用模板（调试用）"""
        templates = list(self.global_templates.keys())
        for module_name, module_config in self.modules.items():
            for template_name in module_config.get('template', {}).keys():
                templates.append(f"module/{module_name}/{template_name}")
        return templates


class LLMApplication:
    """LLM应用上下文"""

    def __init__(
        self,
        client: _client.LLMClient,
        config_dict: Optional[Dict[str, Any]] = None,
        processors: Optional[Dict[str, Callable[[str], Any]]] = None,
        channels: Optional[Dict[str, _channel.TextChannel]] = None,
        context_id: Optional[str] = None
    ):
        self.client: _client.LLMClient = client
        self.context_id: str = context_id or util.get_id()
        self.vars: Dict[str, Any] = {}
        self.channels: Dict[str, _channel.TextChannel] = {}
        self.processors: Dict[str, Callable[[str], Any]] = postprocess.DEFAULT_PROCESSORS.copy()
        self.modules: Dict[str, LLMModule] = {}

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
        self.jinja_env: jinja2.Environment = jinja2.Environment(
            loader=ConfigTemplateLoader(config_dict),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True
        )

        if processors:
            self.processors.update(processors)

        if channels:
            self.channels.update(channels)
        if 'stdin' not in self.channels:
            self.channels['stdin'] = _channel.TextChannel()
        if 'stdout' not in self.channels:
            self.channels['stdout'] = _channel.TextChannel(read_fragments=True)
        if 'reasoning' not in self.channels:
            self.channels['reasoning'] = _channel.TextChannel(read_fragments=True)

    def init_all_modules(self):
        """从配置加载所有模块"""
        if 'module' not in self.config:
            return

        for module_name, module_config in self.config['module'].items():
            # 创建模块配置对象
            self.init_module_from_config(module_name, module_config)

    def init_module_from_config(self, module_name, module_config):
        """从配置初始化模块"""
        # 合并模块默认配置
        config = self.config.get('module_default', {}).copy()
        config.update(module_config)

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
        module_config_obj = ModuleConfig(
            name=module_name,
            model=config.get('model', ''),
            templates=config.get('template', {}),
            tools=resolved_tools,
            stream=config.get('stream', False),
            output_channel=output_channel,
            reasoning_channel=reasoning_channel,
            post_processor=config.get('post_processor'),
            save_context=config.get('save_context', False),
            options=config.get('options', {})
        )

        module = DefaultLLMModule(self, module_config_obj)
        self.modules[module_name] = module
        return module

    def __getattr__(self, name: str) -> LLMModule:
        """通过属性访问模块"""
        if name in self.modules:
            return self.modules[name]
        if name in self.config['module']:
            return self.init_module_from_config(name, self.config['module'][name])
        raise AttributeError(f"No LLM module named '{name}'")

    def add_processor(self, name: str, processor: Callable):
        """添加后处理器"""
        self.processors[name] = processor

    def get_processor(self, name: str) -> Optional[Callable]:
        """获取后处理器"""
        return self.processors.get(name)

    def render_template(self, template_name: str, **kwargs) -> str:
        """渲染命名模板"""
        all_vars = {**self.vars, **kwargs}
        return self.jinja_env.get_template(template_name).render(**all_vars)

    def add_channel(self, name: str, channel: _channel.TextChannel):
        """添加自定义通道"""
        self.channels[name] = channel

    async def audit_event(self, event_type: str, **kwargs):
        """触发用户自定义审计事件"""
        await self.client.audit_event(self.context_id, event_type, **kwargs)

    @classmethod
    def factory(
            cls,
            client: _client.LLMClient,
            config_dict: Optional[Dict[str, Any]] = None,
            processors: Optional[Dict[str, Callable[[str], Any]]] = None,
            channels: Optional[Dict[str, _channel.TextChannel]] = None,
    ) -> Callable[..., 'LLMApplication']:
        """创建应用工厂函数"""
        return functools.partial(
            cls,
            client=client,
            config_dict=config_dict,
            processors=processors,
            channels=channels
        )
