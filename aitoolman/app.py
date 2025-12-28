import logging
from typing import Any, Dict, Optional, Callable

import jinja2

from . import util
from . import postprocess
from . import client as _client
from . import channel as _channel
from .module import LLMModule, ModuleConfig, DefaultLLMModule


logger = logging.getLogger(__name__)


class LLMApplication:
    """LLM应用上下文"""

    def __init__(
        self,
        client: _client.LLMClient,
        config_dict: Optional[Dict[str, Any]] = None,
        channels: Optional[Dict[str, _channel.TextChannel]] = None,
        context_id: Optional[str] = None
    ):
        self.client: _client.LLMClient = client
        self.context_id: str = context_id or util.get_id()
        self.vars: Dict[str, Any] = {}
        self.channels: Dict[str, _channel.TextChannel] = {}
        self.processors: Dict[str, Callable] = postprocess.DEFAULT_PROCESSORS.copy()
        self.modules: Dict[str, LLMModule] = {}
        self.templates: Dict[str, jinja2.Template] = {}

        # 初始化Jinja2环境
        self.jinja_env: jinja2.Environment = jinja2.Environment(
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True
        )
        for name, template in config_dict.get('template', {}).items():
            self.templates[name] = self.jinja_env.from_string(template)

        # 加载配置
        self.config = config_dict
        self.config.setdefault('module', {})

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
        config = self.config.get('module_default', {}).copy()
        config.update(module_config)
        channel_name = config.get('output_channel')
        output_channel = self.channels[channel_name] if channel_name else None
        channel_name = config.get('reasoning_channel')
        reasoning_channel = self.channels[
            channel_name] if channel_name else None
        config = ModuleConfig(
            name=module_name,
            model=config.get('model', ''),
            templates=config.get('template', {}),
            tools=config.get('tools', {}),
            stream=config.get('stream', False),
            output_channel=output_channel,
            reasoning_channel=reasoning_channel,
            post_processor=config.get('post_processor'),
            save_context=config.get('save_context', False),
            options=config.get('options', {})
        )
        module = DefaultLLMModule(self, config)
        self.modules[module_name] = module
        logger.debug("Loaded module: %s", module_name)
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
        return self.templates[template_name].render(**all_vars)

    def add_channel(self, name: str, channel: _channel.TextChannel):
        """添加自定义通道"""
        self.channels[name] = channel
