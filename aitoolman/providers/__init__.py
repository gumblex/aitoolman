"""LLM提供商格式策略包：格式策略注册表"""
import logging

from .common import LLMFormatStrategy, ProviderStreamEvent
from .openai import OpenAICompatibleFormat, ResponsesFormat
from .anthropic import AnthropicFormat

# 包级 logger 默认 WARNING：外部作为库调用时，默认不输出 providers 内的 INFO/DEBUG 日志
logging.getLogger(__name__).setLevel(logging.WARNING)

# api_type → 格式策略类注册表
FORMAT_STRATEGY_REGISTRY = {
    "openai": OpenAICompatibleFormat,
    "responses": ResponsesFormat,
    "anthropic": AnthropicFormat,
}

__all__ = [
    'LLMFormatStrategy',
    'ProviderStreamEvent',
    'OpenAICompatibleFormat',
    'ResponsesFormat',
    'AnthropicFormat',
    'FORMAT_STRATEGY_REGISTRY',
]
