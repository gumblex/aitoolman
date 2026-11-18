"""LLM提供商格式策略包：格式策略注册表"""
from .common import LLMFormatStrategy, ProviderStreamEvent
from .openai import OpenAICompatibleFormat, ResponsesFormat
from .anthropic import AnthropicFormat

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
