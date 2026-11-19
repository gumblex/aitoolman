"""LLM提供商管理器：API调用、资源管理与请求重试（格式策略实现见 providers 包）"""
import typing
import math
import time
import asyncio
import logging
import random
from typing import Optional, Dict, List, Tuple, Any, Callable, Set, Union

from . import sse
from . import util
from .util import calculate_rank_weights, calc_message_length, estimate_text_token
from .channel import ChannelEvent
from .loop_detect import LoopDetector
from .model import LLMProviderRequest, LLMProviderResponse, FinishReason, Message, ModelInfo, LLMNoAvailableModelError
from .resmanager import ResourceManager
from ._version import VERSION
from .providers import (
    FORMAT_STRATEGY_REGISTRY,
    LLMFormatStrategy,
    ProviderStreamEvent,
    OpenAICompatibleFormat,
    AnthropicFormat,
    ResponsesFormat,
)

import httpx

logger = logging.getLogger(__name__)

__all__ = [
    'LLMFormatStrategy',
    'ProviderStreamEvent',
    'OpenAICompatibleFormat',
    'AnthropicFormat',
    'ResponsesFormat',
    'LLMProviderManager',
]


def _get_retry_after(response: httpx.Response) -> Optional[int]:
    """解析HTTP响应中的Retry-After头，返回重试时间（秒）"""
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None and retry_after.isdigit():
        return int(retry_after)
    return None


class RequestTask(typing.NamedTuple):
    request: LLMProviderRequest
    task: asyncio.Task


def _split_global_rank(text: str) -> Tuple[str, int]:
    """解析整串末尾的 ":N" 全局旋转后缀，返回（标签串, N）；无后缀时 N=1，后缀非法时抛 LLMNoAvailableModelError"""
    tag_text, sep, suffix = text.rpartition(':')
    if not sep:
        return text, 1
    try:
        return tag_text, int(suffix)
    except ValueError:
        raise LLMNoAvailableModelError(f"Invalid rank suffix in tags: {text}") from None


def _rotate_by_shift(items: List[Any], shift: int) -> List[Any]:
    """按 shift 循环左移列表；长度不大于 1 或 shift 为 0 时原样返回"""
    if len(items) <= 1:
        return items
    shift = shift % len(items)
    if shift == 0:
        return items
    return items[shift:] + items[:shift]


class LLMProviderManager:
    """LLM提供商管理器，负责API调用、资源管理与请求重试"""
    format_strategies = FORMAT_STRATEGY_REGISTRY
    default_bytes_per_token: float = 4.0
    default_rank_adjust_ratio: float = 0.25

    def __init__(self, config: Dict[str, Any]):
        self._initialized: bool = False
        self._apply_config(config)

        self.http_client = httpx.AsyncClient(
            timeout=self.timeout, http2=True,
            headers={'User-Agent': f'aitoolman/{VERSION}'}
        )

        # 资源管理器（控制模型并行度）
        self.resource_manager = ResourceManager(self._parse_model_capacities())

        # 活跃请求列表（用于取消操作）
        self.active_requests: Dict[str, RequestTask] = {}

    def _apply_config(self, config: Dict[str, Any]):
        """应用全量配置（为初始化和热更新提供统一逻辑）"""
        self.default_config = config['default']
        self.api_config = config['api']
        self.disabled_models: Set[str] = set()

        self.model_tag = config.get('model_tag', {})

        # 读取并校验 model_provider 配置
        self.model_provider: Dict[str, List[str]] = {}
        for provider_name, providers in config.get('model_provider', {}).items():
            if provider_name in self.api_config:
                raise ValueError(
                    f"Model provider '{provider_name}' conflicts with api config")
            if provider_name in self.model_tag:
                raise ValueError(
                    f"Model provider '{provider_name}' conflicts with tag config")
            if not isinstance(providers, list) or not providers:
                raise ValueError(
                    f"Model provider '{provider_name}' must be a non-empty list")
            for m in providers:
                if not isinstance(m, str):
                    raise ValueError(
                        f"Provider '{m}' in model_provider '{provider_name}' must be a string")
                if m not in self.api_config:
                    raise ValueError(
                        f"Provider '{m}' in model_provider '{provider_name}' not found in api config")
            self.model_provider[provider_name] = list(providers)

        # 模型名集合：api 模型名与逻辑模型名
        self.model_names: Set[str] = set(self.api_config) | set(self.model_provider)

        # 校验 model_tag 中的模型名
        for tag, models in self.model_tag.items():
            for m in self._flatten_model_list(models):
                if m not in self.model_names:
                    raise ValueError(f"Model '{m}' in tag '{tag}' not found in config")

        for model_name, model_config in self.api_config.items():
            if not model_config.get('enable', True):
                self.disabled_models.add(model_name)

        self.rank_adjust_ratio = self.default_config.get(
            'rank_adjust_ratio', LLMProviderManager.default_rank_adjust_ratio)

        # 预计算标签权重
        self.model_tag_weights: Dict[str, Dict[str, float]] = {}
        self._calculate_all_tag_weights()

        self.timeout: float = self.default_config.get('timeout', 10.0)
        self.timeout_batch: float = self.default_config.get('timeout_batch', 300.0)
        self.max_retries: int = self.default_config.get('max_retries', 0)
        self.default_parallel: int = self.default_config.get('parallel', 10)
        self.retry_duration = self.default_config.get('retry_duration', 0.5)
        self.retry_factor = self.default_config.get('retry_factor', 1.5)

    def _parse_model_capacities(self) -> Dict[str, int]:
        """从配置中解析各模型的并行处理能力"""
        capacities = {}
        for model_name, model_config in self.api_config.items():
            capacities[model_name] = model_config.get("parallel", self.default_parallel)
        return capacities

    def _calculate_all_tag_weights(self):
        """预计算所有模型-标签的权重"""
        self.model_tag_weights = {
            tag: calculate_rank_weights(models, self.rank_adjust_ratio)
            for tag, models in self.model_tag.items()
        }

    @staticmethod
    def _flatten_model_list(models: List[Union[str, List[str]]]) -> List[str]:
        """将可能包含嵌套列表的模型列表展平为模型名列表"""
        result = []
        for item in models:
            if isinstance(item, list):
                result.extend(item)
            else:
                result.append(item)
        return result

    def get_queue_length(self, model_name: str) -> int:
        """获取模型当前并发负载量（活跃请求数 + 等待队列长度）"""
        return self.resource_manager.get_queue_length(model_name)

    def _within_token_limit(self, model_name: str, msg_length) -> bool:
        """判断消息长度是否在模型输入 Token 限制内"""
        model_config = self.api_config.get(model_name, {})
        max_input_tokens = model_config.get(
            'max_input_tokens', self.default_config.get('max_input_tokens'))
        if max_input_tokens is None:
            return True
        bytes_per_token = model_config.get(
            'bytes_per_token',
            self.default_config.get('bytes_per_token', self.default_bytes_per_token)
        )
        return estimate_text_token(msg_length, bytes_per_token) <= max_input_tokens

    def _resolve_tag_weights(self, input_tags: List[str]) -> Dict[str, Dict[str, float]]:
        """解析标签，返回匹配到的标签/模型权重表"""
        tag_matched: Dict[str, Dict[str, float]] = {}
        for tag_input in input_tags:
            if tag_input in self.model_tag_weights:
                tag_matched[tag_input] = self.model_tag_weights[tag_input]
            elif tag_input in self.model_names:
                tag_matched[tag_input] = {tag_input: 1.0}
        return tag_matched

    def _model_queue_length(self, model_name: str) -> int:
        """模型层队列长度：逻辑模型取未禁用提供商队列长度的最小值"""
        if model_name in self.model_provider:
            queues = [self.get_queue_length(p)
                      for p in self.model_provider[model_name]
                      if p not in self.disabled_models]
            return min(queues) if queues else 0
        return self.get_queue_length(model_name)

    def _resolve_model_providers(
            self, model_name: str, messages: Optional[List[Message]] = None,
            context_id: str = ''
    ) -> List[str]:
        """将模型名或逻辑模型名解析为排序后的提供商列表

        Args:
            model_name: 模型名或逻辑模型名
            messages: 可选消息列表，用于 Token 数估算过滤
            context_id: 可选上下文 ID，用于稳定随机排序的种子

        Returns:
            提供商名列表

        Raises:
            LLMNoAvailableModelError: 无可用提供商
        """
        if model_name in self.model_provider:
            providers = [m for m in self.model_provider[model_name]
                         if m not in self.disabled_models]
            if messages is not None:
                msg_length = calc_message_length(messages)
                providers = [m for m in providers
                             if self._within_token_limit(m, msg_length)]
            if not providers:
                raise LLMNoAvailableModelError(
                    f"No available provider for model: {model_name}")
            rng = random.Random(context_id or None)
            return sorted(providers, key=lambda m: (
                self.get_queue_length(m), rng.random()))
        if model_name in self.api_config and model_name not in self.disabled_models:
            if messages is not None:
                msg_length = calc_message_length(messages)
                if not self._within_token_limit(model_name, msg_length):
                    raise LLMNoAvailableModelError(
                        f"No available provider for model: {model_name}")
            return [model_name]
        raise LLMNoAvailableModelError(
            f"No available provider for model: {model_name}")

    def resolve_model(
            self, input_tags: str, messages: Optional[List[Message]] = None,
            context_id: str = '', model_rank: int = 0, rank_by_model: bool = False
    ) -> List[Tuple[str, float]]:
        """解析出最终使用的提供商名及其评分

        Args:
            input_tags: 标签/模型名/逻辑模型名，支持以英文逗号分隔多个标签，末尾可带 :N 全局旋转后缀
            messages: 可选消息列表，用于 Token 数估算过滤
            context_id: 可选上下文 ID，用于稳定随机排序的种子
            model_rank: 候选偏移量，旋转对象由 rank_by_model 决定
            rank_by_model: 为 True 时 model_rank 在模型层旋转后再按提供商展开，否则作用于展开后的提供商列表

        Returns:
            提供商名与模型层评分的列表

        Raises:
            LLMNoAvailableModelError: 无可用模型
        """
        tag_text, global_rank = _split_global_rank(input_tags)
        input_tags = tag_text.split(',')
        if not input_tags:
            raise LLMNoAvailableModelError("No tags provided.")
        for tag in input_tags:
            if tag not in self.model_tag_weights and tag not in self.model_names:
                raise LLMNoAvailableModelError(
                    f"No available model for tags: {input_tags}")

        # 1. 精确匹配优先
        models: Optional[List[Tuple[str, float]]] = None
        for tag in input_tags:
            if tag in self.api_config and tag not in self.disabled_models:
                models = [(tag, 1.0)]
                break
            if tag in self.model_provider and any(
                    p not in self.disabled_models for p in self.model_provider[tag]):
                models = [(tag, 1.0)]
                break

        if models is None:
            # 2. 标签匹配与权重计算
            tag_matched = self._resolve_tag_weights(input_tags)
            if not tag_matched:
                raise LLMNoAvailableModelError(
                    f"No available model for tags: {input_tags}")
            all_model_sets = [set(weights.keys()) for weights in tag_matched.values()]
            common_models = all_model_sets[0]
            for s in all_model_sets[1:]:
                common_models = common_models & s

            # 过滤禁用模型（api 模型直接禁用，或逻辑模型所有提供商均禁用）
            common_models = common_models - self.disabled_models
            for model_name in list(common_models):
                if model_name in self.model_provider and all(
                        p in self.disabled_models for p in self.model_provider[model_name]):
                    common_models.discard(model_name)
            if not common_models:
                raise LLMNoAvailableModelError(
                    f"No available model for tags: {input_tags}")

            # 3. Token 限制过滤（api 模型按自身限制，逻辑模型留到展开阶段）
            if messages is not None:
                msg_length = calc_message_length(messages)
                for model_name in list(common_models):
                    if model_name in self.model_provider:
                        continue
                    if not self._within_token_limit(model_name, msg_length):
                        common_models.discard(model_name)
                if not common_models:
                    raise LLMNoAvailableModelError(
                        f"No available model within token limit for tags: {input_tags}")

            # 4. 权重求和排序
            model_total_weights: Dict[str, float] = {}
            for model_name in common_models:
                model_total_weights[model_name] = round(sum(
                    math.log(tag_weights.get(model_name, 1e-8))
                    for tag_weights in tag_matched.values()
                ), 6)

            rng = random.Random(context_id or None)
            best_models = sorted(common_models, key=lambda m: (
                model_total_weights[m],
                -self._model_queue_length(m),
                rng.random()
            ), reverse=True)
            models = [(m, model_total_weights[m]) for m in best_models]

        # 5. :N 恒定按模型层旋转（不受 rank_by_model 影响）
        models = _rotate_by_shift(models, global_rank - 1)

        # 6. model_rank 按 rank_by_model 选择旋转对象并展开提供商
        if rank_by_model:
            rotated = _rotate_by_shift(models, model_rank)
            result: List[Tuple[str, float]] = []
            for model_name, score in rotated:
                for provider in self._resolve_model_providers(
                        model_name, messages=messages, context_id=context_id):
                    result.append((provider, score))
            return result

        expanded: List[Tuple[str, float]] = []
        for model_name, score in models:
            for provider in self._resolve_model_providers(
                    model_name, messages=messages, context_id=context_id):
                expanded.append((provider, score))
        return _rotate_by_shift(expanded, model_rank)

    def list_models(self, tags: Optional[str] = None) -> List[ModelInfo]:
        """列出匹配的可用模型

        Args:
            tags: 可选标签/模型名/逻辑模型名，无 tag 返回所有可用 api 模型

        Returns:
            ModelInfo 列表，按权重降序排列
        """
        if not tags:
            models = []
            for name in self.api_config:
                if name in self.disabled_models:
                    continue
                models.append(self._make_model_info(name))
            return models

        candidates = self.resolve_model(tags)
        return [self._make_model_info(name) for name, _ in candidates]

    def list_tags(self) -> Dict[str, List[ModelInfo]]:
        """列出 [model_tag] 与 [model_provider] 的逻辑模型及其模型信息

        Returns:
            名称 -> 模型信息列表。model_tag 展开嵌套列表，逻辑模型替换为其提供商；
            model_provider 条目以提供商列表呈现
        """
        result: Dict[str, List[ModelInfo]] = {}
        for tag, models in self.model_tag.items():
            names: List[str] = []
            for m in self._flatten_model_list(models):
                if m in self.model_provider:
                    names.extend(p for p in self.model_provider[m]
                                 if p not in self.disabled_models)
                elif m in self.api_config and m not in self.disabled_models:
                    names.append(m)
            result[tag] = [self._make_model_info(m) for m in names]
        for provider_name, providers in self.model_provider.items():
            result[provider_name] = [
                self._make_model_info(m) for m in providers
                if m not in self.disabled_models]
        return result

    def _make_model_info(self, model_name: str) -> ModelInfo:
        """构造ModelInfo对象"""
        config = self.api_config[model_name]
        api_type = config.get("type", self.default_config.get('api_type', 'openai'))
        body_options = config.get('body_options', self.default_config.get('body_options') or {})

        tags = {}
        for tag_name, tag_weights in self.model_tag_weights.items():
            if model_name in tag_weights:
                tags[tag_name] = tag_weights[model_name]
                continue
            for provider_name, providers in self.model_provider.items():
                if model_name in providers and provider_name in tag_weights:
                    tags[tag_name] = tag_weights[provider_name]
                    break

        return ModelInfo(
            name=model_name,
            url=config['url'],
            model=config['model'],
            parallel=config.get('parallel', self.default_parallel),
            timeout=config.get("timeout", self.timeout),
            timeout_batch=config.get("timeout_batch", self.timeout_batch),
            api_type=api_type,
            body_options=body_options,
            max_input_tokens=config.get(
                "max_input_tokens", self.default_config.get('max_input_tokens')),
            bytes_per_token=config.get(
                'bytes_per_token',
                self.default_config.get('bytes_per_token', self.default_bytes_per_token)
            ),
            tags=tags,
        )

    async def initialize(self):
        """初始化ZeroMQ sockets和HTTP客户端"""
        if not self._initialized:
            await self.http_client.__aenter__()
            self._initialized = True

    def _merge_headers(self, model_headers: Dict[str, str]) -> Dict[str, str]:
        """合并默认请求头与模型请求头（模型请求头优先）"""
        return {**self.default_config.get('headers', {}), **model_headers}

    async def _end_request_with_error(
        self, request: LLMProviderRequest, response: LLMProviderResponse, error_text: str, finish_reason: FinishReason
    ):
        response.error_text = error_text
        response.finish_reason = finish_reason.value
        await request.output_channel.write(ChannelEvent('reasoning', None))
        await request.output_channel.write(ChannelEvent('response', None))

    async def _async_post_with_retry(self, request: LLMProviderRequest, url: str, **kwargs) -> Dict:
        """带重试的HTTP POST请求（处理限流、超时等异常）"""
        retries = 0
        while True:
            retry_after = None
            status_code = None

            try:
                response = await self.http_client.post(url, **kwargs)
                status_code = response.status_code
                response.raise_for_status()  # 抛出HTTP错误（如4xx/5xx）
                return util.decode_json(response.content)

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

            logger.warning(
                "[%s] Retry (%d/%d) after %.1fs, status code: %s",
                request.request_id, retries + 1, self.max_retries, retry_after, status_code
            )
            await asyncio.sleep(retry_after)
            retries += 1

    async def _handle_batch_request(
            self, request: LLMProviderRequest, response: LLMProviderResponse,
            model_config: Dict, request_body: Dict,
            format_strategy: LLMFormatStrategy
    ):
        """处理非流式请求（完整响应）"""
        await self.initialize()
        start_time = time.monotonic()
        logger.debug("Batch POST: %s, %s", model_config['url'], request_body)
        try:
            # 发送带重试的HTTP请求
            response_data = await self._async_post_with_retry(
                request=request,
                url=model_config["url"],
                json=request_body,
                headers=self._merge_headers(model_config.get("headers", {})),
                timeout=model_config.get("timeout_batch", self.timeout_batch)
            )
        except Exception as e:
            logger.warning("_handle_batch_request error.", exc_info=True)
            await self._end_request_with_error(
                request, response, f"{type(e).__qualname__}: {str(e)}", FinishReason.error_request
            )
            return

        # 时间统计
        total_duration = time.monotonic() - start_time
        response.total_response_time = total_duration
        response.time_to_first_token = total_duration  # 非流式请求无部分响应，首次响应时间等于总时间

        try:
            format_strategy.parse_batch_response(response, response_data)
        except Exception as e:
            logger.warning("parse_batch_response error", exc_info=True)
            await self._end_request_with_error(
                request, response,
                f"Failed to parse batch response: {str(e)}",
                FinishReason.error_format
            )
            return

        if not response.finish_reason:
            response.finish_reason = FinishReason.error_app.value
            response.error_text = 'No finish_reason set after parse_batch_response'

        # 发送完整响应到Channel
        await request.output_channel.write(ChannelEvent(
            'reasoning', response.response_reasoning))
        await request.output_channel.write(ChannelEvent('reasoning', None))
        await request.output_channel.write(ChannelEvent(
            'response', response.response_text))
        await request.output_channel.write(ChannelEvent('response', None))

    async def _handle_stream_request(
            self, request: LLMProviderRequest, response: LLMProviderResponse,
            model_config: Dict, request_body: Dict,
            format_strategy: LLMFormatStrategy
    ):
        """处理流式请求（分块响应）"""
        await self.initialize()
        start_time = time.monotonic()
        first_token_received = False
        thinking_end = False

        # 流式输出死循环检测：loop_detect_length 两级继承（模型配置优先），<=0 关闭
        loop_detect_length = model_config.get(
            'loop_detect_length', self.default_config.get('loop_detect_length', 0))
        if loop_detect_length <= 0:
            loop_detector = None
        elif loop_detect_length <= 10:
            logger.warning("[%s] loop_detect_length <= 10, disabled loop detector", response.model_name)
            loop_detector = None
        else:
            loop_detector = LoopDetector(loop_length=loop_detect_length)

        try:
            # 发送流式HTTP请求
            logger.debug("[%s] Stream POST: %s, %s",
                         response.model_name, model_config['url'], request_body)
            async with self.http_client.stream(
                method="POST",
                url=model_config["url"],
                json=request_body,
                headers=self._merge_headers(model_config.get("headers", {})),
                timeout=model_config.get("timeout", self.timeout)
            ) as http_response:
                http_response.raise_for_status()

                # 逐行解析流式响应（SSE格式：data: ...）
                timeout = model_config.get("timeout", self.timeout)

                # 保存异步迭代器，逐事件等待，记录最近一次收到有效内容（data 非空）的时间
                ait = sse.aiter_sse(http_response)
                last_content_time = time.monotonic()
                while True:
                    remaining = timeout - (time.monotonic() - last_content_time)
                    try:
                        sse_event = await asyncio.wait_for(anext(ait), timeout=max(remaining, 0))
                    except StopAsyncIteration:
                        break  # 流正常结束

                    if sse_event.data:
                        # 有效内容：刷新最近有效内容时间
                        last_content_time = time.monotonic()
                    elif time.monotonic() - last_content_time >= timeout:
                        # 收到事件但距上次有效内容累计已达阈值（心跳保活无法阻止超时）
                        raise TimeoutError(f"No valid content received within {timeout}s")
                    if request.is_cancelled:
                        response.finish_reason = FinishReason.cancelled.value
                        response.error_text = 'cancelled'
                        break
                    event = format_strategy.parse_stream_event(response, sse_event)
                    # logger.debug("Event: %s", event)

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
                        for message in event.response_message:
                            response.response_message.append(format_strategy.parse_response_message(
                                response, message))

                    # 发送分块内容到Channel
                    if event.reasoning:
                        await request.output_channel.write(ChannelEvent(
                            'reasoning', event.reasoning
                        ))
                    if not thinking_end and event.content:
                        await request.output_channel.write(
                            ChannelEvent('reasoning', None))
                        thinking_end = True
                    elif thinking_end and event.reasoning:
                        # reasoning after content
                        thinking_end = False
                    if request.output_channel and event.content:
                        await request.output_channel.write(
                            ChannelEvent('response', event.content))

                    # 流式输出死循环检测：增量内容喂入检测器，连续热窗口报警则中断
                    if loop_detector is not None and (event.content or event.reasoning):
                        if event.content:
                            loop_detector.update(event.content)
                        if event.reasoning:
                            loop_detector.update(event.reasoning)
                        if loop_detector.check():
                            score = loop_detector.score()
                            response.finish_reason = FinishReason.error_format.value
                            response.error_text = 'Loop detected in stream output, score: %.3f' % score
                            logger.warning(
                                "[%s] Loop detected in stream output (model: %s, score: %.3f)",
                                request.request_id, request.model_name, score)
                            break
                    if event.is_end:
                        break

            if not response.finish_reason:
                response.finish_reason = FinishReason.error_request.value
                response.error_text = 'Unexpected end of stream'

            # 流式响应结束，记录总时间
            response.total_response_time = time.monotonic() - start_time

            # 发送结束标记到Channel
            if not thinking_end:
                await request.output_channel.write(ChannelEvent('reasoning', None))
            await request.output_channel.write(ChannelEvent('response', None))

        except Exception as e:
            logger.warning("_handle_stream_request error.", exc_info=True)
            await self._end_request_with_error(
                request, response,
                f"{type(e).__qualname__}: {str(e)}",
                FinishReason.error_request
            )

    async def _call_llm_api(self, request: LLMProviderRequest, response: LLMProviderResponse):
        """核心API调用入口，返回完整的LLM响应对象"""

        # 解析真实提供商
        try:
            providers = self._resolve_model_providers(
                request.model_name, messages=request.messages,
                context_id=request.context_id)
        except LLMNoAvailableModelError as e:
            await self._end_request_with_error(
                request, response, str(e), FinishReason.error_request
            )
            return
        real_model_name = providers[0]
        model_config = self.api_config.get(real_model_name, {}).copy()
        if not model_config:
            await self._end_request_with_error(
                request, response,
                f"Model not found: {real_model_name}",
                FinishReason.error_request
            )
            return
        response.model_name = real_model_name

        api_type = model_config.get("type", self.default_config.get('api_type', 'openai'))
        if api_type not in self.format_strategies:
            await self._end_request_with_error(
                request, response,
                "Unsupported API type: %s" % api_type,
                FinishReason.error_request
            )
            logger.error("[%s] Unsupported API type: %s", request.request_id, api_type)
            return

        if model_config.get('body_options') is None:
            model_config['body_options'] = self.default_config.get('body_options') or {}

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
        response.queue_length = self.resource_manager.get_queue_length(real_model_name)

        # 获取资源锁（控制模型并行度）
        async with self.resource_manager.acquire(real_model_name, request.request_id):
            response.queue_time = time.monotonic() - queue_start

            # 根据请求类型处理
            if request.stream:
                await self._handle_stream_request(
                    request, response, model_config, request_body, format_strategy)
            else:
                await self._handle_batch_request(
                    request, response, model_config, request_body, format_strategy)
            if response.finish_reason is None:
                logger.error(
                    "[%s] Request finish_reason is None: %s",
                    request.request_id, response.finish_reason, response.error_text)
            elif response.finish_reason.startswith('error') or response.finish_reason in ('cancelled', 'unknown'):
                logger.warning(
                    "[%s] Request error (%s): %s",
                    request.request_id, response.finish_reason, response.error_text)
            else:
                logger.info(
                    "[%s] Request end (%s). %s/%s tokens, queue %.1fs/first %.1fs/total %.1fs",
                    request.request_id, response.finish_reason,
                    response.prompt_tokens, response.completion_tokens,
                    response.queue_time or 0, response.time_to_first_token or 0,
                    response.total_response_time or 0
                )

    async def _request_wrapper(self, request: LLMProviderRequest, callback=None) -> LLMProviderResponse:
        # 初始化响应对象
        response = LLMProviderResponse(
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
            logger.info("[%s] Request cancelled.", request.request_id)
            await self._end_request_with_error(
                request, response,
                "cancelled",
                FinishReason.cancelled
            )
        except Exception as e:
            logger.exception("[%s] API call error", request.request_id)
            await self._end_request_with_error(
                request, response,
                f"API call error: {type(e).__qualname__} - {str(e)}",
                FinishReason.error_request
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
            self, request: LLMProviderRequest,
            callback: Optional[Callable[[LLMProviderRequest], typing.Coroutine]] = None
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
            logger.warning("[%s] Request not found to cancel", request_id)
            return
        rt.request.is_cancelled = True
        rt.task.cancel()
        try:
            await rt.task
        except asyncio.CancelledError:
            logger.info("[%s] Cancelled", request_id)
        finally:
            try:
                del self.active_requests[request_id]
            except KeyError:
                pass

    async def cancel_all_requests(self, client_id: str, context_id: Optional[str] = None) -> List[str]:
        """取消指定客户端/上下文的所有请求"""
        cancelled_ids = []
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
                cancelled_ids.append(request_id)

        logger.info(
            "[Client: %s, Context: %s] Cancelled %d/%d requests",
            client_id, context_id, cancelled_count, total_count)
        return cancelled_ids

    def update_config(self, new_config: Dict[str, Any]):
        """热更新全量配置，不中断现有请求"""
        self._apply_config(new_config)
        # 替换 resource_manager
        self.resource_manager = ResourceManager(self._parse_model_capacities())

    def update_api_config(self, model_name: str, model_config: Dict[str, Any]):
        """更新单个模型配置"""
        if model_name in self.model_provider:
            raise ValueError(
                f"Api model '{model_name}' conflicts with model provider")
        self.api_config[model_name] = model_config.copy()
        self.model_names = set(self.api_config) | set(self.model_provider)

        # 更新 disabled_models
        if not model_config.get('enable', True):
            self.disabled_models.add(model_name)
        else:
            self.disabled_models.discard(model_name)

        # 重新计算权重
        self._calculate_all_tag_weights()
        # 更新 resource_manager 的容量
        self.resource_manager = ResourceManager(self._parse_model_capacities())

    def change_api_status(self, model_name: str, enabled: bool):
        """启用/禁用模型"""
        if model_name not in self.api_config:
            raise ValueError(f"Model '{model_name}' not found")
        if enabled:
            self.disabled_models.discard(model_name)
        else:
            self.disabled_models.add(model_name)

    def update_model_tag(self, tag: str, models: List[Union[str, List[str]]]):
        """更新 model_tag"""
        if tag in self.model_provider:
            raise ValueError(
                f"Tag '{tag}' conflicts with model provider")
        for m in self._flatten_model_list(models):
            if m not in self.model_names:
                raise ValueError(f"Model '{m}' in tag '{tag}' not found in config")
        self.model_tag[tag] = models
        self._calculate_all_tag_weights()

    def update_model_provider(self, provider_name: str, providers: List[str]):
        """更新 model_provider"""
        if provider_name in self.api_config:
            raise ValueError(
                f"Model provider '{provider_name}' conflicts with api config")
        if provider_name in self.model_tag:
            raise ValueError(
                f"Model provider '{provider_name}' conflicts with tag config")
        if not providers:
            raise ValueError(
                f"Model provider '{provider_name}' must be a non-empty list")
        for m in providers:
            if not isinstance(m, str):
                raise ValueError(
                    f"Provider '{m}' in model_provider '{provider_name}' must be a string")
            if m not in self.api_config:
                raise ValueError(
                    f"Provider '{m}' in model_provider '{provider_name}' not found in api config")
        self.model_provider[provider_name] = list(providers)
        self.model_names = set(self.api_config) | set(self.model_provider)

    async def close(self):
        """清理资源"""
        for rt in self.active_requests.values():
            rt.request.is_cancelled = True
            rt.task.cancel()
        self.active_requests.clear()
        if self._initialized:
            await self.http_client.aclose()
            self._initialized = False
