import re
import time
import typing
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Set

logger = logging.getLogger(__name__)


class TextFragmentInput(typing.Protocol):
    async def read_message(self) -> Optional[str]:
        ...

    async def read_fragment(self) -> Optional[str]:
        ...


class TextFragmentOutput(typing.Protocol):
    async def write_message(self, message: Optional[str]):
        ...

    async def write_fragment(self, text: str, end: bool = False):
        ...


class Channel:
    """基础通道类，支持完整消息的异步读写"""

    def __init__(self):
        self._message_queue = asyncio.Queue()
        self.closed = False

    async def read_message(self) -> Any:
        """读取一条完整消息"""
        return await self._message_queue.get()

    async def write_message(self, message: Any):
        """写入一条完整消息"""
        if self.closed:
            raise IOError("Channel is closed")
        await self._message_queue.put(message)

    def close(self):
        """关闭通道"""
        self.closed = True


class TextChannel(Channel):
    """文本通道，支持消息片段读写"""

    def __init__(self, read_fragments=False):
        super().__init__()
        # 当前消息的片段缓冲区
        self._current_fragment_buffer = []
        # 片段读取相关的状态
        self._fragment_queue = asyncio.Queue()
        self._read_fragments = read_fragments

    async def read_message(self) -> Optional[str]:
        """读取完整消息"""
        if self._read_fragments:
            raise RuntimeError("Cannot read message while reading fragments")
        elif self.closed:
            raise IOError("Channel is closed")
        return await self._message_queue.get()

    async def write_message(self, message: Optional[str]):
        if self.closed:
            raise IOError("Channel is closed")

        await self._message_queue.put(message)
        if self._read_fragments:
            await self._fragment_queue.put(message)
            await self._fragment_queue.put(None)  # 结束标记

    async def read_fragment(self) -> Optional[str]:
        """读取消息片段，返回 None 表示结束"""
        if self.closed or not self._read_fragments:
            return None

        return await self._fragment_queue.get()

    async def write_fragment(self, text: str, end: bool = False):
        """写入消息片段"""
        if self.closed:
            raise IOError("Channel is closed")

        self._current_fragment_buffer.append(text)

        if self._read_fragments:
            await self._fragment_queue.put(text)

        if end:
            complete_message = ''.join(self._current_fragment_buffer)
            await self._message_queue.put(complete_message)
            if self._read_fragments:
                await self._fragment_queue.put(None)
            self._current_fragment_buffer.clear()


class BaseXmlTagFilter(ABC):
    def __init__(self, tags: Set[str]):
        self.tags = tags
        self.current_tag: Optional[str] = None  # 当前激活的指定标签
        self.current_content: list[str] = []  # 当前标签的内容缓冲区
        self.pending_text: str = ""  # 跨片段的不完整标签缓冲区

        # 匹配所有XML标签的正则（支持命名空间和特殊字符）
        self.tag_pattern = re.compile(r'<(/?)([a-zA-Z_][\w.:-]*)>')
        # 闭合标签模板（动态生成当前标签的闭合匹配）
        self.closing_tag_template = r'</%s>'

    @abstractmethod
    async def on_message_tag(self, tag: Optional[str], message: str, end: bool):
        """处理完整消息的标签内容回调"""
        pass

    @abstractmethod
    async def on_fragment_tag(self, tag: Optional[str], text: str, end: bool):
        """处理消息片段的标签内容回调"""
        pass

    async def _on_tag(self, tag: Optional[str], message: str, end: bool, is_fragment: bool):
        if is_fragment:
            await self.on_fragment_tag(tag, message, end)
        else:
            await self.on_message_tag(tag, message, end)

    async def write_message(self, message: Optional[str]) -> None:
        """处理完整XML消息"""
        self._reset_state()
        try:
            remaining = await self._parse_content(message, is_fragment=False, end=True)
            if remaining:
                await self._on_tag(None, remaining, end=True, is_fragment=False)
        finally:
            self._reset_state()

    async def write_fragment(self, text: str, end: bool = False) -> None:
        """处理XML消息片段"""
        if not text and not end:
            return

        full_text = self.pending_text + text
        remaining = await self._parse_content(full_text, is_fragment=True, end=end)
        self.pending_text = remaining

        if end:
            await self._finalize_fragment()

    def _reset_state(self) -> None:
        """重置所有解析状态"""
        self.current_tag = None
        self.current_content = []
        self.pending_text = ""

    async def _parse_content(self, text: str, is_fragment: bool, end: bool) -> str:
        """
        核心解析逻辑：递归处理文本内容
        返回值：未解析的剩余文本（用于跨片段处理）
        """
        pos = 0
        len_text = len(text)

        # 状态1：当前处于指定标签内部（仅搜索当前标签的闭合）
        if self.current_tag is not None:
            closing_tag = self.closing_tag_template % self.current_tag
            closing_pos = text.find(closing_tag, pos)

            if closing_pos != -1:
                # 找到闭合标签：处理内容并重置状态
                self.current_content.append(text[pos:closing_pos])
                await self._emit_content(is_fragment)

                # 继续解析闭合标签后的内容（递归）
                pos = closing_pos + len(closing_tag)
                return await self._parse_content(text[pos:], is_fragment, end)
            else:
                # 未找到闭合标签：保存所有内容
                self.current_content.append(text[pos:])
                return ""

        # 状态2：处于顶层（解析所有标签）
        while pos < len_text:
            match = self.tag_pattern.search(text, pos)
            if not match:
                return await self._handle_top_level_remaining(text[pos:], is_fragment, end)

            # 处理标签前的普通文本
            before_tag = text[pos:match.start()]
            if before_tag:
                await self._on_tag(None, before_tag, False, is_fragment)

            # 解析标签信息
            is_closing = match.group(1) == '/'
            tag_name = match.group(2)
            tag_text = match.group(0)

            if tag_name in self.tags:
                if not is_closing:
                    # 处理指定标签的打开：进入标签内部状态
                    self.current_tag = tag_name
                    self.current_content = []
                    pos = match.end()
                    return await self._parse_content(text[pos:], is_fragment, end)
                else:
                    # 孤立的闭合标签：作为普通文本处理
                    await self._on_tag(None, tag_text, False, is_fragment)
                    pos = match.end()
            else:
                # 非指定标签：作为普通文本处理
                await self._on_tag(None, tag_text, False, is_fragment)
                pos = match.end()

        return ""

    async def _handle_top_level_remaining(self, remaining: str, is_fragment: bool, end: bool) -> str:
        """处理顶层未解析的剩余文本（处理不完整标签）"""
        if not remaining:
            return ""

        # 查找最后一个<的位置（判断是否有不完整标签）
        last_less_than = remaining.rfind('<')
        if last_less_than == -1:
            # 无标签结构：全部作为普通文本
            await self._on_tag(None, remaining, end, is_fragment)
            return ""
        else:
            # 分割完整文本与不完整标签
            complete_part = remaining[:last_less_than]
            if complete_part:
                await self._on_tag(None, complete_part, False, is_fragment)
            return remaining[last_less_than:]  # 返回不完整部分

    async def _emit_content(self, is_fragment: bool) -> None:
        """发射当前标签的内容（非空时）"""
        content = ''.join(self.current_content).strip()
        if content:
            await self._on_tag(self.current_tag, content, True, is_fragment)
        self.current_tag = None
        self.current_content = []

    async def _finalize_fragment(self) -> None:
        """处理最后一个片段的未完成状态"""
        if self.current_tag is not None:
            # 处理未闭合的标签内容
            if self.pending_text:
                self.current_content.append(self.pending_text)
            await self._emit_content(True)
            self.pending_text = ""
        elif self.pending_text:
            # 处理未完成的普通文本
            await self.on_fragment_tag(None, self.pending_text, True)
            self.pending_text = ""


class XmlTagToChannelFilter(BaseXmlTagFilter):
    def __init__(self, default_channel: 'TextChannel', channel_map: Dict[str, 'TextChannel']):
        tags = set(channel_map.keys())
        super().__init__(tags)
        self.default_channel = default_channel
        self.channel_map = channel_map

    async def on_message_tag(self, tag: Optional[str], message: str, end: bool):
        """将完整消息分发到对应通道"""
        if tag and tag in self.channel_map:
            await self.channel_map[tag].write_message(message)
        else:
            await self.default_channel.write_message(message)

    async def on_fragment_tag(self, tag: Optional[str], text: str, end: bool):
        """将消息片段分发到对应通道"""
        if tag and tag in self.channel_map:
            await self.channel_map[tag].write_fragment(text, end)
        else:
            await self.default_channel.write_fragment(text, end)


class ChannelEvent(typing.NamedTuple):
    """通道事件结构"""
    channel: str  # 通道名称，如 'reasoning', 'response'
    message: Any  # 消息内容
    is_fragment: bool  # 是否为片段
    is_end: bool  # 是否为结束标记


async def collect_text_channels(
        channels: Dict[str, TextChannel],
        read_fragments: bool = True,
        timeout: Optional[float] = None
) -> typing.AsyncGenerator[ChannelEvent, None]:
    """
    通用通道收集器，同时监听多个TextChannel，生成通道事件。
    Args:
        channels: 通道字典，键为通道名称，值为TextChannel对象
        read_fragments: 是否以片段模式读取（True=片段，False=完整消息）
        timeout: 总体超时时间（秒），超过则抛出TimeoutError
    Yields:
        ChannelEvent: 通道事件，包含通道名、消息内容、是否为片段、是否为结束标记
    """
    # 初始化：为每个通道创建第一个读取任务（future）
    _priority_timeout = 1.0
    _priority_real_timeout = (
        min(timeout, _priority_timeout) if timeout else _priority_timeout)
    last_output_channel = None
    pending_futures: Dict[asyncio.Future, str] = {}  # future -> 通道名称
    for channel_name, channel in channels.items():
        # 根据读取模式选择对应的读取方法
        coro = channel.read_fragment() if read_fragments else channel.read_message()
        fut = asyncio.create_task(coro)
        pending_futures[fut] = channel_name

    try:
        while pending_futures:
            # 保存当前未完成的future映射（避免wait后pending_futures被覆盖）
            current_futures = pending_futures.copy()
            done_futures = {}
            pending_set = set()
            if last_output_channel in current_futures.values():
                wait_futures = [
                    fut for fut, channel_name in current_futures.items()
                    if channel_name == last_output_channel
                ]
                start_time = time.monotonic()

                _done, _pending = await asyncio.wait(
                    wait_futures,
                    timeout=_priority_real_timeout,
                    return_when=asyncio.FIRST_COMPLETED  # 有一个完成就返回
                )
                if timeout and time.monotonic() - start_time > timeout:
                    raise TimeoutError(
                        f"collect_text_channels timed out after {timeout} seconds")
                for fut in _done:
                    done_futures[fut] = current_futures[fut]
                for fut in current_futures.keys():
                    if fut in done_futures:
                        continue
                    if fut.done():
                        done_futures[fut] = current_futures[fut]
                    else:
                        pending_set.add(fut)
            else:
                # 等待任意future完成，或超时（返回已完成和未完成的future分组）
                _done, _pending = await asyncio.wait(
                    current_futures.keys(),
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED  # 有一个完成就返回
                )
                # 处理超时：无任何future完成时触发
                if not _done:
                    raise TimeoutError(
                        f"collect_text_channels timed out after {timeout} seconds")
                for fut in _done:
                    done_futures[fut] = current_futures[fut]
                pending_set = _pending

            # 更新pending_futures为未完成的任务（后续继续等待）
            pending_futures = {fut: current_futures[fut] for fut in pending_set}

            # 处理已完成的future
            for fut, channel_name in done_futures.items():
                channel = channels[channel_name]

                try:
                    result = fut.result()  # 获取读取结果（可能抛出异常）
                except Exception:
                    logger.exception(f"Failed to read from channel '{channel_name}'")
                    # 异常视为通道结束，生成结束事件
                    yield ChannelEvent(
                        channel=channel_name,
                        message=None,
                        is_fragment=read_fragments,
                        is_end=True
                    )
                    continue  # 跳过后续处理（该通道不再读取）

                # 判断通道是否结束（收到None即为结束）
                is_end = result is None

                # 生成通道事件（严格对应读取结果）
                yield ChannelEvent(
                    channel=channel_name,
                    message=result,  # 消息内容（None表示结束）
                    is_fragment=read_fragments,  # 是否为片段模式
                    is_end=is_end  # 是否是通道结束标记
                )
                last_output_channel = channel_name

                # 未结束的通道：继续添加下一次读取任务
                if not is_end:
                    next_coro = channel.read_fragment() if read_fragments else channel.read_message()
                    next_fut = asyncio.create_task(next_coro)
                    pending_futures[next_fut] = channel_name

    finally:
        # 清理资源：取消所有未完成的读取任务
        for fut in pending_futures:
            fut.cancel()
        # 等待所有任务取消完成（避免资源泄漏）
        await asyncio.gather(*pending_futures.keys(), return_exceptions=True)
