import re
import abc
import asyncio
import logging
from typing import Optional, Dict, Set, Generic, TypeVar

logger = logging.getLogger(__name__)


T = TypeVar("T")


_EOF = object()


class Channel(Generic[T]):

    def __init__(self):
        self._message_queue = asyncio.Queue()

    async def read(self) -> T:
        result = await self._message_queue.get()
        if result == _EOF:
            raise EOFError
        return result

    async def write(self, message: T):
        await self._message_queue.put(message)

    async def write_complete(self):
        await self._message_queue.put(_EOF)


class TextFragmentChannel(Channel[Optional[str]]):
    """
    文本片段通道，支持按文本片段读写

    None 为一整条消息的结束符
    """

    async def read_whole_message(self) -> str:
        """读取完整消息"""
        buffer = []
        while True:
            result = await self._message_queue.get()
            if result is None:
                break
            elif result == _EOF:
                raise EOFError
            buffer.append(result)
        return ''.join(buffer)


class BaseXmlTagFilter(abc.ABC):
    def __init__(self, tags: Set[str]):
        self.tags = tags
        self.current_tag: Optional[str] = None  # 当前激活的指定标签
        self.current_content: list[str] = []  # 当前标签的内容缓冲区
        self.pending_text: str = ""  # 跨片段的不完整标签缓冲区
        self.pending_closing_tag = ""  # 不完整闭合标签缓冲区

        # 匹配所有XML标签的正则（支持命名空间和特殊字符）
        self.tag_pattern = re.compile(r'<(/?)([a-zA-Z_][\w.:-]*)>')
        # 闭合标签模板（动态生成当前标签的闭合匹配）
        self.closing_tag_template = r'</%s>'

    @abc.abstractmethod
    async def on_tag(self, tag: Optional[str], text: str, end: bool):
        """处理消息片段的标签内容回调"""
        pass

    async def write(self, message: Optional[str]) -> None:
        if message is None:
            await self._finalize_fragment()
            return
        if not message:
            return

        full_text = self.pending_text + message
        remaining = await self._parse_content(full_text, end=False)
        self.pending_text = remaining

    def _reset_state(self) -> None:
        """重置所有解析状态"""
        self.current_tag = None
        self.current_content = []
        self.pending_text = ""
        self.pending_closing_tag = ""

    async def _parse_content(self, text: str, end: bool) -> str:
        """
        核心解析逻辑：递归处理文本内容
        返回值：未解析的剩余文本（用于跨片段处理）
        """
        pos = 0
        len_text = len(text)

        # 状态1：当前处于指定标签内部（仅搜索当前标签的闭合）
        if self.current_tag is not None:
            closing_tag = self.closing_tag_template % self.current_tag
            closing_len = len(closing_tag)

            # 拼接之前的不完整闭合标签与当前文本
            full_text = self.pending_closing_tag + text
            self.pending_closing_tag = ""

            closing_pos = full_text.find(closing_tag, pos)

            if closing_pos != -1:
                # 找到闭合标签：处理内容并重置状态
                self.current_content.append(full_text[pos:closing_pos])
                await self._emit_content()

                # 继续解析闭合标签后的内容（递归）
                remaining_text = full_text[closing_pos + closing_len:]
                return await self._parse_content(remaining_text, end)
            else:
                # 未找到完整闭合标签：检查是否有部分闭合标签在末尾
                found_partial = False
                # 从最长可能的前缀开始检查，避免误匹配
                for l in range(min(closing_len, len(full_text)), 0, -1):
                    suffix = full_text[-l:]
                    if closing_tag.startswith(suffix):
                        # 保存不完整的闭合标签部分
                        self.pending_closing_tag = suffix
                        # 剩余部分加入内容
                        if len(full_text) > l:
                            self.current_content.append(full_text[:-l])
                        found_partial = True
                        break
                if not found_partial:
                    # 没有部分匹配，全部加入内容
                    self.current_content.append(full_text)
                return ""

        # 状态2：处于顶层（解析所有标签）
        while pos < len_text:
            match = self.tag_pattern.search(text, pos)
            if not match:
                return await self._handle_top_level_remaining(text[pos:], end)

            # 处理标签前的普通文本
            before_tag = text[pos:match.start()]
            if before_tag:
                await self.on_tag(None, before_tag, False)

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
                    return await self._parse_content(text[pos:], end)
                else:
                    # 孤立的闭合标签：作为普通文本处理
                    await self.on_tag(None, tag_text, False)
                    pos = match.end()
            else:
                # 非指定标签：作为普通文本处理
                await self.on_tag(None, tag_text, False)
                pos = match.end()

        return ""

    async def _handle_top_level_remaining(self, remaining: str, end: bool) -> str:
        """处理顶层未解析的剩余文本（处理不完整标签）"""
        if not remaining:
            return ""

        # 查找最后一个<的位置（判断是否有不完整标签）
        last_less_than = remaining.rfind('<')
        if last_less_than == -1:
            # 无标签结构：全部作为普通文本
            await self.on_tag(None, remaining, end)
            return ""
        else:
            # 分割完整文本与不完整标签
            complete_part = remaining[:last_less_than]
            if complete_part:
                await self.on_tag(None, complete_part, False)
            return remaining[last_less_than:]  # 返回不完整部分

    async def _emit_content(self) -> None:
        """发射当前标签的内容（非空时）"""
        content = ''.join(self.current_content).strip()
        if content:
            await self.on_tag(self.current_tag, content, True)
        self.current_tag = None
        self.current_content = []

    async def _finalize_fragment(self) -> None:
        """处理最后一个片段的未完成状态"""
        if self.current_tag is not None:
            # 处理未闭合的标签内容
            if self.pending_text:
                self.current_content.append(self.pending_text)
            # 加入剩余的不完整闭合标签部分
            if self.pending_closing_tag:
                self.current_content.append(self.pending_closing_tag)
            await self._emit_content()
            self.pending_text = ""
            self.pending_closing_tag = ""  # 新增：清空缓冲区
        elif self.pending_text:
            # 处理未完成的普通文本
            await self.on_tag(None, self.pending_text, True)
            self.pending_text = ""
        else:
            await self.on_tag(None, "", True)


class XmlTagToChannelFilter(BaseXmlTagFilter):
    def __init__(self, default_channel: 'TextFragmentChannel', channel_map: Dict[str, 'TextFragmentChannel']):
        tags = set(channel_map.keys())
        super().__init__(tags)
        self.default_channel = default_channel
        self.channel_map = channel_map

    async def on_tag(self, tag: Optional[str], text: str, end: bool):
        """将消息片段分发到对应通道"""
        if tag and tag in self.channel_map:
            await self.channel_map[tag].write(text)
            if end:
                await self.channel_map[tag].write(None)
        else:
            await self.default_channel.write(text)
            if end:
                await self.default_channel.write(None)


class ChannelCollector(abc.ABC):
    def __init__(self, channels: Dict[str, Channel]):
        self.channels = channels
        self.running = True
        self.last_channel_timeout = 0.1

        self._pending_futures: Dict[asyncio.Future, str] = {}  # future -> 通道名称
        self._last_output_channel: Optional[str] = None

    @abc.abstractmethod
    async def on_channel_start(self, channel_name: str):
        """通道开始本次输出"""
        ...

    @abc.abstractmethod
    async def on_channel_read(self, channel_name: str, message):
        """通道输出内容"""
        ...

    @abc.abstractmethod
    async def on_channel_end(self, channel_name: str):
        """通道结束本次输出"""
        ...

    @abc.abstractmethod
    async def on_channel_eof(self, channel_name: str):
        """通道结束所有输出"""
        ...

    async def _channel_read(self, channel_name: str, message):
        if channel_name != self._last_output_channel:
            if self._last_output_channel is not None:
                await self.on_channel_end(self._last_output_channel)
            await self.on_channel_start(channel_name)
        if message is None:
            if self._last_output_channel is not None:
                await self.on_channel_end(channel_name)
                self._last_output_channel = None
            return
        await self.on_channel_read(channel_name, message)
        self._last_output_channel = channel_name

    def close(self):
        self.running = False

    async def start_listening(self):
        for channel_name, channel in self.channels.items():
            # 根据读取模式选择对应的读取方法
            fut = asyncio.create_task(channel.read())
            self._pending_futures[fut] = channel_name

        try:
            while self._pending_futures and self.running:
                # 保存当前未完成的future映射（避免wait后pending_futures被覆盖）
                current_futures = self._pending_futures.copy()
                done_futures = {}
                pending_set = set()
                if self._last_output_channel in current_futures.values():
                    wait_futures = [
                        fut for fut, channel_name in current_futures.items()
                        if channel_name == self._last_output_channel
                    ]
                    _done, _pending = await asyncio.wait(
                        wait_futures,
                        timeout=self.last_channel_timeout,
                        return_when=asyncio.FIRST_COMPLETED  # 有一个完成就返回
                    )
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
                        timeout=self.last_channel_timeout,
                        return_when=asyncio.FIRST_COMPLETED  # 有一个完成就返回
                    )
                    for fut in _done:
                        done_futures[fut] = current_futures[fut]
                    pending_set = _pending

                # 更新pending_futures为未完成的任务（后续继续等待）
                self._pending_futures = {fut: current_futures[fut] for fut in pending_set}

                # 处理已完成的future
                for fut, channel_name in done_futures.items():
                    channel = self.channels[channel_name]

                    try:
                        result = fut.result()  # 获取读取结果（可能抛出异常）
                    except EOFError:
                        await self.on_channel_eof(channel_name)
                        continue
                    except Exception:
                        logger.exception("Failed to read from channel: " + channel_name)
                        await self.on_channel_eof(channel_name)
                        continue

                    await self._channel_read(channel_name, result)

                    next_fut = asyncio.create_task(channel.read())
                    self._pending_futures[next_fut] = channel_name
            if self._last_output_channel is not None:
                await self.on_channel_end(self._last_output_channel)
                await self.on_channel_eof(self._last_output_channel)
                self._last_output_channel = None
        except asyncio.CancelledError:
            pass
        finally:
            # 清理资源：取消所有未完成的读取任务
            for fut in self._pending_futures:
                fut.cancel()
            # 等待所有任务取消完成（避免资源泄漏）
            await asyncio.gather(*self._pending_futures.keys(), return_exceptions=True)


class DefaultTextChannelCollector(ChannelCollector):
    async def on_channel_start(self, channel_name: str):
        print('[%s]' % channel_name)

    async def on_channel_read(self, channel_name: str, message):
        if message is None:
            print('')
        elif message:
            print(message, end='', flush=True)

    async def on_channel_end(self, channel_name: str):
        print('\n', flush=True)

    async def on_channel_eof(self, channel_name: str):
        print('=' * 30)

