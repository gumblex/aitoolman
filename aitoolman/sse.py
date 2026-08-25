"""
Adapted from https://pypi.org/project/httpx-sse v0.4.3

Copyright (c) 2022 Florimond Manca
License: MIT
"""
import json
from typing import Any, AsyncGenerator, Generator, List, NamedTuple, Optional

import httpx


class SSEError(httpx.TransportError):
    """Raised when an SSE-related error occurs."""


class ServerSentEvent(NamedTuple):
    event: str = "message"
    data: bytes = b""
    id: str = ""
    retry: Optional[int] = None

    def text(self) -> str:
        return self.data.decode('utf-8')

    def json(self) -> Any:
        return json.loads(self.data.decode('utf-8'))


def check_content_type(response: httpx.Response) -> None:
    content_type = response.headers.get("content-type", "").partition(";")[0]
    if "text/event-stream" not in content_type:
        raise SSEError(
            "Expected response header Content-Type to contain 'text/event-stream', "
            f"got {content_type!r}"
        )


class SSEDecoder:
    """Incremental SSE decoder operating directly on bytes.

    Uses a single ``bytearray`` buffer and returns lists of completed
    :class:`ServerSentEvent` objects, avoiding the overhead of nested
    generators and multiple buffering layers.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._event: bytes = b""
        self._data: List[bytes] = []
        self._last_event_id: bytes = b""
        self._retry: Optional[int] = None

    def decode(self, chunk: bytes, final: bool = False) -> List[ServerSentEvent]:
        """Feed a chunk of bytes and return any completed events.

        When *final* is ``True`` any remaining buffered bytes are processed
        as a final (newline-less) line.
        """
        self._buffer.extend(chunk)
        events: List[ServerSentEvent] = []

        buf = self._buffer
        start = 0

        while start < len(buf):
            nl = buf.find(b"\n", start)
            cr = buf.find(b"\r", start)

            if nl == -1 and cr == -1:
                break

            if cr == -1 or (nl != -1 and nl < cr):
                # \n comes first (or is the only match)
                line = bytes(buf[start:nl])
                start = nl + 1
            else:
                # \r comes first
                if cr + 1 < len(buf):
                    if buf[cr + 1] == 10:  # \r\n
                        line = bytes(buf[start:cr])
                        start = cr + 2
                    else:  # \r only
                        line = bytes(buf[start:cr])
                        start = cr + 1
                elif not final:
                    # \r at end of buffer; could be start of \r\n
                    break
                else:
                    line = bytes(buf[start:cr])
                    start = cr + 1

            event = self._process_line(line)
            if event is not None:
                events.append(event)

        # Compact the buffer
        if start > 0:
            del buf[:start]

        if final and buf:
            event = self._process_line(bytes(buf))
            if event is not None:
                events.append(event)
            buf.clear()

        return events

    def _process_line(self, line: bytes) -> Optional[ServerSentEvent]:
        # See: https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation

        if not line:
            # Blank line — dispatch event
            if (
                not self._event
                and not self._data
                and not self._last_event_id
                and self._retry is None
            ):
                return None

            sse = ServerSentEvent(
                event=self._event.decode('utf-8') if self._event else "message",
                data=b"\n".join(self._data),
                id=self._last_event_id.decode('utf-8'),
                retry=self._retry,
            )

            # NOTE: as per the SSE spec, do not reset last_event_id.
            self._event = b""
            self._data = []
            self._retry = None

            return sse

        if line.startswith(b":"):
            return None

        idx = line.find(b":")
        if idx == -1:
            fieldname = line
            value = b""
        else:
            fieldname = line[:idx]
            value = line[idx + 1:]

        if value.startswith(b" "):
            value = value[1:]

        if fieldname == b"event":
            self._event = value
        elif fieldname == b"data":
            self._data.append(value)
        elif fieldname == b"id":
            if b"\0" not in value:
                self._last_event_id = value
        elif fieldname == b"retry":
            try:
                self._retry = int(value)
            except (TypeError, ValueError):
                pass

        return None


def iter_sse(response: httpx.Response) -> Generator[ServerSentEvent, None, None]:
    """Synchronously iterate over SSE events from an httpx streaming response."""
    check_content_type(response)
    decoder = SSEDecoder()
    for chunk in response.iter_bytes():
        for event in decoder.decode(chunk):
            yield event
    for event in decoder.decode(b"", final=True):
        yield event


async def aiter_sse(response: httpx.Response) -> AsyncGenerator[ServerSentEvent, None]:
    """Asynchronously iterate over SSE events from an httpx streaming response."""
    check_content_type(response)
    decoder = SSEDecoder()
    async for chunk in response.aiter_bytes():
        for event in decoder.decode(chunk):
            yield event
    for event in decoder.decode(b"", final=True):
        yield event


__all__ = [
    "ServerSentEvent",
    "SSEError",
    "SSEDecoder",
    "check_content_type",
    "iter_sse",
    "aiter_sse",
]
