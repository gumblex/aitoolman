#!/usr/bin/env python3
"""Mock OpenAI-compatible LLM server for AIToolMan network tests.

Implements POST /v1/chat/completions supporting batch JSON and SSE streaming
responses. Scenarios are selected by the ``model`` field in the request body:

- ``mock-normal``: normal batch JSON response
- ``mock-stream``: normal SSE streaming response
- ``mock-slow-batch``: delayed batch response (for timeout_batch tests)
- ``mock-slow-stream``: SSE stream with a pause (for timeout tests)
- ``mock-auth-error``: always returns 401

Delay durations can be controlled via the ``X-Mock-Response-Delay`` request
header (in seconds). If absent, built-in default delays are used for the slow
scenarios.

Usage:
    python test/mock_llm_server.py --host 127.0.0.1 --port 8765 --client-token secret
"""

import argparse
import asyncio
import json
import time
from typing import Any, Dict, Optional

from aiohttp import web
from aiohttp_sse import sse_response

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

# Scenario model names
MODEL_NORMAL = "mock-normal"
MODEL_STREAM = "mock-stream"
MODEL_SLOW_BATCH = "mock-slow-batch"
MODEL_SLOW_STREAM = "mock-slow-stream"
MODEL_AUTH_ERROR = "mock-auth-error"

# Default delays (seconds) for slow scenarios
DEFAULT_SLOW_BATCH_DELAY = 5.0
DEFAULT_SLOW_STREAM_DELAY = 2.0

# Interval between SSE chunks in normal streaming
STREAM_CHUNK_INTERVAL = 0.1

# Response content parts for streaming scenarios
STREAM_CONTENT_PARTS = ["Hello", " from ", "mock", " server!"]
SLOW_STREAM_CONTENT_PARTS = ["Hello", " from ", "slow", " stream!"]

# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


def _make_response_id(model: str) -> str:
    """Generate a unique response ID."""
    return f"chatcmpl-{int(time.time() * 1000)}-{model}"


def _make_batch_response(model: str, content: str) -> Dict[str, Any]:
    """Build an OpenAI-compatible non-streaming response body."""
    return {
        "id": _make_response_id(model),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


def _make_stream_chunk(
    model: str,
    content: str = "",
    role: str = "",
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an OpenAI-compatible streaming chunk body."""
    delta: Dict[str, Any] = {}
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content
    return {
        "id": _make_response_id(model),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _error_response(message: str, code: str, status: int = 400) -> web.Response:
    """Build an OpenAI-compatible error response."""
    return web.json_response(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": code,
            }
        },
        status=status,
    )


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------


def _get_delay(request: web.Request, default: float) -> float:
    """Read delay from X-Mock-Response-Delay header, fallback to default."""
    header = request.headers.get("X-Mock-Response-Delay")
    if header is None:
        return default
    try:
        return max(0.0, float(header))
    except ValueError:
        return default


async def _handle_chat_completions(request: web.Request) -> web.StreamResponse:
    """Handle POST /v1/chat/completions."""
    # 1. Authentication check
    client_token = request.app["client_token"]
    if client_token is not None:
        auth_header = request.headers.get("Authorization", "")
        if auth_header != f"Bearer {client_token}":
            return _error_response(
                "Invalid authentication credentials",
                "invalid_api_key",
                401,
            )

    # 2. Parse request body
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return _error_response("Invalid JSON body", "invalid_json")

    model = body.get("model", MODEL_NORMAL)
    stream = body.get("stream", False)

    # 3. Auth-error scenario (after authentication check, so tests can verify
    #    both invalid-token rejection and server-side auth failure)
    if model == MODEL_AUTH_ERROR:
        return _error_response(
            "Invalid authentication credentials",
            "invalid_api_key",
            401,
        )

    # 4. Slow batch scenario: delay then return a batch response
    if model == MODEL_SLOW_BATCH:
        delay = _get_delay(request, DEFAULT_SLOW_BATCH_DELAY)
        await asyncio.sleep(delay)
        return web.json_response(
            _make_batch_response(model, "This is a slow batch response.")
        )

    # 5. Batch responses for non-streaming requests and mock-normal
    if not stream or model == MODEL_NORMAL:
        return web.json_response(
            _make_batch_response(model, "This is a mock response.")
        )

    # 6. Streaming scenarios (model in MODEL_STREAM / MODEL_SLOW_STREAM)
    if model == MODEL_SLOW_STREAM:
        content_parts = SLOW_STREAM_CONTENT_PARTS
        pause_delay = _get_delay(request, DEFAULT_SLOW_STREAM_DELAY)
    else:
        content_parts = STREAM_CONTENT_PARTS
        pause_delay = 0.0

    async with sse_response(request) as resp:
        # First chunk: assistant role
        await resp.send(json.dumps(_make_stream_chunk(model, role="assistant")))
        await asyncio.sleep(STREAM_CHUNK_INTERVAL)

        # Content chunks (with optional pause for slow-stream scenario)
        for i, part in enumerate(content_parts):
            if model == MODEL_SLOW_STREAM and i == 1:
                await asyncio.sleep(pause_delay)
            await resp.send(json.dumps(_make_stream_chunk(model, content=part)))
            await asyncio.sleep(STREAM_CHUNK_INTERVAL)

        # Finish chunk
        await resp.send(
            json.dumps(_make_stream_chunk(model, finish_reason="stop"))
        )
        await resp.send("[DONE]")

    return resp


async def _handle_index(request: web.Request) -> web.StreamResponse:
    """Health check endpoint."""
    return web.Response(
        text="Mock OpenAI LLM server is running.", content_type="text/plain"
    )


# ---------------------------------------------------------------------------
# Application and entry point
# ---------------------------------------------------------------------------


def create_app(client_token: Optional[str]) -> web.Application:
    """Create the aiohttp application."""
    app = web.Application()
    app["client_token"] = client_token
    app.router.add_post("/v1/chat/completions", _handle_chat_completions)
    app.router.add_get("/", _handle_index)
    return app


def main() -> None:
    """Parse command-line arguments and start the server."""
    parser = argparse.ArgumentParser(
        description="Mock OpenAI-compatible LLM server for AIToolMan tests"
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host to bind (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to bind (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--client-token",
        default=None,
        help="Expected Bearer token for Authorization header",
    )
    args = parser.parse_args()

    app = create_app(args.client_token)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
