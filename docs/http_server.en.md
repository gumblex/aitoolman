# HTTP Server

aitoolman provides an OpenAI-compatible HTTP server that wraps configuration management, multi-model routing, and request capabilities into a standard `chat/completions` API, allowing direct integration with the OpenAI SDK, LangChain, and other ecosystems.

## Installation

The HTTP server is based on aiohttp. Install the optional dependency group:

```bash
pip install 'aitoolman[http]'
```

## Features

- `POST /v1/chat/completions`: chat completions, supporting both blocking and SSE streaming responses
- `GET /v1/models`: list available models and model tags
- `GET /health`: health check (no authentication required)
- Bearer Token authentication
- Direct/remote dual backend modes
- Built-in mock scenarios for testing
- Unrecognized keys in the request body are passed through to the model request options (taking priority over the configured body_options)

## Quick Start

1. Add an `[http]` section to `llm_provider.toml`:

```toml
[http]
host = "127.0.0.1"
port = 8000
client_tokens = ["sk-my-token"]
```

2. Start the server:

```bash
python3 -m aitoolman http
```

3. Send a request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-my-token" \
  -H "Content-Type: application/json" \
  -d '{"model": "fast", "messages": [{"role": "user", "content": "What is 4651*236?"}]}'
```

## Endpoints

### POST /v1/chat/completions

The request body follows the OpenAI chat/completions format:

| Field | Type | Description |
|---|---|---|
| `model` | String | Required. Supports model names, tags, `tag:rank`, comma-separated multiple tags, and a trailing `/N` rotation; resolved to the actual model via model routing |
| `messages` | Array | Required. OpenAI message format, supporting text and `image_url` image content; unrecognized forms are passed through as-is |
| `stream` | Boolean | Default false. When true, the response is returned as an SSE stream |
| `tools` | Array | Optional. OpenAI tool format, converted to aitoolman tool configuration |
| `stream_options` | Object | When `{"include_usage": true}`, a usage chunk is appended at the end of the stream |
| Other fields | Any | Passed through into the request options (e.g., temperature, max_tokens), taking priority over the model's configured body_options |

Response:

- `id` looks like `chatcmpl-<request_id>`, and `model` is the actual model name
- The response header `x-aitoolman-resolved-model` exposes the actual model resolved by routing
- `choices[0].message` contains `content`, `reasoning_content` (when thinking content exists), and `tool_calls` (when the model initiates tool calls)
- `usage` provides prompt_tokens/completion_tokens/total_tokens (filled with 0 when missing upstream)

### GET /v1/models

Returns the OpenAI-format model list, with model tags also listed as model entries (`id` is the tag name; when a tag name duplicates a model name, the model takes precedence, and empty tags are omitted):

```json
{"object": "list", "data": [{"id": "model or tag name", "object": "model", "created": 0, "owned_by": "aitoolman"}]}
```

Tag entries can be used directly as the `model` parameter of a request (resolved to the actual model via model routing).

### GET /health

No authentication required; returns `{"status": "ok", "mode": "direct"}` (`"remote"` in remote mode). The root path `/` returns the same status information.

## Authentication

All endpoints except `/` and `/health` require Bearer authentication:

```
Authorization: Bearer <one of client_tokens>
```

Tokens are configured in `[http].client_tokens` (or appended via the CLI `--client-token`). Failed validation returns 401 with a `WWW-Authenticate: Bearer` header.

## Configuration

The `[http]` section of `llm_provider.toml`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `host` | String | `"127.0.0.1"` | HTTP bind address |
| `port` | Integer | `8000` | HTTP bind port |
| `zmq_endpoint` | String | None | Endpoint of an existing ZMQ server (remote mode); omit for direct mode |
| `zmq_auth_token` | String | None | ZMQ authentication token for remote mode (required in remote mode) |
| `client_tokens` | String list | None | Client Bearer token list; startup is rejected when empty |
| `mock` | Boolean | `false` | Enable built-in mock scenarios |

CLI parameters take priority over configuration: `--host`/`--port`/`--zmq-endpoint`/`--auth`/`--client-token`/`--mock` can override each item (`--client-token` is append-only, merged with the configuration list).

Startup validation (fail-fast):

- Empty `client_tokens` → error and exit (anonymous access is not allowed)
- Remote mode missing `zmq_auth_token` → error and exit
- aiohttp not installed → prompts `pip install 'aitoolman[http]'` and exits

## CLI Usage

```
usage: python3 -m aitoolman http [-h] [-v] [-c CONFIG] [--zmq-endpoint ZMQ_ENDPOINT]
                                 [--auth AUTH] [--host HOST] [--port PORT]
                                 [--client-token TOKEN] [--mock]
```

| Parameter | Description |
|---|---|
| `-c, --config` | TOML configuration file path, default `llm_provider.toml` |
| `--zmq-endpoint` | Endpoint of an existing ZMQ server (remote mode); omit for direct mode |
| `--auth` | ZMQ authentication token for remote mode |
| `--host` / `--port` | HTTP bind address/port |
| `--client-token` | Client Bearer token, can be specified multiple times |
| `--mock` | Enable built-in mock scenarios |

## Dual Backend Modes

- **Direct mode** (default, `zmq_endpoint` not configured): the HTTP server serves directly using the provider details from the configuration file; the same configuration file defines both the model providers and the HTTP service.
- **Remote mode** (`zmq_endpoint` + `zmq_auth_token` configured): the HTTP server connects to an existing ZMQ server, and the HTTP layer and the ZMQ server can be deployed independently.

HTTP behavior is identical in both modes.

## Streaming Response

When `stream=true`, the response is returned as SSE (`Content-Type: text/event-stream`), with each frame formatted as `data: {json}`:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hi"}}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[],"usage":{...}}

data: [DONE]
```

- `delta.content` carries the content increments, `delta.reasoning_content` the reasoning increments
- Unset fields in the delta and chunk are omitted from the JSON (no null-valued keys)
- The usage chunk is sent only when `stream_options.include_usage=true` (with an empty choices array)
- When an error occurs mid-stream, an error envelope frame is sent, followed by `data: [DONE]`
- Upstream requests are cancelled automatically when the client disconnects

## Mock Mode

When `--mock` or `[http].mock=true` is enabled, requests whose `model` matches one of the following scenario names return a fixed response directly (intercepted after authentication and before model resolution), without a real upstream:

| model name | Behavior |
|---|---|
| `mock-normal` | Fixed blocking reply |
| `mock-stream` | SSE chunked reply (blocking reply when `stream=false`) |
| `mock-slow-batch` | Blocking reply after a delay (default 5 seconds) |
| `mock-slow-stream` | Pauses before the second content chunk (default 2 seconds) |
| `mock-heartbeat` | Only heartbeat events, ending after 5 seconds |
| `mock-heartbeat-content` | Heartbeats interleaved with content chunks |
| `mock-auth-error` | Returns 401, simulating upstream authentication failure |

- The request header `X-Mock-Response-Delay: <seconds>` overrides the slow-scenario delay
- The response `model` field echoes the requested model name for easy assertions

## Error Format

Unified error envelope `{"error": {"message", "type", "code"}}`:

| Scenario | HTTP Status | type |
|---|---|---|
| Missing/invalid Bearer token | 401 | `authentication_error` |
| Invalid JSON body, missing `model`, invalid `messages`/`tools` | 400 | `invalid_request_error` |
| No available model (routing failure) | 404 | `model_not_found` |
| Input exceeds the model token limit | 400 | `context_length_exceeded` |
| Content filter triggered | 400 | `content_filter` |
| Upstream request failure | 502 | `upstream_error` |
| Other internal errors | 500 | `internal_error` |

## curl Examples

Blocking request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-my-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fast",
    "messages": [{"role": "user", "content": "Introduce yourself in one sentence"}],
    "temperature": 0.7
  }'
```

Streaming request (`-N` disables buffering):

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-my-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fast",
    "messages": [{"role": "user", "content": "Tell a story"}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

Model list and health check:

```bash
curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer sk-my-token"
curl http://127.0.0.1:8000/health
```
