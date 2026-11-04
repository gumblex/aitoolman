# HTTP 服务

aitoolman 提供 OpenAI 兼容的 HTTP 服务，将配置管理、多模型路由与请求能力包装为标准的 `chat/completions` API，便于与 OpenAI SDK、LangChain 等生态直接集成。

## 安装

HTTP 服务基于 aiohttp，需要安装可选依赖组：

```bash
pip install 'aitoolman[http]'
```

## 功能概述

- `POST /v1/chat/completions`：对话补全，支持阻塞与 SSE 流式两种响应
- `GET /v1/models`：列出可用模型与模型标签
- `GET /health`：健康检查（无需认证）
- Bearer Token 认证
- 直连/远程双后端模式
- 请求 body 中未识别的键全量透传给模型请求 options（优先级高于配置的 body_options）

## 快速开始

1. 在 `llm_provider.toml` 中增加 `[http]` 节：

```toml
[http]
host = "127.0.0.1"
port = 8000
client_tokens = ["sk-my-token"]
```

2. 启动服务：

```bash
python3 -m aitoolman http
```

3. 发起请求：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-my-token" \
  -H "Content-Type: application/json" \
  -d '{"model": "fast", "messages": [{"role": "user", "content": "直接输出 4651*236=?"}]}'
```

## 端点说明

### POST /v1/chat/completions

请求体为 OpenAI chat/completions 格式：

| 字段 | 类型 | 说明 |
|---|---|---|
| `model` | 字符串 | 必填。支持模型名、标签、`tag:rank`、逗号分隔多标签与末尾 `/N` 旋转，经模型路由解析为实际模型 |
| `messages` | 数组 | 必填。OpenAI 消息格式，支持文本与 `image_url` 图片内容；无法识别的形态原样透传 |
| `stream` | 布尔值 | 默认 false。true 时以 SSE 流式返回 |
| `tools` | 数组 | 可选。OpenAI 工具格式，转换为 aitoolman 工具配置 |
| `stream_options` | 对象 | `{"include_usage": true}` 时流式末尾附加 usage chunk |
| 其余字段 | 任意 | 全量透传进请求 options（如 temperature、max_tokens），优先级高于模型配置的 body_options |

响应：

- `id` 形如 `chatcmpl-<request_id>`，`model` 为实际模型名
- 响应头 `x-aitoolman-resolved-model` 透出路由解析后的实际模型
- `choices[0].message` 含 `content`、`reasoning_content`（有思考内容时）、`tool_calls`（模型发起工具调用时）
- `usage` 提供 prompt_tokens/completion_tokens/total_tokens（上游缺失时补 0）

### GET /v1/models

返回 OpenAI 格式模型列表，模型标签同时作为模型条目列出（`id` 为标签名；标签名与模型名重名时模型优先，空标签不输出）：

```json
{"object": "list", "data": [{"id": "模型名或标签名", "object": "model", "created": 0, "owned_by": "aitoolman"}]}
```

标签条目可直接用作请求的 `model` 参数，经模型路由解析为实际模型。

### GET /health

无需认证，返回 `{"status": "ok", "mode": "direct"}`（远程模式为 `"remote"`）。根路径 `/` 返回相同状态信息。

## 认证

除 `/` 与 `/health` 外的所有端点均需 Bearer 认证：

```
Authorization: Bearer <client_tokens 之一>
```

令牌在 `[http].client_tokens` 中配置（或 CLI `--client-token` 追加）。校验失败返回 401 与 `WWW-Authenticate: Bearer` 头。

## 配置

`llm_provider.toml` 的 `[http]` 节：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `host` | 字符串 | `"127.0.0.1"` | HTTP 监听地址 |
| `port` | 整数 | `8000` | HTTP 监听端口 |
| `zmq_endpoint` | 字符串 | 无 | 已有 ZMQ 服务端点（远程模式）；缺省为直连模式 |
| `zmq_auth_token` | 字符串 | 无 | 远程模式 ZMQ 认证令牌（远程模式必填） |
| `client_tokens` | 字符串列表 | 无 | 客户端 Bearer 令牌列表，为空时拒绝启动 |

CLI 参数优先级高于配置：`--host`/`--port`/`--zmq-endpoint`/`--auth`/`--client-token` 可逐项覆盖（`--client-token` 为追加语义，与配置列表合并）。

启动校验（fail-fast）：

- `client_tokens` 为空 → 报错退出（不允许匿名访问）
- 远程模式缺少 `zmq_auth_token` → 报错退出
- aiohttp 未安装 → 提示 `pip install 'aitoolman[http]'` 后退出

## CLI 用法

```
usage: python3 -m aitoolman http [-h] [-v] [-c CONFIG] [--zmq-endpoint ZMQ_ENDPOINT]
                                 [--auth AUTH] [--host HOST] [--port PORT]
                                 [--client-token TOKEN]
```

| 参数 | 说明 |
|---|---|
| `-c, --config` | TOML 配置文件路径，默认 `llm_provider.toml` |
| `--zmq-endpoint` | 已有 ZMQ 服务端点（远程模式）；缺省为直连模式 |
| `--auth` | 远程模式 ZMQ 认证令牌 |
| `--host` / `--port` | HTTP 监听地址/端口 |
| `--client-token` | 客户端 Bearer 令牌，可多次指定 |

## 双后端模式

- **直连模式**（默认，未配置 `zmq_endpoint`）：HTTP 服务通过配置的提供商详情直接提供服务，同一份配置文件同时定义模型提供商与 HTTP 服务。
- **远程模式**（配置 `zmq_endpoint` + `zmq_auth_token`）：HTTP 服务连接已有 ZMQ 服务，HTTP 层与 ZMQ 服务可独立部署。

两种模式下 HTTP 层行为完全一致。

## 流式响应

`stream=true` 时以 SSE 返回（`Content-Type: text/event-stream`），每帧格式为 `data: {json}`：

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"你"}}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[],"usage":{...}}

data: [DONE]
```

- `delta.content` 为正文增量、`delta.reasoning_content` 为思考内容增量
- delta 与 chunk 中未设置的字段不会出现在 JSON 中（不输出 null 值键）
- usage chunk 仅在 `stream_options.include_usage=true` 时发送（choices 为空数组）
- 流中发生错误时发送一帧错误信封，随后仍以 `data: [DONE]` 结束
- 客户端断连时自动取消上游请求

## 错误格式

统一错误信封 `{"error": {"message", "type", "code"}}`：

| 场景 | HTTP 状态 | type |
|---|---|---|
| 缺少/无效 Bearer 令牌 | 401 | `authentication_error` |
| body 非法 JSON、缺 `model`、`messages`/`tools` 非法 | 400 | `invalid_request_error` |
| 无可用模型（路由失败） | 404 | `model_not_found` |
| 输入超过模型 Token 限制 | 400 | `context_length_exceeded` |
| 触发内容过滤 | 400 | `content_filter` |
| 上游请求失败 | 502 | `upstream_error` |
| 其他内部错误 | 500 | `internal_error` |

## curl 示例

阻塞请求：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-my-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fast",
    "messages": [{"role": "user", "content": "用一句话介绍自己"}],
    "temperature": 0.7
  }'
```

流式请求（`-N` 关闭缓冲）：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-my-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fast",
    "messages": [{"role": "user", "content": "讲个故事"}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

模型列表与健康检查：

```bash
curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer sk-my-token"
curl http://127.0.0.1:8000/health
```
