
# 配置文件文档

## 1. 服务器配置 (llm_config.toml)

### 1.1 [server] 部分
配置 ZeroMQ 服务器的网络端口。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `zmq_router_rpc` | 字符串 | 无（必须配置） | ZeroMQ ROUTER socket 的绑定地址，用于处理客户端请求。格式：`tcp://*:端口号` 或 `tcp://IP地址:端口号` |
| `zmq_pub_event` | 字符串 | 无（必须配置） | ZeroMQ PUB socket 的绑定地址，用于发布审计日志。格式同上 |

### 1.2 [default] 部分
默认配置，会被所有模型配置继承。

| 参数               | 类型  | 默认值      | 说明                                  |
|------------------|-----|----------|-------------------------------------|
| `timeout`        | 整数  | 600      | HTTP 请求超时时间（秒）                      |
| `max_retries`    | 整数  | 3        | 请求失败时的最大重试次数                        |
| `parallel`       | 整数  | 1        | 默认模型并行处理能力（同时处理的请求数）                |
| `retry_duration` | 浮点数 | 0.5      | 初始重试间隔（秒）                           |
| `retry_factor`   | 浮点数 | 1.5      | 重试间隔乘数，用于指数退避算法                     |
| `api_type`       | 字符串 | "openai" | 默认 API 类型，可选值："openai"、"anthropic"  |
| `headers`        | 字典  | `{}`     | 默认 HTTP 请求头，可在此配置通用认证信息             |
| `body_options`   | 字典  | `{}`     | 默认的请求选项，如 max_tokens, temperature 等 |

### 1.3 [api."模型名称"] 部分
每个模型的具体配置，模型名称可以自定义。

| 参数                | 类型  | 默认值                      | 说明                                     |
|-------------------|-----|--------------------------|----------------------------------------|
| `url`             | 字符串 | 无（必须配置）                  | API 端点 URL                             |
| `type`            | 字符串 | 继承自 `[default].api_type` | API 类型，可选值："openai"、"anthropic"        |
| `model`           | 字符串 | 无（必须配置）                  | 提供商侧的模型标识符                             |
| `parallel`        | 整数  | 继承自 `[default].parallel` | 该模型的并行处理能力                             |
| `headers`         | 字典  | 继承自 `[default].headers`  | 该模型特定的 HTTP 请求头                        |
| `timeout`         | 整数  | 继承自 `[default].timeout`  | 该模型特定的超时时间                             |
| `body_options` | 字典  | 继承自 `[default].body_options`  | 该模型默认的请求选项，如 max_tokens, temperature 等 |

### 配置示例
```toml
[server]
zmq_router_rpc = "tcp://*:5555"
zmq_pub_event = "tcp://*:5556"

[default]
timeout = 600
max_retries = 3
parallel = 1
retry_duration = 0.5
retry_factor = 1.5
api_type = "openai"

[api."Doubao-Seed-1.6"]
url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
type = "openai"
model = "ep-aaa"
headers = {Authorization = "Bearer xxx"}

[api."batch-Doubao-Seed-1.6-flash-250828"]
url = "https://ark.cn-beijing.volces.com/api/v3/batch/chat/completions"
type = "openai"
model = "ep-bbb"
parallel = 100000
headers = {Authorization = "Bearer xxx"}
```

## 2. 应用模块配置 (app_prompt.toml)

### 2.1 [module_default] 部分
默认模块配置，会被所有模块继承。

| 参数                  | 类型  | 默认值         | 说明                                        |
|---------------------|-----|-------------|-------------------------------------------|
| `model`             | 字符串 | 无（建议配置）     | 默认使用的模型名称，需在 llm_config.toml 的 [api] 部分定义 |
| `stream`            | 布尔值 | false       | 是否使用流式输出                                  |
| `output_channel`    | 字符串 | "stdout"    | 默认输出通道名称                                  |
| `reasoning_channel` | 字符串 | "reasoning" | 默认推理通道名称                                  |
| `save_context`      | 布尔值 | false       | 是否保存对话上下文                                 |
| `post_processor`    | 字符串 | 无           | 后处理器名称，需在应用中注册                            |
| `options`           | 字典  | `{}`        | 默认请求选项，如 temperature、max_tokens 等         |

### 2.2 [module."模块名称"] 部分
每个模块的具体配置，模块名称可以自定义。

| 参数                  | 类型  | 默认值                                      | 说明                                   |
|---------------------|-----|------------------------------------------|--------------------------------------|
| `model`             | 字符串 | 继承自 `[module_default].model`             | 该模块使用的模型名称                           |
| `stream`            | 布尔值 | 继承自 `[module_default].stream`            | 该模块是否使用流式输出                          |
| `output_channel`    | 字符串 | 继承自 `[module_default].output_channel`    | 该模块的输出通道                             |
| `reasoning_channel` | 字符串 | 继承自 `[module_default].reasoning_channel` | 该模块的推理通道                             |
| `save_context`      | 布尔值 | 继承自 `[module_default].save_context`      | 该模块是否保存上下文                           |
| `post_processor`    | 字符串 | 继承自 `[module_default].post_processor`    | 该模块的后处理器                             |
| `options`           | 字典  | 继承自 `[module_default].options`           | 该模块的请求选项                             |
| `template`          | 字典  | `{}`                                     | 模板配置，必须包含 `user` 模板，可选包含 `system` 模板 |
| `tools`             | 字典  | `{}`                                     | 工具调用配置，格式详见下文                        |

### 2.3 [template] 部分
全局模板，可以用 `LLMApplication.render_template` 渲染。
```toml
[template]
"模板1" = "模板内容{{text}}"
"模板2" = "模板内容{{text2}}"
```

### 2.4 模板语法
模板使用 Jinja2 语法，支持变量替换和基本控制结构。

**可用变量：**
- `app.vars` 中定义的全局变量
- 调用模块时传入的关键字参数

**示例：**
```jinja2
文章标题：{{title}}
文章内容：<article>{{content}}</article>
```

### 2.5 工具调用配置格式
工具配置采用 TOML 的嵌套结构：

```toml
[module."模块名称"]
tools."工具名称".type = "function"  # 固定值
tools."工具名称".description = "工具描述"

tools."工具名称".param."参数名".type = "string"  # 参数类型：string、integer、boolean 等
tools."工具名称".param."参数名".description = "参数描述"
tools."工具名称".param."参数名".required = true   # 是否必需
```

### 配置示例
```toml
[module_default]
model = "Doubao-Seed-1.6"
stream = false
output_channel = "stdout"
save_context = false

[module.raw]
template.user = """{{content}}"""

[module.summerize]
model = "Doubao-Seed-1.6"
template.user = """文章标题：{{title}}
文章内容：<article>{{content}}</article>
请根据文章内容：
1. 按文章的结构列出论点和重要观点
2. 列出文中的案例及说明的问题
3. 总结这篇文章"""

[module.task_adder]
model = "Doubao-Seed-1.6"
stream = true
save_context = true
template.user = "你作为一个日程助手，可以帮用户添加待办事项。分析用户指令，如果有具体的待办事项则调用工具，没有则面向用户，让用户详细说明代表事项。用户说：{{user_input}}"

tools."add_task".type = "function"
tools."add_task".description = "添加日程"

tools."add_task".param."datetime".type = "string"
tools."add_task".param."datetime".description = "日期时间，如 2025-12-31 12:34:56"
tools."add_task".param."datetime".required = false

tools."add_task".param."content".type = "string"
tools."add_task".param."content".description = "待办事项"
tools."add_task".param."content".required = true
```

## 3. 使用示例

### 3.1 加载配置
```python
import aitoolman

# 加载服务器配置
server_config = aitoolman.load_config('llm_config.toml')

# 加载应用配置
app_config = aitoolman.load_config('app_prompt.toml')

# 创建客户端
client = aitoolman.LLMLocalClient(server_config)

# 创建应用
app = aitoolman.LLMApplication(client, app_config)
```

### 3.2 调用模块
```python
# 调用 summerize 模块
result = await app.summerize(
    title="人工智能发展",
    content="人工智能近年来取得了突破性进展..."
)

# 调用 task_adder 模块（支持工具调用）
result = await app.task_adder(
    user_input="明天下午3点开会"
)
```

### 3.3 动态添加配置
```python
# 动态添加模块配置
app.config['module']['new_module'] = {
    'model': 'Doubao-Seed-1.6',
    'template': {
        'user': '{{query}}'
    }
}

# 动态添加后处理器
app.add_processor('custom_parser', lambda x: x.split('\n'))
```

## 4. 注意事项

1. **模型名称一致性**：`app_prompt.toml` 中的 `model` 必须在 `llm_config.toml` 的 `[api]` 部分有对应配置。

2. **通道管理**：默认提供三个通道：
   - `stdin`：标准输入（非片段模式）
   - `stdout`：标准输出（片段模式）
   - `reasoning`：推理输出（片段模式）

3. **模板变量**：模板中使用的变量必须在调用时提供，否则会渲染失败。

4. **流式输出**：当 `stream=true` 时，输出会通过通道的 `write_fragment` 方法分片发送。

5. **上下文保存**：当 `save_context=true` 时，模块会自动保存对话历史，用于后续调用。

6. **工具调用**：工具配置必须包含完整的参数定义，否则可能无法正确解析。
