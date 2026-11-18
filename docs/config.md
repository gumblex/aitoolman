# 配置文件文档

## 1. 服务器配置 (llm_provider.toml)

### 1.1 [server] 部分
配置 ZeroMQ 服务器的网络端口。

| 参数                   | 类型                 | 默认值        | 说明                                                                       |
|----------------------|--------------------|------------|--------------------------------------------------------------------------|
| `zmq_router_rpc`     | 字符串                | 无（必须配置）    | ZeroMQ ROUTER socket 的绑定地址，用于处理客户端请求。格式：`tcp://*:端口号` 或 `tcp://IP地址:端口号` |
| `zmq_pub_event`      | 字符串                | 无（可选）      | ZeroMQ PUB socket 地址，用于发布审计日志。格式同上                                       |
| `zmq_pub_event_type` | 无/"bind"/"connect" | "bind"（可选） | ZeroMQ PUB socket 的绑定方式，监听新端口或连接已有端口                                     |
| `zmq_auth_token`     | 字符串                | 无（必须配置）    | 任意字符串，用于认证                                                               |
| `zmq_manage_token`   | 字符串                | 无（可选）      | 管理权限认证令牌，用于客户端管理操作的权限验证                                                   |


### 1.2 [default] 部分
默认配置，会被所有模型配置继承。

| 参数               | 类型  | 默认值      | 说明                                  |
|------------------|-----|----------|-------------------------------------|
| `timeout`        | 浮点数 | 10       | 网络活动超时时间（秒），stream=true 的模型使用该超时   |
| `timeout_batch`  | 浮点数 | 300      | 网络活动超时时间（秒），stream=false 的模型使用该超时  |
| `max_retries`    | 整数  | 0        | 请求失败时的最大重试次数                        |
| `parallel`       | 整数  | 10       | 默认模型并行处理能力（同时处理的请求数）                |
| `retry_duration` | 浮点数 | 0.5      | 初始重试间隔（秒）                           |
| `retry_factor`   | 浮点数 | 1.5      | 重试间隔乘数，用于指数退避算法                     |
| `api_type`       | 字符串 | "openai" | 默认 API 类型，可选值："openai"、"anthropic"、"responses" |
| `headers`        | 字典  | `{}`     | 默认 HTTP 请求头，可在此配置通用认证信息             |
| `body_options`   | 字典  | `{}`     | 默认的请求选项，如 max_tokens, temperature 等 |
| `max_input_tokens` | 整数  | 无（不限）    | 模型支持的最大输入Token数                    |
| `bytes_per_token`  | 浮点数 | 4.0      | 单Token对应的UTF-8字节数估算值，用于Token数估算     |
| `rank_adjust_ratio`| 浮点数 | 0.25     | 权重调整参数，用于标签内候选优先级评分          |
| `loop_detect_length` | 整数 | 0 | 流式死循环检测的预期能检测到的循环长度（字符），<=0 关闭，推荐 400 |



### 1.3 [api."模型名称"] 部分
每个模型的具体配置，模型名称可以自定义。

| 参数             | 类型  | 默认值                              | 说明                              |
|----------------|-----|----------------------------------|---------------------------------|
| `url`          | 字符串 | 无（必须配置）                          | API 端点 URL                      |
| `type`         | 字符串 | 继承自 `[default].api_type`         | API 类型，可选值："openai"、"anthropic"、"responses" |
| `model`        | 字符串 | 无（必须配置）                          | 提供商侧的模型标识符                      |
| `parallel`     | 整数  | 继承自 `[default].parallel`         | 该模型的并行处理能力                      |
| `headers`      | 字典  | 继承自 `[default].headers`          | 该模型特定的 HTTP 请求头                 |
| `timeout`      | 浮点数 | 继承自 `[default].timeout`          | 该模型特定的流式请求（stream=true）超时时间        |
| `timeout_batch`| 浮点数 | 继承自 `[default].timeout_batch`    | 该模型特定的批量请求（stream=false）超时时间      |
| `body_options` | 字典  | 继承自 `[default].body_options`     | 该模型默认的请求选项，覆盖 default 中的所有选项    |
| `max_input_tokens` | 整数  | 继承自 `[default].max_input_tokens` | 该模型支持的最大输入Token数 |
| `bytes_per_token`  | 浮点数 | 继承自 `[default].bytes_per_token`  | 该模型单Token对应的UTF-8字节数估算值 |
| `loop_detect_length` | 整数 | 继承自 `[default].loop_detect_length` | 该模型流式死循环检测的预期能检测到的循环长度（字符） |
| `enable`           | 布尔值 | true                            | 设为false则暂停该模型的调度使用   |


其中：
* "openai" 格式，headers 应有 `Authorization`
* "anthropic" 格式，headers 应有 `"X-Api-Key"`, `"anthropic-version" = "2023-06-01"`；body_options 应有 `max_tokens`
* "responses" 格式，headers 应有 `Authorization`；body_options 可按提供商文档配置，如 `store`、`reasoning` 等


### 1.4 [model_tag] / [model_provider] 部分
模型路由配置分为 `model_provider` 与 `model_tag` 两类：`model_provider` 用于把效果相同的多个提供商组织成一个逻辑模型，主要用于负载均衡和提高可用性；`model_tag` 用于根据成本/效果选择具体模型。

业务代码可以使用标签，无需关心底层具体模型，支持灵活切换和路由。模型路由支持同时传入多个标签，系统会自动匹配同时符合所有标签的可用模型，结合输入Token数限制、模型优先级自动选择最优模型。例如传入 `"code,multimodal"` 可自动选择支持多模态的代码模型，传入 `"cheap,fast"` 可自动选择低成本且响应快的模型，适配各类交叉场景。

`[model_provider]` 使用 `逻辑模型名 = ["[api] 配置名", ...]`。逻辑模型名可用于 `[model_tag]` 或直接调用；其权重与逻辑模型名所在位置的优先级共享，同一逻辑模型的提供商按队列长度及 `context_id`（未提供时为 `client_id`）稳定随机选择。逻辑模型名不得与 `[api]` 配置名或标签同名。未在此声明的 `[api]` 配置仍可独立使用。

`标签名 = 字符串列表（支持一层嵌套）`。键为标签名，值为模型列表；列表顺序指定候选优先级，嵌套列表中的模型共享该位置的优先级。模型可为 `[api]` 配置名/模型名或 `[model_provider]` 逻辑模型名。业务代码可使用单个标签，或用英文逗号分隔多个标签；组合标签取同时满足所有标签的可用模型，综合各标签权重计算评分，选择评分最高者，并结合输入 Token 数限制与模型可用性完成选择。

**示例：**
```toml
[model_provider]
deepseek = ["deepseek-provider1", "deepseek-provider2"]
"glm-5.3-flash" = ["glm-5.3-flash-provider1", "glm-5.3-flash-provider2"]

[model_tag]
image = [["deepseek", "glm-5.3-flash"], 'doubao-seed-2.0-pro']
fast = ['deepseek','qwen-flash']
cheap = ["doubao-seed-2.0-mini", "qwen-flash", "deepseek"]
```

**权重计算：**
列表中模型的权重按位置计算，越靠前权重越高。嵌套列表中的所有模型共享其所在位置的权重；组合标签按综合评分排序。`rank_adjust_ratio` 参数控制排名权重的衰减速率，默认 0.25。

**全局 :N 语法：**
调用 `resolve_model` 或模块 `model_name` 时，可传入单个字符串，多个标签以英文逗号分隔。整串末尾可附加 `:N` 后缀，用于从第 N 位候选开始，前面的模型移动到队列最后。`:N` 仅操作模型层候选列表；`model_rank` 的旋转对象由 `rank_by_model` 决定：`rank_by_model=True` 时 `model_rank` 与 `:N` 在模型层叠加，`rank_by_model=False`（默认）时 `model_rank` 仅作用于展开后的提供商列表。在用户代码中遇到可重试错误时，可将 `model_rank` 加 1 后重试，切换到下一个候选。后缀非整数、或分隔项无匹配时抛出 `LLMNoAvailableModelError`。


### 1.5 [http] 部分
HTTP 服务（`python3 -m aitoolman http`）的配置，详见 docs/http_server.md。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `host` | 字符串 | "127.0.0.1" | HTTP 监听地址 |
| `port` | 整数 | 8000 | HTTP 监听端口 |
| `zmq_endpoint` | 字符串 | 无（可选） | 已有 ZMQ 服务端点，配置后为远程模式；缺省为直连模式（HTTP 服务通过配置的提供商详情直接提供服务） |
| `zmq_auth_token` | 字符串 | 无（可选） | 远程模式的 ZMQ 认证令牌，远程模式必须配置 |
| `client_tokens` | 字符串列表 | 无（必须非空） | 客户端 Bearer 认证令牌列表，为空则拒绝启动 |

CLI 参数 `--host`/`--port`/`--zmq-endpoint`/`--auth`/`--client-token` 可逐项覆盖此节配置（CLI > [http] 节 > 内置默认），`--client-token` 为追加语义。

### 1.6 配置示例
```toml
[server]
zmq_router_rpc = "tcp://*:5555"
zmq_pub_event = "tcp://*:5556"
zmq_auth_token = "YOUR_SECRET_TOKEN"  # 接口认证令牌
zmq_manage_token = "YOUR_MANAGE_TOKEN"  # 管理权限认证令牌（可选）

[default]
timeout = 10
timeout_batch = 300
max_retries = 3
parallel = 1
retry_duration = 0.5
retry_factor = 1.5
api_type = "openai"
loop_detect_length = 400

[model_tag]
"cheap" = ["doubao-seed-2.0-mini", "qwen-flash", "deepseek-v4-flash"]
"fast" = ["deepseek-v4-flash", "qwen-flash", "doubao-seed-2.0-lite"]
"creative" = ["deepseek-v4-pro", "kimi-k2.5"]
"precise" = ["Doubao-Seed-2.0-pro", "glm-5.1"]
"code" = ["glm-5.1", "qwen3.7-max", "deepseek-v4-pro", "kimi-k2.5"]
"multimodal" = ["Doubao-Seed-2.0-pro", "doubao-seed-2.0-mini", "qwen-vl"]

[api."deepseek-v4-pro"]
url = "https://api.deepseek.com/chat/completions"
type = "openai"
model = "deepseek-v4-pro"
parallel = 10
headers = {Authorization = "Bearer sk-xxx"}
body_options.thinking.type = "enabled"
body_options.reasoning_effort = "max"

[api."Doubao-Seed-2.0-pro"]
url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
type = "openai"
model = "ep-aaa"
headers = {Authorization = "Bearer xxx"}

[api."batch-Doubao-Seed-2.0-lite"]
url = "https://ark.cn-beijing.volces.com/api/v3/batch/chat/completions"
type = "openai"
model = "ep-bbb"
parallel = 100000
headers = {Authorization = "Bearer xxx"}

[http]
host = "127.0.0.1"
port = 8000
client_tokens = ["sk-xxx"]
# zmq_endpoint = "tcp://127.0.0.1:5555"  # 配置后为远程模式，连接已有 ZMQ 服务
# zmq_auth_token = "YOUR_SECRET_TOKEN"   # 远程模式认证令牌
```

## 2. 应用模块配置 (app_prompt.toml)

### 2.1 [module_default] 部分
默认模块配置，会被所有模块继承。

| 参数                  | 类型  | 默认值         | 说明                                |
|---------------------|-----|-------------|-----------------------------------|
| `model`             | 字符串 | 无（建议配置）     | 默认使用的模型名称或别名                      |
| `stream`            | 布尔值 | false       | 是否使用流式输出                          |
| `post_processor`    | 字符串 | 无           | 后处理器名称，需在应用中注册                    |
| `options`           | 字典  | `{}`        | 默认请求选项，如 temperature、max_tokens 等 |

### 2.2 [module."模块名称"] 部分
每个模块的具体配置，模块名称可以自定义。

| 参数                  | 类型  | 默认值                                      | 说明                                   |
|---------------------|-----|------------------------------------------|--------------------------------------|
| `description`       | 字符串 | `''`                                     | 该模块的描述文本                             |
| `model`             | 字符串 | 继承自 `[module_default].model`             | 该模块使用的模型名称或别名                        |
| `stream`            | 布尔值 | 继承自 `[module_default].stream`            | 该模块是否使用流式输出                          |
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

### 2.4 [model_alias] 部分
应用层模型别名配置，用于灵活配置具体模型。

| 参数   | 类型        | 默认值 | 说明            |
|------|-----------|-----|---------------|
| `别名` | 字符串 | 无   | 映射到实际的模型或标签名称 |

### 2.5 模板语法
模板使用 Jinja2 语法，支持变量替换和基本控制结构。

**可用变量：**
- `app.vars` 中定义的全局变量
- 调用模块时传入的关键字参数

**示例：**
```jinja2
文章标题：{{title}}
文章内容：<article>{{content}}</article>
```

### 2.6 工具调用配置格式
工具配置采用 TOML 的嵌套结构：

```toml
[module."模块名称"]
tools."工具名称".type = "function"  # 固定值
tools."工具名称".description = "工具描述"

tools."工具名称".param."参数名".type = "string"  # 参数类型：string、integer、boolean 等
tools."工具名称".param."参数名".description = "参数描述"
tools."工具名称".param."参数名".required = true   # 是否必需
```

### 2.7 配置示例
```toml
[module_default]
model = "Fast-Model"  # 使用模型别名
stream = false

[module.raw]
template.user = """{{content}}"""

[module.summerize]
model = "precise"  # 使用模型别名
template.user = """文章标题：{{title}}
文章内容：<article>{{content}}</article>
请根据文章内容：
1. 按文章的结构列出论点和重要观点
2. 列出文中的案例及说明的问题
3. 总结这篇文章"""

[module.creative_writing]
model = "creative"  # 使用模型别名
template.user = """请以{{style}}风格创作一篇关于{{topic}}的文章，字数要求{{word_count}}字左右"""
options = {temperature = 0.8, max_tokens = 2000}

[module.code_generator]
model = "code"  # 使用模型别名
template.user = """请使用{{language}}语言编写一个{{functionality}}的代码示例，并添加详细注释"""

[module.task_adder]
model = "fast,cheap"
stream = true
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
server_config = aitoolman.load_config('llm_provider.toml')

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
result = await app['summerize'](
    title="人工智能发展",
    content="人工智能近年来取得了突破性进展..."
)

# 调用 task_adder 模块（支持工具调用）
result = await app['task_adder'](
    user_input="明天下午3点开会"
)
```

### 3.3 动态添加配置
```python
# 动态添加模块配置
app.add_module(module_config: ModuleConfig)

# 动态添加后处理器
app.add_processor('custom_parser', lambda x: x.split('\n'))
```

## 4. 注意事项

1. **模型名称一致性**：`app_prompt.toml` 中的 `model` 可以是 `[api]` 配置名、`[model_provider]` 逻辑模型名、`[model_tag]` 标签名、英文逗号分隔的标签组合，也可以是 `[model_alias]` 部分定义的别名。

2. **模板变量**：模板中使用的变量必须在调用时提供，否则会渲染失败。

3. **流式输出**：当 `stream=true` 时，输出会通过通道的 `write` 方法分片发送。

4. **工具调用**：工具配置必须包含完整的参数定义，否则可能无法正确解析。

5. **模型路由**：`model_tag` 用于按成本/效果选择具体模型，列表顺序指定候选优先级，嵌套列表表示相同优先级；多个标签组合时综合各标签权重计算评分，选择评分最高者。
