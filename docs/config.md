# 配置文件文档

## 1. 服务器配置 (llm_provider.toml)

### 1.1 [server] 部分
配置 ZeroMQ 服务器的网络端口。

| 参数                   | 类型                 | 默认值        | 说明                                                                       |
|----------------------|--------------------|------------|--------------------------------------------------------------------------|
| `zmq_router_rpc`     | 字符串                | 无（必须配置）    | ZeroMQ ROUTER socket 的绑定地址，用于处理客户端请求。格式：`tcp://*:端口号` 或 `tcp://IP地址:端口号` |
| `zmq_pub_event`      | 字符串                | 无（可选）      | ZeroMQ PUB socket 地址，用于发布审计日志。格式同上                                       |
| `zmq_pub_event_type` | 无/"bind"/"connect" | "bind"（可选） | ZeroMQ PUB socket 的绑定方式，监听新端口或连接已有端口                                     |
| `zmq_auth_token`     | 字符串                | 无（可选）      | 任意字符串，用于认证                                                               |
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
| `api_type`       | 字符串 | "openai" | 默认 API 类型，可选值："openai"、"anthropic"  |
| `headers`        | 字典  | `{}`     | 默认 HTTP 请求头，可在此配置通用认证信息             |
| `body_options`   | 字典  | `{}`     | 默认的请求选项，如 max_tokens, temperature 等 |
| `max_input_tokens` | 整数  | 无（不限）    | 模型支持的最大输入Token数                    |
| `bytes_per_token`  | 浮点数 | 4.0      | 单Token对应的UTF-8字节数估算值，用于Token数估算     |
| `rank_adjust_ratio`| 浮点数 | 0.25     | 权重调整参数，用于模型路由算法中的排名权重计算          |


### 1.3 [api."模型名称"] 部分
每个模型的具体配置，模型名称可以自定义。

| 参数             | 类型  | 默认值                              | 说明                              |
|----------------|-----|----------------------------------|---------------------------------|
| `url`          | 字符串 | 无（必须配置）                          | API 端点 URL                      |
| `type`         | 字符串 | 继承自 `[default].api_type`         | API 类型，可选值："openai"、"anthropic" |
| `model`        | 字符串 | 无（必须配置）                          | 提供商侧的模型标识符                      |
| `parallel`     | 整数  | 继承自 `[default].parallel`         | 该模型的并行处理能力                      |
| `headers`      | 字典  | 继承自 `[default].headers`          | 该模型特定的 HTTP 请求头                 |
| `timeout`      | 浮点数 | 继承自 `[default].timeout`          | 该模型特定的流式请求（stream=true）超时时间        |
| `timeout_batch`| 浮点数 | 继承自 `[default].timeout_batch`    | 该模型特定的批量请求（stream=false）超时时间      |
| `body_options` | 字典  | 继承自 `[default].body_options`     | 该模型默认的请求选项，覆盖 default 中的所有选项    |
| `max_input_tokens` | 整数  | 继承自 `[default].max_input_tokens` | 该模型支持的最大输入Token数 |
| `bytes_per_token`  | 浮点数 | 继承自 `[default].bytes_per_token`  | 该模型单Token对应的UTF-8字节数估算值 |
| `enable`           | 布尔值 | true                            | 设为false则暂停该模型的调度使用   |


其中：
* "openai" 格式，headers 应有 `Authorization`
* "anthropic" 格式，headers 应有 `"X-Api-Key"`, `"anthropic-version" = "2023-06-01"`；body_options 应有 `max_tokens`


### 1.4 [model_tag] 部分
模型标签配置，用于模型路由和分组调度。按优先级排序，应尽量写全。

业务代码可以使用标签，无需关心底层具体模型，支持灵活切换和路由。模型路由支持同时传入多个标签，系统会自动匹配同时符合所有标签的可用模型，结合输入Token数限制、模型优先级自动选择最优模型。例如传入 `["code", "multimodal"]` 可自动选择支持多模态的代码模型，传入 `["low_cost", "fast"]` 可自动选择低成本且响应快的模型，适配各类交叉场景。

原有 `[model_alias]` 配置（`别名 = 模型名`）会自动合并到 `[model_tag]`，每个别名作为标签，对应单元素列表。

`标签名 = 字符串列表（支持一层嵌套）`

键为标签名，值为模型列表。列表顺序为推荐优先级（越靠前权重越高）。模型名必须存在于 `[api]` 部分。元素可为字符串或字符串列表，嵌套列表中的模型共享该位置的排名权重，随机排序同权重模型。可用于轮流使用多个提供商的相同（类似）模型。

**示例：**
```toml
[model_tag]
image = ['doubao-seed-2.0-pro']
cheap = ['doubao-seed-2.0-mini','qwen-flash']
fast = ['deepseek-v4-flash','qwen-flash']
low_cost = ["doubao-seed-2.0-mini", "qwen-flash", ["deepseek-v4-flash-a", "deepseek-v4-flash-b", "deepseek-v4-flash-c"]]
# 多个提供商轮流使用
"deepseek-v4-flash" = [["deepseek-v4-flash-a", "deepseek-v4-flash-b", "deepseek-v4-flash-c"]]
```

**权重计算：**
列表中模型的权重按位置计算，越靠前权重越高。嵌套列表中的所有模型共享其所在位置的权重。权重公式中 `rank_adjust_ratio` 参数控制权重衰减速率，默认0.25。

**tag:rank 语法：**

在调用 `resolve_model` 时，可以在标签名后附加 `:rank` 后缀来动态调整标签内模型的排序优先级。语法格式为 `tag_name:rank`，其中 `rank` 为整数。

- `rank=1`：不调整，保持原始顺序（等同于不使用 rank）
- `rank=2`：将第1项移到末尾，第2项变为最高优先
- `rank=N`：将前 N-1 项移到末尾，第 N 项变为最高优先
- 超出范围的 rank 会自动取模循环（例如4个模型传入 rank=6 等同于 rank=2）

移位在列表的外层元素级别进行，嵌套列表视为单个元素整体移位，嵌套列表内的模型共享该位置的权重。

**示例：**
```python
# 假设 "code" 标签配置为 ["glm-5.1", "qwen3.7-max", "deepseek-v4-pro", "kimi-k2.5"]
# 传入 ['code:2'] → qwen3.7-max 优先级最高
# 传入 ['code:3'] → deepseek-v4-pro 优先级最高
# 传入 ['code:5'] → 等同于 rank=1，glm-5.1 优先级最高
# 混合使用：['code:2', 'image:3'] 各标签独立调整
```

不使用 `:rank` 后缀的标签行为完全不变。`tag:rank` 语法仅适用于标签名，不适用于直接模型名。

### 1.5 配置示例
```toml
[server]
zmq_router_rpc = "tcp://*:5555"
zmq_pub_event = "tcp://*:5556"
zmq_auth_token = "YOUR_SECRET_TOKEN"  # 接口认证令牌（可选）
zmq_manage_token = "YOUR_MANAGE_TOKEN"  # 管理权限认证令牌（可选）

[default]
timeout = 10
timeout_batch = 300
max_retries = 3
parallel = 1
retry_duration = 0.5
retry_factor = 1.5
api_type = "openai"

[model_tag]
"low_cost" = ["doubao-seed-2.0-mini", "qwen-flash", "deepseek-v4-flash"]
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
| `别名` | 字符串/字符串列表 | 无   | 映射到实际的模型或标签名称 |

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
model = ["fast", "low_cost"]
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

1. **模型名称一致性**：`app_prompt.toml` 中的 `model` 可以是 `llm_provider.toml` 的 `[api]` 部分的模型名称，也可以是 `[model_alias]` 部分定义的别名。

2. **模板变量**：模板中使用的变量必须在调用时提供，否则会渲染失败。

3. **流式输出**：当 `stream=true` 时，输出会通过通道的 `write` 方法分片发送。

4. **工具调用**：工具配置必须包含完整的参数定义，否则可能无法正确解析。

5. **模型路由**：使用 `model_tag` 进行模型路由时，系统会根据标签权重和模型可用性自动选择最优模型。支持 `tag:rank` 语法动态调整标签内模型排序优先级（详见 1.4 节）。传入 `messages` 时，会自动估算Token数并过滤超出限制的模型。
