# aitoolman LLM应用框架使用指南

## 1. 核心概念与架构

### 1.1 框架设计理念
aitoolman是一个面向开发者的LLM应用框架，核心思想是"AI作为工具人"——让LLM像实习生一样执行明确指令，而不是自主决策。框架强调**流程可控、数据流透明、提示词模板化**。

### 1.2 核心组件关系

```
用户代码 → LLMApplication → LLMModule → LLMClient → 服务层
```

#### 1.2.1 LLMClient（客户端）
**作用**：管理与LLM服务的连接，封装API调用逻辑
**对应类**：
- `LLMClient`：抽象基类，定义客户端接口
- `LLMLocalClient`：本地客户端，直接调用ProviderManager
- `LLMZmqClient`：ZMQ客户端，连接远程微服务

**使用场景**：
- 一个用户进程通常启动一个`LLMClient`实例
- 所有`LLMApplication`共享同一个客户端实例
- 支持异步上下文管理器（`async with`），或手动开启关闭（`initialize`/`close`）

#### 1.2.2 LLMApplication（应用上下文）
**作用**：管理一套完整的LLM应用逻辑，包括模块、模板、变量和通道
**对应类**：`LLMApplication`

**关键特性**：
- 管理多个`LLMModule`，每个模块对应一个特定的LLM任务
- 维护Jinja2模板引擎，支持变量渲染
- 管理`TextChannel`通道，实现数据流控制
- 支持后处理器（post-processor）扩展
- 支持`factory`生成工厂函数，用于并行处理（互不干扰）

#### 1.2.3 LLMModule（模块）
**作用**：封装"输入→模板→LLM→输出"的完整流程
**对应类**：
- `LLMModule`：抽象基类
- `DefaultLLMModule`：默认实现，从配置自动生成

**关键特性**：
- 通过TOML配置定义，无需编写代码
- 支持系统提示词、用户提示词模板
- 支持工具调用（function calling）
- 支持上下文保存（多轮对话）
- 支持流式/非流式输出

#### 1.2.4 TextChannel（文本通道）
**作用**：实现异步数据流传输，支持片段化读取
**对应类**：
- `Channel`：基础通道类
- `TextChannel`：文本通道，支持片段读写
- `XmlTagToChannelFilter`：XML标签分发过滤器

**关键特性**：
- 支持完整消息和消息片段两种模式
- 异步队列机制，实现生产-消费模式
- 支持多通道并行收集（`collect_text_channels`）
- 支持XML标签自动分发到不同通道

### 1.3 配置文件结构

#### 1.3.1 LLM服务配置（llm_config.toml）
```toml
# 服务器配置（ZMQ模式）
[server]
zmq_router_rpc = "tcp://*:5555"    # RPC端点
zmq_pub_event = "tcp://*:5556"     # 事件发布端点

# 默认配置
[default]
timeout = 600          # 超时时间（秒）
max_retries = 3        # 最大重试次数
parallel = 1           # 默认并行度
api_type = "openai"    # 默认API类型

# API配置（每个模型一个配置段）
[api."Doubao-Seed-1.6"]
url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
type = "openai"        # API类型（openai/anthropic）
model = "ep-aaa"       # 模型名称
headers = {Authorization = "Bearer xxx"}  # 认证头
parallel = 1          # 该模型的并行度
```

**作用**：
- 定义LLM服务连接参数
- 配置模型并行度和资源管理
- 支持多种API提供商（现支持 `openai`, `anthropic` 格式）

#### 1.3.2 应用提示词配置（app_prompt.toml）
```toml
# 模块默认配置
[module_default]
model = "Doubao-Seed-1.6"    # 默认模型
stream = false                # 非流式输出
output_channel = "stdout"     # 输出通道
save_context = false          # 不保存上下文

# 原始文本处理模块
[module.raw]
template.user = """{{content}}"""

# 文章总结模块
[module.summerize]
model = "Doubao-Seed-1.6"
template.user = """
文章标题：{{title}}
文章内容：<article>{{content}}</article>
请根据文章内容：
1. 按文章的结构列出论点和重要观点
2. 列出文中的案例及说明的问题
3. 总结这篇文章
"""
```

**作用**：
- 定义`LLMModule`的提示词模板
- 配置模块的输入输出行为
- 用 Jinja2 支持变量插值（`{{variable}}`）和其他复杂模板逻辑

## 2. 数据流详解

### 2.1 完整数据流路径

```
用户输入
    ↓
LLMApplication.render_template()   # 模板渲染
    ↓
Message列表构建                   # 包含角色、内容、媒体
    ↓
LLMClient.request()               # 创建LLMRequest
    ↓
LLMProviderManager.process_request() # 资源管理+格式转换
    ↓
HTTP API调用                      # 实际LLM服务调用
    ↓
响应解析                         # 流式/批量解析
    ↓
TextChannel写入                  # 通道分发
    ↓
后处理器处理                     # 可选：如JSON/XML解析
    ↓
LLMModuleResult返回              # 最终结果
```

### 2.2 关键数据流转节点

#### 2.2.1 模板渲染流程
```
# 用户调用
result = await app.summerize(title="AI发展", content="...")

# 内部流程
1. app.vars + kwargs → 合并变量
2. jinja_env.from_string(template) → 模板对象
3. template.render(**all_vars) → 渲染后的提示词
```

#### 2.2.2 通道数据流
```
# 流式输出场景
async for event in collect_text_channels(channels, read_fragments=True):
    if event.channel == 'reasoning':
        print(f"[思考] {event.message}")
    elif event.channel == 'response':
        print(f"[回复] {event.message}")

# XML标签分发场景

<response>这是最终回复</response>
# 自动分发到不同通道
```

#### 2.2.3 工具调用流程
```toml
# 配置工具
[module.task_planner]
tools.add_task.type = "function"
tools.add_task.description = "添加日程"
tools.add_task.param.datetime.type = "string"
tools.add_task.param.datetime.description = "日期时间"
```

```python
# 控制流模式
# 即用工具调用作为流程控制的一种手段
next_step = 'task_planner'
next_step_args = None
while True:
    if next_step == 'task_planner':
        result = await app.task_planner(user_request="明天开会")
        if result.tool_calls:
            if 'add_task' in result.tool_calls:
                next_step = 'add_task'
                next_step_args = result.tool_calls['add_task'].arguments
                continue
    elif next_step == 'add_task':
        ...
```

```python
# 工具调用模式
result = await app.task_planner(user_request="明天开会")
if result.tool_calls:
    for tool_name, tool_call in result.tool_calls.items():
        # 执行工具调用
        execute_tool(tool_name, tool_call.arguments)
```

## 3. 快速开始

### 3.1 最小可用配置

#### 3.1.1 服务配置（llm_config_simple.toml）
```toml
[default]
timeout = 30
max_retries = 2
parallel = 1
api_type = "openai"

[api."gpt-4"]
url = "https://api.openai.com/v1/chat/completions"
type = "openai"
model = "gpt-4"
headers = {Authorization = "Bearer YOUR_API_KEY"}
```

#### 3.1.2 应用配置（app_simple.toml）
```toml
[module_default]
model = "gpt-4"
stream = false
output_channel = "stdout"

[module.chat]
template.system = "你是一个有帮助的助手"
template.user = "{{message}}"
```

### 3.2 基础使用示例

```python
import asyncio
import aitoolman

async def simple_chat():
    # 1. 加载配置
    api_config = aitoolman.load_config('llm_config_simple.toml')
    app_config = aitoolman.load_config('app_simple.toml')

    # 2. 创建客户端（自动管理资源）
    async with aitoolman.LLMLocalClient(api_config) as client:
        # 3. 创建应用上下文
        app = aitoolman.LLMApplication(client, app_config)

        # 4. 调用模块
        while True:
            user_input = input("你: ")
            if user_input.lower() == 'exit':
                break

            # 调用chat模块，传入message参数
            result = await app.chat(message=user_input)
            print(f"助手: {result.text}")

if __name__ == "__main__":
    asyncio.run(simple_chat())
```

### 3.3 进阶示例：多模块工作流

```python
import asyncio
import aitoolman
from typing import Dict, Any

async def document_processor(app: aitoolman.LLMApplication, document: Dict[str, Any]):
    """文档处理工作流：总结→分析→生成报告"""

    # 1. 总结文档
    summary = await app.summerize(
        title=document["title"],
        content=document["content"]
    )

    # 2. 分析关键点
    analysis = await app.analyze(
        text=summary.text,
        aspects=["技术要点", "商业价值", "实施风险"]
    )

    # 3. 生成执行摘要
    report = await app.generate_report(
        summary=summary.text,
        analysis=analysis.data,  # 后处理器解析后的结构化数据
        format="markdown"
    )

    return {
        "summary": summary.text,
        "analysis": analysis.data,
        "report": report.text
    }

async def main():
    # 加载配置
    api_config = aitoolman.load_config('llm_config.toml')
    app_config = aitoolman.load_config('app_prompt.toml')

    async with aitoolman.LLMLocalClient(api_config) as client:
        app = aitoolman.LLMApplication(client, app_config)

        # 处理文档
        document = {
            "title": "人工智能发展趋势",
            "content": "人工智能正在从感知智能向认知智能发展..."
        }

        result = await document_processor(app, document)
        print("处理完成:", result)

if __name__ == "__main__":
    asyncio.run(main())
```

## 4. 配置文件详解

### 4.1 模块配置参数

| 参数                  | 类型      | 默认值      | 说明                            |
|---------------------|---------|----------|-------------------------------|
| `model`             | string  | 必填       | 使用的模型名称（对应api配置）              |
| `stream`            | boolean | false    | 是否使用流式输出                      |
| `output_channel`    | string  | "stdout" | 输出通道名称                        |
| `reasoning_channel` | string  | 可选       | 推理过程输出通道                      |
| `save_context`      | boolean | false    | 是否保存对话上下文                     |
| `post_processor`    | string  | 可选       | 后处理器名称（如"builtin.parse_json"） |
| `options`           | string  | 可选       | 请求选项                          |
| `template.system`   | string  | 可选       | 系统提示词模板                       |
| `template.user`     | string  | 必填       | 用户提示词模板                       |
| `tools.*`           | object  | 可选       | 工具定义（function calling）        |

### 4.2 工具定义示例

```toml
[module.weather_assistant]
model = "gpt-4"
tools.get_weather.type = "function"
tools.get_weather.description = "获取指定城市的天气信息"
tools.get_weather.param.city.type = "string"
tools.get_weather.param.city.description = "城市名称"
tools.get_weather.param.city.required = true
tools.get_weather.param.date.type = "string"
tools.get_weather.param.date.description = "日期（YYYY-MM-DD）"
```

### 4.3 变量系统

```toml
# 应用级变量（在LLMApplication.vars中）
[module_default]
model = "gpt-4"

# 模块模板中使用变量
[module.greeting]
template.user = """
用户信息：
- 姓名: {{user_name}}
- 等级: {{user_level}}
- 历史对话: {{history}}

请根据以上信息生成个性化问候。
"""

# 调用时传入变量
result = await app.greeting(
    user_name="张三",
    user_level="VIP",
    history="上次讨论了AI发展趋势"
)
```

## 5. 最佳实践

### 5.1 配置管理建议
1. **环境分离**：为开发、测试、生产环境准备不同的配置
2. **版本控制**：提示词模板纳入版本控制，记录迭代历史

### 5.2 性能优化
1. **连接复用**：多个`LLMApplication`共享同一个`LLMClient`
2. **资源限制**：根据提供商合理设置`parallel`参数（一般为1~3）

### 5.3 错误处理
```python
import logging
import aitoolman

try:
    result = await app.module_name(**params)
    result.raise_for_status()  # 自动处理异常

    if result.status == aitoolman.FinishReason.tool_calls:
        # 执行工具调用
        tool_results = result.call({
            "add_task": add_task_function,
            "query_task": query_task_function
        })
        print(f"工具调用结果: {tool_results}")
    else:
        print(f"处理结果: {result.data if result.data else result.text}")
except aitoolman.LLMLengthLimitError:
    print("响应超出长度限制，请优化提示词或调整max_tokens")
except aitoolman.LLMContentFilterError:
    print("内容触发过滤器，请检查输入")
except aitoolman.LLMResponseFormatError as e:
    print(f"响应格式错误: {e}")
except Exception as e:
    print(f"系统错误: {e}")
    # 记录详细日志
    logging.exception("处理任务失败")
```

## 总结

aitoolman框架通过清晰的架构设计和灵活的配置系统，让开发者能够：

1. **快速构建**：通过TOML配置定义LLM应用，无需编写复杂代码
2. **精细控制**：完全掌控数据流，实现透明可调试的AI应用
3. **轻松扩展**：支持自定义处理器、通道和格式策略
4. **灵活部署**：支持本地调用和微服务架构，适应不同场景需求

通过理解`LLMClient`、`LLMApplication`、`LLMModule`和`TextChannel`的协作关系，开发者可以高效构建稳定、可维护的LLM应用。
