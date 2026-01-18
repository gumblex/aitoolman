# aitoolman 框架开发者文档

## 1. 框架概述

### 1.1 设计理念
aitoolman 是一个面向开发者的 LLM 应用框架，核心思想是 **"AI 作为工具人"** ——让 LLM 像实习生一样执行明确指令，而非自主决策。框架强调：
- **流程可控**：所有程序逻辑由用户代码主导，LLM 无自主权
- **数据流透明**：用户可自定义所有发往 LLM 的数据，清晰查看请求/响应内容
- **提示词模板化**：将提示词封装为可复用模板，避免"万能对话框"的混乱

### 1.2 核心数据流
```
用户代码 → LLMApplication → LLMModule → LLMClient → ProviderManager → HTTP API
    ↑            ↑              ↑           ↑            ↑
    └─ 结果返回 ──┴─ 模板渲染 ───┴─ 构建请求 ──┴─ 格式转换 ──┴─ 响应解析
```

**关键流转节点**：
1. **模板渲染**：Jinja2 模板 + 变量替换 → 最终提示词
2. **消息构建**：渲染后的提示词 → Message 对象列表
3. **请求发送**：Message 列表 → LLMRequest → HTTP API
4. **响应处理**：API 响应 → TextChannel 流 → 后处理 → 最终结果

### 1.3 核心组件
- **LLMApplication**：应用上下文，管理模块、模板、变量、通道和后处理器
- **LLMModule**：封装"输入→模板→LLM→输出"的完整流程
- **LLMClient**：抽象 LLM 提供商调用，支持本地和远程（ZeroMQ）模式
- **TextChannel**：异步消息通道，支持流式片段传输
- **FormatStrategy**：抽象不同 LLM 提供商的消息格式转换

## 2. 核心 API 参考

### 2.1 LLMApplication - 应用上下文

**作用**：管理 LLM 模块、模板、变量、通道、后处理器和全局工具的容器

```python
class LLMApplication:
    def __init__(
        self,
        client: LLMClient,                      # LLM 客户端实例
        config_dict: Optional[Dict[str, Any]] = None,  # 配置字典（从 TOML 加载）
        processors: Optional[Dict[str, Callable[[str], Any]]] = None,  # 自定义后处理器
        channels: Optional[Dict[str, TextChannel]] = None,  # 自定义通道
        context_id: Optional[str] = None      # 上下文 ID
    ) -> None
```

**关键属性**：
- `client: LLMClient` - LLM 客户端实例
- `channels: Dict[str, TextChannel]` - 通道字典（默认包含 stdin/stdout/reasoning）
- `vars: Dict[str, Any]` - 全局变量，可在所有模板中访问
- `modules: Dict[str, LLMModule]` - 已加载的模块实例
- `processors: Dict[str, Callable[[str], Any]]` - 后处理器字典
- `global_tools: Dict[str, Any]` - 全局工具集合，所有模块可引用

**关键方法**：
```python
# 动态访问模块（延迟加载）
module = app.module_name  # 自动从配置初始化模块

# 添加后处理器
app.add_processor("custom.parse_xml", parse_xml_function)

# 获取后处理器
processor = app.get_processor("builtin.parse_json")

# 渲染全局模板（支持模板交叉引用）
text = app.render_template("template_name", **variables)

# 添加自定义通道
app.add_channel("custom", TextChannel(read_fragments=True))

# 创建应用工厂（批量创建应用实例）
@classmethod
def factory(
    cls,
    client: LLMClient,
    config_dict: Optional[Dict[str, Any]] = None,
    processors: Optional[Dict[str, Callable[[str], Any]]] = None,
    channels: Optional[Dict[str, TextChannel]] = None,
) -> Callable[..., 'LLMApplication']:
    """创建可复用的应用工厂函数"""
    pass

# 使用工厂创建实例
app_factory = LLMApplication.factory(
    client=client,
    config_dict=app_config,
    processors={"custom.parser": custom_parser}
)
app = app_factory()

# 给LLMClient/LLMZmqServer发送自定义的审计事件
await app.audit_event(event_type, **kwargs)
```

### 2.2 LLMModule / DefaultLLMModule - LLM 模块

**作用**：封装 LLM 调用的逻辑单元，每个模块对应一个特定任务

```python
class DefaultLLMModule(LLMModule):
    def __init__(self, app: LLMApplication, config: ModuleConfig) -> None
```

**推荐调用方式**（依赖注入模式，支持批量处理）：
```python
async def process_task(app: aitoolman.LLMApplication, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """单个任务处理函数"""
    result: LLMModuleResult = await app.module_name(
        _media=MediaContent(...),  # 可选：多媒体内容
        **task_data                # 模板变量
    )
    result.raise_for_status()  # 自动处理异常
    return {
        "task_id": task_data["id"],
        "result": result.data if result.data else result.text
    }

# 批量处理示例
async with asyncio.TaskGroup() as tg:
    tasks = [tg.create_task(process_task(app_factory(), task)) for task in task_list]
    for task in tasks:
        result = await task
        print(f"任务 {result['task_id']} 结果: {result['result']}")
```

**ModuleConfig 数据类**：
```python
@dataclass
class ModuleConfig:
    name: str                               # 模块名称
    model: str                              # 使用的模型名称或别名
    templates: Dict[str, str]              # 模板字典（必须包含 user，可选 system）
    tools: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # 工具配置
    stream: bool = False                    # 是否流式输出
    output_channel: Optional[TextFragmentOutput] = None  # 输出通道
    reasoning_channel: Optional[TextFragmentOutput] = None  # 推理通道
    post_processor: Optional[str] = None    # 后处理器名称（对应app.processors中的键）
    save_context: bool = False              # 是否保存对话上下文
    options: Dict[str, Any] = field(default_factory=dict)  # 请求选项（temperature 等）
```

**工具配置说明**：
- 模块中 `tools` 字典的空值（`{}`）表示引用**全局工具**定义
- 若需覆盖全局工具配置，可直接在模块中定义完整工具参数

### 2.3 LLMClient - LLM 客户端

**作用**：抽象 LLM 提供商调用，支持本地和远程模式

```python
class LLMClient(abc.ABC):
    def __init__(self) -> None:
        ...

    async def initialize(self) -> None:
        """初始化客户端（自动在__aenter__中调用）"""
        pass

    async def close(self) -> None:
        """关闭客户端（自动在__aexit__中调用）"""
        pass
```

**关键方法**：
```python
# 发送请求（由模块自动调用，通常不直接使用）
request: LLMRequest = await client.request(
    model_name: str,                        # 模型名称或别名
    messages: List[Message],               # 消息列表
    tools: Dict[str, Dict[str, Any]] = None,  # 工具配置
    options: Optional[Dict[str, Any]] = None,  # 请求选项
    stream: bool = False,                   # 是否流式
    context_id: Optional[str] = None,      # 上下文 ID
    response_channel: Optional[TextChannel] = None,  # 响应通道
    reasoning_channel: Optional[TextChannel] = None   # 推理通道
)

# 取消请求
await client.cancel(request_id: str)

# 替代 with 的开始和结束方法
await client.initialize()
await client.close()
```

**实现类**：
- `LLMLocalClient`：本地客户端，直接调用 ProviderManager
- `LLMZmqClient`：ZMQ 客户端，连接远程微服务

**使用模式**：
```python
# 作为异步上下文管理器（推荐）
async with LLMLocalClient(api_config) as client:
    app = LLMApplication(client, app_config)
    result = await app.module_name(...)
```

### 2.4 Channel / TextChannel - 通道系统

**作用**：异步消息传递通道，支持完整消息和片段传输

```python
class TextChannel(Channel):
    def __init__(self, read_fragments: bool = False) -> None
```

**关键方法**：
```python
# 写入完整消息
await channel.write_message("完整消息内容")

# 写入消息片段（流式）
await channel.write_fragment("片段1", end=False)
await channel.write_fragment("片段2", end=False)
await channel.write_fragment("片段3", end=True)  # 标记结束

# 读取完整消息（非流式模式）
message: Optional[str] = await channel.read_message()

# 读取消息片段（流式模式）
fragment: Optional[str] = await channel.read_fragment()  # None 表示结束
```

**默认通道**：
- `stdin`：标准输入（非片段模式）
- `stdout`：标准输出（片段模式）
- `reasoning`：推理输出（片段模式）

### 2.5 XmlTagToChannelFilter - XML 标签分发

**作用**：解析 XML 标签并将内容分发到不同通道

```python
class XmlTagToChannelFilter(BaseXmlTagFilter):
    def __init__(
        self,
        default_channel: TextChannel,           # 默认通道（未匹配标签）
        channel_map: Dict[str, TextChannel]    # 标签到通道的映射
    ) -> None
```

**使用示例**：
```python
# 创建通道
response_channel = TextChannel(read_fragments=True)
reasoning_channel = TextChannel(read_fragments=True)

# 创建过滤器
filter = XmlTagToChannelFilter(
    default_channel=response_channel,
    channel_map={"reasoning": reasoning_channel}
)

# 处理 LLM 输出（自动分发）
await filter.write_fragment("<reasoning>思考过程...</reasoning>", end=False)
await filter.write_fragment("<response>最终答案</response>", end=True)
```

### 2.6 collect_text_channels - 多通道收集器

**作用**：同时监听多个 TextChannel，生成统一的事件流

```python
async def collect_text_channels(
    channels: Dict[str, TextChannel],           # 通道字典（名称→通道）
    read_fragments: bool = True,               # 是否以片段模式读取
    timeout: Optional[float] = None            # 超时时间（秒）
) -> AsyncGenerator[ChannelEvent, None]
```

**ChannelEvent 结构**：
```python
class ChannelEvent(NamedTuple):
    channel: str                               # 通道名称
    message: Any                               # 消息内容
    is_fragment: bool                          # 是否为片段
    is_end: bool                               # 是否为结束标记
```

**使用示例**：
```python
channels = {
    'response': app.channels['stdout'],
    'reasoning': app.channels['reasoning']
}

async for event in collect_text_channels(channels, read_fragments=True):
    if event.channel == 'reasoning':
        print(f"[思考] {event.message}", end="", flush=True)
    elif event.channel == 'response':
        print(f"[回复] {event.message}", end="", flush=True)
```

### 2.7 数据模型（model.py）

**Message - 消息对象**：
```python
@dataclass
class Message:
    role: Optional[str] = None                # 角色：system/user/assistant/tool
    content: Optional[str] = None            # 文本内容
    media_content: Optional[MediaContent] = None  # 多媒体内容
    reasoning_content: Optional[str] = None  # 推理内容
    tool_call_id: Optional[str] = None       # 工具调用 ID
    raw_value: Optional[Dict] = None         # 原始值（直接传递给提供商）
```

**LLMRequest - 请求对象**：
```python
@dataclass
class LLMRequest:
    client_id: str                            # 客户端 ID
    context_id: Optional[str]                 # 上下文 ID
    request_id: str                          # 请求 ID
    model_name: str                          # 模型名称或别名
    messages: List[Message]                  # 消息列表
    tools: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # 工具配置
    options: Dict[str, Any] = field(default_factory=dict)  # 请求选项
    stream: bool = False                     # 是否流式
    response_channel: Optional[TextChannel] = None  # 响应通道
    reasoning_channel: Optional[TextChannel] = None  # 推理通道
    is_cancelled: bool = False               # 是否已取消
    response: asyncio.Future[LLMResponse] = field(default_factory=asyncio.Future)
```

**LLMResponse - 响应对象**：
```python
@dataclass
class LLMResponse:
    client_id: str
    context_id: str
    request_id: str
    model_name: str
    stream: bool
    # 时间统计
    start_time: Optional[float] = None
    queue_time: Optional[float] = None
    queue_length: Optional[int] = None
    time_to_first_token: Optional[float] = None
    total_response_time: Optional[float] = None
    # 响应内容
    response_text: str = ""
    response_reasoning: str = ""
    response_tool_calls: List[ToolCall] = field(default_factory=list)
    # 完成信息
    finish_reason: Optional[str] = None
    error_text: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    # 完整请求/响应数据
    response_message: Optional[Dict[str, Any]] = None

    def raise_for_status(self) -> None:
        """根据finish_reason抛出对应的异常"""
        pass

    def call(self, fn_map: Dict[str, Callable]) -> Dict[str, Any]:
        """
        执行工具调用

        Args:
            fn_map: 工具名称到函数的映射

        Returns:
            工具调用ID到结果的字典

        Raises:
            LLMError: 若调用失败
            LLMResponseFormatError: 若工具未找到
        """
        pass
```

**LLMModuleResult - 模块结果**：
```python
@dataclass
class LLMModuleResult:
    response_text: str = ""                   # 原始响应文本
    response_reasoning: str = ""            # 原始推理内容
    text: str = ""                           # 处理后文本（后处理前的原始文本）
    tool_calls: Dict[str, ToolCall] = field(default_factory=dict)  # 工具调用
    status: FinishReason = FinishReason.stop  # 完成状态
    error_text: Optional[str] = None         # 错误信息
    request_params: Dict[str, Any] = field(default_factory=dict)  # 请求参数
    request_messages: List[Message] = field(default_factory=list)  # 请求消息
    response_message: Optional[Dict[str, Any]] = None  # 原始响应消息
    data: Any = None                         # 后处理结果

    def raise_for_status(self) -> None:
        """根据status抛出对应的异常"""
        pass

    def call(self, fn_map: Dict[str, Callable]) -> Dict[str, Any]:
        """
        执行工具调用

        Args:
            fn_map: 工具名称到函数的映射

        Returns:
            工具调用ID到结果的字典
        """
        pass
```

### 2.8 postprocess - 后处理工具

**内置工具函数**：
```python
import aitoolman.postprocess
# 提取XML根标签内容（不解析，仅提取原始内容，不含标签）
xml_content = aitoolman.postprocess.get_xml_tag_content(xml_string, "result")
# 含标签
xml_document = aitoolman.postprocess.get_xml_tag_content(xml_string, "result", with_tag=True)

# JSON 解析（自动修复格式错误）
data = aitoolman.postprocess.parse_json(json_string)

# XML 解析（提取指定根标签并解析为字典）
xml_dict = aitoolman.postprocess.parse_xml(xml_string, "root_tag")
```

**默认处理器字典**：
```python
DEFAULT_PROCESSORS = {
    "builtin.parse_json": parse_json,
}
```

**自定义处理器**：
```python
def custom_xml_processor(text: str) -> Tuple[str, str]:
    """自定义XML解析处理器"""
    content = aitoolman.postprocess.get_xml_tag_content(text, root="classification")
    main_category = re.search(r"<main_category>(.*?)</main_category>", content).group(1).strip()
    sub_category = re.search(r"<sub_category>(.*?)</sub_category>", content).group(1).strip()
    return (main_category, sub_category)

# 注册到应用
app.add_processor("custom.parse_classification", custom_xml_processor)

# 在模块配置中使用
# post_processor = "custom.parse_classification"
```

## 3. 提示词配置文件格式（app_prompt.toml）

### 3.1 文件结构

```toml
# 模块默认配置（所有模块继承）
[module_default]
model = "Doubao-Seed-1.6"    # 默认模型（可以是模型名称或别名，需在llm_config.toml中定义）
stream = false                # 默认非流式输出
output_channel = "stdout"     # 默认输出通道
reasoning_channel = "reasoning"  # 默认推理通道
save_context = false          # 默认不保存上下文
options = { temperature = 0.7, max_tokens = 4000 }  # 默认请求选项

# 全局模板（可用 LLMApplication.render_template 渲染）
[template]
"template1" = "模板内容 {{variable}}"
"template2" = "另一个模板"

# 全局工具定义（所有模块可引用）
[tools]

# 模块定义（可定义多个）
[module."模块名称"]
model = "Creative-Model"    # 使用模型别名，对应llm_config.toml中的model_alias映射
stream = true                 # 覆盖默认流式设置
template.user = "用户模板 {{input}}"
template.system = "系统指令"
tools = { }                   # 工具配置（见下文）
post_processor = "builtin.parse_json"  # 后处理器名称（对应app.processors中的键）
save_context = true           # 覆盖默认上下文设置
options = { temperature = 0.5 }  # 覆盖默认选项
```

### 3.2 模板语法

使用 **Jinja2** 语法，支持：
- **变量替换**：`{{variable_name}}`
- **控制结构**：`{% if condition %}...{% endif %}`
- **循环**：`{% for item in list %}...{% endfor %}`
- **过滤器**：`{{text|upper}}`
- **模板交叉引用**：`{% include %}` 指令引用其他模板（含其他跨文件引用指令），支持两种形式：
  - 引用全局模板：`{% include 'header' %}`（对应[template]块中的模板）
  - 引用模块模板：`{% include 'module/task_planner/user' %}`（格式：`module/模块名/模板名`）

**可用变量**：
- `app.vars` 中定义的全局变量
- 调用模块时传入的所有关键字参数

**示例**：
```toml
[module.summerize]
template.user = """
文章标题：{{title}}
文章内容：<article>{{content}}</article>

请根据文章内容：
1. 按文章的结构列出论点和重要观点
2. 列出文中的案例及说明的问题
3. 总结这篇文章

输出格式：
<response>
<point>论点1</point>
<point>论点2</point>
<case>案例1</case>
<case>案例2</case>
<summery>总结</summery>
</response>
"""
```

**模板引用示例**：
```
# 全局模板header
[template]
"header" = "【任务ID：{{task_id}}】处理开始"

[module.task_planner]
template.user = """
{% include 'header' %}
用户指令：{{user_input}}
请分析指令并调用合适的工具
"""
```

### 3.3 工具调用配置

工具配置采用 TOML 嵌套结构：

```toml
[module."模块名称"]
tools."工具名称".type = "function"  # 固定值
tools."工具名称".description = "工具描述"

tools."工具名称".param."参数名".type = "string"  # 参数类型：string/integer/boolean
tools."工具名称".param."参数名".description = "参数描述"
tools."工具名称".param."参数名".required = true   # 是否必需
```

**全局工具配置**：
在`[tools]`块中定义可被所有模块共享的工具，配置结构与模块内工具一致：
```toml
[tools."工具名称"]
type = "function"
description = "工具功能描述"
param."参数名1".type = "string/integer/boolean"
param."参数名1".description = "参数说明"
param."参数名1".required = true
param."参数名2".type = "string"
param."参数名2".description = "可选参数"
param."参数名2".required = false
```

**模块引用全局工具**：
在模块的`tools`配置中，使用空字典`{}`表示引用全局工具定义：
```toml
[module."task_planner"]
tools."add_task" = {}
```

**完整示例**：
```toml
[module.task_planner]
model = "Fast-Model"  # 使用快速推理模型别名
stream = true
save_context = true
template.user = """
你作为日程助手，分析用户指令：
- 如果有具体待办事项，调用 add_task 工具
- 如果没有，请用户详细说明

用户说：{{user_input}}
"""

tools.add_task.type = "function"
tools.add_task.description = "添加日程"

tools.add_task.param.datetime.type = "string"
tools.add_task.param.datetime.description = "日期时间，如 2025-12-31 12:34:56"
tools.add_task.param.datetime.required = false

tools.add_task.param.content.type = "string"
tools.add_task.param.content.description = "待办事项内容"
tools.add_task.param.content.required = true
```

## 4. 示例配置文件

### 4.1 完整 app_prompt.toml 示例

```toml
# 模块默认配置
[module_default]
model = "Fast-Model"         # 默认使用快速推理模型
stream = false
output_channel = "stdout"
reasoning_channel = "reasoning"
save_context = false
options = { temperature = 0.7, max_tokens = 4000 }

# 全局模板
[template]
"greeting" = "你好，{{name}}！欢迎使用 aitoolman 框架。"

# 原始文本处理模块
[module.raw]
template.user = "{{content}}"

# 文章总结模块
[module.summerize]
model = "Creative-Model"     # 使用创意模型
template.user = """
文章标题：{{title}}
文章内容：<article>{{content}}</article>

请根据文章内容：
1. 按文章的结构列出论点和重要观点
2. 列出文中的案例及说明的问题
3. 总结这篇文章

输出格式：
<response>
<point>论点1</point>
<point>论点2</point>
<case>案例1</case>
<case>案例2</case>
<summery>总结</summery>
</response>
"""
post_processor = "builtin.parse_json"

# 日程规划模块（支持工具调用）
[module.task_planner]
model = "Fast-Model"         # 使用快速推理模型
stream = true
save_context = true
template.user = """
你作为日程助手，分析用户指令：
- 如果有具体待办事项，调用 add_task 工具
- 如果没有，请用户详细说明

用户说：{{user_input}}
"""

tools.add_task.type = "function"
tools.add_task.description = "添加日程"

tools.add_task.param.datetime.type = "string"
tools.add_task.param.datetime.description = "日期时间，如 2025-12-31 12:34:56"
tools.add_task.param.datetime.required = false

tools.add_task.param.content.type = "string"
tools.add_task.param.content.description = "待办事项内容"
tools.add_task.param.content.required = true

# JSON 提取模块
[module.json_extractor]
model = "Precise-Model"      # 使用高精度模型
template.user = """
从以下文本中提取结构化信息：
{{text}}

输出格式：
<response>
{
  "字段1": "值1",
  "字段2": "值2"
}
</response>
"""
post_processor = "builtin.parse_json"

# 多轮对话模块
[module.chat]
model = "Doubao-Seed-1.6"
stream = true
save_context = true
template.user = "{{message}}"
```

## 5. 示例应用结构

### 5.1 项目目录结构

```
my_llm_app/
├── config/
│   ├── llm_config.toml          # API 配置（模型、密钥）
│   └── app_prompt.toml          # 提示词配置（模块、模板）
├── src/
│   ├── __init__.py
│   ├── main.py                  # 应用入口
│   ├── processors.py            # 自定义后处理器
│   └── workflows.py             # 工作流逻辑
└── requirements.txt
```

### 5.2 应用入口（main.py）

```python
import re
import asyncio
import aitoolman
from typing import Dict, Any, List, Tuple

# 自定义后处理器
def parse_classification(text: str) -> Tuple[str, str]:
    content = aitoolman.postprocess.get_xml_tag_content(text, root="classification")
    main_category = re.search(r"<main_category>(.*?)</main_category>", content).group(1).strip()
    sub_category = re.search(r"<sub_category>(.*?)</sub_category>", content).group(1).strip()
    return (main_category, sub_category)

# 单个任务处理函数
async def process_ticket(app: aitoolman.LLMApplication, ticket: Dict[str, Any]) -> Dict[str, Any]:
    """处理工单分类"""
    result = await app.classify_ticket(**ticket)
    result.raise_for_status()  # 自动处理异常
    return {
        "ticket_id": ticket["工单ID"],
        "main_category": result.data[0],
        "sub_category": result.data[1]
    }

async def main():
    # 1. 加载配置
    api_config = aitoolman.load_config("config/llm_config.toml")
    app_config = aitoolman.load_config("config/app_prompt.toml")

    # 2. 创建客户端并初始化
    async with aitoolman.LLMLocalClient(api_config) as client:
        # 3. 创建应用工厂
        app_factory = aitoolman.LLMApplication.factory(
            client=client,
            config_dict=app_config,
            processors={
                "custom.parse_classification": parse_classification
            }
        )

        # 4. 批量处理工单
        tickets: List[Dict[str, Any]] = [
            {"工单ID": "123", "内容描述": "工单内容1", "诉求地址": "地址1"},
            {"工单ID": "456", "内容描述": "工单内容2", "诉求地址": "地址2"}
        ]

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(process_ticket(app_factory(), ticket)) for ticket in tickets]
            for task in tasks:
                result = await task
                print(f"工单 {result['ticket_id']} 分类结果: {result['main_category']} / {result['sub_category']}")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(main())
```

### 5.3 自定义后处理器（processors.py）

```python
import re
from typing import Tuple, List
import aitoolman.postprocess

def parse_xml_classification(xml_str: str) -> Tuple[str, str]:
    """解析分类结果XML"""
    content = aitoolman.postprocess.get_xml_tag_content(xml_str, root="result")
    if not content:
        raise ValueError("无效的分类结果")

    main_category = re.search(r"<main>(.*?)</main>", content, re.DOTALL).group(1).strip()
    sub_category = re.search(r"<sub>(.*?)</sub>", content, re.DOTALL).group(1).strip()
    return (main_category, sub_category)

def extract_keywords(text: str) -> List[str]:
    """提取文本关键词"""
    # 实现自定义关键词提取逻辑
    return ["关键词1", "关键词2"]
```

### 5.4 复杂工作流示例（workflows.py）

```python
import aitoolman
from typing import Dict, Any, List

class DocumentPipeline:
    """文档处理流水线"""

    def __init__(self, app: aitoolman.LLMApplication):
        self.app = app

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """情感分析"""
        result = await self.app.sentiment_analysis(text=text)
        result.raise_for_status()
        return result.data

    async def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """实体提取"""
        result = await self.app.entity_extraction(text=text)
        result.raise_for_status()
        return result.data.get("entities", [])

    async def generate_summary(
        self,
        title: str,
        content: str,
        max_length: int = 300
    ) -> str:
        """生成摘要"""
        result = await self.app.summerize(
            title=title,
            content=content,
            max_length=max_length
        )
        result.raise_for_status()
        return result.text

    async def process_document(
        self,
        document: Dict[str, str],
        enable_sentiment: bool = True,
        enable_entities: bool = True
    ) -> Dict[str, Any]:
        """完整文档处理工作流"""
        results = {
            "metadata": document,
            "summary": await self.generate_summary(
                document["title"],
                document["content"]
            )
        }

        if enable_sentiment:
            results["sentiment"] = await self.analyze_sentiment(
                document["content"]
            )

        if enable_entities:
            results["entities"] = await self.extract_entities(
                document["content"]
            )

        return results

# 使用示例
async def run_pipeline():
    api_config = aitoolman.load_config("config/llm_config.toml")
    app_config = aitoolman.load_config("config/app_prompt.toml")

    async with aitoolman.LLMLocalClient(api_config) as client:
        app = aitoolman.LLMApplication(client, app_config)
        pipeline = DocumentPipeline(app)

        doc = {
            "title": "AI 技术突破",
            "content": "最新研究表明，AI 在图像识别领域取得重大进展..."
        }

        result = await pipeline.process_document(doc)
        print(f"摘要: {result['summary']}")
        print(f"情感: {result['sentiment']}")
        print(f"实体: {result['entities']}")
```

### 5.5 配置示例（llm_config.toml）

```toml
# 服务器配置（ZMQ 模式）
[server]
zmq_router_rpc = "tcp://*:5555"
zmq_pub_event = "tcp://*:5556"

# 默认配置
[default]
timeout = 600
max_retries = 3
parallel = 1
api_type = "openai"

# 模型别名映射
# 业务配置中使用别名，无需关心底层模型具体信息
[model_alias]
"Creative-Model" = "DeepSeek-v3.2-251201"
"Precise-Model" = "GPT-4o"
"Fast-Model" = "Doubao-Seed-1.6-flash-250828"
"Cheap-Model" = "Doubao-Mini-1.5"
"Code-Model" = "CodeLlama-70B-Instruct"

# API 配置
[api."Doubao-Seed-1.6"]
url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
type = "openai"
model = "ep-xxx"
headers = { Authorization = "Bearer YOUR_API_KEY" }

[api."GPT-4"]
url = "https://api.openai.com/v1/chat/completions"
type = "openai"
model = "gpt-4"
headers = { Authorization = "Bearer YOUR_OPENAI_KEY" }
```

## 6. 最佳实践

### 6.1 配置管理
- **环境分离**：为开发/测试/生产准备不同的 `llm_config.toml`
- **密钥管理**：可通过指定配置文件、数据库等方式载入 llm_config
- **版本控制**：将 `app_prompt.toml` 纳入 Git，记录提示词迭代历史
- **模板复用**：使用全局模板和交叉引用减少重复提示词，提升维护效率
- **工具复用**：将通用工具定义为全局工具，避免模块间重复配置
- **模型别名**：通过`model_alias`统一管理模型映射，方便最终用户切换模型，切换模型时只需修改llm_config.toml
- **分层配置**：在module_default中设置通用模型，特定模块按需使用别名覆盖

### 6.2 错误处理
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

### 6.3 性能优化
- **连接复用**：多个 `LLMApplication` 共享同一个 `LLMClient`
- **资源限制**：根据提供商配额合理设置 `parallel` 参数（通常 1-3）
- **流式输出**：使用自定义 Channel 输出工作流状态；长文本使用 `stream=true` 实时输出内容，提升用户体验
- **批量处理**：使用 `asyncio.TaskGroup` 并行处理多个任务，提升吞吐量
- **缓存策略**：对重复请求结果进行缓存，减少不必要的 LLM 调用
- **模型选型**：根据任务类型定义并选择合适的模型别名，再给别名配置合适的具体模型

## 7. 总结

aitoolman 框架通过清晰的架构设计和灵活的配置系统，让开发者能够：

1. **快速构建**：通过 TOML 配置定义 LLM 应用，无需编写复杂代码
2. **精细控制**：完全掌控数据流，实现透明可调试的 AI 应用
3. **轻松扩展**：支持自定义处理器、通道和格式策略，适配不同业务需求
4. **灵活部署**：支持本地调用和微服务架构，适应不同场景需求

通过理解 `LLMApplication`、`LLMModule`、`LLMClient` 和 `TextChannel` 的协作关系，开发者可以高效构建稳定、可维护的 LLM 应用。
