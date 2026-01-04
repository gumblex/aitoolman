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
- **LLMApplication**：应用上下文，管理模块、模板、变量和通道
- **LLMModule**：封装"输入→模板→LLM→输出"的完整流程
- **LLMClient**：抽象 LLM 提供商调用，支持本地和远程（ZeroMQ）模式
- **TextChannel**：异步消息通道，支持流式片段传输
- **FormatStrategy**：抽象不同 LLM 提供商的消息格式转换

## 2. 核心 API 参考

### 2.1 LLMApplication - 应用上下文

**作用**：管理 LLM 模块、模板、变量和通道的容器

```python
class LLMApplication:
    def __init__(
        self,
        client: LLMClient,                      # LLM 客户端实例
        config_dict: Optional[Dict[str, Any]] = None,  # 配置字典（从 TOML 加载）
        channels: Optional[Dict[str, TextChannel]] = None,  # 自定义通道
        context_id: Optional[str] = None      # 上下文 ID
    ) -> None
```

**关键属性**：
- `client: LLMClient` - LLM 客户端实例
- `channels: Dict[str, TextChannel]` - 通道字典（默认包含 stdin/stdout/reasoning）
- `vars: Dict[str, Any]` - 全局变量，可在所有模板中访问
- `modules: Dict[str, LLMModule]` - 已加载的模块实例
- `templates: Dict[str, jinja2.Template]` - 全局模板

**关键方法**：
```python
# 动态访问模块（延迟加载）
module = app.module_name  # 自动从配置初始化模块

# 添加后处理器
app.add_processor("parse_json", lambda s: json.loads(s))

# 渲染全局模板
text = app.render_template("template_name", **variables)

# 添加自定义通道
app.add_channel("custom", TextChannel(read_fragments=True))
```

### 2.2 LLMModule / DefaultLLMModule - LLM 模块

**作用**：封装 LLM 调用的逻辑单元，每个模块对应一个特定任务

```python
class DefaultLLMModule(LLMModule):
    def __init__(self, app: LLMApplication, config: ModuleConfig) -> None
```

**调用方式**：
```python
# 异步调用模块
result: LLMModuleResult = await app.module_name(
    _media=MediaContent(...),  # 可选：多媒体内容
    **kwargs                      # 模板变量
)
```

**ModuleConfig 数据类**：
```python
@dataclass
class ModuleConfig:
    name: str                               # 模块名称
    model: str                              # 使用的模型名称
    templates: Dict[str, str]              # 模板字典（必须包含 user，可选 system）
    tools: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # 工具配置
    stream: bool = False                    # 是否流式输出
    output_channel: Optional[TextFragmentOutput] = None  # 输出通道
    reasoning_channel: Optional[TextFragmentOutput] = None  # 推理通道
    post_processor: Optional[Callable[[str], Any]] = None  # 后处理器函数
    save_context: bool = False              # 是否保存对话上下文
    options: Dict[str, Any] = field(default_factory=dict)  # 请求选项（temperature 等）
```

### 2.3 LLMClient - LLM 客户端

**作用**：抽象 LLM 提供商调用，支持本地和远程模式

```python
class LLMClient(abc.ABC):
    def __init__(self) -> None
```

**关键方法**：
```python
# 发送请求（由模块自动调用，通常不直接使用）
request: LLMRequest = await client.request(
    model_name: str,                        # 模型名称
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
```

**实现类**：
- `LLMLocalClient`：本地客户端，直接调用 ProviderManager
- `LLMZmqClient`：ZMQ 客户端，连接远程微服务

**使用模式**：
```python
# 作为异步上下文管理器
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
    model_name: str                          # 模型名称
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
```

**LLMModuleResult - 模块结果**：
```python
@dataclass
class LLMModuleResult:
    response_text: str = ""                   # 原始响应文本
    response_reasoning: str = ""            # 原始推理内容
    text: str = ""                           # 处理后文本（后处理器输出）
    tool_calls: Dict[str, ToolCall] = field(default_factory=dict)  # 工具调用
    status: FinishReason = FinishReason.stop  # 完成状态
    error_text: Optional[str] = None         # 错误信息
    data: Any = None                         # 后处理结果
```

### 2.8 postprocess.py - 后处理工具

**内置处理器**：
```python
# JSON 解析（自动修复格式）
data = postprocess.parse_json(json_string)

# XML 解析（提取指定根标签）
xml_dict = postprocess.parse_xml(xml_string, root="root_tag")

# 默认处理器字典
DEFAULT_PROCESSORS = {
    "builtin.parse_json": parse_json,
}
```

**自定义处理器**：
```python
def custom_processor(text: str) -> Any:
    # 实现自定义解析逻辑
    return processed_data

# 注册到应用
app.add_processor("custom", custom_processor)

# 在模块配置中使用
# post_processor = "custom"
```

## 3. 提示词配置文件格式（app_prompt.toml）

### 3.1 文件结构

```toml
# 模块默认配置（所有模块继承）
[module_default]
model = "Doubao-Seed-1.6"    # 默认模型（必须在 llm_config.toml 中定义）
stream = false                # 默认非流式输出
output_channel = "stdout"     # 默认输出通道
reasoning_channel = "reasoning"  # 默认推理通道
save_context = false          # 默认不保存上下文
options = { temperature = 0.7, max_tokens = 4000 }  # 默认请求选项

# 全局模板（可用 LLMApplication.render_template 渲染）
[template]
"template1" = "模板内容 {{variable}}"
"template2" = "另一个模板"

# 模块定义（可定义多个）
[module."模块名称"]
model = "Doubao-Seed-1.6"    # 覆盖默认模型
stream = true                 # 覆盖默认流式设置
templates = { user = "用户模板 {{input}}", system = "系统指令" }
tools = { }                   # 工具配置（见下文）
post_processor = "builtin.parse_json"  # 后处理器名称
save_context = true           # 覆盖默认上下文设置
options = { temperature = 0.5 }  # 覆盖默认选项
```

### 3.2 模板语法

使用 **Jinja2** 语法，支持：
- **变量替换**：`{{variable_name}}`
- **控制结构**：`{% if condition %}...{% endif %}`
- **循环**：`{% for item in list %}...{% endfor %}`
- **过滤器**：`{{text|upper}}`

**可用变量**：
- `app.vars` 中定义的全局变量
- 调用模块时传入的所有关键字参数

**示例**：
```toml
[module.summerize]
templates.user = """
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

**完整示例**：
```toml
[module.task_planner]
model = "Doubao-Seed-1.6"
stream = true
save_context = true
templates.user = """
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
model = "Doubao-Seed-1.6"
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
templates.user = "{{content}}"

# 文章总结模块
[module.summerize]
model = "Doubao-Seed-1.6"
templates.user = """
文章标题：{{title}}
文章内容：<article>{{content}}</article>

请根据文章内容：
1. 按文章的结构列出论点和重要观点
2. 列出文中的案例及说明的问题
3. 总结这篇文章

输出格式：
<response>
{
  "论点": [...],
  "案例": [...],
  "总结": "..."
}
</response>
"""
post_processor = "builtin.parse_json"

# 日程规划模块（支持工具调用）
[module.task_planner]
model = "Doubao-Seed-1.6"
stream = true
save_context = true
templates.user = """
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
model = "Doubao-Seed-1.6"
templates.user = """
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
templates.user = "{{message}}"
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
import asyncio
import aitoolman
from typing import Dict, Any

# 应用逻辑函数（依赖注入模式）
async def process_document(
    app: aitoolman.LLMApplication,
    title: str,
    content: str
) -> Dict[str, Any]:
    """
    文档处理工作流：总结 → 提取关键点 → 生成报告
    """
    # 1. 总结文档
    summary_result = await app.summerize(title=title, content=content)

    if summary_result.status != aitoolman.FinishReason.stop:
        raise ValueError(f"总结失败: {summary_result.error_text}")

    summary_data = summary_result.data  # 后处理器已解析 JSON

    # 2. 生成执行摘要
    report_result = await app.generate_report(
        summary=summary_data,
        format="markdown"
    )

    return {
        "summary": summary_data,
        "report": report_result.text
    }

async def main():
    # 1. 加载配置
    api_config = aitoolman.load_config("config/llm_config.toml")
    app_config = aitoolman.load_config("config/app_prompt.toml")

    # 2. 创建客户端（异步上下文管理器）
    async with aitoolman.LLMLocalClient(api_config) as client:
        # 3. 创建应用上下文
        app = aitoolman.LLMApplication(client, app_config)

        # 4. 设置全局变量（可选）
        app.vars["app_name"] = "My LLM App"
        app.vars["version"] = "1.0.0"

        # 文档处理
        document = {
            "title": "人工智能发展趋势",
            "content": "人工智能正在从感知智能向认知智能发展..."
        }
        result = await process_document(app, **document)
        print("\n=== 处理结果 ===")
        print(f"总结: {result['summary']}")
        print(f"报告: {result['report']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 5.3 自定义后处理器（processors.py）

```python
import json
import re
from typing import Dict, Any, List

def extract_markdown_sections(text: str) -> Dict[str, str]:
    """
    从 Markdown 文本中提取章节
    """
    sections = {}
    pattern = r'^##\s+(.*?)\n(.*?)(?=^##|\Z)'  # 匹配二级标题
    matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)

    for match in matches:
        title = match.group(1).strip()
        content = match.group(2).strip()
        sections[title] = content

    return sections

# 在应用启动时注册
# app.add_processor("extract_sections", extract_markdown_sections)
```

### 5.4 复杂工作流示例（workflows.py）

```python
import aitoolman
from typing import Dict, Any, Optional

class DocumentPipeline:
    """
    文档处理流水线（面向对象封装）
    """

    def __init__(self, app: aitoolman.LLMApplication):
        self.app = app

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """情感分析"""
        result = await self.app.sentiment_analysis(text=text)
        return result.data

    async def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """实体提取"""
        result = await self.app.entity_extraction(text=text)
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
        return result.text

    async def process_document(
        self,
        document: Dict[str, str],
        enable_sentiment: bool = True,
        enable_entities: bool = True
    ) -> Dict[str, Any]:
        """
        完整文档处理
        """
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
async def process_with_pipeline(app: aitoolman.LLMApplication):
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
- **密钥管理**：使用环境变量或密钥管理服务，可通过数据库等方式载入 llm_config
- **版本控制**：将 `app_prompt.toml` 纳入 Git，记录提示词迭代历史

### 6.2 错误处理
```python
try:
    result = await app.module_name(**params)

    if result.status == aitoolman.FinishReason.error:
        print(f"LLM 调用错误: {result.error_text}")
    elif result.status == aitoolman.FinishReason.tool_calls:
        # 处理工具调用
        handle_tool_calls(result.tool_calls)
    elif result.status == aitoolman.FinishReason.stop:
        # 正常完成
        print(result.text)
except Exception as e:
    print(f"框架错误: {e}")
    # 检查网络、配置格式等
```

### 6.3 性能优化
- **连接复用**：多个 `LLMApplication` 共享同一个 `LLMClient`
- **资源限制**：根据提供商合理设置 `parallel` 参数（通常 1-3）
- **流式输出**：使用自定义 Channel 输出工作流状态；长文本使用 `stream=true` 提升用户体验

## 7. 总结

aitoolman 框架通过清晰的架构设计和灵活的配置系统，让开发者能够：

1. **快速构建**：通过 TOML 配置定义 LLM 应用，无需编写复杂代码
2. **精细控制**：完全掌控数据流，实现透明可调试的 AI 应用
3. **轻松扩展**：支持自定义处理器、通道和格式策略
4. **灵活部署**：支持本地调用和微服务架构，适应不同场景需求

通过理解 `LLMApplication`、`LLMModule`、`LLMClient` 和 `TextChannel` 的协作关系，开发者可以高效构建稳定、可维护的 LLM 应用。
