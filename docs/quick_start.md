# aitoolman 框架开发者文档

## 1. 框架概述

### 1.1 设计理念
aitoolman 是一个面向开发者的 LLM 应用框架，核心思想是 **AI 作为工具人** ——让 LLM 像企业中的基层员工一样，仅在开发者预设的规则和流程内执行明确指令，没有自主决策权。框架的角色分工清晰：
- 最终用户 = 甲方：提出比较模糊的需求
- 应用开发者 = 企业老板：定义所有规则、流程、提示词模板，掌握100%的决策权
- LLMWorkflow = 中层管理者：按照预设流程调度任务，根据预设或LLM的返回结果切换流程分支
- LLM模块 = 基层员工：仅完成分配的单一、明确的任务，输出严格符合预设的格式要求

框架强调：
- **用户完全控制**：所有提示词、数据流、控制流均由用户代码主导，LLM 仅作为执行工具，无任何隐藏逻辑、无意外行为
- **流程透明可调试**：所有发往LLM和从LLM返回的数据均可自定义、可审计，便于排查问题和优化提示词
- **供应商无关**：通过抽象层统一适配多种LLM提供商，轻松切换模型且充分利用各提供商的特色功能
- **模块化设计**：组件职责单一，易于测试、替换和复用
- **生产级特性**：内置资源管理、错误处理、微服务部署、监控审计能力，可直接用于生产环境

无论是简单的单次查询，还是复杂的多步骤业务流程，aitoolman 都能提供稳定、可靠、可维护的解决方案。框架鼓励开发者深入理解业务逻辑，精心设计提示词，将 AI 能力无缝集成到现有系统中。

### 1.2 与传统 Agent 框架的区别
| 维度 | aitoolman | 传统 Agent 框架 |
|------|-----------|----------------|
| 定位 | LLM是工具人，仅执行预设指令 | LLM是智能体，可自主决策 |
| 控制权 | 用户完全控制流程 | 框架隐含控制流 |
| 提示词 | 开发者编写所有提示词，完全自定义 | 自带大量默认提示词，适配非英语场景成本高 |
| 多模型适配 | 原生支持多厂商、多模型，切换成本低 | 多为单一平台优化，适配成本高 |
| 功能边界 | 专注LLM功能编排，无冗余依赖 | 内置向量索引、RAG等大量功能，依赖库臃肿 |
| 适用场景 | 企业级可控流程编排、批量任务处理 | 开放式自主智能体、探索性应用 |

### 1.3 使用场景
1. 专业应用：输入输出范围明确
   - **文本处理**：总结、翻译、数据标注、提取结构化信息
   - **报告生成**：基于现有结构化数据，生成较为标准的文字报告
2. 助手类应用：输入不确定的用户需求
   - **多轮对话**：通过工作流编排处理复杂用户请求
   - **智能规划**：分解复杂任务为可执行步骤
   - **动态决策**：根据上下文和工具调用结果调整流程

在本框架中，提示词模板是核心，对话上下文是辅助。鼓励通过精心设计的模板参数编排，代替对话上下文的堆砌，从而避免信息遗忘和误导。

### 1.4 架构总览
框架采用分层架构：
1. 用户应用层：业务逻辑实现
2. 应用层 (LLMApplication / LLMWorkflow)：模板管理、流程编排、结果处理
3. 传输层 (LLMClient / Channel)：请求发送、流式响应传输、微服务通信
4. 数据接口层 (ProviderManager)：多厂商适配、请求调度、限流重试
5. LLM 提供商 API (OpenAI / Anthropic 等)：底层LLM服务

## 2. 数据模型类

### 消息结构
`Message` 类表示发送给 LLM 的消息，支持文本、多媒体内容、工具调用响应等。

```python
class Message(typing.NamedTuple):
    """发送给 LLM 的消息"""
    role: Optional[str] = None                # 角色：system/user/assistant/tool
    content: Optional[str] = None             # 文本内容
    media_content: Optional[MediaContent] = None  # 多媒体内容
    reasoning_content: Optional[str] = None   # assistant 的推理内容
    tool_call_id: Optional[str] = None        # 工具调用 ID（用于 tool 角色）
    raw_value: Optional[Dict] = None          # 提供商原始消息（忽略以上所有字段）
```

`MediaContent` 用于封装图片、视频等多媒体内容，支持本地文件、二进制数据、远程URL等多种来源，统一多模态输入的格式。

```python
class MediaContent(typing.NamedTuple):
    """多媒体内容（图像/视频等）"""
    media_type: str                    # 媒体类型，如 "image"、"video"
    # 按以下优先顺序
    # 1. raw_value
    raw_value: Optional[Dict] = None   # 提供商的原始值（优先）
    # 2. data+mime_type
    data: Optional[bytes] = None       # 原始二进制数据
    mime_type: Optional[str] = None    # MIME 类型，如 "image/jpeg"
    # 3. filename
    filename: Optional[str] = None     # 文件名（自动读取）
    # 4. url
    url: Optional[str] = None          # 远程 URL
    options: Optional[Dict] = None     # 提供商特定选项
```

`ToolCall` 类表示 LLM 返回的工具调用请求。

```python
class ToolCall(typing.NamedTuple):
    """LLM 返回的工具调用请求"""
    name: str                     # 工具函数名
    arguments_text: str           # 参数字符串（原始 JSON）
    arguments: Optional[Dict[str, Any]]  # 解析后的参数字典
    id: Optional[str] = None      # 工具调用 ID
    type: str = 'function'        # 类型，默认 'function'
```


### 应用层请求/响应
`LLMDirectRequest`: 直接请求参数，无需配置模板模块，适合经典工具调用、多轮对话、动态生成请求的场景。

```python
class LLMDirectRequest(typing.NamedTuple):
    """应用层直接请求参数"""
    model_name: str
    messages: List[Message]
    tools: Optional[Dict[str, Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    stream: bool = False
    output_channel: Union[str, TextFragmentChannel, None] = None
    reasoning_channel: Union[str, TextFragmentChannel, None] = None
```

`LLMModuleRequest`: 基于配置的模板模块发送请求，自动渲染提示词、加载预设的工具和模型配置，可覆盖默认配置。

```python
class LLMModuleRequest(typing.NamedTuple):
    """应用层模板请求参数（模块配置）"""
    module_name: str                    # 模块名称
    template_params: Dict[str, Any]     # 模板参数
    model_name: Optional[str] = None    # 覆盖模块默认模型
    context_messages: List[Message] = []  # 上下文消息
    media_content: Optional[MediaContent] = None  # 多媒体内容

    # 覆盖原始配置
    tools: Optional[Dict[str, Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = None
    output_channel: Union[str, TextFragmentChannel, None] = None
    reasoning_channel: Union[str, TextFragmentChannel, None] = None
```

`LLMModuleResult`: 统一封装LLM返回的所有结果，包含原始响应、处理后的文本、工具调用、状态信息等。

```python
@dataclass
class LLMModuleResult:
    """应用层（模板）请求响应"""
    module_name: str                    # 模块名称
    request: LLMDirectRequest = None    # 原始请求参数
    response_text: str = ""             # 原始响应文本
    response_reasoning: str = ""        # 原始推理文本
    text: str = ""                      # 处理后的文本
    tool_calls: List[ToolCall] = field(default_factory=list)  # 工具调用
    status: FinishReason = FinishReason.stop  # 完成状态
    error_text: Optional[str] = None    # 错误信息
    request_params: Dict[str, Any] = field(default_factory=dict)  # 原始模板参数
    response_message: Optional[Message] = None  # 原始响应消息
    data: Any = None                    # 后处理结果

    def raise_for_status(self):
        """按照 status 状态报错"""

    async def run_tool_calls(self, fn_map: Dict[str, Callable]) -> List[Message]:
        """运行工具调用，并返回 Message 上下文列表"""
```


### 数据接口层请求/响应
用于与LLM提供商交互，上层应用无需关注。

`LLMProviderRequest`：发送给模型提供商的请求，包含完整的请求数据和通道配置。

```python
@dataclass
class LLMProviderRequest:
    """发送给模型提供商的请求"""
    client_id: str                    # 客户端标识
    context_id: Optional[str]         # 上下文标识（用于关联对话）
    request_id: str                   # 请求唯一 ID
    model_name: str                   # 模型名称
    messages: List[Message]           # 消息列表
    tools: Dict[str, Dict[str, Any]]  # 工具定义
    options: Dict[str, Any]           # 提供商特定选项
    stream: bool = False              # 是否流式响应
    output_channel: Optional[TextFragmentChannel]  # 输出通道
    reasoning_channel: Optional[TextFragmentChannel]  # 推理通道
    is_cancelled: bool = False        # 是否已取消
    response: asyncio.Future[LLMProviderResponse]  # 响应 Future
```

`LLMProviderResponse`：模型提供商的响应，包含完整的时间统计和内容信息。

```python
@dataclass
class LLMProviderResponse:
    """模型提供商的响应"""
    client_id: str
    context_id: str
    request_id: str
    model_name: str
    stream: bool

    # 时间统计
    start_time: Optional[float] = None      # 请求开始时间
    queue_time: Optional[float] = None      # 排队时间
    queue_length: Optional[int] = None      # 排队时队列长度
    time_to_first_token: Optional[float] = None  # 首 token 时间
    total_response_time: Optional[float] = None  # 总响应时间

    # 响应内容
    response_text: str = ""                 # 完整响应文本
    response_reasoning: str = ""            # 完整推理文本
    response_tool_calls: List[ToolCall] = field(default_factory=list)  # 工具调用

    # 完成信息
    finish_reason: Optional[str] = None     # 完成原因
    error_text: Optional[str] = None        # 错误信息
    prompt_tokens: Optional[int] = None     # 输入 token 数
    completion_tokens: Optional[int] = None # 输出 token 数

    # 原始数据
    response_message: Optional[Dict[str, Any]] = None  # 原始响应消息
```

### 状态与错误类型
`FinishReason` 枚举定义了所有可能的完成原因，用于判断请求结果状态。

```python
class FinishReason(enum.Enum):
    # 提供商原因
    stop = "stop"                     # 正常结束
    length = "length"                 # 长度限制
    content_filter = "content_filter" # 内容审核
    tool_calls = "tool_calls"         # 调用了工具

    # 本地原因
    error = "error"                   # 通用错误
    error_request = "error: request"  # 请求错误
    error_format = "error: format"    # 返回格式错误
    error_app = "error: application"  # 应用错误
    cancelled = "cancelled"           # 被取消

    unknown = "unknown"               # 未知原因
```

LLMModuleResult 或 FinishReason 中有 `raise_for_status()` 方法自动将完成原因转换为对应的异常类型。

```python
class LLMError(RuntimeError): ...
class LLMLengthLimitError(LLMError): ...      # 响应长度限制
class LLMContentFilterError(LLMError): ...    # 内容被审核过滤
class LLMApiRequestError(LLMError): ...       # API 请求错误
class LLMResponseFormatError(LLMError): ...   # 响应格式错误
class LLMApplicationError(LLMError): ...      # 应用程序代码错误
class LLMCancelledError(LLMError): ...        # 请求被取消
class LLMUnknownError(LLMError): ...          # 未知完成原因
class GenericError(LLMError): ...             # 通用错误
```

## 3. 应用层

### 3.1 LLMApplication 类
LLMApplication 是框架的核心入口类，负责管理配置、模板渲染、LLM调用、通道和后处理器，是所有LLM应用的基础载体。

#### 3.1.1 核心功能
LLMApplication 是框架的主要入口，负责：
- 加载和管理配置文件中的模块
- 渲染提示词模板
- 调用 LLM 并处理响应
- 管理上下文变量和通道

#### 3.1.2 初始化
创建LLM应用实例，绑定客户端、加载配置、注册后处理器和通道，每个应用实例对应一个独立的上下文。

```python
class LLMApplication:
    def __init__(
        self,
        client: LLMClient,                     # LLM 客户端
        config_dict: Optional[Dict[str, Any]] = None,  # 配置文件字典
        processors: Optional[Dict[str, Callable[[str], Any]]] = None,  # 后处理器
        channels: Optional[Dict[str, TextFragmentChannel]] = None,  # 自定义通道
        context_id: Optional[str] = None       # 上下文 ID（用于追踪客户端、调试审计）
    ): ...
```

#### 3.1.3 主要接口
通过模块名快速获取可调用的模块对象，直接传入模板参数即可完成LLM调用。
```python
# 通过下标访问模块（返回可调用对象）
result: LLMModuleResult = await app['module_name'](template_param1='value1', ...)
```

通用请求入口，支持传入模块请求或直接请求，适合需要动态构造请求的场景。
```python
# 直接调用 LLM（绕过模块配置）
async def call(
    self,
    request: Union[LLMModuleRequest, LLMDirectRequest]
) -> LLMModuleResult: ...
```

渲染指定名称的模板，用于自定义生成消息内容。
```python
# 渲染模板
def render_template(self, template_name: str, **kwargs) -> str: ...
```

注册自定义后处理器，用于解析LLM返回的特定格式内容，比如提取JSON、XML、代码块等。
```python
# 添加自定义后处理器
def add_processor(self, name: str, processor: Callable): ...
```

注册自定义通道，用于接收流式响应、推理内容等，实现实时输出到前端、文件等自定义场景。
```python
# 添加自定义通道
def add_channel(self, name: str, channel: TextFragmentChannel): ...
```

发送自定义审计事件，用于记录业务层面的操作，便于后续排查问题和统计业务数据。LLMClient 后端接收，统一处理。
```python
# 触发审计事件
async def audit_event(self, event_type: str, **kwargs): ...
```

创建应用工厂，用于批量处理任务时生成多个独立的应用实例，避免上下文互相干扰，支持并发处理。
```python
# 对批量任务使用 LLMApplication.factory 创建多个实例并行处理
@classmethod
def factory(
        cls,
        client: _client.LLMClient,
        config_dict: Optional[Dict[str, Any]] = None,
        processors: Optional[Dict[str, Callable[[str], Any]]] = None,
        channels: Optional[Dict[str, _channel.TextFragmentChannel]] = None,
) -> Callable[..., 'LLMApplication']: ...
```

#### 3.1.4 使用示例
```python
# 方式1：通过模块调用
result = await app['translator'](text="Hello", target_lang="zh")
print(result.data)  # 后处理后的翻译结果

# 方式2：直接调用
direct_request = LLMDirectRequest(
    model_name="gpt-4",
    messages=[Message(role="user", content="Hello")],
    stream=True
)
result = await app.call(direct_request)
```

### 3.2 LLMWorkflow 类

#### 3.2.1 核心概念
LLMWorkflow 扩展自 LLMApplication，支持动态工作流执行，提供两种构建模式：

1. **串行模式**：通过 `next_task` 属性连接任务，使用 `run()` 方法执行
2. **并行模式**：通过 `add_task()` 构建 DAG（有向无环图），使用 `wait_tasks()` 方法执行

两种模式可混合使用。

#### 3.2.2 任务定义
继承 `LLMTask` 类，重写 `pre_process()` 和 `post_process()` 方法：
- `pre_process()`: 准备请求数据，可动态修改输入
- `post_process()`: 处理响应结果，可生成下一个任务

工具调用处理：
- `on_tool_call_goto()`: 将工具调用转为下一步任务
- `run_tool_calls()`: 执行工具调用并继续对话

```python
# 任务状态枚举
class LLMTaskStatus(enum.Enum):
    INIT = 0
    WAITING = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4
    DEPENDENCY_FAILED = 5


class LLMTask:
    """LLM 任务基类"""
    module_name: ClassVar[str] = ''  # 默认使用的模块名

    # 主要方法（用户可重写）
    async def pre_process(self) -> Union[LLMModuleRequest, LLMDirectRequest, None]:
        """
        前处理钩子：在调用LLM模块之前执行

        默认实现：
        - input_data 为 LLMModuleRequest/LLMDirectRequest：直接调用
        - input_data 为 dict：作为模板参数
        - 其他：报错

        用户可以重写此方法以实现：
        - 动态修改输入数据
        - 添加上下文消息
        - 添加多媒体内容
        """

    async def post_process(self):
        """
        后处理钩子：在LLM模块返回结果后执行

        默认实现：将 module_result.data 赋值给 output_data
        用户可以重写此方法以实现：
        - 解析和验证输出
        - 根据结果动态生成下一个任务
        - 处理工具调用
        - 实现分支逻辑
        """

    # 工具调用辅助方法
    # 1. 工具调用作为意图识别
    def on_tool_call_goto(self, **kwargs: Callable[[], 'LLMTask']):
        """
        用于 post_process，将工具调用转为下一步的 LLMTask
        * 非工具调用，直接返回
        * 对第一个调用，设置 next_task 为相应 LLMTask，结束当前任务
        * 无匹配的调用，报错
        """

    # 2. 经典“工具调用”模式，将调用结果加入上下文
    async def run_tool_calls(self, **kwargs: Callable):
        """
        用于 post_process，工具调用作为函数调用，生成下一步的 LLMTask，结束当前任务
        """


class LLMTaskCompleted(Exception):
    """
    提前结束LLMTask，用于 LLMTask.post_process
    """
```

#### 3.2.3 工作流接口
```python
class LLMWorkflow(LLMApplication):
    def add_task(self, current_task: Optional[LLMTask], dependent_task: LLMTask):
        """
        添加后台任务（不立即执行）
        dependent_task 为 current_task 之前要运行的任务
        current_task 为 None 时，直接将 dependent_task 加入任务列表
        """

    # 等待指定任务完成
    async def wait_tasks(self, *tasks: LLMTask, timeout: Optional[float] = None): ...

    # 运行串行工作流
    async def run(self, start_task: LLMTask) -> LLMTask: ...
```

#### 3.2.4 使用示例
```python
# 定义任务类
class TranslationTask(LLMTask):
    module_name = 'translator'

    async def post_process(self):
        # 根据工具调用决定下一步
        self.on_tool_call_goto(
            refine=self.refine_task,
            finalize=self.finalize_task
        )

    def refine_task(self):
        return RefinementTask()

    def finalize_task(self):
        return FinalizationTask()

# 运行工作流
workflow = LLMWorkflow(client, config)
start_task = TranslationTask(input_data={"text": "Hello"})
final_task = await workflow.run(start_task)
```

## 4. 传输层

客户端与通道

### 4.1 Channel 通道系统

通道系统用于异步传输流式响应、推理内容等，用于实现实时输出。

#### 4.1.1 基础通道
Channel是通用的异步消息通道，用于不同组件之间的异步通信。

TextFragmentChannel专门用于传输文本片段的通道，支持流式接收LLM的输出片段。

```python
class Channel(Generic[T]):
    """通用通道基类"""
    async def read(self) -> T: ...
    async def write(self, message: T): ...
    # 标记通道写入结束，EOF
    async def write_complete(self): ...

class TextFragmentChannel(Channel[Optional[str]]):
    """
    文本片段通道
    None 为一整条消息的结束符
    """
    # 读取所有片段，并合并为一条完整消息
    async def read_whole_message(self) -> str: ...
```

#### 4.1.2 通道收集器
主要用于编写应用时，同时监听多个通道，将不同来源的输出统一处理。

```python
class ChannelCollector(abc.ABC):
    """多通道收集器基类"""
    async def start_listening(self): ...
    def close(self): ...

    @abc.abstractmethod
    async def on_channel_start(self, channel_name: str):
        """通道开始本次输出"""

    @abc.abstractmethod
    async def on_channel_read(self, channel_name: str, message):
        """通道输出内容"""

    @abc.abstractmethod
    async def on_channel_end(self, channel_name: str):
        """通道结束本次输出"""

    @abc.abstractmethod
    async def on_channel_eof(self, channel_name: str):
        """通道结束所有输出"""

class DefaultTextChannelCollector(ChannelCollector):
    """默认文本通道收集器（打印到控制台）"""
```

#### 4.1.3 XML 标签过滤器
BaseXmlTagFilter从流式文本中自动识别XML标签，主要用于LLM输出单层XML标签，表示不同类型的文本，输出到不同通道。例如：输出当前状态、给用户的输出、给应用程序的处理结果。

```python
class BaseXmlTagFilter(abc.ABC):
    """XML 标签过滤基类"""
    async def write(self, message: Optional[str]) -> None: ...

class XmlTagToChannelFilter(BaseXmlTagFilter):
    """XML 标签分发到不同通道"""
    def __init__(self, default_channel: TextFragmentChannel,
                 channel_map: Dict[str, TextFragmentChannel]): ...
```

### 4.2 LLMClient 客户端抽象

LLMClient是LLM客户端的抽象基类，统一不同部署方式的调用接口。

#### 4.2.1 客户端接口
```python
class LLMClient(abc.ABC):
    """LLM 客户端抽象基类"""
    async def request(
        self,
        model_name: str,
        messages: List[Message],
        tools: Dict[str, Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        context_id: Optional[str] = None,
        output_channel: Optional[TextFragmentChannel] = None,
        reasoning_channel: Optional[TextFragmentChannel] = None
    ) -> LLMProviderRequest

    async def cancel(self, request_id: str): ...
    async def audit_event(self, context_id: str, event_type: str, **kwargs): ...
```

#### 4.2.2 本地客户端
LLMLocalClient是本地客户端，直接调用LLM提供商的API。

```python
class LLMLocalClient(LLMClient):
    """本地客户端（直接调用 ProviderManager）"""
    def __init__(self, config: Dict[str, Any]): ...
```

#### 4.2.3 ZeroMQ 客户端（微服务）
LLMZmqClient是ZeroMQ远程客户端，连接远程的LLM微服务。

```python
class LLMZmqClient(LLMClient):
    """ZeroMQ 客户端（连接远程服务）"""
    def __init__(self, router_endpoint: str, auth_token: Optional[str] = None): ...

    # 取消指定上下文的所有请求，适合用户退出会话时终止所有未完成的请求
    async def cancel_all(self, context_id: Optional[str] = None): ...
```

## 5. 数据接口层

LLM提供商管理

### 5.1 LLMFormatStrategy 格式策略
统一处理不同LLM提供商的请求/响应格式，实现供应商无关。

```python
class LLMFormatStrategy(abc.ABC):
    """LLM 请求/响应格式转换策略"""
    def serialize_tool_description(self, tools_configs: Dict[str, Dict[str, Any]]) -> List[Dict]: ...
    def parse_tool_calls(self, tool_calls: List[Dict]) -> List[ToolCall]: ...
    def serialize_message(self, message: Message) -> Dict[str, Any]: ...
    def make_request_body(self, request: LLMProviderRequest) -> Dict[str, Any]: ...
    def parse_batch_response(self, response: LLMProviderResponse, response_data: Dict[str, Any]): ...
    def parse_stream_event(self, response: LLMProviderResponse, event: httpx_sse.ServerSentEvent) -> StreamEvent: ...


class OpenAICompatibleFormat(LLMFormatStrategy):
    """OpenAI API 兼容格式"""


class AnthropicFormat(LLMFormatStrategy):
    """Anthropic Claude API 格式"""
```


### 5.2 LLMProviderManager 提供商管理器
```python
class LLMProviderManager:
    """管理多个 LLM 提供商，处理 API 调用、重试、资源限制"""
    def __init__(self, config: Dict[str, Any])

    def process_request(
        self,
        request: LLMProviderRequest,
        callback: Optional[Callable[[LLMProviderRequest], typing.Coroutine]] = None
    ) -> RequestTask: ...

    async def cancel_request(self, request_id: str): ...
    async def cancel_all_requests(self, client_id: str, context_id: Optional[str] = None): ...
```

## 6. 实用工具

### 6.1 配置文件

用于加载和管理TOML格式的配置文件，支持文件和字符串两种来源。

```python
# 载入 toml 格式的配置文件
aitoolman.load_config(filename)

# 载入 toml 格式的配置文件文本
aitoolman.load_config_str(s)
```

### 6.2 后处理器（aitoolman.postprocess）
提供常用的文本后处理函数，用于解析LLM输出。

```python
# JSON 解析（自动修复格式错误）
parse_json(s: str) -> Any

# XML 内容提取
get_xml_tag_content(s: str, root: str, with_tag: bool = False) -> Optional[str]

# XML 解析为字典
parse_xml(s: str, root: str) -> Optional[Dict]
```

### 6.3 资源管理器
```python
class ResourceManager:
    """管理模型并行处理资源，防止超额请求和资源竞争"""
    def __init__(self, capacities: Dict[str, int] = None): ...

    @asynccontextmanager
    async def acquire(self, key: str, task_name: Optional[str] = None): ...
    async def add_resource(self, key: str, capacity: int): ...
    async def remove_resource(self, key: str, force: bool = False): ...
    def get_queue_length(self, key: str) -> int: ...
    def get_stats(self, key: str) -> Dict: ...
```


## 7. 配置文件

详细的配置文件参数参考《[配置文件文档](./config.md)》。

### 7.1 提供商配置文件 (llm_config.toml)

```toml
# 服务器配置（ZeroMQ 微服务）
[server]
zmq_router_rpc = "tcp://*:5555" # ZeroMQ ROUTER 端点
zmq_pub_event = "tcp://*:5556"  # ZeroMQ PUB 端点（审计日志）
zmq_auth_token = "YOUR_SECRET_TOKEN"  # 接口认证令牌

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

### 7.2 提示词配置文件 (app_prompt.toml)

```toml
# 模块默认配置
[module_default]
model = "Fast-Model"         # 默认使用快速推理模型
stream = false
output_channel = "stdout"
reasoning_channel = "reasoning"
options = { max_tokens = 4000 }

# 全局模板（可被模块引用）
[template]
"greeting" = "你好，{{name}}！"

# 全局工具定义
[tools."工具名称"]
type = "function"
description = "工具功能描述"
param."参数名1".type = "string/integer/boolean"
param."参数名1".description = "参数说明"
param."参数名1".required = true
param."参数名2".type = "string"
param."参数名2".description = "可选参数"
param."参数名2".required = false

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
template.user = "{{message}}"
```

## 8. 示例应用

### 8.1 项目目录结构
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

### 8.2 简单问答：代码助手
实现代码修改工具：
- 加载配置文件和提示词模板
- 创建客户端和应用实例
- 调用代码编辑模块，处理用户输入
- 使用后处理器提取代码块

```python
import asyncio
import aitoolman

def extract_code(text: str) -> str:
    ...

async def main():
    # 加载配置
    api_config = aitoolman.load_config("config/llm_config.toml")
    prompt_config = aitoolman.load_config("config/app_prompt.toml")
    
    # 创建客户端和应用
    async with aitoolman.LLMLocalClient(api_config) as client:
        app = aitoolman.LLMApplication(client, prompt_config)
        app.add_processor("extract_code", extract_code)
        
        # 监听输出通道
        collector = aitoolman.DefaultTextChannelCollector({
            '思考过程': app.channels['reasoning'],
            '代码输出': app.channels['stdout']
        })
        output_task = asyncio.create_task(collector.start_listening())
        
        # 调用代码编辑器模块
        result = await app['code_editor'](
            code_content=open("app.py").read(),
            instruction="添加错误处理逻辑",
            references=[{"filename": "utils.py", "content": open("utils.py").read()}]
        )
        result.raise_for_status()
        
        # 保存结果
        with open("app_modified.py", "w") as f:
            f.write(result.data)
        
        output_task.close()
        await output_task

if __name__ == "__main__":
    asyncio.run(main())
```

### 8.3 LLM 作为函数：工单批量分类
使用 `LLMApplication.factory()` 创建应用工厂，批量处理工单：
- 定义分类模块和后处理器
- 创建异步任务组并发处理
- 收集并输出分类结果

```python
import asyncio
import aitoolman

def parse_classification(text: str) -> dict:
    ...

async def process_ticket(app_factory, ticket):
    """处理单个工单"""
    app = app_factory()
    result = await app['ticket_classifier'](
        ticket_content=ticket['content'],
        ticket_type=ticket['type']
    )
    result.raise_for_status()
    return {
        "ticket_id": ticket['id'],
        "category": result.data['main'],
        "sub_category": result.data['sub']
    }

async def main():
    # 加载配置
    api_config = aitoolman.load_config("config/llm_config.toml")
    prompt_config = aitoolman.load_config("config/app_prompt.toml")
    
    # 创建应用工厂
    async with aitoolman.LLMLocalClient(api_config) as client:
        app_factory = aitoolman.LLMApplication.factory(
            client=client,
            config_dict=prompt_config,
            processors={
                "classify": parse_classification
            }
        )
        
        # 模拟批量工单
        tickets = [
            {"id": "1", "type": "技术支持", "content": "系统登录失败"},
            {"id": "2", "type": "业务咨询", "content": "发票如何申请"},
            # 更多工单...
        ]
        
        # 并行处理
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(process_ticket(app_factory, t)) for t in tickets]
            
        # 收集结果
        results = [t.result() for t in tasks]
        for res in results:
            print(f"工单 {res['ticket_id']}: {res['category']} > {res['sub_category']}")

if __name__ == "__main__":
    asyncio.run(main())
```


### 8.4 静态工作流：数据分析流水线
应用场景：已知任务依赖关系，比如先获取数据，再分析，最后生成报告

使用 `LLMWorkflow` 构建静态 DAG：
- 定义多个分析任务
- 使用 `add_task()` 建立依赖关系
- 使用 `wait_tasks()` 等待所有任务完成
- 合并分析结果

```python
import aitoolman
from src.tasks import (
    DataFetchTask,
    DataAnalysisTask,
    ReportGenerationTask
)

async def main():
    # 初始化工作流
    api_config = aitoolman.load_config("config/llm_config.toml")
    prompt_config = aitoolman.load_config("config/app_prompt.toml")
    
    async with aitoolman.LLMLocalClient(api_config) as client:
        workflow = aitoolman.LLMWorkflow(client, prompt_config)
        
        # 创建任务
        fetch_task = DataFetchTask(input_data={"query": "2024年Q1销售数据"})
        analysis_task = DataAnalysisTask()
        report_task = ReportGenerationTask(input_data={"format": "markdown"})
        
        # 建立依赖关系：fetch → analysis → report
        workflow.add_task(analysis_task, fetch_task)
        workflow.add_task(report_task, analysis_task)
        
        # 等待所有任务完成
        await workflow.wait_tasks(report_task)
        
        # 获取结果
        print("分析报告生成完成：")
        print(report_task.output_data)

if __name__ == "__main__":
    asyncio.run(main())
```

### 8.5 动态工作流：文件夹批量分析
递归分析文件夹结构：
- 定义文件夹分析任务，输出子项列表
- 在 `pre_process()` 中根据分析内容动态添加子任务
- 使用 `add_task()` 和 `wait_tasks()` 管理递归依赖
- 处理文件内容分析、分类等子任务

### 8.6 串行工作流：多步骤决策
使用 `LLMTask.next_task` 构建串行流程：
- 定义任务链：分析 → 规划 → 执行 → 验证
- 每个任务根据输出决定下一步
- 使用 `workflow.run()` 执行整个流程
- 支持工具调用作为流程分支点

内容审核流水线：

```python
import aitoolman

class ContentSubmitTask(aitoolman.LLMTask):
    module_name = "content_validator"
    
    async def post_process(self):
        # 根据验证结果决定下一步
        if self.output_data['status'] == "valid":
            self.next_task = AIAuditTask(input_data={"content": self.input_data['content']})
        else:
            self.next_task = RejectionTask(input_data={"reason": self.output_data['reason']})

class AIAuditTask(aitoolman.LLMTask):
    module_name = "content_auditor"
    
    async def post_process(self):
        if self.output_data['risk_level'] <= 1:
            self.next_task = PublishTask(input_data={"content": self.input_data['content']})
        else:
            self.next_task = ManualReviewTask(input_data={"content": self.input_data['content'], "risk": self.output_data['risk_details']})

class ManualReviewTask(aitoolman.LLMTask):
    module_name = "review_coordinator"
    
    async def post_process(self):
        if self.output_data['approved']:
            self.next_task = PublishTask(input_data={"content": self.input_data['content']})
        else:
            self.next_task = RevisionTask(input_data={"content": self.input_data['content'], "feedback": self.output_data['feedback']})

# 其他任务类：PublishTask, RevisionTask, RejectionTask...

async def main():
    # 初始化工作流
    api_config = aitoolman.load_config("config/llm_config.toml")
    prompt_config = aitoolman.load_config("config/app_prompt.toml")
    
    async with aitoolman.LLMLocalClient(api_config) as client:
        workflow = aitoolman.LLMWorkflow(client, prompt_config)
        
        # 启动工作流
        start_task = ContentSubmitTask(input_data={
            "content": "待发布的文章内容...",
            "type": "article"
        })
        final_task = await workflow.run(start_task)
        
        print(f"流程完成，最终状态：{final_task.task_name}")
        print(f"结果：{final_task.output_data}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 9. 微服务

### 9.1 应用场景
aitoolman 微服务架构适用于以下场景：

1. **多项目协作**：多个项目共享同一套 LLM 基础设施
2. **资源集中管理**：统一管理 API 密钥、模型配额和访问控制
3. **高可用部署**：通过负载均衡和故障转移确保服务稳定性
4. **审计与监控**：集中记录所有 LLM 调用日志和性能指标
5. **安全隔离**：敏感 API 密钥不暴露给客户端应用

### 9.2 功能特性
- **ZeroMQ 通信**：高性能、低延迟的进程间通信
- **认证授权**：支持令牌认证，确保接口安全
- **请求队列**：智能调度，避免超额请求
- **实时监控**：通过 PUB 接口发布审计日志
- **客户端管理**：支持请求取消、批量取消等操作

### 9.3 使用方法

#### 9.3.1 启动服务端
```bash
# 使用默认配置文件
python3 -m aitoolman server -c llm_config.toml

# 启用详细日志
python3 -m aitoolman server -c llm_config.toml -v
```

服务端启动后，会绑定两个 ZeroMQ 端点：
- **ROUTER 端点**（默认：tcp://*:5555）：处理客户端请求和响应
- **PUB 端点**（默认：tcp://*:5556）：发布审计日志和监控数据

#### 9.3.2 客户端连接
```python
from aitoolman.zmqclient import LLMZmqClient
client = LLMZmqClient(
    router_endpoint="tcp://localhost:5555",
    auth_token="your-secret-token"  # 可选，与服务器配置一致
)
```

命令行客户端测试：
```bash
# 交互式测试
python3 -m aitoolman client \
  -r tcp://localhost:5555 \
  -m gpt-4 \
  -a your-auth-token

# 指定模型别名
python3 -m aitoolman client \
  -r tcp://localhost:5555 \
  -m Creative-Model \
  -a your-auth-token
```

#### 9.3.4 监控与审计
```bash
# 控制台实时监控
python3 -m aitoolman monitor --pub-endpoint tcp://localhost:5556

# 存储到 SQLite 数据库
python3 -m aitoolman monitor \
  --pub-endpoint tcp://localhost:5556 \
  --db-path llm_audit.db
```

监控器会显示以下信息：
- 请求时间统计（排队时间、首token时间、总响应时间）
- Token 使用情况
- 完成原因和错误信息
- 自定义审计事件

## 10. 最佳实践

### 10.1 提示词设计
用“**简答题**”的格式设计提示词。格式：

1. **阅读材料**，需要分析的大段文字，或提供参考资料，历史对话上下文
2. **问题背景**，阐述场景设计和主要目标，是简答题的题干
3. **任务说明**，清晰指出需要做什么，是简答题的一个小问
4. **具体要求**，列出需要遵循的规则、方法和要点，是题型设计、得分点和括号中的提示。
5. 可选的**输出范例**，直接用要输出的格式编写

提示词设计的原则：
- **专注单个任务**：一次只让AI处理一个任务或一组数据
- **首尾衔接**：提示词的结尾要和AI的答案开头能自然连贯；简短（一两行）的输入内容，可以放在提示词的最后
- **语言一致**：用主要工作语言编写提示词，目标语言的文本放在最后
- **简洁清晰**：用词简洁，尽量多写“要做什么”，少写“不要做什么”；避免让大模型看到无关内容
- **输出格式明确**：指定清晰的输出格式（如JSON、XML、Markdown），便于解析

### 10.2 模块设计

- **流程优先**：避免让大模型执行确定性的任务，或仅用提示词限制大模型行为
- **模块化设计**：将复杂提示拆分为多个模块、全局模板，每个模块职责单一
- **模板变量**：使用 `{{ variable }}` 和其他 Jinja2 模板语法
- **上下文控制**：尽量少用上下文消息，优先优化提示词质量
- **工具描述**：为工具提供清晰、具体的描述和参数说明；不提供无用工具

### 10.3 错误处理
```python
try:
    result = await app['module'](...)
    result.raise_for_status()  # 检查完成状态
    processed_data = result.data
except LLMLengthLimitError as e:
    # 处理长度限制：分段处理或换模型
    pass
except LLMApiRequestError as e:
    # API 错误：重试或直接报错
    pass
except LLMResponseFormatError as e:
    # 返回格式错误：重试或调整提示词
    pass
```

### 10.4 性能优化
- **并行度配置**：根据模型配额合理设置 `parallel` 参数
- **流式响应**：对长文本使用流式，提升用户体验
- **批量请求**：对于批量/后台任务，采用批量请求，或专用的批量接口，提升并行度
- **资源管理**：使用 `ResourceManager` 避免超额请求
- **缓存策略**：对重复查询实现结果缓存

### 10.5 调试技巧
- **通道监听**：使用 `ChannelCollector` 实时查看 LLM 输出
- **审计日志**：启用监控器记录所有请求和响应
- **逐步执行**：复杂工作流可先测试单个任务
- **提供商日志**：启用 `logging.DEBUG` 查看原始 API 交互
