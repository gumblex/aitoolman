# aitoolman 框架开发者文档

## 1. 框架概述

### 1.1 设计理念
aitoolman 是一个面向开发者的 LLM 应用框架，核心思想是 **AI 作为工具人** ——让 LLM 像企业中的基层员工一样，仅在开发者预设的规则和流程内执行明确指令，没有自主决策权。框架的角色分工清晰：
- 最终用户 = 甲方：提出比较模糊的需求
- 应用开发者 = 企业老板：定义所有规则、流程、提示词模板，掌握100%的决策权
- LLMWorkflow = 中层管理者：按照预设流程调度任务，根据预设或LLM的返回结果切换流程分支
- LLM模块 = 基层员工：仅完成分配的单一、明确的任务，输出严格符合预设的格式要求

框架强调：
- **用户完全控制**：所有提示词、数据流、控制流均由用户代码主导，LLM 仅作为执行工具，无隐藏业务逻辑、无意外行为
- **流程透明可调试**：所有发往LLM和从LLM返回的数据均可自定义、可审计，便于排查问题和优化提示词
- **供应商无关**：通过抽象层统一适配多种LLM提供商，轻松切换模型且充分利用各提供商的特色功能
- **模块化设计**：组件职责单一，易于测试、替换和复用
- **生产级特性**：内置资源管理、错误处理、微服务部署、监控审计、热更新能力，可直接用于生产环境

无论是简单的单次查询，还是复杂的多步骤业务流程，aitoolman 都能提供稳定、可靠、可维护的解决方案。框架鼓励开发者深入理解业务逻辑，精心设计提示词，将 AI 能力无缝集成到现有系统中。

### 1.2 与传统 Agent 框架的区别
| 维度    | aitoolman | 传统 Agent 框架           |
|-------|-----------|-----------------------|
| LLM角色 | **LLM是工具人，仅执行预设指令** | LLM是智能体，可自主决策         |
| 控制权   | 用户完全控制流程 | 框架隐含控制流               |
| 提示词   | 开发者编写所有提示词，完全自定义 | 自带大量默认提示词，适配自定义场景成本高  |
| 多模型适配 | 原生支持多厂商、多模型，切换成本低 | 多为单一平台优化，适配成本高        |
| 功能边界  | 专注LLM功能编排，无冗余依赖 | 内置向量索引、RAG等大量功能，依赖库臃肿 |
| 适用场景  | 企业级可控流程编排、批量任务处理 | 开放式自主智能体、探索性应用        |

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
4. 数据接口层 (ProviderManager)：多厂商适配、请求调度、限流重试、模型路由
5. LLM 提供商 API (OpenAI / Anthropic 等)：底层LLM服务

## 2. 数据模型类

### 消息结构
`Message` 类表示发送给 LLM 的消息，支持文本、多媒体内容、工具调用响应等。

```python
class Message(typing.NamedTuple):
    """发送给 LLM 的消息"""
    role: Optional[str] = None                # 角色：system/user/assistant/tool
    content: Optional[str] = None             # 文本内容
    media_content: Optional[List[MediaContent]] = None  # 多媒体内容
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
    output_channel: Optional[ChannelWriter] = None  # 输出通道，用于接收流式响应
    post_processor: Union[str, Callable[[str], Any], None] = None
```

`LLMModuleRequest`: 基于配置的模板模块发送请求，自动渲染提示词、加载预设的工具和模型配置，可覆盖默认配置。

```python
class LLMModuleRequest(typing.NamedTuple):
    """应用层模板请求参数（模块配置）"""
    module_name: str                    # 模块名称
    template_params: Dict[str, Any]     # 模板参数
    model_name: Union[str, List[str], None] = None  # 指定模型名/标签/标签列表，覆盖模块默认配置
    model_rank: int = 0   # 模型路由候选排名（从0开始，超出自动取模，详见模型路由功能）
    context_messages: List[Message] = []  # 上下文消息
    media_content: Optional[List[MediaContent]] = None  # 多媒体内容

    # 覆盖原始配置
    tools: Optional[Dict[str, Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = None
    output_channel: Optional[ChannelWriter] = None  # 输出通道，用于接收流式响应
    post_processor: Union[str, Callable[[str], Any], None] = None
```

`LLMModuleResult`: 统一封装LLM返回的所有结果，包含原始响应、处理后的文本、工具调用、状态信息等。

```python
@dataclass
class LLMModuleResult:
    """应用层（模板）请求响应"""
    model_name: str                     # 实际使用的模型名称
    module_name: Optional[str]          # 模块名称
    request: LLMDirectRequest           # 实际请求参数
    post_processor: Optional[str] = None  # 后处理器
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

    async def run_tool_calls(self, fn_map: Dict[str, Callable]) -> Optional[LLMDirectRequest]:
        """运行工具调用，并返回下一次请求参数 LLMDirectRequest"""
```

### 应用层请求状态
`LLMModuleRequestState` 表示应用层请求的中间状态，由 `LLMApplication.send_request` 返回，包含模块请求、实际发送的直接请求和底层提供商请求。通过它可以手动控制请求的等待与后处理。

```python
class LLMModuleRequestState(typing.NamedTuple):
    """应用层请求响应（中间结果）"""
    module_request: Optional[LLMModuleRequest]    # 原始模块请求（无则为 None）
    direct_request: LLMDirectRequest              # 实际发送的请求
    provider_request: LLMProviderRequest          # 底层提供商请求

    def has_response(self) -> bool:
        """检查响应是否已就绪（不阻塞）"""
        return self.provider_request.response.done()

    async def wait_response(self) -> LLMProviderResponse:
        """等待响应完成，返回 LLMProviderResponse"""
        return await self.provider_request.response
```


### 数据接口层请求/响应
用于与LLM提供商交互，上层应用无需关注。

`LLMProviderRequest`：发送给模型提供商的请求，包含完整的请求数据和通道配置。
> LLMClient 输出的通道内容固定使用两个 topic：`reasoning` 表示模型的思考推理内容，`response` 表示模
型的实际响应内容。

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
    output_channel: ChannelWriter = field(default_factory=NullChannel) # 输出通道
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
    response_message: Optional[Message] = None  # 原始响应消息
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
    error_request = "error: request"  # 请求错误
    error_format = "error: format"    # 返回格式错误
    error_app = "error: application"  # 应用错误
    cancelled = "cancelled"           # 被取消

    unknown = "unknown"               # 未知原因
```

LLMModuleResult 或 FinishReason 中有 `raise_for_status()` 方法自动将完成原因转换为对应的异常类型。

```python
class LLMError(RuntimeError): ...
class LLMRetriableError(LLMError): ...  # 可重试错误，建议更换同类模型重试，使用模型路由时可将model_rank+1切换下一个候选模型
class LLMProviderConfigError(LLMError): ... # 模型提供商配置错误
class LLMNoAvailableModelError(LLMProviderConfigError): ... # 无可用模型错误
class LLMPermissionDeniedError(LLMProviderConfigError): ... # 权限不足错误
class LLMLengthLimitError(LLMRetriableError): ...      # 响应长度限制
class LLMContentFilterError(LLMRetriableError): ...    # 内容被审核过滤
class LLMApiRequestError(LLMRetriableError): ...       # API 请求错误
class LLMResponseFormatError(LLMRetriableError): ...   # 响应格式错误
class LLMApplicationError(LLMError): ...      # 应用程序代码错误
class LLMCancelledError(LLMError): ...        # 请求被取消
class LLMUnknownError(LLMError): ...          # 未知完成原因
```

## 3. 应用层

### 3.1 LLMApplication 类
LLMApplication 是框架的核心入口类，负责管理配置、模板渲染、LLM调用和后处理器，是所有LLM应用的基础载体。

#### 3.1.1 核心功能
LLMApplication 是框架的主要入口，负责：
- 加载和管理配置文件中的模块
- 渲染提示词模板
- 调用 LLM 并处理响应
- 管理上下文变量

#### 3.1.2 初始化
创建LLM应用实例，绑定客户端、加载配置、注册后处理器，每个应用实例对应一个独立的上下文。

```python
class LLMApplication:
    def __init__(
        self,
        client: LLMClient,                     # LLM 客户端
        config_dict: Optional[Dict[str, Any]] = None,  # 配置文件字典
        processors: Optional[Dict[str, Callable[[str], Any]]] = None,  # 后处理器
        context_id: Optional[str] = None       # 上下文 ID（用于追踪客户端、调试审计）
    ): ...
```

#### 3.1.3 主要接口
通过模块名快速获取可调用的模块对象，直接传入模板参数即可完成LLM调用。如果需要实时流式输出，可通过`output_channel`参数传入自行创建的Channel实例。
```python
# 通过下标访问模块（返回可调用对象）
result: LLMModuleResult = await app['module_name'](
    template_param1='value1',
    _output_channel=your_channel  # 可选，传入自定义通道接收实时输出
)
```

通用请求入口，支持传入模块请求或直接请求，适合需要动态构造请求的场景。`call` 方法内部等价于依次执行 `send_request` → `wait_response` → `post_process`，一般情况下使用 `call` 即可完成整个请求流程。如果需要在发送请求后先处理其他事务、并发等待多个请求、在响应返回前执行中断逻辑，或对后处理进行自定义控制，也可以手动拆解这三个步骤：

```python
# 一键调用（等同于 send_request + wait_response + post_process）
async def call(
    self,
    request: Union[LLMModuleRequest, LLMDirectRequest]
) -> LLMModuleResult: ...

# 手动分步调用
request_state = await app.send_request(request)
# ... 可在此并发处理其他任务 ...
response = await request_state.wait_response()   # 等待响应完成
result = await app.post_process(request_state)   # 后处理（内部会复用已完成的响应）
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
LLMWorkflow 扩展自 LLMApplication，支持动态任务链和并行子任务执行，工作流路径可以预先定义，也可以在执行过程中根据LLM输出或任务结果动态调整。

核心概念：
- 通过 `submit(task)` 提交任务执行，内置并发控制自动限制并行任务数，避免资源耗尽。
- 任务完成后，如果设置了 `next_task`，会自动提交下一个任务（链式执行）。
- 使用 `wait_tasks(*tasks)` 提交一组任务并等待它们全部完成。
- 使用 `run(start_task)` 执行一条完整的任务链（依次跟随 `next_task`）。
- 任务内部可以通过 `workflow.wait_tasks`/`submit` 启动支线（子任务链）并等待。
- 使用 `release_worker()` 异步上下文管理器，在等待子任务时临时释放并发配额，避免嵌套任务死锁，提升并发利用率。

#### 3.2.2 任务定义
`Task` 是通用任务基类，支持两种使用方式：
1. 继承 `Task` 并重写 `run()` 方法
2. 使用 `Task.set_func()` 指定执行函数

`LLMTask` 是专门用于LLM调用的任务类，处理LLM请求和响应，支持工具调用处理。

```python
# 任务状态枚举
class TaskStatus(enum.Enum):
    INIT = 0    # 初始化
    WAITING = 1 # 待执行
    RUNNING = 2 # 执行中
    COMPLETED = 3  # 已完成
    FAILED = 4     # 已失败
```

```python
class Task:
    """
    通用任务基类，执行自定义函数
    重写 run() 或用 Task.set_func 指定具体函数
    """
    def __init__(self,
        input_data: Optional[Dict[str, Any]] = None,
        workflow: Optional['LLMWorkflow'] = None
    ): ...

    # 设置任务执行函数
    def set_func(self, fn: Callable): ...

    # 任务执行逻辑，可重写
    async def run(self):
        raise NotImplementedError

    # 克隆任务（用于工具调用等场景）
    def clone(self): ...

    # 获取当前任务后续的所有链式任务
    def following_tasks(self) -> List['Task']: ...
```

```python
class LLMTask(Task):
    """LLM任务类，专注于LLM调用和工具调用处理"""
    def __init__(
            self,
            input_data: Union[_model.LLMModuleRequest, _model.LLMDirectRequest, None] = None,
            workflow: Optional['LLMWorkflow'] = None
    ): ...

    # 后处理钩子：在LLM模块返回结果后执行
    async def post_process(self, module_result: _model.LLMModuleResult):
        """
        后处理钩子：在LLM模块返回结果后执行
        可重写此方法处理工具调用、生成下一个任务等
        """
        pass

    # 工具调用处理：将工具调用转为下一步任务
    def on_tool_call_goto(self, **kwargs: Callable[[], 'Task']):
        """
        用于 post_process，将工具调用转为下一步的 Task
        * 非工具调用，直接返回
        * 对第一个调用，设置 next_task 为相应 Task，结束当前任务
        * 无匹配的调用，报错
        """

    # 工具调用处理：执行工具调用并继续对话
    async def run_tool_calls(self, **kwargs: Callable):
        """
        用于 post_process，工具调用作为函数调用，生成下一步的 LLMTask，结束当前任务
        """
```

```python
class LLMWorkflowError(_model.LLMApplicationError):
    """工作流执行错误基类"""
```

#### 3.2.3 工作流接口
```python
class LLMWorkflow(LLMApplication):
    # 提交任务，任务完成后自动提交next_task
    async def submit(self, task: Task): ...

    # 提交多个任务（如果尚未提交），并行等待所有任务完成
    async def wait_tasks(self, *tasks: Task, timeout: Optional[float] = None): ...

    # 执行一条任务链：从start_task开始，依次执行next_task直到结束，返回最后一个完成的任务
    async def run(self, start_task: Task) -> Task: ...

    # 停止所有任务，清理资源
    async def stop(self): ...
    
    # 临时释放当前任务的并发配额，用于等待子任务时避免死锁
    @asynccontextmanager
    async def release_worker(self):
        """
        可在任务执行逻辑（run/post_process）内调用；在外部调用不做任何操作
        典型场景：
            async with self.workflow.release_worker():
                await self.workflow.wait_tasks(subtask1, subtask2)
        """
```
LLMWorkflow 支持 `async with` 上下文管理器，退出时自动停止工作流、清理资源。

#### 3.2.4 使用示例

**通用任务示例**：
```python
# 方式1：继承Task并重写run方法
class SimpleTask(aitoolman.Task):
    async def run(self):
        return self.input_data['x'] + self.input_data['y']

# 方式2：使用set_func指定函数
def simple_func(x, y):
    return x + y

task = aitoolman.Task({"x":1, "y":2})
task.set_func(simple_func)
```

**LLM任务示例**：
```python
# 定义LLM任务类
class TranslationTask(aitoolman.LLMTask):
    async def post_process(self, module_result):
        # 根据工具调用决定下一步
        self.on_tool_call_goto(
            refine=RefinementTask,
            finalize=FinalizationTask
        )


# 运行工作流
async with aitoolman.LLMWorkflow(client, config) as workflow:
    start_task = TranslationTask(
        aitoolman.LLMModuleRequest(
            module_name="translator",
            template_params={"text": "Hello"}
        )
    )
    final_task = await workflow.run(start_task)
```

**嵌套任务（release_worker使用）示例**：
```python
class FolderAnalysisTask(aitoolman.Task):
    async def run(self):
        folder_path = self.input_data['path']
        # 分析文件夹得到子文件/子文件夹列表
        sub_items = scan_folder(folder_path)
        sub_tasks = []
        for item in sub_items:
            if item.is_dir:
                sub_tasks.append(FolderAnalysisTask({"path": item.path}))
            else:
                sub_tasks.append(FileAnalysisTask({"path": item.path}))

        # 等待子任务时释放当前配额，避免死锁，提升并发效率
        async with self.workflow.release_worker():
            await self.workflow.wait_tasks(*sub_tasks)

        # 合并子任务结果
        return merge_results([t.output_data for t in sub_tasks])
```

## 4. 传输层

客户端与通道

### 4.1 Channel 通道系统

通道系统用于异步传输流式响应、推理内容等，用于实现实时输出。LLMClient 输出的通道内容固定使用两个 topic：`reasoning`表示模型的思考推理内容，`response`表示模型的实际响应内容。

```python
# 通道事件，包含topic和数据
class ChannelEvent(NamedTuple):
    topic: str
    data: Any

class Channel:
    """通用通道类，支持读写ChannelEvent"""
    async def read(self) -> ChannelEvent: ...
    async def write(self, message: ChannelEvent): ...
    # 标记通道写入结束，EOF
    async def write_complete(self): ...

class NullChannel(Channel):
    """空通道，丢弃所有写入内容，默认使用"""
```

框架提供便捷函数直接打印通道内容到控制台：
```python
async def print_channel_output(
    channel: ChannelReader,
    topic_names: Dict[str, str],
    header: bool = False
):
    """
    将Channel内容直接打印到stdout
    :param channel: 监听的 Channel
    :param topic_names: topic 名称和显示名称映射，比如 {'reasoning': '思考过程', 'response': '输出内
容'}
    :param header: 是否打印 topic 名称标题
    """
```

Channel 拆分为子通道：
```python
class ChannelDemux:
    """
    使用方式：
        async with ChannelDemux(source_channel) as demux:
            reader1 = demux.get_reader("topic1")
            reader2 = demux.get_reader("topic2", "topic3") # 订阅多个topic
            reader3 = demux.get_reader() # 订阅所有topic
            # 分别使用 reader 并行消费事件
            # 不需要时可调用 await reader.close() 取消订阅
    或使用 start(), close()
    """
    def __init__(self, source: ChannelReader, topics: Optional[Set[str]] = None):
        """
        :param source: 输入 ChannelReader，用于读取 ChannelEvent
        :param topics: 可选的预定义 topic 集合，用于提前初始化订阅列表（不影响动态订阅）
        """

    def get_reader(self, *topics: str) -> DemuxChannelReader:
        """
        订阅指定 topic，并返回专属的 DemuxReader。
        不传入topics时订阅所有topic，可同时传入多个topic同时订阅。
        每次调用都会创建一个独立的消费通道，因此多个消费者可以同时订阅同一个 topic 并各自接收完整的事件流。
        """

    async def start(self): ...

    async def close(self): ...

    async def __aenter__(self): ...

    async def __aexit__(self, exc_type, exc_val, exc_tb): ...
```

#### 4.1.1 XML 标签过滤器
BaseXmlTagFilter从流式文本中自动识别XML标签，主要用于LLM输出单层XML标签，表示不同类型的文本，输出到不同主题。例如：输出当前状态、给用户的输出、给应用程序的处理结果。

```python
class BaseXmlTagFilter(ChannelWriter):
    """XML 标签过滤基类"""
    async def write(self, message: ChannelEvent) -> None: ...

class XmlTagToChannelFilter(BaseXmlTagFilter):
    """XML 标签分发到不同主题"""
    def __init__(self, output_channel: ChannelWriter, tags: Set[str], input_topic: str = 'response'): ...
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
        output_channel: Optional[ChannelWriter] = None
    ) -> LLMProviderRequest

    async def cancel(self, request_id: str): ...
    async def cancel_all(self, context_id: Optional[str] = None): ...
    async def audit_event(self, context_id: str, event_type: str, **kwargs): ...
    
    async def list_models(self, tag: Optional[str] = None) -> List[ModelInfo]: ... # 列出可用模型
    # 解析实际候选模型
    async def resolve_model(self, tags: Union[str, List[str]], messages: Optional[List[Message]] = None) -> List[str]: ...
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

#### 4.2.4 Mock测试客户端
在 test.mock_llmclient 中：

```python
class MockLLMClient(LLMClient):
    """模拟LLM客户端，用于单元测试，不实际调用远程API"""
    def __init__(self, response_generator: Optional[Callable[[LLMProviderRequest], LLMProviderResponse]] = None):
        """用户提供模拟函数：输入 LLMProviderRequest，输出 LLMProviderResponse"""

def make_simple_response(
    request: LLMProviderRequest,
    response_content: Union[str, List[model.ToolCall]]
) -> LLMProviderResponse:
    """直接根据要返回的文本或工具调用创建 LLMProviderResponse 对象"""
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
    """管理多个 LLM 提供商，处理 API 调用、重试、资源限制、模型路由"""
    def __init__(self, config: Dict[str, Any])

    def process_request(
        self,
        request: LLMProviderRequest,
        callback: Optional[Callable[[LLMProviderRequest], typing.Coroutine]] = None
    ) -> RequestTask: ...

    async def cancel_request(self, request_id: str): ...
    async def cancel_all_requests(self, client_id: str, context_id: Optional[str] = None): ...

    # 解析候选模型
    def resolve_model(self, tags: List[str], messages: Optional[List[Message]] = None) -> List[str]: ...
    def list_models(self, tag: Optional[str] = None) -> List[ModelInfo]: ... # 列出可用模型
```

### 5.3 模型路由
框架内置灵活的模型路由机制，通过标签分组实现业务代码与具体模型的解耦，支持多标签匹配、自动权重计算、Token超限过滤、多候选降级重试等能力，便于统一调整模型资源、平衡成本和效果。

#### 5.3.1 配置方式
在提供商配置文件的 `[model_tag]` 段定义标签与模型的映射关系，每个标签下的模型列表按推荐优先级排序（越靠前权重越高）。业务代码中直接使用标签，无需关心底层具体模型，支持随时调整模型组合而不修改业务代码。
详细配置说明参考《[配置文件文档](./config.md)》的 model_tag 部分。

#### 5.3.2 路由算法
输入标签列表（`input_tags`）后，系统按以下规则计算最优候选模型列表：
1. **精确匹配优先**：遍历 input_tags，若存在与真实模型名完全匹配的项，直接返回该模型（单元素列表）
2. **标签交集匹配**：查找每个输入标签对应的模型集合及权重，取所有标签对应模型的交集（单个模型名/别名视为单元素集合，权重为1）
3. **Token限制过滤**：若传入 `messages` 参数，自动估算输入Token总数，过滤掉配置了 `max_input_tokens` 且Token总数超出限制的模型
4. **权重排序**：对每个模型在所有标签中的权重求和，按总权重从高到低排序
5. **排队长度排序**：权重相同的同级模型，按当前排队任务数从低到高排序，优先选择排队较少的模型
6. **稳定随机排序**：若权重和并发量仍相同，以 `context_id`（如有）或 `client_id` 为随机种子进行稳定排序，同一上下文的多次调用结果稳定，不同上下文之间实现负载均衡

返回候选模型列表

#### 5.3.3 使用方式
- **resolve_model 接口**：直接调用 `client.resolve_model(tags, messages, context_id=context_id)`，传入标签列表、可选的消息列表和上下文 ID（可选），得到排序后的候选模型列表
- **模块调用传参**：通过 `LLMModuleRequest` 的 `model_name` 参数或 `app['module'](_model_name=xxx)` 的 `_model_name` 参数，可传入单个标签字符串或多个标签的列表，系统自动路由到最优模型
- **model_rank 参数**：用于选择候选列表中的第N个模型（从0开始计数），若超过候选列表长度则自动取模。当遇到 `LLMRetriableError` 可重试错误时，可将 `model_rank` 加1后重试，自动切换到下一个优先级的候选模型，实现故障降级和负载均衡
- **context_id 参数**：`resolve_model` 的 `context_id` 参数用于作为稳定随机排序的随机种子，同一 `context_id` 的多次调用结果保持一致；未显式传入时使用 `client_id` 作为种子

#### 5.3.4 应用场景
- 按业务场景分组：如 `fast`（快速响应）、`precise`（高精度）、`low_cost`（低成本）、`code`（代码处理）、`multimodal`（多模态）等，业务代码直接使用场景标签
- 多标签交叉匹配：如传入 `["code", "multimodal"]` 自动选择同时支持代码和多模态能力，在两者中最优的模型
- 故障自动降级：主模型不可用时自动切换到备用模型，提升服务可用性
- 成本动态调整：根据业务优先级自动选择不同成本档位的模型

## 6. 实用工具

### 6.1 命令行工具
aitoolman 提供了命令行工具集，可快速完成模型测试、服务启停、监控审计、代码修改、运维管理等常用操作，适合快速验证效果、调试配置和运维管理。

主要命令包括：
* `server`: 启动 LLM 微服务端，统一管理模型资源、对外提供调用接口
* `client`: 通用 LLM 测试客户端，支持本地/远程服务调用、多模态输入、流式输出
* `monitor`: 微服务监控工具，实时查看请求统计、Token使用、错误日志，支持数据持久化到SQLite
* `code-edit`: 代码智能修改工具，支持参考文件上下文、单/多文件新建或修改，自动保存修改后代码
* `manage`: 远程服务运维工具，支持列出模型、热更新配置、启停模型等操作

命令的详细参数和使用示例可通过添加 `--help` 参数查看：

```bash
# 查看所有支持的命令
python3 -m aitoolman --help

# 查看单个子命令的详细用法，例如查看server命令参数
python3 -m aitoolman server --help
```

### 6.2 配置文件

用于加载和管理TOML格式的配置文件，支持文件和字符串两种来源。

```python
# 载入 toml 格式的配置文件
aitoolman.load_config(filename)

# 载入 toml 格式的配置文件文本
aitoolman.load_config_str(s)
```

### 6.3 后处理工具
提供常用的文本后处理函数，用于解析LLM输出。
推荐直接使用 [fix-llm-xml](https://pypi.org/project/fix-llm-xml/) 库解析 XML。

```python
# JSON 解析（自动修复格式错误）
aitoolman.postprocess.parse_json(s: str) -> Any

# XML 内容提取
fix_llm_xml.find_xml_document(s: str, root: str, with_tag: bool = False) -> Optional[str]

# XML 解析为字典 (xmltodict)
fix_llm_xml.parse_xml(s: str, root: str, **kwargs) -> Optional[Dict]
```

### 6.4 资源管理器
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

### 7.1 提供商配置文件 (llm_provider.toml)

```toml
# 服务器配置（ZeroMQ 微服务）
[server]
zmq_router_rpc = "tcp://*:5555" # ZeroMQ ROUTER 端点
zmq_pub_event = "tcp://*:5556"  # ZeroMQ PUB 端点（审计日志）
zmq_auth_token = "YOUR_SECRET_TOKEN"  # 接口认证令牌
zmq_manage_token = "YOUR_MANAGE_TOKEN" # 管理权限认证令牌（可选）

# 默认配置
[default]
timeout = 10          # 流式请求（stream=True）超时，网络活动超时
timeout_batch = 300   # 批量请求（stream=False）超时
max_retries = 3
parallel = 1
api_type = "openai"
rank_adjust_ratio = 0.25 # 模型路由权重调整参数，可选

# 模型标签配置，用于模型路由，按优先级排序，尽量写全
# 业务代码使用标签，无需关心底层具体模型，支持灵活切换和路由
# 模型路由支持同时传入多个标签，系统会自动匹配同时符合所有标签的可用模型，结合输入Token数限制、模型优先级自动选择最优模型。
# 例如传入 `["code", "multimodal"]` 可自动选择支持多模态的代码模型，传入 `["low_cost", "fast"]` 可自动选择低成本且响应快的模型，适配各类交叉场景。
[model_tag]
"low_cost" = ["doubao-seed-2.0-mini", "qwen-flash", "deepseek-v4-flash"]  # 低成本模型组，适合批量文本处理
"fast" = ["deepseek-v4-flash", "qwen-flash", "doubao-seed-2.0-lite"]  # 高速响应模型组，适合快速处理任务
"creative" = ["deepseek-v4-pro", "kimi-k2.5"]  # 创意写作模型组，适合内容生成
"precise" = ["Doubao-Seed-2.0-pro", "glm-5.1"]  # 高精度任务模型组，适合精准修改、信息提取
"code" = ["glm-5.1", "qwen3.7-max", "deepseek-v4-pro", "kimi-k2.5"]  # 代码处理模型组，适合代码修改、生成
"multimodal" = ["Doubao-Seed-2.0-pro", "doubao-seed-2.0-mini", "qwen-vl"]  # 多模态模型组，适合图片/视频理解

# API 配置
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
```

### 7.2 提示词配置文件 (app_prompt.toml)

```toml
# 模块默认配置
[module_default]
model = "fast"         # 默认使用快速推理模型组
stream = false
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
model = "creative"     # 使用创意模型组
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
model = ["fast", "low_cost"]
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
model = ["precise", "fast"]
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
model = "Doubao-Seed-2.0-pro"
stream = true
template.user = "{{message}}"
```

## 8. 示例应用

### 8.1 项目目录结构
```
my_llm_app/
├── config/
│   ├── llm_provider.toml          # API 配置（模型、密钥、路由标签）
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
- 创建自定义Channel接收实时输出，传入模块调用
- 使用后处理器提取代码块

```python
import asyncio
import aitoolman

def extract_code(text: str) -> str:
    ...

async def main():
    # 加载配置
    api_config = aitoolman.load_config("config/llm_provider.toml")
    prompt_config = aitoolman.load_config("config/app_prompt.toml")

    # 创建客户端和应用
    async with aitoolman.LLMLocalClient(api_config) as client:
        app = aitoolman.LLMApplication(client, prompt_config)
        app.add_processor("extract_code", extract_code)

        # 创建通道，监听输出
        output_channel = aitoolman.Channel()
        output_task = asyncio.create_task(aitoolman.print_channel_output(
            output_channel,
            topic_names={'reasoning': '思考过程', 'response': '代码输出'},
            header=True
        ))

        # 调用代码编辑器模块，指定使用代码+多模态标签组，自动选择最优模型
        result = await app['code_editor'](
            code_content=open("app.py").read(),
            instruction="参考截图添加错误处理逻辑",
            references=[{"filename": "utils.py", "content": open("utils.py").read()}],
            _model_name=["code", "multimodal"],
            _output_channel=output_channel
        )
        result.raise_for_status()

        # 等待输出任务完成
        await output_task

        # 保存结果
        with open("app_modified.py", "w") as f:
            f.write(result.data)

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
        ticket_type=ticket['type'],
        _model_name=["low_cost", "fast"] # 使用低成本+高速模型组
    )
    result.raise_for_status()
    return {
        "ticket_id": ticket['id'],
        "category": result.data['main'],
        "sub_category": result.data['sub']
    }

async def main():
    # 加载配置
    api_config = aitoolman.load_config("config/llm_provider.toml")
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

使用 `LLMWorkflow` 构建串行任务链：
- 定义多个分析任务
- 使用 `next_task` 建立串行依赖关系，并在任务内部传递数据
- 使用 `run()` 执行整个任务链
- 合并分析结果

```python
import asyncio
import aitoolman

class DataFetchTask(aitoolman.Task):
    async def run(self):
        query = self.input_data['query']
        # 模拟数据获取
        sales_data = [100, 200, 300]
        # 将结果传递给下一个任务
        if self.next_task:
            self.next_task.input_data['sales_data'] = sales_data
        return sales_data

class DataAnalysisTask(aitoolman.Task):
    async def run(self):
        sales_data = self.input_data['sales_data']
        total = sum(sales_data)
        average = total / len(sales_data)
        result = {'total': total, 'average': average}
        if self.next_task:
            self.next_task.input_data['analysis_result'] = result
        return result

class ReportGenerationTask(aitoolman.Task):
    async def run(self):
        fmt = self.input_data['format']
        analysis_result = self.input_data['analysis_result']
        if fmt == 'markdown':
            report = f"""
# 销售数据分析报告
- 总销售额: {analysis_result['total']}
- 平均销售额: {analysis_result['average']}
"""
            return report
        return str(analysis_result)

async def main():
    api_config = aitoolman.load_config("config/llm_provider.toml")
    prompt_config = aitoolman.load_config("config/app_prompt.toml")

    async with aitoolman.LLMLocalClient(api_config) as client:
        async with aitoolman.LLMWorkflow(client, prompt_config) as workflow:
            # 创建任务
            fetch_task = DataFetchTask({"query": "2024年Q1销售数据"})
            analysis_task = DataAnalysisTask({})
            report_task = ReportGenerationTask({"format": "markdown"})

            # 建立任务链
            fetch_task.next_task = analysis_task
            analysis_task.next_task = report_task

            # 执行整个链
            last_task = await workflow.run(fetch_task)

            print("分析报告生成完成：")
            print(last_task.output_data)

if __name__ == "__main__":
    asyncio.run(main())
```

### 8.5 动态工作流：文件夹批量分析
递归分析文件夹结构：
- 定义文件夹分析任务，输出子项列表
- 在 `run()` 中根据分析内容动态创建子任务
- 调用 `release_worker()` 释放配额后使用 `wait_tasks()` 等待所有子任务并行完成
- 处理文件内容分析、分类等子任务

### 8.6 串行工作流：多步骤决策
使用 `Task.next_task` 构建串行流程：
- 定义任务链：分析 → 规划 → 执行 → 验证
- 每个任务根据输出决定下一步
- 使用 `workflow.run()` 执行整个流程
- 支持工具调用作为流程分支点

内容审核流水线：

```python
import asyncio
import aitoolman

class ContentSubmitTask(aitoolman.LLMTask):
    def __init__(self, input_data):
        super().__init__(
            aitoolman.LLMModuleRequest(
                module_name="content_validator",
                template_params=input_data
            )
        )

    async def post_process(self, module_result):
        # 根据验证结果决定下一步
        if module_result.data['status'] == "valid":
            self.next_task = AIAuditTask({
                "content": self.input_data.template_params['content']
            })
        else:
            self.next_task = RejectionTask({
                "reason": module_result.data['reason']
            })

class AIAuditTask(aitoolman.LLMTask):
    def __init__(self, input_data):
        super().__init__(
            aitoolman.LLMModuleRequest(
                module_name="content_auditor",
                template_params=input_data
            )
        )

    async def post_process(self, module_result):
        if module_result.data['risk_level'] <= 1:
            self.next_task = PublishTask({
                "content": self.input_data.template_params['content']
            })
        else:
            self.next_task = ManualReviewTask({
                "content": self.input_data.template_params['content'],
                "risk": module_result.data['risk_details']
            })

class ManualReviewTask(aitoolman.LLMTask):
    def __init__(self, input_data):
        super().__init__(
            aitoolman.LLMModuleRequest(
                module_name="review_coordinator",
                template_params=input_data
            )
        )

    async def post_process(self, module_result):
        if module_result.data['approved']:
            self.next_task = PublishTask({
                "content": self.input_data.template_params['content']
            })
        else:
            self.next_task = RevisionTask({
                "content": self.input_data.template_params['content'],
                "feedback": module_result.data['feedback']
            })

class PublishTask(aitoolman.Task):
    async def run(self):
        content = self.input_data['content']
        # 模拟发布操作
        return {"status": "published", "content": content}

class RevisionTask(aitoolman.Task):
    async def run(self):
        content = self.input_data['content']
        feedback = self.input_data['feedback']
        # 模拟返回修订建议
        return {"status": "revision_needed", "feedback": feedback}

class RejectionTask(aitoolman.Task):
    async def run(self):
        reason = self.input_data['reason']
        # 模拟拒绝操作
        return {"status": "rejected", "reason": reason}

# 运行工作流
async def main():
    # 初始化工作流
    api_config = aitoolman.load_config("config/llm_provider.toml")
    prompt_config = aitoolman.load_config("config/app_prompt.toml")

    async with aitoolman.LLMLocalClient(api_config) as client:
        async with aitoolman.LLMWorkflow(client, prompt_config) as workflow:
            # 启动工作流
            start_task = ContentSubmitTask({
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
5. **热更新运维**：无需重启服务即可调整模型配置、路由规则、启停模型
6. **安全隔离**：敏感 API 密钥不暴露给客户端应用

### 9.2 功能特性
- **ZeroMQ 通信**：高性能、低延迟的进程间通信
- **认证授权**：支持普通接口令牌和管理权限令牌两级认证，确保接口安全
- **请求队列**：智能调度，避免超额请求
- **模型路由**：自动按配置中的自定义标签选择最优模型，支持Token超限过滤
- **实时监控**：通过 PUB 接口发布审计日志
- **客户端管理**：支持请求取消、批量取消等操作
- **热更新能力**：运行时修改配置、模型状态、标签规则，无需重启服务

### 9.3 使用方法

#### 9.3.1 启动服务端
```bash
# 使用默认配置文件
python3 -m aitoolman server -c llm_provider.toml

# 启用详细日志
python3 -m aitoolman server -c llm_provider.toml -v
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
# 交互式测试，指定单个模型标签
python3 -m aitoolman client \
  -z tcp://localhost:5555 \
  -m fast \
  -a your-auth-token

# 指定多个标签，自动路由到匹配的最优模型
python3 -m aitoolman client \
  -z tcp://localhost:5555 \
  -m code -m multimodal \
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

#### 9.3.5 运维管理
使用`manage`命令可对远程服务进行热运维，无需重启服务：
```bash
# 查看所有可用模型
python3 -m aitoolman manage -z tcp://localhost:5555 -a manage_token list_models

# 按标签过滤可用模型
python3 -m aitoolman manage -z tcp://localhost:5555 -a manage_token list_models --tag code

# 禁用指定模型
python3 -m aitoolman manage -z tcp://localhost:5555 -a manage_token change_api_status --model doubao-mini --disable

# 启用指定模型
python3 -m aitoolman manage -z tcp://localhost:5555 -a manage_token change_api_status --model doubao-mini --enable

# 热更新全量配置
python3 -m aitoolman manage -z tcp://localhost:5555 -a manage_token update_config -c new_config.toml
```

## 10. 最佳实践

### 10.1 提示词设计
用试卷中的“**简答题**”（案例分析题）的格式设计提示词。格式：

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
- **标签分组**：根据业务场景将模型分组为不同标签，业务代码使用标签而非具体模型名，便于后续统一调整模型

### 10.3 错误处理
```python
try:
    result = await app['module'](...)
    result.raise_for_status()  # 检查完成状态
    processed_data = result.data
except LLMNoAvailableModelError as e:
    # 无匹配可用模型：调整标签或扩容模型资源
    pass
except LLMLengthLimitError as e:
    # 处理长度限制：分段处理或换大模型
    pass
except LLMApiRequestError as e:
    # API 错误：重试或直接报错
    pass
except LLMResponseFormatError as e:
    # 返回格式错误：重试或调整提示词
    pass
```

### 10.4 性能优化
- **并行度配置**：根据模型配额合理设置 `max_parallel_tasks` 参数
- **流式响应**：对长文本使用流式，提升用户体验
- **批量请求**：对于批量/后台任务，采用批量请求，或专用的批量接口，提升并行度
- **资源管理**：使用 `ResourceManager` 避免超额请求
- **缓存策略**：对重复查询实现结果缓存
- **嵌套任务**：多层嵌套任务场景下使用 `release_worker()` 释放配额，避免死锁，提升并发利用率
- **模型路由**：根据任务特性选择合适的标签组合，平衡成本、速度和效果

### 10.5 调试技巧
- **通道监听**：创建自定义Channel并传入请求，使用 `print_channel_output` 实时查看 LLM 输出
- **审计日志**：启用监控器记录所有请求和响应
- **逐步执行**：复杂工作流可先测试单个任务
- **提供商日志**：启用 `logging.DEBUG` 查看原始 API 交互
- **单元测试**：使用 `MockLLMClient` 模拟LLM返回，无需调用外部服务即可验证业务逻辑
- **模型解析验证**：使用`resolve_model`接口验证标签匹配的实际模型，调试路由规则
