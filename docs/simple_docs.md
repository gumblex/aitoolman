
# aitoolman 框架简略文档

## 一、框架概览
### 1.1 设计理念
aitoolman 是面向开发者的 LLM 应用框架，核心思想是**AI 作为工具人**：LLM 仅在开发者预设规则内执行明确指令，无自主决策权。开发者掌握100%控制权，所有提示词、数据流、控制流均由代码主导，无隐藏逻辑。框架具备流程可调试、多模型兼容、模块化、生产级特性等优势。

### 1.2 与传统 Agent 框架的区别
| 维度 | aitoolman | 传统 Agent 框架 |
|------|-----------|----------------|
| LLM角色 | 仅执行预设指令的工具人 | 可自主决策的智能体 |
| 控制权 | 开发者完全控制流程 | 框架隐含控制流 |
| 提示词 | 完全自定义，开发者编写所有模板 | 自带大量默认提示词，自定义成本高 |
| 多模型适配 | 原生支持多厂商，切换成本低 | 多为单一平台优化，适配成本高 |
| 功能边界 | 专注LLM编排，无冗余依赖 | 内置RAG等大量功能，依赖臃肿 |
| 适用场景 | 企业级可控流程、批量任务 | 开放式智能体、探索性应用 |

### 1.3 核心功能与适用场景
#### 核心功能
- 提示词模板管理与渲染
- 单步/多步工作流编排
- 多厂商LLM统一适配
- 流式响应、工具调用支持
- 生产级错误处理、监控审计
- 微服务部署能力

#### 适用场景
1. 专业文本处理：总结、翻译、数据标注、结构化信息提取
2. 标准报告生成：基于结构化数据生成规范文档
3. 可控多轮对话：按预设流程处理复杂用户请求
4. 批量任务处理：工单分类、内容审核等大规模异步任务

### 1.4 架构总览
框架采用分层架构：
1. **用户应用层**：业务逻辑实现
2. **应用层**：LLMApplication（模板管理、单步调用）、LLMWorkflow（流程编排）
3. **传输层**：LLMClient（请求发送）、Channel（流式数据传输）
4. **数据接口层**：ProviderManager（多厂商适配、请求调度、限流重试）
5. **LLM提供商API**：底层模型服务

## 二、核心数据模型
### 完整核心类定义
```python
import typing
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field

class Message(typing.NamedTuple):
    """发送给 LLM 的消息"""
    role: Optional[str] = None                # 角色：system/user/assistant/tool
    content: Optional[str] = None             # 文本内容
    media_content: Optional[MediaContent] = None  # 多媒体内容
    reasoning_content: Optional[str] = None   # assistant 的推理内容
    tool_call_id: Optional[str] = None        # 工具调用 ID（用于 tool 角色）
    raw_value: Optional[Dict] = None          # 提供商原始消息（忽略以上所有字段）

class LLMDirectRequest(typing.NamedTuple):
    """应用层直接请求参数"""
    model_name: str
    messages: List[Message]
    tools: Optional[Dict[str, Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    stream: bool = False
    output_channel: Union[str, TextFragmentChannel, None] = None
    reasoning_channel: Union[str, TextFragmentChannel, None] = None
    post_processor: Optional[str] = None

class LLMModuleRequest(typing.NamedTuple):
    """应用层模板请求参数（模块配置）"""
    module_name: str                    # 模块名称
    template_params: Dict[str, Any]     # 模板参数
    model_name: Optional[str] = None    # 覆盖模块默认模型
    context_messages: List[Message] = []  # 上下文消息
    media_content: Optional[MediaContent] = None  # 多媒体内容
    tools: Optional[Dict[str, Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = None
    output_channel: Union[str, TextFragmentChannel, None] = None
    reasoning_channel: Union[str, TextFragmentChannel, None] = None

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

### 其他核心类说明
- `MediaContent`：封装图片、视频等多媒体内容，统一多模态输入格式
- `ToolCall`：LLM返回的工具调用请求
- `LLMProviderRequest/Response`：数据接口层与LLM提供商交互的请求/响应模型，上层应用无需关注
- `FinishReason`：枚举所有请求完成状态，支持自动转换为对应异常类型

## 三、核心组件说明
### 3.1 应用层
#### LLMApplication
- **使用场景**：单步LLM任务、模板化调用场景，是框架核心入口
- **方法签名**：
  ```python
  def __init__(
      self,
      client: LLMClient,                     # LLM 客户端实例
      config_dict: Optional[Dict[str, Any]] = None,  # 提示词配置字典
      processors: Optional[Dict[str, Callable[[str], Any]]] = None,  # 自定义后处理器映射
      channels: Optional[Dict[str, TextFragmentChannel]] = None,  # 自定义通道映射
      context_id: Optional[str] = None       # 上下文ID，用于请求追踪、审计
  ):
  ```
  ```python
  async def call(
      self,
      request: Union[LLMModuleRequest, LLMDirectRequest]
  ) -> LLMModuleResult:
  ```
  ```python
  def render_template(self, template_name: str, **kwargs) -> str:
  ```
- **主要方法说明**：
  - 下标访问：`app['module_name'](**params)` 直接调用配置好的模板模块
  - `call()`：通用请求入口，支持直接请求/模板请求两种方式
  - `render_template()`：渲染指定名称的提示词模板
  - `add_processor()`/`add_channel()`：注册自定义后处理器、通道
  - `factory()`：创建应用工厂，适合批量并发任务场景

#### LLMWorkflow
- **使用场景**：多步骤复杂流程编排，支持串行/并行两种任务调度
- **方法签名**：
  ```python
  def add_task(self, task: Task, next_task: Optional[Task] = None):
  ```
  ```python
  async def run(self, start_task: Task) -> Task:
  ```
  ```python
  async def wait_tasks(self, *tasks: Task, timeout: Optional[float] = None):
  ```
- **主要方法说明**：
  - `run()`：执行串行任务流
  - `add_task()`：添加任务并配置依赖关系，构建DAG
  - `wait_tasks()`：等待指定并行任务完成

#### 配套任务类
- `Task`：通用自定义任务基类，支持重写run方法或指定执行函数
  - 方法说明：
    - `__init__(input_data: Optional[Dict[str, Any]] = None, workflow: Optional['LLMWorkflow'] = None)`：初始化任务实例，绑定输入参数和所属工作流。
    - `set_func(fn: Callable)`：设置任务执行函数，无需重写run方法即可快速定义自定义任务。
    - `async def run(self, **input_data)`：任务核心执行逻辑，默认需重写或通过set_func指定执行函数。
    - `clone()`：克隆当前任务实例，用于工具调用、多实例并行等需要复用任务配置的场景。
- `LLMTask`：LLM调用专用任务，内置工具调用处理、后处理钩子
  - 方法说明：
    - `__init__(input_data: Union[LLMModuleRequest, LLMDirectRequest, None] = None, workflow: Optional['LLMWorkflow'] = None)`：初始化LLM任务实例，绑定LLM请求参数和所属工作流。
    - `async def post_process(self, module_result: LLMModuleResult)`：LLM返回结果后的后处理钩子，可重写实现结果加工、流程分支跳转逻辑。
    - `on_tool_call_goto(**kwargs: Callable[[], 'Task'])`：根据LLM返回的工具调用名称，自动配置下一跳转任务。
    - `async def run_tool_calls(self, **kwargs: Callable)`：自动执行LLM返回的工具调用，生成下一步LLM对话请求任务。

### 3.2 传输层
#### Channel 通道系统
- **使用场景**：异步传输流式响应、推理内容，实现实时输出
- `Channel`：通用异步消息通道基类，用于框架不同组件之间的跨线程、跨进程异步数据传输，支持泛型类型的消息收发。
  - 方法说明：
    - `async def read(self) -> T`：异步阻塞读取通道中的一条消息。
    - `async def write(self, message: T)`：异步写入一条消息到通道。
    - `async def write_complete(self)`：标记通道当前消息写入完成，发送EOF标识通知读取端。
- `TextFragmentChannel`：专门用于传输文本片段的通道实现，适配LLM流式输出场景，支持文本分片读写和完整消息自动合并。
  - 方法说明：
    - `async def read(self) -> Optional[str]`：异步读取单条文本片段，返回None表示当前消息传输结束。
    - `async def write(self, message: Optional[str])`：异步写入单条文本片段，写入None标记当前消息结束。
    - `async def write_whole_message(self, message: str)`：一次性写入完整文本消息，自动拆分片段并添加结束标记。
    - `async def read_whole_message(self) -> str`：异步读取当前消息的所有文本片段，自动合并为完整字符串后返回。
- **其他配套类**：
  - `ChannelCollector`：多通道收集器，统一处理多来源输出
  - `XmlTagToChannelFilter`：自动识别流式输出中的XML标签，分发到对应通道

#### LLMClient 客户端
- **使用场景**：统一不同部署方式的LLM调用接口
- **核心实现**：
  - `LLMLocalClient`：本地客户端，直接调用LLM提供商API，适合单机应用
  - `LLMZmqClient`：ZeroMQ远程客户端，连接LLM微服务，适合多项目共享场景
- **主要方法**：
  - `request()`：发送LLM请求
  - `cancel()`：取消指定请求
  - `audit_event()`：上报自定义审计事件

### 3.3 数据接口层
#### LLMFormatStrategy
- **使用场景**：统一不同LLM提供商的请求/响应格式，实现供应商无关
- **内置实现**：OpenAI兼容格式、Anthropic Claude格式，可扩展其他厂商适配

#### LLMProviderManager
- **使用场景**：管理多个LLM提供商，处理API调用、重试、资源限制、请求调度，上层应用无需关注

## 四、配置文件
### 4.1 llm_provider.toml
统一管理LLM服务商API配置、模型别名、服务端参数、密钥与配额，无需在业务代码中硬编码接口信息。

### 4.2 app_prompt.toml
管理提示词模板、模块配置、全局工具定义，与应用层调用一一对应：
- 每个`[module.xxx]`配置对应`app['xxx']`的调用
- `template`中的`{{param}}`对应调用时传入的模板参数
- `post_processor`字段对应`add_processor()`注册的后处理器
- `tools`配置对应LLM工具调用的定义

```toml
# 模块默认配置
[module_default]
model = "Fast-Model"
stream = false
options = { max_tokens = 4000 }

# 翻译模块示例
[module.translator]
template.user = """
请翻译下列文本为{{target_lang}}，仅输出翻译结果，不要多余内容：
{{text}}
"""
```

## 五、使用示例
```python
import asyncio
import aitoolman

async def main():
    # 加载配置
    api_config = aitoolman.load_config("config/llm_provider.toml")
    prompt_config = aitoolman.load_config("config/app_prompt.toml")

    # 创建客户端与应用
    async with aitoolman.LLMLocalClient(api_config) as client:
        app = aitoolman.LLMApplication(client, prompt_config)
        
        # 调用翻译模块
        result = await app['translator'](
            text="Hello World",
            target_lang="中文"
        )
        
        # 检查请求状态，异常自动抛出
        result.raise_for_status()
        
        # 输出结果
        print("翻译结果：", result.text)

if __name__ == "__main__":
    asyncio.run(main())
```

## 六、最佳实践
LLM是执行工具，所有流程、提示词、输出规则均由开发者明确定义。

### 提示词设计
- 按「阅读材料→问题背景→任务说明→具体要求→输出范例」结构设计，单次仅分配一个明确任务
- 输出格式明确，优先使用JSON/XML等可结构化解析的格式，避免模糊的自然语言输出
- 语言简洁，多写「要做什么」，少写「不要做什么」，避免无关信息干扰

### 模块设计
- 复杂流程拆分为多个单一职责的小模块，避免单个提示词承担过多任务
- 优先使用模板变量传递参数，尽量减少上下文消息堆砌，避免LLM遗忘和无关信息干扰
- 工具定义清晰具体，仅提供当前模块需要的工具
