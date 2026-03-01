# aitoolman Framework Developer Documentation

## 1. Framework Overview

### 1.1 Design Philosophy
aitoolman is a developer-oriented LLM application framework with the core idea of **AI as a Normal Worker** — treating LLM like a junior employee in an enterprise, only executing clear instructions within the rules and processes preset by developers, without autonomous decision-making power. The framework has clear role divisions:
- End User = Client: Proposes vague requirements
- Application Developer = Business Owner: Defines all rules, processes, prompt templates, and holds 100% decision-making power
- LLMWorkflow = Middle Manager: Schedules tasks according to preset workflows, switches process branches based on presets or LLM return results
- LLM Module = Junior Employee: Only completes assigned single, clear tasks, with output strictly adhering to preset format requirements

The framework emphasizes:
- **Full User Control**: All prompts, data flows, and control flows are dominated by user code; LLM only serves as an execution tool with no hidden business logic or unexpected behavior
- **Transparent and Debuggable Workflow**: All data sent to and received from LLM can be customized and audited, facilitating problem troubleshooting and prompt optimization
- **Vendor Agnostic**: Unified adaptation of multiple LLM providers through abstraction layer, easy model switching while fully utilizing each provider's unique features
- **Modular Design**: Components have single responsibilities, making them easy to test, replace, and reuse
- **Production-Grade Features**: Built-in resource management, error handling, microservice deployment, monitoring, and auditing capabilities, ready for direct production use

Whether it's simple one-time queries or complex multi-step business processes, aitoolman provides a stable, reliable, and maintainable solution. The framework encourages developers to deeply understand business logic, carefully design prompts, and seamlessly integrate AI capabilities into existing systems.

### 1.2 Differences from Traditional Agent Frameworks
| Dimension              | aitoolman                                                        | Traditional Agent Frameworks                                              |
|------------------------|------------------------------------------------------------------|---------------------------------------------------------------------------|
| LLM's Role             | **LLM is a "worker", only executes predefined instructions**     | LLM is an autonomous "agent" with decision-making capabilities            |
| Control                | User has full control over workflows                             | Framework implies hidden control flows                                    |
| Prompts                | Developers write all prompts with full customization             | Includes many default prompts, high adaptation cost for custom scenarios  |
| Multi-model Adaptation | Natively supports multi-vendor with low switching cost           | Optimized for single platforms, high adaptation cost                      |
| Feature Boundaries     | Focuses on LLM function orchestration, no redundant dependencies | Bulky dependencies with built-in vector indexing, RAG, and other features |
| Use Cases              | Controllable workflow orchestration, batch task processing       | Autonomous agents, exploratory applications                               |

### 1.3 Use Cases
1. Professional Applications: Clear input and output scope
   - **Text Processing**: Summarization, translation, data annotation, structured information extraction
   - **Report Generation**: Generate standardized text reports based on existing structured data
2. Assistant Applications: Uncertain user input requirements
   - **Multi-Round Dialogue**: Handle complex user requests through workflow orchestration
   - **Intelligent Planning**: Decompose complex tasks into executable steps
   - **Dynamic Decision Making**: Adjust workflows based on context and tool call results

In this framework, prompt templates are core, and dialogue context is supplementary. It encourages optimizing prompt quality through careful template parameter orchestration instead of stacking dialogue context, thereby avoiding information forgetting and misguidance.

### 1.4 Architecture Overview
The framework adopts a layered architecture:
1. User Application Layer: Business logic implementation
2. Application Layer (LLMApplication / LLMWorkflow): Template management, workflow orchestration, result processing
3. Transport Layer (LLMClient / Channel): Request sending, streaming response transmission, microservice communication
4. Data Interface Layer (ProviderManager): Multi-vendor adaptation, request scheduling, rate limiting and retries
5. LLM Provider API (OpenAI / Anthropic, etc.): Underlying LLM services

## 2. Data Model Classes

### Message Structure
The `Message` class represents messages sent to LLM, supporting text, multimedia content, tool call responses, etc.

```python
class Message(typing.NamedTuple):
    """Message sent to LLM"""
    role: Optional[str] = None                # Role: system/user/assistant/tool
    content: Optional[str] = None             # Text content
    media_content: Optional[MediaContent] = None  # Multimedia content
    reasoning_content: Optional[str] = None   # Assistant's reasoning content
    tool_call_id: Optional[str] = None        # Tool call ID (for tool role)
    raw_value: Optional[Dict] = None          # Provider's raw message (ignores all above fields)
```

`MediaContent` is used to encapsulate multimedia content such as images and videos, supporting multiple sources like local files, binary data, and remote URLs, unifying the format of multimodal inputs.

```python
class MediaContent(typing.NamedTuple):
    """Multimedia content (images/videos, etc.)"""
    media_type: str                    # Media type, e.g., "image", "video"
    # Priority order:
    # 1. raw_value
    raw_value: Optional[Dict] = None   # Provider's raw value (highest priority)
    # 2. data + mime_type
    data: Optional[bytes] = None       # Raw binary data
    mime_type: Optional[str] = None    # MIME type, e.g., "image/jpeg"
    # 3. filename
    filename: Optional[str] = None     # Filename (auto-read)
    # 4. url
    url: Optional[str] = None          # Remote URL
    options: Optional[Dict] = None     # Provider-specific options
```

The `ToolCall` class represents tool call requests returned by LLM.

```python
class ToolCall(typing.NamedTuple):
    """Tool call request returned by LLM"""
    name: str                     # Tool function name
    arguments_text: str           # Argument string (raw JSON)
    arguments: Optional[Dict[str, Any]]  # Parsed argument dictionary
    id: Optional[str] = None      # Tool call ID
    type: str = 'function'        # Type, default 'function'
```


### Application Layer Request/Response
`LLMDirectRequest`: Direct request parameters without template module configuration, suitable for classic tool calls, multi-round dialogues, and dynamically generated request scenarios.

```python
class LLMDirectRequest(typing.NamedTuple):
    """Application layer direct request parameters"""
    model_name: str
    messages: List[Message]
    tools: Optional[Dict[str, Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    stream: bool = False
    output_channel: Union[str, TextFragmentChannel, None] = None
    reasoning_channel: Union[str, TextFragmentChannel, None] = None
    post_processor: Optional[str] = None
```

`LLMModuleRequest`: Sends requests based on configured template modules, automatically renders prompts, loads preset tool and model configurations, and allows overriding default configurations.

```python
class LLMModuleRequest(typing.NamedTuple):
    """Application layer template request parameters (module configuration)"""
    module_name: str                    # Module name
    template_params: Dict[str, Any]     # Template parameters
    model_name: Optional[str] = None    # Override module's default model
    context_messages: List[Message] = []  # Context messages
    media_content: Optional[MediaContent] = None  # Multimedia content

    # Override original configuration
    tools: Optional[Dict[str, Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = None
    output_channel: Union[str, TextFragmentChannel, None] = None
    reasoning_channel: Union[str, TextFragmentChannel, None] = None
```

`LLMModuleResult`: Unified encapsulation of all results returned by LLM, including raw responses, processed text, tool calls, status information, etc.

```python
@dataclass
class LLMModuleResult:
    """Application layer (template) request response"""
    model_name: str                     # Actual model name
    module_name: Optional[str]          # Module name
    request: LLMDirectRequest           # Actual request parameters
    post_processor: Optional[str] = None  # Post-processor name
    response_text: str = ""             # Raw response text
    response_reasoning: str = ""        # Raw reasoning text
    text: str = ""                      # Processed text
    tool_calls: List[ToolCall] = field(default_factory=list)  # Tool calls
    status: FinishReason = FinishReason.stop  # Completion status
    error_text: Optional[str] = None    # Error message
    request_params: Dict[str, Any] = field(default_factory=dict)  # Original template parameters
    response_message: Optional[Message] = None  # Original response message
    data: Any = None                    # Post-processed result

    def raise_for_status(self):
        """Raise error based on status"""

    async def run_tool_calls(self, fn_map: Dict[str, Callable]) -> Optional[LLMDirectRequest]:
        """Execute tool calls and return next request parameters"""
```


### Data Interface Layer Request/Response
Used for interacting with LLM providers, no need for upper-layer applications to关注.

`LLMProviderRequest`: Request sent to model provider, containing complete request data and channel configurations.

```python
@dataclass
class LLMProviderRequest:
    """Request sent to model provider"""
    client_id: str                    # Client identifier
    context_id: Optional[str]         # Context identifier (for dialogue association)
    request_id: str                   # Unique request ID
    model_name: str                   # Model name
    messages: List[Message]           # Message list
    tools: Dict[str, Dict[str, Any]]  # Tool definitions
    options: Dict[str, Any]           # Provider-specific options
    stream: bool = False              # Whether to use streaming response
    output_channel: Optional[TextFragmentChannel]  # Output channel
    reasoning_channel: Optional[TextFragmentChannel]  # Reasoning channel
    is_cancelled: bool = False        # Whether cancelled
    response: asyncio.Future[LLMProviderResponse]  # Response Future
```

`LLMProviderResponse`: Model provider's response, containing complete time statistics and content information.

```python
@dataclass
class LLMProviderResponse:
    """Model provider response"""
    client_id: str
    context_id: str
    request_id: str
    model_name: str
    stream: bool

    # Time statistics
    start_time: Optional[float] = None      # Request start time
    queue_time: Optional[float] = None      # Queue time
    queue_length: Optional[int] = None      # Queue length when requesting
    time_to_first_token: Optional[float] = None  # Time to first token
    total_response_time: Optional[float] = None  # Total response time

    # Response content
    response_text: str = ""                 # Complete response text
    response_reasoning: str = ""            # Complete reasoning text
    response_tool_calls: List[ToolCall] = field(default_factory=list)  # Tool calls

    # Completion information
    finish_reason: Optional[str] = None     # Completion reason
    error_text: Optional[str] = None        # Error message
    prompt_tokens: Optional[int] = None     # Input token count
    completion_tokens: Optional[int] = None # Output token count

    # Raw data
    response_message: Optional[Message] = None  # Original response message
```

### Status and Error Types
The `FinishReason` enum defines all possible completion reasons, used to determine request result status.

```python
class FinishReason(enum.Enum):
    # Provider reasons
    stop = "stop"                     # Normal completion
    length = "length"                 # Length limit reached
    content_filter = "content_filter" # Content filtered by moderation
    tool_calls = "tool_calls"         # Tool calls invoked

    # Local reasons
    error = "error"                   # General error
    error_request = "error: request"  # Request error
    error_format = "error: format"    # Response format error
    error_app = "error: application"  # Application error
    cancelled = "cancelled"           # Request cancelled

    unknown = "unknown"               # Unknown reason
```

The `raise_for_status()` method in LLMModuleResult or FinishReason automatically converts completion reasons into corresponding exception types.

```python
class LLMError(RuntimeError): ...
class LLMLengthLimitError(LLMError): ...      # Response length limit reached
class LLMContentFilterError(LLMError): ...    # Content filtered by moderation
class LLMApiRequestError(LLMError): ...       # API request error
class LLMResponseFormatError(LLMError): ...   # Response format error
class LLMApplicationError(LLMError): ...      # Application code error
class LLMCancelledError(LLMError): ...        # Request cancelled
class LLMUnknownError(LLMError): ...          # Unknown completion reason
class GenericError(LLMError): ...             # General error
```

## 3. Application Layer

### 3.1 LLMApplication Class
LLMApplication is the core entry class of the framework, responsible for managing configurations, template rendering, LLM calls, channels, and post-processors, serving as the basic carrier for all LLM applications.

#### 3.1.1 Core Features
LLMApplication is the main entry point of the framework, responsible for:
- Loading and managing modules in configuration files
- Rendering prompt templates
- Calling LLM and processing responses
- Managing context variables and channels

#### 3.1.2 Initialization
Create an LLM application instance, bind client, load configurations, register post-processors and channels. Each application instance corresponds to an independent context.

```python
class LLMApplication:
    def __init__(
        self,
        client: LLMClient,                     # LLM client
        config_dict: Optional[Dict[str, Any]] = None,  # Configuration file dictionary
        processors: Optional[Dict[str, Callable[[str], Any]]] = None,  # Post-processors
        channels: Optional[Dict[str, TextFragmentChannel]] = None,  # Custom channels
        context_id: Optional[str] = None       # Context ID (for client tracking, debugging, and auditing)
    ): ...
```

#### 3.1.3 Main Interfaces
Quickly retrieve callable module objects by module name, and complete LLM calls directly by passing template parameters.
```python
# Access module via subscript (returns callable object)
result: LLMModuleResult = await app['module_name'](template_param1='value1', ...)
```

General request entry, supports passing module requests or direct requests, suitable for dynamically constructing requests.
```python
# Directly call LLM (bypasses module configuration)
async def call(
    self,
    request: Union[LLMModuleRequest, LLMDirectRequest]
) -> LLMModuleResult: ...
```

Render a template with the specified name, used for custom message content generation.
```python
# Render template
def render_template(self, template_name: str, **kwargs) -> str: ...
```

Register custom post-processors for parsing specific format content returned by LLM, such as extracting JSON, XML, code blocks, etc.
```python
# Add custom post-processor
def add_processor(self, name: str, processor: Callable): ...
```

Register custom channels for receiving streaming responses, reasoning content, etc., enabling real-time output to front-end, files, and other custom scenarios.
```python
# Add custom channel
def add_channel(self, name: str, channel: TextFragmentChannel): ...
```

Send custom audit events for recording business-level operations, facilitating subsequent problem troubleshooting and business data statistics. Received and unified processed by LLMClient backend.
```python
# Trigger audit event
async def audit_event(self, event_type: str, **kwargs): ...
```

Create an application factory for generating multiple independent application instances during batch task processing, avoiding context interference and supporting concurrent processing.
```python
# Use LLMApplication.factory for batch tasks to generate multiple instances for parallel processing
@classmethod
def factory(
        cls,
        client: _client.LLMClient,
        config_dict: Optional[Dict[str, Any]] = None,
        processors: Optional[Dict[str, Callable[[str], Any]]] = None,
        channels: Optional[Dict[str, _channel.TextFragmentChannel]] = None,
) -> Callable[..., 'LLMApplication']: ...
```

#### 3.1.4 Usage Examples
```python
# Method 1: Call via module
result = await app['translator'](text="Hello", target_lang="zh")
print(result.data)  # Post-processed translation result

# Method 2: Direct call
direct_request = LLMDirectRequest(
    model_name="gpt-4",
    messages=[Message(role="user", content="Hello")],
    stream=True
)
result = await app.call(direct_request)
```

### 3.2 LLMWorkflow Class

#### 3.2.1 Core Concepts
LLMWorkflow extends LLMApplication, supporting dynamic workflow execution with two construction modes:

1. **Sequential Mode**: Connect tasks via `next_task` attribute, execute using `run()` method
2. **Parallel Mode**: Build DAG (Directed Acyclic Graph) via `add_task()` method, execute using `wait_tasks()` method

The two modes can be mixed.

#### 3.2.2 Task Definition
`Task` is a generic task base class that supports two usage methods:
1. Inherit `Task` and override the `run()` method
2. Use `Task.set_func()` to specify the execution function

`LLMTask` is a task class specifically designed for LLM calls, handling LLM requests and responses, and supporting tool call processing.

```python
# Task status enum
class TaskStatus(enum.Enum):
    INIT = 0    # Initialized
    WAITING = 1 # Pending execution
    RUNNING = 2 # Executing
    COMPLETED = 3  # Completed
    FAILED = 4     # Failed
    DEPENDENCY_FAILED = 5  # Dependency failed
```

```python
class Task:
    """
    Generic task base class that executes custom functions
    Override run() or use Task.set_func to specify the execution function
    """
    def __init__(
        self,
        input_data: Optional[Dict[str, Any]] = None,
        workflow: Optional['LLMWorkflow'] = None
    ): ...

    # Set task execution function
    def set_func(self, fn: Callable): ...

    # Task execution logic, can be overridden
    async def run(self, **input_data):
        raise NotImplementedError

    # Clone task (used in tool call scenarios, etc.)
    def clone(self): ...
```

```python
class LLMTask(Task):
    """LLM task class focused on LLM calls and tool call processing"""
    def __init__(
            self,
            input_data: Union[_model.LLMModuleRequest, _model.LLMDirectRequest, None] = None,
            workflow: Optional['LLMWorkflow'] = None
    ): ...

    # Post-processing hook: Executed after LLM module returns result
    async def post_process(self, module_result: _model.LLMModuleResult):
        """
        Post-processing hook: Executed after LLM module returns result
        Can be overridden to handle tool calls, generate next task, etc.
        """
        pass

    # Tool call processing: Convert tool calls to next task
    def on_tool_call_goto(self, **kwargs: Callable[[], 'Task']):
        """
        Used in post_process to convert tool calls to next Task
        * Not a tool call: Return directly
        * First call: Set next_task to corresponding Task and end current task
        * No matching call: Throw error
        """

    # Tool call processing: Execute tool calls and continue dialogue
    async def run_tool_calls(self, **kwargs: Callable):
        """
        Used in post_process: Tool calls as function calls, generate next LLMTask and end current task
        """
```

```python
class TaskDependencyError(LLMWorkflowError):
    """Dependency task execution error, containing all failed tasks"""
```

#### 3.2.3 Workflow Interfaces
```python
class LLMWorkflow(LLMApplication):
    # Add task to workflow
    def add_task(self, task: Task, next_task: Optional[Task] = None):
        """
        Add background task task, not executed immediately
        task is the task to run before next_task

        Args:
            task: Task to add
            next_task: Task to execute after the added task, or None
        """

    # Wait for specified tasks to complete
    async def wait_tasks(self, *tasks: Task, timeout: Optional[float] = None): ...

    # Run sequential workflow
    async def run(self, start_task: Task) -> Task: ...
```

#### 3.2.4 Usage Examples

**Generic Task Example**:
```python
# Method 1: Inherit Task and override run method
class SimpleTask(aitoolman.Task):
    async def run(self, x, y):
        return x + y

# Method 2: Use set_func to specify function
def simple_func(x, y):
    return x + y

task = aitoolman.Task({"x":1, "y":2})
task.set_func(simple_func)
```

**LLM Task Example**:
```python
# Define LLM task class
class TranslationTask(aitoolman.LLMTask):
    async def post_process(self, module_result):
        # Determine next step based on tool call
        self.on_tool_call_goto(
            refine=RefinementTask,
            finalize=FinalizationTask
        )


# Run workflow
workflow = aitoolman.LLMWorkflow(client, config)
start_task = TranslationTask(
    aitoolman.LLMModuleRequest(
        module_name="translator",
        template_params={"text": "Hello"}
    )
)
final_task = await workflow.run(start_task)
```

## 4. Transport Layer

### 4.1 Channel System
The channel system is used for asynchronous transmission of streaming responses, reasoning content, etc., enabling real-time output.

#### 4.1.1 Basic Channels
Channel is a general asynchronous message channel for asynchronous communication between different components.

TextFragmentChannel is specifically used for transmitting text fragments, supporting streaming reception of LLM output fragments.

```python
class Channel(Generic[T]):
    """General channel base class"""
    async def read(self) -> T: ...
    async def write(self, message: T): ...
    # Mark channel write completion, EOF
    async def write_complete(self): ...

class TextFragmentChannel(Channel[Optional[str]]):
    """
    Text fragment channel
    None indicates end of a complete message
    """
    # Read all fragments and merge into a complete message
    async def read_whole_message(self) -> str: ...
```

#### 4.1.2 Channel Collectors
Mainly used in application development to listen to multiple channels simultaneously and unify processing of outputs from different sources.

```python
class ChannelCollector(abc.ABC):
    """Multi-channel collector base class"""
    async def start_listening(self): ...
    def close(self): ...

    @abc.abstractmethod
    async def on_channel_start(self, channel_name: str):
        """Channel starts current output"""

    @abc.abstractmethod
    async def on_channel_read(self, channel_name: str, message):
        """Channel outputs content"""

    @abc.abstractmethod
    async def on_channel_end(self, channel_name: str):
        """Channel ends current output"""

    @abc.abstractmethod
    async def on_channel_eof(self, channel_name: str):
        """Channel ends all outputs"""

class DefaultTextChannelCollector(ChannelCollector):
    """Default text channel collector (prints to console)"""
```

#### 4.1.3 XML Tag Filters
BaseXmlTagFilter automatically identifies XML tags from streaming text, mainly used when LLM outputs single-layer XML tags representing different types of text, which are then output to different channels. For example: output current status, output to users, processing results to applications.

```python
class BaseXmlTagFilter(abc.ABC):
    """XML tag filter base class"""
    async def write(self, message: Optional[str]) -> None: ...

class XmlTagToChannelFilter(BaseXmlTagFilter):
    """Distribute XML tags to different channels"""
    def __init__(self, default_channel: TextFragmentChannel,
                 channel_map: Dict[str, TextFragmentChannel]): ...
```

### 4.2 LLMClient Abstraction
LLMClient is the abstract base class for LLM clients, unifying the calling interfaces of different deployment methods.

#### 4.2.1 Client Interfaces
```python
class LLMClient(abc.ABC):
    """LLM client abstract base class"""
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

#### 4.2.2 Local Client
LLMLocalClient is a local client that directly calls LLM provider APIs.

```python
class LLMLocalClient(LLMClient):
    """Local client (directly calls ProviderManager)"""
    def __init__(self, config: Dict[str, Any]): ...
```

#### 4.2.3 ZeroMQ Client (Microservice)
LLMZmqClient is a ZeroMQ remote client that connects to remote LLM microservices.

```python
class LLMZmqClient(LLMClient):
    """ZeroMQ client (connects to remote service)"""
    def __init__(self, router_endpoint: str, auth_token: Optional[str] = None): ...

    # Cancel all requests for the specified context, suitable for terminating all incomplete requests when user exits session
    async def cancel_all(self, context_id: Optional[str] = None): ...
```

## 5. Data Interface Layer

LLM Provider Management

### 5.1 LLMFormatStrategy Format Strategy
Unified handling of request/response formats for different LLM providers, achieving vendor agnosticism.

```python
class LLMFormatStrategy(abc.ABC):
    """LLM request/response format conversion strategy"""
    def serialize_tool_description(self, tools_configs: Dict[str, Dict[str, Any]]) -> List[Dict]: ...
    def parse_tool_calls(self, tool_calls: List[Dict]) -> List[ToolCall]: ...
    def serialize_message(self, message: Message) -> Dict[str, Any]: ...
    def make_request_body(self, request: LLMProviderRequest) -> Dict[str, Any]: ...
    def parse_batch_response(self, response: LLMProviderResponse, response_data: Dict[str, Any]): ...
    def parse_stream_event(self, response: LLMProviderResponse, event: httpx_sse.ServerSentEvent) -> StreamEvent: ...


class OpenAICompatibleFormat(LLMFormatStrategy):
    """OpenAI API compatible format"""


class AnthropicFormat(LLMFormatStrategy):
    """Anthropic Claude API format"""
```


### 5.2 LLMProviderManager Provider Manager
```python
class LLMProviderManager:
    """Manages multiple LLM providers, handling API calls, retries, and resource limits"""
    def __init__(self, config: Dict[str, Any])

    def process_request(
        self,
        request: LLMProviderRequest,
        callback: Optional[Callable[[LLMProviderRequest], typing.Coroutine]] = None
    ) -> RequestTask: ...

    async def cancel_request(self, request_id: str): ...
    async def cancel_all_requests(self, client_id: str, context_id: Optional[str] = None): ...
```

## 6. Utility Tools

### 6.1 Configuration Files
Used for loading and managing TOML format configuration files, supporting both file and string sources.

```python
# Load toml format configuration file
aitoolman.load_config(filename)

# Load toml format configuration file text
aitoolman.load_config_str(s)
```

### 6.2 Post-processors (aitoolman.postprocess)
Provides common text post-processing functions for parsing LLM outputs.

```python
# JSON parsing (automatically fixes format errors)
parse_json(s: str) -> Any

# XML content extraction
get_xml_tag_content(s: str, root: str, with_tag: bool = False) -> Optional[str]

# XML parsing to dictionary
parse_xml(s: str, root: str) -> Optional[Dict]
```

### 6.3 Resource Manager
```python
class ResourceManager:
    """Manages model parallel processing resources, preventing excessive requests and resource competition"""
    def __init__(self, capacities: Dict[str, int] = None): ...

    @asynccontextmanager
    async def acquire(self, key: str, task_name: Optional[str] = None): ...
    async def add_resource(self, key: str, capacity: int): ...
    async def remove_resource(self, key: str, force: bool = False): ...
    def get_queue_length(self, key: str) -> int: ...
    def get_stats(self, key: str) -> Dict: ...
```


## 7. Configuration Files

Refer to the [Configuration File Documentation](./config.md) for detailed configuration parameters.

### 7.1 Provider Configuration File (llm_config.toml)

```toml
# Server Configuration (ZeroMQ Microservice)
[server]
zmq_router_rpc = "tcp://*:5555" # ZeroMQ ROUTER endpoint
zmq_pub_event = "tcp://*:5556"  # ZeroMQ PUB endpoint (audit logs)
zmq_auth_token = "YOUR_SECRET_TOKEN"  # Interface authentication token

# Default Configuration
[default]
timeout = 600
max_retries = 3
parallel = 1
api_type = "openai"

# Model Alias Mapping
# Use aliases in business configurations without worrying about underlying model details
[model_alias]
"Creative-Model" = "DeepSeek-v3.2-251201"
"Precise-Model" = "GPT-4o"
"Fast-Model" = "Doubao-Seed-1.6-flash-250828"
"Cheap-Model" = "Doubao-Mini-1.5"
"Code-Model" = "CodeLlama-70B-Instruct"

# API Configuration
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

### 7.2 Prompt Configuration File (app_prompt.toml)

```toml
# Module Default Configuration
[module_default]
model = "Fast-Model"         # Default to fast inference model
stream = false
output_channel = "stdout"
reasoning_channel = "reasoning"
options = { max_tokens = 4000 }

# Global Templates (can be referenced by modules)
[template]
"greeting" = "Hello, {{name}}!"

# Global Tool Definitions
[tools."Tool Name"]
type = "function"
description = "Tool function description"
param."Parameter Name 1".type = "string/integer/boolean"
param."Parameter Name 1".description = "Parameter description"
param."Parameter Name 1".required = true
param."Parameter Name 2".type = "string"
param."Parameter Name 2".description = "Optional parameter"
param."Parameter Name 2".required = false

# Raw Text Processing Module
[module.raw]
template.user = "{{content}}"

# Article Summarization Module
[module.summerize]
model = "Creative-Model"     # Use creative model
template.user = """
Article Title: {{title}}
Article Content: <article>{{content}}</article>

Based on the article content:
1. List arguments and important points according to the article structure
2. List cases in the article and the problems they illustrate
3. Summarize this article

Output Format:
<response>
<point>Argument 1</point>
<point>Argument 2</point>
<case>Case 1</case>
<case>Case 2</case>
<summery>Summary</summery>
</response>
"""
post_processor = "builtin.parse_json"

# Schedule Planning Module (supports tool calls)
[module.task_planner]
model = "Fast-Model"         # Use fast inference model
stream = true
template.user = """
As a scheduling assistant, analyze user instructions:
- If there are specific to-do items, call the add_task tool
- If not, ask the user for more details

User says: {{user_input}}
"""

tools.add_task.type = "function"
tools.add_task.description = "Add schedule"

tools.add_task.param.datetime.type = "string"
tools.add_task.param.datetime.description = "Date and time, e.g., 2025-12-31 12:34:56"
tools.add_task.param.datetime.required = false

tools.add_task.param.content.type = "string"
tools.add_task.param.content.description = "To-do item content"
tools.add_task.param.content.required = true

# JSON Extraction Module
[module.json_extractor]
model = "Precise-Model"      # Use high-precision model
template.user = """
Extract structured information from the following text:
{{text}}

Output Format:
<response>
{
  "Field 1": "Value 1",
  "Field 2": "Value 2"
}
</response>
"""
post_processor = "builtin.parse_json"

# Multi-Round Dialogue Module
[module.chat]
model = "Doubao-Seed-1.6"
stream = true
template.user = "{{message}}"
```

## 8. Example Applications

### 8.1 Project Directory Structure
```
my_llm_app/
├── config/
│   ├── llm_config.toml          # API configuration (models, keys)
│   └── app_prompt.toml          # Prompt configuration (modules, templates)
├── src/
│   ├── __init__.py
│   ├── main.py                  # Application entry
│   ├── processors.py            # Custom post-processors
│   └── workflows.py             # Workflow logic
└── requirements.txt
```

### 8.2 Simple Q&A: Code Assistant
Implement a code modification tool:
- Load configuration files and prompt templates
- Create client and application instances
- Call code editing module to process user input
- Use post-processor to extract code blocks

```python
import asyncio
import aitoolman

def extract_code(text: str) -> str:
    ...

async def main():
    # Load configurations
    api_config = aitoolman.load_config("config/llm_config.toml")
    prompt_config = aitoolman.load_config("config/app_prompt.toml")
    
    # Create client and application
    async with aitoolman.LLMLocalClient(api_config) as client:
        app = aitoolman.LLMApplication(client, prompt_config)
        app.add_processor("extract_code", extract_code)
        
        # Monitor output channels
        collector = aitoolman.DefaultTextChannelCollector({
            'Reasoning Process': app.channels['reasoning'],
            'Code Output': app.channels['stdout']
        })
        output_task = asyncio.create_task(collector.start_listening())
        
        # Call code editor module
        result = await app['code_editor'](
            code_content=open("app.py").read(),
            instruction="Add error handling logic",
            references=[{"filename": "utils.py", "content": open("utils.py").read()}]
        )
        result.raise_for_status()
        
        # Save results
        with open("app_modified.py", "w") as f:
            f.write(result.data)
        
        output_task.close()
        await output_task

if __name__ == "__main__":
    asyncio.run(main())
```

### 8.3 LLM as a Function: Batch Ticket Classification
Use `LLMApplication.factory()` to create an application factory for batch ticket processing:
- Define classification module and post-processor
- Create asynchronous task group for concurrent processing
- Collect and output classification results

```python
import asyncio
import aitoolman

def parse_classification(text: str) -> dict:
    ...

async def process_ticket(app_factory, ticket):
    """Process single ticket"""
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
    # Load configurations
    api_config = aitoolman.load_config("config/llm_config.toml")
    prompt_config = aitoolman.load_config("config/app_prompt.toml")
    
    # Create application factory
    async with aitoolman.LLMLocalClient(api_config) as client:
        app_factory = aitoolman.LLMApplication.factory(
            client=client,
            config_dict=prompt_config,
            processors={
                "classify": parse_classification
            }
        )
        
        # Simulate batch tickets
        tickets = [
            {"id": "1", "type": "Technical Support", "content": "System login failed"},
            {"id": "2", "type": "Business Inquiry", "content": "How to apply for invoice"},
            # More tickets...
        ]
        
        # Parallel processing
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(process_ticket(app_factory, t)) for t in tickets]
            
        # Collect results
        results = [t.result() for t in tasks]
        for res in results:
            print(f"Ticket {res['ticket_id']}: {res['category']} > {res['sub_category']}")

if __name__ == "__main__":
    asyncio.run(main())
```


### 8.4 Static Workflow: Data Analysis Pipeline
Application Scenario: Known task dependencies, such as first fetching data, then analyzing, and finally generating reports

Use `LLMWorkflow` to build a static DAG:
- Define multiple analysis tasks
- Use `add_task()` to establish dependencies
- Use `wait_tasks()` to wait for all tasks to complete
- Merge analysis results

```python
import asyncio
import aitoolman

# Define task classes
class DataFetchTask(aitoolman.Task):
    async def run(self, query):
        # Simulate data fetching
        return {"sales_data": [100, 200, 300]}

class DataAnalysisTask(aitoolman.Task):
    async def run(self, sales_data):
        # Simulate data analysis
        return {
            "total": sum(sales_data),
            "average": sum(sales_data)/len(sales_data)
        }

class ReportGenerationTask(aitoolman.Task):
    async def run(self, format, analysis_result):
        # Simulate report generation
        if format == "markdown":
            return f"""
# Sales Data Analysis Report
- Total Sales: {analysis_result['total']}
- Average Sales: {analysis_result['average']}
"""
        return str(analysis_result)

async def main():
    # Initialize workflow
    api_config = aitoolman.load_config("config/llm_config.toml")
    prompt_config = aitoolman.load_config("config/app_prompt.toml")

    async with aitoolman.LLMLocalClient(api_config) as client:
        workflow = aitoolman.LLMWorkflow(client, prompt_config)

        # Create tasks
        fetch_task = DataFetchTask({"query": "Q1 2024 sales data"})
        analysis_task = DataAnalysisTask()
        report_task = ReportGenerationTask({"format": "markdown"})

        # Establish dependencies: fetch → analysis → report
        workflow.add_task(fetch_task, analysis_task)
        workflow.add_task(analysis_task, report_task)

        # Wait for all tasks to complete
        await workflow.wait_tasks(report_task)

        # Get results
        print("Analysis report generated successfully:")
        print(report_task.output_data)

if __name__ == "__main__":
    asyncio.run(main())
```

### 8.5 Dynamic Workflow: Batch Folder Analysis
Recursively analyze folder structure:
- Define folder analysis task to output sub-item list
- Dynamically add sub-tasks in `run()` based on analysis content
- Use `add_task()` and `wait_tasks()` to manage recursive dependencies
- Handle file content analysis, classification, and other sub-tasks

### 8.6 Sequential Workflow: Multi-step Decision Making
Use `Task.next_task` to build sequential processes:
- Define task chain: analysis → planning → execution → verification
- Each task determines the next step based on output
- Use `workflow.run()` to execute the entire process
- Support tool calls as process branch points

Content Review Pipeline:

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
        # Determine next step based on verification results
        if module_result.data['status'] == "valid":
            self.next_task = AIAuditTask({
                "content": self.input_data.request.template_params['content']
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
                "content": self.input_data.request.template_params['content']
            })
        else:
            self.next_task = ManualReviewTask({
                "content": self.input_data.request.template_params['content'],
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
                "content": self.input_data.request.template_params['content']
            })
        else:
            self.next_task = RevisionTask({
                "content": self.input_data.request.template_params['content'],
                "feedback": module_result.data['feedback']
            })

class PublishTask(aitoolman.Task):
    async def run(self, content):
        # Simulate publishing operation
        return {"status": "published", "content": content}

class RevisionTask(aitoolman.Task):
    async def run(self, content, feedback):
        # Simulate returning revision suggestions
        return {"status": "revision_needed", "feedback": feedback}

class RejectionTask(aitoolman.Task):
    async def run(self, reason):
        # Simulate rejection operation
        return {"status": "rejected", "reason": reason}

# Run workflow
async def main():
    # Initialize workflow
    api_config = aitoolman.load_config("config/llm_config.toml")
    prompt_config = aitoolman.load_config("config/app_prompt.toml")

    async with aitoolman.LLMLocalClient(api_config) as client:
        workflow = aitoolman.LLMWorkflow(client, prompt_config)

        # Start workflow
        start_task = ContentSubmitTask({
            "content": "Article content to be published...",
            "type": "article"
        })
        final_task = await workflow.run(start_task)

        print(f"Process completed, final status: {final_task.task_name}")
        print(f"Result: {final_task.output_data}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 9. Microservices

### 9.1 Application Scenarios
The aitoolman microservice architecture is suitable for the following scenarios:

1. **Multi-Project Collaboration**: Multiple projects share the same set of LLM infrastructure
2. **Centralized Resource Management**: Unified management of API keys, model quotas, and access control
3. **High-Availability Deployment**: Ensure service stability through load balancing and failover
4. **Audit and Monitoring**: Centralized logging of all LLM call logs and performance metrics
5. **Security Isolation**: Sensitive API keys are not exposed to client applications

### 9.2 Feature Highlights
- **ZeroMQ Communication**: High-performance, low-latency inter-process communication
- **Authentication and Authorization**: Support token authentication to ensure interface security
- **Request Queue**: Intelligent scheduling to avoid excessive requests
- **Real-Time Monitoring**: Publish audit logs via PUB interface
- **Client Management**: Support request cancellation, batch cancellation, and other operations

### 9.3 Usage Methods

#### 9.3.1 Start Server
```bash
# Use default configuration file
python3 -m aitoolman server -c llm_config.toml

# Enable detailed logging
python3 -m aitoolman server -c llm_config.toml -v
```

After starting the server, two ZeroMQ endpoints will be bound:
- **ROUTER Endpoint** (default: tcp://*:5555): Handles client requests and responses
- **PUB Endpoint** (default: tcp://*:5556): Publishes audit logs and monitoring data

#### 9.3.2 Client Connection
```python
from aitoolman.zmqclient import LLMZmqClient
client = LLMZmqClient(
    router_endpoint="tcp://localhost:5555",
    auth_token="your-secret-token"  # Optional, must match server configuration
)
```

Command-line client test:
```bash
# Interactive test
python3 -m aitoolman client \
  -r tcp://localhost:5555 \
  -m gpt-4 \
  -a your-auth-token

# Specify model alias
python3 -m aitoolman client \
  -r tcp://localhost:5555 \
  -m Creative-Model \
  -a your-auth-token
```

#### 9.3.4 Monitoring and Audit
```bash
# Console real-time monitoring
python3 -m aitoolman monitor --pub-endpoint tcp://localhost:5556

# Store to SQLite database
python3 -m aitoolman monitor \
  --pub-endpoint tcp://localhost:5556 \
  --db-path llm_audit.db
```

The monitor will display the following information:
- Request time statistics (queue time, time to first token, total response time)
- Token usage
- Completion reasons and error messages
- Custom audit events

## 10. Best Practices

### 10.1 Prompt Design
Design prompts using the "**Short-Answer Question**" format. The format is:

1.  **Reading Material**: Long text to be analyzed, reference materials, or historical conversation context.
2.  **Question Background**: Elaborate on the scenario design and main objective; this is the stem of the short-answer question.
3.  **Task Instructions**: Clearly state what needs to be done; this is a sub-question within the short-answer question.
4.  **Specific Requirements**: List the rules, methods, and key points to follow; this includes question type design, scoring points, and hints (in parentheses).
5.  Optional **Output Example**: Write directly in the format that should be output.

Principles of prompt design:
- **Focus on a single task**: Make the AI process only one task or one set of data at a time.
- **Seamless connection**: The end of the prompt should connect naturally to the beginning of the AI's answer; for brief (one or two lines) input content, it can be placed at the very end of the prompt.
- **Language consistency**: Write the prompt in the primary working language; place target language text at the end.
- **Be concise and clear**: Use concise wording, focus more on writing "what to do" and less on "what not to do"; avoid exposing the large language model to irrelevant content.
- **Explicit output format**: Specify a clear output format (e.g., JSON, XML, Markdown) for easy parsing.

### 10.2 Module Design

- **Workflow first**: Avoid having the large language model perform deterministic tasks, or restrict large model behavior only with prompts.
- **Modular design**: Break down complex prompts into multiple modules and global templates, with each module having a single responsibility.
- **Template variables**: Use `{{ variable }}` and other Jinja2 template syntax.
- **Context control**: Minimize the use of context messages; prioritize optimizing prompt quality.
- **Tool description**: Provide clear, specific descriptions and parameter explanations for tools; do not provide useless tools.

### 10.3 Error Handling
```python
try:
    result = await app['module'](...)
    result.raise_for_status()  # Check completion status
    processed_data = result.data
except LLMLengthLimitError as e:
    # Handle length limit: process in segments or switch models
    pass
except LLMApiRequestError as e:
    # API error: retry or directly report error
    pass
except LLMResponseFormatError as e:
    # Response format error: retry or adjust prompt
    pass
```

### 10.4 Performance Optimization
- **Parallelism Configuration**: Set `parallel` according to model quotas
- **Streaming Response**: Use streaming for long text to improve user experience
- **Batch Requests**: For batch/background tasks, use batch requests or dedicated batch interfaces to improve parallelism
- **Resource Management**: Use `ResourceManager` to avoid excessive requests
- **Caching Strategy**: Implement result caching for repeated queries

### 10.5 Debugging Tips
- **Channel Monitoring**: Use `ChannelCollector` to view LLM output in real-time
- **Audit Logs**: Enable monitor to record all requests and responses
- **Step-by-Step Execution**: Test individual tasks first for complex workflows
- **Provider Logs**: Enable `logging.DEBUG` to view raw API interactions
