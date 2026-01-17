# aitoolman Framework Developer Documentation

## 1. Framework Overview

### 1.1 Design Philosophy
aitoolman is a developer-focused LLM application framework with the core concept of **"AI as a Toolman"** — enabling LLMs to execute explicit instructions like interns, rather than making autonomous decisions. The framework emphasizes:
- **Controllable Workflow**: All program logic is dominated by user code, with no autonomy for LLMs
- **Transparent Data Flow**: Users can customize all data sent to LLMs and clearly view request/response content
- **Template-based Prompting**: Encapsulate prompts into reusable templates to avoid the chaos of "one-size-fits-all dialog boxes"

### 1.2 Core Data Flow
```
User Code → LLMApplication → LLMModule → LLMClient → ProviderManager → HTTP API
    ↑            ↑              ↑           ↑            ↑
    └─ Result Return ──┴─ Template Rendering ───┴─ Request Construction ──┴─ Format Conversion ──┴─ Response Parsing
```

**Key Flow Nodes**:
1. **Template Rendering**: Jinja2 template + variable substitution → final prompt
2. **Message Construction**: Rendered prompt → list of Message objects
3. **Request Sending**: Message list → LLMRequest → HTTP API
4. **Response Handling**: API response → TextChannel stream → post-processing → final result

### 1.3 Core Components
- **LLMApplication**: Application context that manages modules, templates, variables, channels, and post-processors
- **LLMModule**: Encapsulates the complete flow of "input → template → LLM → output"
- **LLMClient**: Abstracts LLM provider calls, supporting local and remote (ZeroMQ) modes
- **TextChannel**: Asynchronous message channel that supports streaming fragment transmission
- **FormatStrategy**: Abstracts message format conversion for different LLM providers

## 2. Core API Reference

### 2.1 LLMApplication - Application Context

**Purpose**: Container for managing LLM modules, templates, variables, channels, post-processors, and global tools

```python
class LLMApplication:
    def __init__(
        self,
        client: LLMClient,                      # LLM client instance
        config_dict: Optional[Dict[str, Any]] = None,  # Configuration dictionary (loaded from TOML)
        processors: Optional[Dict[str, Callable[[str], Any]]] = None,  # Custom post-processors
        channels: Optional[Dict[str, TextChannel]] = None,  # Custom channels
        context_id: Optional[str] = None      # Context ID
    ) -> None
```

**Key Attributes**:
- `client: LLMClient` - LLM client instance
- `channels: Dict[str, TextChannel]` - Channel dictionary (includes stdin/stdout/reasoning by default)
- `vars: Dict[str, Any]` - Global variables accessible in all templates
- `modules: Dict[str, LLMModule]` - Loaded module instances
- `processors: Dict[str, Callable[[str], Any]]` - Post-processor dictionary
- `global_tools: Dict[str, Any]` - Global tool set referenced by all modules

**Key Methods**:
```python
# Dynamically access modules (lazy loading)
module = app.module_name  # Auto-initializes module from config

# Add post-processor
app.add_processor("custom.parse_xml", parse_xml_function)

# Get post-processor
processor = app.get_processor("builtin.parse_json")

# Render global templates (supports cross-template references)
text = app.render_template("template_name", **variables)

# Add custom channel
app.add_channel("custom", TextChannel(read_fragments=True))

# Create application factory (batch create application instances)
@classmethod
def factory(
    cls,
    client: LLMClient,
    config_dict: Optional[Dict[str, Any]] = None,
    processors: Optional[Dict[str, Callable[[str], Any]]] = None,
    channels: Optional[Dict[str, TextChannel]] = None,
) -> Callable[..., 'LLMApplication']:
    """Create reusable application factory function"""
    pass

# Use factory to create instances
app_factory = LLMApplication.factory(
    client=client,
    config_dict=app_config,
    processors={"custom.parser": custom_parser}
)
app = app_factory()

# Send custom audit event to LLMClient/LLMZmqServer
await app.audit_event(event_type, **kwargs)
```

### 2.2 LLMModule / DefaultLLMModule - LLM Module

**Purpose**: Encapsulates logical units for LLM calls, with each module corresponding to a specific task

```python
class DefaultLLMModule(LLMModule):
    def __init__(self, app: LLMApplication, config: ModuleConfig) -> None
```

**Recommended Calling Method** (dependency injection pattern, supports batch processing):
```python
async def process_task(app: aitoolman.LLMApplication, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Single task processing function"""
    result: LLMModuleResult = await app.module_name(
        _media=MediaContent(...),  # Optional: Multimedia content
        **task_data                # Template variables
    )
    result.raise_for_status()  # Auto-handle exceptions
    return {
        "task_id": task_data["id"],
        "result": result.data if result.data else result.text
    }

# Batch processing example
async with asyncio.TaskGroup() as tg:
    tasks = [tg.create_task(process_task(app_factory(), task)) for task in task_list]
    for task in tasks:
        result = await task
        print(f"Task {result['task_id']} Result: {result['result']}")
```

**ModuleConfig Data Class**:
```python
@dataclass
class ModuleConfig:
    name: str                               # Module name
    model: str                              # Model name or alias to use
    templates: Dict[str, str]              # Template dictionary (must include user, system is optional)
    tools: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Tool configuration
    stream: bool = False                    # Whether to use streaming output
    output_channel: Optional[TextFragmentOutput] = None  # Output channel
    reasoning_channel: Optional[TextFragmentOutput] = None  # Reasoning channel
    post_processor: Optional[str] = None    # Post-processor name (corresponds to key in app.processors)
    save_context: bool = False              # Whether to save conversation context
    options: Dict[str, Any] = field(default_factory=dict)  # Request options (temperature, etc.)
```

**Tool Configuration Notes**:
- Empty values (`{}`) in the module's `tools` dictionary indicate referencing **global tool** definitions
- To override global tool configurations, define complete tool parameters directly in the module

### 2.3 LLMClient - LLM Client

**Purpose**: Abstracts LLM provider calls, supporting local and remote modes

```python
class LLMClient(abc.ABC):
    def __init__(self) -> None:
        ...

    async def initialize(self) -> None:
        """Initialize client (automatically called in __aenter__)"""
        pass

    async def close(self) -> None:
        """Close client (automatically called in __aexit__)"""
        pass
```

**Key Methods**:
```python
# Send request (automatically called by modules, not typically used directly)
request: LLMRequest = await client.request(
    model_name: str,                        # Model name or alias
    messages: List[Message],               # List of messages
    tools: Dict[str, Dict[str, Any]] = None,  # Tool configuration
    options: Optional[Dict[str, Any]] = None,  # Request options
    stream: bool = False,                   # Whether to use streaming
    context_id: Optional[str] = None,      # Context ID
    response_channel: Optional[TextChannel] = None,  # Response channel
    reasoning_channel: Optional[TextChannel] = None   # Reasoning channel
)

# Cancel request
await client.cancel(request_id: str)

# Alternatives to with statement
await client.initialize()
await client.close()
```

**Implementation Classes**:
- `LLMLocalClient`: Local client that directly calls ProviderManager
- `LLMZmqClient`: ZMQ client that connects to remote microservices

**Usage Pattern**:
```python
# As asynchronous context manager (recommended)
async with LLMLocalClient(api_config) as client:
    app = LLMApplication(client, app_config)
    result = await app.module_name(...)
```

### 2.4 Channel / TextChannel - Channel System

**Purpose**: Asynchronous message passing channel that supports complete messages and fragment transmission

```python
class TextChannel(Channel):
    def __init__(self, read_fragments: bool = False) -> None
```

**Key Methods**:
```python
# Write complete message
await channel.write_message("Complete message content")

# Write message fragments (streaming)
await channel.write_fragment("Fragment 1", end=False)
await channel.write_fragment("Fragment 2", end=False)
await channel.write_fragment("Fragment 3", end=True)  # Mark as end

# Read complete message (non-streaming mode)
message: Optional[str] = await channel.read_message()

# Read message fragments (streaming mode)
fragment: Optional[str] = await channel.read_fragment()  # None indicates end
```

**Default Channels**:
- `stdin`: Standard input (non-fragment mode)
- `stdout`: Standard output (fragment mode)
- `reasoning`: Reasoning output (fragment mode)

### 2.5 XmlTagToChannelFilter - XML Tag Routing

**Purpose**: Parses XML tags and routes content to different channels

```python
class XmlTagToChannelFilter(BaseXmlTagFilter):
    def __init__(
        self,
        default_channel: TextChannel,           # Default channel (for unmatched tags)
        channel_map: Dict[str, TextChannel]    # Mapping of tags to channels
    ) -> None
```

**Usage Example**:
```python
# Create channels
response_channel = TextChannel(read_fragments=True)
reasoning_channel = TextChannel(read_fragments=True)

# Create filter
filter = XmlTagToChannelFilter(
    default_channel=response_channel,
    channel_map={"reasoning": reasoning_channel}
)

# Process LLM output (auto-routing)
await filter.write_fragment("<reasoning>Thinking process...</reasoning>", end=False)
await filter.write_fragment("<response>Final answer</response>", end=True)
```

### 2.6 collect_text_channels - Multi-channel Collector

**Purpose**: Listens to multiple TextChannels simultaneously and generates a unified event stream

```python
async def collect_text_channels(
    channels: Dict[str, TextChannel],           # Channel dictionary (name → channel)
    read_fragments: bool = True,               # Whether to read in fragment mode
    timeout: Optional[float] = None            # Timeout in seconds
) -> AsyncGenerator[ChannelEvent, None]
```

**ChannelEvent Structure**:
```python
class ChannelEvent(NamedTuple):
    channel: str                               # Channel name
    message: Any                               # Message content
    is_fragment: bool                          # Whether it's a fragment
    is_end: bool                               # Whether it's the end marker
```

**Usage Example**:
```python
channels = {
    'response': app.channels['stdout'],
    'reasoning': app.channels['reasoning']
}

async for event in collect_text_channels(channels, read_fragments=True):
    if event.channel == 'reasoning':
        print(f"[Reasoning] {event.message}", end="", flush=True)
    elif event.channel == 'response':
        print(f"[Response] {event.message}", end="", flush=True)
```

### 2.7 Data Models (model.py)

**Message - Message Object**:
```python
@dataclass
class Message:
    role: Optional[str] = None                # Role: system/user/assistant/tool
    content: Optional[str] = None            # Text content
    media_content: Optional[MediaContent] = None  # Multimedia content
    reasoning_content: Optional[str] = None  # Reasoning content
    tool_call_id: Optional[str] = None       # Tool call ID
    raw_value: Optional[Dict] = None         # Raw value (directly passed to provider)
```

**LLMRequest - Request Object**:
```python
@dataclass
class LLMRequest:
    client_id: str                            # Client ID
    context_id: Optional[str]                 # Context ID
    request_id: str                          # Request ID
    model_name: str                          # Model name or alias
    messages: List[Message]                  # List of messages
    tools: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Tool configuration
    options: Dict[str, Any] = field(default_factory=dict)  # Request options
    stream: bool = False                     # Whether to use streaming
    response_channel: Optional[TextChannel] = None  # Response channel
    reasoning_channel: Optional[TextChannel] = None  # Reasoning channel
    is_cancelled: bool = False               # Whether request is cancelled
    response: asyncio.Future[LLMResponse] = field(default_factory=asyncio.Future)
```

**LLMResponse - Response Object**:
```python
@dataclass
class LLMResponse:
    client_id: str
    context_id: str
    request_id: str
    model_name: str
    stream: bool
    # Time statistics
    start_time: Optional[float] = None
    queue_time: Optional[float] = None
    queue_length: Optional[int] = None
    time_to_first_token: Optional[float] = None
    total_response_time: Optional[float] = None
    # Response content
    response_text: str = ""
    response_reasoning: str = ""
    response_tool_calls: List[ToolCall] = field(default_factory=list)
    # Completion information
    finish_reason: Optional[str] = None
    error_text: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    # Complete request/response data
    response_message: Optional[Dict[str, Any]] = None

    def raise_for_status(self) -> None:
        """Raise corresponding exceptions based on finish_reason"""
        pass

    def call(self, fn_map: Dict[str, Callable]) -> Dict[str, Any]:
        """
        Execute tool calls

        Args:
            fn_map: Mapping of tool names to functions

        Returns:
            Dictionary of tool call IDs to results

        Raises:
            LLMError: If call fails
            LLMResponseFormatError: If tool not found
        """
        pass
```

**LLMModuleResult - Module Result**:
```python
@dataclass
class LLMModuleResult:
    response_text: str = ""                   # Raw response text
    response_reasoning: str = ""            # Raw reasoning content
    text: str = ""                           # Processed text (raw text before post-processing)
    tool_calls: Dict[str, ToolCall] = field(default_factory=dict)  # Tool calls
    status: FinishReason = FinishReason.stop  # Completion status
    error_text: Optional[str] = None         # Error message
    request_params: Dict[str, Any] = field(default_factory=dict)  # Request parameters
    request_messages: List[Message] = field(default_factory=list)  # Request messages
    response_message: Optional[Dict[str, Any]] = None  # Raw response message
    data: Any = None                         # Post-processing result

    def raise_for_status(self) -> None:
        """Raise corresponding exceptions based on status"""
        pass

    def call(self, fn_map: Dict[str, Callable]) -> Dict[str, Any]:
        """
        Execute tool calls

        Args:
            fn_map: Mapping of tool names to functions

        Returns:
            Dictionary of tool call IDs to results
        """
        pass
```

### 2.8 postprocess.py - Post-processing Tools

**Built-in Utility Functions**:
```python
# Extract XML root tag content (no parsing, only raw content extraction)
xml_content = postprocess.get_xml_tag_content(xml_string, root="result")

# JSON parsing (auto-fixes format errors)
data = postprocess.parse_json(json_string)

# XML parsing (extracts specified root tag and parses into dictionary)
xml_dict = postprocess.parse_xml(xml_string, root="root_tag")
```

**Default Processor Dictionary**:
```python
DEFAULT_PROCESSORS = {
    "builtin.parse_json": parse_json,
}
```

**Custom Processor**:
```python
def custom_xml_processor(text: str) -> Tuple[str, str]:
    """Custom XML parsing processor"""
    content = postprocess.get_xml_tag_content(text, root="classification")
    main_category = re.search(r"<main_category>(.*?)</main_category>", content).group(1).strip()
    sub_category = re.search(r"<sub_category>(.*?)</sub_category>", content).group(1).strip()
    return (main_category, sub_category)

# Register to application
app.add_processor("custom.parse_classification", custom_xml_processor)

# Use in module configuration
# post_processor = "custom.parse_classification"
```

## 3. Prompt Configuration File Format (app_prompt.toml)

### 3.1 File Structure

```toml
# Module Default Configuration (inherited by all modules)
[module_default]
model = "Doubao-Seed-1.6"    # Default model (can be model name or alias, defined in llm_config.toml)
stream = false                # Default non-streaming output
output_channel = "stdout"     # Default output channel
reasoning_channel = "reasoning"  # Default reasoning channel
save_context = false          # Default context saving disabled
options = { temperature = 0.7, max_tokens = 4000 }  # Default request options

# Global Templates (renderable with LLMApplication.render_template)
[template]
"template1" = "Template content {{variable}}"
"template2" = "Another template"

# Global Tool Definitions (referenced by all modules)
[tools]

# Module Definitions (multiple can be defined)
[module."module_name"]
model = "Creative-Model"    # Use model alias, maps to model_alias in llm_config.toml
stream = true                 # Override default streaming setting
template.user = "User template {{input}}"
template.system = "System instruction"
tools = { }                   # Tool configuration (see below)
post_processor = "builtin.parse_json"  # Post-processor name (corresponds to key in app.processors)
save_context = true           # Override default context setting
options = { temperature = 0.5 }  # Override default options
```

### 3.2 Template Syntax

Uses **Jinja2** syntax, supporting:
- **Variable Substitution**: `{{variable_name}}`
- **Control Structures**: `{% if condition %}...{% endif %}`
- **Loops**: `{% for item in list %}...{% endfor %}`
- **Filters**: `{{text|upper}}`
- **Cross-template References**: `{% include %}` directive to reference other templates (including cross-file references), supporting two forms:
  - Reference global template: `{% include 'header' %}` (corresponds to template in [template] block)
  - Reference module template: `{% include 'module/task_planner/user' %}` (format: `module/module_name/template_name`)

**Available Variables**:
- Global variables defined in `app.vars`
- All keyword parameters passed when calling the module

**Example**:
```toml
[module.summarize]
template.user = """
Article Title: {{title}}
Article Content: <article>{{content}}</article>

Please do the following based on the article content:
1. List arguments and key points according to the article structure
2. List cases in the article and the problems they illustrate
3. Summarize the article

Output Format:
<response>
<point>Argument 1</point>
<point>Argument 2</point>
<case>Case 1</case>
<case>Case 2</case>
<summary>Summary</summary>
</response>
"""
```

**Template Reference Example**:
```
# Global template header
[template]
"header" = "【Task ID: {{task_id}}】Processing started"

[module.task_planner]
template.user = """
{% include 'header' %}
User Instruction: {{user_input}}
Please analyze the instruction and call appropriate tools
"""
```

### 3.3 Tool Call Configuration

Tool configuration uses TOML nested structure:

```toml
[module."module_name"]
tools."tool_name".type = "function"  # Fixed value
tools."tool_name".description = "Tool description"

tools."tool_name".param."param_name".type = "string"  # Parameter type: string/integer/boolean
tools."tool_name".param."param_name".description = "Parameter description"
tools."tool_name".param."param_name".required = true   # Whether required
```

**Global Tool Configuration**:
Define tools shared by all modules in the `[tools]` block, with the same structure as module-specific tools:
```toml
[tools."tool_name"]
type = "function"
description = "Tool function description"
param."param_name1".type = "string/integer/boolean"
param."param_name1".description = "Parameter description"
param."param_name1".required = true
param."param_name2".type = "string"
param."param_name2".description = "Optional parameter"
param."param_name2".required = false
```

**Module Reference to Global Tools**:
In the module's `tools` configuration, use an empty dictionary `{}` to reference global tool definitions:
```toml
[module."task_planner"]
tools."add_task" = {}
```

**Complete Example**:
```toml
[module.task_planner]
model = "Fast-Model"  # Use fast inference model alias
stream = true
save_context = true
template.user = """
As a schedule assistant, analyze the user's instruction:
- If there are specific to-do items, call the add_task tool
- If not, ask the user for more details

User says: {{user_input}}
"""

tools.add_task.type = "function"
tools.add_task.description = "Add schedule item"

tools.add_task.param.datetime.type = "string"
tools.add_task.param.datetime.description = "Date and time, e.g., 2025-12-31 12:34:56"
tools.add_task.param.datetime.required = false

tools.add_task.param.content.type = "string"
tools.add_task.param.content.description = "To-do item content"
tools.add_task.param.content.required = true
```

## 4. Example Configuration Files

### 4.1 Complete app_prompt.toml Example

```toml
# Module Default Configuration
[module_default]
model = "Fast-Model"         # Default to fast inference model
stream = false
output_channel = "stdout"
reasoning_channel = "reasoning"
save_context = false
options = { temperature = 0.7, max_tokens = 4000 }

# Global Templates
[template]
"greeting" = "Hello, {{name}}! Welcome to the aitoolman framework."

# Raw Text Processing Module
[module.raw]
template.user = "{{content}}"

# Article Summarization Module
[module.summarize]
model = "Creative-Model"     # Use creative model
template.user = """
Article Title: {{title}}
Article Content: <article>{{content}}</article>

Please do the following based on the article content:
1. List arguments and key points according to the article structure
2. List cases in the article and the problems they illustrate
3. Summarize the article

Output Format:
<response>
<point>Argument 1</point>
<point>Argument 2</point>
<case>Case 1</case>
<case>Case 2</case>
<summary>Summary</summary>
</response>
"""
post_processor = "builtin.parse_json"

# Schedule Planning Module (supports tool calls)
[module.task_planner]
model = "Fast-Model"         # Use fast inference model
stream = true
save_context = true
template.user = """
As a schedule assistant, analyze the user's instruction:
- If there are specific to-do items, call the add_task tool
- If not, ask the user for more details

User says: {{user_input}}
"""

tools.add_task.type = "function"
tools.add_task.description = "Add schedule item"

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
  "field1": "value1",
  "field2": "value2"
}
</response>
"""
post_processor = "builtin.parse_json"

# Multi-turn Dialogue Module
[module.chat]
model = "Doubao-Seed-1.6"
stream = true
save_context = true
template.user = "{{message}}"
```

## 5. Example Application Structure

### 5.1 Project Directory Structure

```
my_llm_app/
├── config/
│   ├── llm_config.toml          # API configuration (models, keys)
│   └── app_prompt.toml          # Prompt configuration (modules, templates)
├── src/
│   ├── __init__.py
│   ├── main.py                  # Application entry point
│   ├── processors.py            # Custom post-processors
│   └── workflows.py             # Workflow logic
└── requirements.txt
```

### 5.2 Application Entry Point (main.py)

```python
import re
import asyncio
import aitoolman
from typing import Dict, Any, List, Tuple

# Custom Post-processor
def parse_classification(text: str) -> Tuple[str, str]:
    content = aitoolman.postprocess.get_xml_tag_content(text, root="classification")
    main_category = re.search(r"<main_category>(.*?)</main_category>", content).group(1).strip()
    sub_category = re.search(r"<sub_category>(.*?)</sub_category>", content).group(1).strip()
    return (main_category, sub_category)

# Single Task Processing Function
async def process_ticket(app: aitoolman.LLMApplication, ticket: Dict[str, Any]) -> Dict[str, Any]:
    """Process ticket classification"""
    result = await app.classify_ticket(**ticket)
    result.raise_for_status()  # Auto-handle exceptions
    return {
        "ticket_id": ticket["ticket_id"],
        "main_category": result.data[0],
        "sub_category": result.data[1]
    }

async def main():
    # 1. Load configurations
    api_config = aitoolman.load_config("config/llm_config.toml")
    app_config = aitoolman.load_config("config/app_prompt.toml")

    # 2. Create and initialize client
    async with aitoolman.LLMLocalClient(api_config) as client:
        # 3. Create application factory
        app_factory = aitoolman.LLMApplication.factory(
            client=client,
            config_dict=app_config,
            processors={
                "custom.parse_classification": parse_classification
            }
        )

        # 4. Batch process tickets
        tickets: List[Dict[str, Any]] = [
            {"ticket_id": "123", "content": "Ticket content 1", "location": "Location 1"},
            {"ticket_id": "456", "content": "Ticket content 2", "location": "Location 2"}
        ]

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(process_ticket(app_factory(), ticket)) for ticket in tickets]
            for task in tasks:
                result = await task
                print(f"Ticket {result['ticket_id']} Classification: {result['main_category']} / {result['sub_category']}")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(main())
```

### 5.3 Custom Post-processors (processors.py)

```python
import re
from typing import Tuple, List
import aitoolman.postprocess

def parse_xml_classification(xml_str: str) -> Tuple[str, str]:
    """Parse classification result XML"""
    content = aitoolman.postprocess.get_xml_tag_content(xml_str, root="result")
    if not content:
        raise ValueError("Invalid classification result")

    main_category = re.search(r"<main>(.*?)</main>", content, re.DOTALL).group(1).strip()
    sub_category = re.search(r"<sub>(.*?)</sub>", content, re.DOTALL).group(1).strip()
    return (main_category, sub_category)

def extract_keywords(text: str) -> List[str]:
    """Extract text keywords"""
    # Implement custom keyword extraction logic
    return ["keyword1", "keyword2"]
```

### 5.4 Complex Workflow Example (workflows.py)

```python
import aitoolman
from typing import Dict, Any, List

class DocumentPipeline:
    """Document processing pipeline"""

    def __init__(self, app: aitoolman.LLMApplication):
        self.app = app

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Sentiment analysis"""
        result = await self.app.sentiment_analysis(text=text)
        result.raise_for_status()
        return result.data

    async def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Entity extraction"""
        result = await self.app.entity_extraction(text=text)
        result.raise_for_status()
        return result.data.get("entities", [])

    async def generate_summary(
        self,
        title: str,
        content: str,
        max_length: int = 300
    ) -> str:
        """Generate summary"""
        result = await self.app.summarize(
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
        """Complete document processing workflow"""
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

# Usage Example
async def run_pipeline():
    api_config = aitoolman.load_config("config/llm_config.toml")
    app_config = aitoolman.load_config("config/app_prompt.toml")

    async with aitoolman.LLMLocalClient(api_config) as client:
        app = aitoolman.LLMApplication(client, app_config)
        pipeline = DocumentPipeline(app)

        doc = {
            "title": "AI Technology Breakthrough",
            "content": "Latest research shows that AI has made significant progress in image recognition..."
        }

        result = await pipeline.process_document(doc)
        print(f"Summary: {result['summary']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Entities: {result['entities']}")
```

### 5.5 Configuration Example (llm_config.toml)

```toml
# Server Configuration (ZMQ Mode)
[server]
zmq_router_rpc = "tcp://*:5555"
zmq_pub_event = "tcp://*:5556"

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

## 6. Best Practices

### 6.1 Configuration Management
- **Environment Separation**: Prepare different `llm_config.toml` files for development/testing/production
- **Secret Management**: Load llm_config via configuration files, databases, etc.
- **Version Control**: Include `app_prompt.toml` in Git to track prompt iteration history
- **Template Reuse**: Use global templates and cross-references to reduce duplicate prompts and improve maintainability
- **Tool Reuse**: Define common tools as global tools to avoid duplicate configurations between modules
- **Model Aliases**: Unified model mapping via `model_alias` for easy model switching by end users; modify only llm_config.toml when switching models
- **Layered Configuration**: Set a common model in module_default, and override with aliases in specific modules as needed

### 6.2 Error Handling
```python
import logging
import aitoolman

try:
    result = await app.module_name(**params)
    result.raise_for_status()  # Auto-handle exceptions

    if result.status == aitoolman.FinishReason.tool_calls:
        # Execute tool calls
        tool_results = result.call({
            "add_task": add_task_function,
            "query_task": query_task_function
        })
        print(f"Tool Call Results: {tool_results}")
    else:
        print(f"Processing Result: {result.data if result.data else result.text}")
except aitoolman.LLMLengthLimitError:
    print("Response exceeded length limit; optimize prompt or adjust max_tokens")
except aitoolman.LLMContentFilterError:
    print("Content triggered filter; please check input")
except aitoolman.LLMResponseFormatError as e:
    print(f"Response format error: {e}")
except Exception as e:
    print(f"System error: {e}")
    # Record detailed logs
    logging.exception("Task processing failed")
```

### 6.3 Performance Optimization
- **Connection Reuse**: Share the same `LLMClient` across multiple `LLMApplication` instances
- **Resource Limits**: Set the `parallel` parameter appropriately based on provider quotas (typically 1-3)
- **Streaming Output**: Use custom Channels to output workflow status; use `stream=true` for long texts to output content in real time and improve user experience
- **Batch Processing**: Use `asyncio.TaskGroup` to process multiple tasks in parallel and increase throughput
- **Caching Strategy**: Cache results of repeated requests to reduce unnecessary LLM calls
- **Model Selection**: Define and select appropriate model aliases based on task type, then configure suitable specific models for each alias

## 7. Summary

The aitoolman framework, through clear architecture design and flexible configuration system, enables developers to:

1. **Rapid Construction**: Define LLM applications via TOML configuration without writing complex code
2. **Fine-grained Control**: Complete control over data flow to build transparent and debuggable AI applications
3. **Easy Extension**: Support custom processors, channels, and format strategies to adapt to different business needs
4. **Flexible Deployment**: Support local invocation and microservice architecture to adapt to different scenario requirements

By understanding the collaborative relationship between `LLMApplication`, `LLMModule`, `LLMClient`, and `TextChannel`, developers can efficiently build stable and maintainable LLM applications.
