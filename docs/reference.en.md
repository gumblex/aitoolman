
# aitoolman LLM Framework Documentation

## Class Documentation

### LLMApplication - Application Context
**Description**: The main application class, a container for managing LLM modules, channels, variables and templates, providing unified configuration management and dependency injection.

**Upstream/Downstream Relationships**:
- **Upstream**: User code initializes the application, passing in configurations and clients
- **Downstream**: Manages multiple LLMModule instances and coordinates data flow between them

**Key Methods**:
- `__init__(client: _client.LLMClient, config_dict: Optional[Dict[str, Any]] = None, processors: Optional[Dict[str, Callable[[str], Any]]] = None, channels: Optional[Dict[str, _channel.TextChannel]] = None, context_id: Optional[str] = None)`: Initialize the application context, supports passing custom post-processors and channels
- `init_all_modules()`: Loads all modules from configuration
- `init_module_from_config(module_name, module_config)`: Initializes a single module from configuration
- `__getattr__(name)`: Access modules via properties (lazy loading)
- `add_processor(name, processor)`: Adds a post-processor
- `render_template(template_name, **kwargs)`: Renders a named template
- `add_channel(name, channel)`: Adds a custom channel
- `factory(client, config_dict=None, processors=None, channels=None)`: Creates an application factory function for batch creation of application instances
- `audit_event(event_type: str, **kwargs)`: Sends a custom audit event

### LLMModule / DefaultLLMModule - LLM Module
**Description**: Encapsulates logical units for LLM calls, each module has independent configurations, templates and context.

**Upstream/Downstream Relationships**:
- **Upstream**: LLMApplication manages module lifecycle
- **Downstream**: Calls LLMClient to send requests and process responses

**Key Methods**:
- `__call__(_media=None, **kwargs)`: Executes module invocation, supports multimedia input
- `render_messages(kwargs, media_content)`: Renders a message list containing context
- `render_template(template_name, **kwargs)`: Renders module-specific templates

### LLMClient / LLMLocalClient / LLMZmqClient - LLM Client
**Description**: Abstracts LLM provider calls, supports both local and remote (ZeroMQ) modes.

**Upstream/Downstream Relationships**:
- **Upstream**: LLMModule calls the client to send requests
- **Downstream**: LLMProviderManager handles actual API calls

**Key Methods**:
- `request(model_name, messages, tools, options, stream, context_id, response_channel, reasoning_channel)`: Sends LLM requests, supports streaming output and channel callbacks
- `cancel(request_id)`: Cancels a specified request
- `initialize()`: Initializes the client (can be replaced with async for initialize/close)
- `close()`: Closes the client
- `audit_event(context_id: str, event_type: str, **kwargs)`: Records custom audit events

### LLMProviderManager - Provider Manager
**Description**: Manages multiple LLM providers, handles API calls, resource management, retries and error handling, supports OpenAI and Anthropic format compatibility.

**Upstream/Downstream Relationships**:
- **Upstream**: LLMClient forwards requests
- **Downstream**: Calls specific LLM provider APIs (HTTP/SSE streaming)

**Key Methods**:
- `process_request(request, callback)`: Processes LLM requests and returns async tasks
- `cancel_request(request_id)`: Cancels a specified request
- `cancel_all_requests(client_id, context_id)`: Cancels all requests for a specified client or context
- `initialize()`: Initializes HTTP clients and resources
- `cleanup()`: Cleans up all active requests and resources

### LLMFormatStrategy - Format Strategy
**Description**: Abstracts message format conversion for different LLM providers, with the following concrete implementations:
- `OpenAICompatibleFormat`: Compatible with OpenAI and OpenAI-like APIs (e.g., GPT series, Qwen, Llama, etc.)
- `AnthropicFormat`: Adapts to Anthropic Claude series API formats

**Upstream/Downstream Relationships**:
- **Upstream**: LLMProviderManager invokes format conversion
- **Downstream**: Generates provider-specific HTTP request bodies and parses response content

**Key Methods**:
- `serialize_tool_description(tools_configs)`: Converts tool configurations
- `serialize_message(message)`: Converts message formats
- `make_request_body(request)`: Constructs request bodies
- `parse_batch_response(response, response_data)`: Parses batch responses
- `parse_stream_event(response, event)`: Parses streaming events

### Channel / TextChannel - Channel System
**Description**: Asynchronous message passing channel, supports full message and fragment transmission.

**Upstream/Downstream Relationships**:
- **Upstream**: LLM modules write responses
- **Downstream**: User code reads responses or forwards to other processing logic

**Key Methods**:
- `read_message()`: Reads full messages
- `write_message(message)`: Writes full messages
- `read_fragment()`: Reads message fragments
- `write_fragment(text, end)`: Writes message fragments

### BaseXmlTagFilter / XmlTagToChannelFilter - XML Filter
**Description**: Parses XML tags and distributes content to different channels, enabling structured output.

**Upstream/Downstream Relationships**:
- **Upstream**: LLM responses contain XML tags
- **Downstream**: Distributes content of different tags to corresponding channels

**Key Methods**:
- `write_message(message)`: Processes complete XML messages
- `write_fragment(text, end)`: Processes XML message fragments

### ResourceManager - Resource Manager
**Description**: Manages model parallelism, controls the number of concurrent requests.

**Upstream/Downstream Relationships**:
- **Upstream**: LLMProviderManager acquires resource locks
- **Downstream**: Controls concurrent execution of HTTP requests

**Key Methods**:
- `acquire(key, task_name)`: Acquires resource locks
- `cancel_request(key, task_name)`: Cancels resource requests
- `get_stats(key)`: Retrieves resource statistics
- `get_queue_length(key)`: Gets the current wait queue length for a specified model
- `add_resource(key, capacity)`: Dynamically adds new model resource configurations

### ModuleConfig - Module Configuration
**Description**: Data class that stores module configuration parameters.

**Key Fields**:
- `name`: Module name
- `model`: Model name to use
- `templates`: Template dictionary
- `tools`: Tool configurations
- `stream`: Whether to enable streaming output
- `output_channel`: Output channel
- `reasoning_channel`: Reasoning channel
- `post_processor`: Post-processor function
- `save_context`: Whether to save context
- `options`: Additional model parameters (e.g., temperature, max_tokens, etc.)

