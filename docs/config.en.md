
# Configuration File Documentation

## 1. Server Configuration (llm_config.toml)

### 1.1 [server] Section
Configure network ports for the ZeroMQ server.

| Parameter         | Type    | Default Value       | Description                                                                 |
|-------------------|---------|---------------------|-----------------------------------------------------------------------------|
| `zmq_router_rpc`  | String  | None (Required)     | Bind address for the ZeroMQ ROUTER socket, used to handle client requests. Format: `tcp://*:port` or `tcp://IP:port` |
| `zmq_pub_event`   | String  | None (Required)     | Bind address for the ZeroMQ PUB socket, used to publish audit logs. Format same as above |

### 1.2 [default] Section
Default configuration inherited by all model configurations.

| Parameter         | Type    | Default Value       | Description                                                                 |
|-------------------|---------|---------------------|-----------------------------------------------------------------------------|
| `timeout`         | Integer | 600                 | HTTP request timeout duration (seconds)                                     |
| `max_retries`     | Integer | 3                   | Maximum number of retries when request fails                                |
| `parallel`        | Integer | 1                   | Default model parallel processing capacity (number of concurrent requests)  |
| `retry_duration`  | Float   | 0.5                 | Initial retry interval (seconds)                                             |
| `retry_factor`    | Float   | 1.5                 | Retry interval multiplier for exponential backoff algorithm                 |
| `api_type`        | String  | "openai"            | Default API type, available options: "openai", "anthropic"                  |
| `headers`         | Dict    | `{}`                | Default HTTP request headers, universal authentication info can be configured here |
| `body_options`    | Dict    | `{}`                | Default request options such as max_tokens, temperature, etc.               |

### 1.3 [api."Model Name"] Section
Specific configuration for each model, model names can be customized.

| Parameter         | Type    | Default Value                          | Description                                                                 |
|-------------------|---------|----------------------------------------|-----------------------------------------------------------------------------|
| `url`             | String  | None (Required)                        | API endpoint URL                                                            |
| `type`            | String  | Inherited from `[default].api_type`    | API type, available options: "openai", "anthropic"                          |
| `model`           | String  | None (Required)                        | Model identifier on the provider's side                                     |
| `parallel`        | Integer | Inherited from `[default].parallel`    | Parallel processing capacity for this model                                  |
| `headers`         | Dict    | Inherited from `[default].headers`     | Model-specific HTTP request headers                                         |
| `timeout`         | Integer | Inherited from `[default].timeout`     | Model-specific timeout duration                                             |
| `body_options`    | Dict    | Inherited from `[default].body_options` | Default request options for this model, such as max_tokens, temperature, etc. |

### 1.4 [model_alias] Section
Model alias configuration to simplify model name usage and enable friendly names in application configurations.

| Parameter | Type    | Default Value | Description                                                                 |
|-----------|---------|---------------|-----------------------------------------------------------------------------|
| `Alias`   | String  | None          | Maps to the actual model name (must have corresponding configuration in [api] section) |

### Configuration Example
```toml
[server]
zmq_router_rpc = "tcp://*:5555"
zmq_pub_event = "tcp://*:5556"

[default]
timeout = 600
max_retries = 3
parallel = 1
retry_duration = 0.5
retry_factor = 1.5
api_type = "openai"

[model_alias]
"Creative-Model" = "DeepSeek-v3.2-251201"
"Precise-Model" = "GPT-4o-mini-251119"
"Fast-Model" = "Doubao-Seed-1.6-flash-250828"
"Cheap-Model" = "Llama-3.1-8B-Instruct"
"Code-Model" = "CodeLlama-70B-Instruct"

[api."Doubao-Seed-1.6"]
url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
type = "openai"
model = "ep-aaa"
headers = {Authorization = "Bearer xxx"}

[api."batch-Doubao-Seed-1.6-flash-250828"]
url = "https://ark.cn-beijing.volces.com/api/v3/batch/chat/completions"
type = "openai"
model = "ep-bbb"
parallel = 100000
headers = {Authorization = "Bearer xxx"}
```

## 2. Application Module Configuration (app_prompt.toml)

### 2.1 [module_default] Section
Default module configuration inherited by all modules.

| Parameter           | Type    | Default Value       | Description                                                                 |
|---------------------|---------|---------------------|-----------------------------------------------------------------------------|
| `model`             | String  | None (Recommended)  | Default model name or alias to use                                          |
| `stream`            | Boolean | false               | Whether to use streaming output                                             |
| `output_channel`    | String  | "stdout"            | Default output channel name                                                 |
| `reasoning_channel` | String  | "reasoning"         | Default reasoning channel name                                              |
| `save_context`      | Boolean | false               | Whether to save conversation context                                        |
| `post_processor`    | String  | None                | Post-processor name, must be registered in the application                  |
| `options`           | Dict    | `{}`                | Default request options such as temperature, max_tokens, etc.               |

### 2.2 [module."Module Name"] Section
Specific configuration for each module, module names can be customized.

| Parameter           | Type    | Default Value                                      | Description                                                                 |
|---------------------|---------|----------------------------------------------------|-----------------------------------------------------------------------------|
| `model`             | String  | Inherited from `[module_default].model`             | Model name or alias used by this module                                      |
| `stream`            | Boolean | Inherited from `[module_default].stream`            | Whether this module uses streaming output                                     |
| `output_channel`    | String  | Inherited from `[module_default].output_channel`    | Output channel for this module                                                |
| `reasoning_channel` | String  | Inherited from `[module_default].reasoning_channel` | Reasoning channel for this module                                             |
| `save_context`      | Boolean | Inherited from `[module_default].save_context`      | Whether this module saves conversation context                               |
| `post_processor`    | String  | Inherited from `[module_default].post_processor`    | Post-processor for this module                                                |
| `options`           | Dict    | Inherited from `[module_default].options`           | Request options for this module                                              |
| `template`          | Dict    | `{}`                                               | Template configuration, must include a `user` template, optionally a `system` template |
| `tools`             | Dict    | `{}`                                               | Tool call configuration, see format details below                            |

### 2.3 [template] Section
Global templates, can be rendered using `LLMApplication.render_template`.
```toml
[template]
"Template1" = "Template content {{text}}"
"Template2" = "Template content {{text2}}"
```

### 2.4 Template Syntax
Templates use Jinja2 syntax, supporting variable substitution and basic control structures.

**Available Variables:**
- Global variables defined in `app.vars`
- Keyword arguments passed when calling the module

**Example:**
```jinja2
Article Title: {{title}}
Article Content: <article>{{content}}</article>
```

### 2.5 Tool Call Configuration Format
Tool configuration uses nested TOML structure:

```toml
[module."Module Name"]
tools."Tool Name".type = "function"  # Fixed value
tools."Tool Name".description = "Tool description"

tools."Tool Name".param."Parameter Name".type = "string"  # Parameter types: string, integer, boolean, etc.
tools."Tool Name".param."Parameter Name".description = "Parameter description"
tools."Tool Name".param."Parameter Name".required = true   # Whether the parameter is required
```

### Configuration Example
```toml
[module_default]
model = "Fast-Model"  # Using model alias
stream = false
output_channel = "stdout"
save_context = false

[module.raw]
template.user = """{{content}}"""

[module.summerize]
model = "Precise-Model"  # Using model alias
template.user = """Article Title: {{title}}
Article Content: <article>{{content}}</article>
Based on the article content:
1. List arguments and key points according to the article structure
2. List cases in the article and the issues they illustrate
3. Summarize this article"""

[module.creative_writing]
model = "Creative-Model"  # Using model alias
template.user = """Please write an article about {{topic}} in {{style}} style, with approximately {{word_count}} words"""
options = {temperature = 0.8, max_tokens = 2000}

[module.code_generator]
model = "Code-Model"  # Using model alias
template.user = """Please write a code example for {{functionality}} using {{language}} language, with detailed comments"""

[module.task_adder]
model = "Fast-Model"  # Using model alias
stream = true
save_context = true
template.user = "As a schedule assistant, you can help users add to-do items. Analyze the user's instruction: if there are specific to-do items, call the tool; if not, ask the user to provide detailed information about the task. User says: {{user_input}}"

tools."add_task".type = "function"
tools."add_task".description = "Add schedule item"

tools."add_task".param."datetime".type = "string"
tools."add_task".param."datetime".description = "Date and time, e.g., 2025-12-31 12:34:56"
tools."add_task".param."datetime".required = false

tools."add_task".param."content".type = "string"
tools."add_task".param."content".description = "To-do item content"
tools."add_task".param."content".required = true
```

## 3. Usage Examples

### 3.1 Loading Configuration
```python
import aitoolman

# Load server configuration
server_config = aitoolman.load_config('llm_config.toml')

# Load application configuration
app_config = aitoolman.load_config('app_prompt.toml')

# Create client
client = aitoolman.LLMLocalClient(server_config)

# Create application
app = aitoolman.LLMApplication(client, app_config)
```

### 3.2 Calling Modules
```python
# Call the summerize module
result = await app.summerize(
    title="Development of Artificial Intelligence",
    content="Artificial intelligence has made breakthrough progress in recent years..."
)

# Call the task_adder module (supports tool calls)
result = await app.task_adder(
    user_input="Meeting at 3 PM tomorrow"
)
```

### 3.3 Dynamically Adding Configuration
```python
# Dynamically add module configuration
app.config['module']['new_module'] = {
    'model': 'Cheap-Model',  # Using model alias
    'template': {
        'user': '{{query}}'
    }
}

# Dynamically add post-processor
app.add_processor('custom_parser', lambda x: x.split('\n'))
```

## 4. Notes

1. **Model Name Consistency**: The `model` field in `app_prompt.toml` can be either a model name from the `[api]` section of `llm_config.toml`, or an alias defined in the `[model_alias]` section.

2. **Channel Management**: Three channels are provided by default:
   - `stdin`: Standard input (non-fragmented mode)
   - `stdout`: Standard output (fragmented mode)
   - `reasoning`: Reasoning output (fragmented mode)

3. **Template Variables**: Variables used in templates must be provided during invocation, otherwise rendering will fail.

4. **Streaming Output**: When `stream=true`, output will be sent in fragments via the channel's `write_fragment` method.

5. **Context Saving**: When `save_context=true`, the module will automatically save conversation history for subsequent calls.

6. **Tool Calls**: Tool configuration must include complete parameter definitions, otherwise parsing may fail.

7. **Model Alias Mapping**: Aliases in `[model_alias]` must map to model names already defined in the `[api]` section. It is recommended to use aliases in app_prompt.toml to facilitate model replacement by end users.
