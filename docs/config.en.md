
# Configuration File Documentation

## 1. Server Configuration (llm_provider.toml)

### 1.1 [server] Section
Configure network ports for the ZeroMQ server.

| Parameter            | Type                  | Default Value     | Description                                                                                                          |
|----------------------|-----------------------|-------------------|----------------------------------------------------------------------------------------------------------------------|
| `zmq_router_rpc`     | String                | None (Required)   | Bind address for the ZeroMQ ROUTER socket, used to handle client requests. Format: `tcp://*:port` or `tcp://IP:port` |
| `zmq_pub_event`      | String                | None (Optional)   | Bind address for the ZeroMQ PUB socket, used to publish audit logs. Format same as above                             |
| `zmq_pub_event_type` | None/"bind"/"connect" | "bind" (Optional) | Bind method for the ZeroMQ PUB socket. Bind to a new port, or connect to an existing port                            |
| `zmq_auth_token`     | String                | None (Optional)   | Any string for authentication                                                                                        |
| `zmq_manage_token`   | String                | None (Optional)   | Management authentication token for client management operations                                                     |


### 1.2 [default] Section
Default configuration inherited by all model configurations.

| Parameter         | Type    | Default Value   | Description                                                                 |
|-------------------|---------|-----------------|-----------------------------------------------------------------------------|
| `timeout`         | Float   | 10              | Network activity timeout duration (seconds), used by models with stream=true  |
| `timeout_batch`   | Float   | 300             | Network activity timeout duration (seconds), used by models with stream=false |
| `max_retries`     | Integer | 0               | Maximum number of retries when request fails                                |
| `parallel`        | Integer | 10              | Default model parallel processing capacity (number of concurrent requests)  |
| `retry_duration`  | Float   | 0.5             | Initial retry interval (seconds)                                             |
| `retry_factor`    | Float   | 1.5             | Retry interval multiplier for exponential backoff algorithm                 |
| `api_type`        | String  | "openai"        | Default API type, available options: "openai", "anthropic"                  |
| `headers`         | Dict    | `{}`            | Default HTTP request headers, universal authentication info can be configured here |
| `body_options`    | Dict    | `{}`            | Default request options such as max_tokens, temperature, etc.               |
| `max_input_tokens` | Integer | None (No limit) | Maximum input token count supported by the model                            |
| `bytes_per_token`  | Float   | 4.0             | Estimated UTF-8 bytes per token, used for token count estimation            |
| `rank_adjust_ratio`| Float   | 0.25            | Weight adjustment parameter for rank weight calculation in model routing    |


### 1.3 [api."Model Name"] Section
Specific configuration for each model, model names can be customized.

| Parameter         | Type    | Default Value                               | Description                                                                           |
|-------------------|---------|---------------------------------------------|---------------------------------------------------------------------------------------|
| `url`             | String  | None (Required)                             | API endpoint URL                                                                      |
| `type`            | String  | Inherited from `[default].api_type`         | API type, available options: "openai", "anthropic"                                    |
| `model`           | String  | None (Required)                             | Model identifier on the provider's side                                               |
| `parallel`        | Integer | Inherited from `[default].parallel`         | Parallel processing capacity for this model                                           |
| `headers`         | Dict    | Inherited from `[default].headers`          | Model-specific HTTP request headers                                                   |
| `timeout`         | Float   | Inherited from `[default].timeout`          | Model-specific timeout duration for stream=true requests                            |
| `timeout_batch`   | Float   | Inherited from `[default].timeout_batch`    | Model-specific timeout duration for stream=false requests                           |
| `body_options`    | Dict    | Inherited from `[default].body_options`     | Default request options for this model, overrides all options in `[default]`. |
| `max_input_tokens` | Integer | Inherited from `[default].max_input_tokens` | Maximum input token count for this model            |
| `bytes_per_token`  | Float   | Inherited from `[default].bytes_per_token`  | Estimated UTF-8 bytes per token for this model      |
| `enable`           | Boolean | true                                        | Set to false to suspend scheduling of this model    |


Additional notes:
* For "openai" format, `headers` should include `Authorization`
* For "anthropic" format, `headers` should include `"X-Api-Key"` and `"anthropic-version" = "2023-06-01"`; `body_options` should include `max_tokens`


### 1.4 [model_tag] Section
Model tag configuration for model routing and grouped scheduling, sorted by priority. Try to define all needed tags here.

Business code can use tags without caring about underlying specific models, which supports flexible switching and routing. Model routing supports passing multiple tags at the same time, the system will automatically match all available models that satisfy all tags, and select the optimal model combined with input token limit and model priority. For example, passing `["code", "multimodal"]` will automatically select multimodal models that support code use cases, passing `["low_cost", "fast"]` will automatically select low cost and low latency models, to adapt to all kinds of cross scenarios.

Original `[model_alias]` config (`alias = model_name`) will be merged into `[model_tag]`. Each alias is a tag, corresponding to a single-item list.

`"Tag Name" = List of strings (supports one level of nesting)`

Key is the tag name, value is a list of model names. List order determines recommendation priority (earlier entries have higher weight). Model names must exist in the `[api]` section. Elements can be strings or string lists. Models in a nested list share the rank weight of their position, and same-weight models will be shuffled randomly. This can be used to rotate usage of identical/similar models across multiple providers.

**Example:**
```toml
[model_tag]
image = ['doubao-seed-2.0-pro']
cheap = ['doubao-seed-2.0-mini','qwen-flash']
fast = ['deepseek-v4-flash','qwen-flash']
low_cost = ["doubao-seed-2.0-mini", "qwen-flash", ["deepseek-v4-flash-a", "deepseek-v4-flash-b", "deepseek-v4-flash-c"]]
# Use models from multiple providers in rotation
"deepseek-v4-flash" = [["deepseek-v4-flash-a", "deepseek-v4-flash-b", "deepseek-v4-flash-c"]]
```

**Weight Calculation:**
Model weights within a tag list are calculated by position, with earlier entries receiving higher weights. All models in a nested list share the rank weight of their position. The `rank_adjust_ratio` parameter controls the weight decay rate, defaulting to 0.25.

**tag:rank Syntax:**

When calling `resolve_model`, you can append a `:rank` suffix to a tag name to dynamically adjust the model priority order within that tag. The syntax is `tag_name:rank`, where `rank` is an integer.

- `rank=1`: No adjustment, keeps original order (equivalent to not using rank)
- `rank=2`: Moves the 1st item to the end, making the 2nd item highest priority
- `rank=N`: Moves the first N-1 items to the end, making the N-th item highest priority
- Ranks exceeding the range automatically wrap around using modulo (e.g., with 4 models, rank=6 is equivalent to rank=2)

Shifting operates at the outer element level of the list, with nested lists treated as single elements that shift together. Models within a nested list share the weight of that position.

**Example:**
```python
# Assuming "code" tag is configured as ["glm-5.1", "qwen3.7-max", "deepseek-v4-pro", "kimi-k2.5"]
# Passing ['code:2'] → qwen3.7-max has highest priority
# Passing ['code:3'] → deepseek-v4-pro has highest priority
# Passing ['code:5'] → Equivalent to rank=1, glm-5.1 has highest priority
# Mixed usage: ['code:2', 'image:3'] each tag adjusts independently
```

Tags without the `:rank` suffix behave exactly as before. The `tag:rank` syntax applies only to tag names, not to direct model names.

### 1.5 Configuration Example

```toml
[server]
zmq_router_rpc = "tcp://*:5555"
zmq_pub_event = "tcp://*:5556"
zmq_auth_token = "YOUR_SECRET_TOKEN"  # Optional auth token
zmq_manage_token = "YOUR_MANAGE_TOKEN"  # Management auth token (optional)

[default]
timeout = 10
timeout_batch = 300
max_retries = 3
parallel = 1
retry_duration = 0.5
retry_factor = 1.5
api_type = "openai"

[model_tag]
"low_cost" = ["doubao-seed-2.0-mini", "qwen-flash", "deepseek-v4-flash"]
"fast" = ["deepseek-v4-flash", "qwen-flash", "doubao-seed-2.0-lite"]
"creative" = ["deepseek-v4-pro", "kimi-k2.5"]
"precise" = ["Doubao-Seed-2.0-pro", "glm-5.1"]
"code" = ["glm-5.1", "qwen3.7-max", "deepseek-v4-pro", "kimi-k2.5"]
"multimodal" = ["Doubao-Seed-2.0-pro", "doubao-seed-2.0-mini", "qwen-vl"]

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

[api."batch-Doubao-Seed-2.0-lite"]
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
| `post_processor`    | String  | None                | Post-processor name, must be registered in the application                  |
| `options`           | Dict    | `{}`                | Default request options such as temperature, max_tokens, etc.               |

### 2.2 [module."Module Name"] Section
Specific configuration for each module, module names can be customized.

| Parameter           | Type    | Default Value                                       | Description                                                                            |
|---------------------|---------|-----------------------------------------------------|----------------------------------------------------------------------------------------|
| `description`       | String  | `''`                                                | Description text for this module                                                       |
| `model`             | String  | Inherited from `[module_default].model`             | Model name or alias used by this module, also supports a list of multiple tags         |
| `stream`            | Boolean | Inherited from `[module_default].stream`            | Whether this module uses streaming output                                              |
| `post_processor`    | String  | Inherited from `[module_default].post_processor`    | Post-processor for this module                                                         |
| `options`           | Dict    | Inherited from `[module_default].options`           | Request options for this module                                                        |
| `template`          | Dict    | `{}`                                                | Template configuration, must include a `user` template, optionally a `system` template |
| `tools`             | Dict    | `{}`                                                | Tool call configuration, see format details below                                      |

### 2.3 [template] Section
Global templates, can be rendered using `LLMApplication.render_template`.
```toml
[template]
"Template1" = "Template content {{text}}"
"Template2" = "Template content {{text2}}"
```

### 2.4 [model_alias] Section
Application-layer model alias configuration, which is used for flexible configuration of specific models.

| Parameter | Type                    | Default Value | Description                                     |
|-----------|-------------------------|---------------|-------------------------------------------------|
| (Alias)   | String / List of String | None          | Maps to the actual model name or tag names list |

### 2.5 Template Syntax
Templates use Jinja2 syntax, supporting variable substitution and basic control structures.

**Available Variables:**
- Global variables defined in `app.vars`
- Keyword arguments passed when calling the module

**Example:**
```jinja2
Article Title: {{title}}
Article Content: <article>{{content}}</article>
```

### 2.6 Tool Call Configuration Format
Tool configuration uses nested TOML structure:

```toml
[module."Module Name"]
tools."Tool Name".type = "function"  # Fixed value
tools."Tool Name".description = "Tool description"

tools."Tool Name".param."Parameter Name".type = "string"  # Parameter types: string, integer, boolean, etc.
tools."Tool Name".param."Parameter Name".description = "Parameter description"
tools."Tool Name".param."Parameter Name".required = true   # Whether the parameter is required
```

### 2.7 Configuration Example
```toml
[module_default]
model = "Fast-Model"  # Using model alias
stream = false

[module.raw]
template.user = """{{content}}"""

[module.summerize]
model = "precise"  # Using model tag
template.user = """Article Title: {{title}}
Article Content: <article>{{content}}</article>
Based on the article content:
1. List arguments and key points according to the article structure
2. List cases in the article and the issues they illustrate
3. Summarize this article"""

[module.creative_writing]
model = "creative"  # Using model tag
template.user = """Please write an article about {{topic}} in {{style}} style, with approximately {{word_count}} words"""
options = {temperature = 0.8, max_tokens = 2000}

[module.code_generator]
model = "code"  # Using model tag
template.user = """Please write a code example for {{functionality}} using {{language}} language, with detailed comments"""

[module.task_adder]
model = ["fast", "low_cost"]  # Use multiple tags to filter models that satisfy both conditions
stream = true
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
server_config = aitoolman.load_config('llm_provider.toml')

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
result = await app['summerize'](
    title="Development of Artificial Intelligence",
    content="Artificial intelligence has made breakthrough progress in recent years..."
)

# Call the task_adder module (supports tool calls)
result = await app['task_adder'](
    user_input="Meeting at 3 PM tomorrow"
)
```

### 3.3 Dynamically Adding Configuration
```python
# Dynamically add module configuration
app.add_module(ModuleConfig(...))

# Dynamically add post-processor
app.add_processor('custom_parser', lambda x: x.split('\n'))
```

## 4. Notes

1. **Model Name Consistency**: The `model` field in `app_prompt.toml` can be either a model name from the `[api]` section of `llm_provider.toml`, a tag defined in the `[model_tag]` section, or an alias defined in the `[model_alias]` section.

2. **Template Variables**: Variables used in templates must be provided during invocation, otherwise rendering will fail.

3. **Streaming Output**: When `stream=true`, output will be sent in fragments via the channel's `write` method.

4. **Tool Calls**: Tool configuration must include complete parameter definitions, otherwise parsing may fail.

5. **Model Routing**: When using `model_tag` for model routing, the system automatically selects the optimal model based on tag weights and model availability. Supports `tag:rank` syntax to dynamically adjust model priority within tags (see Section 1.4 for details). When `messages` are provided, token count is automatically estimated and models exceeding the `max_input_tokens` limit are filtered out.
