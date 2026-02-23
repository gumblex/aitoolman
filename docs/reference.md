# aitoolman LLM框架文档

## 类文档

### LLMApplication - 应用上下文
**说明**: 主应用类，管理LLM模块、通道、变量和模板的容器，提供统一的配置管理和依赖注入。

**上下游关系**:
- **上游**: 用户代码初始化应用，传入配置和客户端
- **下游**: 管理多个LLMModule实例，协调它们之间的数据流

**主要方法**:
- `__init__(client: _client.LLMClient, config_dict: Optional[Dict[str, Any]] = None, processors: Optional[Dict[str, Callable[[str], Any]]] = None, channels: Optional[Dict[str, _channel.TextFragmentChannel]] = None, context_id: Optional[str] = None)`: 初始化应用上下文，支持传入自定义后处理器和通道
- `__getattr__(name)`: 通过属性访问模块（延迟加载）
- `add_processor(name, processor)`: 添加后处理器
- `render_template(template_name, **kwargs)`: 渲染命名模板
- `add_channel(name, channel)`: 添加自定义通道
- `factory(client, config_dict=None, processors=None, channels=None)`: 创建应用工厂函数，用于批量创建应用实例
- `audit_event(event_type: str, **kwargs)`: 发送自定义审计事件

### LLMModule / DefaultLLMModule - LLM模块
**说明**: 封装LLM调用的逻辑单元，每个模块有独立的配置、模板和上下文。

**上下游关系**:
- **上游**: LLMApplication管理模块生命周期
- **下游**: 调用LLMClient发送请求，处理响应

**主要方法**:
- `__call__(_media=None, **kwargs)`: 执行模块调用，支持多媒体输入
- `render_messages(kwargs, media_content)`: 渲染包含上下文的消息列表
- `render_template(template_name, **kwargs)`: 渲染模块专属模板

### LLMClient / LLMLocalClient / LLMZmqClient - LLM客户端
**说明**: 抽象LLM提供商调用，支持本地和远程（ZeroMQ）两种模式。

**上下游关系**:
- **上游**: LLMModule调用客户端发送请求
- **下游**: LLMProviderManager处理实际API调用

**主要方法**:
- `request(model_name, messages, tools, options, stream, context_id, output_channel, reasoning_channel)`: 发送LLM请求，支持流式输出和通道回调
- `cancel(request_id)`: 取消指定请求
- `initialize()`: 初始化客户端（用 async with 可以替代 initialize/close）
- `close()`: 关闭客户端
- `audit_event(context_id: str, event_type: str, **kwargs)`: 记录自定义审计事件

### LLMProviderManager - 提供商管理器
**说明**: 管理多个LLM提供商，处理API调用、资源管理、重试和错误处理，支持OpenAI和Anthropic格式兼容。

**上下游关系**:
- **上游**: LLMClient转发请求
- **下游**: 调用具体LLM提供商API（HTTP/SSE流式传输）

**主要方法**:
- `process_request(request, callback)`: 处理LLM请求，返回异步任务
- `cancel_request(request_id)`: 取消指定请求
- `cancel_all_requests(client_id, context_id)`: 取消指定客户端或上下文的所有请求
- `initialize()`: 初始化HTTP客户端和资源
- `cleanup()`: 清理所有活跃请求和资源

### LLMFormatStrategy - 格式策略
**说明**: 抽象不同LLM提供商的消息格式转换，提供以下具体实现：
- `OpenAICompatibleFormat`: 兼容OpenAI及类OpenAI风格API（如GPT系列、Qwen、Llama等）
- `AnthropicFormat`: 适配Anthropic Claude系列API格式

**上下游关系**:
- **上游**: LLMProviderManager调用格式转换
- **下游**: 生成特定提供商的HTTP请求体，解析响应内容

**主要方法**:
- `serialize_tool_description(tools_configs)`: 转换工具配置
- `serialize_message(message)`: 转换消息格式
- `make_request_body(request)`: 构建请求体
- `parse_batch_response(response, response_data)`: 解析批量响应
- `parse_stream_event(response, event)`: 解析流式事件

### Channel / TextFragmentChannel - 通道系统
**说明**: 异步消息传递通道，支持完整消息和片段传输。

**上下游关系**:
- **上游**: LLM模块写入响应
- **下游**: 用户代码读取响应或转发到其他处理逻辑

**主要方法**:
- `read_message()`: 读取完整消息
- `write_message(message)`: 写入完整消息
- `read_fragment()`: 读取消息片段
- `write_fragment(text, end)`: 写入消息片段

### BaseXmlTagFilter / XmlTagToChannelFilter - XML过滤器
**说明**: 解析XML标签并将内容分发到不同通道，实现结构化输出。

**上下游关系**:
- **上游**: LLM响应包含XML标签
- **下游**: 将不同标签内容分发到对应通道

**主要方法**:
- `write_message(message)`: 处理完整XML消息
- `write_fragment(text, end)`: 处理XML消息片段

### ResourceManager - 资源管理器
**说明**: 管理模型并行度，控制并发请求数量。

**上下游关系**:
- **上游**: LLMProviderManager获取资源锁
- **下游**: 控制HTTP请求的并发执行

**主要方法**:
- `acquire(key, task_name)`: 获取资源锁
- `cancel_request(key, task_name)`: 取消资源请求
- `get_stats(key)`: 获取资源统计
- `get_queue_length(key)`: 获取指定模型的当前等待队列长度
- `add_resource(key, capacity)`: 动态添加新的模型资源配置

### ModuleConfig - 模块配置
**说明**: 数据类，存储模块的配置参数。

**主要字段**:
- `name`: 模块名称
- `model`: 使用的模型名称
- `templates`: 模板字典
- `tools`: 工具配置
- `stream`: 是否流式输出
- `output_channel`: 输出通道
- `reasoning_channel`: 推理通道
- `post_processor`: 后处理器函数
- `options`: 额外模型参数（如temperature、max_tokens等）
