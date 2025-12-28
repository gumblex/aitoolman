# aitoolman LLM框架文档

## 类文档

### LLMApplication - 应用上下文
**说明**: 主应用类，管理LLM模块、通道、变量和模板的容器，提供统一的配置管理和依赖注入。

**上下游关系**:
- **上游**: 用户代码初始化应用，传入配置和客户端
- **下游**: 管理多个LLMModule实例，协调它们之间的数据流

**主要方法**:
- `__init__(client, config_dict, context_id)`: 初始化应用上下文
- `init_all_modules()`: 从配置加载所有模块
- `__getattr__(name)`: 通过属性访问模块（延迟加载）
- `add_processor(name, processor)`: 添加后处理器
- `render_template(template_name, **kwargs)`: 渲染命名模板
- `add_channel(name, channel)`: 添加自定义通道

### LLMModule / DefaultLLMModule - LLM模块
**说明**: 封装LLM调用的逻辑单元，每个模块有独立的配置、模板和上下文。

**上下游关系**:
- **上游**: LLMApplication管理模块生命周期
- **下游**: 调用LLMClient发送请求，处理响应

**主要方法**:
- `__call__(_media=None, **kwargs)`: 执行模块调用
- `render_messages(kwargs, media_content)`: 渲染消息列表
- `render_template(template_name, **kwargs)`: 渲染模块模板

### LLMClient / LLMLocalClient / LLMZmqClient - LLM客户端
**说明**: 抽象LLM提供商调用，支持本地和远程（ZeroMQ）两种模式。

**上下游关系**:
- **上游**: LLMModule调用客户端发送请求
- **下游**: LLMProviderManager处理实际API调用

**主要方法**:
- `request(model_name, messages, tools, options, stream, context_id, response_channel, reasoning_channel)`: 发送请求
- `cancel(request_id)`: 取消请求
- `make_request(...)`: 创建请求对象

### LLMProviderManager - 提供商管理器
**说明**: 管理多个LLM提供商，处理API调用、资源管理、重试和错误处理。

**上下游关系**:
- **上游**: LLMClient转发请求
- **下游**: 调用具体LLM提供商API（HTTP/SSE）

**主要方法**:
- `process_request(request, callback)`: 处理请求
- `cancel_request(request_id)`: 取消指定请求
- `cancel_all_requests(client_id, context_id)`: 取消所有请求

### LLMFormatStrategy - 格式策略
**说明**: 抽象不同LLM提供商的消息格式转换，支持OpenAI和Anthropic格式。

**上下游关系**:
- **上游**: LLMProviderManager调用格式转换
- **下游**: 生成特定提供商的HTTP请求体

**主要方法**:
- `serialize_tool_description(tools_configs)`: 转换工具配置
- `serialize_message(message)`: 转换消息格式
- `make_request_body(request)`: 构建请求体
- `parse_batch_response(response, response_data)`: 解析批量响应
- `parse_stream_event(response, event)`: 解析流式事件

### Channel / TextChannel - 通道系统
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
- `save_context`: 是否保存上下文
- `options`: 其他选项
