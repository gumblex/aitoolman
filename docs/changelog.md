# Changelog

## 0.4.2

### 修改
- 完善LLMMonitor日志 

## 0.4.1

### 新增
- ChannelDemux 通道拆分

### 修改
- 修复 PUB 接口 response_message 格式

## 0.4.0

### 修改
- 重构 LLMWorkflow，取消自结果倒推依赖的 DAG 调度模式，实现自开始节点串行执行的逻辑：已知开始的 Task，具体执行路径动态决定（或预先通过创建所有的 Task 直接连接在一起实现固定路径），直到结束。
- 修复 anthropic 格式调用问题

## 0.3.2

### 修改
- 重构 LLMWorkflow.start/run
- 代码编辑器支持图片，支持递归文件夹，增加长文件、异常文件过滤

## 0.3.1

### 修改
- 修复 LLMZmqClient 的结束符

## 0.3.0

### 新增
- 原测试中的代码编辑功能集成到命令行工具子命令 code-edit
- XML 解析支持（不规范的）嵌套 CDATA 块

### 修改
- Channel 类重构，从多通道改为单通道，用 topic='reasoning'/'response' 标识推理、回复内容
- LLMApplication 不再管理通道，如需LLM实时输出需传入 ChannelWriter 实例
- 统一命令行工具中的参数名称

## 0.2.1

### 新增
- TextFragmentChannel.write_whole_message
- LLMModuleRequest.post_processor
- `app_prompt.toml` 支持应用层的模型别名

### 修改
- 配置文件：api.name.body_options 覆盖 default.body_options 所有选项
- 示例客户端（python -maitoolman client）参数修改
- Message.media_content 改为传入 `Optional[List[MediaContent]]`
- LLMDirectRequest/LLMModuleRequest.post_processor 支持直接传入（同步/异步）函数
- 修复 ChannelCollector 乱序问题
- 提高 API 兼容性
- 文档中 LLM 提供商配置文件默认名称改为 `llm_provider.toml`

## 0.2.0

### 新增
- 安全：新增 ZeroMQ 接口认证功能，支持通过配置 `zmq_auth_token` 实现客户端认证
- 新增工作流模块（workflow）
  - Task：任务基类
  - LLMTask：包装调用LLM的任务及前后处理逻辑，支持工具调用（Agent）、工具调用作为意图识别
  - LLMWorkflow：任务调度器，支持DAG（有向无环图）拓扑调度、并行任务执行。两种运行模式：
    1. 通过 run，在 Task 中设置 next_task，串行执行工作流
    2. 用 add_task 生成嵌套任务，用 wait_tasks 等待任务完成
- LLMDirectRequest：应用层 LLM 实际请求对象
- LLMApplication
  - has_template：模板是否存在
  - add_module：从 ModuleConfig 配置动态添加模块
  - render_direct_request：根据 LLMModuleRequest 渲染真实的 LLMDirectRequest 请求
  - call：直接传入 LLMDirectRequest/LLMModuleRequest 发送 LLM 请求
- 模块调用支持直接指定模型名
- 模块配置新增 `description` 字段，支持添加模块描述文本

### 修改
- 请求、响应模型调整：
  - 网络层：LLMProviderRequest（原 LLMRequest），LLMProviderResponse（原 LLMResponse）
  - 应用层：LLMDirectRequest，LLMModuleRequest，LLMModuleResult
- 模块调用方式从属性访问（`app.module_name`）改为下标访问（`app['module_name']`）
- 后处理器支持异步函数
- ZeroMQ 协议优化：新增错误消息类型，优化认证流程与超时处理
- model 中对象根据实际用途，改为 typing.NamedTuple
- LLMModuleResult.call 改为 LLMModuleResult.run_tool_calls，改为返回所有上下文
- LLMFormatStrategy 负责序列化/反序列化 Message 对象

### 删除
- 原 LLMModule 实现合并至 LLMApplication，不再有单独一层 LLMModule
- 移除未实现的 `save_context` 配置
- 移除 `LLMApplication.init_all_modules` 方法，初始化时载入所有 ModuleConfig
- 暂时移除未完整实现的 Websocket 支持

## 0.1.x

原始设计
