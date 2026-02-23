# Changelog

## 0.2.0

### 新增
- 安全：新增 ZeroMQ 接口认证功能，支持通过配置 `zmq_auth_token` 实现客户端认证
- 新增工作流模块（workflow）
  - LLMTask：LLM任务基类，包装调用LLM的任务及前后处理逻辑，支持工具调用（Agent）、工具调用作为意图识别
  - LLMWorkflow：任务调度器，支持DAG（有向无环图）拓扑调度、并行任务执行。两种运行模式：
    1. 通过 run，在 LLMTask 中设置 next_task，串行执行工作流
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

### 删除
- 原 LLMModule 实现合并至 LLMApplication，不再有单独一层 LLMModule
- 移除未实现的 `save_context` 配置
- 移除 `LLMApplication.init_all_modules` 方法，初始化时载入所有 ModuleConfig
- 暂时移除未完整实现的 Websocket 支持

## 0.1.x

原始设计
