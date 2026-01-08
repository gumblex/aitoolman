# aitoolman - 可控、透明的LLM应用框架


## 项目介绍
aitoolman 是一个面向开发者的LLM（大语言模型）应用框架，旨在解决现有框架的**供应商锁定、流程不清晰、调试困难**等痛点。框架将AI定位为“工具人”，强调**用户直接控制所有提示词、数据流、控制流**，帮助开发者快速构建稳定、可调试的LLM应用。


### 设计理念
1. **AI是工具人**：AI应像“985实习生”一样，执行明确指令，而非自主决策。
2. **流程可控**：所有程序逻辑由用户代码主导，**无意外操作、无隐藏提示词**，LLM也无自主权。
3. **数据流透明**：用户可自定义所有发往LLM的数据，充分利用各提供商特色功能。
4. **提示词模板化**：将提示词在固定的地方封装为可复用模板，避免提示词散落各处。


### 核心优势
- **供应商无关**：直接带入HTTP头和自定义选项，可抽象请求/响应格式，轻松切换供应商。
- **模块化设计**：组件职责单一，易于测试和替换。
- **流式支持**：通过`Channel`实现实时数据传输（如思考过程、响应片段）。
- **工具调用**：支持LLM工具调用，作为流程控制机制（如用调用A工具代表走A流程；或在需要调用工具时暂停，等待用户确认）。
- **微服务支持**：可将LLM调度器作为独立服务，统一资源管理和审计日志。

## 项目结构设计
### 数据流顺序

用户输入 → LLMApplication → LLMModule → LLMClient → LLMProviderManager → 
LLMFormatStrategy → HTTP API → 响应解析 → 后处理 → 通道输出 → 用户接收

### 类调用关系图

```mermaid
graph TB
    subgraph "用户代码"
        User[用户应用]
    end
    
    subgraph "应用层"
        App[LLMApplication]
        Module[LLMModule]
    end
    
    subgraph "客户端层"
        Client[LLMClient]
        LocalClient[LLMLocalClient]
        ZmqClient[LLMZmqClient]
    end
    
    subgraph "微服务"
        ZmqServer[LLMZmqServer]
    end
    
    subgraph "服务层"
        ProviderManager[LLMProviderManager]
        ResourceMgr[ResourceManager]
        FormatStrategy[LLMFormatStrategy]
        OpenAIFormat[OpenAICompatibleFormat]
    end
    
    subgraph "通道层"
        Channel[Channel]
        TextChannel[TextChannel]
    end
    
    subgraph "数据模型"
        Request[LLMRequest]
        Response[LLMResponse]
        Message[Message]
        LLMModuleResult[LLMModuleResult]
    end
    
    User --> App
    App --> Module
    Module --> Client
    Module --> Channel
    
    Client --> LocalClient
    Client --> ZmqClient
    LocalClient --> ProviderManager
    ZmqClient --> ZmqServer
    ZmqServer --> ProviderManager
    
    ProviderManager --> ResourceMgr
    ProviderManager --> FormatStrategy
    FormatStrategy --> OpenAIFormat
    
    ProviderManager --> Request
    Request --> Response
    Request --> Message
    Response --> LLMModuleResult
    
    Channel --> TextChannel
```

### 数据流图
```mermaid
flowchart TD
    subgraph "输入处理"
        USER[用户输入] --> TEMPLATE[模板渲染]
        TEMPLATE --> MESSAGES[Message列表]
    end
    
    subgraph "请求构建"
        MESSAGES --> REQUEST[LLMRequest]
        REQUEST --> FORMAT[格式策略]
        FORMAT --> REQ_BODY[请求体]
    end
    
    subgraph "API调用"
        REQ_BODY --> PROVIDER[ProviderManager]
        PROVIDER --> RESOURCE[资源管理]
        RESOURCE --> HTTP[HTTP请求]
        HTTP --> STREAM{流式?}
        
        STREAM -->|是| SSE[SSE流式解析]
        STREAM -->|否| BATCH[批量响应解析]
    end
    
    subgraph "响应处理"
        SSE --> FRAGMENT[片段写入]
        BATCH --> FULL[完整写入]
        
        FRAGMENT --> CHANNEL[TextChannel]
        FULL --> CHANNEL
        
        CHANNEL --> MULTI_CHANNEL[多通道分发]
        
        MULTI_CHANNEL --> REASONING[推理通道]
        MULTI_CHANNEL --> RESPONSE[响应通道]
    end
    
    subgraph "后处理"
        RESPONSE --> POSTPROCESS[后处理器]
    end
    
    subgraph "输出"
        POSTPROCESS --> RESULT[LLMModuleResult]
        RESULT --> CONTEXT[上下文更新]
        CONTEXT --> NEXT[下一步决策]
    end
```

### 数据流时序图
```mermaid
sequenceDiagram
    participant User as 用户应用
    participant App as LLMApplication
    participant Module as LLMModule
    participant Client as LLMClient
    participant Provider as ProviderManager
    participant Resource as ResourceManager
    participant API as 外部API
    participant Channel as TextChannel
    
    User->>App: 创建应用上下文
    App->>Module: 初始化模块
    App->>Client: 配置客户端
    
    User->>Module: 调用模块(参数)
    
    Module->>Module: 渲染模板
    Module->>Module: 构建Message列表
    Module->>Client: request(model, messages, tools)
    
    Client->>Provider: process_request(LLMRequest)
    
    Provider->>Resource: acquire(模型资源)
    Resource-->>Provider: 资源许可
    
    Provider->>Provider: 构建请求体(格式策略)
    Provider->>API: HTTP POST请求
    
    alt 流式响应
        API-->>Provider: SSE流式数据
        Provider->>Channel: write_fragment(片段)
        Channel-->>User: 实时输出片段
    else 批量响应
        API-->>Provider: 完整响应
        Provider->>Channel: write_message(完整)
        Channel-->>User: 完整输出
    end
    
    Provider->>Resource: release(资源)
    
    Provider-->>Client: LLMResponse
    Client-->>Module: LLMModuleResult
    Module-->>User: 处理结果
```

