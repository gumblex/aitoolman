# aitoolman - A Controllable, Transparent LLM Application Framework

* [中文简介](./docs/README.zh.md) | [完整介绍（可用于AI理解）](./docs/quick_start.md) （所有文档以中文为准）
* [One-page Quick Start (for human and AI)](./docs/quick_start.en.md)

## Project Introduction
aitoolman is a developer-oriented LLM (Large Language Model) application framework designed to address common problems in existing frameworks such as **vendor lock-in, unclear workflows, and difficult debugging**. The framework positions AI as a "Toolman", emphasizing that **users directly control all prompts, data flows, and control flows** to help developers quickly build stable, debuggable LLM applications.

### Design Philosophy
1. **AI is not an Agent, but a tool**: AI should act like a "top university intern" that executes clear instructions rather than making independent decisions.
2. **Controllable Workflows**: All program logic is dominated by user code, **with no unexpected operations, no hidden prompts**, and no autonomy granted to the LLM.
3. **Transparent Data Flows**: Users can customize all data sent to the LLM, fully leveraging the unique features of each provider.
4. **Template-Driven Prompts**: Encapsulate prompts as reusable templates in fixed locations, preventing prompts from being scattered across the codebase.

### Core Advantages
- **Vendor-Agnostic**: Directly pass in HTTP headers and custom options, abstract request/response formats, and easily switch between providers.
- **Modular Design**: Components have single responsibilities, making them easy to test and replace.
- **Streaming Support**: Implement real-time data transmission (e.g., thinking processes, response fragments) via `Channel`.
- **Tool Calling**: Support LLM tool invocation as a workflow control mechanism (e.g., calling Tool A represents following Workflow A; or pausing to wait for user confirmation when tool invocation is needed).
- **Microservice Support**: The LLM scheduler can be deployed as an independent service for unified resource management and audit logging.

## Project Structure Design
### Data Flow Sequence

User Input → LLMApplication → LLMModule → LLMClient → LLMProviderManager → 
LLMFormatStrategy → HTTP API → Response Parsing → Post-Processing → Channel Output → User Reception

### Class Call Relationship Diagram

```mermaid
graph TB
    subgraph "User Code"
        User[User Application]
    end
    
    subgraph "Application Layer"
        App[LLMApplication]
        Module[LLMModule]
    end
    
    subgraph "Client Layer"
        Client[LLMClient]
        LocalClient[LLMLocalClient]
        ZmqClient[LLMZmqClient]
    end
    
    subgraph "Microservices"
        ZmqServer[LLMZmqServer]
    end
    
    subgraph "Service Layer"
        ProviderManager[LLMProviderManager]
        ResourceMgr[ResourceManager]
        FormatStrategy[LLMFormatStrategy]
        OpenAIFormat[OpenAICompatibleFormat]
    end
    
    subgraph "Channel Layer"
        Channel[Channel]
        TextFragmentChannel[TextFragmentChannel]
    end
    
    subgraph "Data Models"
        Request[LLMProviderRequest]
        Response[LLMProviderResponse]
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
    
    Channel --> TextFragmentChannel
```

### Data Flow Diagram
```mermaid
flowchart TD
    subgraph "Input Processing"
        USER[User Input] --> TEMPLATE[Template Rendering]
        TEMPLATE --> MESSAGES[Message List]
    end
    
    subgraph "Request Construction"
        MESSAGES --> REQUEST[LLMProviderRequest]
        REQUEST --> FORMAT[Format Strategy]
        FORMAT --> REQ_BODY[Request Body]
    end
    
    subgraph "API Call"
        REQ_BODY --> PROVIDER[ProviderManager]
        PROVIDER --> RESOURCE[Resource Management]
        RESOURCE --> HTTP[HTTP Request]
        HTTP --> STREAM{Streaming?}
        
        STREAM -->|Yes| SSE[SSE Streaming Parsing]
        STREAM -->|No| BATCH[Batch Response Parsing]
    end
    
    subgraph "Response Processing"
        SSE --> FRAGMENT[Write Fragment]
        BATCH --> FULL[Write Full Response]
        
        FRAGMENT --> CHANNEL[TextFragmentChannel]
        FULL --> CHANNEL
        
        CHANNEL --> MULTI_CHANNEL[Multi-Channel Distribution]
        
        MULTI_CHANNEL --> REASONING[Reasoning Channel]
        MULTI_CHANNEL --> RESPONSE[Response Channel]
    end
    
    subgraph "Post-Processing"
        RESPONSE --> POSTPROCESS[Post-Processor]
    end
    
    subgraph "Output"
        POSTPROCESS --> RESULT[LLMModuleResult]
        RESULT --> CONTEXT[Context Update]
        CONTEXT --> NEXT[Next Step Decision]
    end
```

### Data Flow Sequence Diagram
```mermaid
sequenceDiagram
    participant User as User Application
    participant App as LLMApplication
    participant Module as LLMModule
    participant Client as LLMClient
    participant Provider as ProviderManager
    participant Resource as ResourceManager
    participant API as External API
    participant Channel as TextFragmentChannel
    
    User->>App: Create application context
    App->>Module: Initialize module
    App->>Client: Configure client
    
    User->>Module: Invoke module (parameters)
    
    Module->>Module: Render template
    Module->>Module: Build message list
    Module->>Client: request(model, messages, tools)
    
    Client->>Provider: process_request(LLMProviderRequest)
    
    Provider->>Resource: acquire(model resource)
    Resource-->>Provider: Resource granted
    
    Provider->>Provider: Build request body (format strategy)
    Provider->>API: HTTP POST request
    
    alt Streaming Response
        API-->>Provider: SSE streaming data
        Provider->>Channel: write_fragment(fragment)
        Channel-->>User: Real-time fragment output
    else Batch Response
        API-->>Provider: Full response
        Provider->>Channel: write_message(full response)
        Channel-->>User: Full output
    end
    
    Provider->>Resource: release(resource)
    
    Provider-->>Client: LLMProviderResponse
    Client-->>Module: LLMModuleResult
    Module-->>User: Processing result
```
