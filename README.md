# aitoolman - A Controllable, Transparent LLM Middleware

* [中文简介](./docs/README.zh.md) | [完整介绍（可用于AI理解）](docs/quick_start.md) （所有文档以中文为准）
* [One-page Quick Start (for human and AI)](./docs/quick_start.en.md)

## Project Introduction
aitoolman is a developer-focused LLM (Large Language Model) middleware designed to address pain points such as **vendor lock-in of native SDKs** and **unclear workflows and debugging difficulties of application frameworks**. The library unifies streaming and batch inference interfaces, centrally manages multiple models and providers, and provides simple prompt templates and post-processing pipelines, helping developers quickly build stable, debuggable LLM applications.

## Design Principles
1. **Full User Control**: All prompts, data flows, and control flows are dominated by user code; the LLM serves as an execution tool that follows predefined instructions, with **no unexpected operations, hidden prompts and business logic**.
2. **Transparent & Debuggable Workflows**: All data sent to and received from LLMs is customizable and auditable, making it easy to troubleshoot issues and optimize prompts.
3. **Vendor-agnostic Architecture**: A unified abstraction layer natively supports multiple LLM providers, enabling seamless model switching while leveraging each provider's unique features.
4. **Modular Design**: Components have single responsibilities, making them easy to test, replace, and reuse.
5. **Deployment-friendly Features**: Built-in resource management, error handling, microservice/HTTP integration, monitoring and auditing, ready for production use.

| Dimension      | Native LLM SDK              | aitoolman                                                                    | Full Agent Framework                                         |
|----------------|-----------------------------|------------------------------------------------------------------------------|--------------------------------------------------------------|
| LLM's Role     | Called directly by app code | LLM is a worker executing predefined instructions                            | LLM is an autonomous agent with decision-making capabilities |
| Control        | Managed by app code         | User has full control over workflows                                         | Framework implies hidden control flows                       |
| Prompts        | Assembled in app code       | Developers write templates with full customization                           | Includes many default prompts                                |
| Multi-provider | Per-SDK adaptation          | Native multi-provider abstraction and model routing                          | Usually single platform or plugin-based                      |
| Boundaries     | Low-level API only          | Provider abstraction, application layer, lightweight orchestration           | Built-in vector index, RAG, planning, memory                 |
| Use Cases      | Simple API calls            | Unified access, template-based calls, model routing, on-demand orchestration | Autonomous agents, exploratory applications                  |

## Core Features
- **Multi-provider Unified Abstraction**: Unified data models and per-provider format strategies with a unified streaming/batch inference interface, plus retries and cancellation across LLM vendors.
- **Application Layer Abstraction**: Module configuration, Jinja2 prompt templates, unified module results, post-processing and audit events.
- **Model Routing**: Group models by tags, support multiple providers per model, multi-tag matching and weight calculation, filter by input token limits, and enable fallback across candidates.
- **Streaming Processing**: Streaming output, reasoning content, XML tag distribution and channel demultiplexing.
- **Lightweight Orchestration**: Task chains, concurrency limits, stop control and tool-call transitions.
- **Deployment & Integration**: CLI, OpenAI-compatible HTTP service, ZeroMQ microservice and monitoring for deployment and integration.

## Use Cases
aitoolman is suitable for the following scenarios:

- **Text Processing Pipelines**: Build stable pipelines for classification, summarization, translation and structured extraction with prompt templates and post-processing.
- **Unified Configuration Management**: Centrally manage API keys, model parameters and provider settings across multiple models and providers.
- **Model Routing & Fallback**: Select models by tags and priority, automatically falling back to maintain a balance of cost, speed and quality.
- **Task Orchestration**: Support serial and parallel LLM tasks with concurrency limits and stop control.

## Architecture Overview
1. User Application Layer: Business logic implementation
2. Application Layer (LLMApplication / LLMWorkflow): Module configuration, template rendering, result processing and lightweight orchestration
3. Transport Layer (LLMClient / Channel): Request sending, streaming response transmission, microservice communication
4. Data Interface Layer (ProviderManager): Multi-vendor adaptation, request scheduling, retry, rate limiting and model routing
5. LLM Provider APIs (OpenAI / Anthropic, etc.): Underlying LLM services

## Quick Start
1. `pip install aitoolman`
2. Refer to the [Developer Documentation](docs/quick_start.en.md) for detailed framework docs, API references, and sample code
3. [Configure](docs/config.en.md) LLM provider API keys and model parameters (llm_provider.toml)
4. Write prompt template configurations (app_prompt.toml)
5. Build application logic with LLMApplication; use LLMWorkflow for serial or parallel task chains

## Best Practices

### Prompt Design
Design prompts using the "**Case-study Question**" format in exam papers. The format is:

1.  **Reading Material**: Long text to be analyzed, reference materials, or historical conversation context.
2.  **Question Background**: Elaborate on the scenario design and main objective; this is the stem of the question.
3.  **Task Instructions**: Clearly state what needs to be done; this is a sub-question within the question.
4.  **Specific Requirements**: List the rules, methods, and key points to follow; this includes question type design, scoring points, and hints (in parentheses).
5.  Optional **Output Example**: Write directly in the format that should be output.

Principles of prompt design:
- **Focus on a single task**: Make the AI process only one task or one set of data at a time.
- **Seamless connection**: The end of the prompt should connect naturally to the beginning of the AI's answer; for brief (one or two lines) input content, it can be placed at the very end of the prompt.
- **Language consistency**: Write the prompt in the primary working language; place target language text at the end.
- **Be concise and clear**: Use concise wording, focus more on writing "what to do" and less on "what not to do"; avoid exposing the large language model to irrelevant content.
- **Explicit output format**: Specify a clear output format (e.g., JSON, XML, Markdown) and provide examples.

### Module Design

- **Template variables**: Use `{{ variable }}` and other Jinja2 template syntax.
- **Context control**: Minimize the use of context messages; prioritize optimizing prompt quality.
- **Modular design**: Break down complex prompts into multiple modules and global templates, with each module having a single responsibility.
- **Workflow first**: Avoid having the large language model perform deterministic tasks, or restrict large model behavior only with prompts.
- **Tool description**: Provide clear, specific descriptions and parameter explanations for tools; do not provide useless tools.
