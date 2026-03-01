# aitoolman - A Controllable, Transparent LLM Application Framework

* [中文简介](./docs/README.zh.md) | [完整介绍（可用于AI理解）](docs/quick_start.md) （所有文档以中文为准）
* [One-page Quick Start (for human and AI)](./docs/quick_start.en.md)

## Project Introduction
aitoolman is a developer-focused LLM (Large Language Model) application framework designed to address pain points of existing frameworks such as **vendor lock-in, unclear workflows, and debugging difficulties**. The framework positions AI as a normal worker, emphasizing **direct user control over all prompts, data flows, and control flows** to help developers quickly build stable, debuggable LLM applications.

## Design Principles
1. **Full User Control**: All prompts, data flows, and control flows are dominated by user code. The LLM only acts as an execution tool, with **no unexpected operations, hidden prompts and business logic**.
2. **Transparent & Debuggable Workflows**: All data sent to and received from LLMs is customizable and auditable, making it easy to troubleshoot issues and optimize prompts.
3. **Vendor-agnostic Architecture**: Unified abstraction layer natively supports multiple LLM providers, enabling seamless model switching while leveraging each provider's unique features.
4. **Modular Design**: Components have single responsibilities, making them easy to test, replace, and reuse.
5. **Production-grade Features**: Built-in resource management, error handling, microservice deployment, and monitoring capabilities, ready for direct production use.

| Dimension              | aitoolman                                                        | Traditional Agent Frameworks                                              |
|------------------------|------------------------------------------------------------------|---------------------------------------------------------------------------|
| LLM's Role             | **LLM is a "worker", only executes predefined instructions**     | LLM is an autonomous "agent" with decision-making capabilities            |
| Control                | User has full control over workflows                             | Framework implies hidden control flows                                    |
| Prompts                | Developers write all prompts with full customization             | Includes many default prompts, high adaptation cost for custom scenarios  |
| Multi-model Adaptation | Natively supports multi-vendor with low switching cost           | Optimized for single platforms, high adaptation cost                      |
| Feature Boundaries     | Focuses on LLM function orchestration, no redundant dependencies | Bulky dependencies with built-in vector indexing, RAG, and other features |
| Use Cases              | Controllable workflow orchestration, batch task processing       | Autonomous agents, exploratory applications                               |

## Core Features
- **Prompt Templating**: Encapsulate prompts as reusable Jinja2 templates for centralized management, avoiding scattered prompt definitions.
- **Flexible Workflow Orchestration**: Supports serial, parallel, and dynamic DAG (Directed Acyclic Graph) workflows to handle complex multistep tasks easily.
- **Native Tool Call Support**: Treats tool calls as a workflow control mechanism, enabling intent recognition or classic function call patterns.
- **Real-time Streaming**: Channel system enables real-time output of response content and reasoning processes, enhancing interactive experiences.
- **Microservice Architecture**: Deploy LLM schedulers as independent services for unified resource management, key isolation, and global auditing.

## Use Cases
aitoolman is suitable for various scenarios requiring controllable, reliable LLM integration:

- **Professional Text Processing**: Summarization, translation, data annotation, structured information extraction.
- **Report Generation**: Generate standardized text reports based on structured data.
- **Controllable Multi-turn Dialogues**: Handle complex user requests through predefined workflows to ensure compliance with business rules.
- **Intelligent Task Orchestration**: Decompose complex business tasks into executable steps, dynamically adjusting workflow branches based on results.
- **Batch Task Processing**: Efficiently process large volumes of standardized tasks in parallel, such as ticket classification, content moderation, and data cleansing.

## Architecture Overview
1. User Application Layer: Business logic implementation
2. Application Layer (LLMApplication / LLMWorkflow): Template management, workflow orchestration, result processing
3. Transport Layer (LLMClient / Channel): Request sending, streaming response transmission, microservice communication
4. Data Interface Layer (ProviderManager): Multi-vendor adaptation, request scheduling, rate limiting & retries
5. LLM Provider APIs (OpenAI / Anthropic, etc.): Underlying LLM services

## Quick Start
1. `pip install aitoolman`
2. Refer to the [Developer Documentation](docs/quick_start.en.md) for detailed framework docs, API references, and sample code
3. Configure LLM provider API keys and model parameters (llm_config.toml)
4. Write prompt template configurations (app_prompt.toml)
5. Build application logic using LLMApplication or LLMWorkflow

## Best Practices

### Prompt Design
Design prompts using the "**Short-Answer Question**" format. The format is:

1.  **Reading Material**: Long text to be analyzed, reference materials, or historical conversation context.
2.  **Question Background**: Elaborate on the scenario design and main objective; this is the stem of the short-answer question.
3.  **Task Instructions**: Clearly state what needs to be done; this is a sub-question within the short-answer question.
4.  **Specific Requirements**: List the rules, methods, and key points to follow; this includes question type design, scoring points, and hints (in parentheses).
5.  Optional **Output Example**: Write directly in the format that should be output.

Principles of prompt design:
- **Focus on a single task**: Make the AI process only one task or one set of data at a time.
- **Seamless connection**: The end of the prompt should connect naturally to the beginning of the AI's answer; for brief (one or two lines) input content, it can be placed at the very end of the prompt.
- **Language consistency**: Write the prompt in the primary working language; place target language text at the end.
- **Be concise and clear**: Use concise wording, focus more on writing "what to do" and less on "what not to do"; avoid exposing the large language model to irrelevant content.
- **Explicit output format**: Specify a clear output format (e.g., JSON, XML, Markdown) for easy parsing.

### Module Design

- **Workflow first**: Avoid having the large language model perform deterministic tasks, or restrict large model behavior only with prompts.
- **Modular design**: Break down complex prompts into multiple modules and global templates, with each module having a single responsibility.
- **Template variables**: Use `{{ variable }}` and other Jinja2 template syntax.
- **Context control**: Minimize the use of context messages; prioritize optimizing prompt quality.
- **Tool description**: Provide clear, specific descriptions and parameter explanations for tools; do not provide useless tools.
