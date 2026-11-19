# 命令行工具

## 日志级别

各子命令的 `-v` 均为计数参数，控制日志输出级别：

- 默认：`aitoolman.providers` 包日志级别为 WARNING，其余 `aitoolman` 模块为 INFO
- `-v`：`aitoolman` 包日志级别为 DEBUG，`aitoolman.providers` 包为 INFO
- `-vv`：`aitoolman` 包与 `aitoolman.providers` 包均为 DEBUG

## ZeroMQ 服务
```
usage: python3 -m aitoolman server [-h] [-v] [-c CONFIG]

options:
  -h, --help           show this help message and exit
  -v, --verbose        Print debug log
  -c, --config CONFIG  Path to the TOML config file
```

## HTTP 服务
OpenAI 兼容的 HTTP 服务（chat/completions、models），支持阻塞与 SSE 流式响应，详见 docs/http_server.md。

```
usage: python3 -m aitoolman http [-h] [-v] [-c CONFIG] [--zmq-endpoint ZMQ_ENDPOINT]
                                 [--auth AUTH] [--host HOST] [--port PORT]
                                 [--client-token TOKEN]

options:
  -h, --help            show this help message and exit
  -v, --verbose         Print debug log
  -c, --config CONFIG   Path to the TOML config file
  --zmq-endpoint ZMQ_ENDPOINT
                        ZeroMQ ROUTER endpoint of an existing server (remote mode);
                        omit for direct mode
  --auth AUTH           ZeroMQ auth token (remote mode)
  --host HOST           HTTP bind host (default: 127.0.0.1)
  --port PORT           HTTP bind port (default: 8000)
  --client-token TOKEN  Client Bearer token for authentication, can be specified
                        multiple times
```

其中，未指定 --zmq-endpoint 时为直连模式（同一份配置文件同时提供 HTTP 服务与提供商访问）；指定后为远程模式，连接已有 ZMQ 服务。--client-token 可多次指定，与配置文件 [http].client_tokens 合并。

示例：
```
# 直连模式
python3 -m aitoolman http --client-token sk-xxx

# 远程模式
python3 -m aitoolman http --zmq-endpoint tcp://localhost:5555 --auth TOKEN --client-token sk-xxx
```

## LLM 客户端
```
usage: python3 -m aitoolman client [-h] [-v] (-c CONFIG | -z ZMQ_ENDPOINT) [-a AUTH] -m MODEL
                                   [-p PROMPT] [-M [MEDIA ...]] [-b BODY] [--batch]
                                   [--no-think] [-o OUTPUT]

options:
  -h, --help            show this help message and exit
  -v, --verbose         Print debug log
  -c, --config CONFIG   Path to the TOML config file (for Local Client)
  -z, --zmq-endpoint ZMQ_ENDPOINT
                        ZeroMQ ROUTER endpoint (e.g., tcp://localhost:5555) (for ZMQ Client)
  -a, --auth AUTH       ZeroMQ auth token
  -m, --model MODEL     Model name/tag to use (separate multiple tags with commas)
  -p, --prompt PROMPT   Path to prompt text file
  -M, --media [MEDIA ...]
                        Path to media files (images/videos), e.g., -M img1.jpg img2.mp4
  -b, --body BODY       JSON string for LLMDirectRequest.options
  --batch               Run in batch mode (stream=False)
  --no-think            Only output stdout channel to standard output (suppress
                        reasoning/thinking)
  -o, --output OUTPUT   Path to output text file
```

其中，用 -c 为本地配置文件，-z 为用远程 ZeroMQ 服务端

## 代码修改工具
```
usage: python3 -m aitoolman code-edit [-h] [-v] (-c CONFIG | -z ZMQ_ENDPOINT) [-a AUTH]
                                      -m MODEL [-r REFERENCE [REFERENCE ...]]
                                      [-i INPUT [INPUT ...]] [-o OUTPUT] [-O RAW_OUTPUT]
                                      [-p PROMPT] [-M [MEDIA ...]] [-L SIZE_LIMIT]
                                      [--batch] [--no-system] [--overwrite]

LLM文本文件编辑工具 - 使用AI助手修改/生成文本文件

options:
  -h, --help            show this help message and exit
  -v, --verbose         Print debug log
  -c, --config CONFIG   LLM客户端配置文件路径（TOML格式）
  -z, --zmq-endpoint ZMQ_ENDPOINT
                        ZeroMQ服务端点（如: tcp://localhost:5555）
  -a, --auth AUTH       ZeroMQ auth token
  -m, --model MODEL     指定模型名称/标签（多个标签以英文逗号分隔，如: "code,multimodal"）
  -r, --reference REFERENCE [REFERENCE ...]
                        参考文件路径（提供上下文，可多个）
  -i, --input INPUT [INPUT ...]
                        输入文件路径（支持多个文件，如：-i file.py file2.py）
  -o, --output OUTPUT   输出文件路径：可以是单个文件名（单文件）或目录路径（多文件）
  -O, --raw-output RAW_OUTPUT
                        原始输出内容保存文件名
  -p, --prompt PROMPT   提示词文件路径
  -M, --media [MEDIA ...]
                        图片/视频文件路径（多模态）
  -L, --size-limit SIZE_LIMIT
                        文件大小限制（字节），默认 200KB，设为 0 禁用限制
  --batch               批处理模式（不实时显示思考过程）
  --no-system           不使用系统提示词
  --overwrite           覆盖现有文件（默认情况下会生成.new后缀的文件）

使用示例:
    # 单文件处理
    python3 -m aitoolman code-edit -i input.py -o output.py -c llm_provider.toml

    # 多文件处理（输出到目录）
    python3 -m aitoolman code-edit -i file1.py file2.py -o output_dir -c llm_provider.toml


    # 使用参考文件
    python3 -m aitoolman code-edit -i app.py -o output.py --reference api.py utils.py -c llm_provider.toml

    # 批处理模式（不实时显示思考过程）
    python3 -m aitoolman code-edit -i input.py -o output.py --batch --model DeepSeek-v3 -c llm_provider.toml

    # 覆盖现有文件
    python3 -m aitoolman code-edit -i input.py -o input.py --overwrite -c llm_provider.toml

    # 使用远程ZMQ服务
    python3 -m aitoolman code-edit -i input.py -o output.py --zmq-endpoint tcp://localhost:5555 --au
th TOKEN --model Code-Model
```

## LLM请求监控
```
usage: python3 -m aitoolman monitor [-h] [-v] [--pub-endpoint PUB_ENDPOINT] [--db-path DB_PATH]

options:
  -h, --help            show this help message and exit
  -v, --verbose         Print debug log
  --pub-endpoint PUB_ENDPOINT
                        ZeroMQ PUB endpoint (e.g., tcp://localhost:5556)
  --db-path DB_PATH     SQLite database path for DB monitor
```

## LLM 服务管理
```
usage: python3 -m aitoolman manage [-h] [-v] -z ZMQ_ENDPOINT -a AUTH
                                   {list_models,update_config,change_api_status} ...

Manage LLM ZMQ Server

positional arguments:
  {list_models,update_config,change_api_status}
                        Management action
    list_models         List available models
    update_config       Update full server config
    change_api_status   Enable/disable a model

options:
  -h, --help            show this help message and exit
  -v, --verbose         Print debug log
  -z, --zmq-endpoint ZMQ_ENDPOINT
                        ZeroMQ ROUTER endpoint
  -a, --auth AUTH       ZeroMQ management auth token
```

## 版本
```
usage: python3 -m aitoolman version [-h] [-v]

options:
  -h, --help     show this help message and exit
  -v, --verbose  Print debug log
```

打印当前版本，如 `aitoolman 0.10.0`。
