# 命令行工具
## ZeroMQ 服务
```
usage: python3 -m aitoolman server [-h] [-v] [-c CONFIG]

options:
  -h, --help           show this help message and exit
  -v, --verbose        Print debug log
  -c, --config CONFIG  Path to the TOML config file
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
  -a, --auth AUTH       ZeroMQ ROUTER auth token
  -m, --model MODEL     Model name to use
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
                                      [-p PROMPT] [-M [MEDIA ...]] [--batch] [--no-system]
                                      [--overwrite]

LLM代码修改工具 - 使用AI助手修改代码文件

options:
  -h, --help            show this help message and exit
  -v, --verbose         Print debug log
  -c, --config CONFIG   LLM客户端配置文件路径（TOML格式）
  -z, --zmq-endpoint ZMQ_ENDPOINT
                        ZeroMQ服务端点（如: tcp://localhost:5555）
  -a, --auth AUTH       ZeroMQ认证令牌
  -m, --model MODEL     指定模型名称（如: Kimi-K2, DeepSeek-v3）
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
  --batch               批处理模式（不实时显示思考过程）
  --no-system           不使用系统提示词
  --overwrite           覆盖现有文件（默认情况下会生成.new后缀的文件）

使用示例:
    # 单文件处理
    python3 -m aitoolman code-edit -i input.py -o output.py --llm-config llm_provider.toml

    # 多文件处理（输出到目录）
    python3 -m aitoolman code-edit -i file1.py file2.py -o output_dir --llm-config llm_provider.toml


    # 使用参考文件
    python3 -m aitoolman code-edit -i app.py -o output.py --reference api.py utils.py --llm-config l
lm_provider.toml

    # 批处理模式（不实时显示思考过程）
    python3 -m aitoolman code-edit -i input.py -o output.py --batch --model DeepSeek-v3 --llm-config
 llm_provider.toml

    # 覆盖现有文件
    python3 -m aitoolman code-edit -i input.py -o input.py --overwrite --llm-config llm_provider.tom
l

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
