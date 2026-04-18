import sys
import json
import logging
import asyncio
import argparse
import mimetypes
from typing import Optional, List

from . import app as _app
from . import util
from . import client as _client
from . import channel as _channel
from . import code_editor
from .model import Message, MediaContent, LLMDirectRequest

logger = logging.getLogger(__name__)


async def run_client_session(
    config_file: Optional[str],
    zmq_endpoint: Optional[str],
    auth_token: Optional[str],
    model_name: str,
    prompt_file: Optional[str],
    media_files: Optional[List[str]],
    body_json: Optional[str],
    no_think: bool,
    batch_mode: bool,
    output: Optional[str]
):
    """Unified client session runner supporting both local and ZMQ clients."""

    # 1. Initialize Client
    if config_file:
        config = util.load_config(config_file)
        client = _client.LLMLocalClient(config)
    elif zmq_endpoint:
        from .zmqclient import LLMZmqClient
        client = LLMZmqClient(zmq_endpoint, auth_token)
    else:
        # Should be caught by argparse, but just in case
        raise ValueError("Either --config or --zmq-endpoint must be provided.")

    async with client:
        # Initialize App (config_dict can be empty as we use DirectRequest)
        app = _app.LLMApplication(client, config_dict={})

        # 2. Prepare Prompt Content
        if prompt_file:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
        else:
            # Fallback to stdin if no prompt file provided
            print("Enter your messages (end with a single '.' on a line):")
            lines = []
            while True:
                line = sys.stdin.readline()
                if not line:  # EOF
                    break
                stripped_line = line.strip()
                if stripped_line == ".":
                    break
                lines.append(line.rstrip("\n"))
            prompt_text = "\n".join(lines)

        # 3. Prepare Media Content
        media_content_list = []
        if media_files:
            for f in media_files:
                with open(f, 'rb') as file:
                    file_data = file.read()
                mime_type, _ = mimetypes.guess_type(f)
                if not mime_type:
                    raise ValueError("Unknown file type: " + f)
                media_content_list.append(MediaContent(
                    media_type=mime_type.split('/', 1)[0],
                    data=file_data,
                    mime_type=mime_type
                ))

        # 4. Prepare Options
        options = {}
        if body_json:
            try:
                options = json.loads(body_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON for --body: " + body_json)

        # 5. Construct LLMDirectRequest
        output_channel = _channel.Channel()
        direct_request = LLMDirectRequest(
            model_name=model_name,
            messages=[Message(
                role="user",
                content=prompt_text,
                media_content=media_content_list if media_content_list else None
            )],
            options=options,
            stream=not batch_mode,
            output_channel=output_channel,
        )

        # 6. Handle Output
        output_task = None

        if not no_think:
            output_task = asyncio.create_task(_channel.print_channel_output(
                output_channel,
                topic_names={'reasoning': 'Thinking', 'response': 'Response'},
                header=True
            ))
        elif not batch_mode:
            output_task = asyncio.create_task(_channel.print_channel_output(
                output_channel, {'response': 'response'}))

        # 7. Execute Request
        response = await app.call(direct_request)

        # 8. Cleanup
        if output_task:
            await output_task
        response.raise_for_status()

        # If batch mode and no-think, we haven't printed anything yet (stream=False)
        # Print the full result text now.
        if batch_mode and no_think:
            print(response.text)

        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(response.text)


async def run_code_editor(args):
    if args.config:
        api_config = util.load_config(args.config)
        client = _client.LLMLocalClient(api_config)
    elif args.zmq_endpoint:
        from .zmqclient import LLMZmqClient
        client = LLMZmqClient(args.zmq_endpoint, args.auth)
    else:
        # Should be caught by argparse, but just in case
        raise ValueError("Either --config or --zmq-endpoint must be provided.")

    async with client:
        # 初始化应用
        app_config = util.load_config_str(code_editor.APP_CONFIG)
        llm_app = _app.LLMApplication(client, app_config)
        llm_app.add_processor("extract_code_blocks", code_editor.extract_code_blocks)

        # 处理文件
        result = await code_editor.process_files(
            llm_app=llm_app,
            model_name=args.model,
            reference_files=args.reference,
            input_files=args.input,
            prompt_file=args.prompt,
            output_arg=args.output,
            batch_mode=args.batch,
            overwrite=args.overwrite,
            use_system=(not args.no_system)
        )
        if args.raw_output:
            with open(args.raw_output, 'w', encoding='utf-8') as f:
                f.write(result.text)


def run_zmqserver(config_file):
    from . import zmqserver
    config = util.load_config(config_file)
    service = zmqserver.LLMZmqServer(config)
    asyncio.run(service.run())


def run_monitor(pub_endpoint, db_path=None):
    from . import zmqclient
    if db_path:
        monitor = zmqclient.DBLLMMonitor(pub_endpoint, db_path)
    else:
        monitor = zmqclient.LLMMonitor(pub_endpoint)
    monitor.start()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    parser = argparse.ArgumentParser(description="LLM client/server toolkit.")
    subparsers = parser.add_subparsers(dest='subparser_name', required=True, help='Command')

    # --- Server Command ---
    parser_server = subparsers.add_parser('server')
    parser_server.add_argument(
        "-v", "--verbose", action='store_true',
        help="Print debug log"
    )
    parser_server.add_argument(
        "-c", "--config", type=str, default='llm_provider.toml',
        help="Path to the TOML config file"
    )

    # --- Client Command (Merged local and zmq) ---
    parser_client = subparsers.add_parser('client')
    parser_client.add_argument(
        "-v", "--verbose", action='store_true',
        help="Print debug log"
    )

    # Connection: Mutually exclusive
    connection_group = parser_client.add_mutually_exclusive_group(required=True)
    connection_group.add_argument(
        "-c", "--config", type=str,
        help="Path to the TOML config file (for Local Client)"
    )
    connection_group.add_argument(
        '-z', '--zmq-endpoint', type=str,
        help='ZeroMQ ROUTER endpoint (e.g., tcp://localhost:5555) (for ZMQ Client)'
    )

    parser_client.add_argument(
        '-a', '--auth', help='ZeroMQ ROUTER auth token')
    parser_client.add_argument('-m', '--model', required=True, help='Model name to use')

    # Inputs
    parser_client.add_argument(
        '-p', '--prompt', type=str,
        help='Path to prompt text file'
    )
    parser_client.add_argument(
        '-M', '--media', nargs='*', type=str,
        help='Path to media files (images/videos), e.g., -M img1.jpg img2.mp4'
    )

    # Request Options
    parser_client.add_argument(
        '-b', '--body', type=str,
        help='JSON string for LLMDirectRequest.options'
    )
    parser_client.add_argument(
        '--batch', action='store_true',
        help='Run in batch mode (stream=False)'
    )

    # Output Options
    parser_client.add_argument(
        '--no-think', action='store_true',
        help='Only output stdout channel to standard output (suppress reasoning/thinking)'
    )
    parser_client.add_argument(
        '-o', '--output', type=str,
        help='Path to output text file'
    )

    parser_code_edit = subparsers.add_parser(
        'code-edit',
        description="LLM代码修改工具 - 使用AI助手修改代码文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    使用示例:
    # 单文件处理
    python3 -m aitoolman code-edit -i input.py -o output.py --llm-config llm_provider.toml

    # 多文件处理（输出到目录）
    python3 -m aitoolman code-edit -i file1.py file2.py -o output_dir --llm-config llm_provider.toml

    # 使用参考文件
    python3 -m aitoolman code-edit -i app.py -o output.py --reference api.py utils.py --llm-config llm_provider.toml

    # 批处理模式（不实时显示思考过程）
    python3 -m aitoolman code-edit -i input.py -o output.py --batch --model DeepSeek-v3 --llm-config llm_provider.toml

    # 覆盖现有文件
    python3 -m aitoolman code-edit -i input.py -o input.py --overwrite --llm-config llm_provider.toml

    # 使用远程ZMQ服务
    python3 -m aitoolman code-edit -i input.py -o output.py --zmq-endpoint tcp://localhost:5555 --auth TOKEN --model Code-Model  
            """.strip()
    )
    parser_code_edit.add_argument(
        "-v", "--verbose", action='store_true',
        help="Print debug log"
    )

    # LLM客户端配置（二选一）
    group = parser_code_edit.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-c", "--config", type=str,
        help="LLM客户端配置文件路径（TOML格式）"
    )
    group.add_argument(
        "-z", "--zmq-endpoint", type=str,
        help="ZeroMQ服务端点（如: tcp://localhost:5555）"
    )
    parser_code_edit.add_argument(
        "-a", "--auth", type=str,
        help="ZeroMQ认证令牌"
    )
    parser_code_edit.add_argument(
        "-m", "--model", type=str, default='DeepSeek-v3.2-251201',
        help="指定模型名称（如: Kimi-K2, DeepSeek-v3）"
    )

    parser_code_edit.add_argument(
        "-r", "--reference", type=str, nargs='+', default=[],
        help="参考文件路径（提供上下文，可多个）"
    )
    parser_code_edit.add_argument(
        "-i", "--input", type=str, nargs='+', default=[],
        help="输入文件路径（支持多个文件，如：-i file.py file2.py）"
    )
    parser_code_edit.add_argument(
        "-o", "--output", type=str, required=False,
        help="输出文件路径：可以是单个文件名（单文件）或目录路径（多文件）"
    )
    parser_code_edit.add_argument(
        "-O", "--raw-output", type=str, required=False,
        help="原始输出内容保存文件名"
    )
    parser_code_edit.add_argument(
        "-p", "--prompt", type=str, required=False,
        help="提示词文件路径"
    )
    parser_code_edit.add_argument(
        "--batch", action="store_true",
        help="批处理模式（不实时显示思考过程）"
    )
    parser_code_edit.add_argument(
        "--no-system", action="store_true",
        help="不使用系统提示词"
    )
    parser_code_edit.add_argument(
        "--overwrite", action="store_true",
        help="覆盖现有文件（默认情况下会生成.new后缀的文件）"
    )

    # --- Monitor Command ---
    parser_monitor = subparsers.add_parser('monitor')
    parser_monitor.add_argument(
        "-v", "--verbose", action='store_true',
        help="Print debug log"
    )
    parser_monitor.add_argument(
        '--pub-endpoint', default='tcp://localhost:5556',
        help='ZeroMQ PUB endpoint (e.g., tcp://localhost:5556)')
    parser_monitor.add_argument(
        '--db-path',
        help='SQLite database path for DB monitor')

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger('aitoolman').setLevel(logging.DEBUG)

    if args.subparser_name == 'server':
        run_zmqserver(args.config)
    elif args.subparser_name == 'client':
        asyncio.run(run_client_session(
            config_file=args.config,
            zmq_endpoint=args.zmq_endpoint,
            auth_token=args.auth,
            model_name=args.model,
            prompt_file=args.prompt,
            media_files=args.media,
            body_json=args.body,
            no_think=args.no_think,
            batch_mode=args.batch,
            output=args.output
        ))
    elif args.subparser_name == 'code-edit':
        asyncio.run(run_code_editor(args))
    elif args.subparser_name == 'monitor':
        run_monitor(args.pub_endpoint, args.db_path)


if __name__ == "__main__":
    main()
