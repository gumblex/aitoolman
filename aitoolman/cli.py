import sys
import json
import logging
import asyncio
import argparse

from . import app as _app
from . import util
from . import client as _client
from . import channel as _channel
from .model import Message, MediaContent, LLMDirectRequest

logger = logging.getLogger(__name__)


# --- Subparser definition functions ---

def subparser_server(subparsers):
    subp = subparsers.add_parser('server')
    subp.add_argument(
        "-v", "--verbose", action='count', default=0,
        help="Print debug log"
    )
    subp.add_argument(
        "-c", "--config", type=str, default='llm_provider.toml',
        help="Path to the TOML config file"
    )
    return subp


def subparser_client(subparsers):
    subp = subparsers.add_parser('client')
    subp.add_argument(
        "-v", "--verbose", action='store_true',
        help="Print debug log"
    )

    # Connection: Mutually exclusive
    connection_group = subp.add_mutually_exclusive_group(required=True)
    connection_group.add_argument(
        "-c", "--config", type=str,
        help="Path to the TOML config file (for Local Client)"
    )
    connection_group.add_argument(
        '-z', '--zmq-endpoint', type=str,
        help='ZeroMQ ROUTER endpoint (e.g., tcp://localhost:5555) (for ZMQ Client)'
    )

    subp.add_argument(
        '-a', '--auth', help='ZeroMQ ROUTER auth token')
    subp.add_argument(
        '-m', '--model', required=True, action='append',
        help='Model name/tag to use, can specify multiple for model routing')

    # Inputs
    subp.add_argument(
        '-p', '--prompt', type=str,
        help='Path to prompt text file'
    )
    subp.add_argument(
        '-M', '--media', nargs='*', type=str,
        help='Path to media files (images/videos), e.g., -M img1.jpg img2.mp4'
    )

    # Request Options
    subp.add_argument(
        '-b', '--body', type=str,
        help='JSON string for LLMDirectRequest.options'
    )
    subp.add_argument(
        '--batch', action='store_true',
        help='Run in batch mode (stream=False)'
    )

    # Output Options
    subp.add_argument(
        '--no-think', action='store_true',
        help='Only output stdout channel to standard output (suppress reasoning/thinking)'
    )
    subp.add_argument(
        '-o', '--output', type=str,
        help='Path to output text file'
    )
    return subp


def subparser_code_edit(subparsers):
    subp = subparsers.add_parser(
        'code-edit',
        description="LLM Code Editing Tool - Modify code files with AI assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
# Single file processing
python3 -m aitoolman code-edit -i input.py -o output.py --llm-config llm_provider.toml

# Multi-file processing (output to directory)
python3 -m aitoolman code-edit -i file1.py file2.py -o output_dir --llm-config llm_provider.toml

# Use reference files
python3 -m aitoolman code-edit -i app.py -o output.py --reference api.py utils.py --llm-config llm_provider.toml

# Batch mode (no real-time reasoning display)
python3 -m aitoolman code-edit -i input.py -o output.py --batch --model DeepSeek-v3 --llm-config llm_provider.toml

# Overwrite existing files
python3 -m aitoolman code-edit -i input.py -o input.py --overwrite --llm-config llm_provider.toml

# Use remote ZMQ service
python3 -m aitoolman code-edit -i input.py -o output.py --zmq-endpoint tcp://localhost:5555 --auth TOKEN --model Code-Model  
        """.strip()
    )
    subp.add_argument(
        "-v", "--verbose", action='store_true',
        help="Print debug log"
    )

    # LLM client configuration (one required)
    group = subp.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-c", "--config", type=str,
        help="Path to LLM client configuration file (TOML format)"
    )
    group.add_argument(
        "-z", "--zmq-endpoint", type=str,
        help="ZeroMQ service endpoint (e.g., tcp://localhost:5555)"
    )
    subp.add_argument(
        "-a", "--auth", type=str,
        help="ZeroMQ authentication token"
    )
    subp.add_argument(
        "-m", "--model", type=str, required=True, action='append',
        help="Model name/tag to use, can specify multiple for model routing"
    )

    subp.add_argument(
        "-r", "--reference", type=str, nargs='+', default=[],
        help="Reference file paths (provide context, supports multiple files)"
    )
    subp.add_argument(
        "-i", "--input", type=str, nargs='+', default=[],
        help="Input file paths (supports multiple files, e.g., -i file.py file2.py)"
    )
    subp.add_argument(
        "-o", "--output", type=str, required=False,
        help="Output path: can be a single filename (for single output) or directory path (for multiple outputs)"
    )
    subp.add_argument(
        "-O", "--raw-output", type=str, required=False,
        help="File path to save raw LLM output content"
    )
    subp.add_argument(
        "-p", "--prompt", type=str, required=False,
        help="Path to custom prompt file"
    )
    subp.add_argument(
        '-M', '--media', nargs='*', type=str,
        help='Image/video file paths (for multimodal models)'
    )
    subp.add_argument(
        '-L', '--size-limit', type=int, default=204800,
        help='File size limit in bytes, default 200KB, set to 0 to disable limit'
    )
    subp.add_argument(
        "--batch", action="store_true",
        help="Batch mode (disable real-time reasoning output)"
    )
    subp.add_argument(
        "--no-system", action="store_true",
        help="Do not add built-in system prompt"
    )
    subp.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing files (default: generate new files with .new suffix)"
    )
    return subp


def subparser_monitor(subparsers):
    subp = subparsers.add_parser('monitor')
    subp.add_argument(
        "-v", "--verbose", action='store_true',
        help="Print detailed content (verbose mode)"
    )
    subp.add_argument(
        '--pub-endpoint', default='tcp://localhost:5556',
        help='ZeroMQ PUB endpoint (e.g., tcp://localhost:5556)')
    subp.add_argument(
        '--pub-type', default='bind',
        choices=('bind', 'connect'),
        help='ZeroMQ PUB endpoint type (bind or connect)')
    subp.add_argument(
        '--db-path',
        help='SQLite database path for DB monitor')
    return subp


def subparser_manage(subparsers):
    subp = subparsers.add_parser('manage', description='Manage LLM ZMQ Server')
    subp.add_argument(
        "-v", "--verbose", action='count', default=0,
        help="Print debug log"
    )
    subp.add_argument('-z', '--zmq-endpoint', type=str, required=True, help='ZeroMQ ROUTER endpoint')
    subp.add_argument('-a', '--auth', type=str, help='Management auth token')

    manage_subparsers = subp.add_subparsers(dest='manage_action', required=True, help='Management action')

    # list_models subcommand
    list_p = manage_subparsers.add_parser('list_models', help='List available models')
    list_p.add_argument('-t', '--tag', type=str, action='append',
                        help='Filter models by tag, can specify multiple')

    # update_config subcommand
    update_p = manage_subparsers.add_parser('update_config', help='Update full server config')
    update_p.add_argument('-c', '--config', type=str, required=True, help='Path to new TOML config file')

    # change_api_status subcommand
    status_p = manage_subparsers.add_parser('change_api_status', help='Enable/disable a model')
    status_p.add_argument('-m', '--model', type=str, required=True, help='Model name to change status')
    status_group = status_p.add_mutually_exclusive_group(required=True)
    status_group.add_argument('--enable', action='store_true', help='Enable the model')
    status_group.add_argument('--disable', action='store_true', help='Disable the model')

    return subp


# --- Run functions ---

async def _run_client_session(args):
    """Unified client session runner supporting both local and ZMQ clients."""

    # 1. Initialize Client
    if args.config:
        config = util.load_config(args.config)
        client = _client.LLMLocalClient(config)
    elif args.zmq_endpoint:
        from .zmqclient import LLMZmqClient
        client = LLMZmqClient(args.zmq_endpoint, args.auth)
    else:
        # Should be caught by argparse, but just in case
        raise ValueError("Either --config or --zmq-endpoint must be provided.")

    async with client:
        # Initialize App (config_dict can be empty as we use DirectRequest)
        app = _app.LLMApplication(client, config_dict={})

        # 2. Prepare Prompt Content
        if args.prompt:
            with open(args.prompt, 'r', encoding='utf-8') as f:
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
        if args.media:
            media_content_list = [MediaContent.load_from_path(m) for m in args.media]

        # 4. Prepare Options
        options = {}
        if args.body:
            try:
                options = json.loads(args.body)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON for --body: " + args.body)

        # 5. Construct messages
        messages = [Message(
            role="user",
            content=prompt_text,
            media_content=media_content_list if media_content_list else None
        )]

        # Resolve actual model
        model_names = await client.resolve_model(args.model, messages=messages)
        actual_model = model_names[0]
        logger.info("Resolved model from candidates: %s", ', '.join(
            (f'[{m}]' if i == 0 else m)
            for i, m in enumerate(model_names)
        ))

        # 6. Construct LLMDirectRequest
        output_channel = _channel.Channel()
        direct_request = LLMDirectRequest(
            model_name=actual_model,
            messages=messages,
            options=options,
            stream=not args.batch,
            output_channel=output_channel,
        )

        # 7. Handle Output
        output_task = None

        if not args.no_think:
            output_task = asyncio.create_task(_channel.print_channel_output(
                output_channel,
                topic_names={'reasoning': 'Thinking', 'response': 'Response'},
                header=True
            ))
        elif not args.batch:
            output_task = asyncio.create_task(_channel.print_channel_output(
                output_channel, {'response': 'response'}))

        # 8. Execute Request
        response = await app.call(direct_request)

        # 9. Cleanup
        if output_task:
            await output_task
        response.raise_for_status()

        # If batch mode and no-think, we haven't printed anything yet (stream=False)
        # Print the full result text now.
        if args.batch and args.no_think:
            print(response.text)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(response.text)


async def _run_code_edit_session(args):
    from . import code_editor
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
        # Initialize application
        app_config = util.load_config_str(code_editor.APP_CONFIG)
        llm_app = _app.LLMApplication(client, app_config)
        llm_app.add_processor("extract_code_blocks", code_editor.extract_code_blocks)

        # Process files
        result = await code_editor.process_files(
            llm_app=llm_app,
            model_name=args.model,
            reference_files=args.reference,
            input_files=args.input,
            prompt_file=args.prompt,
            output_arg=args.output,
            batch_mode=args.batch,
            overwrite=args.overwrite,
            use_system=(not args.no_system),
            media_files=args.media,
            file_size_limit=args.size_limit,
        )
        if args.raw_output:
            with open(args.raw_output, 'w', encoding='utf-8') as f:
                f.write(result.text)


async def _run_manage_session(args):
    from .zmqclient import LLMZmqClient
    client = LLMZmqClient(args.zmq_endpoint, args.auth)
    async with client:
        if args.manage_action == 'list_models':
            models = await client.list_models(tag=args.tag)
            print(f"Available models of tag: {args.tag or '(all)'}")
            for model in models:
                print(f"\n- Name: {model.name}")
                print(f"  API Type: {model.api_type}")
                print(f"  Parallel: {model.parallel}")
                print(f"  Tags: {', '.join(f'{k}({v:.2f})' for k, v in model.tags.items())}")
        elif args.manage_action == 'update_config':
            new_config = util.load_config(args.config)
            await client.update_config(new_config)
            logger.info("Config updated successfully")
        elif args.manage_action == 'change_api_status':
            enabled = args.enable
            await client.change_api_status(args.model, enabled)
            logger.info(f"Model {args.model} {'enabled' if enabled else 'disabled'} successfully")


def run_server(args):
    if args.verbose > 1:
        for name in logging.root.manager.loggerDict:
            if name.startswith('aitoolman'):
                logging.getLogger(name).setLevel(logging.DEBUG)
    from . import zmqserver
    config = util.load_config(args.config)
    service = zmqserver.LLMZmqServer(config)
    asyncio.run(service.run())


def run_client(args):
    asyncio.run(_run_client_session(args))


def run_code_edit(args):
    asyncio.run(_run_code_edit_session(args))


def run_monitor(args):
    from . import zmqclient
    if args.db_path:
        monitor = zmqclient.DBLLMMonitor(args.pub_endpoint, args.pub_type, verbose=args.verbose, db_path=args.db_path)
    else:
        monitor = zmqclient.LLMMonitor(args.pub_endpoint, args.pub_type, verbose=args.verbose)
    monitor.start()


def run_manage(args):
    asyncio.run(_run_manage_session(args))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    parser = argparse.ArgumentParser(description="LLM client/server toolkit.")
    subparsers = parser.add_subparsers(dest='subparser_name', required=True, help='Command')

    subparser_server(subparsers)
    subparser_client(subparsers)
    subparser_code_edit(subparsers)
    subparser_monitor(subparsers)
    subparser_manage(subparsers)

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger('aitoolman').setLevel(logging.DEBUG)

    globals()['run_' + args.subparser_name.replace('-', '_')](args)


if __name__ == "__main__":
    main()
