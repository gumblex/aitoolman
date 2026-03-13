import os
import sys
import json
import logging
import asyncio
import argparse
from typing import Optional, List

from . import app as _app
from . import util
from . import client as _client
from . import channel as _channel
from .model import Message, MediaContent, LLMDirectRequest

logger = logging.getLogger(__name__)


def _get_media_type(filename: str) -> str:
    """Simple heuristic to determine media type from filename extension."""
    ext = os.path.splitext(filename)[1].lower()
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}
    if ext in video_extensions:
        return 'video'
    return 'image'


async def stream_stdout(channel):
    while True:
        chunk = await channel.read()
        if chunk is None:  # EOF
            break
        sys.stdout.write(chunk)
        sys.stdout.flush()


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
                m_type = _get_media_type(f)
                media_content_list.append(MediaContent(media_type=m_type, filename=f))

        # 4. Prepare Options
        options = {}
        if body_json:
            try:
                options = json.loads(body_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON for --body: " + body_json)

        # 5. Construct LLMDirectRequest
        direct_request = LLMDirectRequest(
            model_name=model_name,
            messages=[Message(
                role="user",
                content=prompt_text,
                media_content=media_content_list if media_content_list else None
            )],
            options=options,
            stream=not batch_mode,
            output_channel="stdout",
            reasoning_channel=None if no_think else "reasoning"
        )

        # 6. Handle Output
        output_task = None
        collector = None

        if not no_think:
            # Use DefaultTextChannelCollector for full output (Thinking + Response)
            collector = _channel.DefaultTextChannelCollector({
                'Thinking': app.channels['reasoning'],
                'Response': app.channels['stdout']
            })
            output_task = asyncio.create_task(collector.start_listening())
        elif not batch_mode:
            # Stream mode + no-think: Only output stdout channel directly
            output_task = asyncio.create_task(stream_stdout(app.channels['stdout']))

        # 7. Execute Request
        response = await app.call(direct_request)

        # 8. Cleanup
        if collector:
            collector.close()
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
        "-c", "--config", type=str, default='llm_provider.toml',
        help="Path to the TOML config file"
    )
    parser_server.add_argument(
        "-v", "--verbose", action='store_true',
        help="Print debug log"
    )

    # --- Client Command (Merged local and zmq) ---
    parser_client = subparsers.add_parser('client')

    # Connection: Mutually exclusive
    connection_group = parser_client.add_mutually_exclusive_group(required=True)
    connection_group.add_argument(
        "-c", "--config", type=str,
        help="Path to the TOML config file (for Local Client)"
    )
    connection_group.add_argument(
        '-r', '--zmq-endpoint', type=str,
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
    parser_client.add_argument(
        "-v", "--verbose", action='store_true',
        help="Print debug log"
    )

    # --- Monitor Command ---
    parser_monitor = subparsers.add_parser('monitor')
    parser_monitor.add_argument(
        '--pub-endpoint', default='tcp://localhost:5556',
        help='ZeroMQ PUB endpoint (e.g., tcp://localhost:5556)')
    parser_monitor.add_argument(
        '--db-path',
        help='SQLite database path for DB monitor')
    parser_monitor.add_argument(
        "-v", "--verbose", action='store_true',
        help="Print debug log"
    )

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
    elif args.subparser_name == 'monitor':
        run_monitor(args.pub_endpoint, args.db_path)


if __name__ == "__main__":
    main()
