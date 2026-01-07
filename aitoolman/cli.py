import sys
import logging
import asyncio
import argparse

from . import app as _app
from . import util
from . import client as _client
from . import channel as _channel

logger = logging.getLogger(__name__)


async def print_channel_text(
        response_channel: _channel.TextChannel,
        reasoning_channel: _channel.TextChannel
):
    channels = {
        'response': response_channel,
        'reasoning': reasoning_channel,
    }
    reasoning_started = False
    response_started = False
    async for event in _channel.collect_text_channels(
            channels, read_fragments=True
    ):
        if event.channel == 'reasoning' and not reasoning_started:
            print("\n[Thinking]", flush=True)
            reasoning_started = True
        elif (
            ((event.channel == 'reasoning' and event.is_end) or event.channel == 'response') and
            not response_started
        ):
            print("\n[Response]", flush=True)
            response_started = True
        if event.message:
            print(event.message, end="", flush=True)
    print("\n")


DEMO_CLIENT_CONFIG = '''
[module.user_input]
stream = true
output_channel = "stdout"
reasoning_channel = "reasoning"
save_context = true

[module.user_input.template]
user = "{{user_input}}"
'''


async def demo_stream_client(client: _client.LLMClient, model_name: str):
    logger = logging.getLogger("demo_stream_client")
    app_config = util.load_config_str(DEMO_CLIENT_CONFIG)
    app_config['module_default'] = {'model': model_name}

    async with client:
        app = _app.LLMApplication(client, app_config)
        print("Enter your messages (end with a single '.' on a line):")
        lines = []
        while True:
            line = sys.stdin.readline()
            if not line:  # 处理EOF
                break
            stripped_line = line.strip()
            if stripped_line == ".":
                break
            lines.append(line.rstrip("\n"))  # 保留用户输入的换行
        user_input = "\n".join(lines)

        output_task = asyncio.create_task(print_channel_text(
            app.channels['stdout'], app.channels['reasoning']
        ))
        response = await app.user_input(user_input=user_input)
        await output_task
        logger.info("Response: %s", response)


def run_zmqserver(config_file):
    from . import zmqserver
    config = util.load_config(config_file)
    service = zmqserver.LLMZmqServer(config)
    asyncio.run(service.run())


def run_zmqclient(router_endpoint, model_name):
    from . import zmqclient
    client = zmqclient.LLMZmqClient(router_endpoint)
    asyncio.run(demo_stream_client(client, model_name))


def run_localclient(config_file, model_name):
    config = util.load_config(config_file)
    client = _client.LLMLocalClient(config)
    asyncio.run(demo_stream_client(client, model_name))


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

    parser = argparse.ArgumentParser(description="LLM server.")
    subparsers = parser.add_subparsers(dest='subparser_name', required=True, help='Command')
    parser_server = subparsers.add_parser('server')
    parser_server.add_argument(
        "-c", "--config", type=str, default='llm_config.toml',
        help="Path to the TOML config file"
    )
    parser_server.add_argument(
        "-v", "--verbose", action='store_true',
        help="Print debug log"
    )
    parser_client = subparsers.add_parser('client')
    parser_client.add_argument(
        '--router-endpoint', default='tcp://localhost:5555',
        help='ZeroMQ ROUTER endpoint (e.g., tcp://localhost:5555)')
    parser_client.add_argument('-m', '--model-name', required=True, help='Model name to use')
    parser_client.add_argument(
        "-v", "--verbose", action='store_true',
        help="Print debug log"
    )
    parser_local = subparsers.add_parser('local')
    parser_local.add_argument(
        "-c", "--config", type=str, default='llm_config.toml',
        help="Path to the TOML config file"
    )
    parser_local.add_argument('-m', '--model-name', required=True, help='Model name to use')
    parser_local.add_argument(
        "-v", "--verbose", action='store_true',
        help="Print debug log"
    )
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
        run_zmqclient(args.router_endpoint, args.model_name)
    elif args.subparser_name == 'local':
        run_localclient(args.router_endpoint, args.model_name)
    elif args.subparser_name == 'monitor':
        run_monitor(args.pub_endpoint, args.db_path)


if __name__ == "__main__":
    main()
