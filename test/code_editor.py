#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import sys
import asyncio
import logging
import argparse
import mimetypes
from typing import List
from pathlib import Path

import aitoolman
import aitoolman.cli
import aitoolman.zmqclient

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ------------------------------
# 后处理器：提取 <output> 标签内容
# ------------------------------
def extract_code_block(text: str) -> str:
    """
    从LLM输出中提取 <output></output> 包围的代码块
    如果没有找到，返回原始文本（可能LLM直接输出了代码）
    """
    match = re.search(r'<output><!\[CDATA\[(.*)]]></output>', text, re.DOTALL)
    if match:
        return match.group(1)
    # 如果没有找到标签，可能LLM直接输出了代码
    logger.warning("未找到 <output> 标签，返回原始文本")
    return text


def detect_file_language(filename):
    mimetype, encoding = mimetypes.guess_file_type(filename)
    if not mimetype:
        return os.path.splitext(filename)[1].lstrip('.')
    text_type = mimetype.split('/', 1)[-1]
    if text_type.startswith('x-'):
        text_type = text_type[2:]
    if text_type in ('csrc', 'chdr'):
        return 'c'
    elif text_type in ('c++src', 'c++hdr'):
        return 'c++'
    return text_type


APP_CONFIG = '''
[module_default]
model = "Code-Model"
output_channel = "stdout"
reasoning_channel = "reasoning"

[module.code_edit]
template.user = """{% if references %}
# 参考文件
{% for ref in references %}
<file name="{{ ref.filename }}">{{ref.content}}</file>
{% endfor %}
{% endif %}
{% if input_content -%}
# 当前文件
<file name="{{ input_filename }}">{{input_content}}</file>
{%- endif %}
# 要求
{{user_instruction}}
{% if use_system -%}
将修改后的文件用 <output><![CDATA[文件内容]]></output> 包围
{% endif %}"""
post_processor = "extract_code"
'''


# ------------------------------
# 异步处理函数
# ------------------------------
async def process_single_file(
        app: aitoolman.LLMApplication,
        model_name: str,
        reference_files: List[str],
        input_file: str,
        prompt_file: str,
        output: str,
        input_language: str,
        batch_mode: bool,
        use_system: bool = True
) -> None:
    """处理单个文件"""
    logger.info(f"使用模型: {model_name}")

    logger.info(f"开始处理: {input_file}")

    references = []
    for ref_file in reference_files:
        file_path = Path(ref_file)
        with open(ref_file, 'r', encoding='utf-8') as f:
            content = f.read()
        references.append({
            'filename': str(file_path),
            'content': content,
            'language': detect_file_language(file_path.name)
        })

    language = input_language or ''
    if input_file:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_content = f.read()
        if not input_language:
            language = detect_file_language(input_file)
    else:
        input_content = read_user_input("请输入原始代码")

    user_instruction = None
    if prompt_file:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            user_instruction = f.read()
    if not user_instruction:
        user_instruction = read_user_input("请输入修改指令")

    channel_collector = aitoolman.DefaultTextChannelCollector({
        'Thinking': app.channels['reasoning'],
        'Response': app.channels['stdout']
    })
    output_task = asyncio.create_task(channel_collector.start_listening())

    template_params = {
        'user_instruction': user_instruction, 'language': language,
        'input_filename': input_file, 'input_content': input_content,
        'references': references, 'use_system': use_system
    }
    result = await app.call(aitoolman.LLMModuleRequest(
        module_name='code_edit',
        template_params=template_params,
        model_name=model_name,
        stream=(not batch_mode)
    ))
    result.raise_for_status()
    channel_collector.close()
    await output_task
    # 保存结果
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(result.data)
        logger.info(f"处理完成: {output}")


def read_user_input(prompt) -> str:
    print(prompt + "（结束后输入单独的一行 . ）")
    lines = []
    while True:
        line = sys.stdin.readline()
        if not line:  # 处理EOF
            break
        if line.rstrip('\r\n') == ".":
            break
        lines.append(line.rstrip("\n"))  # 保留用户输入的换行
    return "\n".join(lines).strip()


# ------------------------------
# 主程序
# ------------------------------
async def main(args):
    # 确定模型名称
    model_name = args.model
    if not model_name:
        # 从LLM配置文件中获取默认模型
        if args.llm_config:
            llm_config = aitoolman.load_config(args.llm_config)
            model_name = llm_config.get('model_default', {}).get('model')

    if not model_name:
        raise ValueError("未指定模型名称，且无法从配置中推断")

    # 创建客户端
    if args.zmq_endpoint:
        client = aitoolman.zmqclient.LLMZmqClient(args.zmq_endpoint, args.auth)
    else:
        api_config = aitoolman.load_config(args.llm_config)
        client = aitoolman.LLMLocalClient(api_config)

    async with client:
        app_config = aitoolman.load_config_str(APP_CONFIG)
        app = aitoolman.LLMApplication(client, app_config)
        app.add_processor("extract_code", extract_code_block)
        await process_single_file(
            app=app,
            model_name=model_name,
            reference_files=args.reference,
            input_file=args.input,
            prompt_file=args.prompt,
            output=args.output,
            input_language=args.language,
            batch_mode=args.batch,
            use_system=(not args.no_system)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM代码修改工具 - 使用AI助手修改代码文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
# 基本用法（交互模式）
python3 llm_code_edit.py -i input.py -o output.py --llm-config llm_config.toml

# 使用参考文件
python3 llm_code_edit.py -i input.py -o output.py --reference api.py utils.py --llm-config llm_config.toml

# 批处理模式（不实时显示）
python3 llm_code_edit.py -i input.py -o output.py --batch --model Kimi-K2 --llm-config llm_config.toml

# 使用远程ZMQ服务
python3 llm_code_edit.py -i input.py -o output.py --zmq-endpoint tcp://localhost:5555 --auth TOKEN
        """.strip()
    )

    parser.add_argument(
        "-i", "--input",
        type=str, required=False,
        help="输入文件路径（待修改的代码文件）"
    )

    parser.add_argument(
        "-p", "--prompt",
        type=str, required=False,
        help="输入提示词文本"
    )

    parser.add_argument(
        "-o", "--output",
        type=str, required=False,
        help="输出文件路径（修改后的代码文件）"
    )

    parser.add_argument(
        "-l", "--language",
        type=str, required=False,
        help="输入代码语言"
    )

    parser.add_argument(
        "-r", "--reference",
        type=str, nargs='+', default=[],
        help="参考文件路径（提供上下文，可多个）"
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="批处理模式（不实时显示思考过程）"
    )

    parser.add_argument(
        "--no-system",
        action="store_true",
        help="不使用系统提示词"
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        default='DeepSeek-v3.2-251201',
        help="指定模型名称（如: Kimi-K2, DeepSeek-v3）"
    )

    # LLM客户端配置（二选一）
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-lc", "--llm-config",
        type=str,
        help="LLM客户端配置文件路径（TOML格式）"
    )
    group.add_argument(
        "-z", "--zmq-endpoint",
        type=str,
        help="ZeroMQ服务端点（如: tcp://localhost:5555）"
    )
    parser.add_argument(
        "-a", "--auth",
        type=str,
        help="ZeroMQ认证"
    )

    args = parser.parse_args()
    asyncio.run(main(args))
