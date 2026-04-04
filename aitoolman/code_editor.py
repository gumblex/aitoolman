#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path
import sys
import asyncio
import logging
from datetime import datetime
from typing import List, Dict
from pathlib import Path

from . import util, app, client, postprocess, channel, zmqclient
from .model import LLMModuleRequest

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ------------------------------
# 后处理器：解析<output>中的多文件内容
# ------------------------------
def extract_code_blocks(text: str) -> List[Dict]:
    """
    从LLM输出中解析<output>标签内的多个<file>内容
    返回格式: [{'filename': str, 'content': str}, ...]
    """
    output_content = postprocess.get_xml_tag_content(text, 'output', with_tag=True)
    if not output_content:
        logger.warning("未找到 <output> 标签，将所有内容作为单个文件处理")
        return [{'filename': None, 'content': text.strip()}]

    xml_dict = postprocess.parse_xml(output_content, 'output')
    if not xml_dict or 'file' not in xml_dict.get('output', {}):
        logger.warning("未找到 <file> 标签，将所有内容作为单个文件处理")
        return [{'filename': None, 'content': output_content.strip()}]

    files_data = xml_dict['output']['file']
    if isinstance(files_data, dict):
        files_data = [files_data]

    result = []
    for file_item in files_data:
        filename = file_item.get('@name')
        content = file_item.get('#text', '').strip()
        result.append({'filename': filename, 'content': content})

    return result


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
{% if input_files -%}
# 当前文件
{% for file in input_files -%}
<file name="{{ file.filename }}">{{file.content}}</file>
{% endfor %}
{%- endif %}
# 要求
{{user_instruction}}
{% if use_system -%}
输出用 XML 格式：
<output>
{% if input_files|length == 1 -%}
<file name="{{ input_files[0].filename }}"><![CDATA[文件内容]]></file>
{%- else -%}
<file name="输出文件名1"><![CDATA[文件内容1]]></file>
<file name="输出文件名2"><![CDATA[文件内容2]]></file>
{% endif %}
</output>
{% endif %}"""
post_processor = "extract_code_blocks"
'''


# ------------------------------
# 输出路径处理函数
# ------------------------------
def get_output_path(output_arg: str, filename: str, input_files: List[str]) -> Path:
    """根据输出参数和文件名确定最终输出路径"""
    output_path = Path(output_arg)

    if output_path.is_dir():
        if filename:
            return output_path / filename
        elif len(input_files) == 1:
            return output_path / Path(input_files[0]).name
        else:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            return output_path / f"output_{timestamp}.md"
    else:
        return output_path


def handle_existing_file(file_path: Path, overwrite: bool) -> Path:
    """处理已存在的文件，返回最终路径"""
    if not file_path.exists():
        return file_path

    if overwrite:
        logger.warning("文件已存在，覆盖: %s", file_path)
        return file_path
    else:
        stem = file_path.stem
        suffix = file_path.suffix
        new_path = file_path.with_name(f"{stem}.new{suffix}")
        logger.warning("文件已存在，写入新文件: %s", new_path)
        return new_path


def read_user_input(prompt) -> str:
    print(prompt + "（结束后输入单独的一行 . ）")
    lines = []
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        if line.rstrip('\r\n') == ".":
            break
        lines.append(line.rstrip("\n"))
    return "\n".join(lines).strip()


# ------------------------------
# 异步处理函数
# ------------------------------
async def process_files(
        llm_app: app.LLMApplication,
        model_name: str,
        reference_files: List[str],
        input_files: List[str],
        prompt_file: str,
        output_arg: str,
        batch_mode: bool,
        overwrite: bool,
        use_system: bool = True
) -> app.LLMModuleResult:
    """处理多个文件"""
    logger.info("使用模型: %s", model_name)
    logger.info("输入文件: %s", ', '.join(input_files))

    # 处理参考文件
    references = []
    for ref_file in reference_files:
        file_path = Path(ref_file)
        with open(ref_file, 'r', encoding='utf-8') as f:
            content = f.read()
        references.append({'filename': str(file_path), 'content': content})

    # 处理输入文件
    input_files_list = []
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        input_files_list.append({
            'filename': os.path.relpath(input_file, os.curdir),
            'content': content
        })

    # 获取用户指令
    user_instruction = None
    if prompt_file:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            user_instruction = f.read()
    if not user_instruction:
        user_instruction = read_user_input("请输入修改指令")

    # 启动通道收集器
    channel_collector = channel.DefaultTextChannelCollector({
        'Thinking': llm_app.channels['reasoning'],
        'Response': llm_app.channels['stdout']
    })
    output_task = asyncio.create_task(channel_collector.start_listening())

    template_params = {
        'user_instruction': user_instruction,
        'input_files': input_files_list,
        'references': references,
        'use_system': use_system
    }
    result = await llm_app.call(LLMModuleRequest(
        module_name='code_edit',
        template_params=template_params,
        model_name=model_name,
        stream=(not batch_mode)
    ))
    result.raise_for_status()
    channel_collector.close()
    await output_task

    # 处理输出文件
    if output_arg:
        file_results: List[Dict] = result.data
        for file_item in file_results:
            filename = file_item['filename']
            content = file_item['content']
            output_path = None
            try:
                output_path = get_output_path(output_arg, filename, input_files)
                output_path = handle_existing_file(output_path, overwrite)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"写入文件: {output_path}")
            except Exception as e:
                logger.exception(f"写入文件失败: {output_path or filename}")
    return result
