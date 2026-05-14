#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from . import app, postprocess
from .model import LLMModuleRequest, MediaContent
from .channel import Channel, print_channel_output

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# 大小限制常量
WARN_FILE_CHARS = 128 * 1024
WARN_PROMPT_CHARS = 200 * 1024


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
<file name="{{ ref.filename }}"><![CDATA[{{ref.content}}]]></file>
{% endfor %}
{% endif %}
{% if input_files -%}
# 当前关注文件
{% for file in input_files -%}
<file name="{{ file.filename }}"><![CDATA[{{file.content}}]]></file>
{% endfor %}
{%- endif %}
# 用户需求
{{user_instruction}}
{% if use_system -%}

***

# 系统输出格式
用 XML 格式输出根据需求编写或修改后的文件内容（不是指代码可能使用的输出格式）：
<output>
{% if input_files|length == 1 and output_file -%}
<file name="{{ output_file }}"><![CDATA[文件内容]]></file>
{%- else -%}
<file name="输出文件名1"><![CDATA[文件内容1]]></file>
<file name="输出文件名2"><![CDATA[文件内容2]]></file>
{%- endif %}
</output>
{% endif %}"""
post_processor = "extract_code_blocks"
'''


# ------------------------------
# 输出路径处理函数
# ------------------------------
def get_output_path(
    output_arg: str, filename: str, input_files: List[str], result_num: int
) -> Path:
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
    elif result_num > 1:
        if output_path.is_file():
            return output_path.parent / filename
        else:
            return output_path / filename
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


def load_files_from_paths(
    paths: List[str], relative_to: Optional[Path] = None,
    file_size_limit: Optional[int] = None
) -> List[Dict]:
    """
    从路径列表加载文件，支持递归遍历目录
    :param paths: 输入的文件/目录路径列表
    :param relative_to: 如果不为None，返回的文件名转为相对于该路径的相对路径
    :param file_size_limit: 跳过超过大小的文件
    :return: 加载成功的文件列表，每个元素为 {'filename': str, 'content': str}
    """
    loaded = []
    base_path = (relative_to or Path.cwd()).resolve()

    def _process_path(p: Path, is_direct_specified: bool = True):
        if not p.exists():
            logger.warning(f"路径不存在，跳过: {p}")
            return
        # 处理目录
        if p.is_dir():
            for child in p.iterdir():
                # 非直接指定的路径跳过.开头的隐藏文件/目录
                if not is_direct_specified and child.name.startswith('.'):
                    logger.debug(f"跳过隐藏文件/目录: {child}")
                    continue
                _process_path(child, is_direct_specified=False)
            return
        # 处理文件
        if p.is_file():
            if not is_direct_specified and p.name.startswith('.'):
                logger.debug(f"跳过隐藏文件: {p}")
                return

            # 检查文件大小是否超过最大限制
            file_size = p.stat().st_size
            if file_size_limit and file_size > file_size_limit:
                logger.warning(f"文件大小 {file_size} > {file_size_limit}，跳过: {p}")
                return

            # 读取文件校验
            try:
                content = p.read_text(encoding='utf-8')
                # 排除含有NUL字符的文件
                if '\x00' in content:
                    logger.warning(f"二进制文件，跳过: {p}")
                    return

                if len(content) > WARN_FILE_CHARS:
                    logger.warning("大文件（字符数 %s）: %s", len(content), p)

                # 处理文件名路径
                filename = str(p)
                if relative_to:
                    try:
                        filename = str(p.relative_to(base_path, walk_up=True).as_posix())
                    except ValueError:
                        filename = str(p)
                loaded.append({'filename': filename, 'content': content})
            except UnicodeDecodeError:
                logger.warning(f"文件不是UTF-8编码，跳过: {p}")
            except Exception as e:
                logger.warning(f"读取文件失败 {p}: {str(e)}")

    for path_str in paths:
        p = Path(path_str).resolve()
        _process_path(p, is_direct_specified=True)
    return loaded


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
        use_system: bool = True,
        media_files: Optional[List[str]] = None,
        file_size_limit: Optional[int] = None
) -> app.LLMModuleResult:
    """处理多个文件"""
    logger.info("使用模型: %s", model_name)

    references = load_files_from_paths(
        reference_files, relative_to=Path.cwd(), file_size_limit=file_size_limit)
    input_files_list = load_files_from_paths(
        input_files, relative_to=Path.cwd(), file_size_limit=file_size_limit)

    if references:
        logger.info("参考文件: %s", ', '.join(f['filename'] for f in references))
    if input_files_list:
        logger.info("输入文件: %s", ', '.join(f['filename'] for f in input_files_list))

    # 校验是否还有有效文件
    if input_files and not input_files_list:
        raise ValueError("所有输入文件都因无法读取或大小超出限制被排除，无法继续处理")
    if not input_files and reference_files and not references:
        raise ValueError("所有参考文件都因无法读取或大小超出限制被排除，无法继续处理")

    media_content_list = []
    if media_files:
        media_content_list = [MediaContent.load_from_path(m) for m in media_files]

    # 获取用户指令
    user_instruction = None
    if prompt_file:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            user_instruction = f.read()
    if not user_instruction:
        user_instruction = read_user_input("请输入修改指令")

    # 启动通道收集器
    output_channel = Channel()
    output_task = asyncio.create_task(print_channel_output(
        output_channel,
        topic_names={'reasoning': 'Thinking', 'response': 'Response'},
        header=True
    ))

    output_file = None
    if output_arg:
        output_path = Path(output_arg)
        if output_path.is_file() or '.' in output_path.name:
            output_file = output_arg

    template_params = {
        'user_instruction': user_instruction,
        'input_files': input_files_list,
        'output_file': output_file,
        'references': references,
        'use_system': use_system
    }
    total_chars = sum(len(f['content']) + len(f['filename']) for f in references) + \
                  sum(len(f['content']) + len(f['filename']) for f in input_files_list) + \
                  len(user_instruction)
    if total_chars > WARN_PROMPT_CHARS:
        logger.warning(f"提示词中，总文件长度较大 {total_chars} > {WARN_PROMPT_CHARS}，可能导致处理变慢或无法处理")

    result = await llm_app.call(LLMModuleRequest(
        module_name='code_edit',
        template_params=template_params,
        media_content=media_content_list,
        model_name=model_name,
        stream=(not batch_mode),
        output_channel=output_channel
    ))
    result.raise_for_status()
    await output_task

    # 处理输出文件
    file_results: List[Dict] = result.data
    if use_system and not output_arg and file_results:
        output_arg = '.'
    if output_arg:
        for file_item in file_results:
            filename = file_item['filename']
            content = file_item['content']
            output_path = None
            try:
                output_path = get_output_path(
                    output_arg, filename, input_files, len(file_results)
                )
                output_path = handle_existing_file(output_path, overwrite)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                if filename != output_path:
                    logger.info(f"写入文件: {output_path}（原名 {filename}）")
                else:
                    logger.info(f"写入文件: {output_path}")
            except Exception as e:
                logger.exception(f"写入文件失败: {output_path or filename}")
    return result
