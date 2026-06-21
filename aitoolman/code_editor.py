#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from . import app
from .model import LLMModuleRequest, MediaContent
from .channel import Channel, print_channel_output

import fix_llm_xml

# Log configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Size limit constants
TOKEN_RATIO = 4.0
WARN_FILE_K_TOKENS = 64
WARN_PROMPT_K_TOKENS = 120


# ------------------------------
# Postprocessor: parse multi-file content in <output>
# ------------------------------
def extract_code_blocks(text: str) -> List[Dict]:
    """
    Parse multiple <file> contents inside <output> tags from LLM output
    Return format: [{'filename': str, 'content': str}, ...]
    """
    output_content = fix_llm_xml.find_xml_document(text, 'output', with_tag=True)
    if not output_content:
        logger.warning("<output> tag not found, treat all content as a single file")
        return [{'filename': None, 'content': text.strip()}]

    xml_dict = fix_llm_xml.parse_xml(
        output_content, 'output',
        text_tags=['file'],
        strip_whitespace=False
    )
    if not xml_dict or 'file' not in xml_dict.get('output', {}):
        logger.warning("<file> tag not found, treat all content as a single file")
        return [{'filename': None, 'content': output_content.strip()}]

    files_data = xml_dict['output']['file']
    if isinstance(files_data, dict):
        files_data = [files_data]

    result = []
    for file_item in files_data:
        filename = file_item.get('@name')
        content = file_item.get('#text', '')
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
# Output path handling functions
# ------------------------------
def get_output_path(
    output_arg: str, filename: str, input_files: List[str], result_num: int
) -> Path:
    """Determine final output path based on output parameter and filename"""
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
    """Handle existing files, return final path"""
    if not file_path.exists():
        return file_path

    if overwrite:
        logger.warning("File already exists, overwriting: %s", file_path)
        return file_path
    else:
        stem = file_path.stem
        suffix = file_path.suffix
        new_path = file_path.with_name(f"{stem}.new{suffix}")
        logger.warning("File already exists, writing to new file: %s", new_path)
        return new_path


def load_files_from_paths(
    paths: List[str], relative_to: Optional[Path] = None,
    file_size_limit: Optional[int] = None
) -> List[Dict]:
    """
    Load files from path list, support recursive directory traversal
    :param paths: List of input file/directory paths
    :param relative_to: If not None, convert returned filename to relative path against this path
    :param file_size_limit: Skip files exceeding size limit
    :return: List of successfully loaded files, each element is {'filename': str, 'content': str}
    """
    loaded = []
    base_path = (relative_to or Path.cwd()).resolve()

    def _process_path(p: Path, is_direct_specified: bool = True):
        if not p.exists():
            logger.warning(f"Path does not exist, skipping: {p}")
            return
        # Process directory
        if p.is_dir():
            for child in p.iterdir():
                # Skip hidden files/directories for non-direct specified paths
                if not is_direct_specified and child.name.startswith('.'):
                    logger.debug(f"Skip hidden file/directory: {child}")
                    continue
                _process_path(child, is_direct_specified=False)
            return
        # Process file
        if p.is_file():
            if not is_direct_specified and p.name.startswith('.'):
                logger.debug(f"Skip hidden file: {p}")
                return

            # Check if file size exceeds limit
            file_size = p.stat().st_size
            if file_size_limit and file_size > file_size_limit:
                logger.warning(f"File size {file_size} > {file_size_limit}, skipping: {p}")
                return

            # Read and validate file
            try:
                content = p.read_text(encoding='utf-8')
                # Exclude files containing NUL characters
                if '\x00' in content:
                    logger.warning(f"Binary file, skipping: {p}")
                    return

                estimated_file_k_tokens = len(content.encode('utf-8'))/TOKEN_RATIO/1024.0
                if estimated_file_k_tokens > WARN_FILE_K_TOKENS:
                    logger.warning("Large file (estimated %.1fK tokens): %s", estimated_file_k_tokens, p)

                # Process filename path
                filename = str(p)
                if relative_to:
                    try:
                        filename = str(p.relative_to(base_path, walk_up=True).as_posix())
                    except ValueError:
                        filename = str(p)
                loaded.append({'filename': filename, 'content': content})
            except UnicodeDecodeError:
                logger.warning(f"File is not UTF-8 encoded, skipping: {p}")
            except Exception as e:
                logger.warning(f"Failed to read file {p}: {str(e)}")

    for path_str in paths:
        p = Path(path_str).resolve()
        _process_path(p, is_direct_specified=True)
    return loaded


def read_user_input(prompt) -> str:
    print(prompt + " (enter a separate line with . to finish)")
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
# Asynchronous processing functions
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
    """Process multiple files"""
    references = load_files_from_paths(
        reference_files, relative_to=Path.cwd(), file_size_limit=file_size_limit)
    input_files_list = load_files_from_paths(
        input_files, relative_to=Path.cwd(), file_size_limit=file_size_limit)

    if references:
        logger.info("Reference files: %s", ', '.join(f['filename'] for f in references))
    if input_files_list:
        logger.info("Input files: %s", ', '.join(f['filename'] for f in input_files_list))

    # Validate if there are valid files left
    if input_files and not input_files_list:
        raise ValueError("All input files are excluded due to unreadable or size exceeding limit, cannot proceed")
    if not input_files and reference_files and not references:
        raise ValueError("All reference files are excluded due to unreadable or size exceeding limit, cannot proceed")

    media_content_list = []
    if media_files:
        media_content_list = [MediaContent.load_from_path(m) for m in media_files]

    # Get user instruction
    user_instruction = None
    if prompt_file:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            user_instruction = f.read()
    if not user_instruction:
        user_instruction = read_user_input("Please enter instruction to modify")

    # Start channel collector
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
    rendered = await llm_app.render_direct_request(LLMModuleRequest(
        module_name='code_edit',
        template_params=template_params,
        media_content=media_content_list,
        model_name=model_name,
        stream=(not batch_mode),
        output_channel=output_channel
    ))
    logger.info("Resolved actual model: %s", rendered.model_name)
    total_bytes = sum(len((m.content or '').encode('utf-8')) for m in rendered.messages)
    estimated_total_k_tokens = total_bytes / TOKEN_RATIO / 1024.0
    if estimated_total_k_tokens >= WARN_PROMPT_K_TOKENS:
        logger.warning(
            "Prompt total length is large, estimated %.1fK tokens >= %dK tokens, may reach model's limit",
            estimated_total_k_tokens, WARN_PROMPT_K_TOKENS
        )

    result = await llm_app.call(rendered)
    result.raise_for_status()
    await output_task

    # Process output files
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
                    logger.info(f"Written to file: {output_path} (original name {filename})")
                else:
                    logger.info(f"Written to file: {output_path}")
            except Exception as e:
                logger.exception(f"Failed to write file: {output_path or filename}")
    return result
