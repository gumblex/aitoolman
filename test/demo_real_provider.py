"""真实提供商端点实测脚本：三协议（OpenAI Chat Completions / Responses / Anthropic）
流式与批量模式下的算术与工具用例实测。

用法示例:
# 全部内置端点、流式+批量、全部用例
python3 test/demo_real_provider.py --config llm_provider.toml

# 指定模型与模式
python3 test/demo_real_provider.py --config llm_provider.toml -m qwen4-max,qwen4-max-r --mode stream --cases tool

每个请求打印 finish_reason、token 用量、响应摘要与 raw_value 完整内容；
异常区分协议问题（4xx/响应解析格式错误）与服务端问题（5xx/超时）；
个别端点不可用时记录后继续测其余端点；每个条目完成后即时追加至
test/demo_real_provider_entries.jsonl，并按 model+mode+case 去重刷新
test/demo_real_provider_report.md，分批运行或中断均不丢已完成结果。
"""

import os
import sys
import re
import json
import time
import asyncio
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aitoolman import util
from aitoolman import client as _client
from aitoolman.model import (
    Message,
    LLMProviderRequest,
    LLMApiRequestError,
    LLMResponseFormatError,
)

# 默认内置的 17 个实测端点
DEFAULT_MODELS = [
    'deepseek-v4-flash-ds',
    'deepseek-v4-flash-ds-a',
    'deepseek-v4-flash-ds-r',
    'deepseek-v4-flash-baidu',
    'deepseek-v4-flash-baidu-a',
    'deepseek-v4-flash-baidu-r',
    'Doubao-Seed-2.1-lite-260915',
    'Doubao-Seed-2.1-lite-260915-r',
    'mimo-v2.6-flash-xiaomi',
    'mimo-v2.6-flash-xiaomi-r',
    'mimo-v2.6-flash-xiaomi-a',
    'glm-5.3-flash-tencent',
    'glm-5.3-flash-tencent-r',
    'glm-5.3-flash-tencent-a',
    'qwen4-max',
    'qwen4-max-r',
    'qwen4-max-a',
]

# 单请求脚本层等待上限（秒），防止异常端点长时间挂起
REQUEST_TIMEOUT = 360.0

# 实测报告文件路径（固定文件名，便于修复后回归时覆盖更新）
REPORT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'demo_real_provider_report.md')

# 条目累积文件（JSONL，一行一条 entry）：报告由全部历史条目去重生成，
# 支持分批运行与中断后不丢结果
ENTRIES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'demo_real_provider_entries.jsonl')

ARITH_PROMPT = '直接输出 16852*4793=?'
ARITH_ANSWER = '80771636'


def normalize_number_text(text):
    """剥离千位分隔符（中英文逗号）与空白字符，用于数值包含校验"""
    return re.sub(r'[,，\s]', '', text or '')


def check_arithmetic_text(text):
    """校验响应文本（剥离千位分隔符与空白后）包含正确乘积"""
    return ARITH_ANSWER in normalize_number_text(text)

TOOL_PROMPT = '请查询北京 2026-04-01 的天气，必须调用 get_weather 工具获取。'
WEATHER_TOOL = {
    'get_weather': {
        'type': 'function',
        'description': '查询指定城市与日期的天气',
        'param': {
            'city': {
                'type': 'string',
                'description': '城市名称，如 北京',
                'required': True,
            },
            'date': {
                'type': 'string',
                'description': '日期，如 2026-04-01',
                'required': True,
            },
        },
    },
}


def mock_get_weather(city, date):
    """get_weather mock 工具实现"""
    return '%s %s 天气：晴，3~17℃，南风3级，湿度45%%。' % (city, date)


def setup_logging(verbose):
    """与 cli 一致的日志级别矩阵：0=默认、-v=INFO(providers)/DEBUG(其他)、-vv=DEBUG"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    if verbose:
        for name in logging.root.manager.loggerDict:
            if name.startswith('aitoolman.providers'):
                if verbose > 1:
                    logging.getLogger(name).setLevel(logging.DEBUG)
                else:
                    logging.getLogger(name).setLevel(logging.INFO)
            elif name.startswith('aitoolman'):
                logging.getLogger(name).setLevel(logging.DEBUG)


def new_entry(model_name, mode, case):
    return {
        'model': model_name,
        'mode': mode,
        'case': case,
        'pass': False,
        'finish_reason': '',
        'error_kind': '',
        'error_text': '',
        'prompt_tokens': '',
        'completion_tokens': '',
        'cached_tokens': '',
        'elapsed': 0.0,
        'summary': '',
        'raw_value': '',
        'note': '',
    }


def classify_error(exc):
    """异常二分类: protocol=协议问题(4xx/响应解析格式错误)、server=服务端问题(5xx/超时)"""
    if isinstance(exc, LLMResponseFormatError):
        return 'protocol'
    text = str(exc)
    if not text:
        text = type(exc).__name__
    if 'No available provider' in text or 'No available model' in text:
        return 'config'
    lowered = text.lower()
    if 'timeout' in lowered or 'timed out' in lowered:
        return 'server'
    for token in re.findall(r'\b[45]\d{2}\b', text):
        code = int(token)
        if 400 <= code < 500:
            return 'protocol'
        if 500 <= code < 600:
            return 'server'
    if isinstance(exc, LLMApiRequestError):
        return 'server'
    return 'server'


def record_error(entry, exc):
    entry['pass'] = False
    entry['error_kind'] = classify_error(exc)
    entry['error_text'] = '%s: %s' % (type(exc).__name__, exc)


def summarize_text(text, limit=120):
    text = (text or '').replace('\n', ' ').strip()
    if len(text) > limit:
        return text[:limit] + '...'
    return text


def dump_raw_value(response):
    """展开 response_message 各消息的 raw_value 完整内容"""
    lines = []
    for i, m in enumerate(response.response_message):
        raw = m.raw_value
        if raw is None:
            lines.append('#%d raw_value=None' % i)
            continue
        try:
            lines.append('#%d %s' % (i, json.dumps(raw, ensure_ascii=False)))
        except (TypeError, ValueError):
            lines.append('#%d %r' % (i, raw))
    return '\n'.join(lines) if lines else '(response_message empty)'


def fill_common(entry, response, elapsed):
    entry['finish_reason'] = response.finish_reason or ''
    entry['elapsed'] = elapsed
    entry['prompt_tokens'] = response.prompt_tokens
    entry['completion_tokens'] = response.completion_tokens
    entry['cached_tokens'] = response.cached_tokens
    entry['summary'] = summarize_text(response.response_text)


async def execute_request(client, model_name, messages, tools, stream):
    """经 provider 层执行单次请求，返回 (LLMProviderResponse, elapsed)"""
    manager = client.provider_manager
    await manager.initialize()
    request = LLMProviderRequest(
        client_id='demo_real_provider',
        context_id='',
        request_id=util.get_id(),
        model_name=model_name,
        messages=list(messages),
        tools=tools or {},
        stream=stream,
        options={'stream': stream},
    )
    start = time.monotonic()
    rt = manager.process_request(request)
    try:
        response = await asyncio.wait_for(rt.task, timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        raise asyncio.TimeoutError(
            'script timeout: no response within %ss' % REQUEST_TIMEOUT) from None
    elapsed = time.monotonic() - start
    response.raise_for_status()
    return response, elapsed


async def run_arithmetic(client, model_name, mode):
    """算术用例：校验响应文本包含正确乘积"""
    stream = mode == 'stream'
    entry = new_entry(model_name, mode, 'arithmetic')
    messages = [Message(role='user', content=ARITH_PROMPT)]
    try:
        response, elapsed = await execute_request(
            client, model_name, messages, None, stream)
        fill_common(entry, response, elapsed)
        entry['raw_value'] = dump_raw_value(response)
        entry['pass'] = check_arithmetic_text(response.response_text)
        entry['note'] = ('answer should contain %s (thousand separators/spaces ignored)'
                         % ARITH_ANSWER)
    except Exception as exc:  # 单端点失败不影响其余端点
        record_error(entry, exc)
    return entry


async def run_tool_case(client, model_name, mode):
    """工具用例：两轮对话，第二轮以 raw_value 原样回传 assistant 响应消息"""
    stream = mode == 'stream'
    entry = new_entry(model_name, mode, 'tool')
    user_msg = Message(role='user', content=TOOL_PROMPT)
    raw_parts = []
    try:
        # 第一轮：获取 tool_calls
        response1, elapsed = await execute_request(
            client, model_name, [user_msg], WEATHER_TOOL, stream)
        fill_common(entry, response1, elapsed)
        raw_parts.append('[round1] ' + dump_raw_value(response1))
        entry['raw_value'] = '\n'.join(raw_parts)
        if response1.finish_reason != 'tool_calls' or not response1.response_tool_calls:
            entry['note'] = ('round1 finished without tool_calls (finish_reason=%s), '
                             'round2 skipped' % response1.finish_reason)
            return entry

        # 第二轮：原 user 消息 + assistant 原始响应消息（raw_value 原样回传）+ tool 结果
        context = [user_msg]
        for m in response1.response_message:
            if m.raw_value is not None:
                context.append(Message(raw_value=m.raw_value))
            else:
                context.append(m)
        for tc in response1.response_tool_calls:
            try:
                tool_result = mock_get_weather(**(tc.arguments or {}))
            except TypeError as te:
                tool_result = 'mock tool error: %s' % te
            context.append(Message(
                role='tool', content=str(tool_result), tool_call_id=tc.id))

        response2, elapsed2 = await execute_request(
            client, model_name, context, WEATHER_TOOL, stream)
        fill_common(entry, response2, elapsed2)
        raw_parts.append('[round2] ' + dump_raw_value(response2))
        entry['raw_value'] = '\n'.join(raw_parts)
        entry['pass'] = True
        entry['note'] = ('round1 tool_calls=%d; round2 with raw_value passthrough ok'
                         % len(response1.response_tool_calls))
    except Exception as exc:  # 单端点失败不影响其余端点
        entry['raw_value'] = '\n'.join(raw_parts)
        record_error(entry, exc)
    return entry


async def run_prompt(client, model_name, mode, prompt):
    """自由提问用例：校验响应非空，用于性能剖析实测"""
    stream = mode == 'stream'
    entry = new_entry(model_name, mode, 'prompt')
    messages = [Message(role='user', content=prompt)]
    try:
        response, elapsed = await execute_request(
            client, model_name, messages, None, stream)
        fill_common(entry, response, elapsed)
        entry['raw_value'] = dump_raw_value(response)
        entry['pass'] = bool((response.response_text or '').strip())
        entry['note'] = 'free prompt case for profiling'
    except Exception as exc:
        entry['raw_value'] = ''
        record_error(entry, exc)
    return entry


def print_entry(entry):
    print('=' * 78)
    print('[%s] model=%s mode=%s case=%s' % (
        'PASS' if entry['pass'] else 'FAIL',
        entry['model'], entry['mode'], entry['case']))
    print('finish_reason: %s' % (entry['finish_reason'] or '-'))
    print('tokens: prompt=%s completion=%s cached=%s elapsed=%.1fs' % (
        entry['prompt_tokens'], entry['completion_tokens'],
        entry['cached_tokens'], entry['elapsed']))
    if entry['summary']:
        print('summary: %s' % entry['summary'])
    if entry['note']:
        print('note: %s' % entry['note'])
    if entry['error_text']:
        print('error(%s): %s' % (entry['error_kind'] or '-', entry['error_text']))
    print('raw_value:')
    print(entry['raw_value'] or '-')


def expand_modes(mode):
    if mode == 'both':
        return ('stream', 'batch')
    return (mode,)


def append_entries(entries):
    """条目追加写入 JSONL 累积文件"""
    with open(ENTRIES_PATH, 'a', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')


def load_entries():
    """读取 JSONL 累积文件中的全部条目"""
    if not os.path.exists(ENTRIES_PATH):
        return []
    entries = []
    with open(ENTRIES_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except ValueError:
                continue
    return entries


def dedupe_entries(entries):
    """按 (model, mode, case) 去重，值取最后一次出现，顺序按首次出现"""
    order = []
    by_key = {}
    for e in entries:
        key = (e.get('model'), e.get('mode'), e.get('case'))
        if key not in by_key:
            order.append(key)
        by_key[key] = e
    return [by_key[k] for k in order]


def write_report(entries, args, path):
    passed = sum(1 for e in entries if e['pass'])
    fail_kinds = {}
    for e in entries:
        if not e['pass']:
            kind = e['error_kind'] or 'assert'
            fail_kinds[kind] = fail_kinds.get(kind, 0) + 1
    lines = []
    lines.append('# 真实提供商端点实测报告')
    lines.append('')
    lines.append('- 生成时间: %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    lines.append('- 配置文件: %s' % args.config)
    lines.append('- 模式: %s；用例: %s' % (args.mode, args.cases))
    lines.append('- 条目: %d（通过 %d / 失败 %d）' % (
        len(entries), passed, len(entries) - passed))
    lines.append('- 失败分类: %s' % (
        ', '.join('%s=%d' % (k, v) for k, v in sorted(fail_kinds.items())) or 'none'))
    lines.append('')
    lines.append('## 汇总')
    lines.append('')
    lines.append('| 模型 | 模式 | 用例 | 结果 | finish_reason | 错误分类 | 耗时s | tokens(p/c) | 摘要 |')
    lines.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for e in entries:
        lines.append('| %s | %s | %s | %s | %s | %s | %.1f | %s/%s | %s |' % (
            e['model'], e['mode'], e['case'],
            'PASS' if e['pass'] else 'FAIL',
            e['finish_reason'] or '-', e['error_kind'] or '-',
            e['elapsed'], e['prompt_tokens'], e['completion_tokens'],
            summarize_text(e['summary'], 60).replace('|', '\\|')))
    lines.append('')
    lines.append('## 详情')
    lines.append('')
    for e in entries:
        lines.append('### %s / %s / %s' % (e['model'], e['mode'], e['case']))
        lines.append('')
        lines.append('- 结果: %s' % ('PASS' if e['pass'] else 'FAIL'))
        lines.append('- finish_reason: %s' % (e['finish_reason'] or '-'))
        lines.append('- tokens: prompt=%s completion=%s cached=%s elapsed=%.1fs' % (
            e['prompt_tokens'], e['completion_tokens'], e['cached_tokens'], e['elapsed']))
        if e['note']:
            lines.append('- note: %s' % e['note'])
        if e['error_text']:
            lines.append('- error(%s): %s' % (e['error_kind'] or '-', e['error_text']))
        lines.append('- 响应摘要: %s' % (e['summary'] or '-'))
        lines.append('- raw_value:')
        lines.append('```')
        lines.append(e['raw_value'] or '(empty)')
        lines.append('```')
        lines.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def parse_args():
    parser = argparse.ArgumentParser(
        description='真实提供商端点实测脚本（流式/批量、算术/工具用例）')
    parser.add_argument(
        '-c', '--config', type=str, default='llm_provider.toml',
        help='Path to the TOML config file')
    parser.add_argument(
        '-m', '--model', type=str, default='',
        help='逗号分隔的模型名列表（默认内置 17 端点全测）')
    parser.add_argument(
        '--mode', choices=('stream', 'batch', 'both'), default='both',
        help='请求模式：stream/batch/both')
    parser.add_argument(
        '--cases', choices=('arithmetic', 'tool', 'all'), default='all',
        help='测试用例选择')
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='Print debug log (-v: providers INFO & others DEBUG; -vv: all DEBUG)')
    parser.add_argument(
        '--prompt', type=str, default='',
        help='自由提问内容；指定后仅运行该提问用例（用于性能剖析实测）')
    parser.add_argument(
        '--profile', action='store_true',
        help='启用 line_profiler 剖析三协议解析与 provider 处理函数（LoopDetector 不剖析）')
    return parser.parse_args()


def resolve_models(args):
    if args.model:
        return [m.strip() for m in args.model.split(',') if m.strip()]
    return list(DEFAULT_MODELS)


async def run_all(args, models, report_path):
    config = util.load_config(args.config)
    client = _client.LLMLocalClient(config)
    print('start real provider test: models=%d config=%s mode=%s cases=%s' % (
        len(models), args.config, args.mode, args.cases), flush=True)
    async with client:
        for model_name in models:
            for mode in expand_modes(args.mode):
                round_entries = []
                if args.prompt:
                    entry = await run_prompt(client, model_name, mode, args.prompt)
                    print_entry(entry)
                    round_entries.append(entry)
                else:
                    if args.cases in ('arithmetic', 'all'):
                        entry = await run_arithmetic(client, model_name, mode)
                        print_entry(entry)
                        round_entries.append(entry)
                    if args.cases in ('tool', 'all'):
                        entry = await run_tool_case(client, model_name, mode)
                        print_entry(entry)
                        round_entries.append(entry)
                # 条目即时累积并刷新报告，分批运行或中断均不丢已完成结果
                append_entries(round_entries)
                write_report(dedupe_entries(load_entries()), args, report_path)
    all_entries = dedupe_entries(load_entries())
    write_report(all_entries, args, report_path)
    passed = sum(1 for e in all_entries if e['pass'])
    print('-' * 78)
    print('total=%d passed=%d failed=%d' % (
        len(all_entries), passed, len(all_entries) - passed))
    print('report: %s' % report_path)


PROFILE_METHODS = (
    'make_request_body',
    'serialize_message',
    'parse_batch_response',
    'parse_stream_event',
    '_accumulate_delta',
    '_build_stream_messages',
    '_handle_end_event',
)


def _add_profile_target(profile, func):
    """按 code 对象去重注册剖析函数"""
    if func is None:
        return
    code = getattr(func, '__code__', None)
    if code is None:
        code = getattr(getattr(func, '__func__', None), '__code__', None)
    if code is None:
        return
    code_map = getattr(profile, 'code_map', None)
    if code_map is not None and code in code_map:
        return
    profile.add_function(func)


def setup_profiler():
    """--profile：line_profiler API 直接剖析（python3 直接运行，不依赖 kernprof）

    剖析对象：三协议策略（OpenAICompatibleFormat/ResponsesFormat/AnthropicFormat）的
    make_request_body/serialize_message/parse_batch_response/parse_stream_event
    及 _accumulate_delta/_build_stream_messages/_handle_end_event；
    provider.py 的 _call_llm_api/_handle_stream_request/_handle_batch_request。
    LoopDetector 不剖析。
    """
    try:
        from line_profiler import LineProfiler
    except ImportError:
        print('line_profiler not installed, --profile ignored.')
        return None
    from aitoolman.providers import openai as openai_fmt
    from aitoolman.providers import anthropic as anthropic_fmt
    from aitoolman import provider as provider_mod

    profile = LineProfiler()
    for module, class_names in (
            (openai_fmt, ('OpenAICompatibleFormat', 'ResponsesFormat')),
            (anthropic_fmt, ('AnthropicFormat',))):
        for class_name in class_names:
            cls = getattr(module, class_name, None)
            if cls is None:
                continue
            for name in PROFILE_METHODS:
                _add_profile_target(profile, getattr(cls, name, None))
        for name in PROFILE_METHODS:
            # 模块级辅助函数（如 _accumulate_delta）
            _add_profile_target(profile, getattr(module, name, None))
    for name in ('_call_llm_api', '_handle_stream_request', '_handle_batch_request'):
        for obj in list(vars(provider_mod).values()):
            if isinstance(obj, type):
                _add_profile_target(profile, getattr(obj, name, None))
    return profile


def main():
    args = parse_args()
    setup_logging(args.verbose)
    models = resolve_models(args)
    profile = setup_profiler() if args.profile else None
    if profile is not None:
        print('profiling: line_profiler enabled for protocol/provider functions')
    try:
        if profile is not None:
            profile.enable()
        asyncio.run(run_all(args, models, REPORT_PATH))
    finally:
        if profile is not None:
            profile.disable()
            profile.print_stats()


if __name__ == '__main__':
    main()
