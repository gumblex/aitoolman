import asyncio
import logging
import argparse
from typing import Union

import aitoolman
import aitoolman.cli


CONFIG_FILE = '''
[module_default]
model = "test_model"
stream = false
output_channel = "stdout"
reasoning_channel = "reasoning"

[tools."clarify"]
type = "function"
description = "向用户追问细节"
param."question".type = "string"
param."question".description = "直接向用户提问的内容"
param."question".required = true

[tools."check_order"]
type = "function"
description = "查询订单详情"
param."order_id".type = "string"
param."order_id".description = "订单号"
param."order_id".required = true

[module."entry"]
template.user = """## 相关信息
{{ background_info|join('\\n') }}

## 用户提问
原始提问：{{ user_question }}
{{ context|join('\\n') }}

## 要求
{% if available_tools %}如果问题和细节不清楚，调用相应工具了解问题细节；否则{% endif %}直接向用户回答。
"""
'''


def mock_response(request):
    messages = request.messages
    if not messages:
        raise ValueError("no messages")
    last_msg = messages[-1].content
    if '状态：' in last_msg and '追问：' in last_msg:
        return aitoolman.LLMProviderResponse(
            client_id=request.client_id,
            context_id=request.context_id,
            request_id=request.request_id,
            model_name=request.model_name,
            stream=False,
            finish_reason=aitoolman.FinishReason.stop.value,
            response_text="你的订单15641651，当前状态是已发货",
            response_message={"content": "你的订单15641651，当前状态是已发货"}
        )
    elif '追问：' in last_msg:
        tool_call = aitoolman.ToolCall(
            name="check_order",
            arguments_text='',
            arguments={"order_id": "15641651"}
        )
        return aitoolman.LLMProviderResponse(
            client_id=request.client_id,
            context_id=request.context_id,
            request_id=request.request_id,
            model_name=request.model_name,
            stream=False,
            finish_reason=aitoolman.FinishReason.tool_calls.value,
            response_tool_calls=[tool_call],
            response_message={"tool_calls": [tool_call._asdict()]}
        )
    else:
        tool_call = aitoolman.ToolCall(
            name="clarify",
            arguments_text='',
            arguments={"question": "你的订单号是多少？"}
        )
        return aitoolman.LLMProviderResponse(
            client_id=request.client_id,
            context_id=request.context_id,
            request_id=request.request_id,
            model_name=request.model_name,
            stream=False,
            finish_reason=aitoolman.FinishReason.tool_calls.value,
            response_tool_calls=[tool_call],
            response_message={"tool_calls": [tool_call._asdict()]}
        )


logger_wf = logging.getLogger(__name__ + ".workflow")


class EntryTask(aitoolman.LLMTask):
    module_name = 'entry'

    async def pre_process(self) -> Union[aitoolman.LLMModuleRequest, aitoolman.LLMDirectRequest, None]:
        await self.workflow.channels['status'].write("当前任务：%s" % self.task_name)
        await self.workflow.channels['status'].write(None)
        tools = {
            x: self.workflow.global_tools[x]
            for x in self.workflow.vars['available_tools']
        }
        req = await super().pre_process()
        return req._replace(tools=tools)

    async def post_process(self):
        if not self.module_result:
            raise RuntimeError("no module_result")
        self.module_result.raise_for_status()
        if self.module_result.status == aitoolman.FinishReason.tool_calls:
            await self.workflow.channels['status'].write("调用工具：%s" % ','.join(
                map(str, self.module_result.tool_calls)))
            await self.workflow.channels['status'].write(None)
            self.on_tool_call_goto(
                clarify=ClarifyTask,
                check_order=OrderTask
            )
        else:
            self.output_data = self.module_result.data


class ClarifyTask(EntryTask):
    module_name = 'entry'

    async def pre_process(self):
        question = self.input_data['question']
        await self.workflow.channels['stdout'].write(question)
        user_answer = await self.workflow.channels['stdin'].read()
        self.workflow.vars['context'].append('')
        self.workflow.vars['context'].append('追问：' + question.strip())
        self.workflow.vars['context'].append('用户回答：' + user_answer.strip())
        self.workflow.vars['available_tools'].discard('clarify')
        return await super().pre_process()


class OrderTask(EntryTask):
    module_name = 'entry'

    async def pre_process(self):
        order_id = self.input_data['order_id']
        order_status = '已发货，暂无物流信息'
        self.workflow.vars['background_info'].append('订单 %s 的状态：%s' % (order_id, order_status))
        self.workflow.vars['available_tools'].discard('check_order')
        return await super().pre_process()


def create_workflow(client: aitoolman.LLMClient, model_name: str):
    app_config = aitoolman.load_config_str(CONFIG_FILE)
    app_config['module_default']['model'] = model_name

    app = aitoolman.LLMWorkflow(
        client, config_dict=app_config,
        channels={'status': aitoolman.Channel()}
    )
    app.vars['available_tools'] = {'clarify', 'check_order'}
    app.vars['background_info'] = []
    app.vars['context'] = []
    return app


async def run(client: aitoolman.LLMClient, model_name: str):
    async with client:
        app = create_workflow(client, model_name)
        channel_collector = aitoolman.DefaultTextChannelCollector({
            k: v for k, v in app.channels.items()
            if k in ('stdout', 'reasoning', 'status')
        })
        output_task = asyncio.create_task(channel_collector.start_listening())

        await app.channels['stdin'].write('我的订单现在什么状态？')
        await app.channels['stdin'].write('15641651')
        user_question = await app.channels['stdin'].read()
        app.vars['user_question'] = user_question
        await app.run(EntryTask(input_data={
            'user_question': user_question
        }))
        channel_collector.close()
        await output_task


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.WARN,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    parser = argparse.ArgumentParser(description="LLM server.")
    parser.add_argument(
        "-v", "--verbose", action='store_true',
        help="Print debug log"
    )
    parser.add_argument(
        '-r', '--router-endpoint', default='tcp://localhost:5555',
        help='ZeroMQ ROUTER endpoint (e.g., tcp://localhost:5555)')
    parser.add_argument(
        '-a', '--auth-token', help='ZeroMQ ROUTER auth token')
    parser.add_argument('-m', '--model-name', required=True, help='Model name to use')
    args = parser.parse_args()

    from aitoolman import zmqclient

    client = zmqclient.LLMZmqClient(args.router_endpoint, args.auth_token)
    asyncio.run(run(client, args.model_name))
