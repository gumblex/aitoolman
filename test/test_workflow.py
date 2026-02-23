import dataclasses
import json
import inspect
import asyncio
import unittest
from typing import Dict, List, Any, Callable, Optional, Union

from aitoolman.workflow import (
    LLMWorkflow, LLMTask, LLMTaskStatus, LLMTaskDependencyError
)
from aitoolman.app import LLMApplication
from aitoolman.client import LLMClient
from aitoolman.model import (
    LLMProviderResponse, Message, ToolCall, FinishReason,
    LLMProviderRequest, LLMModuleResult, LLMModuleRequest, LLMDirectRequest
)

from mock_llmclient import MockLLMClient, default_llm_response


TEST_CONFIG = {
    "module": {
        "test_module": {
            "model": "test_model",
            "template": {"user": "test"},
            "post_processor": "builtin.parse_json"
        },
        "test_module_with_input": {
            "model": "test_model",
            "template": {"user": "{{input}}"},
            "post_processor": "builtin.parse_json"
        }
    }
}


class LLMTaskDefaultModule(LLMTask):
    module_name = 'test_module_with_input'


class TestLLMTask(unittest.IsolatedAsyncioTestCase):
    """测试LLMTask的基础功能"""

    async def test_task_success(self):
        """测试任务成功执行"""
        client = MockLLMClient()
        app = LLMWorkflow(client, config_dict=TEST_CONFIG)

        task = LLMTaskDefaultModule(
            input_data={"input": "a"}
        )

        result_task = await app.run(task)
        self.assertIs(result_task, task)
        self.assertEqual(task.status, LLMTaskStatus.COMPLETED)
        self.assertEqual(task.output_data, {"result": "a"})
        self.assertEqual(len(client.requests), 1)

    async def test_task_failure(self):
        """测试任务执行失败"""
        def failure_response(request):
            resp = default_llm_response(request)
            resp.finish_reason = FinishReason.error_app.value
            resp.error_text = "模拟任务失败"
            return resp

        client = MockLLMClient(failure_response)
        app = LLMWorkflow(client, config_dict=TEST_CONFIG)

        task = LLMTaskDefaultModule({"input": "a"})

        with self.assertRaises(LLMTaskDependencyError) as context:
            await app.run(task)
        self.assertEqual(context.exception.error_tasks, [task])
        self.assertEqual(task.status, LLMTaskStatus.FAILED)
        self.assertIn("模拟任务失败", str(context.exception))
        self.assertEqual(len(client.requests), 1)

    async def test_pre_process_failure(self):
        """测试前处理阶段失败"""
        class PreProcessFailedTask(LLMTaskDefaultModule):
            async def pre_process(self):
                raise ValueError("前处理失败")

        client = MockLLMClient()
        app = LLMWorkflow(client, config_dict={})

        task = PreProcessFailedTask()

        with self.assertRaises(LLMTaskDependencyError) as context:
            await app.run(task)

        self.assertEqual(task.status, LLMTaskStatus.FAILED)
        self.assertIn("前处理失败", str(context.exception))
        self.assertEqual(len(client.requests), 0)  # 没有调用LLM

    async def test_post_process_failure(self):
        """测试后处理阶段失败"""
        class PostProcessFailedTask(LLMTaskDefaultModule):
            async def post_process(self):
                raise ValueError("后处理失败")

        client = MockLLMClient()
        app = LLMWorkflow(client, config_dict=TEST_CONFIG)

        task = PostProcessFailedTask(input_data={"input": "a"})

        with self.assertRaises(LLMTaskDependencyError) as context:
            await app.run(task)

        self.assertEqual(task.status, LLMTaskStatus.FAILED)
        self.assertIn("后处理失败", str(context.exception))
        self.assertEqual(len(client.requests), 1)


class TestLLMWorkflow(unittest.IsolatedAsyncioTestCase):
    async def test_workflow_dependency_failure(self):
        """测试调用链失败：依赖项失败导致后续任务失败"""

        def failure_response(request):
            resp = default_llm_response(request)
            resp.finish_reason = FinishReason.error_app.value
            resp.error_text = "模拟依赖任务失败"
            return resp

        client = MockLLMClient(failure_response)
        app = LLMWorkflow(client, config_dict=TEST_CONFIG)

        # 依赖任务
        class DependentTask(LLMTask):
            module_name = "test_module_with_input"

        # 根任务
        class RootTask(LLMTask):
            module_name = "test_module_with_input"
            async def post_process(self):
                # 注意：由于任务在 raise_for_status 阶段就失败了，这里的代码永远不会执行
                self.next_task = DependentTask(
                    input_data={"input": "依赖任务"}
                )

        start_task = RootTask(
            input_data={"input": "根任务"}
        )

        with self.assertRaises(LLMTaskDependencyError) as context:
            await app.run(start_task)

        # 根任务失败
        self.assertEqual(start_task.status, LLMTaskStatus.FAILED)

        self.assertIsNone(start_task.next_task)
        self.assertIn("模拟依赖任务失败", str(context.exception))
        self.assertEqual(len(client.requests), 1)

    async def test_run_cannot_parallel(self):
        """测试run方法不能并行执行"""
        async def slow_response(request: LLMProviderRequest) -> LLMProviderResponse:
            await asyncio.sleep(0.5)
            return default_llm_response(request)

        client = MockLLMClient(slow_response)
        app = LLMWorkflow(client, config_dict=TEST_CONFIG)

        class SimpleTask(LLMTask):
            module_name = "test_module"

        # 创建两个任务
        task1 = SimpleTask(task_id="task1")
        task2 = SimpleTask(task_id="task2")

        # 同时启动两个run
        coro1 = app.run(task1)
        coro2 = app.run(task2)
        with self.assertRaises(RuntimeError):
            await asyncio.gather(coro1, coro2)

    async def test_max_parallel_consumers(self):
        """测试并行consumer个数不超过指定值"""
        max_parallel = 2
        client = MockLLMClient()
        app = LLMWorkflow(client, config_dict=TEST_CONFIG, max_parallel_tasks=max_parallel)

        # 记录consumer数量变化
        consumer_counts = []
        original_start = app._start_consumers
        def track_consumers(task_num):
            original_start(task_num)
            consumer_counts.append(len(app._consumers))
        app._start_consumers = track_consumers

        # 创建5个并行任务
        tasks = [
            LLMTaskDefaultModule(
                workflow=app,
                task_id=f"task{i}",
                input_data={"task": i}
            ) for i in range(5)
        ]

        await app.wait_tasks(*tasks)

        # 检查consumer数量从未超过最大值
        self.assertTrue(all(count <= max_parallel for count in consumer_counts))
        self.assertTrue(all(t.status == LLMTaskStatus.COMPLETED for t in tasks))
        self.assertEqual(len(client.requests), 5)

    async def test_no_residual_tasks_after_completion(self):
        """测试任务完成后无残留的task和consumer"""
        client = MockLLMClient()
        app = LLMWorkflow(client, config_dict=TEST_CONFIG)

        # 创建并执行任务
        tasks = [
            LLMTaskDefaultModule(
                task_id=f"task{i}",
                input_data={'input': i}
            ) for i in range(3)
        ]
        await app.wait_tasks(*tasks)

        # 等待资源清理
        await asyncio.sleep(1.0)

        # 检查无残留
        self.assertEqual(len(app._queued_tasks), 0)
        self.assertEqual(len(app._consumers), 0)
        self.assertTrue(app._pending_queue.empty())
        self.assertEqual(len(app._graph), 0)
        self.assertEqual(len(app._reverse_graph), 0)


# class TestLLMWorkflowSerialUseCase(unittest.IsolatedAsyncioTestCase):
#     CONFIG_FILE = '''
# [module_default]
# model = "test_model"
# stream = false
#
# [tools."clarify"]
# type = "function"
# description = "向用户追问细节"
# param."question".type = "string"
# param."question".description = "直接向用户提问的内容"
# param."question".required = true
#
# [tools."check_order"]
# type = "function"
# description = "查询订单详情"
# param."order_id".type = "string"
# param."order_id".description = "订单号"
# param."order_id".required = true
#
# [module."entry"]
# model = "test_model"
# template.user = """({{ state }})
# {{ context|join('\\n') }}
#
# 用户提问：{{ user_question }}
# {{ user_clarify|join('\\n') }}
#
# 如果问题和细节不清楚，调用相应工具了解问题细节；否则直接向用户回答。
# """
# tools."clarify" = {}
# tools."check_order" = {}
#     '''
#
#     @staticmethod
#     def entry_response(request):
#         """模拟EntryTask返回澄清工具调用"""
#         tool_call = ToolCall(
#             name="ClarifyTask",
#             arguments_text='{"question": "你的订单号是多少？", "original_question": "我的订单在哪里？"}',
#             arguments={
#                 "question": "你的订单号是多少？", "original_question": "我的订单在哪里？"
#             }
#         )
#         resp = LLMProviderResponse(
#             client_id=request.client_id,
#             context_id=request.context_id,
#             request_id=request.request_id,
#             model_name=request.model_name,
#             stream=False,
#             finish_reason=FinishReason.tool_calls.value,
#             response_tool_calls=[tool_call],
#             response_message={"tool_calls": [tool_call._asdict()]}
#         )
#         return resp
#
#     @staticmethod
#     def clarify_response(request):
#         """模拟ClarifyTask返回订单查询工具调用"""
#         tool_call = ToolCall(
#             name="OrderTask",
#             arguments_text='{"order_id": "12345", "original_question": "我的订单在哪里？", "clarification": "12345"}',
#             arguments={
#                 "order_id": "12345", "original_question": "我的订单在哪里？",
#                 "clarification": "12345"
#             }
#         )
#         resp = LLMProviderResponse(
#             client_id=request.client_id,
#             context_id=request.context_id,
#             request_id=request.request_id,
#             model_name=request.model_name,
#             stream=False,
#             finish_reason=FinishReason.tool_calls.value,
#             response_tool_calls=[tool_call],
#             response_message={"tool_calls": [tool_call._asdict()]}
#         )
#         return resp
#
#     @staticmethod
#     def order_response(request):
#         """模拟OrderTask返回查询结果"""
#         resp = LLMProviderResponse(
#             client_id=request.client_id,
#             context_id=request.context_id,
#             request_id=request.request_id,
#             model_name=request.model_name,
#             stream=False,
#             finish_reason=FinishReason.stop.value,
#             response_text=json.dumps({"result": "你的订单12345状态是已发货"}),
#             response_message={"content": "你的订单12345状态是已发货"}
#         )
#         return resp
#
#     @staticmethod
#     def dynamic_response(request):
#         messages = request.messages
#         if not messages:
#             return default_llm_response(request)
#         last_msg = messages[-1].content
#         if last_msg.startswith('(ClarifyTask)'):
#             return TestLLMWorkflowSerialUseCase.clarify_response(request)
#         elif last_msg.startswith('(OrderTask)'):
#             return TestLLMWorkflowSerialUseCase.order_response(request)
#         elif "我的订单在哪里？" in last_msg:
#             return TestLLMWorkflowSerialUseCase.entry_response(request)
#         return default_llm_response(request)
#
#     # 定义EntryTask
#     class EntryTask(LLMTask):
#         async def post_process(self):
#             self.on_tool_call_goto(ClarifyTask=lambda )
#             if self.module_result and self.module_result.tool_calls:
#                 tool_call = self.module_result.tool_calls[0]
#                 if tool_call.name == "ClarifyTask":
#                     self.next_task = ClarifyTask(
#                         workflow=self.workflow,
#                         task_id="clarify",
#                         module_name="test_module_with_input",
#                         input_data={
#                             "input": f"(ClarifyTask) 原始问题：{tool_call.arguments['original_question']} 追问：{tool_call.arguments['question']}"
#                         }
#                     )
#             elif self.module_result:
#                 self.output_data = self.module_result.data
#
#     # 定义ClarifyTask
#     class ClarifyTask(LLMTask):
#         async def pre_process(self) -> Union[LLMModuleRequest, LLMDirectRequest, None]:
#             # 应该要问用户问题
#             return await super().pre_process()
#
#         async def post_process(self):
#             if self.module_result and self.module_result.tool_calls:
#                 tool_call = self.module_result.tool_calls[0]
#                 if tool_call.name == "OrderTask":
#                     self.next_task = OrderTask(
#                         workflow=self.workflow,
#                         task_id="order",
#                         module_name="test_module_with_input",
#                         input_data={
#                             "input": f"(OrderTask) 原始问题：{tool_call.arguments['original_question']} 澄清信息：{tool_call.arguments['clarification']} 查询订单号{tool_call.arguments['order_id']}"
#                         }
#                     )
#             elif self.module_result:
#                 self.output_data = self.module_result.data
#
#     # 定义OrderTask
#     class OrderTask(LLMTask):
#         async def post_process(self):
#             if self.module_result:
#                 self.output_data = self.module_result.data
#
#
#     async def test_serial_workflow_customer_service(self):
#         """测试串行工作流：客服助手"""
#         client = MockLLMClient(self.dynamic_response)
#         app = LLMWorkflow(client, config_dict=TEST_CONFIG)
#
#         # 启动工作流
#         start_task = EntryTask(
#             workflow=app,
#             task_id="entry",
#             module_name="test_module_with_input",
#             input_data={"input": "我的订单在哪里？"}
#         )
#
#         result_task = await app.run(start_task)
#         self.assertEqual(result_task.status, LLMTaskStatus.COMPLETED)
#         self.assertEqual(result_task.output_data, {"result": "你的订单12345状态是已发货"})
#         self.assertEqual(len(client.requests), 3)
#
#     async def test_nested_workflow_file_scan(self):
#         """测试嵌套工作流：逐层扫描文件"""
#         def entry_response(request):
#             tool_call = ToolCall(
#                 name="FolderAnalyzeTask",
#                 arguments_text='{"folder_path": "root"}',
#                 arguments={"folder_path": "root"}
#             )
#             resp = LLMProviderResponse(
#                 client_id=request.client_id,
#                 context_id=request.context_id,
#                 request_id=request.request_id,
#                 model_name=request.model_name,
#                 stream=False,
#                 finish_reason=FinishReason.tool_calls.value,
#                 response_tool_calls=[tool_call],
#                 response_message={"tool_calls": [tool_call._asdict()]}
#             )
#             return resp
#
#         def folder_response(request):
#             folder_path = request.messages[-1].content.split(":")[1].strip()
#             resp = LLMProviderResponse(
#                 client_id=request.client_id,
#                 context_id=request.context_id,
#                 request_id=request.request_id,
#                 model_name=request.model_name,
#                 stream=False,
#                 finish_reason=FinishReason.stop.value,
#                 response_text=json.dumps({
#                     "result": f"{folder_path}包含file1.txt和subfolder"
#                 }),
#                 response_message={"content": f"{folder_path}包含file1.txt和subfolder"}
#             )
#             return resp
#
#         def file_response(request):
#             file_path = request.messages[-1].content.split(":")[1].strip()
#             resp = LLMProviderResponse(
#                 client_id=request.client_id,
#                 context_id=request.context_id,
#                 request_id=request.request_id,
#                 model_name=request.model_name,
#                 stream=False,
#                 finish_reason=FinishReason.stop.value,
#                 response_text=json.dumps({
#                     "result": f"{file_path}内容：测试文件内容"
#                 }),
#                 response_message={"content": f"{file_path}内容：测试文件内容"}
#             )
#             return resp
#
#         def dynamic_response(request):
#             messages = request.messages
#             if not messages:
#                 return default_llm_response(request)
#             last_msg = messages[-1].content
#             if "扫描root文件夹" in last_msg:
#                 return entry_response(request)
#             elif "folder:" in last_msg:
#                 return folder_response(request)
#             elif "file:" in last_msg:
#                 return file_response(request)
#             return default_llm_response(request)
#
#         client = MockLLMClient(dynamic_response)
#         app = LLMWorkflow(client, config_dict=TEST_CONFIG, max_parallel_tasks=2)
#
#         # 文件分析任务
#         class FileAnalyzeTask(LLMTask):
#             module_name = 'test_module_with_input'
#
#             async def pre_process(self):
#                 return LLMModuleRequest(
#                     module_name=self.module_name,
#                     template_params={"input": f"file:{self.input_data['file_path']}"}
#                 )
#
#             async def post_process(self):
#                 if self.module_result:
#                     self.output_data = self.module_result.data
#
#         # 文件夹分析任务（递归）
#         class FolderAnalyzeTask(LLMTask):
#             module_name = 'test_module_with_input'
#             # input_data
#             # path: str
#             # is_root: bool
#
#             async def pre_process(self):
#                 folder_path = self.input_data['folder_path']
#
#                 # 模拟生成子任务
#                 tasks = []
#                 if folder_path == "root":
#                     tasks.append(FileAnalyzeTask(
#                         {"file_path": "root/file1.txt"}
#                     ))
#                     tasks.append(FolderAnalyzeTask(
#                         {"folder_path": "root/subfolder"}
#                     ))
#                 elif folder_path == "root/subfolder1":
#                     tasks.append(FileAnalyzeTask(
#                         input_data={"file_path": "root/subfolder1/file2.txt"}
#                     ))
#
#                 if tasks:
#                     await self.workflow.wait_tasks(*tasks)
#                     self.output_data["children"] = [t.output_data for t in tasks]
#
#                 return LLMModuleRequest(
#                     module_name=self.module_name,
#                     template_params={"input": f"folder:{self.input_data['folder_path']}"}
#                 )
#
#             async def post_process(self):
#                 if not self.module_result:
#                     return
#
#                 folder_path = self.input_data['folder_path']
#                 self.output_data = {
#                     "folder": folder_path,
#                     "result": self.module_result.data
#                 }
#
#
#         # 入口任务
#         class EntryTask(LLMTask):
#             module_name = "test_module_with_input"
#
#             async def post_process(self):
#                 self.on_tool_call_goto(FolderAnalyzeTask=FolderAnalyzeTask)
#
#         fs = {
#             "root": ["file1.txt", "sub1/"],
#             "root/sub1": ["file2.txt", "sub2/"],
#             "root/sub1/sub2": ["file3.txt", "file4.txt"],
#         }
#         # 启动工作流
#         start_task = EntryTask(
#             task_id="entry",
#             input_data={"input": "用户提问：扫描root文件夹"}
#         )
#
#         result_task = await app.run(start_task)
#         self.assertEqual(result_task.status, LLMTaskStatus.COMPLETED)
#         self.assertEqual(result_task.output_data["folder"], "root")
#         self.assertEqual(len(result_task.output_data["children"]), 2)
#         self.assertEqual(len(client.requests), 5)  # entry + root + file1 + subfolder + file2


class TestLLMWorkflowNestedUseCase(unittest.IsolatedAsyncioTestCase):
    async def test_nested_workflow_file_scan(self):
        """测试嵌套工作流：逐层扫描文件"""
        def entry_response(request):
            tool_call = ToolCall(
                name="FolderAnalyzeTask",
                arguments_text='{"folder_path": "root"}',
                arguments={"folder_path": "root"}
            )
            resp = LLMProviderResponse(
                client_id=request.client_id,
                context_id=request.context_id,
                request_id=request.request_id,
                model_name=request.model_name,
                stream=False,
                finish_reason=FinishReason.tool_calls.value,
                response_tool_calls=[tool_call],
                response_message={"tool_calls": [tool_call._asdict()]}
            )
            return resp

        def folder_response(request):
            folder_path = request.messages[-1].content.split(":")[1].strip()
            resp = LLMProviderResponse(
                client_id=request.client_id,
                context_id=request.context_id,
                request_id=request.request_id,
                model_name=request.model_name,
                stream=False,
                finish_reason=FinishReason.stop.value,
                response_text=json.dumps({
                    "result": f"{folder_path}包含file1.txt和subfolder"
                }),
                response_message={"content": f"{folder_path}包含file1.txt和subfolder"}
            )
            return resp

        def file_response(request):
            file_path = request.messages[-1].content.split(":")[1].strip()
            resp = LLMProviderResponse(
                client_id=request.client_id,
                context_id=request.context_id,
                request_id=request.request_id,
                model_name=request.model_name,
                stream=False,
                finish_reason=FinishReason.stop.value,
                response_text=json.dumps({
                    "result": f"{file_path}内容：测试文件内容"
                }),
                response_message={"content": f"{file_path}内容：测试文件内容"}
            )
            return resp

        def dynamic_response(request):
            messages = request.messages
            if not messages:
                return default_llm_response(request)
            last_msg = messages[-1].content
            if "扫描root文件夹" in last_msg:
                return entry_response(request)
            elif "folder:" in last_msg:
                return folder_response(request)
            elif "file:" in last_msg:
                return file_response(request)
            return default_llm_response(request)

        client = MockLLMClient(dynamic_response)
        app = LLMWorkflow(client, config_dict=TEST_CONFIG, max_parallel_tasks=2)

        # 文件分析任务
        class FileAnalyzeTask(LLMTask):
            module_name = 'test_module_with_input'

            async def pre_process(self):
                return LLMModuleRequest(
                    module_name=self.module_name,
                    template_params={"input": f"file:{self.input_data['file_path']}"}
                )

            async def post_process(self):
                if self.module_result:
                    self.output_data = self.module_result.data

        # 文件夹分析任务（递归）
        class FolderAnalyzeTask(LLMTask):
            module_name = 'test_module_with_input'
            # input_data
            # path: str
            # is_root: bool

            async def pre_process(self):
                folder_path = self.input_data['folder_path']

                # 模拟生成子任务
                tasks = []
                if folder_path == "root":
                    tasks.append(FileAnalyzeTask(
                        {"file_path": "root/file1.txt"}
                    ))
                    tasks.append(FolderAnalyzeTask(
                        {"folder_path": "root/subfolder"}
                    ))
                elif folder_path == "root/subfolder1":
                    tasks.append(FileAnalyzeTask(
                        input_data={"file_path": "root/subfolder1/file2.txt"}
                    ))

                if tasks:
                    await self.workflow.wait_tasks(*tasks)
                    self.output_data["children"] = [t.output_data for t in tasks]

                return LLMModuleRequest(
                    module_name=self.module_name,
                    template_params={"input": f"folder:{self.input_data['folder_path']}"}
                )

            async def post_process(self):
                if not self.module_result:
                    return

                folder_path = self.input_data['folder_path']
                self.output_data = {
                    "folder": folder_path,
                    "result": self.module_result.data
                }


        # 入口任务
        class EntryTask(LLMTask):
            module_name = "test_module_with_input"

            async def post_process(self):
                self.on_tool_call_goto(FolderAnalyzeTask=FolderAnalyzeTask)

        fs = {
            "root": ["file1.txt", "sub1/"],
            "root/sub1": ["file2.txt", "sub2/"],
            "root/sub1/sub2": ["file3.txt", "file4.txt"],
        }
        # 启动工作流
        start_task = EntryTask(
            task_id="entry",
            input_data={"input": "用户提问：扫描root文件夹"}
        )

        result_task = await app.run(start_task)
        self.assertEqual(result_task.status, LLMTaskStatus.COMPLETED)
        self.assertEqual(result_task.output_data["folder"], "root")
        self.assertEqual(len(result_task.output_data["children"]), 2)
        self.assertEqual(len(client.requests), 5)  # entry + root + file1 + subfolder + file2


class TestLLMWorkflowEdgeCases(unittest.IsolatedAsyncioTestCase):
    """测试工作流的边界情况和高级特性"""

    async def test_wait_tasks_dependency_graph(self):
        """测试wait_tasks处理显式依赖图：C依赖A和B，A/B完成后C才执行"""
        client = MockLLMClient()
        app = LLMWorkflow(client, config_dict=TEST_CONFIG)

        class SimpleTask(LLMTask):
            module_name = "test_module_with_input"

        task_a = SimpleTask(workflow=app, task_id="A", input_data={"input": "A"})
        task_b = SimpleTask(workflow=app, task_id="B", input_data={"input": "B"})
        task_c = SimpleTask(workflow=app, task_id="C", input_data={"input": "C"})

        # 构建依赖: C -> A, C -> B
        app.add_task(task_c, task_a)
        app.add_task(task_c, task_b)

        # 等待所有任务完成
        await app.wait_tasks(task_a, task_b, task_c)

        # 检查状态
        self.assertEqual(task_a.status, LLMTaskStatus.COMPLETED)
        self.assertEqual(task_b.status, LLMTaskStatus.COMPLETED)
        self.assertEqual(task_c.status, LLMTaskStatus.COMPLETED)

        # 检查请求顺序：A和B必须在C之前完成（虽然A和B之间可能并行）
        # 模板内容是 "{{input}}"，所以response_text会包含 "A", "B", "C"
        # 我们通过client.requests中的response_text来推断顺序
        req_contents = [r.messages[-1].content for r in client.requests]
        # 找到 "A", "B", "C" 的索引
        idx_a = req_contents.index("A")
        idx_b = req_contents.index("B")
        idx_c = req_contents.index("C")

        self.assertLess(idx_a, idx_c, "Task A should finish before Task C")
        self.assertLess(idx_b, idx_c, "Task B should finish before Task C")

    async def test_wait_tasks_timeout(self):
        """测试wait_tasks超时机制"""
        async def slow_response(request):
            await asyncio.sleep(1.0)
            return default_llm_response(request)

        client = MockLLMClient(slow_response)
        app = LLMWorkflow(client, config_dict=TEST_CONFIG)

        task = LLMTaskDefaultModule(workflow=app, task_id="slow", input_data={})

        with self.assertRaises(asyncio.TimeoutError):
            await app.wait_tasks(task, timeout=0.1)

        # 任务可能还在运行或失败，这里主要测试wait_tasks抛出异常
        # 由于超时，wait_tasks不会等待任务完成，任务状态可能是RUNNING或COMPLETED（如果竞态）
        # 但我们主要关注异常抛出

    async def test_wait_tasks_child_failure(self):
        """测试wait_tasks中子任务失败导致整体失败及依赖传播"""
        def fail_response(request):
            resp = default_llm_response(request)
            resp.finish_reason = FinishReason.error_app.value
            resp.error_text = "Child failed"
            return resp

        client = MockLLMClient(fail_response)
        app = LLMWorkflow(client, config_dict=TEST_CONFIG)

        task_a = LLMTaskDefaultModule(workflow=app, task_id="A", input_data={})
        task_b = LLMTaskDefaultModule(workflow=app, task_id="B", input_data={})

        app.add_task(task_b, task_a)  # B depends on A

        with self.assertRaises(LLMTaskDependencyError):
            await app.wait_tasks(task_a, task_b)

        self.assertEqual(task_a.status, LLMTaskStatus.FAILED)
        self.assertEqual(task_b.status, LLMTaskStatus.DEPENDENCY_FAILED)

    async def test_task_double_execution_protection(self):
        """测试任务不会被执行两次"""
        client = MockLLMClient()
        app = LLMWorkflow(client, config_dict=TEST_CONFIG)

        task = LLMTaskDefaultModule(workflow=app, task_id="double", input_data={})

        await app.wait_tasks(task)
        await app.wait_tasks(task)

        self.assertEqual(len(client.requests), 1)
        self.assertEqual(task.status, LLMTaskStatus.COMPLETED)

    async def test_wait_tasks_empty_list(self):
        """测试wait_tasks传入空列表"""
        client = MockLLMClient()
        app = LLMWorkflow(client, config_dict=TEST_CONFIG)

        # 不应抛出异常
        await app.wait_tasks()
        self.assertEqual(len(client.requests), 0)

    async def test_complex_dependency_chain_failure(self):
        """测试复杂依赖链中的失败传播：A->B->C，A失败导致B和C都失败"""
        def fail_response(request):
            resp = default_llm_response(request)
            resp.finish_reason = FinishReason.error_app.value
            resp.error_text = "Root failed"
            return resp

        client = MockLLMClient(fail_response)
        app = LLMWorkflow(client, config_dict=TEST_CONFIG)

        task_a = LLMTaskDefaultModule(task_id="A", input_data={})
        task_b = LLMTaskDefaultModule(task_id="B", input_data={})
        task_c = LLMTaskDefaultModule(task_id="C", input_data={})

        app.add_task(task_b, task_a) # B depends on A
        app.add_task(task_c, task_b) # C depends on B

        with self.assertRaises(LLMTaskDependencyError):
            await app.wait_tasks(task_a, task_b, task_c)

        self.assertEqual(task_a.status, LLMTaskStatus.FAILED)
        self.assertEqual(task_b.status, LLMTaskStatus.DEPENDENCY_FAILED)
        self.assertEqual(task_c.status, LLMTaskStatus.DEPENDENCY_FAILED)
        # 只有A被执行了
        self.assertEqual(len(client.requests), 1)


if __name__ == '__main__':
    unittest.main()
