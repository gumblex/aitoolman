import re
import logging
import asyncio
import unittest

import aitoolman

from aitoolman.test import mock_llmclient

logging.basicConfig(level="DEBUG", format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger('asyncio').setLevel(logging.INFO)


TEST_CONFIG = aitoolman.load_config_str("""
[module.test_module_with_input]
model = "test_model"
template.user = "{{input}}"
""")

# ---------------------------------------------------------------------------
# 辅助：基础 Task 子类
# ---------------------------------------------------------------------------

class SuccessTask(aitoolman.Task):
    async def run(self):
        return self.input_data['x'] * 2

class FailingTask(aitoolman.Task):
    async def run(self):
        raise ValueError("Dependency failed")

class SlowTask(aitoolman.Task):
    async def run(self):
        await asyncio.sleep(self.input_data['sec'])
        return self.input_data['sec']

class CounterTask(aitoolman.Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0

    async def run(self):
        self.counter += 1
        return self.counter

class RecorderTask(aitoolman.Task):
    execution_order = []

    async def run(self):
        self.execution_order.append(self.task_id)
        return self.task_id

# ---------------------------------------------------------------------------
# 基础测试
# ---------------------------------------------------------------------------

class TestTaskBasic(unittest.IsolatedAsyncioTestCase):
    """测试 Task 的基础功能"""

    async def test_task_success(self):
        """测试任务成功执行"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        task = SuccessTask({"x": 1, "y": 2}, workflow=app)
        await app.wait_tasks(task)

        self.assertEqual(task.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(task.output_data, 2)

    async def test_task_failure(self):
        """测试任务执行失败"""
        def failing_func():
            raise ValueError("Function failed")

        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        task = aitoolman.Task({}, workflow=app)
        task.set_func(failing_func)

        with self.assertRaises(aitoolman.LLMWorkflowError) as ctx:
            await app.wait_tasks(task)

        self.assertEqual(task.status, aitoolman.TaskStatus.FAILED)
        self.assertIn("Function failed", str(task.error))
        self.assertIn("1 tasks failed", str(ctx.exception))

    async def test_inherit_task_override_run(self):
        """测试继承 Task 并重写 run 方法"""
        class CustomTask(aitoolman.Task):
            def __init__(self, value, workflow):
                super().__init__()
                self.input_data = {"value": value}
                self.workflow = workflow

            async def run(self):
                return self.input_data['value'] * 2

        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        task = CustomTask(5, workflow=app)
        await app.wait_tasks(task)

        self.assertEqual(task.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(task.output_data, 10)


# ---------------------------------------------------------------------------
# 串行执行逻辑
# ---------------------------------------------------------------------------

class TestSerialExecution(unittest.IsolatedAsyncioTestCase):
    """测试串行执行链（使用 next_task 和 run/wait_tasks）"""

    async def test_single_task_with_run(self):
        """单个任务通过 run 执行"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        task = SuccessTask({"x": 3})
        last = await app.run(task)

        self.assertEqual(last, task)
        self.assertEqual(last.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(last.output_data, 6)

    async def test_chain_of_two_tasks(self):
        """两个任务串行：task_a -> task_b"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        task_a = SuccessTask({"x": 2})
        task_b = SuccessTask({"x": 5})
        task_a.next_task = task_b

        last = await app.run(task_a)

        self.assertEqual(last, task_b)
        self.assertEqual(task_a.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(task_b.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(task_a.output_data, 4)
        self.assertEqual(task_b.output_data, 10)

    async def test_chain_with_failure(self):
        """链中某个任务失败，run 应该抛出异常并停止"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        task_a = SuccessTask({"x": 2})
        task_b = FailingTask({})
        task_c = SuccessTask({"x": 3})
        task_a.next_task = task_b
        task_b.next_task = task_c

        with self.assertRaises(aitoolman.LLMWorkflowError):
            await app.run(task_a)

        self.assertEqual(task_a.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(task_b.status, aitoolman.TaskStatus.FAILED)
        # task_c 不应该被执行
        self.assertEqual(task_c.status, aitoolman.TaskStatus.INIT)

    async def test_chain_via_submit(self):
        """通过 submit 提交链首任务，自动推进整个链"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        task_a = SuccessTask({"x": 1})
        task_b = SuccessTask({"x": 2})
        task_c = SuccessTask({"x": 3})
        task_a.next_task = task_b
        task_b.next_task = task_c

        # 只提交链首，链式自动推进
        await app.submit(task_a)
        # 等待链上所有任务完成
        for t in [task_a, task_b, task_c]:
            await t.status_event.wait()

        self.assertEqual(task_a.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(task_b.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(task_c.status, aitoolman.TaskStatus.COMPLETED)


# ---------------------------------------------------------------------------
# 支线与子任务
# ---------------------------------------------------------------------------

class TestBranchingSubTasks(unittest.IsolatedAsyncioTestCase):
    """测试在任务内部启动支线（wait_tasks / submit）"""

    async def test_subtask_wait_inside_task(self):
        """父任务内部 wait_tasks 等待两个子任务并行完成（使用 release_worker 释放并发配额）"""
        sub_log = []

        class SubTask(aitoolman.Task):
            async def run(self):
                await asyncio.sleep(0.05)
                sub_log.append(self.input_data['name'])
                return len(sub_log)

        class ParentTask(aitoolman.Task):
            async def run(self):
                sub_a = SubTask({"name": "A"})
                sub_b = SubTask({"name": "B"})
                async with self.workflow.release_worker():
                    await self.workflow.wait_tasks(sub_a, sub_b)
                return sub_a.output_data + sub_b.output_data

        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        parent = ParentTask()
        await app.wait_tasks(parent)

        self.assertEqual(parent.status, aitoolman.TaskStatus.COMPLETED)
        # 俩子任务都执行，A 先写入，B 后写入，但 sub_log 至少包含 A 和 B
        self.assertIn("A", sub_log)
        self.assertIn("B", sub_log)
        self.assertEqual(parent.output_data, 3)  # 1+2

    async def test_nested_chains(self):
        """子任务链嵌套：父任务启动一个链，等待链尾任务"""
        class LeafTask(aitoolman.Task):
            async def run(self):
                return self.input_data['value'] * 10

        class ParentTask(aitoolman.Task):
            async def run(self):
                t1 = LeafTask({"value": 1})
                t2 = LeafTask({"value": 2})
                t1.next_task = t2
                async with self.workflow.release_worker():
                    # 只提交链首，自动推进到链尾
                    await self.workflow.submit(t1)
                    await t2.status_event.wait()
                return t2.output_data

        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        parent = ParentTask()
        await app.wait_tasks(parent)
        self.assertEqual(parent.output_data, 20)

    async def test_submit_without_wait(self):
        """submit 之后不等待，父任务先完成，子任务继续执行"""
        log = []

        class DelayedTask(aitoolman.Task):
            async def run(self):
                await asyncio.sleep(0.1)
                log.append("delayed_done")
                return True

        class FireAndForgetTask(aitoolman.Task):
            async def run(self):
                sub = DelayedTask()
                await self.workflow.submit(sub)
                log.append("parent_done")
                return "fired"

        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        task = FireAndForgetTask()
        await app.wait_tasks(task)
        # 父任务已完成，但子任务可能未完成
        self.assertEqual(task.output_data, "fired")
        # 给子任务一些时间完成
        await asyncio.sleep(0.2)
        self.assertIn("delayed_done", log)
        # 日志顺序：父任务先完成，子任务后完成（因为不等待）
        self.assertLess(log.index("parent_done"), log.index("delayed_done"))


# ---------------------------------------------------------------------------
# 静态依赖路径与 DAG
# ---------------------------------------------------------------------------

class TestStaticDependencyDAG(unittest.IsolatedAsyncioTestCase):
    """通过 next_task 和内部 wait_tasks 实现静态 DAG（扇出/扇入）"""

    async def test_fan_out_fan_in(self):
        """扇出：A -> [B, C] -> D（汇聚）"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        class FanOutTask(aitoolman.Task):
            async def run(self):
                self.b = SuccessTask({"x": 2})
                self.c = SuccessTask({"x": 3})
                async with self.workflow.release_worker():
                    await self.workflow.wait_tasks(self.b, self.c)
                # 将汇聚任务作为 next_task 交给工作流自动执行，
                # 或者直接在此执行并返回。这里选择返回给 next_task。
                self.next_task = FanInTask(self.b, self.c)
                return "fanout_done"

        class FanInTask(aitoolman.Task):
            def __init__(self, b, c, **kwargs):
                super().__init__(**kwargs)
                self.b = b
                self.c = c

            async def run(self):
                return self.b.output_data + self.c.output_data

        # 仅用 run 执行链：FanOut -> FanIn
        start = FanOutTask()
        last = await app.run(start)

        self.assertIsInstance(last, FanInTask)
        self.assertEqual(last.output_data, 4 + 6)  # 2*2 + 2*3
        self.assertEqual(start.status, aitoolman.TaskStatus.COMPLETED)

    async def test_diamond_dag(self):
        """钻石型 DAG：A -> B, A -> C, B -> D, C -> D"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        class TaskA(aitoolman.Task):
            async def run(self):
                self.b = SuccessTask({"x": 2})
                self.c = SuccessTask({"x": 3})
                async with self.workflow.release_worker():
                    await self.workflow.wait_tasks(self.b, self.c)
                self.next_task = TaskD(self.b, self.c)
                return "A_done"

        class TaskD(aitoolman.Task):
            def __init__(self, b, c, **kwargs):
                super().__init__(**kwargs)
                self.b = b
                self.c = c

            async def run(self):
                return self.b.output_data * self.c.output_data

        start = TaskA()
        last = await app.run(start)

        self.assertIsInstance(last, TaskD)
        self.assertEqual(last.output_data, 4 * 6)  # 2*2 * 2*3
        self.assertEqual(start.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(start.b.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(start.c.status, aitoolman.TaskStatus.COMPLETED)

    async def test_static_chain_via_next_task(self):
        """静态链：创建任务时已经连接好 next_task"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        t1 = SuccessTask({"x": 1})
        t2 = SuccessTask({"x": 2})
        t3 = SuccessTask({"x": 3})
        t1.next_task = t2
        t2.next_task = t3

        last = await app.run(t1)
        self.assertEqual(last, t3)
        self.assertEqual(t1.output_data, 2)
        self.assertEqual(t2.output_data, 4)
        self.assertEqual(t3.output_data, 6)

    async def test_dependency_failure_propagation_in_dag(self):
        """当汇聚任务依赖的一个任务失败时，wait_tasks 会抛出异常并导致父任务失败"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        class ParentWithFailingChild(aitoolman.Task):
            async def run(self):
                good = SuccessTask({"x": 5})
                bad = FailingTask({})
                try:
                    async with self.workflow.release_worker():
                        await self.workflow.wait_tasks(good, bad)
                except aitoolman.LLMWorkflowError:
                    raise RuntimeError("child task failed, cannot proceed")

        task = ParentWithFailingChild()
        with self.assertRaises(aitoolman.LLMWorkflowError):
            await app.wait_tasks(task)

        self.assertEqual(task.status, aitoolman.TaskStatus.FAILED)
        self.assertIn("child task failed", str(task.error))

    async def test_wait_tasks_timeout(self):
        """测试 wait_tasks 超时机制"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        task = SlowTask({"sec": 0.5})

        with self.assertRaises(asyncio.TimeoutError):
            await app.wait_tasks(task, timeout=0.1)

    async def test_task_double_execution_protection(self):
        """测试任务不会被执行两次"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        task = CounterTask()

        await app.wait_tasks(task)
        await app.wait_tasks(task)

        self.assertEqual(task.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(task.counter, 1)

    async def test_max_parallel_tasks(self):
        """测试并行执行任务数不超过 max_parallel_tasks"""
        max_parallel = 2
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG, max_parallel_tasks=max_parallel)

        current_running = 0
        max_observed = 0

        class TrackedSlowTask(aitoolman.Task):
            async def run(self):
                nonlocal current_running, max_observed
                current_running += 1
                max_observed = max(max_observed, current_running)
                await asyncio.sleep(0.1)
                current_running -= 1
                return True

        tasks = [TrackedSlowTask() for _ in range(5)]
        await app.wait_tasks(*tasks)

        self.assertTrue(all(t.status == aitoolman.TaskStatus.COMPLETED for t in tasks))
        self.assertLessEqual(max_observed, max_parallel,
                             f"Observed {max_observed} concurrent tasks, exceeding max {max_parallel}")


# ---------------------------------------------------------------------------
# release_worker 机制
# ---------------------------------------------------------------------------

class TestReleaseWorker(unittest.IsolatedAsyncioTestCase):
    """测试 release_worker 上下文管理器"""

    async def test_release_worker_prevents_deadlock(self):
        """低并行度下，release_worker 防止父子任务死锁"""
        max_parallel = 1
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG, max_parallel_tasks=max_parallel)

        class ParentWithSubtasks(aitoolman.Task):
            async def run(self):
                sub_a = SuccessTask({"x": 1})
                sub_b = SuccessTask({"x": 2})
                async with self.workflow.release_worker():
                    await self.workflow.wait_tasks(sub_a, sub_b)
                return sub_a.output_data + sub_b.output_data

        task = ParentWithSubtasks()
        await app.wait_tasks(task)
        self.assertEqual(task.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(task.output_data, 2 + 4)  # 1*2 + 2*2

    async def test_release_worker_outside_task_raises(self):
        """在任务外部调用 release_worker 应抛出 RuntimeError"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        with self.assertRaises(RuntimeError):
            async with app.release_worker():
                pass

    async def test_release_worker_nested_subtasks(self):
        """release_worker 支持多层嵌套子任务"""
        max_parallel = 2
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG, max_parallel_tasks=max_parallel)

        class InnerTask(aitoolman.Task):
            async def run(self):
                return self.input_data['x'] * 3

        class MiddleTask(aitoolman.Task):
            async def run(self):
                inner = InnerTask({"x": self.input_data['x']})
                async with self.workflow.release_worker():
                    await self.workflow.wait_tasks(inner)
                return inner.output_data + 1

        class OuterTask(aitoolman.Task):
            async def run(self):
                mid = MiddleTask({"x": 5})
                async with self.workflow.release_worker():
                    await self.workflow.wait_tasks(mid)
                return mid.output_data + 10

        task = OuterTask()
        await app.wait_tasks(task)
        self.assertEqual(task.status, aitoolman.TaskStatus.COMPLETED)
        # MiddleTask: 5*3 + 1 = 16; OuterTask: 16 + 10 = 26
        self.assertEqual(task.output_data, 26)

    async def test_release_worker_with_run(self):
        """release_worker 配合 workflow.run 使用"""
        max_parallel = 1
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG, max_parallel_tasks=max_parallel)

        class ParentWithChain(aitoolman.Task):
            async def run(self):
                t1 = SuccessTask({"x": 3})
                t2 = SuccessTask({"x": 4})
                t1.next_task = t2
                async with self.workflow.release_worker():
                    last = await self.workflow.run(t1)
                return last.output_data

        task = ParentWithChain()
        await app.wait_tasks(task)
        self.assertEqual(task.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(task.output_data, 8)  # 4*2

    async def test_release_worker_exception_safety(self):
        """测试release_worker在with块抛出异常时仍能正确重新获取配额"""
        max_parallel = 1
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG, max_parallel_tasks=max_parallel)

        class ExceptionInReleaseTask(aitoolman.Task):
            async def run(self):
                try:
                    async with self.workflow.release_worker():
                        raise RuntimeError("Intentional error in release block")
                except RuntimeError:
                    pass
                # 确认退出with块后仍然持有配额，可以正常执行后续逻辑
                sub = SuccessTask({"x": 5})
                async with self.workflow.release_worker():
                    await self.workflow.wait_tasks(sub)
                return sub.output_data

        task = ExceptionInReleaseTask()
        await app.wait_tasks(task)
        self.assertEqual(task.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(task.output_data, 10)
        # 确认配额正常，后续任务可以执行
        task2 = SuccessTask({"x": 10})
        await app.wait_tasks(task2)
        self.assertEqual(task2.output_data, 20)


# ---------------------------------------------------------------------------
# 工作流资源清理
# ---------------------------------------------------------------------------

class TestWorkflowCleanup(unittest.IsolatedAsyncioTestCase):
    async def test_no_residual_tasks_after_completion(self):
        """测试任务完成后无残留的 task 和 exec_task"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        tasks = [
            SuccessTask({"x": i}, workflow=app)
            for i in range(3)
        ]
        await app.wait_tasks(*tasks)

        # 等待 done_callback 清理 _exec_tasks
        await asyncio.sleep(0.05)

        self.assertEqual(len(app._active_tasks), 0)
        self.assertEqual(len(app._exec_tasks), 0)

    async def test_workflow_async_context_manager(self):
        """测试 async with 语法自动停止工作流"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG, max_parallel_tasks=2)

        async with app:
            t = SlowTask({"sec": 0.1})
            await app.wait_tasks(t)

        # 退出上下文后工作流应停止
        self.assertTrue(app._stopped)

    async def test_stop_cancels_running_tasks(self):
        """测试stop方法会取消所有正在运行的任务"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        task_started = asyncio.Event()
        task_cancelled = False

        class CancellableTask(aitoolman.Task):
            async def run(self):
                nonlocal task_cancelled
                task_started.set()
                try:
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    task_cancelled = True
                    raise

        task = CancellableTask()
        # 提交任务不等待
        await app.submit(task)
        # 等待任务开始运行
        await task_started.wait()
        # 停止工作流
        await app.stop()
        # 确认任务被取消
        await asyncio.sleep(0.05)
        self.assertEqual(task.status, aitoolman.TaskStatus.FAILED)
        self.assertTrue(task_cancelled)
        self.assertTrue(app._stopped)
        self.assertEqual(len(app._active_tasks), 0)
        self.assertEqual(len(app._exec_tasks), 0)


# ---------------------------------------------------------------------------
# LLMTask 工具调用支持
# ---------------------------------------------------------------------------

class TestLLMTaskToolSupport(unittest.IsolatedAsyncioTestCase):
    """测试 LLMTask 的工具调用处理（on_tool_call_goto 和 run_tool_calls）"""

    async def test_on_tool_call_goto(self):
        """测试 on_tool_call_goto 跳转到下一个 Task"""
        client = mock_llmclient.MockLLMClient(lambda req: mock_llmclient.make_simple_response(
            req, [aitoolman.ToolCall.new("next_task", x=42)]
        ))
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        class StartTask(aitoolman.LLMTask):
            async def post_process(self, module_result):
                self.on_tool_call_goto(next_task=SuccessTask)

        start_task = StartTask(aitoolman.LLMModuleRequest(
            module_name="test_module_with_input",
            template_params={"input": "[start_task]"}
        ))

        result_task = await app.run(start_task)
        self.assertEqual(result_task.status, aitoolman.TaskStatus.COMPLETED)
        self.assertIsInstance(result_task, SuccessTask)
        self.assertEqual(result_task.output_data, 84)

    async def test_run_tool_calls_single(self):
        """测试 run_tool_calls 单轮工具调用"""
        def _response_generator(request):
            last_msg = request.messages[-1]
            if last_msg.role == aitoolman.MessageRole.user.value:
                return mock_llmclient.make_simple_response(
                    request, [aitoolman.ToolCall.new("test_tool", x=42)])
            else:
                return mock_llmclient.make_tool_call_response(request)

        client = mock_llmclient.MockLLMClient(_response_generator)
        app_prompt = aitoolman.load_config_str("""
        [module.tool_call_module]
        model = "test_model"
        template.user = "[start_task]"
        post_processor = "builtin.parse_json"

        [module.tool_call_module.tools."test_tool"]
        type = "function"
        description = "Tool description"
        param."x".type = "integer"
        param."x".description = "x"
        param."x".required = true
        """)
        app = aitoolman.LLMWorkflow(client, config_dict=app_prompt)

        def test_tool(x):
            return x * 2

        class ToolTask(aitoolman.LLMTask):
            async def post_process(self, module_result):
                await self.run_tool_calls(test_tool=test_tool)

        start_task = ToolTask(
            aitoolman.LLMModuleRequest(module_name="tool_call_module", template_params={})
        )

        result_task = await app.run(start_task)
        self.assertEqual(result_task.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(result_task.output_data[0]['result'], "84")

    async def test_run_tool_calls_multiple(self):
        """测试 run_tool_calls 一次调用多个工具"""
        def _response_generator(request):
            last_msg = request.messages[-1]
            if last_msg.role == aitoolman.MessageRole.user.value:
                return mock_llmclient.make_simple_response(request, [
                    aitoolman.ToolCall.new("multiply", x=2, y=3),
                    aitoolman.ToolCall.new("add", a=5, b=7)
                ])
            else:
                return mock_llmclient.make_tool_call_response(request)

        client = mock_llmclient.MockLLMClient(_response_generator)
        app_prompt = aitoolman.load_config_str("""
        [module.multi_tool_module]
        model = "test_model"
        template.user = "[start_task]"
        post_processor = "builtin.parse_json"

        [module.multi_tool_module.tools."multiply"]
        type = "function"
        description = "Multiply two numbers"
        param."x".type = "integer"
        param."y".type = "integer"
        param."x".required = true
        param."y".required = true

        [module.multi_tool_module.tools."add"]
        type = "function"
        description = "Add two numbers"
        param."a".type = "integer"
        param."b".type = "integer"
        param."a".required = true
        param."b".required = true
        """)
        app = aitoolman.LLMWorkflow(client, config_dict=app_prompt)

        def multiply(x, y):
            return x * y

        def add(a, b):
            return a + b

        class MultiToolTask(aitoolman.LLMTask):
            async def post_process(self, module_result):
                await self.run_tool_calls(multiply=multiply, add=add)

        start_task = MultiToolTask(
            aitoolman.LLMModuleRequest(module_name="multi_tool_module", template_params={})
        )

        result_task = await app.run(start_task)
        self.assertEqual(result_task.status, aitoolman.TaskStatus.COMPLETED)

        results = result_task.output_data
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['arguments'], {'x': 2, 'y': 3})
        self.assertEqual(results[0]['result'], "6")
        self.assertEqual(results[1]['arguments'], {'a': 5, 'b': 7})
        self.assertEqual(results[1]['result'], "12")


# ---------------------------------------------------------------------------
# 工具调用典型用例（使用 Tool Calls）
# ---------------------------------------------------------------------------

def get_weather(city, date):
    if city == "北京":
        return f"{date}{city}的天气是：晴，20-28度"
    elif city == "上海":
        return f"{date}{city}的天气是：多云，22-30度"
    return "未知天气"


class TestLLMWorkflowToolUseCase(unittest.IsolatedAsyncioTestCase):
    """测试 LLMWorkflow + LLMTask 使用工具调用的常见用例"""

    async def test_tool_call_as_intent_recognition(self):
        """用例：工具调用作为意图识别，跳转到下一个 Task"""
        def _response_generator(request):
            last_msg = request.messages[-1]
            if last_msg.role == aitoolman.MessageRole.user.value:
                user_input = last_msg.content
                if "你好" in user_input:
                    return mock_llmclient.make_simple_response(
                        request, [aitoolman.ToolCall.new("greet", name="用户")])
                elif "计算" in user_input:
                    match = re.search(r'计算(\d+)\+(\d+)', user_input)
                    if match:
                        return mock_llmclient.make_simple_response(request, [
                            aitoolman.ToolCall.new("calculate", a=int(match.group(1)), b=int(match.group(2)))
                        ])
                return mock_llmclient.make_simple_response(request, [aitoolman.ToolCall.new("unknown")])
            else:
                return mock_llmclient.make_tool_call_response(request)

        client = mock_llmclient.MockLLMClient(_response_generator)
        app_prompt = aitoolman.load_config_str("""
        [module.intent_recognition]
        model = "test_model"
        template.user = "{{user_input}}"
        """)
        app = aitoolman.LLMWorkflow(client, config_dict=app_prompt)

        class GreetTask(aitoolman.Task):
            async def run(self):
                return f"你好，{self.input_data['name']}！"

        class CalculateTask(aitoolman.Task):
            async def run(self):
                return f"结果是：{self.input_data['a'] + self.input_data['b']}"

        class UnknownTask(aitoolman.Task):
            async def run(self):
                return "抱歉，我不明白你的意思。"

        class IntentTask(aitoolman.LLMTask):
            async def post_process(self, module_result):
                self.on_tool_call_goto(
                    greet=GreetTask,
                    calculate=CalculateTask,
                    unknown=UnknownTask
                )

        # 测试问候
        start = IntentTask(aitoolman.LLMModuleRequest(
            module_name="intent_recognition", template_params={"user_input": "你好"}))
        result = await app.run(start)
        self.assertEqual(result.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(result.output_data, "你好，用户！")

        # 测试计算
        start = IntentTask(aitoolman.LLMModuleRequest(
            module_name="intent_recognition", template_params={"user_input": "计算1+2"}))
        result = await app.run(start)
        self.assertEqual(result.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(result.output_data, "结果是：3")

        # 测试未知
        start = IntentTask(aitoolman.LLMModuleRequest(
            module_name="intent_recognition", template_params={"user_input": "今天吃什么？"}))
        result = await app.run(start)
        self.assertEqual(result.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(result.output_data, "抱歉，我不明白你的意思。")

    async def test_tool_call_pattern_multi_round(self):
        """用例：多轮工具调用（先问天气，再问明天）"""
        def _response_generator(request):
            last_msg = request.messages[-1]
            if last_msg.role == aitoolman.MessageRole.user.value:
                return mock_llmclient.make_simple_response(request, [
                    aitoolman.ToolCall.new("get_weather", city="北京", date="今天"),
                ])
            round_num = sum(1 for msg in request.messages if msg.role == aitoolman.MessageRole.assistant.value)
            if round_num == 1:
                return mock_llmclient.make_simple_response(request, [
                    aitoolman.ToolCall.new("get_weather", city="北京", date="明天"),
                ])
            return mock_llmclient.make_tool_call_response(request)

        client = mock_llmclient.MockLLMClient(_response_generator)
        app_prompt = aitoolman.load_config_str("""
        [module.weather_module]
        model = "test_model"
        template.user = "{{user_input}}"
        post_processor = "builtin.parse_json"

        [module.weather_module.tools."get_weather"]
        type = "function"
        description = "获取指定城市指定日期的天气"
        param."city".type = "string"
        param."date".type = "string"
        param."city".required = true
        param."date".required = true
        """)
        app = aitoolman.LLMWorkflow(client, config_dict=app_prompt)

        class WeatherQueryTask(aitoolman.LLMTask):
            async def post_process(self, module_result):
                if module_result.status == aitoolman.FinishReason.tool_calls:
                    await self.run_tool_calls(get_weather=get_weather)

        start_task = WeatherQueryTask(aitoolman.LLMModuleRequest(
            module_name="weather_module",
            template_params={"user_input": "北京今天的天气怎么样？明天呢？"}
        ))

        result_task = await app.run(start_task)
        self.assertEqual(result_task.status, aitoolman.TaskStatus.COMPLETED)
        results = result_task.output_data
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        # 两轮结果
        for tool_call in results:
            self.assertEqual(tool_call['result'], get_weather(**tool_call['arguments']))

    async def test_tool_call_pattern_multiple_tools(self):
        """用例：一次调用多个工具（同轮多城市）"""
        def _response_generator(request):
            last_msg = request.messages[-1]
            if last_msg.role == aitoolman.MessageRole.user.value:
                return mock_llmclient.make_simple_response(request, [
                    aitoolman.ToolCall.new("get_weather", city="北京", date="今天"),
                    aitoolman.ToolCall.new("get_weather", city="上海", date="今天")
                ])
            else:
                return mock_llmclient.make_tool_call_response(request)

        client = mock_llmclient.MockLLMClient(_response_generator)
        app_prompt = aitoolman.load_config_str("""
        [module.multi_city_weather]
        model = "test_model"
        template.user = "{{user_input}}"
        post_processor = "builtin.parse_json"

        [module.multi_city_weather.tools."get_weather"]
        type = "function"
        description = "获取指定城市指定日期的天气"
        param."city".type = "string"
        param."date".type = "string"
        param."city".required = true
        param."date".required = true
        """)
        app = aitoolman.LLMWorkflow(client, config_dict=app_prompt)

        class MultiCityWeatherTask(aitoolman.LLMTask):
            async def post_process(self, module_result):
                if module_result.status == aitoolman.FinishReason.tool_calls:
                    await self.run_tool_calls(get_weather=get_weather)

        start_task = MultiCityWeatherTask(aitoolman.LLMModuleRequest(
            module_name="multi_city_weather",
            template_params={"user_input": "北京和上海今天的天气怎么样？"}
        ))

        result_task = await app.run(start_task)
        self.assertEqual(result_task.status, aitoolman.TaskStatus.COMPLETED)
        results = result_task.output_data
        self.assertEqual(len(results), 2)
        for tool_call in results:
            self.assertEqual(tool_call['result'], get_weather(**tool_call['arguments']))


# ---------------------------------------------------------------------------
# 复杂 LLM 流程测试（动作 XML 解析，非 Tool Call 方式）
# ---------------------------------------------------------------------------

class MockLLMForComplexFlow:
    """模拟复杂流程的 LLM 响应：按固定顺序返回预设的 XML 动作列表"""
    def __init__(self):
        self._responses = iter([
            # R1: 开始项目 -> A,B,C
            "<response><progress>0%</progress><actions>A,B,C</actions><memory></memory></response>",
            # R2: A结果 B结果 C结果 -> B,C,D
            "<response><progress>40%</progress><actions>B,C,D</actions><memory>前序结果良好</memory></response>",
            # R3: B结果 C结果 D结果 -> C,D,E
            "<response><progress>70%</progress><actions>C,D,E</actions><memory>需要子任务协助</memory></response>",
            # 子R1: 启动子任务E -> A1,B1,C1
            "<response><progress>子任务0%</progress><actions>A1,B1,C1</actions><memory></memory></response>",
            # 子R2: A1结果 B1结果 C1结果 -> Z
            "<response><progress>子任务100%</progress><actions>Z</actions><memory>子任务完成</memory></response>",
            # R4: C结果 D结果 E结果 -> Z
            "<response><progress>100%</progress><actions>Z</actions><memory>所有任务完成</memory></response>",
            # 额外兜底
            "<response><summary>任务完成</summary></response>",
        ])

    def __call__(self, request):
        try:
            resp_text = next(self._responses)
        except StopIteration:
            resp_text = "<response><summary>任务完成</summary></response>"
        return mock_llmclient.make_simple_response(request, resp_text)


class TestComplexLLMFlow(unittest.IsolatedAsyncioTestCase):
    """测试复杂的 LLM 工作流：动作解析、子任务循环、汇聚"""

    async def test_complex_workflow(self):
        # 全局执行记录
        executed_actions = []
        sub_executed_actions = []

        # ------------------------------
        # 通用动作定义
        # ------------------------------
        class ActionTask(aitoolman.Task):
            """通用普通动作任务"""
            async def run(self):
                action_name = self.input_data['action']
                await asyncio.sleep(0.01)
                executed_actions.append(action_name)
                return f"{action_name}执行结果"

        class SubActionTask(aitoolman.Task):
            """子任务内部动作任务"""
            async def run(self):
                action_name = self.input_data['action']
                await asyncio.sleep(0.01)
                sub_executed_actions.append(action_name)
                return f"子动作{action_name}执行结果"

        class SubWorkflowTask(aitoolman.Task):
            """E子任务：完全复用外层LLM核心逻辑，作为普通Task对外暴露"""
            async def run(self):
                # 子任务内部使用和外层完全相同的LLMCoreTask逻辑
                sub_start_task = LLMCoreTask(
                    aitoolman.LLMModuleRequest(
                        module_name="test_module_with_input",
                        template_params={"input": "启动子任务E"}
                    ),
                    action_map=SUB_ACTION_MAP
                )
                # 释放并发配额后执行子任务链，避免死锁
                async with self.workflow.release_worker():
                    await self.workflow.run(sub_start_task)
                executed_actions.append(self.input_data['action'])
                return "子任务E全部完成"

        # 动作映射表：统一处理所有动作，包括子任务入口E
        MAIN_ACTION_MAP = {
            "A": ActionTask,
            "B": ActionTask,
            "C": ActionTask,
            "D": ActionTask,
            "E": SubWorkflowTask,
            "Z": ActionTask
        }
        SUB_ACTION_MAP = {
            "A1": SubActionTask,
            "B1": SubActionTask,
            "C1": SubActionTask,
            "Z": SubActionTask
        }

        # ------------------------------
        # 通用LLM核心处理逻辑（外层和子任务复用）
        # ------------------------------
        class LLMCoreTask(aitoolman.LLMTask):
            """通用LLM核心处理任务，外层和子任务共用同一套代码"""
            def __init__(self, input_data=None, workflow=None, action_map=None):
                super().__init__(input_data=input_data, workflow=workflow)
                self.action_map = action_map or MAIN_ACTION_MAP

            async def post_process(self, module_result):
                text = module_result.response_text
                # 解析动作列表（统一XML解析逻辑）
                actions_match = re.search(r'<actions>(.*?)</actions>', text)
                if not actions_match:
                    # 无动作表示流程结束
                    return
                actions = [a.strip() for a in actions_match.group(1).split(',') if a.strip()]
                # 无动作结束
                if not actions:
                    return

                # 通用并行执行动作逻辑，所有动作（包括子任务E）统一处理
                parallel_tasks = []
                for act_name in actions:
                    task_cls = self.action_map.get(act_name)
                    if not task_cls:
                        raise ValueError(f"未知动作: {act_name}")
                    parallel_tasks.append(task_cls({"action": act_name}))

                # 释放并发配额后等待所有并行动作执行完成
                async with self.workflow.release_worker():
                    await self.workflow.wait_tasks(*parallel_tasks)

                # 结束动作Z不需要继续循环
                if "Z" in actions:
                    return

                # 通用生成下一轮LLM请求逻辑
                results_text = ' '.join([f"{t.input_data['action']}结果" for t in parallel_tasks])
                next_req = aitoolman.LLMModuleRequest(
                    module_name="test_module_with_input",
                    template_params={"input": results_text}
                )
                self.next_task = LLMCoreTask(next_req, action_map=self.action_map)

        # ------------------------------
        # 流程执行
        # ------------------------------
        client = mock_llmclient.MockLLMClient(MockLLMForComplexFlow())
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        # 启动外层主流程
        start_task = LLMCoreTask(aitoolman.LLMModuleRequest(
            module_name="test_module_with_input",
            template_params={"input": "开始项目"}
        ))
        last_task = await app.run(start_task)

        # ------------------------------
        # 结果验证
        # ------------------------------
        self.assertEqual(last_task.status, aitoolman.TaskStatus.COMPLETED)
        # 主流程执行动作验证
        self.assertIn("A", executed_actions)
        self.assertIn("B", executed_actions)
        self.assertIn("C", executed_actions)
        self.assertIn("D", executed_actions)
        self.assertIn("E", executed_actions)
        self.assertIn("Z", executed_actions)
        # 子任务执行动作验证
        self.assertIn("A1", sub_executed_actions)
        self.assertIn("B1", sub_executed_actions)
        self.assertIn("C1", sub_executed_actions)
        self.assertIn("Z", sub_executed_actions)

        # 资源清理验证
        await asyncio.sleep(0.05)
        self.assertEqual(len(app._active_tasks), 0)
        self.assertEqual(len(app._exec_tasks), 0)


if __name__ == '__main__':
    unittest.main()
