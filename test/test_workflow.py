import json
import logging
import asyncio
import unittest
import functools
import re

import aitoolman

import mock_llmclient

logging.basicConfig(level="DEBUG", format="%(asctime)s - %(levelname)s - %(message)s")

TEST_CONFIG = aitoolman.load_config_str("""
[module.test_module_with_input]
model = "test_model"
template.user = "{{input}}"
""")


class TestTaskBasic(unittest.IsolatedAsyncioTestCase):
    """测试Task的基础功能"""

    async def test_task_success(self):
        """测试任务成功执行"""
        class SimpleTask(aitoolman.Task):
            async def run(self, x, y):
                return x + y

        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        task = SimpleTask({"x": 1, "y": 2}, workflow=app)
        await app.wait_tasks(task)

        self.assertEqual(task.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(task.output_data, 3)

    async def test_task_failure(self):
        """测试任务执行失败"""
        def failing_func():
            raise ValueError("Function failed")

        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        task = aitoolman.Task({}, workflow=app)
        task.set_func(failing_func)

        with self.assertRaises(aitoolman.TaskDependencyError):
            await app.wait_tasks(task)

        self.assertEqual(task.status, aitoolman.TaskStatus.FAILED)
        self.assertIn("Function failed", str(task.error))

    async def test_inherit_task_override_run(self):
        """测试继承Task并重写run方法"""

        class CustomTask(aitoolman.Task):
            def __init__(self, value, workflow):
                super().__init__()
                self.input_data = {"value": value}
                self.workflow = workflow

            async def run(self, value):
                return value * 2

        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        task = CustomTask(5, workflow=app)
        await app.wait_tasks(task)

        self.assertEqual(task.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(task.output_data, 10)


class FailingTask(aitoolman.Task):
    async def run(self):
        raise ValueError("Dependency failed")


class SuccessTask(aitoolman.Task):
    async def run(self, x):
        return x * 2


class SlowTask(aitoolman.Task):
    async def run(self, sec):
        await asyncio.sleep(sec)
        return sec


class TestTaskDependency(unittest.IsolatedAsyncioTestCase):
    """测试任务依赖功能"""

    async def test_dependency_failure_wait_final(self):
        """测试依赖项失败导致后续任务失败"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        task_a = FailingTask({}, workflow=app)
        task_b = SuccessTask({"x": 5}, workflow=app)

        app.add_task(task_a, task_b)

        with self.assertRaises(aitoolman.TaskDependencyError):
            await app.wait_tasks(task_b)

        self.assertEqual(task_a.status, aitoolman.TaskStatus.FAILED)
        self.assertEqual(task_b.status, aitoolman.TaskStatus.DEPENDENCY_FAILED)

    async def test_dependency_failure_wait_all(self):
        """测试依赖项失败导致后续任务失败"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        task_a = FailingTask({}, workflow=app)
        task_b = SuccessTask({"x": 5}, workflow=app)

        app.add_task(task_a, task_b)

        with self.assertRaises(aitoolman.TaskDependencyError):
            await app.wait_tasks(task_a, task_b)

        self.assertEqual(task_a.status, aitoolman.TaskStatus.FAILED)
        self.assertEqual(task_b.status, aitoolman.TaskStatus.DEPENDENCY_FAILED)

    async def test_max_parallel_consumers(self):
        """测试并行consumer个数不超过指定值"""
        max_parallel = 2
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG, max_parallel_tasks=max_parallel)

        # 记录consumer数量变化
        consumer_counts = []
        original_start = app._start_consumers

        def track_consumers(task_num):
            original_start(task_num)
            consumer_counts.append(len(app._consumers))

        app._start_consumers = track_consumers

        tasks = [
            SlowTask({"sec": 0.2}, workflow=app)
            for i in range(5)
        ]

        await app.wait_tasks(*tasks)

        # 检查consumer数量从未超过最大值
        self.assertTrue(all(count <= max_parallel for count in consumer_counts))
        self.assertTrue(all(t.status == aitoolman.TaskStatus.COMPLETED for t in tasks))

    async def test_no_residual_tasks_after_completion(self):
        """测试任务完成后无残留的task和consumer"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        tasks = [
            SuccessTask({"x": i}, workflow=app)
            for i in range(3)
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

    async def test_wait_tasks_dependency_graph(self):
        """测试wait_tasks处理显式依赖图：C依赖A和B，A/B完成后C才执行"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        class RecorderTask(aitoolman.Task):
            execution_order = []

            async def run(self):
                self.execution_order.append(self.task_id)
                return self.task_id

        task_a = RecorderTask()
        task_a.task_id = "A"
        task_b = RecorderTask()
        task_b.task_id = "B"
        task_c = RecorderTask()
        task_c.task_id = "C"
        task_d = RecorderTask()
        task_d.task_id = "D"

        app.add_task(task_a, task_b)
        app.add_task(task_b, task_d)
        app.add_task(task_c, task_d)

        # 等待所有任务完成
        await app.wait_tasks(task_d)

        # 检查状态
        self.assertEqual(task_a.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(task_b.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(task_c.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(task_d.status, aitoolman.TaskStatus.COMPLETED)

        # 检查执行顺序: A和B应该在C之前完成
        idx = {k: i for i, k in enumerate(RecorderTask.execution_order)}
        self.assertLess(idx['A'], idx['B'])
        self.assertLess(idx['B'], idx['D'])
        self.assertLess(idx['C'], idx['D'])

    async def test_wait_tasks_timeout(self):
        """测试wait_tasks超时机制"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        task = SlowTask({"sec": 1.0})

        with self.assertRaises(asyncio.TimeoutError):
            await app.wait_tasks(task, timeout=0.1)

    async def test_task_double_execution_protection(self):
        """测试任务不会被执行两次"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        class CounterTask(aitoolman.Task):
            def __init__(self):
                super().__init__()
                self.counter = 0

            async def run(self):
                self.counter += 1
                return self.counter

        task = CounterTask()

        await app.wait_tasks(task)
        await app.wait_tasks(task)

        # 应该只执行一次
        self.assertEqual(task.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(task.counter, 1)

    async def test_wait_tasks_empty_list(self):
        """测试wait_tasks传入空列表"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        # 不应抛出异常
        await app.wait_tasks()
        self.assertEqual(len(client.requests), 0)

    async def test_complex_dependency_chain_failure(self):
        """测试复杂依赖链中的失败传播：A->B->C，A失败导致B和C都失败"""
        client = mock_llmclient.MockLLMClient()
        app = aitoolman.LLMWorkflow(client, config_dict=TEST_CONFIG)

        task_a = FailingTask({}, workflow=app)
        task_b = SuccessTask({"x": 5}, workflow=app)
        task_c = SuccessTask({"x": 10}, workflow=app)

        app.add_task(task_a, task_b)  # B depends on A
        app.add_task(task_b, task_c)  # C depends on B

        with self.assertRaises(aitoolman.TaskDependencyError):
            await app.wait_tasks(task_a, task_b, task_c)

        self.assertEqual(task_a.status, aitoolman.TaskStatus.FAILED)
        self.assertEqual(task_b.status, aitoolman.TaskStatus.DEPENDENCY_FAILED)
        self.assertEqual(task_c.status, aitoolman.TaskStatus.DEPENDENCY_FAILED)


class TestLLMTaskToolSupport(unittest.IsolatedAsyncioTestCase):
    """测试LLMTask的工具调用处理，和使用工具调用的常见用例"""

    async def test_on_tool_call_goto(self):
        """测试on_tool_call_goto跳转到下一个Task"""
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
        self.assertIsInstance(result_task, aitoolman.Task)
        self.assertEqual(result_task.output_data, 84)

    async def test_run_tool_calls_single(self):
        """测试run_tool_calls单轮工具调用"""
        def _response_generator(request: aitoolman.LLMProviderRequest) -> aitoolman.LLMProviderResponse:
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
            aitoolman.LLMModuleRequest(module_name="tool_call_module", template_params={}),
            app
        )

        result_task = await app.run(start_task)
        self.assertEqual(result_task.status, aitoolman.TaskStatus.COMPLETED)
        self.assertIsInstance(result_task, aitoolman.LLMTask)
        self.assertEqual(result_task.output_data[0]['result'], "84")

    async def test_run_tool_calls_multiple(self):
        """测试run_tool_calls一次调用多个工具"""
        def _response_generator(request: aitoolman.LLMProviderRequest) -> aitoolman.LLMProviderResponse:
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
            aitoolman.LLMModuleRequest(module_name="multi_tool_module", template_params={}),
            app
        )

        result_task = await app.run(start_task)
        self.assertEqual(result_task.status, aitoolman.TaskStatus.COMPLETED)
        self.assertIsInstance(result_task, aitoolman.LLMTask)

        # 验证工具调用结果
        results = result_task.output_data
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['arguments'], {'x': 2, 'y': 3})
        self.assertEqual(results[0]['result'], "6")
        self.assertEqual(results[1]['arguments'], {'a': 5, 'b': 7})
        self.assertEqual(results[1]['result'], "12")


# 模拟天气工具的实现
def get_weather(city, date):
    if city == "北京":
        return f"{date}{city}的天气是：晴，20-28度"
    elif city == "上海":
        return f"{date}{city}的天气是：多云，22-30度"
    return "未知天气"


class TestLLMWorkflowToolUseCase(unittest.IsolatedAsyncioTestCase):
    """测试LLMWorkflow+LLMTask使用工具调用的常见用例"""

    async def test_tool_call_as_intent_recognition(self):
        """用例：工具调用作为意图识别，跳转到下一个Task"""
        def _response_generator(request: aitoolman.LLMProviderRequest) -> aitoolman.LLMProviderResponse:
            last_msg = request.messages[-1]
            if last_msg.role == aitoolman.MessageRole.user.value:
                user_input = last_msg.content
                if "你好" in user_input:
                    return mock_llmclient.make_simple_response(request, [aitoolman.ToolCall.new("greet", name="用户")])
                elif "计算" in user_input:
                    match = re.search(r'计算(\d+)\+(\d+)', user_input)
                    if match:
                        return mock_llmclient.make_simple_response(request, [
                            aitoolman.ToolCall.new(
                                "calculate", a=int(match.group(1)), b=int(match.group(2)))
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

        # 定义不同意图的Task
        class GreetTask(aitoolman.Task):
            async def run(self, name):
                return f"你好，{name}！"

        class CalculateTask(aitoolman.Task):
            async def run(self, a, b):
                return f"结果是：{a+b}"

        class UnknownTask(aitoolman.Task):
            async def run(self):
                return "抱歉，我不明白你的意思。"

        # 起始Task：意图识别
        class IntentTask(aitoolman.LLMTask):
            async def post_process(self, module_result):
                self.on_tool_call_goto(
                    greet=GreetTask,
                    calculate=CalculateTask,
                    unknown=UnknownTask
                )

        # 测试问候意图
        start_task = IntentTask(
            aitoolman.LLMModuleRequest(module_name="intent_recognition", template_params={"user_input": "你好"}),
            app
        )
        result_task = await app.run(start_task)
        self.assertEqual(result_task.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(result_task.output_data, "你好，用户！")

        # 测试计算意图
        start_task = IntentTask(
            aitoolman.LLMModuleRequest(module_name="intent_recognition", template_params={"user_input": "计算1+2"}),
            app
        )
        result_task = await app.run(start_task)
        self.assertEqual(result_task.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(result_task.output_data, "结果是：3")

        # 测试未知意图
        start_task = IntentTask(
            aitoolman.LLMModuleRequest(module_name="intent_recognition", template_params={"user_input": "今天吃什么？"}),
            app
        )
        result_task = await app.run(start_task)
        self.assertEqual(result_task.status, aitoolman.TaskStatus.COMPLETED)
        self.assertEqual(result_task.output_data, "抱歉，我不明白你的意思。")

    async def test_tool_call_pattern_multi_round(self):
        """
        用例：经典工具调用模式，多轮调用

        user: 提问->
        assistant: 调用工具->
        tool: 工具回复->
        assistant: 调用工具->
        tool: 工具回复->
        assistant: 最终回复
        """
        def _response_generator(request: aitoolman.LLMProviderRequest) -> aitoolman.LLMProviderResponse:
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

        # 定义天气查询Task
        class WeatherQueryTask(aitoolman.LLMTask):
            async def post_process(self, module_result):
                # 如果有工具调用，执行工具调用并继续对话
                if module_result.status == aitoolman.FinishReason.tool_calls:
                    await self.run_tool_calls(get_weather=get_weather)

        # 起始Task：用户提问
        start_task = WeatherQueryTask(aitoolman.LLMModuleRequest(
            module_name="weather_module",
            template_params={"user_input": "北京今天的天气怎么样？明天呢？"}
        ))

        # 运行串行工作流
        result_task = await app.run(start_task)
        self.assertEqual(result_task.status, aitoolman.TaskStatus.COMPLETED)
        self.assertIsInstance(result_task.output_data, list)
        self.assertEqual(len(result_task.output_data), 2)
        for i, tool_call in enumerate(result_task.output_data):
            if i == 2:
                self.assertEqual(tool_call['arguments']['date'], '明天')
            self.assertEqual(tool_call['result'], get_weather(**tool_call['arguments']))

    async def test_tool_call_pattern_multiple_tools(self):
        """
        用例：经典工具调用模式，一次调用多个工具

        user: 提问->
        assistant: 调用2个工具->
        tool: 2个工具回复->
        assistant: 最终回复
        """
        def _response_generator(request: aitoolman.LLMProviderRequest) -> aitoolman.LLMProviderResponse:
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

        # 定义多城市天气查询Task
        class MultiCityWeatherTask(aitoolman.LLMTask):
            async def post_process(self, module_result):
                if module_result.status == aitoolman.FinishReason.tool_calls:
                    await self.run_tool_calls(get_weather=get_weather)

        # 起始Task：用户提问
        start_task = MultiCityWeatherTask(aitoolman.LLMModuleRequest(
            module_name="multi_city_weather",
            template_params={"user_input": "北京和上海今天的天气怎么样？"}
        ))

        # 运行串行工作流
        result_task = await app.run(start_task)
        self.assertEqual(result_task.status, aitoolman.TaskStatus.COMPLETED)
        self.assertIsInstance(result_task.output_data, list)
        self.assertEqual(len(result_task.output_data), 2)
        for tool_call in result_task.output_data:
            self.assertEqual(tool_call['result'], get_weather(**tool_call['arguments']))


if __name__ == '__main__':
    unittest.main()
