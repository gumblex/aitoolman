import enum
import asyncio
import logging
import contextlib
import collections
import dataclasses
from typing import Any, Dict, List, Optional, Set, Callable, Union, ClassVar

from .app import LLMApplication
from . import util
from . import model as _model


logger = logging.getLogger(__name__)


class LLMTaskStatus(enum.Enum):
    """任务状态枚举"""
    INIT = 0    # 初始化
    WAITING = 1    # 待执行
    RUNNING = 2    # 执行中
    COMPLETED = 3  # 已完成
    FAILED = 4     # 已失败
    DEPENDENCY_FAILED = 5  # 依赖失败


class LLMWorkflowError(_model.LLMApplicationError):
    """工作流执行错误基类"""
    pass


class LLMTaskDependencyError(LLMWorkflowError):
    """依赖的任务执行错误，包含出错的所有任务"""

    def __init__(self, error_tasks: List['LLMTask'], task_chain: List[List['LLMTask']], context_id=None):
        self.error_tasks = error_tasks
        self.task_chain = task_chain
        self.first_error = error_tasks[0].error
        self.context_id = context_id
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        line = ""
        if self.context_id:
            line = f"[{self.context_id}] "
        line += f"Workflow execution failed: {len(self.error_tasks)} task(s) failed"
        lines = [line]

        # Show the first error
        if self.first_error:
            lines.append(f"\nFirst error: {type(self.first_error).__name__}: {self.first_error}")

        # Show dependency chain if available
        if self.task_chain:
            lines.append("\nDependency chain (root -> leaves):")
            for i, level_tasks in enumerate(self.task_chain):
                task_descs = [f"{t.task_name}[{t.task_id}]" for t in level_tasks]
                lines.append(f"  Level {i}: {', '.join(task_descs)}")

        # Show all failed tasks
        if len(self.error_tasks) > 1:
            lines.append(f"\nAll failed tasks:")
            for task in self.error_tasks:
                lines.append(f"  - {task.task_name}[{task.task_id}]: {task.error}")

        return "\n".join(lines)


class LLMTaskCompleted(Exception):
    """
    提前结束LLMTask，用于 LLMTask.post_process
    """
    pass


@dataclasses.dataclass
class LLMTask:
    """
    LLM任务基类，支持自定义前处理和后处理逻辑

    用户可以通过继承此类并重写 pre_process 和 post_process 方法
    来实现复杂的任务逻辑，如动态生成下一个任务、处理工具调用等
    """
    module_name: ClassVar[str] = ''

    # 默认行为：
    # Dict[str, Any] -> 调用 module_name 作为模板输入
    # LLMDirectRequest -> 直接输入 LLM
    # 其他 -> 需要自定义 pre_process
    input_data: Union[Dict[str, Any], _model.LLMDirectRequest, Any] = None

    task_id: str = dataclasses.field(default_factory=util.get_id)
    _task_name: str = ''
    description: str = ''

    workflow: Optional['LLMWorkflow'] = None
    module_result: Optional[_model.LLMModuleResult] = None
    output_data: Any = None
    next_task: Optional['LLMTask'] = None
    status: LLMTaskStatus = LLMTaskStatus.INIT
    status_event: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
    error: Optional[Exception] = None

    @property
    def task_name(self):
        return self._task_name or self.__class__.__name__

    async def pre_process(self) -> Union[_model.LLMModuleRequest, _model.LLMDirectRequest, None]:
        """
        前处理钩子：在调用LLM模块之前执行

        默认实现：
        - input_data 为 LLMModuleRequest/LLMDirectRequest：直接调用
        - input_data 为 dict：作为模板参数
        - 其他：报错

        用户可以重写此方法以实现：
        - 动态修改输入数据
        - 添加上下文消息
        - 添加多媒体内容
        """
        if isinstance(self.input_data, (_model.LLMModuleRequest, _model.LLMDirectRequest)):
            return self.input_data
        if not isinstance(self.input_data, dict):
            raise ValueError("input_data is not dict")
        return _model.LLMModuleRequest(
            module_name=self.module_name,
            template_params=self.input_data,
            context_messages=[],
            media_content=None
        )

    async def post_process(self):
        """
        后处理钩子：在LLM模块返回结果后执行

        默认实现：将 module_result.data 赋值给 output_data
        用户可以重写此方法以实现：
        - 解析和验证输出
        - 根据结果动态生成下一个任务
        - 处理工具调用
        - 实现分支逻辑
        """
        if self.module_result:
            self.output_data = self.module_result.data

    def on_tool_call_goto(self, **kwargs: Callable[[], 'LLMTask']):
        """
        用于 post_process，将工具调用转为下一步的 LLMTask
        * 非工具调用，直接返回
        * 对第一个调用，设置 next_task 为相应 LLMTask，结束当前任务
        * 无匹配的调用，报错
        """
        if not self.module_result:
            return
        if self.module_result.status != _model.FinishReason.tool_calls:
            return
        if not self.module_result.tool_calls:
            raise _model.LLMResponseFormatError("tool call list is empty")
        tool_call = self.module_result.tool_calls[0]
        fn = kwargs.get(tool_call.name)
        if not fn:
            raise _model.LLMResponseFormatError("tool call not supported: %s" % tool_call)
        next_task = fn()
        next_task.workflow = self.workflow
        next_task.input_data = next_task.input_data or tool_call.arguments or {}
        next_task.task_id = tool_call.id
        next_task._task_name = tool_call.name
        self.next_task = next_task
        raise LLMTaskCompleted()

    async def run_tool_calls(self, **kwargs: Callable):
        """
        用于 post_process，工具调用作为函数调用，生成下一步的 LLMTask，结束当前任务
        """
        if not self.module_result:
            return
        if self.module_result.status != _model.FinishReason.tool_calls:
            return
        if not self.module_result.tool_calls:
            raise _model.LLMResponseFormatError("tool call list is empty")
        context = await self.module_result.run_tool_calls(kwargs)
        original_req = self.module_result.request
        req = _model.LLMDirectRequest(
            model_name=original_req.model_name,
            messages=context,
            tools=original_req.tools,
            options=original_req.options,
            stream=original_req.stream,
            output_channel=original_req.output_channel,
            reasoning_channel=original_req.reasoning_channel
        )
        next_task = self.__class__()
        next_task.workflow = self.workflow
        next_task._task_name = self.task_name
        next_task.input_data = req
        next_task.description = self.description
        self.next_task = next_task
        raise LLMTaskCompleted()


class LLMWorkflow(LLMApplication):
    """
    任务调度器，支持DAG（有向无环图）任务执行

    两种运行模式（可结合）：
    1. 通过 run，在 LLMTask 中设置 next_task，串行执行工作流
    2. 用 add_task 生成嵌套任务，用 wait_tasks 等待任务完成
    """

    def __init__(
        self,
        client: Any,
        config_dict: Optional[Dict[str, Any]] = None,
        processors: Optional[Dict[str, Callable[[str], Any]]] = None,
        channels: Optional[Dict[str, Any]] = None,
        context_id: Optional[str] = None,
        max_parallel_tasks: int = 5  # 默认最大并行任务数
    ):
        # 调用父类初始化
        super().__init__(
            client=client,
            config_dict=config_dict,
            processors=processors,
            channels=channels,
            context_id=context_id
        )
        self.max_parallel_tasks = max_parallel_tasks
        # 用于防止下列状态同时修改
        self._task_lock = asyncio.Lock()
        self._consumer_lock = asyncio.Lock()

        # 内部调度状态
        # 跟踪待运行、运行时的任务，运行完成之后清除
        self._new_tasks: Dict[str, LLMTask] = {}
        self._queued_tasks: Dict[str, LLMTask] = {}
        # task -> 被依赖项 (dependents)
        self._graph: Dict[str, Set[str]] = collections.defaultdict(set)
        # task -> 依赖项 (dependencies)
        self._reverse_graph: Dict[str, Set[str]] = collections.defaultdict(set)
        self._pending_queue: asyncio.Queue[str] = asyncio.Queue()

        self._next_consumer_id: int = 0
        self._consumers: Dict[int, asyncio.Task] = {}
        self.running: bool = False

    def add_task(self, current_task: Optional[LLMTask], dependent_task: LLMTask):
        """
        添加后台任务，不立即执行
        dependent_task 为 current_task 之前要运行的任务
        current_task 为 None 时，单独添加 dependent_task

        这个方法设计为在 LLMTask.pre_process 或 post_process 中调用
        类似 asyncio.create_task，会管理任务依赖和并行度

        Args:
            current_task: 当前正在执行的任务，或为 None
            dependent_task: 要添加的依赖任务
        """
        self._new_tasks[dependent_task.task_id] = dependent_task
        if current_task:
            self._graph[dependent_task.task_id].add(current_task.task_id)
            self._reverse_graph[current_task.task_id].add(dependent_task.task_id)

    async def run_llm_task(self, task: LLMTask):
        task.status = LLMTaskStatus.RUNNING
        task.status_event.clear()

        try:
            req = await task.pre_process()
            if req is None:
                task.module_result = None
            else:
                task.module_result = await self.call(req)
                task.module_result.raise_for_status()
            try:
                await task.post_process()
            except LLMTaskCompleted:
                pass
            if task.next_task:
                self.add_task(None, task.next_task)
            task.status = LLMTaskStatus.COMPLETED
            task.status_event.set()

            # 检查并调度依赖的任务
            await self._queue_dependents(task)
        except LLMTaskDependencyError as ex:
            task.error = ex
            task.status = LLMTaskStatus.DEPENDENCY_FAILED
            task.status_event.set()
            # 传播错误到依赖的任务
            await self._fail_dependents(task, ex)
        except Exception as ex:
            logger.exception("Task %s(%s) failed.",
                             task.task_name, task.task_id)
            task.error = ex
            task.status = LLMTaskStatus.FAILED
            task.status_event.set()
            # 传播错误到依赖的任务
            await self._fail_dependents(task, ex)

    async def _queue_dependents(self, task: LLMTask):
        """检查并调度依赖于此任务的任务"""
        for dependent_id in self._graph.get(task.task_id, []):
            if dependent_id not in self._queued_tasks:
                continue
            dependent = self._queued_tasks[dependent_id]
            # 检查是否所有依赖都已完成
            if all(
                dep_id not in self._queued_tasks or
                self._queued_tasks[dep_id].status == LLMTaskStatus.COMPLETED
                for dep_id in self._reverse_graph.get(dependent_id, [])
            ):
                if dependent.status == LLMTaskStatus.INIT:
                    await self._pending_queue.put(dependent_id)

    async def _fail_dependents(self, task: LLMTask, error: Exception):
        """将错误传播到所有依赖的任务"""
        for dependent_id in self._graph.get(task.task_id, []):
            if dependent_id not in self._queued_tasks:
                continue
            dependent = self._queued_tasks[dependent_id]
            if dependent.status in (LLMTaskStatus.INIT, LLMTaskStatus.WAITING):
                dependent.status = LLMTaskStatus.DEPENDENCY_FAILED
                dependent.error = error
                dependent.status_event.set()
                # 递归传播
                await self._fail_dependents(dependent, error)

    def _clear_task_by_id(self, task_id: str):
        """清除任务的跟踪状态"""
        with contextlib.suppress(KeyError):
            del self._graph[task_id]
        with contextlib.suppress(KeyError):
            del self._reverse_graph[task_id]
        with contextlib.suppress(KeyError):
            del self._queued_tasks[task_id]

    async def _consumer(self, consumer_id: int):
        """后台消费者协程，从队列取任务执行"""
        try:
            while True:
                task_id = None
                try:
                    # 从队列获取任务，设置超时以便定期检查停止信号
                    task_id = await asyncio.wait_for(
                        self._pending_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    # Check if we should exit (no more tasks)
                    if not self._queued_tasks and self._pending_queue.empty():
                        break
                    continue

                task = self._queued_tasks.get(task_id)
                if task is None:
                    logger.error("[Consumer %s] Task %s not in _queued_tasks, skipped",
                                 consumer_id, task_id)
                    continue

                if task.status not in (LLMTaskStatus.INIT, LLMTaskStatus.WAITING):
                    continue

                await self.run_llm_task(task)
        finally:
            with contextlib.suppress(KeyError):
                del self._consumers[consumer_id]

    def _start_consumers(self, task_num: int):
        # Should use with self._consumer_lock
        start_num = min(self.max_parallel_tasks,
                        task_num - len(self._consumers))
        if start_num <= 0:
            return
        for i in range(start_num):
            consumer_id = self._next_consumer_id
            self._next_consumer_id += 1
            self._consumers[consumer_id] = asyncio.create_task(
                self._consumer(consumer_id),
                name=('LLMWorkflow.consumer[%s]' % consumer_id)
            )

    async def _stop_consumers(self):
        """Stop all consumer tasks"""
        # Should use with self._consumer_lock
        for consumer_id, consumer_task in list(self._consumers.items()):
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                pass
            with contextlib.suppress(KeyError):
                del self._consumers[consumer_id]

    async def _reset_state(self):
        """Reset workflow state for a new run"""
        async with self._task_lock:
            self._new_tasks.clear()
            self._queued_tasks.clear()
            self._graph.clear()
            self._reverse_graph.clear()
            # Clear queue
            while not self._pending_queue.empty():
                try:
                    self._pending_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        async with self._consumer_lock:
            await self._stop_consumers()

    def _clear_all_tasks(self):
        """Clear all task tracking data"""
        self._new_tasks.clear()
        self._queued_tasks.clear()
        self._graph.clear()
        self._reverse_graph.clear()

    async def _raise_dependency_error(self, failed_tasks: List[LLMTask]):
        """Raise LLMTaskDependencyError with proper task chain"""
        # Build task chain from dependencies
        task_chain = []
        visited = set()

        def build_chain(level_tasks, level=0):
            if level >= len(task_chain):
                task_chain.append([])

            next_level_tasks = []
            for task in level_tasks:
                if task.task_id not in visited:
                    visited.add(task.task_id)
                    task_chain[level].append(task)
                    # Add dependents for next level
                    for dependent_id in self._graph.get(task.task_id, []):
                        if dependent_id in self._queued_tasks:
                            next_level_tasks.append(self._queued_tasks[dependent_id])

            if next_level_tasks:
                build_chain(next_level_tasks, level + 1)

        build_chain(failed_tasks)

        raise LLMTaskDependencyError(
            error_tasks=failed_tasks,
            task_chain=task_chain,
            context_id=self.context_id
        )

    async def _cleanup_completed_tasks(self):
        """Clean up tasks that are completed and have no active dependents"""
        async with self._task_lock:
            # Collect all tasks that can be removed
            tasks_to_remove = []

            for task_id, task in list(self._queued_tasks.items()):
                if task.status == LLMTaskStatus.COMPLETED:
                    # Check if it has any dependents that are not completed
                    has_active_dependents = False
                    for dependent_id in self._graph.get(task_id, []):
                        dependent = self._queued_tasks.get(dependent_id)
                        if dependent and dependent.status != LLMTaskStatus.COMPLETED:
                            has_active_dependents = True
                            break

                    if not has_active_dependents:
                        tasks_to_remove.append(task_id)

            # Remove tasks
            for task_id in tasks_to_remove:
                self._clear_task_by_id(task_id)

    async def wait_tasks(self, *tasks: LLMTask, timeout: Optional[float] = None):
        """
        等待指定的任务完成

        这个方法设计为在 LLMTask.pre_process 或 post_process 中调用
        如果任务不在已有任务列表内则添加

        Args:
            tasks: 要等待的任务
            timeout: 超时时间（秒）
        """
        if not tasks:
            return

        # Register tasks if not already registered
        for task in tasks:
            if task.task_id not in self._new_tasks and task.task_id not in self._queued_tasks:
                self.add_task(None, task)

        # Check if all tasks are already completed
        all_completed = all(
            task.status in (LLMTaskStatus.COMPLETED, LLMTaskStatus.FAILED, LLMTaskStatus.DEPENDENCY_FAILED)
            for task in tasks
        )

        if all_completed:
            failed_tasks = [t for t in tasks if t.status in (LLMTaskStatus.FAILED, LLMTaskStatus.DEPENDENCY_FAILED)]
            if failed_tasks:
                await self._raise_dependency_error(failed_tasks)
            return

        # Build dependency graph and find ready tasks
        ready_task_ids = []
        async with self._task_lock:
            # Move tasks from _new_tasks to _queued_tasks
            for task in tasks:
                if task.task_id in self._new_tasks:
                    del self._new_tasks[task.task_id]
                    self._queued_tasks[task.task_id] = task

            # Find tasks with no unmet dependencies
            for task_id, task in self._queued_tasks.items():
                if task.status == LLMTaskStatus.INIT and not self._reverse_graph.get(task_id):
                    ready_task_ids.append(task_id)

        # Queue ready tasks
        for task_id in ready_task_ids:
            await self._pending_queue.put(task_id)

        # Start consumers
        async with self._consumer_lock:
            self._start_consumers(len(self._queued_tasks))

        # Wait for completion with timeout support
        wait_coros = [task.status_event.wait() for task in tasks]
        if timeout:
            await asyncio.wait_for(asyncio.gather(*wait_coros), timeout=timeout)
        else:
            await asyncio.gather(*wait_coros)

        # Check for errors
        failed_tasks = [t for t in tasks if t.status in (LLMTaskStatus.FAILED, LLMTaskStatus.DEPENDENCY_FAILED)]
        if failed_tasks:
            await self._raise_dependency_error(failed_tasks)

        # Cleanup completed tasks
        await self._cleanup_completed_tasks()

    async def run(self, start_task: LLMTask) -> LLMTask:
        """
        运行工作流，从起始任务开始

        Args:
            start_task: 起始任务

        Raises:
            LLMTaskDependencyError: 任务失败
        """
        # Ensure single instance
        if self.running:
            raise RuntimeError("parallel run is not supported")
        self.running = True

        # Reset state
        await self._reset_state()

        last_task = start_task
        try:
            current_task = start_task
            while current_task:
                current_task.workflow = self
                logger.debug("Run step: %s", current_task)
                await self.wait_tasks(current_task)
                last_task = current_task
                current_task = current_task.next_task
        finally:
            async with self._consumer_lock:
                await self._stop_consumers()
            self._clear_all_tasks()
            self.running = False
        return last_task

