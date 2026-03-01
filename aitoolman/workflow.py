import abc
import enum
import asyncio
import inspect
import logging
import functools
import contextlib
import collections
import dataclasses
from typing import Any, Dict, List, Optional, Set, Callable, Union, ClassVar

from .app import LLMApplication
from . import util
from . import model as _model


logger = logging.getLogger(__name__)

class TaskStatus(enum.Enum):
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


class TaskDependencyError(LLMWorkflowError):
    """依赖的任务执行错误，包含出错的所有任务"""

    def __init__(self, error_tasks: List['Task'], task_chain: List[List['Task']], context_id=None):
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


class Task:
    """
    任务基类，执行自定义函数
    重写 run() 或用 Task.set_func 指定具体函数
    """
    input_data: Dict[str, Any]
    workflow: Optional['LLMWorkflow']
    task_id: str
    task_name: str
    description: str
    status: TaskStatus
    status_event: asyncio.Event
    output_data: Any
    next_task: Optional['Task']
    error: Optional[Exception]
    _func: Optional[Callable]

    def __init__(self,
        input_data: Optional[Dict[str, Any]] = None,
        workflow: Optional['LLMWorkflow'] = None
    ):
        self.input_data = input_data or {}
        self.workflow = workflow
        self.task_id = util.get_id()
        self.task_name = self.__class__.__name__
        self.description = ''
        self.status = TaskStatus.INIT
        self.status_event = asyncio.Event()
        self.output_data = None
        self.next_task = None
        self.error = None
        self._func: Optional[Callable] = None

    def __repr__(self):
        return '%s(input_data=%r, task_id=%r, task_name=%r, status=%s)' % (
            self.__class__.__name__,
            self.input_data, self.task_id, self.task_name, self.status
        )

    def set_func(self, fn: Callable):
        self._func = fn

    async def start(self):
        """
        执行该任务，并设置 output_data
        用于 LLMWorkflow 内部
        """
        if self._func is not None:
            output_data = self._func(**self.input_data)
            if inspect.isawaitable(output_data):
                self.output_data = await output_data
            else:
                self.output_data = output_data
        else:
            self.output_data = await self.run(**self.input_data)

    async def run(self, **input_data):
        """
        用户函数：输入 self.input_data，返回值为 self.output_data
        """
        raise NotImplementedError

    def clone(self):
        new_task = self.__class__()
        new_task.input_data = self.input_data
        new_task.workflow = self.workflow
        new_task.task_id = self.task_id
        new_task.task_name = self.task_name
        new_task.description = self.description
        new_task._func = self._func
        return new_task


class LLMTaskCompleted(Exception):
    """
    提前结束LLMTask，用于 LLMTask.post_process
    """
    pass


class LLMTask(Task):
    """LLM任务类"""
    input_data: Union[_model.LLMModuleRequest, _model.LLMDirectRequest] = None
    module_result: Optional[_model.LLMModuleResult] = None

    def __init__(
            self,
            input_data: Union[_model.LLMModuleRequest, _model.LLMDirectRequest, None] = None,
            workflow: Optional['LLMWorkflow'] = None
    ):
        super().__init__()
        self.input_data = input_data
        self.workflow = workflow

    async def start(self):
        if isinstance(self.input_data, (_model.LLMModuleRequest, _model.LLMDirectRequest)):
            req = self.input_data
        else:
            raise ValueError("input_data is not LLMModuleRequest/LLMDirectRequest")
        self.module_result = await self.workflow.call(req)
        self.module_result.raise_for_status()
        if self.module_result:
            self.output_data = self.module_result.data
            try:
                await self.post_process(self.module_result)
            except LLMTaskCompleted:
                pass

    async def post_process(self, module_result: _model.LLMModuleResult):
        """
        后处理钩子：在LLM模块返回结果后执行
        """
        pass

    def on_tool_call_goto(self, **kwargs: Callable[[], 'Task']):
        """
        用于 post_process，将工具调用转为下一步的 Task
        * 非工具调用，直接返回
        * 对第一个调用，设置 next_task 为相应 Task，结束当前任务
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
        next_task.input_data = tool_call.arguments or {}
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
        req = await self.module_result.run_tool_calls(kwargs)
        next_task = self.clone()
        next_task.input_data = req
        self.next_task = next_task
        raise LLMTaskCompleted()


class LLMWorkflow(LLMApplication):
    """
    任务调度器，支持DAG（有向无环图）任务执行

    两种运行模式（可结合）：
    1. 通过 run，在 Task 中设置 next_task，串行执行工作流
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
        self._new_tasks: Dict[str, Task] = {}
        self._queued_tasks: Dict[str, Task] = {}
        # task -> 被依赖项 (dependents)
        self._graph: Dict[str, Set[str]] = collections.defaultdict(set)
        # task -> 依赖项 (dependencies)
        self._reverse_graph: Dict[str, Set[str]] = collections.defaultdict(set)
        self._pending_queue: asyncio.Queue[str] = asyncio.Queue()

        self._next_consumer_id: int = 0
        self._consumers: Dict[int, asyncio.Task] = {}
        self.running: bool = False

    def add_task(self, task: Task, next_task: Optional[Task] = None):
        """
        添加后台任务 task，不立即执行
        task 为 next_task 之前要运行的任务

        Args:
            task: 要添加的任务
            next_task: 要添加的任务之后要执行的任务，或为 None
        """
        self._new_tasks[task.task_id] = task
        if next_task:
            self._graph[task.task_id].add(next_task.task_id)
            self._reverse_graph[next_task.task_id].add(task.task_id)

    async def run_task(self, task: Task):
        task.status = TaskStatus.RUNNING
        task.status_event.clear()

        try:
            await task.start()
            if task.next_task:
                self.add_task(task.next_task, None)
            task.status = TaskStatus.COMPLETED
            task.status_event.set()

            # 检查并调度依赖的任务
            await self._queue_dependents(task)
        except TaskDependencyError as ex:
            task.error = ex
            task.status = TaskStatus.DEPENDENCY_FAILED
            task.status_event.set()
            # 传播错误到依赖的任务
            self._fail_dependents(task, ex)
        except Exception as ex:
            logger.exception("Task failed: %r", task)
            task.error = ex
            task.status = TaskStatus.FAILED
            task.status_event.set()
            # 传播错误到依赖的任务
            self._fail_dependents(task, ex)

    async def _queue_dependents(self, task: Task):
        """检查并调度依赖于此任务的任务"""
        for dependent_id in self._graph.get(task.task_id, []):
            if dependent_id not in self._queued_tasks:
                continue
            dependent = self._queued_tasks[dependent_id]
            # 检查是否所有依赖都已完成
            if all(
                    dep_id not in self._queued_tasks or
                    self._queued_tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in self._reverse_graph.get(dependent_id, [])
            ):
                if dependent.status == TaskStatus.INIT:
                    await self._pending_queue.put(dependent_id)

    def _fail_dependents(self, task: Task, error: Exception):
        """将错误传播到所有依赖的任务"""
        for dependent_id in self._graph.get(task.task_id, []):
            if dependent_id not in self._queued_tasks:
                continue
            dependent = self._queued_tasks[dependent_id]
            if dependent.status in (TaskStatus.INIT, TaskStatus.WAITING):
                dependent.status = TaskStatus.DEPENDENCY_FAILED
                dependent.error = error
                dependent.status_event.set()
                # 递归传播
                self._fail_dependents(dependent, error)

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

                if task.status not in (TaskStatus.INIT, TaskStatus.WAITING):
                    continue

                await self.run_task(task)
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

    async def _raise_dependency_error(self, failed_tasks: List[Task]):
        """Raise TaskDependencyError with proper task chain"""
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

        raise TaskDependencyError(
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
                if task.status == TaskStatus.COMPLETED:
                    # Check if it has any dependents that are not completed
                    has_active_dependents = False
                    for dependent_id in self._graph.get(task_id, []):
                        dependent = self._queued_tasks.get(dependent_id)
                        if dependent and dependent.status != TaskStatus.COMPLETED:
                            has_active_dependents = True
                            break

                    if not has_active_dependents:
                        tasks_to_remove.append(task_id)

            # Remove tasks
            for task_id in tasks_to_remove:
                self._clear_task_by_id(task_id)

    async def wait_tasks(self, *tasks: Task, timeout: Optional[float] = None):
        """
        等待指定的任务完成，如果任务不在已有任务列表内则添加

        Args:
            tasks: 要等待的任务
            timeout: 超时时间（秒）
        """
        if not tasks:
            return

        # Register tasks if not already registered
        for task in tasks:
            if task.task_id not in self._new_tasks and task.task_id not in self._queued_tasks:
                self.add_task(task, None)

        # Check if all tasks are already completed
        all_completed = all(
            task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.DEPENDENCY_FAILED)
            for task in tasks
        )

        if all_completed:
            failed_tasks = [t for t in tasks if t.status in (TaskStatus.FAILED, TaskStatus.DEPENDENCY_FAILED)]
            if failed_tasks:
                await self._raise_dependency_error(failed_tasks)
            return

        all_task_ids = set()

        def collect_deps(task_id: str):
            if task_id in all_task_ids:
                return
            all_task_ids.add(task_id)
            # 递归收集所有上游依赖
            for dep_id in self._reverse_graph.get(task_id, set()):
                collect_deps(dep_id)

        for task in tasks:
            collect_deps(task.task_id)

        # Build dependency graph and find ready tasks
        ready_task_ids = []
        async with self._task_lock:
            # 将所有收集到的任务从_new_tasks移到_queued_tasks
            for task_id in all_task_ids:
                if task_id in self._new_tasks:
                    task = self._new_tasks.pop(task_id)
                    self._queued_tasks[task_id] = task

            # Find tasks with no unmet dependencies
            for task_id, task in self._queued_tasks.items():
                if task.status == TaskStatus.INIT and not self._reverse_graph.get(task_id):
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
        failed_tasks = [t for t in tasks if t.status in (TaskStatus.FAILED, TaskStatus.DEPENDENCY_FAILED)]
        if failed_tasks:
            await self._raise_dependency_error(failed_tasks)

        # Cleanup completed tasks
        await self._cleanup_completed_tasks()

    async def run(self, start_task: Task) -> Task:
        """
        运行工作流，从起始任务开始

        Args:
            start_task: 起始任务

        Raises:
            TaskDependencyError: 任务失败
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

