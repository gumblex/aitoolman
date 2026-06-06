import enum
import asyncio
import inspect
import logging
import contextvars
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Callable, Union

from .app import LLMApplication
from . import util
from . import model as _model


logger = logging.getLogger(__name__)

# 用于标记当前协程是否持有 worker semaphore 的上下文变量
_semaphore_held: contextvars.ContextVar[int] = contextvars.ContextVar('semaphore_held', default=0)


class TaskStatus(enum.Enum):
    """任务状态枚举"""
    INIT = 0    # 初始化
    WAITING = 1    # 待执行
    RUNNING = 2    # 执行中
    COMPLETED = 3  # 已完成
    FAILED = 4     # 已失败


class LLMWorkflowError(_model.LLMApplicationError):
    """工作流执行错误"""
    def __init__(self, msg: str, *tasks: 'Task'):
        super().__init__(f'{msg} Task: {tasks}')
        self.msg = msg
        self.tasks = tasks


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
        self.status = TaskStatus.RUNNING
        self.status_event.clear()
        try:
            self.output_data = await self.run()
            self.status = TaskStatus.COMPLETED
        except Exception as ex:
            logger.exception("Task failed: %r", self)
            self.error = ex
            self.status = TaskStatus.FAILED
        finally:
            self.status_event.set()

    async def run(self):
        """
        用户函数：输入 self.input_data，返回值为 self.output_data
        """
        if self._func is not None:
            output_data = self._func(**self.input_data)
            if inspect.isawaitable(output_data):
                return await output_data
            else:
                return output_data
        else:
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

    def following_tasks(self) -> List['Task']:
        """收集当前任务后续的所有链式任务，返回任务列表"""
        tasks = []
        current = self.next_task
        while current:
            tasks.append(current)
            current = current.next_task
        return tasks


class LLMTask(Task):
    """LLM任务类，专注于LLM调用和工具调用处理"""
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

    async def run(self):
        if isinstance(self.input_data, (_model.LLMModuleRequest, _model.LLMDirectRequest)):
            req = self.input_data
        else:
            raise ValueError("input_data is not LLMModuleRequest/LLMDirectRequest")
        self.module_result = await self.workflow.call(req)
        self.module_result.raise_for_status()
        if self.module_result:
            output_data = self.module_result.data
            await self.post_process(self.module_result)
            return output_data
        return None

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
        if req is None:
            return
        next_task = self.clone()
        next_task.input_data = req
        self.next_task = next_task


class LLMWorkflow(LLMApplication):
    """
    任务调度器，支持动态任务链和并行子任务。

    核心概念：
    - 使用 Semaphore 限制并行执行的任务数，避免资源耗尽。
    - 通过 submit(task) 将任务提交执行，任务完成自动提交 next_task 实现链式执行。
    - 使用 wait_tasks(*tasks) 提交一组任务并等待它们全部完成。
    - 使用 run(start_task) 执行一条完整的任务链（依次跟随 next_task）。
    - 任务内部可以通过 workflow.wait_tasks/submit 启动支线（子任务链）并等待。
    - 为避免在等待子任务时阻塞并发配额，可在任务内部使用
      async with workflow.release_worker(): 临时释放配额，允许其他任务执行。
    """

    def __init__(
        self,
        client: Any,
        config_dict: Optional[Dict[str, Any]] = None,
        processors: Optional[Dict[str, Callable[[str], Any]]] = None,
        context_id: Optional[str] = None,
        max_parallel_tasks: int = 5  # 默认最大并行任务数
    ):
        # 调用父类初始化
        super().__init__(
            client=client,
            config_dict=config_dict,
            processors=processors,
            context_id=context_id
        )
        self.max_parallel_tasks = max_parallel_tasks
        self._semaphore = asyncio.Semaphore(max_parallel_tasks)
        self._active_tasks: Dict[str, Task] = {}  # 跟踪已提交且未完成的任务
        self._exec_tasks: Dict[str, asyncio.Task] = {}  # 正在执行的任务协程
        self._stopped = False

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    async def submit(self, task: Task):
        """
        提交任务。如果任务已经提交过（task_id 在活动任务中），则忽略。
        任务完成后，如果 task.next_task 不为 None，会自动提交下一个任务。
        """
        if self._stopped:
            return
        if task.task_id in self._active_tasks:
            return
        task.workflow = self
        self._active_tasks[task.task_id] = task
        # 直接创建协程执行任务，通过 Semaphore 限制并发
        exec_coro = self._execute_task(task)
        exec_task = asyncio.create_task(exec_coro, name=f'LLMWorkflow-{task.task_id}')
        self._exec_tasks[task.task_id] = exec_task
        exec_task.add_done_callback(lambda t, tid=task.task_id: self._exec_tasks.pop(tid, None))

    async def wait_tasks(self, *tasks: Task, timeout: Optional[float] = None):
        """
        提交多个任务（如果尚未提交），并发等待所有任务完成。
        """
        # 提交所有尚未提交的任务
        for task in tasks:
            if task.status == TaskStatus.INIT:
                await self.submit(task)

        # 等待所有任务完成
        wait_coros = [task.status_event.wait() for task in tasks]
        if timeout:
            await asyncio.wait_for(asyncio.gather(*wait_coros), timeout=timeout)
        else:
            await asyncio.gather(*wait_coros)

        # 检查失败的任务
        failed = [t for t in tasks if t.status == TaskStatus.FAILED]
        if failed:
            raise LLMWorkflowError(f"{len(failed)} tasks failed.", *failed)

    async def run(self, start_task: Task) -> Task:
        """
        执行一条任务链，从 start_task 开始，依次跟随 next_task 直到没有下一步。
        返回链上最后一个完成的任务。
        如果在链中任何任务失败，抛出异常并停止执行后续任务。
        """
        current = start_task
        last_task = start_task
        while current:
            await self.submit(current)
            await current.status_event.wait()
            if current.status == TaskStatus.FAILED:
                raise LLMWorkflowError("Task failed.", current) from current.error
            last_task = current
            current = current.next_task
        return last_task

    async def stop(self):
        """停止所有正在执行的任务，清理资源。"""
        if self._stopped:
            return
        self._stopped = True
        exec_tasks = list(self._exec_tasks.values())
        for t in exec_tasks:
            t.cancel()
        if exec_tasks:
            await asyncio.gather(*exec_tasks, return_exceptions=True)
        self._active_tasks.clear()
        self._exec_tasks.clear()

    @asynccontextmanager
    async def release_worker(self):
        """
        异步上下文管理器，用于临时释放当前任务的并发配额，允许其他任务执行。

        必须在任务执行期间调用（即 Task.run() 或 LLMTask.post_process() 内）。
        典型用法：
            async with self.workflow.release_worker():
                await self.workflow.wait_tasks(subtask1, subtask2)
        """
        held = _semaphore_held.get()
        if held <= 0:
            raise RuntimeError("release_worker can only be called from a task that holds the worker semaphore")
        self._semaphore.release()
        _semaphore_held.set(0)
        try:
            yield
        finally:
            await self._semaphore.acquire()
            _semaphore_held.set(1)

    # 支持 async with
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    # ------------------------------------------------------------------
    # 内部调度
    # ------------------------------------------------------------------

    async def _execute_task(self, task: Task):
        """执行单个任务，管理 Semaphore 和后续任务提交。"""
        acquired = False
        try:
            await self._semaphore.acquire()
            acquired = True
            _semaphore_held.set(1)

            # 防止重复执行
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                return

            await task.start()

            # 任务成功完成后自动提交 next_task
            if task.status == TaskStatus.COMPLETED and task.next_task is not None:
                await self.submit(task.next_task)

        except asyncio.CancelledError as ex:
            if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                task.status = TaskStatus.FAILED
                task.error = ex
                task.status_event.set()
            raise
        except Exception as ex:
            logger.exception("Unhandled error in execute_task for task: %r", task)
            if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                task.status = TaskStatus.FAILED
                task.error = ex
                task.status_event.set()
        finally:
            if acquired and _semaphore_held.get() > 0:
                self._semaphore.release()
                _semaphore_held.set(0)
            self._active_tasks.pop(task.task_id, None)
