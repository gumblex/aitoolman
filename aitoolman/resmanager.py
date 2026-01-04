import time
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Optional, Set, List
from dataclasses import dataclass
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class ResourceStatus(Enum):
    PENDING = "pending"
    ACQUIRED = "acquired"
    CANCELLED = "cancelled"


@dataclass
class ResourceRequest:
    future: asyncio.Future
    status: ResourceStatus = ResourceStatus.PENDING
    task_name: Optional[str] = None
    enqueue_time: float = 0.0

    def __eq__(self, other):
        return self.future == other.future

    def __hash__(self):
        return hash(self.future)


class ResourceManager:
    def __init__(self, capacities: Dict[str, int] = None):
        """
        初始化资源管理器

        Args:
            capacities: 资源容量配置，如 {'a': 3, 'b': 10}，可选
        """
        self.capacities: Dict[str, int] = capacities or {}
        self._active_tasks: Dict[str, Set[ResourceRequest]] = {}
        self._waiting_queues: Dict[str, deque] = {}  # 使用deque代替Queue，更轻量
        self._pending_requests: Dict[str, Dict[str, ResourceRequest]] = {}
        self._available_slots: Dict[str, int] = {}  # 可用槽位计数
        self._lock = asyncio.Lock()  # 保护资源操作的锁

        # 初始化现有资源
        for key, capacity in self.capacities.items():
            self._initialize_resource(key, capacity)

    def _initialize_resource(self, key: str, capacity: int):
        """初始化单个资源"""
        self._waiting_queues[key] = deque()
        self._active_tasks[key] = set()
        self._pending_requests[key] = {}
        self._available_slots[key] = capacity

    async def add_resource(self, key: str, capacity: int):
        """
        动态添加资源
        Args:
            key: 资源key
            capacity: 资源容量
        """
        async with self._lock:
            if key in self.capacities:
                raise ValueError(f"Resource '{key}' already exists")

            self.capacities[key] = capacity
            self._initialize_resource(key, capacity)

    async def remove_resource(self, key: str, force: bool = False):
        """
        动态移除资源
        Args:
            key: 资源key
            force: 是否强制移除（即使有活跃任务）
        Returns:
            bool: 是否成功移除
        """
        async with self._lock:
            if key not in self.capacities:
                return False

            active_count = len(self._active_tasks[key])
            if active_count > 0 and not force:
                raise RuntimeError(f"Cannot remove resource '{key}' with {active_count} active tasks")

            # 取消所有等待中的请求
            pending_count = await self._cancel_all_pending(key)

            if force and active_count > 0:
                logger.warning("[%s] Force remove, with %s tasks cancelled.", key, active_count)

            # 清理资源
            del self.capacities[key]
            del self._waiting_queues[key]
            del self._active_tasks[key]
            del self._pending_requests[key]
            del self._available_slots[key]

            logger.info("[%s] Removed. Cancelled %s pending tasks.", key, pending_count)
            return True

    @asynccontextmanager
    async def acquire(self, key: str, task_name: Optional[str] = None):
        """
        异步获取资源，按照排队顺序公平分配

        Args:
            key: 资源key
            task_name: 任务名称（用于调试和取消）

        Yields:
            资源上下文管理器

        Raises:
            KeyError: 资源key不存在
            asyncio.CancelledError: 任务被取消
        """
        if key not in self.capacities:
            raise KeyError(f"Resource '{key}' not found")

        if not task_name:
            task_name = f"task_{id(asyncio.current_task())}"

        # 创建请求对象
        request = ResourceRequest(
            future=asyncio.Future(),
            task_name=task_name,
            enqueue_time=time.monotonic()
        )

        async with self._lock:
            # 记录待处理请求
            self._pending_requests[key][task_name] = request

            # 如果有可用槽位，直接获取
            if self._available_slots[key] > 0:
                self._available_slots[key] -= 1
                can_acquire_immediately = True
            else:
                can_acquire_immediately = False
                # 加入等待队列
                self._waiting_queues[key].append(request.future)

        if can_acquire_immediately:
            # 立即获取资源
            await self._complete_acquisition(key, request, task_name)
        else:
            # 等待资源可用
            queue_position = len(self._waiting_queues[key])
            logger.debug("[%s] Task '%s' waiting at %s", key, task_name, queue_position)

            try:
                await request.future
                await self._complete_acquisition(key, request, task_name)
            except asyncio.CancelledError:
                await self._handle_cancellation(key, request, task_name)
                raise

        try:
            yield request
        finally:
            await self._release_resource(key, request, task_name)

    async def _complete_acquisition(self, key: str, request: ResourceRequest, task_name: str):
        """完成资源获取过程"""
        request.status = ResourceStatus.ACQUIRED
        self._active_tasks[key].add(request)

        if task_name in self._pending_requests[key]:
            del self._pending_requests[key][task_name]

        if not request.future.done():
            request.future.set_result(True)

        wait_time = time.monotonic() - request.enqueue_time
        logger.debug("[%s] Task '%s' acquired resource, waited for %.2f", task_name, key, wait_time)

    async def _handle_cancellation(self, key: str, request: ResourceRequest, task_name: str):
        """处理取消操作"""
        async with self._lock:
            request.status = ResourceStatus.CANCELLED
            if task_name in self._pending_requests[key]:
                del self._pending_requests[key][task_name]

            # 从等待队列中移除
            if request.future in self._waiting_queues[key]:
                self._waiting_queues[key].remove(request.future)

            if not request.future.done():
                request.future.set_exception(asyncio.CancelledError())

    async def _release_resource(self, key: str, request: ResourceRequest, task_name: str):
        """释放资源"""
        async with self._lock:
            if request in self._active_tasks[key]:
                self._active_tasks[key].remove(request)

                # 如果有等待的任务，唤醒下一个
                if self._waiting_queues[key]:
                    next_future = self._waiting_queues[key].popleft()
                    if not next_future.done():
                        next_future.set_result(True)
                else:
                    self._available_slots[key] += 1

    async def cancel_request(self, key: str, task_name: str) -> bool:
        """
        取消指定任务的资源请求

        Args:
            key: 资源key
            task_name: 任务名称

        Returns:
            bool: 是否成功取消
        """
        async with self._lock:
            if key not in self._pending_requests:
                return False

            if task_name in self._pending_requests[key]:
                request = self._pending_requests[key][task_name]
                if request.status == ResourceStatus.PENDING:
                    request.status = ResourceStatus.CANCELLED
                    del self._pending_requests[key][task_name]

                    # 从等待队列中移除
                    if request.future in self._waiting_queues[key]:
                        self._waiting_queues[key].remove(request.future)

                    # 取消future
                    if not request.future.done():
                        request.future.cancel()

                    logging.debug("[%s] Cancelled for task '%s'", key, task_name)
                    return True
        return False

    async def _cancel_all_pending(self, key: str) -> int:
        """
        取消指定资源的所有等待请求，在 self._lock 内部

        Args:
            key: 资源key

        Returns:
            int: 取消的请求数量
        """
        if key not in self._pending_requests:
            return 0

        count = 0
        for task_name, request in list(self._pending_requests[key].items()):
            if request.status == ResourceStatus.PENDING:
                request.status = ResourceStatus.CANCELLED
                del self._pending_requests[key][task_name]

                # 从等待队列中移除
                if request.future in self._waiting_queues[key]:
                    self._waiting_queues[key].remove(request.future)

                if not request.future.done():
                    request.future.cancel()
                count += 1

        logger.debug("[%s] Cancelled %s pending requests", key, count)
        return count

    def get_stats(self, key: str) -> Dict:
        """
        获取资源统计信息

        Args:
            key: 资源key

        Returns:
            Dict: 统计信息
        """
        if key not in self.capacities:
            return {}

        capacity = self.capacities[key]
        active = len(self._active_tasks[key])
        pending = len(self._pending_requests[key])
        queue_size = len(self._waiting_queues[key])

        return {
            'capacity': capacity,
            'active': active,
            'pending': pending,
            'queue_size': queue_size,
            'available': self._available_slots[key]
        }

    def get_queue_length(self, key: str) -> int:
        """
        获取指定资源的当前队列长度
        Args:
            key: 资源key
        Returns:
            int: 队列长度（等待的任务数量）
        Raises:
            KeyError: 资源key不存在
        """
        if key not in self.capacities:
            raise KeyError(f"Resource '{key}' not found")
        return len(self._waiting_queues[key])

    def get_all_stats(self) -> Dict[str, Dict]:
        """
        获取所有资源的统计信息

        Returns:
            Dict: 所有资源的统计信息
        """
        return {key: self.get_stats(key) for key in self.capacities}

    def get_queue_position(self, key: str, task_name: str) -> Optional[int]:
        """
        获取任务在队列中的位置（1表示下一个）

        Args:
            key: 资源key
            task_name: 任务名称

        Returns:
            Optional[int]: 位置（从1开始），如果不在队列中返回None
        """
        if key not in self._pending_requests or task_name not in self._pending_requests[key]:
            return None

        # 查找任务在等待队列中的位置
        request = self._pending_requests[key][task_name]
        try:
            position = list(self._waiting_queues[key]).index(request.future) + 1
            return position
        except ValueError:
            return None

    def list_resources(self) -> List[str]:
        """获取所有资源key列表"""
        return list(self.capacities.keys())

    def get_capacity(self, key: str):
        return self.capacities[key]
