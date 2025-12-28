#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import logging
from aitoolman.resmanager import ResourceManager

logging.basicConfig(level="DEBUG", format="%(asctime)s - %(levelname)s - %(message)s")


# 使用示例和测试代码
async def worker(manager: ResourceManager, resource_key: str, worker_id: int, work_time: float):
    """
    模拟工作任务的协程
    """
    task_name = f"worker_{worker_id}"

    try:
        logging.info(f"[{task_name}] 尝试获取资源 {resource_key}...")

        async with manager.acquire(resource_key, task_name) as request:
            logging.info(f"[{task_name}] 成功获取资源 {resource_key}, 开始工作 {work_time}秒")
            # logging.info(f"[{task_name}] {request}")
            stats = manager.get_stats(resource_key)

            logging.info(f"[{task_name}] 资源状态: {stats}")

            # 模拟工作
            await asyncio.sleep(work_time)
            logging.info(f"[{task_name}] 工作完成，释放资源 {resource_key}")
            logging.info(f"[{task_name}] 资源状态: {stats}")

    except asyncio.CancelledError:
        logging.info(f"[{task_name}] 任务被取消")


async def main():
    """
    演示资源管理器的使用 - 测试公平性
    """
    logging.info("=" * 60)
    logging.info("测试资源管理的公平性（FIFO）")
    logging.info("=" * 60)

    # 创建资源管理器：资源a容量2
    manager = ResourceManager({'a': 2})

    # 创建多个任务竞争资源，测试公平性
    tasks = []

    # 创建5个任务，每个任务工作2秒
    for i in range(5):
        task = asyncio.create_task(worker(manager, 'a', i, 2.0))
        tasks.append(task)
        await asyncio.sleep(0.1)  # 稍微错开启动时间

    # 等待所有任务完成
    await asyncio.gather(*tasks, return_exceptions=True)

    logging.info("")
    logging.info("最终资源状态:")
    for key, stats in manager.get_all_stats().items():
        logging.info(f"资源 {key}: {stats}")


async def demo_cancellation():
    """
    演示取消功能
    """
    logging.info("=" * 60)
    logging.info("演示取消功能")
    logging.info("=" * 60)

    manager = ResourceManager({'test': 1})

    # 创建多个任务，第一个占用资源，其他的等待
    tasks = []
    for i in range(4):
        task = asyncio.create_task(worker(manager, 'test', i, 3.0))
        tasks.append(task)
        await asyncio.sleep(0.1)

    # 等待一会儿让任务进入队列
    await asyncio.sleep(0.5)

    # 取消一个等待中的任务
    logging.info("取消 worker_2 的任务...")
    success = await manager.cancel_request('test', 'worker_2')
    logging.info(f"取消结果: {'成功' if success else '失败'}")

    # 查看当前状态
    logging.info("取消后的资源状态:")
    for key, stats in manager.get_all_stats().items():
        logging.info(f"资源 {key}: {stats}")

    # 等待剩余任务完成
    await asyncio.gather(*tasks, return_exceptions=True)


async def demo_multiple_resources():
    """
    演示多个资源的管理
    """
    logging.info("" + "=" * 60)
    logging.info("演示多个资源的管理")
    logging.info("=" * 60)

    # 创建资源管理器：资源a容量2，资源b容量3
    manager = ResourceManager({'a': 2, 'b': 3})

    # 创建混合任务
    tasks = []

    # 任务使用资源a
    for i in range(4):
        task = asyncio.create_task(worker(manager, 'a', i, 1.0))
        tasks.append(task)

    # 任务使用资源b
    for i in range(4, 8):
        task = asyncio.create_task(worker(manager, 'b', i, 1.5))
        tasks.append(task)

    # 等待所有任务完成
    await asyncio.gather(*tasks, return_exceptions=True)

    logging.info("最终资源状态:")
    for key, stats in manager.get_all_stats().items():
        logging.info(f"资源 {key}: {stats}")


async def demo_dynamic_resources():
    """
    演示动态资源管理
    """
    logging.info("=" * 60)
    logging.info("演示动态资源管理")
    logging.info("=" * 60)

    # 创建空资源管理器
    manager = ResourceManager()

    # 动态添加资源
    await manager.add_resource('dynamic_a', 2)
    await manager.add_resource('dynamic_b', 3)

    logging.info(f"当前资源: {manager.list_resources()}")

    # 使用动态添加的资源
    tasks = []
    for i in range(5):
        task = asyncio.create_task(worker(manager, 'dynamic_a', i, 1.0))
        tasks.append(task)
        await asyncio.sleep(0.1)

    logging.info("asyncio.gather:")

    await asyncio.gather(*tasks, return_exceptions=True)

    logging.info("asyncio.gather complete")

    # 动态移除资源（无活跃任务时）
    await manager.remove_resource('dynamic_b')

    logging.info("manager.remove_resource")

    logging.info(f"最终资源列表: {manager.list_resources()}")
    logging.info("最终资源状态:")
    for key, stats in manager.get_all_stats().items():
        logging.info(f"资源 {key}: {stats}")


if __name__ == "__main__":
    # 运行演示
    # asyncio.run(main())
    # asyncio.run(demo_cancellation())
    # asyncio.run(demo_multiple_resources())
    asyncio.run(demo_dynamic_resources())
