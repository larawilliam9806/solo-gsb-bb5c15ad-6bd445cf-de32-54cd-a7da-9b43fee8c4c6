"""子进程计算执行层测试：配置校验、并发名额、超时回收、异常与关闭清理。"""

import asyncio
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

import app.main as main_module
from app.align_executor import (
    ALIGN_MAX_PROCESSES,
    ALIGN_TIMEOUT_SECONDS,
    AlignBusyError,
    AlignExecutor,
    AlignTimeoutError,
    AlignWorkerError,
    positive_int_env,
)
from app.aligner import align
from app.main import app

ROOT = Path(__file__).resolve().parents[1]

client = TestClient(app)

# 占住名额但不做计算的假 worker；超时/取消/关闭时由执行器终止
SLEEP_WORKER = ("-c", "import time; time.sleep(60)")

# 计算耗时明显超过 1 秒、又不超过默认 10 秒时限的输入规模（300x300 约 3 秒）
SLOW_TEXT = "字" * 300


def test_default_config_values():
    assert ALIGN_MAX_PROCESSES == 2
    assert ALIGN_TIMEOUT_SECONDS == 10


def test_positive_int_env_default_and_valid(monkeypatch):
    monkeypatch.delenv("X_TEST_CFG", raising=False)
    assert positive_int_env("X_TEST_CFG", 2) == 2
    monkeypatch.setenv("X_TEST_CFG", "7")
    assert positive_int_env("X_TEST_CFG", 2) == 7


@pytest.mark.parametrize("bad", ["0", "-3", "abc", "", "1.5", "两"])
def test_positive_int_env_rejects_invalid(monkeypatch, bad):
    monkeypatch.setenv("X_TEST_CFG", bad)
    with pytest.raises(ValueError):
        positive_int_env("X_TEST_CFG", 2)


@pytest.mark.parametrize("name", ["ALIGN_MAX_PROCESSES", "ALIGN_TIMEOUT_SECONDS"])
@pytest.mark.parametrize("bad", ["0", "-1", "abc"])
def test_invalid_config_fails_startup(name, bad):
    env = {**os.environ, name: bad}
    proc = subprocess.run(
        [sys.executable, "-c", "import app.main"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert name in proc.stderr


def test_executor_run_matches_in_process_align():
    executor = AlignExecutor(max_processes=2, timeout_seconds=30)
    result = asyncio.run(executor.run("学而?习之", "时习之不亦"))
    assert result == align("学而?习之", "时习之不亦")
    assert executor.active == 0


def test_executor_concurrent_runs_all_succeed():
    executor = AlignExecutor(max_processes=2, timeout_seconds=30)

    async def scenario():
        r1, r2 = await asyncio.gather(
            executor.run("学而?习之", "时习之不亦"),
            executor.run("甲乙丙丁", "丙丁戊"),
        )
        assert r1["merged"] == "学而时习之不亦"
        assert r2["merged"] == "甲乙丙丁戊"
        assert executor.active == 0

    asyncio.run(scenario())


def test_executor_busy_rejected_immediately_without_queueing():
    executor = AlignExecutor(
        max_processes=1, timeout_seconds=30, worker_argv=SLEEP_WORKER
    )

    async def scenario():
        slow = asyncio.ensure_future(executor.run("甲", "甲"))
        await asyncio.sleep(0.5)  # 让第一个请求占住唯一名额
        t0 = time.perf_counter()
        with pytest.raises(AlignBusyError):
            await executor.run("甲", "甲")
        assert time.perf_counter() - t0 < 1.0  # 立即拒绝，不排队
        # 取消占用中的请求：子进程被终止回收后名额才释放
        slow.cancel()
        with pytest.raises(asyncio.CancelledError):
            await slow
        assert executor.active == 0
        assert not executor._children

    asyncio.run(scenario())


def test_executor_timeout_terminates_and_reclaims_child():
    executor = AlignExecutor(
        max_processes=1, timeout_seconds=1, worker_argv=SLEEP_WORKER
    )

    async def scenario():
        t0 = time.perf_counter()
        with pytest.raises(AlignTimeoutError):
            await executor.run("甲", "甲")
        # 1 秒超时 + 回收，远小于假 worker 的 60 秒睡眠
        assert time.perf_counter() - t0 < 10
        # 回收完成后名额才释放
        assert executor.active == 0
        assert not executor._children

    asyncio.run(scenario())


def test_executor_accepts_new_requests_after_timeout():
    executor = AlignExecutor(max_processes=1, timeout_seconds=1)

    async def scenario():
        with pytest.raises(AlignTimeoutError):
            await executor.run(SLOW_TEXT, SLOW_TEXT)
        # 超时回收后，后续请求可正常接纳
        result = await executor.run("学而?习之", "时习之不亦")
        assert result == align("学而?习之", "时习之不亦")
        assert executor.active == 0

    asyncio.run(scenario())


def test_executor_worker_error_then_recovers():
    failing = AlignExecutor(
        max_processes=1,
        timeout_seconds=10,
        worker_argv=("-c", "import sys; sys.exit(3)"),
    )

    async def scenario():
        with pytest.raises(AlignWorkerError):
            await failing.run("甲", "甲")
        assert failing.active == 0
        assert not failing._children

    asyncio.run(scenario())

    # 子进程异常后，后续请求可正常接纳
    executor = AlignExecutor(max_processes=1, timeout_seconds=10)
    result = asyncio.run(executor.run("学而?习之", "时习之不亦"))
    assert result["merged"] == "学而时习之不亦"


def test_executor_unparseable_output_is_worker_error():
    executor = AlignExecutor(
        max_processes=1,
        timeout_seconds=10,
        worker_argv=("-c", "print('not-json')"),
    )

    async def scenario():
        with pytest.raises(AlignWorkerError):
            await executor.run("甲", "甲")
        assert executor.active == 0

    asyncio.run(scenario())


def test_executor_close_terminates_running_children():
    executor = AlignExecutor(
        max_processes=2, timeout_seconds=60, worker_argv=SLEEP_WORKER
    )

    async def scenario():
        t1 = asyncio.ensure_future(executor.run("甲", "甲"))
        t2 = asyncio.ensure_future(executor.run("乙", "乙"))
        await asyncio.sleep(0.5)
        assert executor.active == 2
        await executor.aclose()
        for t in (t1, t2):
            with pytest.raises(AlignWorkerError):
                await t
        assert executor.active == 0
        assert not executor._children
        # 关闭后不再接纳新请求
        with pytest.raises(AlignBusyError):
            await executor.run("甲", "甲")

    asyncio.run(scenario())


def test_executor_close_waits_for_child_still_being_created():
    """关闭时请求已占用名额但仍在 create_subprocess_exec 返回前：aclose 不得
    提前结束，须等该子进程登记后终止回收、名额释放再返回。"""
    executor = AlignExecutor(
        max_processes=2, timeout_seconds=60, worker_argv=SLEEP_WORKER
    )
    spawn_started = asyncio.Event()
    allow_spawn = asyncio.Event()
    real_spawn = executor._spawn_child

    async def gated_spawn():
        spawn_started.set()
        await allow_spawn.wait()
        return await real_spawn()

    executor._spawn_child = gated_spawn

    async def scenario():
        task = asyncio.ensure_future(executor.run("甲", "甲"))
        await spawn_started.wait()
        # 名额已占用，但子进程尚未创建完成、未登记进 _children
        assert executor.active == 1
        assert not executor._children

        async def close_after_delay():
            await asyncio.sleep(0.2)
            await executor.aclose()

        close_task = asyncio.ensure_future(close_after_delay())
        await asyncio.sleep(0.3)
        # aclose 已开始（标志置位、拒绝新请求），但因创建途中尚未结束而等待
        assert executor._closed is True
        with pytest.raises(AlignBusyError):
            await executor.run("甲", "甲")
        assert not close_task.done()
        assert executor.active == 1

        allow_spawn.set()  # 放行创建：请求自行终止回收并释放名额
        await asyncio.wait_for(close_task, timeout=10)
        with pytest.raises(AlignWorkerError):
            await task
        assert executor.active == 0
        assert not executor._children

    asyncio.run(scenario())


def test_executor_close_waits_for_mixed_running_and_creating_requests():
    """同时存在运行中子进程与创建途中请求时，aclose 覆盖两者并等待名额归零。"""
    executor = AlignExecutor(
        max_processes=2, timeout_seconds=60, worker_argv=SLEEP_WORKER
    )
    spawn_started = asyncio.Event()
    allow_spawn = asyncio.Event()
    real_spawn = executor._spawn_child

    async def gated_spawn():
        spawn_started.set()
        await allow_spawn.wait()
        return await real_spawn()

    async def scenario():
        t_running = asyncio.ensure_future(executor.run("甲", "甲"))
        await asyncio.sleep(0.5)  # 第一个请求的子进程已登记运行
        assert executor.active == 1
        assert len(executor._children) == 1

        executor._spawn_child = gated_spawn
        t_creating = asyncio.ensure_future(executor.run("乙", "乙"))
        await spawn_started.wait()
        assert executor.active == 2
        assert len(executor._children) == 1

        async def close_after_delay():
            await asyncio.sleep(0.2)
            await executor.aclose()

        close_task = asyncio.ensure_future(close_after_delay())
        await asyncio.sleep(0.1)
        assert not close_task.done()
        allow_spawn.set()
        await asyncio.wait_for(close_task, timeout=10)
        for t in (t_running, t_creating):
            with pytest.raises(AlignWorkerError):
                await t
        assert executor.active == 0
        assert not executor._children

    asyncio.run(scenario())


def test_executor_close_idempotent_and_drains_repeatedly():
    """重复 aclose：均等待名额归零；正常请求收尾与关闭不重复释放名额。"""
    executor = AlignExecutor(
        max_processes=1, timeout_seconds=60, worker_argv=SLEEP_WORKER
    )

    async def scenario():
        task = asyncio.ensure_future(executor.run("甲", "甲"))
        await asyncio.sleep(0.5)
        assert executor.active == 1
        await asyncio.gather(executor.aclose(), executor.aclose())
        with pytest.raises(AlignWorkerError):
            await task
        assert executor.active == 0
        await executor.aclose()  # 已关闭且名额为零：立即返回
        assert executor.active == 0

    asyncio.run(scenario())


def test_api_align_success_response_unchanged():
    resp = client.post("/api/align", json={"left": "学而?习之", "right": "时习之不亦"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["score"] == 4
    assert data["merged"] == "学而时习之不亦"
    assert data["resolved_wildcards"] == [
        {"segment": "left", "pos": 2, "resolved_char": "时"}
    ]
    assert main_module.align_executor.active == 0


def test_api_align_validation_still_422_and_not_counted():
    assert client.post("/api/align", json={"left": "", "right": "abc"}).status_code == 422
    assert client.post("/api/align", json={"left": "abc"}).status_code == 422
    assert main_module.align_executor.active == 0


def test_api_align_maps_executor_errors(monkeypatch):
    class FakeExecutor:
        def __init__(self, exc):
            self._exc = exc

        async def run(self, left, right):
            raise self._exc

    for exc, status in (
        (AlignBusyError("名额已满"), 503),
        (AlignTimeoutError("超时"), 504),
        (AlignWorkerError("子进程异常"), 500),
    ):
        monkeypatch.setattr(main_module, "align_executor", FakeExecutor(exc))
        resp = client.post("/api/align", json={"left": "甲", "right": "甲"})
        assert resp.status_code == status
    monkeypatch.undo()


def test_api_align_full_slots_return_503_and_health_not_blocked(monkeypatch):
    executor = AlignExecutor(max_processes=1, timeout_seconds=30)
    monkeypatch.setattr(main_module, "align_executor", executor)
    holder = {}

    def call_slow():
        holder["resp"] = client.post(
            "/api/align", json={"left": SLOW_TEXT, "right": SLOW_TEXT}
        )

    t = threading.Thread(target=call_slow)
    t.start()
    try:
        deadline = time.time() + 10
        while executor.active < 1 and time.time() < deadline:
            time.sleep(0.05)
        assert executor.active == 1
        # 名额占满：新请求立即 503，不排队
        t0 = time.perf_counter()
        resp = client.post("/api/align", json={"left": "甲", "right": "甲"})
        assert resp.status_code == 503
        assert time.perf_counter() - t0 < 2
        # /health 不等待对齐计算
        t0 = time.perf_counter()
        health = client.get("/health")
        assert health.status_code == 200
        assert time.perf_counter() - t0 < 2
    finally:
        t.join(timeout=30)
    assert not t.is_alive()
    assert holder["resp"].status_code == 200
    assert executor.active == 0


def test_api_align_timeout_returns_504_and_recovers(monkeypatch):
    executor = AlignExecutor(max_processes=2, timeout_seconds=1)
    monkeypatch.setattr(main_module, "align_executor", executor)
    resp = client.post("/api/align", json={"left": SLOW_TEXT, "right": SLOW_TEXT})
    assert resp.status_code == 504
    assert executor.active == 0
    # 超时回收后，后续请求可正常接纳
    resp = client.post("/api/align", json={"left": "学而?习之", "right": "时习之不亦"})
    assert resp.status_code == 200
    assert resp.json()["merged"] == "学而时习之不亦"


def test_api_align_client_disconnect_terminates_child(monkeypatch):
    executor = AlignExecutor(max_processes=1, timeout_seconds=30)
    monkeypatch.setattr(main_module, "align_executor", executor)

    async def scenario():
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/align",
            "headers": [],
            "query_string": b"",
        }

        async def receive():
            return {"type": "http.disconnect"}

        request = Request(scope, receive)
        req = main_module.AlignRequest(left=SLOW_TEXT, right=SLOW_TEXT)
        with pytest.raises(HTTPException) as exc_info:
            await main_module.api_align(req, request)
        assert exc_info.value.status_code == 499
        # 断开后子进程被终止回收，名额释放
        assert executor.active == 0
        assert not executor._children

    asyncio.run(scenario())


def test_lifespan_shutdown_reclaims_children(monkeypatch):
    executor = AlignExecutor(
        max_processes=2, timeout_seconds=60, worker_argv=SLEEP_WORKER
    )
    monkeypatch.setattr(main_module, "align_executor", executor)
    holder = {}
    with TestClient(app) as c:
        t = threading.Thread(
            target=lambda: holder.setdefault(
                "resp", c.post("/api/align", json={"left": "甲", "right": "甲"})
            )
        )
        t.start()
        deadline = time.time() + 5
        while executor.active < 1 and time.time() < deadline:
            time.sleep(0.05)
        assert executor.active == 1
    # 退出 with 触发服务关闭：残留子进程被终止回收，在途请求以 500 收尾
    t.join(timeout=10)
    assert not t.is_alive()
    assert holder["resp"].status_code == 500
    assert executor.active == 0
    assert not executor._children


def test_lifespan_shutdown_waits_for_child_still_being_created(monkeypatch):
    """关闭发生在请求已占用名额、子进程仍在创建途中时：lifespan 的 aclose
    等待创建完成并终止回收，不遗留运行进程（关闭竞态回归测试）。"""
    executor = AlignExecutor(
        max_processes=2, timeout_seconds=60, worker_argv=SLEEP_WORKER
    )
    monkeypatch.setattr(main_module, "align_executor", executor)

    spawn_started = threading.Event()
    allow_spawn = threading.Event()
    real_spawn = executor._spawn_child

    async def gated_spawn():
        spawn_started.set()
        # 跨线程事件：在线程上阻塞等待，避免向跨事件循环的事件注册回调
        await asyncio.get_running_loop().run_in_executor(None, allow_spawn.wait)
        return await real_spawn()

    executor._spawn_child = gated_spawn

    holder = {}
    with TestClient(app) as c:
        t = threading.Thread(
            target=lambda: holder.setdefault(
                "resp", c.post("/api/align", json={"left": "甲", "right": "甲"})
            )
        )
        t.start()
        assert spawn_started.wait(timeout=5)
        # 名额已占但子进程尚未登记：此前的实现在此处关闭会漏掉该请求
        assert executor.active == 1
        assert not executor._children
        # 关闭开始后再放行子进程创建，复现“已接纳但创建中”的关闭竞态
        releaser = threading.Timer(0.5, allow_spawn.set)
        releaser.start()
    # lifespan 关闭必须等创建途中的子进程终止回收、名额释放后才结束
    releaser.join(timeout=5)
    t.join(timeout=10)
    assert not t.is_alive()
    assert holder["resp"].status_code == 500
    assert executor.active == 0
    assert not executor._children
