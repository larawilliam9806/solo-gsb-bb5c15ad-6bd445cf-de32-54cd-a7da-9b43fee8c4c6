"""``/api/align`` 的子进程计算执行层。

每个已接纳的对齐请求都在**独立子进程**中运行 ``app.aligner.align``（以
``python -m app.align_worker`` 拉起），主进程用 asyncio 异步等待结果，
事件循环不被计算阻塞，``/health`` 等接口无须等待对齐计算。

并发与超时由两个环境变量控制（均须为**正整数**，非法值在导入本模块时
抛出 ``ValueError``，使服务启动失败）：

- ``ALIGN_MAX_PROCESSES``：同时处于**运行中或清理中**的对齐子进程数上限，
  默认 2；无空位的请求立即失败（接口映射为 503），不排队等待空位；
- ``ALIGN_TIMEOUT_SECONDS``：单次计算从**接纳**起的最长秒数，默认 10；
  超时即终止并回收子进程（接口映射为 504）。

名额自接纳时占用，到该请求的子进程**终止回收完成**后才释放；正常结束、
超时、客户端断开（任务取消）、子进程异常退出与服务关闭都汇入同一条
终止-回收路径，不泄漏进程、不重复释放名额。服务关闭（``aclose``）开始后
拒绝一切新请求，并等待**所有已接纳请求**收尾——包括已占用名额、但仍在
``create_subprocess_exec`` 创建途中尚未登记进 ``_children`` 的请求——直到
其子进程终止回收、名额全部释放后才返回。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# app 包所在目录：子进程以此为工作目录，保证 ``python -m app.align_worker``
# 能导入应用代码，与主进程自身的工作目录无关
_PACKAGE_PARENT = Path(__file__).resolve().parents[1]

# 默认子进程入口：运行包内 worker（测试可注入其他 argv）
_DEFAULT_WORKER_ARGV = ("-m", "app.align_worker")


def positive_int_env(name: str, default: int) -> int:
    """读取正整数环境变量；未设置时返回默认值，非法值抛 ``ValueError``。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip(), 10)
    except ValueError:
        raise ValueError(
            f"环境变量 {name} 必须为正整数，当前值为 {raw!r}，服务无法启动"
        ) from None
    if value <= 0:
        raise ValueError(
            f"环境变量 {name} 必须为正整数，当前值为 {raw!r}，服务无法启动"
        )
    return value


# 启动期配置：非法值在此即抛出，服务启动失败
ALIGN_MAX_PROCESSES = positive_int_env("ALIGN_MAX_PROCESSES", 2)
ALIGN_TIMEOUT_SECONDS = positive_int_env("ALIGN_TIMEOUT_SECONDS", 10)


class AlignBusyError(Exception):
    """没有空闲的对齐子进程名额（接口映射为 503）。"""


class AlignTimeoutError(Exception):
    """对齐计算超过时限（接口映射为 504）。"""


class AlignWorkerError(Exception):
    """对齐子进程异常退出或输出无法解析（接口映射为 500）。"""


class AlignExecutor:
    """以独立子进程执行对齐计算的执行器。

    - 名额（``max_processes``）统计**运行中或清理中**的已接纳请求：接纳时
      占用（即使子进程尚在 ``create_subprocess_exec`` 创建途中），该请求的
      子进程终止回收完成后才释放；
    - 超时、客户端断开（任务被取消）、子进程异常退出都汇入 ``_terminate``
      终止-回收路径；
    - ``aclose`` 在服务关闭时先置关闭标志（此后新请求一律拒绝），再终止并
      回收全部已登记子进程，并等待所有已接纳但尚未收尾的请求（含仍在创建
      途中、子进程还没登记的请求）完成终止-回收、名额归零后才返回。
    """

    def __init__(
        self,
        max_processes: int,
        timeout_seconds: int,
        worker_argv: tuple[str, ...] = _DEFAULT_WORKER_ARGV,
    ) -> None:
        if max_processes <= 0 or timeout_seconds <= 0:
            raise ValueError("max_processes 与 timeout_seconds 必须为正整数")
        self._max_processes = max_processes
        self._timeout_seconds = timeout_seconds
        self._worker_argv = tuple(worker_argv)
        self._active = 0                                  # 已接纳且未释放名额的请求数
        self._children: set[asyncio.subprocess.Process] = set()
        self._closed = False
        # 协调接纳名额的检查/占用/释放与关闭流程：关闭后不得再有新请求占用
        # 名额；名额释放与 aclose 的等待都在此条件变量上同步，避免漏唤醒
        self._condition = asyncio.Condition()

    @property
    def active(self) -> int:
        """当前占用名额的子进程数（运行中或清理中）。"""
        return self._active

    async def run(self, left: str, right: str) -> dict:
        """接纳一次对齐计算；无空位立即失败，超时/被取消即终止并回收子进程。"""
        # 关闭标志与名额占用必须在同一临界区内判定/自增：保证 aclose 置位后
        # 不会再有请求占用名额，也不会有“检查时未关闭、占用时已关闭”的窗口
        async with self._condition:
            if self._closed:
                raise AlignBusyError("服务正在关闭，不再接纳新的对齐请求")
            if self._active >= self._max_processes:
                raise AlignBusyError(
                    f"对齐子进程名额已满（{self._max_processes} 个运行或清理中），请稍后重试"
                )
            self._active += 1
        try:
            # 时限从接纳（占用名额）起算
            deadline = asyncio.get_running_loop().time() + self._timeout_seconds
            return await self._run_in_child(left, right, deadline)
        finally:
            # _run_in_child 返回时子进程必然已回收，此处才释放名额；
            # 与上面的自增一一对应，每个已接纳请求恰好释放一次，关闭流程与
            # 请求自身的收尾不会重复释放
            async with self._condition:
                self._active -= 1
                self._condition.notify_all()  # 唤醒等待名额归零的 aclose

    async def _run_in_child(self, left: str, right: str, deadline: float) -> dict:
        loop = asyncio.get_running_loop()
        if deadline - loop.time() <= 0:
            raise AlignTimeoutError(
                f"对齐计算超过 {self._timeout_seconds} 秒时限，已终止"
            )
        payload = json.dumps(
            {"left": left, "right": right}, ensure_ascii=False
        ).encode("utf-8")
        # 等待子进程创建期间名额已被占用，但 proc 尚未登记进 _children；
        # 关闭流程以名额（而非 _children）为等待依据，因此不会漏掉本请求
        proc = await self._spawn_child()
        async with self._condition:
            self._children.add(proc)
            closing = self._closed
        try:
            if closing:
                # 子进程在关闭开始后才创建完成：关闭快照未能覆盖它，故由本
                # 请求立即终止回收（与 aclose 终止已登记子进程同一条路径），
                # 不投入计算；以 worker 异常路径结束（接口映射 500），与关闭
                # 前已登记子进程被 aclose 终止的请求表现一致
                await self._terminate(proc)
                raise AlignWorkerError("服务正在关闭，对齐子进程已终止并回收")
            try:
                remaining = max(deadline - loop.time(), 0.0)
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=payload), timeout=remaining
                )
            except asyncio.TimeoutError:
                await self._terminate(proc)
                raise AlignTimeoutError(
                    f"对齐计算超过 {self._timeout_seconds} 秒时限，子进程已终止并回收"
                ) from None
            if proc.returncode != 0:
                detail = stderr.decode("utf-8", "replace").strip()
                raise AlignWorkerError(
                    f"对齐子进程异常退出（退出码 {proc.returncode}）"
                    + (f"：{detail[:200]}" if detail else "")
                )
            try:
                return json.loads(stdout.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise AlignWorkerError("对齐子进程输出无法解析") from exc
        finally:
            # 正常结束时为幂等空操作；超时/取消/关闭/异常时负责终止并回收
            try:
                await self._terminate(proc)
            finally:
                async with self._condition:
                    self._children.discard(proc)

    async def _spawn_child(self) -> asyncio.subprocess.Process:
        """拉起 worker 子进程（单独成方法，便于测试注入创建延迟）。"""
        return await asyncio.create_subprocess_exec(
            sys.executable,
            *self._worker_argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(_PACKAGE_PARENT),
        )

    async def _terminate(self, proc: asyncio.subprocess.Process) -> None:
        """终止并回收子进程；返回时进程已被回收。重复调用安全。"""
        if proc.returncode is not None:
            return
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()

    async def aclose(self) -> None:
        """服务关闭：拒绝新请求，等待全部已接纳请求（含创建途中）终止回收。

        返回时所有已接纳请求均已走到终止-回收路径、子进程全部回收、名额
        （``_active``）归零；``run`` 与本方法都只走同一条回收路径，不会重复
        终止子进程或重复释放名额。
        """
        async with self._condition:
            self._closed = True
        # 循环终止：置位关闭标志后，仍可能有请求正卡在 _spawn_child（子进程
        # 尚未登记进 _children）。每轮快照终止当前已登记的子进程，再等待名额
        # 归零；创建途中的请求登记后会自行检测到关闭标志并终止回收、释放名额，
        # 唤醒本循环后继续下一轮，直到没有任何已接纳请求
        while True:
            async with self._condition:
                if self._active == 0:
                    return
                children = list(self._children)
            for proc in children:  # 先全部终止，再统一等待回收
                if proc.returncode is None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
            for proc in children:
                try:
                    await proc.wait()
                except Exception:
                    pass
            # 已登记子进程的终止已完成，但对应 run 的 finally（_terminate 回收、
            # 名额释放）可能还没被调度；创建途中的 run 也需要一轮调度才能完成
            # 登记-终止-回收。让出事件循环，避免在名额释放前就进入条件等待
            await asyncio.sleep(0)
            async with self._condition:
                # 等待已登记子进程的请求收尾释放名额，以及创建途中的请求完成
                # 登记-终止-回收；名额释放时的 notify_all 不会漏掉本等待
                await self._condition.wait_for(lambda: self._active == 0)
