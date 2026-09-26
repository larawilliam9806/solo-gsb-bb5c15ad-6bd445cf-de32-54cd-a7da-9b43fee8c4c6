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

名额自接纳时占用，到子进程**回收完成**后才释放；正常结束、超时、客户端
断开（任务取消）、子进程异常退出与服务关闭都汇入同一条终止-回收路径，
不泄漏进程、不重复释放名额。服务关闭（``aclose``）同时覆盖**已接纳但
子进程仍在创建中**的请求：其子进程登记后立即被终止回收，全部在途请求
收尾、名额释放后 ``aclose`` 才返回；关闭开始后拒绝接纳新请求。
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

    - 名额（``max_processes``）统计**已接纳未收尾**的请求（子进程创建中、
      运行中或清理中）：接纳时占用，子进程回收完成后才释放；
    - 超时、客户端断开（任务被取消）、子进程异常退出都汇入 ``_terminate``
      终止-回收路径；``aclose`` 在服务关闭时终止并回收全部残留子进程，
      并等待已接纳但尚在创建子进程的请求收尾（名额释放）后才返回。
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
        self._active = 0  # 已接纳未收尾的请求数（子进程创建中、运行中或清理中）
        self._children: set[asyncio.subprocess.Process] = set()
        self._closed = False
        # 无在途请求时置位；aclose 借此等待已接纳请求全部收尾
        self._drained = asyncio.Event()
        self._drained.set()

    @property
    def active(self) -> int:
        """当前占用名额的请求数（子进程创建中、运行中或清理中）。"""
        return self._active

    async def run(self, left: str, right: str) -> dict:
        """接纳一次对齐计算；无空位立即失败，超时/被取消即终止并回收子进程。"""
        if self._closed:
            raise AlignBusyError("服务正在关闭，不再接纳新的对齐请求")
        if self._active >= self._max_processes:
            raise AlignBusyError(
                f"对齐子进程名额已满（{self._max_processes} 个运行或清理中），请稍后重试"
            )
        self._active += 1
        self._drained.clear()
        try:
            # 时限从接纳（占用名额）起算
            deadline = asyncio.get_running_loop().time() + self._timeout_seconds
            return await self._run_in_child(left, right, deadline)
        finally:
            # _run_in_child 返回时子进程必然已回收，此处才释放名额；
            # 与上面的自增一一对应，并发收尾也不会重复释放
            self._active -= 1
            if self._active == 0:
                self._drained.set()  # 全部在途请求收尾完毕，唤醒 aclose

    async def _run_in_child(self, left: str, right: str, deadline: float) -> dict:
        loop = asyncio.get_running_loop()
        if deadline - loop.time() <= 0:
            raise AlignTimeoutError(
                f"对齐计算超过 {self._timeout_seconds} 秒时限，已终止"
            )
        payload = json.dumps(
            {"left": left, "right": right}, ensure_ascii=False
        ).encode("utf-8")
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            *self._worker_argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(_PACKAGE_PARENT),
        )
        self._children.add(proc)
        try:
            if self._closed:
                # 关闭已开始：子进程不得继续运行，登记后立即转入下方
                # finally 终止回收，aclose 待名额释放后才返回
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
            # 正常结束时为幂等空操作；超时/取消/异常时负责终止并回收
            try:
                await self._terminate(proc)
            finally:
                self._children.discard(proc)

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
        """服务关闭：拒绝新请求，终止并回收全部残留子进程。

        已接纳但子进程仍在创建中的请求同样在关闭范围内：其子进程登记后
        立即被终止回收；本方法等到全部在途请求收尾、名额释放后才返回，
        返回后不再留有属于本执行器的运行进程。
        """
        self._closed = True
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
        # 尚在创建子进程的请求：登记后自行终止回收并释放名额，在此等齐
        await self._drained.wait()
