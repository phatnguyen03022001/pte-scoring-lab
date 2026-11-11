"""Bound subprocesses and clean up their complete POSIX process groups."""

from __future__ import annotations

import os
import signal
import subprocess
import time
import fcntl
from pathlib import Path
from typing import Any


DEADLINE_ENV = "PTE_EXECUTION_DEADLINE"
PROCESS_GROUP_REGISTRY_ENV = "PTE_PROCESS_GROUP_REGISTRY"
TERMINATION_GRACE_SECONDS = 2.0


def _update_registry(path: str | None, group_id: int, *, remove: bool = False) -> None:
    if not path or os.name != "posix":
        return
    registry = Path(path)
    try:
        with registry.open("a+", encoding="ascii") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.seek(0)
            groups = {int(line) for line in stream if line.strip().isdigit()}
            if remove:
                groups.discard(group_id)
            else:
                groups.add(group_id)
            stream.seek(0)
            stream.truncate()
            stream.writelines(f"{item}\n" for item in sorted(groups))
            stream.flush()
            fcntl.flock(stream, fcntl.LOCK_UN)
    except OSError:
        # Removing a stale entry is best effort; adding one must succeed when
        # an outer scorer relies on it for cancellation cleanup.
        if remove or not path:
            return
        raise


def _terminate_registered_groups(path: str | None) -> None:
    if not path or os.name != "posix":
        return
    try:
        with Path(path).open("a+", encoding="ascii") as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            stream.seek(0)
            groups = [int(line) for line in stream if line.strip().isdigit()]
            fcntl.flock(stream, fcntl.LOCK_UN)
    except OSError:
        return
    for group_id in groups:
        try:
            os.killpg(group_id, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            continue
    end = time.monotonic() + TERMINATION_GRACE_SECONDS
    while time.monotonic() < end:
        active = []
        for group_id in groups:
            try:
                os.killpg(group_id, 0)
                active.append(group_id)
            except (ProcessLookupError, PermissionError):
                pass
        if not active:
            return
        time.sleep(0.02)
    for group_id in groups:
        try:
            os.killpg(group_id, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if os.name != "posix":
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=TERMINATION_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    end = time.monotonic() + TERMINATION_GRACE_SECONDS
    while time.monotonic() < end:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            # Some macOS sandbox profiles permit the termination signal but
            # deny signal-zero probes for a delegated process group.
            return
        time.sleep(0.02)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _terminate_and_reap(
    process: subprocess.Popen[str], registry: str | None = None, *, cleanup_registered: bool = False,
) -> tuple[str | bytes | None, str | bytes | None]:
    _terminate_process_group(process)
    if cleanup_registered:
        _terminate_registered_groups(registry)
    try:
        return process.communicate(timeout=TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        return process.communicate()


def deadline_from_env(env: dict[str, str] | None = None) -> float | None:
    raw = (env or os.environ).get(DEADLINE_ENV)
    if raw is None:
        return None
    try:
        deadline = float(raw)
    except ValueError:
        return None
    return deadline if deadline > 0 else None


def remaining_seconds(deadline: float | None, timeout: float) -> float:
    remaining = timeout if deadline is None else min(timeout, deadline - time.monotonic())
    if remaining <= 0:
        raise subprocess.TimeoutExpired("execution deadline", timeout)
    return remaining


def run_bounded(
    command: list[str], *, timeout: float, deadline: float | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """Run a child with a new session; timeout kills and reaps its process tree."""
    timeout = remaining_seconds(deadline, timeout)
    capture_output = kwargs.pop("capture_output", False)
    check = kwargs.pop("check", False)
    cleanup_registered = kwargs.pop("cleanup_process_groups", False)
    child_env = kwargs.get("env") or os.environ
    registry = child_env.get(PROCESS_GROUP_REGISTRY_ENV)
    process = subprocess.Popen(
        command,
        start_new_session=(os.name == "posix"),
        stdout=subprocess.PIPE if capture_output else kwargs.pop("stdout", None),
        stderr=subprocess.PIPE if capture_output else kwargs.pop("stderr", None),
        text=kwargs.pop("text", False),
        **kwargs,
    )
    if os.name == "posix":
        try:
            _update_registry(registry, process.pid)
        except OSError:
            _terminate_and_reap(process)
            raise
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = _terminate_and_reap(process, registry, cleanup_registered=cleanup_registered)
        _update_registry(registry, process.pid, remove=True)
        raise subprocess.TimeoutExpired(command, timeout, output=stdout or exc.output, stderr=stderr or exc.stderr) from exc
    except BaseException:
        _terminate_and_reap(process, registry, cleanup_registered=cleanup_registered)
        _update_registry(registry, process.pid, remove=True)
        raise
    _terminate_process_group(process)
    _update_registry(registry, process.pid, remove=True)
    if cleanup_registered:
        _terminate_registered_groups(registry)
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if check and completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command, output=stdout, stderr=stderr)
    return completed
