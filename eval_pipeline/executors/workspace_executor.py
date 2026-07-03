"""Local workspace executor for file-producing benchmarks.

This executor gives SubAgent a persistent working directory without Docker.
It is intended for benchmarks where the useful observation is a set of files,
logs, and command outputs rather than an environment-provided reward.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from .base_executor import BaseExecutor


class WorkspaceExecutor(BaseExecutor):
    """Execute commands in a task-specific local workspace."""

    def __init__(
        self,
        task_id: str,
        workspace_dir: Path,
        problem_dir: Path | None = None,
        submission_schema_path: Path | None = None,
        verifier_logs_dir: Path | None = None,
        agent_logs_dir: Path | None = None,
        timeout: int = 7200,
        env_init: Optional[dict[str, str]] = None,
    ):
        workspace_dir = Path(workspace_dir)
        super().__init__(
            task_id=task_id,
            task_dir=workspace_dir,
            task_config={},
            verifier_logs_dir=verifier_logs_dir or (workspace_dir / "logs"),
            agent_logs_dir=agent_logs_dir or (workspace_dir / "logs"),
            timeout=timeout,
            env_init=env_init,
        )
        self.workspace_dir = workspace_dir
        self.problem_dir = Path(problem_dir) if problem_dir else None
        self.submission_schema_path = (
            Path(submission_schema_path) if submission_schema_path else None
        )

    async def start_container(self):
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.verifier_logs_dir.mkdir(parents=True, exist_ok=True)
        self.agent_logs_dir.mkdir(parents=True, exist_ok=True)
        for name in ("code", "results", "logs"):
            (self.workspace_dir / name).mkdir(parents=True, exist_ok=True)

    async def execute_command(self, command: str, timeout: Optional[int] = None) -> Tuple[str, int]:
        await self.start_container()
        env = os.environ.copy()
        env.update(self.env_init)
        env["LANG"] = env.get("LANG") or "C.UTF-8"
        env["LC_ALL"] = env.get("LC_ALL") or "C.UTF-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env.setdefault("OUTPUT_DIR", str(self.workspace_dir))
        if self.problem_dir:
            env.setdefault("PROBLEM_DIR", str(self.problem_dir))
        if self.submission_schema_path:
            env.setdefault("SUBMISSION_SCHEMA_PATH", str(self.submission_schema_path))

        return await asyncio.to_thread(
            self._execute_command_sync,
            command,
            env,
            timeout or self.timeout,
        )

    def _execute_command_sync(
        self,
        command: str,
        env: dict[str, str],
        timeout: int,
    ) -> Tuple[str, int]:
        try:
            if sys.platform.startswith("win"):
                bash = _find_windows_bash()
                if bash:
                    args = [bash, "-lc", command]
                    shell = False
                else:
                    args = command
                    shell = True
            else:
                args = ["/bin/bash", "-lc", command]
                shell = False

            proc = subprocess.run(
                args,
                cwd=str(self.workspace_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                input=None,
                timeout=timeout,
                shell=shell,
            )
            output = _decode(proc.stdout or b"")
            err = _decode(proc.stderr or b"")
            if err:
                output = output + ("\n" if output else "") + err
            return output, int(proc.returncode or 0)
        except subprocess.TimeoutExpired as exc:
            output = _decode(exc.stdout or b"")
            err = _decode(exc.stderr or b"")
            if err:
                output = output + ("\n" if output else "") + err
            return output + f"\n[workspace-executor] timeout after {timeout}s", -1
        except Exception as exc:
            return f"Command execution error: {type(exc).__name__}: {exc}", -1

    async def run_tests(self) -> float:
        # RubricWorkflow scoring is performed after submit by the benchmark.
        return 0.0

    async def cleanup(self):
        return None

    def get_container_id(self) -> Optional[str]:
        return None


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _find_windows_bash() -> str | None:
    candidates = [
        os.environ.get("GIT_BASH"),
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        shutil.which("bash"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.exists():
            continue
        normalized = str(path).lower()
        if normalized.endswith(r"\system32\bash.exe"):
            continue
        return str(path)
    return None
