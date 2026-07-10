"""Remote Terminal-Bench Docker executor client.

This mirrors DockerExecutor's async interface but forwards operations to a
remote server that owns the actual Docker daemon.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

from .base_executor import BaseExecutor


def remote_docker_enabled() -> bool:
    value = os.environ.get("UNO_DOCKER_EXECUTOR", "").strip().lower()
    return value in {"remote", "http", "server"} or bool(os.environ.get("UNO_REMOTE_DOCKER_URL", "").strip())


def _server_url() -> str:
    url = os.environ.get("UNO_REMOTE_DOCKER_URL", "").strip()
    if not url:
        raise RuntimeError("UNO_REMOTE_DOCKER_URL is required when UNO_DOCKER_EXECUTOR=remote")
    return url.rstrip("/")


def _auth_token() -> str:
    return os.environ.get("UNO_REMOTE_DOCKER_TOKEN", "").strip()


class RemoteDockerExecutor(BaseExecutor):
    """Proxy executor that runs Terminal-Bench Docker work on a remote server."""

    def __init__(
        self,
        task_id: str,
        task_dir: Path,
        task_config: dict,
        verifier_logs_dir: Path,
        agent_logs_dir: Path,
        docker_timeout: int = 600,
        env_init: Optional[dict[str, str]] = None,
        server_url: Optional[str] = None,
    ):
        super().__init__(
            task_id=task_id,
            task_dir=task_dir,
            task_config=task_config,
            verifier_logs_dir=verifier_logs_dir,
            agent_logs_dir=agent_logs_dir,
            timeout=docker_timeout,
            env_init=env_init,
        )
        self.server_url = (server_url or _server_url()).rstrip("/")
        self.docker_timeout = docker_timeout
        self.session_id: Optional[str] = None
        self.container_id: Optional[str] = None

    async def start_container(self):
        payload = {
            "task_id": self.task_id,
            "task_config": self.task_config,
            "docker_timeout": self.docker_timeout,
            "env_init": self.env_init,
        }
        data = await self._request("POST", "/start", payload, timeout=self.docker_timeout + 120)
        self.session_id = data["session_id"]
        self.container_id = data.get("container_id")

    async def execute_command(self, command: str, timeout: Optional[int] = None) -> Tuple[str, int]:
        self._require_session()
        data = await self._request(
            "POST",
            "/exec",
            {"session_id": self.session_id, "command": command, "timeout": timeout},
            timeout=(timeout or self.docker_timeout) + 30,
        )
        return str(data.get("output", "")), int(data.get("exit_code", -1))

    async def run_tests(self) -> float:
        self._require_session()
        data = await self._request(
            "POST",
            "/run_tests",
            {"session_id": self.session_id},
            timeout=self.docker_timeout + 300,
        )
        return float(data.get("reward", 0.0) or 0.0)

    async def cleanup(self):
        if not self.session_id:
            return
        try:
            await self._request(
                "POST",
                "/cleanup",
                {"session_id": self.session_id},
                timeout=60,
            )
        finally:
            self.session_id = None
            self.container_id = None

    def get_container_id(self) -> Optional[str]:
        return self.container_id

    def _require_session(self) -> None:
        if not self.session_id:
            raise RuntimeError("Remote Docker session not started")

    async def _request(self, method: str, path: str, payload: dict, timeout: int) -> dict:
        return await asyncio.to_thread(self._request_sync, method, path, payload, timeout)

    def _request_sync(self, method: str, path: str, payload: dict, timeout: int) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.server_url + path,
            data=body,
            method=method,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {_auth_token()}"} if _auth_token() else {}),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"remote docker {path} failed: HTTP {exc.code}: {detail}") from exc
        data = json.loads(raw or "{}")
        if not data.get("ok", False):
            raise RuntimeError(f"remote docker {path} failed: {data.get('error', 'unknown error')}")
        return data

