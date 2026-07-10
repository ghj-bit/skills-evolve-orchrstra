"""HTTP server that exposes Terminal-Bench DockerExecutor sessions.

Run this on the machine that has Docker:

    python -m eval_pipeline.executors.remote_docker_server \
      --tasks-dir /path/to/data/terminal_task_rl --host 0.0.0.0 --port 18080

The client side selects it with:

    UNO_DOCKER_EXECUTOR=remote
    UNO_REMOTE_DOCKER_URL=http://host:18080
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from .docker_executor import DockerExecutor
from .docker_manager import DockerComposeManager


_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE_YAML = Path(__file__).parent / "docker-compose-build.yaml"


class SessionStore:
    def __init__(self, tasks_dir: Path, logs_dir: Path, docker_timeout: int):
        self.tasks_dir = tasks_dir
        self.logs_dir = logs_dir
        self.docker_timeout = docker_timeout
        self.docker_manager = DockerComposeManager(_COMPOSE_YAML)
        self.sessions: dict[str, DockerExecutor] = {}
        self.lock = threading.Lock()

    def task_dir_for(self, task_id: str) -> Path:
        direct = self.tasks_dir / task_id
        if (direct / "task.toml").exists():
            return direct
        nested = list(direct.glob("*/task.toml")) if direct.exists() else []
        if nested:
            return nested[0].parent
        matches = list(self.tasks_dir.glob(f"**/{task_id}/task.toml"))
        if matches:
            return matches[0].parent
        raise FileNotFoundError(f"task.toml not found for task_id={task_id!r} under {self.tasks_dir}")

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        task_id = str(payload["task_id"])
        task_dir = self.task_dir_for(task_id)
        config = payload.get("task_config")
        if not isinstance(config, dict):
            with (task_dir / "task.toml").open("rb") as f:
                config = tomllib.load(f)

        session_id = uuid.uuid4().hex
        base_logs = self.logs_dir / task_id / session_id
        verifier_logs = base_logs / "verifier"
        agent_logs = base_logs / "agent"
        verifier_logs.mkdir(parents=True, exist_ok=True)
        agent_logs.mkdir(parents=True, exist_ok=True)
        executor = DockerExecutor(
            task_id=task_id,
            task_dir=task_dir,
            task_config=config,
            verifier_logs_dir=verifier_logs,
            agent_logs_dir=agent_logs,
            docker_manager=self.docker_manager,
            docker_timeout=int(payload.get("docker_timeout") or self.docker_timeout),
            env_init=payload.get("env_init") or None,
        )
        asyncio.run(executor.start_container())
        with self.lock:
            self.sessions[session_id] = executor
        return {"session_id": session_id, "container_id": executor.get_container_id()}

    def get(self, session_id: str) -> DockerExecutor:
        with self.lock:
            executor = self.sessions.get(session_id)
        if executor is None:
            raise KeyError(f"unknown session_id={session_id!r}")
        return executor

    def exec(self, payload: dict[str, Any]) -> dict[str, Any]:
        executor = self.get(str(payload["session_id"]))
        output, exit_code = asyncio.run(
            executor.execute_command(
                str(payload.get("command") or ""),
                timeout=payload.get("timeout"),
            )
        )
        return {"output": output, "exit_code": exit_code}

    def run_tests(self, payload: dict[str, Any]) -> dict[str, Any]:
        executor = self.get(str(payload["session_id"]))
        reward = asyncio.run(executor.run_tests())
        return {"reward": reward}

    def cleanup(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload["session_id"])
        with self.lock:
            executor = self.sessions.pop(session_id, None)
        if executor is not None:
            asyncio.run(executor.cleanup())
        return {"cleaned": executor is not None}

    def cleanup_all(self) -> None:
        with self.lock:
            items = list(self.sessions.items())
            self.sessions.clear()
        for _, executor in items:
            try:
                asyncio.run(executor.cleanup())
            except Exception:
                pass


def make_handler(store: SessionStore, token: str):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send({"ok": True, "sessions": len(store.sessions)})
            else:
                self._send({"ok": False, "error": "not found"}, status=404)

        def do_POST(self):  # noqa: N802
            if token:
                expected = f"Bearer {token}"
                if self.headers.get("Authorization") != expected:
                    self._send({"ok": False, "error": "unauthorized"}, status=401)
                    return
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                if self.path == "/start":
                    result = store.start(payload)
                elif self.path == "/exec":
                    result = store.exec(payload)
                elif self.path == "/run_tests":
                    result = store.run_tests(payload)
                elif self.path == "/cleanup":
                    result = store.cleanup(payload)
                else:
                    self._send({"ok": False, "error": "not found"}, status=404)
                    return
                self._send({"ok": True, **result})
            except Exception as exc:
                self._send(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-4000:],
                    },
                    status=500,
                )

        def log_message(self, fmt, *args):
            print(f"[remote-docker] {self.address_string()} {fmt % args}", flush=True)

        def _send(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote Terminal-Bench Docker executor server")
    parser.add_argument("--host", default=os.environ.get("UNO_REMOTE_DOCKER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("UNO_REMOTE_DOCKER_PORT", "18080")))
    parser.add_argument(
        "--tasks-dir",
        default=os.environ.get("TERMINAL_BENCH_TASKS_DIR", str(_REPO_ROOT / "data" / "terminal_task_rl")),
    )
    parser.add_argument(
        "--logs-dir",
        default=os.environ.get("UNO_REMOTE_DOCKER_LOGS_DIR", str(_REPO_ROOT / "data" / "remote_docker_logs")),
    )
    parser.add_argument("--docker-timeout", type=int, default=int(os.environ.get("UNO_REMOTE_DOCKER_TIMEOUT", "600")))
    parser.add_argument("--token", default=os.environ.get("UNO_REMOTE_DOCKER_TOKEN", ""))
    args = parser.parse_args()

    store = SessionStore(Path(args.tasks_dir), Path(args.logs_dir), args.docker_timeout)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store, args.token))
    print(
        f"Remote Docker server listening on http://{args.host}:{args.port} "
        f"tasks_dir={args.tasks_dir} logs_dir={args.logs_dir}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        store.cleanup_all()


if __name__ == "__main__":
    main()

