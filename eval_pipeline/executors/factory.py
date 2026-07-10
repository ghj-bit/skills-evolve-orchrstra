"""Executor selection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .docker_manager import DockerComposeManager
from .docker_executor import DockerExecutor
from .remote_docker_executor import RemoteDockerExecutor, remote_docker_enabled


def make_terminalbench_executor(
    *,
    task_id: str,
    task_dir: Path,
    task_config: dict,
    verifier_logs_dir: Path,
    agent_logs_dir: Path,
    docker_manager: DockerComposeManager,
    docker_timeout: int = 600,
    env_init: Optional[dict[str, str]] = None,
):
    """Create a local or remote Terminal-Bench Docker executor.

    Default is local. Set UNO_DOCKER_EXECUTOR=remote and UNO_REMOTE_DOCKER_URL
    to proxy all container operations to a remote Docker server.
    """
    if remote_docker_enabled():
        return RemoteDockerExecutor(
            task_id=task_id,
            task_dir=task_dir,
            task_config=task_config,
            verifier_logs_dir=verifier_logs_dir,
            agent_logs_dir=agent_logs_dir,
            docker_timeout=docker_timeout,
            env_init=env_init,
        )
    return DockerExecutor(
        task_id=task_id,
        task_dir=task_dir,
        task_config=task_config,
        verifier_logs_dir=verifier_logs_dir,
        agent_logs_dir=agent_logs_dir,
        docker_manager=docker_manager,
        docker_timeout=docker_timeout,
        env_init=env_init,
    )

