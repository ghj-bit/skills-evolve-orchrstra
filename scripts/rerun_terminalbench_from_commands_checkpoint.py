#!/usr/bin/env python
"""Replay a Terminal-Bench run up to a delegate breakpoint, then rerun it.

This is a command-replay approximation of a Docker checkpoint:
1. Start a fresh Terminal-Bench task container.
2. Optionally replay commands from the original commands.log before the target tool_call_id.
3. Rerun the target delegate, optionally with skills enabled.
4. Optionally run the verifier.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from eval_pipeline.benchmarks.terminalbench import COMPOSE_YAML, _resolve_model_endpoint
from eval_pipeline.executors.docker_executor import DockerExecutor
from eval_pipeline.executors.docker_manager import DockerComposeManager
from uno_orchestor.agents.subagent import SubAgent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _iter_delegates(trajectory: dict):
    for item in trajectory.get("trajectory", []):
        for tc in item.get("tool_calls", []) or []:
            if tc.get("name") != "delegate_task":
                continue
            args = tc.get("arguments") or {}
            yield {
                "attempt": item.get("attempt"),
                "tool_call_id": tc.get("id"),
                "worker_model": args.get("worker_model", ""),
                "instruction": args.get("instruction", ""),
            }


def _select_delegate(trajectory: dict, tool_call_id: str | None, worker_model: str | None) -> dict:
    delegates = list(_iter_delegates(trajectory))
    if tool_call_id:
        for delegate in delegates:
            if delegate["tool_call_id"] == tool_call_id:
                return delegate
        raise SystemExit(f"tool_call_id not found: {tool_call_id}")
    if worker_model:
        for delegate in delegates:
            if delegate["worker_model"] == worker_model:
                return delegate
        raise SystemExit(f"worker_model not found: {worker_model}")
    for delegate in delegates:
        if "qwen" in delegate["worker_model"].lower():
            return delegate
    raise SystemExit("No Qwen delegate found. Pass --tool-call-id or --worker-model.")


def _target_begin_span(commands_log: str, tool_call_id: str) -> tuple[int, int]:
    marker = "[SubTask BEGIN]"
    pos = 0
    while True:
        start = commands_log.find(marker, pos)
        if start < 0:
            raise SystemExit(f"Cannot find SubTask BEGIN for tool_call_id={tool_call_id}")
        next_start = commands_log.find(marker, start + len(marker))
        end = next_start if next_start >= 0 else len(commands_log)
        block = commands_log[start:end]
        if f'"tool_call_id": "{tool_call_id}"' in block:
            return start, end
        pos = end


def _commands_before_tool_call(commands_log_path: Path, tool_call_id: str) -> list[str]:
    text = commands_log_path.read_text(encoding="utf-8", errors="replace")
    target_start, _ = _target_begin_span(text, tool_call_id)
    prefix = text[:target_start]
    commands = []
    for match in re.finditer(
        r"(?ms)^\[Step\s+\d+\]\s+(.*?)\nExit:\s+[-\d]+",
        prefix,
    ):
        command = match.group(1).strip()
        if command:
            commands.append(command)
    return commands


def _load_task(task_id: str) -> tuple[Path, dict, str]:
    task_dir = ROOT / "data" / "terminal-bench" / "tasks" / task_id
    task_toml = task_dir / "task.toml"
    instruction_md = task_dir / "instruction.md"
    if not task_toml.exists():
        raise SystemExit(f"Missing task.toml: {task_toml}")
    if not instruction_md.exists():
        raise SystemExit(f"Missing instruction.md: {instruction_md}")
    with task_toml.open("rb") as f:
        cfg = tomllib.load(f)
    instruction = instruction_md.read_text(encoding="utf-8-sig").strip()
    return task_dir, cfg, instruction


async def _main_async(args: argparse.Namespace) -> None:
    _load_dotenv(ROOT / ".env")

    trajectory_path = Path(args.trajectory).resolve()
    trajectory = _read_json(trajectory_path)
    selected = _select_delegate(trajectory, args.tool_call_id, args.worker_model)
    task_id = trajectory.get("task_id") or trajectory_path.parent.name
    commands_log = Path(args.commands_log) if args.commands_log else trajectory_path.parent / "agent" / "commands.log"
    commands = (
        _commands_before_tool_call(commands_log, selected["tool_call_id"])
        if args.resume_from_commands_checkpoint
        else []
    )

    output_dir = Path(args.output_dir) if args.output_dir else trajectory_path.parent / "commands_checkpoint_rerun" / selected["tool_call_id"]
    verifier_logs = output_dir / "verifier"
    agent_logs = output_dir / "agent"
    verifier_logs.mkdir(parents=True, exist_ok=True)
    agent_logs.mkdir(parents=True, exist_ok=True)

    task_dir, task_config, original_instruction = _load_task(task_id)
    model = args.model or selected["worker_model"]
    api_base, api_key = _resolve_model_endpoint(
        model,
        os.environ.get("API_BASE") or os.environ.get("DEEPSEEK_API_BASE") or "https://api.deepseek.com",
        os.environ.get("API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or "EMPTY",
    )

    os.environ["SUBAGENT_ENABLE_SKILLS"] = "1" if args.enable_skills else "0"
    os.environ["SUBAGENT_SKILLS_TOP_K"] = str(args.top_k)
    os.environ["SUBAGENT_SKILLS_MODE"] = args.skills_mode
    os.environ["SUBAGENT_SKILLS_MODELS"] = model
    os.environ["TERMINAL_BENCH_SKILLS_PATH"] = str(Path(args.skills).resolve())

    manager = DockerComposeManager(COMPOSE_YAML)
    executor = DockerExecutor(
        task_id=task_id,
        task_dir=task_dir,
        task_config=task_config,
        verifier_logs_dir=verifier_logs,
        agent_logs_dir=agent_logs,
        docker_manager=manager,
        docker_timeout=args.docker_timeout,
    )

    replay_log = output_dir / "replay_commands.log"
    result_payload = {
        "task_id": task_id,
        "source_trajectory": str(trajectory_path),
        "source_commands_log": str(commands_log.resolve()),
        "selected_delegate": selected,
        "resume_from_commands_checkpoint": args.resume_from_commands_checkpoint,
        "replayed_command_count": len(commands),
        "model": model,
        "api_base": api_base,
        "skills": {
            "enabled": bool(args.enable_skills),
            "path": str(Path(args.skills).resolve()),
            "top_k": args.top_k,
            "mode": args.skills_mode,
        },
    }

    try:
        await executor.start_container()
        result_payload["container_id"] = executor.get_container_id()

        with replay_log.open("w", encoding="utf-8") as f:
            if not args.resume_from_commands_checkpoint:
                f.write("Command replay disabled by --no-resume-from-commands-checkpoint.\n")
            for idx, command in enumerate(commands, start=1):
                output, exit_code = await executor.execute_command(command, timeout=args.command_timeout)
                f.write(f"[Replay Step {idx}] {command}\n")
                f.write(f"Exit: {exit_code}\n")
                f.write(output)
                f.write("\n" + "-" * 80 + "\n")
                if exit_code != 0 and args.stop_on_replay_error:
                    raise RuntimeError(f"Replay command {idx} failed with exit={exit_code}")

        subagent = SubAgent(
            api_base=api_base,
            api_key=api_key,
            max_steps=args.max_steps,
            cmd_timeout=args.command_timeout,
        )
        sub_result = await subagent.run(
            model=model,
            task_instruction=selected["instruction"],
            original_question=original_instruction,
            executor=executor,
            agent_logs_dir=agent_logs,
            tool_call_id=selected["tool_call_id"],
        )
        result_payload["sub_result"] = sub_result

        if args.run_tests:
            result_payload["reward"] = await executor.run_tests()

    finally:
        if not args.keep_container:
            await executor.cleanup()
        else:
            result_payload["kept_container"] = executor.get_container_id()
        (output_dir / "rerun_result.json").write_text(
            json.dumps(result_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"Wrote: {output_dir / 'rerun_result.json'}")
    print(f"Replay commands: {len(commands)}")
    if "reward" in result_payload:
        print(f"Reward: {result_payload['reward']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--commands-log")
    parser.add_argument("--tool-call-id")
    parser.add_argument("--worker-model")
    parser.add_argument("--model")
    parser.add_argument("--skills", default=str(ROOT / "terminal_bench_skills_init.json"))
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--skills-mode", default="task_only", choices=["task_only"])
    parser.add_argument("--enable-skills", action="store_true", help="Enable subagent skill retrieval for the rerun")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--docker-timeout", type=int, default=600)
    parser.add_argument("--command-timeout", type=int, default=300)
    parser.add_argument(
        "--resume-from-commands-checkpoint",
        dest="resume_from_commands_checkpoint",
        action="store_true",
        default=True,
        help="Replay commands before the target delegate to approximate checkpoint resume.",
    )
    parser.add_argument(
        "--no-resume-from-commands-checkpoint",
        dest="resume_from_commands_checkpoint",
        action="store_false",
        help="Start from a fresh container and rerun only the selected delegate.",
    )
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument("--stop-on-replay-error", action="store_true")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
