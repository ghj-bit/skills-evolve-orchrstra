#!/usr/bin/env python
"""Prepare a skill-augmented rerun request for a Terminal-Bench delegate step.

This script does not resume Docker state by itself. It extracts the original
delegate instruction from a trajectory, retrieves top-k skills, and writes a
rerun_request.json that can be used by a replay/checkpoint runner.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SKILL_RETRIEVER_PATH = ROOT / "uno_orchestor" / "agents" / "skill_retriever.py"
_spec = importlib.util.spec_from_file_location("skill_retriever", _SKILL_RETRIEVER_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load skill retriever: {_SKILL_RETRIEVER_PATH}")
_skill_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_skill_mod)
SkillRetriever = _skill_mod.SkillRetriever


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return ""


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
                "sub_result": (item.get("delegate") or {}).get("sub_result"),
            }


def _select_delegate(delegates: list[dict], tool_call_id: str | None, worker_model: str | None) -> dict:
    if tool_call_id:
        for d in delegates:
            if d.get("tool_call_id") == tool_call_id:
                return d
        raise SystemExit(f"tool_call_id not found: {tool_call_id}")
    if worker_model:
        for d in delegates:
            if d.get("worker_model") == worker_model:
                return d
        raise SystemExit(f"worker_model not found: {worker_model}")
    for d in delegates:
        if "qwen" in d.get("worker_model", "").lower():
            return d
    raise SystemExit("No Qwen delegate found. Pass --tool-call-id or --worker-model.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True, help="Path to trajectory.json")
    parser.add_argument("--tool-call-id", help="Exact delegate tool call id to rerun")
    parser.add_argument("--worker-model", help="Fallback selector if tool call id is omitted")
    parser.add_argument("--skills", default=str(ROOT / "terminal_bench_skills_init.json"))
    parser.add_argument("--top-k", type=int, default=2, help="Maximum skills to inject")
    parser.add_argument("--output-dir", help="Directory for rerun_request.json")
    parser.add_argument("--enable-skills", action="store_true", help="Retrieve and inject skills into the rerun request")
    args = parser.parse_args()

    trajectory_path = Path(args.trajectory)
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8-sig"))
    delegates = list(_iter_delegates(trajectory))
    selected = _select_delegate(delegates, args.tool_call_id, args.worker_model)

    task_id = trajectory.get("task_id") or trajectory_path.parent.name
    task_instruction = _read_text(ROOT / "data" / "terminal-bench" / "tasks" / task_id / "instruction.md")
    query = "\n\n".join([
        f"Task: {task_id}",
        task_instruction,
        selected.get("instruction", ""),
        json.dumps(selected.get("sub_result") or {}, ensure_ascii=False),
    ])

    skills = []
    augmented_instruction = selected.get("instruction", "")
    if args.enable_skills:
        retriever = SkillRetriever.from_path(args.skills)
        skills = retriever.search(query, top_k=max(1, args.top_k), level="subtask")
        skills_block = retriever.render(skills)
        augmented_instruction = (
            f"{selected.get('instruction', '')}\n\n"
            "== Retrieved Skills ==\n"
            "Use these skills as hard execution constraints. Do not return status=done "
            "unless the skill checks pass.\n"
            f"{skills_block}\n"
        )

    output_dir = Path(args.output_dir) if args.output_dir else trajectory_path.parent / "skill_rerun"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": task_id,
        "source_trajectory": str(trajectory_path),
        "selected_delegate": selected,
        "skills_path": str(Path(args.skills)),
        "top_k": max(1, args.top_k),
        "skills_enabled": bool(args.enable_skills),
        "retrieved_skills": skills,
        "augmented_instruction": augmented_instruction,
        "env": {
            "SUBAGENT_ENABLE_SKILLS": "1" if args.enable_skills else "0",
            "SUBAGENT_SKILLS_TOP_K": str(max(1, args.top_k)),
            "SUBAGENT_SKILLS_MODE": "task_only",
            "TERMINAL_BENCH_SKILLS_PATH": str(Path(args.skills).resolve()),
        },
        "notes": [
            "This file prepares the skill-augmented delegate rerun request.",
            "Exact breakpoint rerun requires either replaying prior commands from commands.log or starting from a Docker checkpoint captured before this delegate.",
        ],
    }
    out_file = output_dir / "rerun_request.json"
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {out_file}")
    print(f"Selected: attempt={selected.get('attempt')} tool_call_id={selected.get('tool_call_id')} model={selected.get('worker_model')}")
    if args.enable_skills:
        print("Retrieved skills:")
        for skill in skills:
            print(f"  {skill['rank']}. {skill['name']} score={skill['retrieval_score']:.4f}")
    else:
        print("Skills disabled. Pass --enable-skills to retrieve and inject skills.")


if __name__ == "__main__":
    main()
