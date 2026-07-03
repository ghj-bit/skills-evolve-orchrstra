"""RubricWorkflow benchmark adapter for Uno planning evaluation."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import DEFAULT_API_BASE, MODEL_POOL, compute_cost
from ..executors.workspace_executor import WorkspaceExecutor
from .base import BaseBenchmark, Task, VerifyResult


DEFAULT_PROBLEM_ID = "高教社杯2020B"
FIXTURE_DIR = Path(__file__).resolve().parent / "rubric_workflow_fixtures"
PROBLEM_FIXTURE_IDS = {
    DEFAULT_PROBLEM_ID: "gaojiaoshebei_2020B",
}
HISTORICAL_ROOTS = [
    "/root/data/moved/rubric_workflow_llm_modules",
    r"D:\vscode_project\rubric_workflow_llm_modules",
]

PLANNER_SYSTEM_PROMPT = """You are a planning controller for a math-modeling benchmark.

You must solve the problem by delegating concrete implementation/report-writing
subtasks to worker agents. The workspace persists across delegate_task calls.
Use submit only after output/results/solution_report.md exists and the report is
ready for final rubric scoring. Do not ask for rubric scores during intermediate
steps."""


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _planner_tools(worker_pool: List[str]) -> List[Dict[str, Any]]:
    worker_schema: Dict[str, Any] = {"type": "string"}
    if worker_pool:
        worker_schema["enum"] = list(worker_pool)
    return [
        {
            "type": "function",
            "function": {
                "name": "delegate_task",
                "description": "Delegate one concrete workspace task to a worker agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "worker_model": worker_schema,
                        "instruction": {
                            "type": "string",
                            "description": "Specific task to execute in the shared workspace.",
                        },
                    },
                    "required": ["worker_model", "instruction"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit",
                "description": "Submit the final solution report for scoring.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "report_path": {
                            "type": "string",
                            "description": "Path to output/results/solution_report.md.",
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["report_path"],
                },
            },
        },
    ]


def _usage_from_response(resp: Dict[str, Any], default_model: str = "") -> Dict[str, Any]:
    prompt_tokens = int(resp.get("prompt_tokens", 0) or 0)
    completion_tokens = int(resp.get("completion_tokens", 0) or 0)
    model = resp.get("model") or default_model
    cost = compute_cost(model, completion_tokens, prompt_tokens) if model else 0.0
    return {
        "model": model,
        "cost": cost,
        "tokens": prompt_tokens + completion_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def _sum_usage(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "cost": sum(float(i.get("cost", 0) or 0) for i in items),
        "tokens": sum(int(i.get("tokens", 0) or 0) for i in items),
        "prompt_tokens": sum(int(i.get("prompt_tokens", 0) or 0) for i in items),
        "completion_tokens": sum(int(i.get("completion_tokens", 0) or 0) for i in items),
    }


class RubricWorkflow(BaseBenchmark):
    scoring_mode = "rubric_workflow_aggregated"
    score_name = "Aggregated rubric score"

    @property
    def name(self) -> str:
        return "RubricWorkflow"

    def __init__(self, root: str | None = None):
        self.fixture_root = Path(root) if root else FIXTURE_DIR
        self.problem_ids = _split_env("RUBRIC_WORKFLOW_TASK_IDS") or [DEFAULT_PROBLEM_ID]
        self.max_attempts = int(os.environ.get("RUBRIC_WORKFLOW_MAX_ATTEMPTS", "8"))
        self.subagent_max_steps = int(os.environ.get("RUBRIC_WORKFLOW_SUBAGENT_MAX_STEPS", "40"))
        self.cmd_timeout = int(os.environ.get("RUBRIC_WORKFLOW_CMD_TIMEOUT", "7200"))

    def load(self, max_tasks: Optional[int] = None) -> List[Task]:
        tasks: List[Task] = []
        for problem_id in self.problem_ids:
            fixture_dir = self._fixture_dir(problem_id)
            prompt_template = self._read_prompt(problem_id)
            problem_dir = fixture_dir / "problem" / problem_id
            submission_schema = fixture_dir / "submission_schema.md"
            scoring_dir = fixture_dir / "scoring"
            for required in (problem_dir, submission_schema, scoring_dir):
                if not required.exists():
                    raise FileNotFoundError(f"RubricWorkflow fixture input missing: {required}")
            task = Task(
                task_id=problem_id,
                raw={
                    "fixture_dir": str(fixture_dir),
                    "problem_id": problem_id,
                    "prompt_template": prompt_template,
                    "problem_dir": str(problem_dir),
                    "submission_schema": str(submission_schema),
                    "scoring_dir": str(scoring_dir),
                },
                question=prompt_template,
                context={"problem_id": problem_id, "problem_dir": str(problem_dir)},
            )
            tasks.append(task)
            if max_tasks and len(tasks) >= max_tasks:
                break
        return tasks

    def extract_answer(self, router_output: str, task: Task) -> str:
        return router_output

    def verify(self, task: Task, answer: str, logs_dir: str = None) -> VerifyResult:
        score_info = self._score_report(task, Path(logs_dir or ".") / task.task_id / "workspace" / "output")
        return VerifyResult(
            task_id=task.task_id,
            reward=score_info.get("reward", 0.0),
            error=score_info.get("error"),
            log=json.dumps(score_info, ensure_ascii=False)[:3000],
        )

    def interactive_verify(self, task: Task, router, logs_dir=None) -> VerifyResult:
        return _run_async(self._run_interactive_async(task, router, Path(logs_dir or ".")))

    async def _run_interactive_async(self, task: Task, router, logs_root: Path) -> VerifyResult:
        base = logs_root / task.task_id
        workspace = base / "workspace" / "output"
        planner_logs = base / "planner"
        agent_logs = base / "agent"
        verifier_logs = base / "verifier"
        for path in (workspace, planner_logs, agent_logs, verifier_logs):
            path.mkdir(parents=True, exist_ok=True)

        problem_id = task.raw["problem_id"]
        problem_dir = Path(task.raw["problem_dir"])
        submission_schema = Path(task.raw["submission_schema"])
        prompt = _render_prompt(task.raw["prompt_template"], problem_id, problem_dir, workspace, submission_schema)
        (base / "rendered_prompt.md").write_text(prompt, encoding="utf-8")

        executor = WorkspaceExecutor(
            task_id=problem_id,
            workspace_dir=workspace,
            problem_dir=problem_dir,
            submission_schema_path=submission_schema,
            verifier_logs_dir=verifier_logs,
            agent_logs_dir=agent_logs,
            timeout=self.cmd_timeout,
        )
        await executor.start_container()

        worker_pool = _split_env("RUBRIC_WORKFLOW_WORKER_MODELS") or list(MODEL_POOL)
        tools = _planner_tools(worker_pool)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        planner_usages: List[Dict[str, Any]] = []
        subagent_usages: List[Dict[str, Any]] = []
        routed_models: List[str] = []
        trajectory: List[Dict[str, Any]] = []
        submit_called = False
        submitted_report = ""
        last_error: Optional[str] = None
        started_at = _now_iso()
        start_perf = time.perf_counter()

        try:
            for attempt in range(self.max_attempts):
                resp = router.chat_completions(messages, tools=tools)
                planner_usage = _usage_from_response(
                    resp,
                    getattr(router, "planner_model", getattr(router, "model_name", "")),
                )
                planner_usages.append(planner_usage)
                content = resp.get("content") or ""
                tool_calls = resp.get("tool_calls") or []
                assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content or None}
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": t["id"],
                            "type": "function",
                            "function": {
                                "name": t["name"],
                                "arguments": json.dumps(t.get("arguments", {}), ensure_ascii=False),
                            },
                        }
                        for t in tool_calls
                    ]
                messages.append(assistant_msg)
                step_record = {
                    "attempt": attempt + 1,
                    "planner_content": content,
                    "planner_usage": planner_usage,
                    "tool_calls": tool_calls,
                }
                trajectory.append(step_record)

                if not tool_calls:
                    last_error = "planner returned no tool call"
                    break

                for tc in tool_calls:
                    name = tc.get("name")
                    args = tc.get("arguments", {}) or {}
                    tc_id = tc.get("id", f"call_{attempt + 1}")

                    if name == "submit":
                        submitted_report = str(args.get("report_path") or workspace / "results" / "solution_report.md")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": _submission_observation(workspace, submitted_report),
                        })
                        step_record["submit"] = {"report_path": submitted_report, "reason": args.get("reason", "")}
                        submit_called = True
                        break

                    if name != "delegate_task":
                        msg = f"Unknown tool '{name}' ignored."
                        messages.append({"role": "tool", "tool_call_id": tc_id, "content": msg})
                        continue

                    worker_model = args.get("worker_model") or (worker_pool[0] if worker_pool else "")
                    if worker_pool and worker_model not in worker_pool:
                        worker_model = worker_pool[0]
                    instruction = args.get("instruction") or ""
                    if not instruction.strip():
                        msg = "Empty instruction; delegate_task skipped."
                        messages.append({"role": "tool", "tool_call_id": tc_id, "content": msg})
                        continue

                    from uno_orchestor.agents.subagent import SubAgent

                    subagent = SubAgent(
                        api_base=getattr(router, "sub_model_api_base", DEFAULT_API_BASE),
                        api_key=getattr(router, "sub_model_api_key", os.environ.get("API_KEY", "EMPTY")),
                        max_steps=self.subagent_max_steps,
                        cmd_timeout=self.cmd_timeout,
                    )
                    sub_result = await subagent.run(
                        model=worker_model,
                        task_instruction=_worker_instruction(instruction, workspace, problem_dir, submission_schema),
                        original_question=prompt,
                        executor=executor,
                        agent_logs_dir=agent_logs,
                        tool_call_id=tc_id,
                    )
                    routed_models.append(worker_model)
                    usage = {
                        "model": worker_model,
                        "cost": float(sub_result.get("cost", 0) or 0),
                        "tokens": int(sub_result.get("tokens", 0) or 0),
                        "prompt_tokens": int(sub_result.get("prompt_tokens", 0) or 0),
                        "completion_tokens": int(sub_result.get("completion_tokens", 0) or 0),
                    }
                    subagent_usages.append(usage)
                    obs = _format_worker_observation(sub_result, workspace)
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": obs})
                    step_record.setdefault("delegates", []).append({
                        "worker_model": worker_model,
                        "instruction": instruction,
                        "usage": usage,
                        "observation": obs[:2000],
                    })

                if submit_called:
                    break
        except Exception as exc:
            last_error = f"interactive rubric workflow failed: {exc}"
        finally:
            await executor.cleanup()

        score_info = self._score_report(task, workspace) if submit_called else {
            "reward": 0.0,
            "error": last_error or "planner did not submit",
        }
        planner_totals = _sum_usage(planner_usages)
        subagent_totals = _sum_usage(subagent_usages)
        total_usage = _sum_usage([planner_totals, subagent_totals])
        trajectory_payload = {
            "task_id": task.task_id,
            "benchmark": self.name,
            "started_at": started_at,
            "ended_at": _now_iso(),
            "elapsed_seconds": round(time.perf_counter() - start_perf, 3),
            "workspace": str(workspace),
            "submit_called": submit_called,
            "submitted_report": submitted_report,
            "last_error": last_error,
            "reward": score_info.get("reward", 0.0),
            "score_info": score_info,
            "route_count": len(routed_models),
            "routed_models": routed_models,
            "routed_skills": ["delegate_task"] * len(routed_models),
            "routed_backends": ["workspace"] * len(routed_models),
            "planner_usage": planner_totals,
            "subagent_usage": subagent_totals,
            **total_usage,
            "trajectory": trajectory,
        }
        _write_json(base / "trajectory.json", trajectory_payload)
        return VerifyResult(
            task.task_id,
            float(score_info.get("reward", 0.0) or 0.0),
            error=score_info.get("error") or last_error,
            log=json.dumps(score_info, ensure_ascii=False)[:3000],
        )

    def _fixture_dir(self, problem_id: str) -> Path:
        fixture_id = PROBLEM_FIXTURE_IDS.get(problem_id)
        if not fixture_id:
            raise FileNotFoundError(f"RubricWorkflow fixture is not migrated for {problem_id}")
        fixture_dir = self.fixture_root / fixture_id
        if not fixture_dir.exists():
            raise FileNotFoundError(f"RubricWorkflow fixture directory not found: {fixture_dir}")
        return fixture_dir

    def _read_prompt(self, problem_id: str) -> str:
        prompt_path = self._fixture_dir(problem_id) / "prompt.md"
        if not prompt_path.exists():
            raise FileNotFoundError(f"RubricWorkflow prompt not found in fixture: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8-sig")

    def _score_report(self, task: Task, workspace: Path) -> Dict[str, Any]:
        report = workspace / "results" / "solution_report.md"
        if not report.exists():
            return {"reward": 0.0, "error": f"missing report: {report}"}

        problem_id = task.raw["problem_id"]
        output_dir = workspace.parent.parent / "scores"
        output_dir.mkdir(parents=True, exist_ok=True)
        scoring_dir = Path(task.raw["scoring_dir"])
        rubric = scoring_dir / "aggregated_refined_rubric.json"
        rubric_mapping = scoring_dir / "aggregated_refined_rubric_mapping.json"
        problem_analysis = scoring_dir / "problem_analysis.md"
        for required in (rubric, rubric_mapping, problem_analysis):
            if not required.exists():
                return {
                    "reward": 0.0,
                    "error": f"missing scoring input: {required}",
                    "report_path": str(report),
                }

        cmd = [
            sys.executable,
            "-m",
            "rubric_workflow.workflow.evaluate_aggregated_report",
            "--report",
            str(report),
            "--problem-path",
            str(task.raw["problem_dir"]),
            "--problem-analysis",
            str(problem_analysis),
            "--rubric",
            str(rubric),
            "--rubric-mapping",
            str(rubric_mapping),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "uno_planner",
            "--timeout",
            os.environ.get("RUBRIC_WORKFLOW_SCORE_TIMEOUT", "600"),
            "--temperature",
            "0.0",
            "--max-retries",
            "2",
        ]
        env = os.environ.copy()
        package_root = os.environ.get("RUBRIC_WORKFLOW_PACKAGE_ROOT", "").strip()
        if package_root:
            env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["LANG"] = env.get("LANG") or "C.UTF-8"
        env["LC_ALL"] = env.get("LC_ALL") or "C.UTF-8"
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(Path.cwd()),
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=int(os.environ.get("RUBRIC_WORKFLOW_SCORE_TIMEOUT", "600")) + 60,
            )
        except Exception as exc:
            return {"reward": 0.0, "error": f"scoring failed: {exc}", "report_path": str(report)}

        score_json = output_dir / "uno_planner_score.json"
        if not score_json.exists():
            return {
                "reward": 0.0,
                "error": f"score json not produced; exit={proc.returncode}",
                "scorer_output": proc.stdout[-3000:],
                "report_path": str(report),
            }
        data = json.loads(score_json.read_text(encoding="utf-8-sig"))
        raw_score = float(data.get("raw_aggregated_score", 0) or 0)
        full_score = float(data.get("raw_aggregated_full_score", 100) or 100)
        normalized = raw_score / full_score if full_score > 0 else 0.0
        return {
            "reward": normalized,
            "raw_score": raw_score,
            "full_score": full_score,
            "score_json": str(score_json),
            "report_path": str(report),
            "scorer_exit_code": proc.returncode,
        }


def _render_prompt(template: str, problem_id: str, problem_dir: Path, output_dir: Path, schema: Path) -> str:
    rendered = template.replace("{{PROBLEM_DIR}}", str(problem_dir))
    rendered = rendered.replace("{{OUTPUT_DIR}}", str(output_dir))
    rendered = rendered.replace("{{SUBMISSION_SCHEMA_PATH}}", str(schema))

    for historical_root in HISTORICAL_ROOTS:
        rendered = rendered.replace(f"{historical_root}/国赛题目/{problem_id}", str(problem_dir))
        rendered = rendered.replace(f"{historical_root}\\国赛题目\\{problem_id}", str(problem_dir))
        rendered = rendered.replace(f"{historical_root}/submission_schema.md", str(schema))
        rendered = rendered.replace(f"{historical_root}\\submission_schema.md", str(schema))
        baseline = f"{historical_root}/experiments/baseline-openclaw-dsv4pro/{problem_id}/output"
        rendered = rendered.replace(baseline, str(output_dir))
        baseline = f"{historical_root}\\experiments\\baseline-openclaw-dsv4pro\\{problem_id}\\output"
        rendered = rendered.replace(baseline, str(output_dir))
    rendered += (
        "\n\n## Uno evaluation path override\n"
        f"- The only output root for this run is: {output_dir}\n"
        f"- The final report must be written to: {output_dir / 'results' / 'solution_report.md'}\n"
        "- Do not write to any external historical experiment directory.\n"
    )
    return rendered


def _worker_instruction(instruction: str, workspace: Path, problem_dir: Path, schema: Path) -> str:
    return (
        f"{instruction}\n\n"
        f"Workspace/output directory: {workspace}\n"
        f"Problem directory: {problem_dir}\n"
        f"Submission schema: {schema}\n"
        "Write code under code/, intermediate outputs under results/, logs under logs/. "
        "The final report must be results/solution_report.md."
    )


def _format_worker_observation(result: Dict[str, Any], workspace: Path) -> str:
    from uno_orchestor.agents.subagent import SubAgent

    snapshot = _workspace_snapshot(workspace)
    return "\n".join([
        SubAgent.format_result_for_planner(result),
        "",
        "Workspace snapshot:",
        json.dumps(snapshot, ensure_ascii=False, indent=2),
    ])


def _workspace_snapshot(workspace: Path) -> Dict[str, Any]:
    files = []
    for path in sorted(workspace.rglob("*")):
        if path.is_file():
            rel = path.relative_to(workspace).as_posix()
            files.append({"path": rel, "bytes": path.stat().st_size})
            if len(files) >= 80:
                break
    report = workspace / "results" / "solution_report.md"
    return {
        "workspace": str(workspace),
        "solution_report_exists": report.exists(),
        "solution_report_bytes": report.stat().st_size if report.exists() else 0,
        "files": files,
    }


def _submission_observation(workspace: Path, report_path: str) -> str:
    report = Path(report_path)
    if not report.is_absolute():
        report = workspace / report
    return json.dumps({
        "submit_received": True,
        "report_path": str(report),
        "report_exists": report.exists(),
        "workspace_snapshot": _workspace_snapshot(workspace),
    }, ensure_ascii=False)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _split_env(name: str) -> List[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]


def _run_async(awaitable):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    return loop.run_until_complete(awaitable)
