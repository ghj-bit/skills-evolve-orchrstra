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

PLANNER_SYSTEM_PROMPT = """\
You are the Planner for a math-modeling benchmark. You do NOT execute commands
or write files directly. Instead, you delegate work to a worker sub-agent that
runs inside a persistent output workspace.

## Tools
- delegate_task(worker_model, instruction)
    Delegate a concrete sub-task to the given worker model. The worker runs
    commands in the workspace (state persists across delegations), then returns
    a structured report: status (done/partial/error), what it did,
    any issues.
- submit(report_path, reason)
    Declare the whole task complete. The harness runs final rubric scoring on
    the submitted report.

## Rules
- Each delegate_task consumes one attempt. The workspace persists, so later
  delegations see the previous worker's changes.
- Start by delegating a concrete subtask, not by describing the whole task.
- After the worker returns `status=done`, inspect its `completed` list and
  `issues`. If it really addressed every requirement and
  results/solution_report.md exists, call `submit`; if not, delegate another
  subtask with explicit instructions for what is missing.
- If the worker returns `status=partial` or `status=error`, the next
  delegate_task must directly address the unresolved `issues`. Do not restart
  from the original task unless the observation says required files are missing.
- Treat the latest tool observation as the source of truth. Before each
  delegate_task, compare the requested instruction with the latest
  `completed`, `issues`, and workspace snapshot.
- Do not repeat completed discovery work. If the latest observation says the
  problem files, submission schema, weather/data files, or workspace listing
  were already read, the next delegation must move to the next unresolved
  modeling, computation, validation, spreadsheet/report-writing, or submission
  step.
- A worker `status=done` only means the delegated subtask finished. If
  `issues` is non-empty or results/solution_report.md is missing, continue from
  the existing workspace and resolve those concrete issues.
- Each new delegate_task must name what prior work it is reusing and what
  exact missing artifact it will produce. Do not ask the worker to "read all
  files" again unless the latest observation explicitly says required files are
  missing or unreadable.
- Prefer small, verifiable delegations over one monolithic "do everything".""" 


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _budget_note(attempt_idx: int, max_attempts: int) -> str:
    remaining = max_attempts - attempt_idx
    if remaining <= 2:
        return (
            f"CRITICAL: Only {remaining} delegation attempt(s) left. "
            "Use the latest observation to finish missing work or submit if the report is ready."
        )
    if attempt_idx > 1:
        return (
            f"Budget: {remaining}/{max_attempts} attempts remaining. "
            "Do not repeat completed work; delegate only the unresolved issues from the latest observation."
        )
    return f"Budget: {remaining}/{max_attempts} attempts remaining."


def _planner_tools(worker_pool: List[str]) -> List[Dict[str, Any]]:
    worker_schema: Dict[str, Any] = {"type": "string"}
    if worker_pool:
        worker_schema["enum"] = list(worker_pool)
    return [
        {
            "type": "function",
            "function": {
                "name": "delegate_task",
                "description": (
                    "Delegate a concrete sub-task to a worker model that runs "
                    "commands in the shared output workspace."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "worker_model": worker_schema,
                        "instruction": {
                            "type": "string",
                            "description": (
                                "Self-contained natural-language instructions for the worker. "
                                "Include the specific missing work from the latest observation, "
                                "and tell the worker to reuse existing workspace files."
                            ),
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
                "description": (
                    "Declare the math-modeling solution complete. Runs final "
                    "rubric scoring on the submitted report."
                ),
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
        self._live_trace_task_selected = False

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
        score_info = self._score_report(
            task,
            (Path(logs_dir or ".") / task.task_id / "workspace" / "output").resolve(),
        )
        return VerifyResult(
            task_id=task.task_id,
            reward=score_info.get("reward", 0.0),
            error=score_info.get("error"),
            log=json.dumps(score_info, ensure_ascii=False)[:3000],
        )

    def interactive_verify(self, task: Task, router, logs_dir=None) -> VerifyResult:
        return _run_async(self._run_interactive_async(task, router, Path(logs_dir or ".")))

    async def _run_interactive_async(self, task: Task, router, logs_root: Path) -> VerifyResult:
        base = (logs_root / task.task_id).resolve()
        workspace = (base / "workspace" / "output").resolve()
        planner_logs = (base / "planner").resolve()
        agent_logs = (base / "agent").resolve()
        verifier_logs = (base / "verifier").resolve()
        for path in (workspace, planner_logs, agent_logs, verifier_logs):
            path.mkdir(parents=True, exist_ok=True)

        problem_id = task.raw["problem_id"]
        problem_dir = Path(task.raw["problem_dir"]).resolve()
        submission_schema = Path(task.raw["submission_schema"]).resolve()
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
        live_trace_enabled = not self._live_trace_task_selected
        self._live_trace_task_selected = True
        live_trace_path = base / "live_trajectory.json"
        live_trace: Optional[Dict[str, Any]] = None
        if live_trace_enabled:
            live_trace = {
                "task_id": task.task_id,
                "benchmark": self.name,
                "started_at": started_at,
                "updated_at": started_at,
                "workspace": str(workspace),
                "rendered_prompt_path": str(base / "rendered_prompt.md"),
                "events": [
                    {
                        "type": "task_start",
                        "timestamp": started_at,
                        "planner_initial_messages": _json_clone(messages),
                        "tools": _json_clone(tools),
                    }
                ],
            }
            _write_live_trace(live_trace_path, live_trace)

        try:
            for attempt in range(self.max_attempts):
                live_messages = messages + [
                    {"role": "system", "content": _budget_note(attempt + 1, self.max_attempts)}
                ]
                planner_event: Optional[Dict[str, Any]] = None
                if live_trace is not None:
                    planner_event = {
                        "type": "planner_call",
                        "attempt": attempt + 1,
                        "started_at": _now_iso(),
                        "input_messages": _json_clone(live_messages),
                        "tools": _json_clone(tools),
                    }
                    live_trace["events"].append(planner_event)
                    _write_live_trace(live_trace_path, live_trace)

                resp = router.chat_completions(live_messages, tools=tools)
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
                if planner_event is not None:
                    planner_event.update({
                        "ended_at": _now_iso(),
                        "output": {
                            "content": content,
                            "tool_calls": _json_clone(tool_calls),
                            "usage": planner_usage,
                            "model": resp.get("model"),
                        },
                    })
                    _write_live_trace(live_trace_path, live_trace)

                if not tool_calls:
                    last_error = "planner returned no tool call"
                    break

                for tc in tool_calls:
                    name = tc.get("name")
                    args = tc.get("arguments", {}) or {}
                    tc_id = tc.get("id", f"call_{attempt + 1}")

                    if name == "submit":
                        submitted_report = str(args.get("report_path") or workspace / "results" / "solution_report.md")
                        submit_observation = _submission_observation(workspace, submitted_report)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": submit_observation,
                        })
                        step_record["submit"] = {"report_path": submitted_report, "reason": args.get("reason", "")}
                        if live_trace is not None:
                            live_trace["events"].append({
                                "type": "submit",
                                "timestamp": _now_iso(),
                                "tool_call_id": tc_id,
                                "arguments": _json_clone(args),
                                "observation": submit_observation,
                            })
                            _write_live_trace(live_trace_path, live_trace)
                        submit_called = True
                        break

                    if name != "delegate_task":
                        msg = f"Unknown tool '{name}' ignored."
                        messages.append({"role": "tool", "tool_call_id": tc_id, "content": msg})
                        if live_trace is not None:
                            live_trace["events"].append({
                                "type": "unknown_tool",
                                "timestamp": _now_iso(),
                                "tool_call_id": tc_id,
                                "name": name,
                                "arguments": _json_clone(args),
                                "observation": msg,
                            })
                            _write_live_trace(live_trace_path, live_trace)
                        continue

                    worker_model = args.get("worker_model") or (worker_pool[0] if worker_pool else "")
                    if worker_pool and worker_model not in worker_pool:
                        worker_model = worker_pool[0]
                    instruction = args.get("instruction") or ""
                    if not instruction.strip():
                        msg = "Empty instruction; delegate_task skipped."
                        messages.append({"role": "tool", "tool_call_id": tc_id, "content": msg})
                        if live_trace is not None:
                            live_trace["events"].append({
                                "type": "delegate_skipped",
                                "timestamp": _now_iso(),
                                "tool_call_id": tc_id,
                                "worker_model": worker_model,
                                "arguments": _json_clone(args),
                                "observation": msg,
                            })
                            _write_live_trace(live_trace_path, live_trace)
                        continue

                    from uno_orchestor.agents.subagent import SubAgent

                    subagent = SubAgent(
                        api_base=getattr(router, "sub_model_api_base", DEFAULT_API_BASE),
                        api_key=getattr(router, "sub_model_api_key", os.environ.get("API_KEY", "EMPTY")),
                        max_steps=self.subagent_max_steps,
                        cmd_timeout=self.cmd_timeout,
                    )
                    worker_prompt = _worker_instruction(instruction, workspace, problem_dir, submission_schema)
                    delegate_event: Optional[Dict[str, Any]] = None
                    if live_trace is not None:
                        delegate_event = {
                            "type": "delegate_task",
                            "tool_call_id": tc_id,
                            "started_at": _now_iso(),
                            "worker_model": worker_model,
                            "arguments": _json_clone(args),
                            "worker_input": {
                                "task_instruction": worker_prompt,
                                "original_question": prompt,
                            },
                        }
                        live_trace["events"].append(delegate_event)
                        _write_live_trace(live_trace_path, live_trace)

                    previous_transcript_env = os.environ.get("SUBAGENT_RETURN_TRANSCRIPT")
                    if live_trace is not None:
                        os.environ["SUBAGENT_RETURN_TRANSCRIPT"] = "1"
                    try:
                        sub_result = await subagent.run(
                            model=worker_model,
                            task_instruction=worker_prompt,
                            original_question=prompt,
                            executor=executor,
                            agent_logs_dir=agent_logs,
                            tool_call_id=tc_id,
                        )
                    finally:
                        if live_trace is not None:
                            if previous_transcript_env is None:
                                os.environ.pop("SUBAGENT_RETURN_TRANSCRIPT", None)
                            else:
                                os.environ["SUBAGENT_RETURN_TRANSCRIPT"] = previous_transcript_env
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
                    if delegate_event is not None:
                        delegate_event.update({
                            "ended_at": _now_iso(),
                            "worker_output": _json_clone(sub_result),
                            "worker_llm_transcript": _json_clone(sub_result.get("llm_transcript", [])),
                            "planner_observation": obs,
                            "usage": usage,
                        })
                        _write_live_trace(live_trace_path, live_trace)
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
            if live_trace is not None:
                live_trace["events"].append({
                    "type": "error",
                    "timestamp": _now_iso(),
                    "error": last_error,
                    "exception_type": type(exc).__name__,
                })
                _write_live_trace(live_trace_path, live_trace)
        finally:
            await executor.cleanup()

        score_info = self._score_report(task, workspace) if submit_called else {
            "reward": 0.0,
            "error": last_error or "planner did not submit",
        }
        planner_totals = _sum_usage(planner_usages)
        subagent_totals = _sum_usage(subagent_usages)
        total_usage = _sum_usage([planner_totals, subagent_totals])
        if live_trace is not None:
            live_trace.update({
                "ended_at": _now_iso(),
                "elapsed_seconds": round(time.perf_counter() - start_perf, 3),
                "submit_called": submit_called,
                "submitted_report": submitted_report,
                "last_error": last_error,
                "reward": score_info.get("reward", 0.0),
                "score_info": score_info,
                "route_count": len(routed_models),
                "routed_models": routed_models,
                "planner_usage": planner_totals,
                "subagent_usage": subagent_totals,
                **total_usage,
            })
            _write_live_trace(live_trace_path, live_trace)
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
    problem_dir = problem_dir.resolve()
    output_dir = output_dir.resolve()
    schema = schema.resolve()
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


def _bash_path(path: Path) -> str:
    resolved = path.resolve()
    text = str(resolved)
    if os.name == "nt" and len(text) >= 2 and text[1] == ":":
        drive = text[0].lower()
        rest = text[2:].replace("\\", "/").lstrip("/")
        return f"/{drive}/{rest}"
    return resolved.as_posix()


def _worker_instruction(instruction: str, workspace: Path, problem_dir: Path, schema: Path) -> str:
    workspace = workspace.resolve()
    problem_dir = problem_dir.resolve()
    schema = schema.resolve()
    return (
        f"{instruction}\n\n"
        "Path rules:\n"
        f"- Workspace/output directory (absolute Windows path): {workspace}\n"
        f"- Workspace/output directory (Git Bash path): {_bash_path(workspace)}\n"
        f"- Problem directory (absolute Windows path): {problem_dir}\n"
        f"- Submission schema (absolute Windows path): {schema}\n"
        "Use the workspace path above exactly; do not prepend the current working directory to it. "
        "The command working directory is already the workspace/output directory. "
        "Write code under code/ or <workspace>/code, intermediate outputs under results/ or <workspace>/results, "
        "and logs under logs/ or <workspace>/logs. "
        "The final report must be <workspace>/results/solution_report.md."
    )


def _format_worker_observation(result: Dict[str, Any], workspace: Path) -> str:
    from uno_orchestor.agents.subagent import SubAgent

    snapshot = _workspace_snapshot(workspace)
    status = str(result.get("status") or "unknown")
    issues = result.get("issues") or []
    completed = result.get("completed") or []
    if status == "done" and snapshot.get("solution_report_exists"):
        next_action = "The final report exists. If it is complete, call submit with results/solution_report.md."
    elif issues:
        next_action = (
            "Continue from the existing workspace. Do not repeat completed work. "
            "The next delegate_task should directly resolve these unresolved issues: "
            + json.dumps(issues, ensure_ascii=False)
        )
    elif completed:
        next_action = (
            "Continue from the existing workspace. Do not repeat completed work. "
            "Delegate the next missing modeling, computation, validation, or report-writing step."
        )
    else:
        next_action = (
            "Use the workspace snapshot and worker message to choose the next concrete step. "
            "Do not restart unless required files are missing."
        )
    completed_guard = (
        "Do not ask for already completed work again. Reuse existing files and "
        "workspace state. If discovery/reading is listed as completed, the next "
        "task must be modeling, computation, validation, report writing, or a "
        "specific fix for an issue."
    )
    return "\n".join([
        "Planner next-action guidance:",
        next_action,
        "",
        "Completed work that must not be repeated:",
        json.dumps(completed, ensure_ascii=False, indent=2),
        "",
        "Unresolved issues to address next:",
        json.dumps(issues, ensure_ascii=False, indent=2),
        "",
        "Anti-repeat rule:",
        completed_guard,
        "",
        "Workspace snapshot:",
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        "",
        "Raw worker summary:",
        SubAgent.format_result_for_planner(result),
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


def _write_live_trace(path: Path, payload: Dict[str, Any]) -> None:
    payload["updated_at"] = _now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


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
