"""SWE-bench Verified benchmark adapter.

Two complete scoring modes:
1. Interactive: router ↔ Docker multi-turn → Uno harness score
2. One-shot: router generates a patch → official-compatible harness score
"""
import re
import os
import json
import subprocess
import tempfile
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from .base import BaseBenchmark, Task, VerifyResult


# ── SWE-bench system prompt for interactive mode ──

SWEBENCH_PLANNER_SYSTEM_PROMPT = """\
You are the Planner for a SWE-Bench issue inside a persistent Docker container.
You do NOT execute shell commands directly. Instead, delegate concrete repair
work to a worker sub-agent that edits and tests the repository.

## Environment
- The repository is at /testbed and is checked out to the base commit.
- Worker commands run in /testbed with conda env "testbed" active.
- The container persists across delegations; later workers see earlier edits.
- The verifier will run the SWE-Bench tests in the same container after submit
  or after the delegation budget is exhausted.

## Tools
- delegate_task(worker_model, instruction)
    Delegate a concrete sub-task to the given worker model. The worker can run
    shell commands, inspect files, edit code, run targeted tests, and report
    completed work and issues.
- submit(reason)
    Declare the issue fixed. The harness runs the SWE-Bench test patch and
    grades FAIL_TO_PASS/PASS_TO_PASS.

## Rules
- Start by delegating a concrete debugging or repair subtask.
- Prefer small, verifiable delegations over one vague instruction.
- Make minimal, targeted code changes. Do not edit tests unless explicitly
  needed for local diagnosis; the verifier resets and applies the hidden tests.
- Submit only when the worker reports a plausible fix or no budget remains.
"""


SWEBENCH_FLAT_SYSTEM_PROMPT = """\
You are fixing a SWE-Bench issue inside a persistent Docker container. You
execute shell commands yourself, one per turn, observing each output before the
next command.

## Environment
- The repository is at /testbed and commands run there with conda env "testbed"
  active.
- Make minimal, targeted code changes.
- Run targeted tests or import checks when practical.

## Tools
- execute_command(command)
    Run one shell command in the container. State persists in the filesystem.
- submit(reason)
    Declare the issue fixed. The SWE-Bench verifier will run after this.
"""

SWEBENCH_INTERACTIVE_PROMPT = SWEBENCH_FLAT_SYSTEM_PROMPT


SWEBENCH_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "delegate_task",
            "description": "Delegate a concrete SWE-Bench repair sub-task to a worker model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "worker_model": {
                        "type": "string",
                        "description": "Worker model id. Use one of the allowed worker models.",
                    },
                    "instruction": {
                        "type": "string",
                        "description": (
                            "Self-contained repair instructions. Include the issue, relevant "
                            "repo facts, and what to inspect or change."
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
            "description": "Declare the SWE-Bench issue fixed and run the verifier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Brief completion reason."}
                },
                "required": ["reason"],
            },
        },
    },
]


SWEBENCH_FLAT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Run one shell command in the SWE-Bench container.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run."}
                },
                "required": ["command"],
            },
        },
    },
    SWEBENCH_TOOLS[1],
]


def _verbose_responses_enabled() -> bool:
    return os.environ.get("UNO_VERBOSE_RESPONSES", "").lower() in {"1", "true", "yes", "on"}


def _print_verbose_block(title: str, text: str, limit: int = 12000) -> None:
    if not _verbose_responses_enabled():
        return
    body = str(text)
    if len(body) > limit:
        body = body[:limit] + f"\n... [truncated {len(body) - limit} chars]"
    encoding = sys.stdout.encoding or "utf-8"
    safe_title = title.encode(encoding, errors="replace").decode(encoding, errors="replace")
    safe_body = body.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(f"\n===== {safe_title} =====")
    print(safe_body)
    print(f"===== /{safe_title} =====", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _planner_tools(worker_pool: Optional[List[str]]) -> List[Dict[str, Any]]:
    import copy

    tools = copy.deepcopy(SWEBENCH_TOOLS)
    if worker_pool:
        worker_schema = tools[0]["function"]["parameters"]["properties"]["worker_model"]
        worker_schema["enum"] = list(worker_pool)
        worker_schema["description"] = "Worker model id. Must be one of: " + ", ".join(worker_pool)
    return tools


def _usage_from_response(resp: Dict[str, Any], default_model: str = "") -> Dict[str, Any]:
    prompt_tokens = int(resp.get("prompt_tokens", 0) or 0)
    completion_tokens = int(resp.get("completion_tokens", 0) or 0)
    model = resp.get("model") or default_model
    cost = 0.0
    if model:
        try:
            from ..config import compute_cost

            cost = compute_cost(model, completion_tokens, prompt_tokens)
        except Exception:
            cost = 0.0
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


def _budget_note(attempt_idx: int, max_attempts: int) -> str:
    remaining = max_attempts - attempt_idx
    if remaining <= 2:
        return f"CRITICAL: Only {remaining} delegation attempt(s) left. Submit if the fix is plausible."
    if remaining <= 4:
        return f"Warning: {remaining} delegation attempts remaining. Keep the next subtask focused."
    return f"Budget: {remaining}/{max_attempts} delegation attempts remaining."


def _resolve_model_endpoint(
    model_id: str,
    default_api_base: str,
    default_api_key: str,
) -> tuple[str, str]:
    try:
        from configs import load_pools

        pools = load_pools()
        for model_cfg in pools.get("raw", {}).get("models", []):
            if model_cfg.get("id") == model_id:
                return (
                    model_cfg.get("api_base") or default_api_base,
                    model_cfg.get("api_key") or default_api_key,
                )
    except Exception:
        pass
    return default_api_base, default_api_key


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


class _SWEBenchCommandExecutor:
    """Adapter for SubAgent/flat tools: run every command in SWE-Bench context."""

    def __init__(self, executor):
        self.executor = executor

    async def execute_command(self, command: str, timeout: Optional[int] = None):
        wrapped = (
            "source /opt/miniconda3/bin/activate && "
            "conda activate testbed && "
            "cd /testbed && "
            f"{command}"
        )
        return await self.executor.execute_command(wrapped, timeout=timeout)

    def get_container_id(self):
        return self.executor.get_container_id()


class SWEBench(BaseBenchmark):
    def __init__(self, dataset="princeton-nlp/SWE-bench_Verified", split="test",
                 conda_env="swebench", eval_timeout=900, eval_workers=4,
                 max_steps=30, docker_timeout=1800, max_attempts=None,
                 subagent_max_steps=20, subagent_cmd_timeout=300):
        self.dataset = dataset
        self.split = split
        self.conda_env = conda_env
        self.eval_timeout = eval_timeout
        self.eval_workers = eval_workers
        self.max_steps = max_steps
        self.max_attempts = max_attempts or max_steps
        self.subagent_max_steps = subagent_max_steps
        self.subagent_cmd_timeout = subagent_cmd_timeout
        self.docker_timeout = docker_timeout

    @property
    def name(self):
        return "SWE-bench_Verified"

    def load(self, max_tasks=None) -> List[Task]:
        from datasets import load_dataset
        ds = load_dataset(self.dataset, split=self.split)
        if max_tasks:
            ds = ds.select(range(min(max_tasks, len(ds))))
        tasks = []
        for inst in ds:
            problem = inst["problem_statement"]
            # Parse FAIL_TO_PASS / PASS_TO_PASS (may be JSON strings)
            f2p = inst.get("FAIL_TO_PASS", [])
            p2p = inst.get("PASS_TO_PASS", [])
            if isinstance(f2p, str):
                try: f2p = json.loads(f2p)
                except: f2p = []
            if isinstance(p2p, str):
                try: p2p = json.loads(p2p)
                except: p2p = []
            tasks.append(Task(
                task_id=inst["instance_id"], raw=inst,
                question=f"Fix this issue in {inst['repo']}:\n\n{problem[:4000]}",
                context={
                    "repo": inst["repo"],
                    "problem_statement": problem,
                    "base_commit": inst["base_commit"],
                    "test_patch": inst.get("test_patch", ""),
                    "hints_text": inst.get("hints_text", ""),
                    "FAIL_TO_PASS": f2p,
                    "PASS_TO_PASS": p2p,
                },
            ))
        return tasks

    def extract_answer(self, router_output: str, task: Task) -> str:
        m = re.search(r"```(?:diff)?\s*\n((?:---|\+\+\+|diff\s).*?)```", router_output, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"((?:---\s+a/|diff\s+--git\s).*?)(?:\n\n|\Z)", router_output, re.DOTALL)
        if m:
            return m.group(1).strip()
        return router_output

    # ─── Interactive mode: router ↔ Docker multi-turn (AOrchestra style) ───

    def _legacy_interactive_verify(self, task: Task, router, logs_dir=None) -> VerifyResult:
        """
        Multi-turn interactive evaluation (AOrchestra style):
        1. Start swebench Docker container via AOrchestra executor
        2. Router sees issue → DISCUSSION + COMMAND
        3. COMMAND executes in container → real output back to router
        4. Repeat until 'submit' or max_steps
        5. Run tests in SAME container (AOrchestra executor.run_tests)
        """
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._legacy_async_interactive_verify(task, router, logs_dir)
            )
        finally:
            loop.close()

    async def _legacy_async_interactive_verify(self, task, router, logs_dir=None):
        from ..executors.swebench_executor import SWEBenchExecutor
        from ..executors.swebench_data_loader import SWEBenchInstance
        from pathlib import Path

        ctx = task.context
        task_logs = Path(logs_dir or "/tmp") / task.task_id.replace("/", "_")
        task_logs.mkdir(parents=True, exist_ok=True)
        log = ""

        # Build SWEBenchInstance from task context
        instance = SWEBenchInstance.from_dict(task.raw)

        # Create AOrchestra executor
        executor = SWEBenchExecutor(
            instance=instance,
            logs_dir=task_logs,
            timeout=self.docker_timeout,
        )

        CMD_RE = re.compile(r"COMMAND\s*\n(.+?)(?:\n\n|\Z)", re.DOTALL)

        try:
            # Start container (AOrchestra handles image pull + checkout)
            await executor.start_container()

            # Build prompt
            instruction = (
                f"## Repository: {ctx['repo']}\n\n"
                f"## Issue\n{ctx['problem_statement'][:6000]}\n"
            )
            if ctx.get("hints_text"):
                instruction += f"\n## Hints\n{ctx['hints_text'][:2000]}\n"

            messages = [
                {"role": "system", "content": SWEBENCH_INTERACTIVE_PROMPT},
                {"role": "user", "content": instruction},
            ]

            # Multi-turn loop
            for step in range(self.max_steps):
                try:
                    resp = router.local.chat.completions.create(
                        model=router.model_name,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=2048,
                    )
                    assistant_text = resp.choices[0].message.content or ""
                    _print_verbose_block(
                        f"SWE-Bench response task={task.task_id} step={step+1}",
                        assistant_text,
                    )
                except Exception as e:
                    log += f"\n[ROUTER ERROR step {step}: {e}]"
                    break

                messages.append({"role": "assistant", "content": assistant_text})
                log += f"\n[STEP {step+1}] ASSISTANT:\n{assistant_text[:500]}\n"

                cmd_match = CMD_RE.search(assistant_text)
                if not cmd_match:
                    log += "[NO COMMAND FOUND]\n"
                    break

                command = cmd_match.group(1).strip().split("\n")[0].strip()

                if command.lower() == "submit":
                    log += "[SUBMIT]\n"
                    break

                # Execute in same container via AOrchestra executor
                exec_cmd = f"cd /testbed && {command}"
                output, exit_code = await executor.execute_command(exec_cmd, timeout=120)
                output = output[-2000:]

                obs = f"[Step {step+1}/{self.max_steps}] exit_code={exit_code}\n{output}"
                log += f"[STEP {step+1}] CMD: {command}\n[OUTPUT] {output[:500]}\n"
                messages.append({"role": "user", "content": obs})

            # ── Run tests in SAME container (AOrchestra's run_tests) ──
            reward, test_results = await executor.run_tests()
            log += f"\n[TEST] reward={reward} summary={test_results.get('summary',{})}\n"

            # Save trace
            with (task_logs / "trace.log").open("w") as f:
                f.write(log)

            return VerifyResult(task.task_id, reward, log=log[-3000:])

        except Exception as e:
            return VerifyResult(task.task_id, 0.0, error=str(e)[:300], log=log[-3000:])
        finally:
            await executor.cleanup()

    # ─── One-shot mode (legacy, for non-interactive routers) ───

    def interactive_verify(self, task: Task, router, logs_dir=None) -> VerifyResult:
        """Adapter used by eval_pipeline.run's common interactive path."""
        import asyncio

        from ..config import DEFAULT_API_BASE, MODEL_POOL

        flat_mode = router.name.startswith("Direct(")
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._async_interactive_verify(
                    task=task,
                    router=router,
                    worker_pool=MODEL_POOL,
                    subagent_api_base=getattr(router, "sub_model_api_base", DEFAULT_API_BASE),
                    subagent_api_key=getattr(router, "sub_model_api_key", "EMPTY"),
                    logs_dir=logs_dir,
                    flat_mode=flat_mode,
                )
            )
        finally:
            loop.close()

    async def _async_interactive_verify(
        self,
        task,
        router,
        worker_pool: Optional[List[str]],
        subagent_api_base: Optional[str],
        subagent_api_key: str,
        logs_dir=None,
        flat_mode: bool = False,
    ):
        from ..executors.swebench_executor import SWEBenchExecutor
        from ..executors.swebench_data_loader import SWEBenchInstance
        from uno_orchestor.agents.subagent import SubAgent

        ctx = task.context
        task_logs = Path(logs_dir or "/tmp") / task.task_id.replace("/", "_")
        agent_logs = task_logs / "agent"
        verifier_logs = task_logs / "verifier"
        agent_logs.mkdir(parents=True, exist_ok=True)
        verifier_logs.mkdir(parents=True, exist_ok=True)

        instance = SWEBenchInstance.from_dict(task.raw)
        executor = SWEBenchExecutor(instance=instance, logs_dir=verifier_logs, timeout=self.docker_timeout)
        command_executor = _SWEBenchCommandExecutor(executor)

        instruction = f"## Repository\n{ctx['repo']}\n\n## Issue\n{ctx['problem_statement'][:6000]}\n"
        if ctx.get("hints_text"):
            instruction += f"\n## Hints\n{ctx['hints_text'][:2000]}\n"
        if ctx.get("FAIL_TO_PASS"):
            instruction += f"\n## FAIL_TO_PASS\n{json.dumps(ctx['FAIL_TO_PASS'][:20])}\n"

        budget = (
            f"You may run up to {self.subagent_max_steps} commands."
            if flat_mode
            else f"You have {self.max_attempts} delegation attempts."
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SWEBENCH_FLAT_SYSTEM_PROMPT if flat_mode else SWEBENCH_PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"{instruction}\n"
                f"## Allowed worker models\n{', '.join(worker_pool or [])}\n\n"
                f"## Budget\n{budget}\n"
            )},
        ]

        trajectory: List[Dict[str, Any]] = []
        planner_usages: List[Dict[str, Any]] = []
        subagent_usages: List[Dict[str, Any]] = []
        routed_models: List[str] = []
        reward = 0.0
        submit_called = False
        last_error: Optional[str] = None
        started_at = _now_iso()
        start_perf = time.perf_counter()
        tools = SWEBENCH_FLAT_TOOLS if flat_mode else _planner_tools(worker_pool)

        try:
            await executor.start_container()
            total_steps = self.subagent_max_steps if flat_mode else self.max_attempts
            for step in range(total_steps):
                live_messages = messages + [
                    {
                        "role": "system",
                        "content": (
                            f"Budget: {total_steps - step} command(s) remaining."
                            if flat_mode
                            else _budget_note(step + 1, self.max_attempts)
                        ),
                    }
                ]
                try:
                    resp = router.chat_completions(live_messages, tools=tools)
                except NotImplementedError as e:
                    last_error = f"router {type(router).__name__} lacks chat_completions: {e}"
                    break
                except Exception as e:
                    last_error = f"planner call failed: {e}"
                    break

                content = resp.get("content") or ""
                tool_calls = resp.get("tool_calls") or []
                _print_verbose_block(f"SWE-Bench planner response task={task.task_id} step={step + 1}", content)
                usage = _usage_from_response(
                    resp,
                    getattr(router, "planner_model", getattr(router, "model_name", getattr(router, "model_id", ""))),
                )
                planner_usages.append(usage)
                assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content or None}
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {"id": t["id"], "type": "function",
                         "function": {"name": t["name"], "arguments": json.dumps(t["arguments"])}}
                        for t in tool_calls
                    ]
                messages.append(assistant_msg)
                trajectory.append({"step": step + 1, "planner_content": content, "planner_usage": usage, "tool_calls": tool_calls})

                if not tool_calls:
                    last_error = "planner returned no tool call"
                    break

                did_submit = False
                for tc in tool_calls:
                    name = tc.get("name")
                    args = tc.get("arguments", {}) or {}
                    tc_id = tc.get("id", f"tc_{step}")

                    if name == "submit":
                        trajectory[-1]["submit"] = {"reason": args.get("reason", "(no reason)")}
                        messages.append({"role": "tool", "tool_call_id": tc_id, "content": "Submission received; verifier will run."})
                        did_submit = True
                        break

                    if flat_mode and name == "execute_command":
                        command = (args.get("command") or "").strip()
                        if not command:
                            messages.append({"role": "tool", "tool_call_id": tc_id, "content": "Empty command; ignored."})
                            continue
                        output, exit_code = await command_executor.execute_command(command, timeout=self.subagent_cmd_timeout)
                        output = output[-3000:]
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": json.dumps({"exit_code": exit_code, "output": output[-2000:]}),
                        })
                        trajectory[-1]["execute"] = {"command": command[:500], "exit_code": exit_code, "output_tail": output[-500:]}
                        continue

                    if name == "delegate_task":
                        worker_model = args.get("worker_model") or ""
                        subtask_instruction = args.get("instruction") or ""
                        if worker_pool and worker_model not in worker_pool:
                            worker_model = worker_pool[0]
                            args["worker_model"] = worker_model
                            for saved_tc in trajectory[-1].get("tool_calls", []):
                                if saved_tc.get("id") == tc_id:
                                    saved_tc["arguments"] = args
                        if not subtask_instruction.strip():
                            msg = "Empty instruction; delegate_task skipped."
                            messages.append({"role": "tool", "tool_call_id": tc_id, "content": msg})
                            trajectory[-1]["delegate"] = {"error": msg}
                            continue
                        try:
                            worker_api_base, worker_api_key = _resolve_model_endpoint(worker_model, subagent_api_base or "", subagent_api_key)
                            subagent = SubAgent(
                                api_base=worker_api_base,
                                api_key=worker_api_key,
                                max_steps=self.subagent_max_steps,
                                cmd_timeout=self.subagent_cmd_timeout,
                            )
                            sub_result = await subagent.run(
                                model=worker_model,
                                task_instruction=subtask_instruction,
                                original_question=instruction,
                                executor=command_executor,
                                agent_logs_dir=agent_logs,
                                tool_call_id=tc_id,
                            )
                        except Exception as e:
                            sub_result = {
                                "status": "error",
                                "completed": [],
                                "issues": [str(e)[:300]],
                                "message": f"SubAgent crashed: {e}",
                                "steps_taken": 0,
                                "model": worker_model,
                                "commands_log": [],
                                "cost": 0.0,
                                "tokens": 0,
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                            }
                        routed_models.append(worker_model)
                        subagent_usages.append({
                            "model": worker_model,
                            "cost": float(sub_result.get("cost", 0) or 0),
                            "tokens": int(sub_result.get("tokens", 0) or 0),
                            "prompt_tokens": int(sub_result.get("prompt_tokens", 0) or 0),
                            "completion_tokens": int(sub_result.get("completion_tokens", 0) or 0),
                        })
                        planner_view = SubAgent.format_result_for_planner(sub_result)
                        messages.append({"role": "tool", "tool_call_id": tc_id, "content": planner_view})
                        trajectory[-1]["delegate"] = {
                            "worker_model": worker_model,
                            "instruction": subtask_instruction[:500],
                            "sub_result": {
                                k: sub_result.get(k)
                                for k in (
                                    "status", "steps_taken", "completed", "issues", "message",
                                    "model", "cost", "tokens", "prompt_tokens", "completion_tokens",
                                    "summary_model", "summary_usage", "step_logs",
                                )
                            },
                        }
                    else:
                        messages.append({"role": "tool", "tool_call_id": tc_id, "content": f"Unknown tool '{name}'; ignored."})

                if did_submit:
                    submit_called = True
                    break

            reward, test_results = await executor.run_tests()
            trajectory.append({"verifier": {"reward": reward, "summary": test_results.get("summary", {})}})
        except Exception as e:
            last_error = str(e)[:300]
        finally:
            await executor.cleanup()

        try:
            ended_at = _now_iso()
            elapsed_seconds = round(time.perf_counter() - start_perf, 3)
            planner_totals = _sum_usage(planner_usages)
            subagent_totals = _sum_usage(subagent_usages)
            total_usage = _sum_usage([planner_totals, subagent_totals])
            _write_json_atomic(task_logs / "trajectory.json", {
                "task_id": task.task_id,
                "mode": "flat" if flat_mode else "planner",
                "started_at": started_at,
                "ended_at": ended_at,
                "elapsed_seconds": elapsed_seconds,
                "reward": reward,
                "submit_called": submit_called,
                "attempts_used": len([x for x in trajectory if "verifier" not in x]),
                "max_attempts": self.subagent_max_steps if flat_mode else self.max_attempts,
                "last_error": last_error,
                "route_count": len(routed_models),
                "routed_models": routed_models or ([getattr(router, "model_id", "")] if flat_mode else []),
                "routed_skills": (["execute_command"] if flat_mode else ["delegate_task"] * len(routed_models)),
                "routed_backends": (["direct_flat"] if flat_mode else ["swebench_subagent"] * len(routed_models)),
                "planner_usage": planner_totals,
                "subagent_usage": subagent_totals,
                **total_usage,
                "trajectory": trajectory,
            })
        except Exception:
            pass

        return VerifyResult(
            task.task_id,
            reward,
            error=last_error,
            log=json.dumps({
                "attempts": len([x for x in trajectory if "verifier" not in x]),
                "submit": submit_called,
                "route_count": len(routed_models),
                "routed_models": routed_models,
                **_sum_usage([_sum_usage(planner_usages), _sum_usage(subagent_usages)]),
            })[:3000],
        )

    def verify(self, task: Task, answer: str, logs_dir=None) -> VerifyResult:
        work_dir = os.path.join(logs_dir or "/tmp", task.task_id.replace("/", "_"))
        os.makedirs(work_dir, exist_ok=True)
        pred_path = os.path.join(work_dir, "predictions.jsonl")
        run_id = f"single_{int(time.time())}"

        with open(pred_path, "w") as f:
            f.write(json.dumps({
                "instance_id": task.task_id,
                "model_name_or_path": "eval",
                "model_patch": answer,
            }) + "\n")

        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python3", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", self.dataset, "--split", self.split,
            "--predictions_path", pred_path,
            "--instance_ids", task.task_id,
            "--max_workers", "1",
            "--run_id", run_id,
            "--timeout", str(self.eval_timeout),
            "--cache_level", "instance",
            "--report_dir", os.path.join(work_dir, "reports"),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=self.eval_timeout + 300)
            log = f"stdout: {proc.stdout[-1000:]}\nstderr: {proc.stderr[-1000:]}"
        except subprocess.TimeoutExpired:
            return VerifyResult(task.task_id, 0.0, error="Harness timeout")
        except Exception as e:
            return VerifyResult(task.task_id, 0.0, error=str(e)[:300])

        resolved = False
        report_dir = os.path.join(work_dir, "reports", run_id)
        for root, dirs, files in os.walk(report_dir):
            for fname in files:
                if fname.endswith(".json"):
                    try:
                        data = json.load(open(os.path.join(root, fname)))
                        if task.task_id in data.get("resolved", []):
                            resolved = True
                    except Exception:
                        pass
        return VerifyResult(task.task_id, 1.0 if resolved else 0.0, log=log[:500])

    def verify_batch(self, tasks: List[Task], answers: List[str],
                     logs_dir: str = None) -> List[VerifyResult]:
        work_dir = logs_dir or tempfile.mkdtemp(prefix="swebench_eval_")
        os.makedirs(work_dir, exist_ok=True)
        pred_path = os.path.join(work_dir, "predictions.jsonl")
        run_id = "eval_run"

        with open(pred_path, "w") as f:
            for task, ans in zip(tasks, answers):
                f.write(json.dumps({
                    "instance_id": task.task_id,
                    "model_name_or_path": "eval",
                    "model_patch": ans,
                }) + "\n")

        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python3", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", self.dataset, "--split", self.split,
            "--predictions_path", pred_path,
            "--max_workers", str(self.eval_workers),
            "--run_id", run_id,
            "--timeout", str(self.eval_timeout),
            "--cache_level", "instance",
            "--report_dir", os.path.join(work_dir, "reports"),
        ]
        print(f"[SWE-bench] Running harness: {' '.join(cmd[:8])}...")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        output_root = Path(work_dir).parent if Path(work_dir).name == "logs" else Path(work_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "swebench_harness_stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (output_root / "swebench_harness_stderr.log").write_text(proc.stderr or "", encoding="utf-8")
        if proc.stdout:
            print(proc.stdout[-4000:])
        if proc.stderr:
            print(proc.stderr[-4000:])

        resolved_ids = set()
        report_payloads = []
        report_dir = os.path.join(work_dir, "reports", run_id)
        for root, dirs, files in os.walk(report_dir):
            for fname in files:
                if fname.endswith(".json"):
                    try:
                        data = json.load(open(os.path.join(root, fname)))
                        report_payloads.append(data)
                        if "resolved" in data:
                            resolved_ids.update(data["resolved"])
                    except Exception:
                        pass
        _write_swebench_failure_report(
            output_root=output_root,
            tasks=tasks,
            answers=answers,
            resolved_ids=resolved_ids,
            report_payloads=report_payloads,
            report_dir=Path(report_dir),
            harness_returncode=proc.returncode,
        )
        return [VerifyResult(t.task_id, 1.0 if t.task_id in resolved_ids else 0.0) for t in tasks]


def _write_swebench_failure_report(
    output_root: Path,
    tasks: List[Task],
    answers: List[str],
    resolved_ids: set[str],
    report_payloads: list[dict],
    report_dir: Path,
    harness_returncode: int,
) -> None:
    submitted_ids = {task.task_id for task in tasks}
    status_by_id: dict[str, str] = {task.task_id: "resolved" if task.task_id in resolved_ids else "unknown" for task in tasks}
    details_by_id: dict[str, dict] = {}

    for payload in report_payloads:
        for key, status in [
            ("resolved", "resolved"),
            ("resolved_ids", "resolved"),
            ("unresolved", "unresolved"),
            ("unresolved_ids", "unresolved"),
            ("error", "error"),
            ("error_ids", "error"),
            ("empty_patch", "empty_patch"),
            ("empty_patch_ids", "empty_patch"),
            ("completed", "completed"),
            ("completed_ids", "completed"),
        ]:
            value = payload.get(key)
            if isinstance(value, list):
                for instance_id in value:
                    if instance_id in submitted_ids and status_by_id.get(instance_id) != "resolved":
                        status_by_id[instance_id] = status
        for instance_id, value in payload.items():
            if instance_id in submitted_ids and isinstance(value, dict):
                details_by_id[instance_id] = value

    instances = []
    for task, patch in zip(tasks, answers):
        patch_text = str(patch or "")
        status = status_by_id.get(task.task_id, "unknown")
        if task.task_id in resolved_ids:
            status = "resolved"
        reason = _swebench_reason(task.task_id, status, patch_text, details_by_id.get(task.task_id), report_dir, output_root)
        instances.append(
            {
                "task_id": task.task_id,
                "status": status,
                "resolved": task.task_id in resolved_ids,
                "patch_chars": len(patch_text),
                "patch_diagnostic": _patch_diagnostic(patch_text),
                "reason": reason,
            }
        )

    failed = [item for item in instances if not item["resolved"]]
    report = {
        "harness_returncode": harness_returncode,
        "total_submitted": len(instances),
        "resolved": len(instances) - len(failed),
        "failed_or_error": len(failed),
        "report_dir": str(report_dir),
        "instances": instances,
    }
    json_path = output_root / "swebench_failure_report.json"
    md_path = output_root / "swebench_failure_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# SWE-Bench Failure Report",
        "",
        f"- Harness return code: `{harness_returncode}`",
        f"- Submitted: `{len(instances)}`",
        f"- Resolved: `{len(instances) - len(failed)}`",
        f"- Failed/error: `{len(failed)}`",
        "",
    ]
    for item in failed:
        lines.extend(
            [
                f"## {item['task_id']}",
                "",
                f"- Status: `{item['status']}`",
                f"- Patch chars: `{item['patch_chars']}`",
                f"- Patch diagnostic: {item['patch_diagnostic']}",
                "",
                "```text",
                item["reason"][:4000],
                "```",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SWE-bench] Failure report: {json_path}")
    print(f"[SWE-bench] Failure report: {md_path}")


def _patch_diagnostic(patch: str) -> str:
    stripped = patch.strip()
    if not stripped:
        return "empty patch"
    if "diff --git " not in stripped and not stripped.startswith("--- "):
        return "not a unified diff patch"
    if "--- " in stripped and "+++ " in stripped:
        return "looks like a unified diff"
    return "diff-like text, but missing expected ---/+++ headers"


def _swebench_reason(
    task_id: str,
    status: str,
    patch: str,
    details: dict | None,
    report_dir: Path,
    output_root: Path,
) -> str:
    diagnostic = _patch_diagnostic(patch)
    parts = [f"status={status}", f"patch_diagnostic={diagnostic}"]
    if diagnostic != "looks like a unified diff":
        parts.append("The model output is unlikely to be applicable by git apply.")
    if details:
        parts.append("Harness details:")
        parts.append(json.dumps(details, ensure_ascii=False, indent=2)[:3000])

    log_excerpt = _find_instance_log_excerpt(task_id, [report_dir, output_root])
    if log_excerpt:
        parts.append("Relevant log excerpt:")
        parts.append(log_excerpt)
    elif status in {"unresolved", "completed"}:
        parts.append("The patch was evaluated but did not resolve all FAIL_TO_PASS/PASS_TO_PASS checks.")
    elif status in {"error", "unknown"}:
        parts.append("No per-instance log was found; check swebench_harness_stdout.log and swebench_harness_stderr.log.")
    return "\n\n".join(parts)


def _find_instance_log_excerpt(task_id: str, roots: list[Path], limit: int = 3000) -> str:
    patterns = ("error", "failed", "fail", "traceback", "git apply", "patch", "exception", "timeout")
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and task_id in str(path) and path.suffix.lower() in {".log", ".txt", ".json"}:
                candidates.append(path)
    for path in candidates[:8]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lines = [line for line in text.splitlines() if any(pattern in line.lower() for pattern in patterns)]
        excerpt = "\n".join(lines[-40:]) if lines else text[-limit:]
        if excerpt.strip():
            return f"{path}\n{excerpt[-limit:]}"
    return ""
