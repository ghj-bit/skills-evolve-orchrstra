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
from pathlib import Path
from typing import List
from .base import BaseBenchmark, Task, VerifyResult


# ── SWE-bench system prompt for interactive mode ──

SWEBENCH_INTERACTIVE_PROMPT = """\
You are a software engineer fixing a GitHub issue inside a Docker container.
The repository is at /testbed, already checked out to the correct commit.

Available actions (respond with EXACTLY one per turn):

DISCUSSION
<your reasoning about what to do next>
COMMAND
<single bash command to execute>

OR when you're done fixing:

DISCUSSION
<explain your fix>
COMMAND
submit

Commands run in /testbed with conda env "testbed" active.
Useful commands: find, grep -rn, cat, python, pytest, git diff, etc.

IMPORTANT:
- Activate conda first: source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed
- Explore before editing. Use grep/find to locate relevant code.
- Make minimal, targeted changes.
- Test your fix before submitting.
"""


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


class SWEBench(BaseBenchmark):

    def __init__(self, dataset="princeton-nlp/SWE-bench_Verified", split="test",
                 conda_env="swebench", eval_timeout=900, eval_workers=4,
                 max_steps=30, docker_timeout=1800):
        self.dataset = dataset
        self.split = split
        self.conda_env = conda_env
        self.eval_timeout = eval_timeout
        self.eval_workers = eval_workers
        self.max_steps = max_steps
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
                question=(
                    f"Fix this issue in {inst['repo']}:\n\n{problem[:4000]}\n\n"
                    "Return a minimal unified diff patch as the final answer. "
                    "The final answer should contain only the patch, preferably in a ```diff``` block."
                ),
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

    def interactive_verify(self, task: Task, router, logs_dir=None) -> VerifyResult:
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
                self._async_interactive_verify(task, router, logs_dir)
            )
        finally:
            loop.close()

    async def _async_interactive_verify(self, task, router, logs_dir=None):
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
