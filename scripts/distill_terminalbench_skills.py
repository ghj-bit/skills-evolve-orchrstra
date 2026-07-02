#!/usr/bin/env python3
"""Distill Terminal-Bench skills from trajectory.json + commands.log.

The reflection prompt follows the Routing_0421 D2Skill style: analyze a full
trajectory, identify the main success/failure causes, then emit exactly two
step-level reflections. The output schema is the same list-of-dicts format used
by terminal_bench_skills_init.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_EVAL_DIR = Path("data/eval/deepseek_v4_pro_qwen3_8b_router_terminalbench")
DEFAULT_EXISTING_SKILLS = Path("terminal_bench_skills_init.json")
DEFAULT_OUTPUT = Path("terminal_bench_skills_gen.json")
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_API_BASE = "https://api.deepseek.com"

EXTERNAL_ERROR_PATTERNS = {
    "qwen_429_tpm": re.compile(r"429|TPM limit reached|rate limiting", re.I),
    "timeout": re.compile(r"Request timed out|timed out|timeout", re.I),
    "context_too_long": re.compile(
        r"context.*(length|window)|maximum context|too long|input.*too long|prompt.*too long|exceed",
        re.I,
    ),
    "docker_proxy_pull": re.compile(
        r"Failed to (start container|pull image)|127\.0\.0\.1:7897|docker\.io|archive\.ubuntu\.com|security\.ubuntu\.com",
        re.I,
    ),
    "planner_no_tool": re.compile(r"planner returned no tool call", re.I),
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return text[:half] + "\n...<truncated>...\n" + text[-half:]


def collect_error_text(data: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key in ("last_error", "error"):
        if data.get(key):
            chunks.append(str(data[key]))
    for item in data.get("trajectory", []) or []:
        delegate = item.get("delegate") or {}
        sub_result = delegate.get("sub_result") or {}
        if sub_result.get("status") in {"error", "partial"}:
            chunks.append(str(sub_result.get("message", "")))
            chunks.extend(str(x) for x in sub_result.get("issues") or [])
    return "\n".join(chunks)


def external_error_labels(data: dict[str, Any]) -> list[str]:
    text = collect_error_text(data)
    return [name for name, pattern in EXTERNAL_ERROR_PATTERNS.items() if pattern.search(text)]


def iter_trajectories(eval_dir: Path, attempts: set[str] | None) -> list[Path]:
    logs = eval_dir / "logs"
    if not logs.exists():
        return []
    rows = []
    for path in sorted(logs.glob("attempt_*/*/trajectory.json")):
        if attempts and path.parent.parent.name not in attempts:
            continue
        rows.append(path)
    return rows


def commands_log_path(traj_path: Path) -> Path:
    return traj_path.parent / "agent" / "commands.log"


def verifier_test_output_path(traj_path: Path) -> Path:
    return traj_path.parent / "verifier" / "test_output.log"


def verifier_ctrf_path(traj_path: Path) -> Path:
    return traj_path.parent / "verifier" / "ctrf.json"


def compact_trajectory(data: dict[str, Any], max_delegates: int) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "task_id": data.get("task_id"),
        "reward": data.get("reward"),
        "submit_called": data.get("submit_called"),
        "attempts_used": data.get("attempts_used"),
        "max_attempts": data.get("max_attempts"),
        "last_error": data.get("last_error"),
        "route_count": data.get("route_count"),
        "routed_models": data.get("routed_models", []),
        "planner_usage": data.get("planner_usage", {}),
        "subagent_usage": data.get("subagent_usage", {}),
        "delegates": [],
    }
    for item in (data.get("trajectory") or [])[:max_delegates]:
        calls = item.get("tool_calls") or []
        delegate = item.get("delegate") or {}
        sub = delegate.get("sub_result") or {}
        compact["delegates"].append({
            "attempt": item.get("attempt"),
            "planner_content": normalize_space(str(item.get("planner_content") or ""))[:800],
            "tool_calls": [
                {
                    "name": c.get("name"),
                    "arguments": c.get("arguments"),
                }
                for c in calls[:2]
            ],
            "worker_model": delegate.get("worker_model"),
            "instruction": normalize_space(str(delegate.get("instruction") or ""))[:1200],
            "sub_result": {
                "status": sub.get("status"),
                "steps_taken": sub.get("steps_taken"),
                "completed": sub.get("completed", []),
                "issues": sub.get("issues", []),
                "message": sub.get("message"),
            },
        })
    return compact


def read_commands_log(path: Path, limit: int) -> str:
    if not path.exists():
        return ""
    return truncate_middle(path.read_text(encoding="utf-8", errors="replace"), limit)


def read_text_file(path: Path, limit: int) -> str:
    if not path.exists():
        return ""
    return truncate_middle(path.read_text(encoding="utf-8", errors="replace"), limit)


def read_json_file_compact(path: Path, limit: int) -> str:
    if not path.exists():
        return ""
    try:
        data = load_json(path)
        text = json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        text = path.read_text(encoding="utf-8", errors="replace")
    return truncate_middle(text, limit)


def build_reflection_prompt(
    *,
    trajectory: dict[str, Any],
    commands_log: str,
    test_output_log: str,
    ctrf_json: str,
    source_path: Path,
) -> str:
    task_id = trajectory.get("task_id") or source_path.parent.name
    reward = trajectory.get("reward")
    submit = trajectory.get("submit_called")
    return f"""You are given one Terminal-Bench planner/sub-agent trajectory, its Docker command log, and verifier outputs.
Analyze the execution and distill exactly two reusable step-level skills for future Terminal-Bench sub-agents.

This prompt follows a D2Skill-style reflection pattern:
analyze the entire trajectory, identify the main success/failure causes, then produce exactly two distinct step-level skills that the agent most needs.

Task ID: {task_id}
Benchmark: Terminal-Bench
Reward: {reward}
Planner submit_called: {submit}
Source trajectory: {source_path.as_posix()}

Compact trajectory JSON:
{json.dumps(trajectory, ensure_ascii=False, indent=2)}

Commands log excerpt:
{commands_log}

Verifier test_output.log excerpt:
{test_output_log or "<missing or empty>"}

Verifier ctrf.json excerpt:
{ctrf_json or "<missing or empty>"}

Output the following in order using the exact section headers:

1) TRAJECTORY_DIAGNOSIS:
   Output 3-6 concise bullet lines in plain text. Explain the main reasons the trajectory succeeded or failed.
   Consider planner decomposition, sub-agent command choices, command outputs, validation behavior, verifier outcome, repeated mistakes,
   and concrete failures reported in test_output.log and ctrf.json.
   If reward is 0, identify verifier failure signals from test_output.log/ctrf.json whenever available.
   If reward is 1, identify command and verifier evidence that confirms success.
   Do not focus on locating the first error; use the whole trajectory.

2) STEP_REFLECTION:
   Output exactly one JSON object in the exact Terminal-Bench skill schema below.
   This must be the most important step/subtask-level skill the agent needs based on the full trajectory.
   Do not output an array, multiple alternatives, or more than one skill under this section.
   Required keys:
   - name: short title
   - level: "subtask"
   - description: one reusable instruction
   - application_scenarios: array of 3-6 strings
   - execution_principles: array of 4-7 actionable rules
   - workflow: array of 4-8 snake_case steps
   - example: object with "situation" and "steps" array
   - failure_cases_prevented: array of 3-6 snake_case strings

3) STEP_REFLECTION_2:
   Output exactly one JSON object in the same schema.
   This must be the second most important, distinct step/subtask-level skill the agent needs based on the full trajectory.
   Do not output an array, multiple alternatives, or more than one skill under this section.
   Required keys are the same, and level must be "subtask".

Requirements:
- Each source trajectory must produce exactly two skills total: STEP_REFLECTION and STEP_REFLECTION_2.
- Both skills must have level set to "subtask".
- Skills must be selected from the most important full-trajectory success/failure causes, not from the first error location.
- Skills must be reusable beyond this exact task_id.
- Skills must be actionable for a smaller model running shell commands in Docker.
- Prefer lessons grounded in concrete command failures, verifier failures, missing validation, or successful repair patterns.
- Use test_output.log and ctrf.json as primary evidence for why the final result passed or failed.
- If verifier output contradicts sub-agent self-reported success, trust the verifier output.
- Do not mention model names, API keys, private paths, or this exact trajectory filename inside skill text.
- Do not produce generic advice like "be careful"; encode operational steps.
- Return JSON objects only under the requested section headers.
- The two skills should not duplicate the same idea.
- Do not add any other section, explanation, markdown fence, bullet list, or extra skill.

Output format:
TRAJECTORY_DIAGNOSIS:
<3-6 concise bullet lines>

STEP_REFLECTION:
<single JSON object>

STEP_REFLECTION_2:
<single JSON object>"""


def ensure_v1_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def call_openai_compatible(
    *,
    prompt: str,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> str:
    url = ensure_v1_url(api_base) + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                obj = json.loads(resp.read().decode("utf-8", errors="replace"))
            content = obj["choices"][0]["message"]["content"] or ""
            if not content.strip():
                raise RuntimeError("empty LLM response content")
            return content
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_error}")


def extract_json_after_label(text: str, label: str) -> dict[str, Any] | None:
    m = re.search(rf"{re.escape(label)}\s*:", text, re.I)
    if not m:
        return None
    raw = text[m.end():]
    fence = re.match(r"\s*```(?:json)?\s*", raw, re.I)
    if fence:
        raw = raw[fence.end():]
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    end = None
    for idx, char in enumerate(raw[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end is None:
        return None
    raw = raw[start:end].strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def extract_diagnosis(text: str) -> str:
    m = re.search(
        r"TRAJECTORY_DIAGNOSIS\s*:\s*(.*?)(?=\nSTEP_REFLECTION\s*:|\Z)",
        text,
        re.I | re.S,
    )
    return normalize_space(m.group(1))[:1200] if m else ""


def parse_reflection(text: str) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    return (
        extract_diagnosis(text),
        extract_json_after_label(text, "STEP_REFLECTION"),
        extract_json_after_label(text, "STEP_REFLECTION_2"),
    )


def normalize_skill(skill: dict[str, Any], *, level: str, source: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(skill, dict):
        return None
    name = str(skill.get("name") or skill.get("title") or "").strip()
    description = str(skill.get("description") or skill.get("principle") or "").strip()
    if not name or not description:
        return None

    def as_list(key: str, fallback: list[str]) -> list[str]:
        val = skill.get(key)
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        if isinstance(val, str) and val.strip():
            return [val.strip()]
        return fallback

    example = skill.get("example")
    if not isinstance(example, dict):
        example = {
            "situation": str(skill.get("when_to_apply") or f"Terminal-Bench {level} execution"),
            "steps": as_list("workflow", []),
        }
    if not isinstance(example.get("steps"), list):
        example["steps"] = [str(example.get("steps"))]

    normalized = {
        "name": name[:120],
        "level": level,
        "description": description[:700],
        "application_scenarios": as_list("application_scenarios", ["Terminal-Bench tasks", "Docker command execution"]),
        "execution_principles": as_list("execution_principles", []),
        "workflow": as_list("workflow", []),
        "example": {
            "situation": str(example.get("situation") or "")[:300],
            "steps": [str(x)[:300] for x in example.get("steps", [])[:10]],
        },
        "failure_cases_prevented": as_list("failure_cases_prevented", []),
        "source": source,
    }
    return normalized


def skill_fingerprint(skill: dict[str, Any]) -> str:
    fields = [
        skill.get("name", ""),
        skill.get("level", ""),
        skill.get("description", ""),
        " ".join(skill.get("execution_principles") or []),
    ]
    return normalize_space(" ".join(fields)).lower()


def load_existing_skills(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = load_json(path)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [x for x in (data.get("task_skills") or []) + (data.get("step_skills") or []) if isinstance(x, dict)]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Distill Terminal-Bench skills from trajectories.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--existing-skills", type=Path, default=DEFAULT_EXISTING_SKILLS)
    parser.add_argument("--merge-existing", action="store_true", help="Include existing skills before distilled skills.")
    parser.add_argument("--attempts", default="attempt_0,attempt_1", help="Comma-separated attempt dirs, or empty for all.")
    parser.add_argument("--tasks", default="", help="Comma-separated task ids to include. Empty means all selected attempts.")
    parser.add_argument("--normal-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-trajectories", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--max-delegates", type=int, default=8)
    parser.add_argument("--commands-chars", type=int, default=16000)
    parser.add_argument("--test-output-chars", type=int, default=12000)
    parser.add_argument("--ctrf-chars", type=int, default=12000)
    parser.add_argument("--prompt-dir", type=Path, default=None)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL_ID", DEFAULT_MODEL))
    parser.add_argument("--api-base", default=os.environ.get("DEEPSEEK_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Select trajectories and write prompts without calling the LLM.")
    args = parser.parse_args()

    if not args.api_key and not args.dry_run:
        print("Missing --api-key or DEEPSEEK_API_KEY.", file=sys.stderr)
        return 2

    attempts = {x.strip() for x in args.attempts.split(",") if x.strip()} if args.attempts.strip() else None
    task_filter = {x.strip() for x in args.tasks.split(",") if x.strip()}
    paths = iter_trajectories(args.eval_dir, attempts)
    selected: list[Path] = []
    skipped_external = 0
    for path in paths:
        task_id = path.parent.name
        if task_filter and task_id not in task_filter:
            continue
        data = load_json(path)
        labels = external_error_labels(data)
        if args.normal_only and labels:
            skipped_external += 1
            continue
        selected.append(path)
        if args.max_trajectories and len(selected) >= args.max_trajectories:
            break

    if args.prompt_dir:
        args.prompt_dir.mkdir(parents=True, exist_ok=True)

    output_skills: list[dict[str, Any]] = load_existing_skills(args.existing_skills) if args.merge_existing else []
    seen = {skill_fingerprint(s) for s in output_skills}
    raw_records: list[dict[str, Any]] = []

    for index, path in enumerate(selected, start=1):
        data = load_json(path)
        compact = compact_trajectory(data, args.max_delegates)
        commands = read_commands_log(commands_log_path(path), args.commands_chars)
        test_output = read_text_file(verifier_test_output_path(path), args.test_output_chars)
        ctrf = read_json_file_compact(verifier_ctrf_path(path), args.ctrf_chars)
        prompt = build_reflection_prompt(
            trajectory=compact,
            commands_log=commands,
            test_output_log=test_output,
            ctrf_json=ctrf,
            source_path=path,
        )
        if args.prompt_dir:
            (args.prompt_dir / f"{index:04d}_{path.parent.parent.name}_{path.parent.name}_prompt.txt").write_text(
                prompt,
                encoding="utf-8",
            )
        print(f"[{index}/{len(selected)}] distilling {path.parent.parent.name}/{path.parent.name}", flush=True)
        if args.dry_run:
            raw_records.append({
                "source": {
                    "task_id": path.parent.name,
                    "attempt": path.parent.parent.name,
                    "trajectory": path.as_posix(),
                    "commands_log": commands_log_path(path).as_posix(),
                    "test_output_log": verifier_test_output_path(path).as_posix(),
                    "ctrf_json": verifier_ctrf_path(path).as_posix(),
                    "reward": data.get("reward"),
                    "submit_called": data.get("submit_called"),
                    "distill_model": args.model,
                },
                "prompt_chars": len(prompt),
                "dry_run": True,
            })
            continue
        try:
            raw = call_openai_compatible(
                prompt=prompt,
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
            )
        except Exception as exc:
            raw_records.append({
                "source": {
                    "task_id": path.parent.name,
                    "attempt": path.parent.parent.name,
                    "trajectory": path.as_posix(),
                    "commands_log": commands_log_path(path).as_posix(),
                    "test_output_log": verifier_test_output_path(path).as_posix(),
                    "ctrf_json": verifier_ctrf_path(path).as_posix(),
                    "reward": data.get("reward"),
                    "submit_called": data.get("submit_called"),
                    "distill_model": args.model,
                },
                "raw_response": "",
                "error": str(exc),
            })
            write_json(args.output.with_suffix(".raw.json"), raw_records)
            continue
        if args.prompt_dir:
            (args.prompt_dir / f"{index:04d}_{path.parent.parent.name}_{path.parent.name}_response.txt").write_text(
                raw,
                encoding="utf-8",
            )
        trajectory_diagnosis, step_skill, step_skill_2 = parse_reflection(raw)
        source = {
            "task_id": path.parent.name,
            "attempt": path.parent.parent.name,
            "trajectory": path.as_posix(),
            "commands_log": commands_log_path(path).as_posix(),
            "test_output_log": verifier_test_output_path(path).as_posix(),
            "ctrf_json": verifier_ctrf_path(path).as_posix(),
            "reward": data.get("reward"),
            "submit_called": data.get("submit_called"),
            "trajectory_diagnosis": trajectory_diagnosis,
            "distill_model": args.model,
        }
        if not step_skill or not step_skill_2:
            raw_records.append({
                "source": source,
                "raw_response": raw,
                "parsed_step": step_skill,
                "parsed_step_2": step_skill_2,
                "error": "failed to parse exactly two complete step-level skills",
            })
            write_json(args.output.with_suffix(".raw.json"), raw_records)
            continue
        for level, raw_skill in (("subtask", step_skill), ("subtask", step_skill_2)):
            normalized = normalize_skill(raw_skill or {}, level=level, source=source)
            if not normalized:
                continue
            fp = skill_fingerprint(normalized)
            if fp in seen:
                continue
            seen.add(fp)
            output_skills.append(normalized)
        raw_records.append({
            "source": source,
            "raw_response": raw,
            "parsed_step": step_skill,
            "parsed_step_2": step_skill_2,
        })
        write_json(args.output, output_skills)
        write_json(args.output.with_suffix(".raw.json"), raw_records)

    write_json(args.output, output_skills)
    write_json(args.output.with_suffix(".raw.json"), raw_records)
    print(
        f"Wrote {len(output_skills)} skills to {args.output} "
        f"(selected={len(selected)}, skipped_external={skipped_external})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
