"""SubAgent: multi-turn Docker executor for TerminalBench pipeline.

Designed to be called from Planner's execute_subtask callback.
Each invocation runs a multi-turn loop:
  1. SubAgent LLM receives task instruction + Docker observation
  2. Outputs structured JSON: {"action":"execute","params":{"command":"..."}}
  3. Command runs in Docker, output returned as next observation
  4. Repeat until SubAgent calls "finish" or hits max_steps

Returns a structured report string that the Planner can interpret.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _step_timestamps_enabled() -> bool:
    return os.environ.get("TERMINALBENCH_STEP_TIMESTAMPS", "1").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def _log_step_start(phase: str, **fields: Any) -> None:
    if not _step_timestamps_enabled():
        return
    details = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    suffix = f" {details}" if details else ""
    print(f"[{_now_iso()}] [SubAgent] phase={phase}{suffix}", flush=True)


def _subagent_verbose_enabled() -> bool:
    return os.environ.get("SUBAGENT_VERBOSE", "0").lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def _subagent_llm_debug_enabled() -> bool:
    return os.environ.get("SUBAGENT_LLM_DEBUG", "0").lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def _subagent_include_step_logs() -> bool:
    return os.environ.get("SUBAGENT_INCLUDE_STEP_LOGS", "0").lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def _subagent_return_transcript() -> bool:
    return os.environ.get("SUBAGENT_RETURN_TRANSCRIPT", "0").lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def _subagent_skills_enabled() -> bool:
    return os.environ.get("SUBAGENT_ENABLE_SKILLS", "0").lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def _subagent_skills_enabled_for_model(model: str) -> bool:
    if not _subagent_skills_enabled():
        return False
    raw = os.environ.get("SUBAGENT_SKILLS_MODELS", "Qwen/Qwen3-8B")
    allowed = [m.strip() for m in raw.split(",") if m.strip()]
    if not allowed:
        return True
    return model in allowed


def _subagent_skill_top_k() -> int:
    try:
        return max(1, int(os.environ.get("SUBAGENT_SKILLS_TOP_K", "2")))
    except ValueError:
        return 2


def _subagent_skill_mode() -> str:
    # Retrieved skills are fixed per delegated sub-agent task. Step-level
    # re-retrieval made prompts drift when command output contained noisy logs.
    return "task_only"


def _subagent_skills_path() -> str:
    return os.environ.get("TERMINAL_BENCH_SKILLS_PATH", "terminal_bench_skills_init.json")


def _safe_prompt_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _subagent_prompt_dump_task_dir(agent_logs_dir: Any) -> Path | None:
    root = os.environ.get("TERMINALBENCH_PROMPT_DUMP_DIR", "").strip()
    if not root:
        return None
    task_id = "unknown_task"
    if agent_logs_dir:
        try:
            task_id = Path(agent_logs_dir).parent.name or task_id
        except Exception:
            pass
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    task_dir = root_path / _safe_prompt_name(task_id)
    if not task_dir.exists():
        try:
            limit = max(1, int(os.environ.get("TERMINALBENCH_PROMPT_DUMP_TASK_LIMIT", "1")))
        except ValueError:
            limit = 1
        existing = [p for p in root_path.iterdir() if p.is_dir()]
        if len(existing) >= limit:
            return None
        task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def _dump_subagent_prompt(
    agent_logs_dir: Any,
    *,
    tool_call_id: str | None,
    model: str,
    step: int,
    attempt: int,
    messages: list[dict],
) -> None:
    task_dir = _subagent_prompt_dump_task_dir(agent_logs_dir)
    if task_dir is None:
        return
    tc = _safe_prompt_name(tool_call_id or "no_tool_call")
    path = task_dir / f"subagent_{tc}_step_{step:02d}_attempt_{attempt + 1}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "metadata": {
            "kind": "subagent",
            "tool_call_id": tool_call_id,
            "model": model,
            "step": step,
            "retry_attempt": attempt + 1,
        },
        "messages": messages,
    }
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def _subagent_summary_model() -> str:
    return os.environ.get("SUBAGENT_SUMMARY_MODEL_ID", "").strip()


def _subagent_summary_api_base(default_api_base: str) -> str:
    return os.environ.get("SUBAGENT_SUMMARY_API_BASE", default_api_base).strip() or default_api_base


def _subagent_summary_api_key(default_api_key: str) -> str:
    return os.environ.get("SUBAGENT_SUMMARY_API_KEY", default_api_key).strip() or default_api_key


def _subagent_summary_max_tokens() -> int:
    try:
        return max(128, int(os.environ.get("SUBAGENT_SUMMARY_MAX_TOKENS", "512")))
    except ValueError:
        return 512


def _subagent_log(title: str, body: Any = "") -> None:
    if not _subagent_verbose_enabled():
        return
    print(f"\n[SubAgent {title}]", flush=True)
    if body is not None and body != "":
        print(str(body), flush=True)



def _message_debug_stats(messages: list[dict]) -> tuple[int, str]:
    total_chars = 0
    roles = []
    for message in messages:
        role = str(message.get("role", "?"))
        roles.append(role)
        content = message.get("content", "")
        if content is None:
            continue
        total_chars += len(str(content))
    return total_chars, ",".join(roles)


def _llm_exception_debug(
    exc: Exception,
    *,
    model: str,
    api_base: str,
    step: int,
    attempt: int,
    messages: list[dict],
    extra_body: dict,
) -> str:
    prompt_chars, roles = _message_debug_stats(messages)
    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    response_text = ""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            response_text = response.text or ""
        except Exception:
            response_text = "<unavailable>"
    body = getattr(exc, "body", None)
    if not response_text and body is not None:
        response_text = str(body)
    if len(response_text) > 1200:
        response_text = response_text[:1200] + "... [truncated]"
    return (
        "[SubAgent LLM debug]\n"
        f"  model={model}\n"
        f"  api_base={api_base}\n"
        f"  step={step}\n"
        f"  retry_attempt={attempt}\n"
        f"  exception_type={type(exc).__name__}\n"
        f"  status_code={status_code}\n"
        f"  request_id={request_id}\n"
        f"  messages={len(messages)}\n"
        f"  message_roles={roles}\n"
        f"  prompt_chars={prompt_chars}\n"
        f"  max_tokens=2048\n"
        f"  temperature=0.1\n"
        f"  extra_body={extra_body}\n"
        f"  response_body={response_text}"
    )


def _append_llm_error_log(agent_logs_dir: Any, text: str) -> None:
    if not agent_logs_dir:
        return
    try:
        from pathlib import Path

        log_file = Path(agent_logs_dir) / "llm_errors.log"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(text)
            f.write("\n" + "-" * 80 + "\n")
    except Exception:
        pass


def _write_subtask_boundary(
    logs_dir: Any,
    *,
    phase: str,
    tool_call_id: str | None,
    worker_model: str,
    instruction: str,
    status: str | None = None,
    steps_taken: int | None = None,
) -> None:
    if not logs_dir:
        return
    try:
        from pathlib import Path
        from datetime import datetime

        log_file = Path(logs_dir) / "commands.log"
        payload = {
            "phase": phase,
            "time": datetime.now().isoformat(timespec="seconds"),
            "tool_call_id": tool_call_id,
            "worker_model": worker_model,
            "instruction": instruction,
        }
        if status is not None:
            payload["status"] = status
        if steps_taken is not None:
            payload["steps_taken"] = steps_taken
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"[SubTask {phase}]\n")
            f.write(json.dumps(payload, ensure_ascii=False, indent=2))
            f.write("\n" + "=" * 80 + "\n")
    except Exception:
        pass


# SubAgent system prompt


SUBAGENT_SYSTEM_PROMPT = """\
You are an autonomous shell executor running INSIDE a Docker container as root.
You DIRECTLY execute commands - you are NOT a chatbot giving advice.
NEVER say "I can't run commands" - you ARE the one running them.

Each turn you receive the output of your previous command. Respond with the next command.

RULES:
- ONE command per turn. Wait for output before the next step.
- For package installs: use DEBIAN_FRONTEND=noninteractive and -y flags.
- If dpkg lock error: run `kill $(lsof -t /var/lib/dpkg/lock-frontend) 2>/dev/null; rm -f /var/lib/dpkg/lock*` first.
- Prefer pip over apt when possible (faster, fewer lock issues).
- Long commands: chain with && to avoid partial failure.
- If a command times out or fails, try a simpler alternative.
- You MUST call finish before running out of steps!

OUTPUT FORMAT:
CRITICAL: Reply with ONLY a raw JSON object. No markdown, no explanation, no prose.

To execute a command:
{"action": "execute", "params": {"command": "your shell command"}, "memory": "key findings"}

When done (or before running out of steps):
{"action": "finish", "params": {"status": "done", "completed": ["step1", "step2"], "issues": [], "message": "summary"}, "memory": "final notes"}
"""


class SubAgent:
    """Multi-turn sub-agent that executes commands in a Docker container.

    Args:
        api_base: OpenAI-compatible API endpoint for the sub-agent model.
        api_key: API key.
        max_steps: Maximum number of interaction turns.
        cmd_timeout: Timeout in seconds for each Docker command.
    """

    def __init__(
        self,
        api_base: str,
        api_key: str,
        max_steps: int = 30,
        cmd_timeout: int = 300,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.client = AsyncOpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=90,
        )
        self.max_steps = max_steps
        self.cmd_timeout = cmd_timeout

    async def run(
        self,
        model: str,
        task_instruction: str,
        original_question: str,
        executor: Any,
        agent_logs_dir: Any = None,
        tool_call_id: str | None = None,
    ) -> dict:
        """Run multi-turn SubAgent loop in Docker.

        Args:
            model: API model ID to use.
            task_instruction: Specific instruction from Planner for this subtask.
            original_question: The original TerminalBench task for reference.
            executor: DockerExecutor instance (must have execute_command method).
            agent_logs_dir: Optional path for command logs.

        Returns:
            dict with keys: status, completed, issues, message, steps_taken, model
        """
        skill_retriever = None
        task_retrieved_skills: list[dict[str, Any]] = []
        step_retrieved_skills: list[dict[str, Any]] = []
        skill_mode = _subagent_skill_mode()
        inject_skills = _subagent_skills_enabled_for_model(model)
        if inject_skills:
            try:
                from .skill_retriever import SkillRetriever

                skill_retriever = SkillRetriever.from_path(_subagent_skills_path())
                if skill_mode == "task_only":
                    task_query = f"{original_question}\n\n{task_instruction}"
                    task_retrieved_skills = skill_retriever.search(
                        task_query,
                        top_k=_subagent_skill_top_k(),
                        level="subtask",
                    )
            except Exception as e:
                logger.warning("[SubAgent] failed to load skills: %s", e)
                skill_retriever = None

        skill_block = ""
        if skill_retriever and task_retrieved_skills:
            rendered = skill_retriever.render(task_retrieved_skills)
            if rendered:
                skill_block = (
                    "\n\n== Retrieved Skills ==\n"
                    "Use these skills as execution constraints. Do not mention them in the final report unless relevant.\n"
                    f"{rendered}\n"
                )

        messages = [
            {"role": "system", "content": SUBAGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"[Step 1/{self.max_steps}]\n\n"
                    f"== Your Task ==\n{task_instruction}\n\n"
                    f"== Original Task (for reference) ==\n{original_question}\n\n"
                    f"{skill_block}"
                    "Environment ready. Begin executing commands."
                ),
            },
        ]

        finish_result = None
        steps_taken = 0
        commands_log = []
        step_logs = []
        llm_transcript = []
        prompt_tokens = 0
        completion_tokens = 0
        total_cost = 0.0

        _write_subtask_boundary(
            agent_logs_dir,
            phase="BEGIN",
            tool_call_id=tool_call_id,
            worker_model=model,
            instruction=task_instruction,
        )

        try:
            for step in range(self.max_steps):
                steps_taken = step + 1
                remaining = self.max_steps - steps_taken
                _log_step_start(
                    "llm_step_start",
                    model=model,
                    step=steps_taken,
                    max_steps=self.max_steps,
                    tool_call_id=tool_call_id,
                )

                if remaining <= 3 and step > 0:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"WARNING: Only {remaining} steps left! "
                                "Use 'finish' NOW to report your progress!"
                            ),
                        }
                    )

                raw = ""
                llm_error = None
                request_extra_body = {"enable_thinking": False}

                for _attempt in range(2):
                    transcript_entry = None
                    if _subagent_return_transcript():
                        transcript_entry = {
                            "step": steps_taken,
                            "retry_attempt": _attempt + 1,
                            "model": model,
                            "api_base": self.api_base,
                            "temperature": 0.1,
                            "max_tokens": 2048,
                            "extra_body": request_extra_body,
                            "input_messages": json.loads(json.dumps(messages, ensure_ascii=False)),
                            "started_at": _now_iso(),
                        }
                    try:
                        _dump_subagent_prompt(
                            agent_logs_dir,
                            tool_call_id=tool_call_id,
                            model=model,
                            step=steps_taken,
                            attempt=_attempt,
                            messages=messages,
                        )
                        resp = await self.client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=0.1,
                            max_tokens=2048,
                            extra_body=request_extra_body,
                        )

                        raw = resp.choices[0].message.content or ""
                        usage = getattr(resp, "usage", None)
                        step_prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                        step_completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                        if transcript_entry is not None:
                            transcript_entry.update({
                                "ended_at": _now_iso(),
                                "output_content": raw,
                                "usage": {
                                    "prompt_tokens": step_prompt_tokens,
                                    "completion_tokens": step_completion_tokens,
                                    "tokens": step_prompt_tokens + step_completion_tokens,
                                },
                            })
                            llm_transcript.append(transcript_entry)
                        prompt_tokens += step_prompt_tokens
                        completion_tokens += step_completion_tokens
                        try:
                            from eval_pipeline.config import compute_cost
                            total_cost += compute_cost(
                                model,
                                step_completion_tokens,
                                step_prompt_tokens,
                            )
                        except Exception:
                            pass
                        if _subagent_include_step_logs():
                            step_logs.append({
                                "step": steps_taken,
                                "type": "llm_response",
                                "prompt_tokens": step_prompt_tokens,
                                "completion_tokens": step_completion_tokens,
                                "content": raw,
                            })

                        _subagent_log(f"step {steps_taken} assistant", raw)
                        llm_error = None
                        break
                    except Exception as e:
                        llm_error = str(e)
                        if transcript_entry is not None:
                            transcript_entry.update({
                                "ended_at": _now_iso(),
                                "error": llm_error,
                                "exception_type": type(e).__name__,
                            })
                            llm_transcript.append(transcript_entry)
                        debug_text = _llm_exception_debug(
                            e,
                            model=model,
                            api_base=self.api_base,
                            step=steps_taken,
                            attempt=_attempt + 1,
                            messages=messages,
                            extra_body=request_extra_body,
                        )

                    logger.warning(
                        "[SubAgent step %d] LLM error (attempt %d): %s\n%s",
                        steps_taken,
                        _attempt + 1,
                        llm_error[:200],
                        debug_text,
                    )

                    if _subagent_llm_debug_enabled():
                        _append_llm_error_log(agent_logs_dir, debug_text)
                        _subagent_log(f"step {steps_taken} llm_error attempt {_attempt + 1}", debug_text)

                    if "quota" in llm_error.lower() or "balance" in llm_error.lower():
                        break

                    import asyncio as _aio
                    await _aio.sleep(2)

                if llm_error:
                    finish_result = {
                        "status": "error",
                        "completed": [],
                        "issues": [f"API error: {llm_error[:300]}"],
                        "message": f"SubAgent API failed at step {steps_taken}: {llm_error[:200]}",
                    }
                    break

                messages.append({"role": "assistant", "content": raw})

                try:
                    action = _parse_action(raw)
                    action_type = action.get("action", "") if isinstance(action, dict) else ""
                    params = action.get("params", {}) if isinstance(action, dict) else {}
                    if isinstance(params, str):
                        if action_type == "execute":
                            params = {"command": params}
                        elif action_type == "finish":
                            params = {"message": params}
                        else:
                            params = {}
                    elif not isinstance(params, dict):
                        params = {}
                except Exception as _parse_err:
                    logger.warning("[SubAgent step %d] parse failure: %s", steps_taken, _parse_err)
                    messages.append({
                        "role": "user",
                        "content": json.dumps({
                            "error": f"Could not parse your response: {str(_parse_err)[:120]}",
                            "hint": "Reply with ONLY a JSON object like "
                            '{"action":"execute","params":{"command":"..."}}',
                        }),
                    })
                    continue

                if action_type == "finish":
                    finish_result = {
                        "status": params.get("status", "done"),
                        "completed": params.get("completed", []),
                        "issues": params.get("issues", []),
                        "message": params.get("message", ""),
                    }
                    _subagent_log(f"step {steps_taken} finish", json.dumps(finish_result, ensure_ascii=False))
                    break

                if action_type == "execute":
                    command = params.get("command", "")
                    if not command:
                        messages.append({
                            "role": "user",
                            "content": json.dumps({
                                "error": "No command provided.",
                                "hint": 'Use {"action":"execute","params":{"command":"..."}}',
                            }),
                        })
                        continue

                    _subagent_log(f"step {steps_taken} command", command)
                    _log_step_start(
                        "command_start",
                        model=model,
                        step=steps_taken,
                        timeout=self.cmd_timeout,
                        tool_call_id=tool_call_id,
                        command=command[:200],
                    )
                    output, exit_code = await self._exec(executor, command)
                    _subagent_log(f"step {steps_taken} output exit={exit_code}", output[:3000])
                    commands_log.append((steps_taken, command, exit_code, output[:500]))
                    if _subagent_include_step_logs():
                        step_logs.append({
                            "step": steps_taken,
                            "type": "execute",
                            "command": command,
                            "exit_code": exit_code,
                            "output": output,
                        })
                    if agent_logs_dir:
                        self._write_cmd_log(agent_logs_dir, steps_taken, command, exit_code, output)
                    obs = {
                        "step": steps_taken,
                        "max_steps": self.max_steps,
                        "command": command,
                        "exit_code": exit_code,
                        "output": output[-2000:],
                    }
                    messages.append({"role": "user", "content": json.dumps(obs)})
                    continue

                fallback_cmd = _extract_single_command(raw)
                if fallback_cmd:
                    _subagent_log(f"step {steps_taken} fallback_command", fallback_cmd)
                    _log_step_start(
                        "fallback_command_start",
                        model=model,
                        step=steps_taken,
                        timeout=self.cmd_timeout,
                        tool_call_id=tool_call_id,
                        command=fallback_cmd[:200],
                    )
                    output, exit_code = await self._exec(executor, fallback_cmd)
                    _subagent_log(f"step {steps_taken} fallback_output exit={exit_code}", output[:3000])
                    commands_log.append((steps_taken, fallback_cmd, exit_code, output[:500]))
                    if _subagent_include_step_logs():
                        step_logs.append({
                            "step": steps_taken,
                            "type": "fallback_execute",
                            "command": fallback_cmd,
                            "exit_code": exit_code,
                            "output": output,
                        })
                    if agent_logs_dir:
                        self._write_cmd_log(agent_logs_dir, steps_taken, fallback_cmd, exit_code, output)
                    obs = {
                        "step": steps_taken,
                        "max_steps": self.max_steps,
                        "exit_code": exit_code,
                        "output": output[-2000:],
                        "note": "Your output was not valid JSON. Reply with ONLY JSON.",
                    }
                    messages.append({"role": "user", "content": json.dumps(obs)})
                else:
                    messages.append({
                        "role": "user",
                        "content": json.dumps({
                            "error": "Invalid response format. Reply with ONLY a JSON object.",
                            "hint": '{"action":"execute","params":{"command":"..."}} or '
                            '{"action":"finish","params":{"status":"done",...}}',
                        }),
                    })

        finally:
            _write_subtask_boundary(
                agent_logs_dir,
                phase="END",
                tool_call_id=tool_call_id,
                worker_model=model,
                instruction=task_instruction,
                status=(finish_result or {}).get("status") if isinstance(finish_result, dict) else None,
                steps_taken=steps_taken,
            )

        if not finish_result:
            finish_result = {
                "status": "partial",
                "completed": [c[1] for c in commands_log[-3:]] if commands_log else [],
                "issues": [
                    f"SubAgent exhausted {steps_taken}/{self.max_steps} steps without finish"
                ],
                "message": f"Ran {steps_taken} steps, {len(commands_log)} commands executed",
            }

        summary_usage = {
            "model": "",
            "cost": 0.0,
            "tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        summary_model = _subagent_summary_model()
        if summary_model:
            summarized, usage = await self._summarize_finish_result(
                model=summary_model,
                task_instruction=task_instruction,
                original_question=original_question,
                finish_result=finish_result,
                commands_log=commands_log,
            )
            if summarized:
                finish_result = summarized
                summary_usage = usage
                prompt_tokens += usage.get("prompt_tokens", 0)
                completion_tokens += usage.get("completion_tokens", 0)
                total_cost += usage.get("cost", 0.0)

        return {
            **finish_result,
            "steps_taken": steps_taken,
            "model": model,
            "summary_model": summary_model,
            "commands_log": commands_log,
            "cost": total_cost,
            "tokens": prompt_tokens + completion_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "summary_usage": summary_usage,
            "retrieved_skills": {
                "enabled": inject_skills,
                "top_k": _subagent_skill_top_k(),
                "mode": skill_mode,
                "allowed_models": [
                    m.strip()
                    for m in os.environ.get("SUBAGENT_SKILLS_MODELS", "Qwen/Qwen3-8B").split(",")
                    if m.strip()
                ],
                "task_skills": task_retrieved_skills,
                "step_skills": step_retrieved_skills,
            },
            **({"step_logs": step_logs} if _subagent_include_step_logs() else {}),
            **({"llm_transcript": llm_transcript} if _subagent_return_transcript() else {}),
        }

    async def _summarize_finish_result(
        self,
        *,
        model: str,
        task_instruction: str,
        original_question: str,
        finish_result: dict,
        commands_log: list,
    ) -> tuple[dict | None, dict]:
        """Optionally rewrite the subtask report with a fixed summary model.

        This is opt-in via SUBAGENT_SUMMARY_MODEL_ID so normal worker routing is
        unaffected. The summary model only sees a compact command tail.
        """
        usage = {
            "model": model,
            "cost": 0.0,
            "tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        client = AsyncOpenAI(
            base_url=_subagent_summary_api_base(self.api_base),
            api_key=_subagent_summary_api_key(self.api_key),
            timeout=90,
        )
        command_tail = []
        for step_n, cmd, exit_code, output in commands_log[-8:]:
            command_tail.append({
                "step": step_n,
                "command": cmd[:500],
                "exit_code": exit_code,
                "output": str(output)[-700:],
            })
        messages = [
            {
                "role": "system",
                "content": (
                    "Summarize a Docker sub-agent execution for a planner. "
                    "Return only a compact JSON object with keys: completed, issues, message. "
                    "completed and issues must be arrays of short strings. "
                    "message must be one concise paragraph. Do not invent success."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({
                    "original_task": original_question[:3000],
                    "subtask_instruction": task_instruction[:3000],
                    "raw_result": finish_result,
                    "recent_commands": command_tail,
                }, ensure_ascii=False),
            },
        ]
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=_subagent_summary_max_tokens(),
            )
            raw = resp.choices[0].message.content or ""
            summary = _parse_json_object(raw)
            completed = summary.get("completed", finish_result.get("completed", []))
            issues = summary.get("issues", finish_result.get("issues", []))
            message = summary.get("message", finish_result.get("message", ""))
            if not isinstance(completed, list):
                completed = [str(completed)]
            if not isinstance(issues, list):
                issues = [str(issues)]
            summarized = {
                **finish_result,
                "completed": [str(x)[:300] for x in completed[:8]],
                "issues": [str(x)[:300] for x in issues[:8]],
                "message": str(message)[:1200],
            }
            resp_usage = getattr(resp, "usage", None)
            prompt = getattr(resp_usage, "prompt_tokens", 0) or 0
            completion = getattr(resp_usage, "completion_tokens", 0) or 0
            usage.update({
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "tokens": prompt + completion,
            })
            try:
                from eval_pipeline.config import compute_cost
                usage["cost"] = compute_cost(model, completion, prompt)
            except Exception:
                pass
            return summarized, usage
        except Exception as e:
            logger.warning("[SubAgent summary] failed with %s: %s", model, e)
            return None, usage

    async def _exec(self, executor, command: str) -> tuple[str, int]:
        """Execute a command in Docker with timeout handling."""
        try:
            output, exit_code = await executor.execute_command(
                command,
                timeout=self.cmd_timeout,
            )
            return output[-3000:], exit_code
        except Exception as e:
            return f"Command execution error: {type(e).__name__}: {e}", -1

    @staticmethod
    def _write_cmd_log(logs_dir, step, command, exit_code, output):
        try:
            from pathlib import Path

            log_file = Path(logs_dir) / "commands.log"
            with log_file.open("a", encoding="utf-8") as f:
                f.write(
                    f"[{_now_iso()}] [Step {step}] {command}\n"
                    f"Exit: {exit_code}\n"
                    f"{output[:2000]}\n{'-' * 60}\n"
                )
        except Exception:
            pass

    @staticmethod
    def format_result_for_planner(result: dict) -> str:
        """Format SubAgent result as a string for the Planner's ToolMessage.

        The Planner sees this as the tool response from plan_subtask().
        """
        status = result.get("status", "?")
        model = result.get("model", "?")
        steps = result.get("steps_taken", 0)
        completed = result.get("completed", [])
        issues = result.get("issues", [])
        message = result.get("message", "")

        parts = [
            f"[SubAgent: {model}, {steps} steps, status={status}]",
        ]
        if completed:
            parts.append(f"Completed: {completed}")
        if issues:
            parts.append(f"Issues: {issues}")
        if message:
            parts.append(f"Message: {message}")

        # Include last few command outputs for context
        commands_log = result.get("commands_log", [])
        if commands_log:
            parts.append("Recent commands:")
            for step_n, cmd, exit_code, output in commands_log[-5:]:
                parts.append(f"  [{step_n}] $ {cmd}")
                parts.append(f"      exit={exit_code}")
                if output.strip():
                    # Truncate long outputs
                    out_lines = output.strip()[:300]
                    parts.append(f"      {out_lines}")

        return "\n".join(parts)


# Parsing helpers


def _parse_action(raw: str) -> dict:
    """Parse SubAgent JSON output with fallbacks."""
    return _parse_json_object(raw)


def _parse_json_object(raw: str) -> dict:
    """Parse a JSON object with markdown/code-fence fallbacks."""
    raw = raw.strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
        raw = raw.strip()

    # Direct JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Find JSON in text
    m = re.search(
        r'\{[^{}]*"action"\s*:\s*"[^"]+?"[^{}]*(?:\{[^{}]*\}[^{}]*)?\}',
        raw,
        re.DOTALL,
    )
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "action" in obj:
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # DISCUSSION/COMMAND format fallback
    cmd_match = re.search(r"COMMAND\s*\n(.+?)(?:\n\n|\Z)", raw, re.DOTALL)
    if cmd_match:
        cmd = cmd_match.group(1).strip().split("\n")[0].strip()
        if cmd.lower() == "finish":
            return {
                "action": "finish",
                "params": {"status": "done", "message": raw[:200]},
            }
        return {"action": "execute", "params": {"command": cmd}}

    return {"action": "unknown", "raw": raw[:500]}


def _extract_single_command(raw: str) -> str:
    """Try to extract a single executable command from free-form text."""
    # ```bash blocks
    m = re.search(r"```(?:bash|sh|shell)?\s*\n(.+?)```", raw, re.DOTALL)
    if m:
        lines = [
            l.strip()
            for l in m.group(1).strip().split("\n")
            if l.strip() and not l.strip().startswith("#")
        ]
        if lines:
            return " && ".join(lines)

    # Lines that look like commands
    for line in raw.split("\n"):
        s = line.strip()
        if (
            s
            and not s.startswith("#")
            and not s.startswith("{")
            and any(
                s.startswith(c)
                for c in [
                    "sudo",
                    "apt",
                    "pip",
                    "make",
                    "gcc",
                    "cd ",
                    "mkdir",
                    "wget",
                    "curl",
                    "git ",
                    "chmod",
                    "cp ",
                    "mv ",
                    "echo ",
                    "export",
                    "python",
                    "npm",
                    "cargo",
                    "cmake",
                    "tar ",
                    "cat ",
                    "tee ",
                    "source",
                    "dnf",
                    "yum",
                    "./",
                    "bash ",
                ]
            )
        ):
            return s
    return ""
