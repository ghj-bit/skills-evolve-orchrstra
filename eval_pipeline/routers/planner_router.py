"""Paper-style Uno evaluation router entrypoint.

The Uno architecture uses a unified policy that emits decomposition and routing
decisions in one assistant stream, so this class reuses the same schema/harness
path as ``UnoSFT``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import openai

from .router_sft import UnoSFT
from ..config import DEFAULT_API_BASE, DEFAULT_LOCAL_BASE, EVAL_MAX_TOKENS


def _write_router_io_log(record: dict) -> None:
    log_path = os.environ.get("UNO_ROUTER_IO_LOG", "").strip()
    if not log_path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return


def _available_tool_names(tools) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            names.add(name)
    return names


def _loads_first_json_object_once(text: str):
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            return obj
        except json.JSONDecodeError:
            continue
    return None


def _decode_escaped_text(text: str) -> str:
    try:
        return json.loads('"' + text.replace("\r", "\\r").replace("\n", "\\n") + '"')
    except json.JSONDecodeError:
        return text


def _loads_first_json_object(text: str):
    parsed = _loads_first_json_object_once(text)
    if parsed is not None:
        return parsed
    decoded = _decode_escaped_text(text)
    if decoded != text:
        return _loads_first_json_object_once(decoded)
    return None


def _parse_text_tool_calls(content: str | None, tools=None) -> list[dict]:
    """Fallback for models that print a tool call instead of using OpenAI tools."""
    if not content:
        return []
    parsed = _loads_first_json_object(content)
    if not isinstance(parsed, dict):
        return []

    calls = parsed.get("tool_calls") if isinstance(parsed.get("tool_calls"), list) else [parsed]
    allowed_names = _available_tool_names(tools)
    tool_calls = []
    for idx, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        args = call.get("arguments", {})
        if not name or (allowed_names and name not in allowed_names):
            continue
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {"_raw": args}
        tool_calls.append({"id": call.get("id", f"text_tool_call_{idx}"), "name": name, "arguments": args})
    return tool_calls


def _is_deepseek_planner(model: str, api_base: str) -> bool:
    model_l = (model or "").lower()
    api_base_l = (api_base or "").lower()
    return "deepseek" in model_l or "deepseek" in api_base_l


class PlannerRouter(UnoSFT):
    """Paper-style unified Uno router."""

    def __init__(
        self,
        planner_model: str = "Qwen/Qwen2.5-7B-Instruct",
        router_model: str | None = None,
        planner_api_base: str = DEFAULT_LOCAL_BASE,
        router_api_base: str | None = None,
        sub_model_api_base: str = DEFAULT_API_BASE,
        planner_api_key: str = "EMPTY",
        router_api_key: str = "EMPTY",
        sub_model_api_key: str = "EMPTY",
        planner_temperature: float = 0.0,
        router_temperature: float = 0.0,
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        model_name = router_model or planner_model
        local_base = router_api_base or planner_api_base
        worker_api_base = sub_model_api_base or api_base or DEFAULT_API_BASE
        worker_api_key = sub_model_api_key or api_key or "EMPTY"
        super().__init__(
            local_base=local_base,
            api_base=worker_api_base,
            api_key=worker_api_key,
            model_name=model_name,
            local_api_key=router_api_key,
        )
        self.sub_model_api_base = worker_api_base
        self.sub_model_api_key = worker_api_key
        self.planner_model = planner_model
        self.router_model = model_name
        self.planner_temperature = planner_temperature
        self.router_temperature = router_temperature
        self.planner_api_base = planner_api_base
        self.chat_client = openai.OpenAI(base_url=planner_api_base, api_key=planner_api_key)

    @property
    def name(self) -> str:
        p = self.planner_model.split("/")[-1]
        r = self.router_model.split("/")[-1]
        return f"UnoRouter({p})" if p == r else f"UnoRouter(P={p},R={r})"

    def chat_completions(self, messages, tools=None, **kw):
        """Expose raw chat completions for interactive Docker benchmarks."""
        call_kw = dict(
            model=self.planner_model,
            messages=messages,
            temperature=kw.get("temperature", self.planner_temperature),
            max_tokens=kw.get("max_tokens", EVAL_MAX_TOKENS),
        )
        if tools:
            call_kw["tools"] = tools
            call_kw["tool_choice"] = kw.get("tool_choice", "auto")
        if _is_deepseek_planner(self.planner_model, self.planner_api_base):
            call_kw["extra_body"] = {"thinking": {"type": "disabled"}}

        request_started_at = datetime.now(timezone.utc).isoformat()
        request_meta = {k: v for k, v in call_kw.items() if k not in {"messages", "tools"}}
        try:
            r = self.chat_client.chat.completions.create(**call_kw)
        except Exception as exc:
            _write_router_io_log(
                {
                    "timestamp": request_started_at,
                    "event": "error",
                    "model": self.planner_model,
                    "request": request_meta,
                    "messages": messages,
                    "tools": tools or [],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            raise

        msg = r.choices[0].message
        tool_calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except Exception:
                args = {"_raw": tc.function.arguments}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        if not tool_calls:
            tool_calls = _parse_text_tool_calls(msg.content, tools=tools)
        _write_router_io_log(
            {
                "timestamp": request_started_at,
                "event": "response",
                "model": self.planner_model,
                "request": request_meta,
                "messages": messages,
                "tools": tools or [],
                "response": {
                    "content": msg.content,
                    "tool_calls": tool_calls,
                    "completion_tokens": getattr(r.usage, "completion_tokens", 0) or 0,
                    "prompt_tokens": getattr(r.usage, "prompt_tokens", 0) or 0,
                    "model": getattr(r, "model", self.planner_model),
                },
            }
        )
        return {
            "content": msg.content,
            "tool_calls": tool_calls,
            "completion_tokens": getattr(r.usage, "completion_tokens", 0) or 0,
            "prompt_tokens": getattr(r.usage, "prompt_tokens", 0) or 0,
            "model": self.planner_model,
        }

