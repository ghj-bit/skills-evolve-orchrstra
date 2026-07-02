"""
Direct prompting baseline — single model, no routing.
Tests: "Is routing worth it at all?"
"""
import os

import openai
from .base import BaseRouter, RouteResult
from ..config import EVAL_MAX_TOKENS, DEFAULT_API_BASE, compute_cost


def _env_enabled(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def _truncate(value, limit: int = 2000) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


class DirectRouter(BaseRouter):
    """No routing: send question directly to a single model."""

    def __init__(self, model_id: str, api_base=DEFAULT_API_BASE, api_key="EMPTY",
                 system_prompt="You are a helpful assistant."):
        self.model_id = model_id
        self.api_base = api_base
        timeout = float(os.environ.get("DIRECT_ROUTER_TIMEOUT", "120"))
        max_retries = int(os.environ.get("DIRECT_ROUTER_MAX_RETRIES", "2"))
        self.api = openai.OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.system_prompt = system_prompt

    def _print_debug(self, exc: Exception, messages, tools=None, call_kw=None) -> None:
        if not _env_enabled("DIRECT_ROUTER_DEBUG"):
            return

        response = getattr(exc, "response", None)
        response_text = ""
        if response is not None:
            try:
                response_text = response.text
            except Exception:
                response_text = ""
        body = getattr(exc, "body", None)
        roles = ",".join(str(m.get("role", "?")) for m in (messages or []))
        prompt_chars = sum(len(str(m.get("content") or "")) for m in (messages or []))
        tool_names = []
        for tool in tools or []:
            fn = tool.get("function", {}) if isinstance(tool, dict) else {}
            tool_names.append(fn.get("name", "?"))

        print("[DirectRouter LLM debug]", flush=True)
        print(f"  model={self.model_id}", flush=True)
        print(f"  api_base={self.api_base}", flush=True)
        print(f"  exception_type={type(exc).__name__}", flush=True)
        print(f"  status_code={getattr(exc, 'status_code', None)}", flush=True)
        print(f"  request_id={getattr(exc, 'request_id', None)}", flush=True)
        print(f"  messages={len(messages or [])}", flush=True)
        print(f"  message_roles={roles}", flush=True)
        print(f"  prompt_chars={prompt_chars}", flush=True)
        print(f"  tools={tool_names}", flush=True)
        if call_kw:
            print(f"  max_tokens={call_kw.get('max_tokens')}", flush=True)
            print(f"  temperature={call_kw.get('temperature')}", flush=True)
            print(f"  tool_choice={call_kw.get('tool_choice')}", flush=True)
        print(f"  response_body={_truncate(body or response_text)}", flush=True)

    @property
    def name(self):
        return f"Direct({self.model_id})"

    def route(self, question: str, context: dict = None) -> RouteResult:
        try:
            r = self.api.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=0.0, max_tokens=EVAL_MAX_TOKENS,
            )
            txt = r.choices[0].message.content or ""
            toks = getattr(r.usage, "completion_tokens", 0) or 0
            prompt_toks = getattr(r.usage, "prompt_tokens", 0) or 0
            cost = compute_cost(self.model_id, toks, prompt_toks)
            return RouteResult(answer=txt, route_count=0, routed_models=[self.model_id],
                               total_cost=cost, total_tokens=toks + prompt_toks,
                               prompt_tokens=prompt_toks, completion_tokens=toks)
        except Exception as e:
            self._print_debug(e, [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ])
            return RouteResult(answer=f"Error: {e}", route_count=0)

    def chat_completions(self, messages, tools=None, **kw):
        """Pass-through to the underlying model, with optional tool definitions."""
        call_kw = dict(
            model=self.model_id,
            messages=messages,
            temperature=kw.get("temperature", 0.0),
            max_tokens=kw.get("max_tokens", EVAL_MAX_TOKENS),
        )
        if tools:
            call_kw["tools"] = tools
            call_kw["tool_choice"] = kw.get("tool_choice", "auto")
        try:
            r = self.api.chat.completions.create(**call_kw)
        except Exception as e:
            self._print_debug(e, messages, tools=tools, call_kw=call_kw)
            raise
        msg = r.choices[0].message
        tool_calls = []
        import json as _json
        for tc in (msg.tool_calls or []):
            try:
                args = _json.loads(tc.function.arguments) if tc.function.arguments else {}
            except Exception:
                args = {"_raw": tc.function.arguments}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        return {
            "content": msg.content,
            "tool_calls": tool_calls,
            "completion_tokens": getattr(r.usage, "completion_tokens", 0) or 0,
            "prompt_tokens": getattr(r.usage, "prompt_tokens", 0) or 0,
            "model": self.model_id,
        }
