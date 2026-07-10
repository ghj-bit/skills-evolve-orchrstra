"""
ToolBench benchmark adapter.

Dataset: Team-ACE/ToolACE or Salesforce/xlam-function-calling-60k
Format:  Tool/function selection given a query + available tools
Verify:  Correct tool name selection (+ argument overlap)
"""
import re
import json
import os
from typing import List
from .base import BaseBenchmark, Task, VerifyResult


def _extract_function_calls(text: str) -> list[dict]:
    """Extract function/tool calls from model output."""
    calls = []

    # JSON format: {"name": "...", "arguments": {...}}
    for m in re.finditer(r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*\}', text):
        try:
            start = m.start()
            brace_count = 0
            end = start
            for i in range(start, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
            obj = json.loads(text[start:end])
            calls.append({
                "name": obj.get("name", ""),
                "arguments": obj.get("arguments", {}),
            })
        except (json.JSONDecodeError, IndexError):
            calls.append({"name": m.group(1), "arguments": {}})

    # Function call format: func_name(arg1, arg2)
    if not calls:
        for m in re.finditer(r'(\w+(?:\.\w+)*)\(([^)]*)\)', text):
            calls.append({"name": m.group(1), "arguments": m.group(2)})

    return calls


class ToolBench(BaseBenchmark):
    """ToolBench: Tool routing and function selection benchmark."""
    scoring_mode = "uno_harness"
    score_name = "Uno harness score"

    def __init__(self, dataset="Team-ACE/ToolACE", split="train",
                 max_default=1000):
        self.dataset = dataset
        self.split = split
        self.max_default = max_default

    @property
    def name(self) -> str:
        return "ToolBench"

    def load(self, max_tasks=None) -> List[Task]:
        from datasets import load_dataset, load_from_disk

        local_data_dir = os.environ.get("TOOLBENCH_DATA_DIR", "").strip()
        if local_data_dir:
            print(f"[ToolBench] loading local dataset from TOOLBENCH_DATA_DIR={local_data_dir}", flush=True)
            if not os.path.isdir(local_data_dir):
                raise FileNotFoundError(
                    f"TOOLBENCH_DATA_DIR does not exist: {local_data_dir}. "
                    "Run scripts/download_toolbench_data.sh first, or unset TOOLBENCH_DATA_DIR "
                    "to load from Hugging Face cache/remote."
                )
            ds_obj = load_from_disk(local_data_dir)
            ds = ds_obj[self.split] if hasattr(ds_obj, "keys") and self.split in ds_obj else ds_obj
        else:
            print(f"[ToolBench] loading remote/cache dataset {self.dataset} split={self.split}", flush=True)
            ds = load_dataset(self.dataset, split=self.split)

        limit = max_tasks or self.max_default
        try:
            print(f"[ToolBench] dataset rows={len(ds)} limit={limit}", flush=True)
        except TypeError:
            print(f"[ToolBench] dataset rows=unknown limit={limit}", flush=True)

        tasks = []
        for i, row in enumerate(ds):
            # ToolACE format: conversations list with system/human/gpt roles
            convos = row.get("conversations", [])
            if not convos:
                continue

            # Extract system prompt (tool definitions), user query, gold response
            system_msg = row.get("system", "") or ""
            user_query = ""
            gold_response = ""
            for msg in convos:
                role = msg.get("from", msg.get("role", ""))
                content = msg.get("value", msg.get("content", ""))
                if role == "system":
                    system_msg = content
                elif role in {"human", "user"}:
                    user_query = content
                elif role in {"gpt", "assistant"}:
                    gold_response = content

            if not user_query or not gold_response:
                continue

            # Build prompt with available tools
            prompt = ""
            if system_msg:
                prompt += f"Available tools:\n{system_msg[:3000]}\n\n"
            prompt += (
                f"User query: {user_query}\n\n"
                f"Select the appropriate tool(s) and provide the function call(s) "
                f"in JSON format: {{\"name\": \"tool_name\", \"arguments\": {{...}}}}"
            )

            tasks.append(Task(
                task_id=f"toolbench_{i}",
                raw=row,
                question=prompt,
                context={"tools": system_msg},
                gold=gold_response,
            ))
            if limit and len(tasks) >= limit:
                break
        print(
            f"[ToolBench] loaded tasks={len(tasks)} "
            f"first_task={(tasks[0].task_id if tasks else 'NONE')}",
            flush=True,
        )
        return tasks

    def extract_answer(self, router_output: str, task: Task) -> str:
        return router_output

    def verify(self, task: Task, answer: str, logs_dir=None) -> VerifyResult:
        print(f"[ToolBench] verify start task={task.task_id} answer_chars={len(answer or '')}", flush=True)
        pred_calls = _extract_function_calls(answer)
        gold_calls = _extract_function_calls(task.gold)

        if not gold_calls:
            # Gold is text, not a tool call (e.g., refusal)
            gold_norm = task.gold.lower().strip()
            pred_norm = answer.lower().strip()
            # Both refusals
            refusal_kw = ["cannot", "can't", "don't have", "unable", "not available"]
            if any(k in gold_norm for k in refusal_kw) and any(k in pred_norm for k in refusal_kw):
                result = VerifyResult(task.task_id, 1.0, log="both_refusal")
                print(f"[ToolBench] verify done task={task.task_id} reward={result.reward} log={result.log}", flush=True)
                return result
            if gold_norm in pred_norm or pred_norm in gold_norm:
                result = VerifyResult(task.task_id, 1.0, log="text_match")
                print(f"[ToolBench] verify done task={task.task_id} reward={result.reward} log={result.log}", flush=True)
                return result
            result = VerifyResult(task.task_id, 0.0, log="no_gold_calls")
            print(f"[ToolBench] verify done task={task.task_id} reward={result.reward} log={result.log}", flush=True)
            return result

        if not pred_calls:
            result = VerifyResult(task.task_id, 0.0, log="no_pred_calls")
            print(f"[ToolBench] verify done task={task.task_id} reward={result.reward} log={result.log}", flush=True)
            return result

        # Tool name overlap
        pred_names = {c["name"].lower() for c in pred_calls}
        gold_names = {c["name"].lower() for c in gold_calls}
        overlap = len(pred_names & gold_names) / len(gold_names)
        correct = overlap >= 0.5

        result = VerifyResult(
            task.task_id, 1.0 if correct else 0.0,
            log=f"overlap={overlap:.2f} pred={pred_names} gold={gold_names}",
        )
        print(f"[ToolBench] verify done task={task.task_id} reward={result.reward} log={result.log}", flush=True)
        return result
