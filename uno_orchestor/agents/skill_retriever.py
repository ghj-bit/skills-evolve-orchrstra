"""Lightweight skill retrieval for Terminal-Bench sub-agents.

This intentionally avoids external embedding dependencies. It mirrors the
Routing_0421 pattern of turning each skill into retrieval text, then selecting
top-k skills for a task or step query.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


_TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1]


def _flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_flatten(v) for v in value)
    if isinstance(value, dict):
        return " ".join(_flatten(v) for v in value.values())
    return str(value)


def skill_to_retrieval_text(skill: dict[str, Any]) -> str:
    parts = []
    for field in (
        "name",
        "level",
        "description",
        "application_scenarios",
        "execution_principles",
        "workflow",
        "example",
        "failure_cases_prevented",
    ):
        val = _flatten(skill.get(field))
        if val:
            parts.append(val)
    source = skill.get("source")
    if isinstance(source, dict):
        for field in ("task_id", "trajectory_diagnosis"):
            val = _flatten(source.get(field))
            if val:
                parts.append(val)
    return ". ".join(parts)


class SkillRetriever:
    def __init__(self, skills: list[dict[str, Any]]):
        self.skills = skills
        self.docs = [skill_to_retrieval_text(s) for s in skills]
        self.doc_tokens = [Counter(_tokens(doc)) for doc in self.docs]
        doc_freq = Counter()
        for toks in self.doc_tokens:
            doc_freq.update(toks.keys())
        n_docs = max(len(self.doc_tokens), 1)
        self.idf = {
            tok: math.log((1 + n_docs) / (1 + freq)) + 1.0
            for tok, freq in doc_freq.items()
        }

    @classmethod
    def from_path(cls, path: str | Path) -> "SkillRetriever":
        with Path(path).open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict):
            skills = (data.get("task_skills") or []) + (data.get("step_skills") or [])
        elif isinstance(data, list):
            skills = data
        else:
            skills = []
        return cls([s for s in skills if isinstance(s, dict)])

    def search(
        self,
        query: str,
        *,
        top_k: int = 2,
        level: str | None = None,
    ) -> list[dict[str, Any]]:
        query_tokens = Counter(_tokens(query))
        if not query_tokens:
            return []
        rows = []
        for idx, (skill, doc_counter) in enumerate(zip(self.skills, self.doc_tokens)):
            if level and skill.get("level") not in {level, "task", "subtask"}:
                continue
            score = 0.0
            for tok, q_count in query_tokens.items():
                if tok in doc_counter:
                    score += min(q_count, doc_counter[tok]) * self.idf.get(tok, 1.0)
            if score <= 0:
                continue
            norm = math.sqrt(sum(v * v for v in doc_counter.values())) or 1.0
            rows.append({
                "rank": 0,
                "name": skill.get("name", f"skill_{idx}"),
                "execution_principles": skill.get("execution_principles", []),
                "workflow": skill.get("workflow", []),
                "retrieval_score": score / norm,
            })
        rows.sort(key=lambda x: x["retrieval_score"], reverse=True)
        for rank, row in enumerate(rows[:top_k], start=1):
            row["rank"] = rank
        return rows[:top_k]

    @staticmethod
    def render(skills: list[dict[str, Any]]) -> str:
        if not skills:
            return ""
        blocks = []
        for skill in skills:
            lines = [f"Name: {skill.get('name')}"]
            principles = skill.get("execution_principles") or []
            if principles:
                lines.append("Principles: " + "; ".join(str(x) for x in principles[:5]))
            workflow = skill.get("workflow") or []
            if workflow:
                lines.append("Workflow: " + " -> ".join(str(x) for x in workflow[:7]))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
