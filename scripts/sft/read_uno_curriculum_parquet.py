"""Inspect and export parquet files from the Uno-Curriculum dataset.

Examples:
    python scripts/sft/read_uno_curriculum_parquet.py
    python scripts/sft/read_uno_curriculum_parquet.py --config sft_traj --limit 3
    python scripts/sft/read_uno_curriculum_parquet.py --config sft_full --export-jsonl tmp/sft_full_sample.jsonl --limit 100
    python scripts/sft/read_uno_curriculum_parquet.py --export-aligned-long-task-json-dir tmp/uno_long_task
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "Uno-Curriculum"
CONFIGS = ("sft_full", "sft_traj", "sft_subtasks")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Uno-Curriculum root directory. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--config",
        choices=CONFIGS,
        help="Only read one config. By default all known configs are inspected.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Number of rows to load/print per parquet file. Use 0 to skip samples.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        help="Optional subset of columns to load for samples or export.",
    )
    parser.add_argument(
        "--export-jsonl",
        type=Path,
        help="Write loaded rows to JSONL. Honors --config, --columns, and --limit.",
    )
    parser.add_argument(
        "--full-export",
        action="store_true",
        help="When exporting, read all rows instead of --limit rows.",
    )
    parser.add_argument(
        "--max-cell-chars",
        type=int,
        default=500,
        help="Maximum characters printed for each sampled cell.",
    )
    parser.add_argument(
        "--export-aligned-long-task-json-dir",
        type=Path,
        help=(
            "Find one long sft_traj task that also has matching sft_full and sft_subtasks rows, "
            "then write three JSON files under this output directory."
        ),
    )
    return parser.parse_args()


def parquet_paths(root: Path, config: str | None) -> list[Path]:
    names = [config] if config else list(CONFIGS)
    paths = [root / name / "train.parquet" for name in names]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing parquet file(s):\n" + "\n".join(missing))
    return paths


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "as_py"):
        return to_jsonable(value.as_py())
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def normalize_question(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"^Question:\s*", "", text)
    text = re.split(r"\n\nCorrect answer \(for your reference; arrive at this through reasoning\):", text)[0]
    return re.sub(r"\s+", " ", text).strip().lower()


def shorten(value: Any, max_chars: int) -> str:
    text = json.dumps(to_jsonable(value), ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def print_metadata(path: Path) -> None:
    pf = pq.ParquetFile(path)
    print(f"\n=== {path.parent.name} ===")
    print(f"path: {path}")
    print(f"rows: {pf.metadata.num_rows:,}")
    print(f"row_groups: {pf.metadata.num_row_groups}")
    print("schema:")
    print(pf.schema_arrow)


def read_rows(path: Path, columns: list[str] | None, limit: int | None) -> pd.DataFrame:
    if limit == 0:
        return pd.DataFrame()
    if limit is None:
        return pd.read_parquet(path, columns=columns)

    pf = pq.ParquetFile(path)
    try:
        batch = next(pf.iter_batches(batch_size=max(limit, 1), columns=columns))
    except StopIteration:
        return pd.DataFrame()
    return batch.to_pandas().head(limit)


def print_quick_stats(df: pd.DataFrame) -> None:
    for column in ("source", "category", "domain", "strategy", "distillation_pass", "routed_model", "routed_skill"):
        if column in df.columns:
            counts = Counter(str(v) for v in df[column].dropna())
            top = ", ".join(f"{k}={v}" for k, v in counts.most_common(8))
            print(f"{column}: {top}")


def print_samples(df: pd.DataFrame, max_cell_chars: int) -> None:
    if df.empty:
        return
    print("sample rows:")
    for row_idx, row in df.iterrows():
        print(f"- row {row_idx}:")
        for column, value in row.items():
            print(f"  {column}: {shorten(value, max_cell_chars)}")


def export_jsonl(path: Path, frames: list[tuple[str, pd.DataFrame]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _, df in frames:
            for row in df.to_dict(orient="records"):
                row = {k: to_jsonable(v) for k, v in row.items()}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nexported: {path}")


def row_to_json_dict(row: pd.Series) -> dict[str, Any]:
    return {k: to_jsonable(v) for k, v in row.to_dict().items()}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"exported: {path}")


def find_aligned_long_task(root: Path) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    traj_meta = pd.read_parquet(
        root / "sft_traj" / "train.parquet",
        columns=["id", "source", "question", "n_delegates", "n_turns", "strategy"],
    )
    full_meta = pd.read_parquet(
        root / "sft_full" / "train.parquet",
        columns=["id", "source", "question", "n_subtasks", "n_plan_rounds"],
    )
    sub_meta = pd.read_parquet(
        root / "sft_subtasks" / "train.parquet",
        columns=["trajectory_id", "subtask_id"],
    )

    full_by_question: dict[str, list[int]] = {}
    for row_idx, row in full_meta.iterrows():
        full_by_question.setdefault(normalize_question(row["question"]), []).append(row_idx)

    subtask_counts = Counter(str(v) for v in sub_meta["trajectory_id"].dropna())
    ranked_traj = traj_meta.sort_values(
        ["n_delegates", "n_turns"],
        ascending=False,
        kind="mergesort",
    )

    for _, traj in ranked_traj.iterrows():
        trajectory_id = str(traj["id"])
        if subtask_counts.get(trajectory_id, 0) == 0:
            continue

        full_matches = full_by_question.get(normalize_question(traj["question"]), [])
        if not full_matches:
            continue

        full_candidates = full_meta.loc[full_matches].copy()
        full_candidates["_same_source"] = full_candidates["source"].astype(str).eq(str(traj["source"]))
        full_candidates = full_candidates.sort_values(
            ["_same_source", "n_subtasks", "n_plan_rounds"],
            ascending=False,
            kind="mergesort",
        )
        full = full_candidates.iloc[0]

        traj_row = pd.read_parquet(root / "sft_traj" / "train.parquet").loc[lambda df: df["id"] == trajectory_id].iloc[0]
        full_row = pd.read_parquet(root / "sft_full" / "train.parquet").loc[lambda df: df["id"] == full["id"]].iloc[0]
        sub_rows = pd.read_parquet(root / "sft_subtasks" / "train.parquet").loc[
            lambda df: df["trajectory_id"] == trajectory_id
        ]
        sub_rows = sub_rows.sort_values("subtask_order", kind="mergesort")
        return full_row, traj_row, sub_rows

    raise RuntimeError("Could not find an aligned long task across sft_full, sft_traj, and sft_subtasks.")


def export_aligned_long_task(root: Path, out_dir: Path) -> None:
    full_row, traj_row, sub_rows = find_aligned_long_task(root)
    question_key = normalize_question(traj_row["question"])
    alignment = {
        "alignment_note": (
            "sft_traj and sft_subtasks are joined by sft_traj.id == sft_subtasks.trajectory_id; "
            "sft_full is matched by normalized question text because its row id can differ."
        ),
        "normalized_question": question_key,
        "sft_full_id": full_row["id"],
        "sft_traj_id": traj_row["id"],
        "sft_subtasks_trajectory_id": traj_row["id"],
        "sft_subtasks_count": len(sub_rows),
    }

    write_json(out_dir / "manifest.json", alignment)
    write_json(out_dir / "sft_full" / "trajectory.json", row_to_json_dict(full_row))
    write_json(out_dir / "sft_traj" / "trajectory.json", row_to_json_dict(traj_row))
    write_json(
        out_dir / "sft_subtasks" / "trajectory.json",
        [{k: to_jsonable(v) for k, v in row.items()} for row in sub_rows.to_dict(orient="records")],
    )


def main() -> None:
    args = parse_args()

    if args.export_aligned_long_task_json_dir:
        export_aligned_long_task(args.root, args.export_aligned_long_task_json_dir)
        return

    paths = parquet_paths(args.root, args.config)
    export_frames: list[tuple[str, pd.DataFrame]] = []

    for path in paths:
        print_metadata(path)

        read_limit = None if args.export_jsonl and args.full_export else args.limit
        df = read_rows(path, args.columns, read_limit)

        if not df.empty:
            print_quick_stats(df)
            print_samples(df, args.max_cell_chars)

        if args.export_jsonl:
            export_frames.append((path.parent.name, df))

    if args.export_jsonl:
        export_jsonl(args.export_jsonl, export_frames)


if __name__ == "__main__":
    main()
