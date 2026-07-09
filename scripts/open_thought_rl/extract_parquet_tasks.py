#!/usr/bin/env python3
"""
Simple script to extract tasks from a parquet file.
Usage: python extract_parquet_tasks.py <parquet_path> <output_dir>
"""

import io
import os
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath

try:
    import pyarrow.parquet as pq
except ImportError:
    print("Error: pyarrow is required. Install with: pip install pyarrow")
    sys.exit(1)


def _is_within(base: Path, target: Path) -> bool:
    try:
        return os.path.commonpath([str(base.resolve()), str(target.resolve())]) == str(base.resolve())
    except Exception:
        return False


def _sanitize_tar_member_name(name: str) -> str:
    p = PurePosixPath(name)
    parts = [part for part in p.parts if part not in ("..", ".", "")]
    while parts and parts[0] == "/":
        parts.pop(0)
    return str(PurePosixPath(*parts)) if parts else ""


def safe_extract_tar(archive_bytes: bytes, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO(archive_bytes)
    with tarfile.open(fileobj=buf, mode="r:*") as tf:
        for member in tf.getmembers():
            member_name = _sanitize_tar_member_name(member.name)
            if not member_name or member_name.endswith("/"):
                (dest_dir / member_name).mkdir(parents=True, exist_ok=True)
                continue
            if ".snapshot" in PurePosixPath(member_name).parts:
                continue
            target = (dest_dir / member_name).resolve()
            if not _is_within(dest_dir, target):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.isfile():
                with tf.extractfile(member) as src:
                    if src is None:
                        continue
                    with open(target, "wb") as dst:
                        dst.write(src.read())
            elif member.isdir():
                target.mkdir(parents=True, exist_ok=True)


def from_parquet(parquet_path: str, base: str, on_exist: str = "overwrite") -> list[Path]:
    """Extract tasks from parquet file to directory."""
    table = pq.read_table(parquet_path)
    cols = {name: i for i, name in enumerate(table.column_names)}
    
    print(f"Parquet columns: {list(cols.keys())}")
    
    if "path" not in cols or "task_binary" not in cols:
        raise RuntimeError(f"Parquet must have columns: 'path', 'task_binary'. Found: {list(cols.keys())}")

    base = Path(base).resolve()
    base.mkdir(parents=True, exist_ok=True)
    
    path_col = table.column(cols["path"]).to_pylist()
    data_col = table.column(cols["task_binary"]).to_pylist()

    written: list[Path] = []
    total = len(path_col)
    
    for i, (rel_path, data) in enumerate(zip(path_col, data_col)):
        if i % 100 == 0:
            print(f"Processing {i}/{total}...")
        
        if not isinstance(rel_path, str):
            print(f"Warning: Row {i}: 'path' must be a string, skipping")
            continue
        if not isinstance(data, (bytes, bytearray, memoryview)):
            print(f"Warning: Row {i}: 'task_binary' must be bytes, skipping")
            continue

        safe_rel = PurePosixPath(rel_path)
        parts = [p for p in safe_rel.parts if p not in ("..", "")]
        rel_norm = Path(*parts) if parts else Path(f"task_{i}")
        target_dir = (base / rel_norm).resolve()
        
        if not _is_within(base, target_dir):
            print(f"Warning: Unsafe target path: {rel_path}, skipping")
            continue

        if target_dir.exists():
            if on_exist == "skip":
                continue
            if on_exist == "error":
                raise FileExistsError(f"Target exists: {target_dir}")
            if on_exist == "overwrite":
                if target_dir.is_dir():
                    shutil.rmtree(target_dir)
                else:
                    target_dir.unlink()

        try:
            safe_extract_tar(bytes(data), target_dir)
            written.append(target_dir)
        except Exception as e:
            print(f"Warning: Failed to extract {rel_path}: {e}")

    print(f"Successfully extracted {len(written)} tasks to {base}")
    return written


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_parquet_tasks.py <parquet_path> <output_dir> [--on-exist skip|overwrite|error]")
        print("\nExample:")
        print("  python extract_parquet_tasks.py train-00000-of-00001.parquet ./extracted_tasks")
        sys.exit(1)
    
    parquet_path = sys.argv[1]
    output_dir = sys.argv[2]
    on_exist = "overwrite"
    
    if "--on-exist" in sys.argv:
        idx = sys.argv.index("--on-exist")
        if idx + 1 < len(sys.argv):
            on_exist = sys.argv[idx + 1]
    
    if not os.path.exists(parquet_path):
        print(f"Error: Parquet file not found: {parquet_path}")
        sys.exit(1)
    
    from_parquet(parquet_path, output_dir, on_exist)

