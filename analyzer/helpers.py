"""evolve 分析产物的最小 JSON 读写工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    """读取一个 JSON 对象。"""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return value


def load_index(experiment_dir: Path) -> dict[str, Any]:
    """读取实验索引。"""
    return load_json(experiment_dir / "index.json")


def load_case(experiment_dir: Path, instance_id: str) -> dict[str, Any]:
    """读取一条 case。"""
    return load_json(experiment_dir / "cases" / f"{instance_id}.json")


def iter_cases(experiment_dir: Path) -> list[dict[str, Any]]:
    """按文件名顺序读取实验中的全部 case。"""
    return [load_json(path) for path in sorted((experiment_dir / "cases").glob("*.json"))]


def write_json(path: Path, value: dict[str, Any]) -> None:
    """以稳定格式写入 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def truncate_text(text: str, max_chars: int) -> str:
    """超长文本保留头尾，并标记省略的字符数。"""
    if max_chars < 2:
        raise ValueError("max_chars 必须至少为 2")
    if len(text) <= max_chars:
        return text
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    omitted = len(text) - max_chars
    return f"{text[:head_chars]}\n… [中间省略 {omitted} 字符] …\n{text[-tail_chars:]}"
