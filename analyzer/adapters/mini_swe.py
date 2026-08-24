"""把 mini-swe-agent 轨迹转换为 evolve 的 case。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..helpers import load_json, write_json


_THINKING_FIELDS = ("reasoning_content", "thinking", "reasoning")


def extract_case(traj: dict[str, Any]) -> dict[str, Any]:
    """提取一条轨迹的任务、步骤、工具调用和终态。"""
    info = traj.get("info") or {}
    model_stats = info.get("model_stats") or {}
    steps: list[dict[str, Any]] = []
    pending_calls: dict[str, dict[str, Any]] = {}
    task: str | None = None

    for message in traj.get("messages") or []:
        role = message.get("role")
        if role == "user" and task is None:
            task = message.get("content") or ""
            continue
        if role == "assistant":
            step = {
                "index": len(steps) + 1,
                "thinking": _thinking(message),
                "tool_calls": _extract_tool_calls(message, len(steps) + 1),
            }
            steps.append(step)
            for call in step["tool_calls"]:
                pending_calls[call["id"]] = call
            continue
        if role == "tool":
            call = pending_calls.pop(message.get("tool_call_id"), None)
            if call is None:
                raise ValueError(f"工具结果没有对应调用: {message.get('tool_call_id')}")
            content = message.get("content") or ""
            call["observation"] = _observation(content)
            call["returncode"] = _returncode(content)

    instance_id = traj.get("instance_id") or info.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("轨迹缺少 instance_id")
    case: dict[str, Any] = {
        "instance_id": instance_id,
        "task": task,
        "exit_status": info.get("exit_status"),
        "submission": info.get("submission"),
        "steps": steps,
    }
    if "cost" in info:
        case["cost"] = info["cost"]
    elif "instance_cost" in model_stats:
        case["cost"] = model_stats["instance_cost"]
    if "api_calls" in model_stats:
        case["api_calls"] = model_stats["api_calls"]
    if info.get("environment_image") is not None:
        case["env"] = {"environment_image": info["environment_image"]}
    return case


def extract_trajectory_path(path: str | Path) -> dict[str, Any]:
    """读取并转换一条 ``.traj.json``。"""
    return extract_case(load_json(Path(path)))


def iter_trajectory_paths(traj_dir: str | Path) -> list[Path]:
    """发现目录下的全部轨迹文件。"""
    root = Path(traj_dir)
    return sorted({path.resolve() for path in root.rglob("*.traj.json") if path.is_file()})


def write_experiment(traj_dir: str | Path, out_dir: str | Path) -> list[dict[str, Any]]:
    """转换整批轨迹并只写入 case。"""
    traj_root = Path(traj_dir).resolve()
    experiment_dir = Path(out_dir)
    cases: list[dict[str, Any]] = []
    for path in iter_trajectory_paths(traj_root):
        case = extract_trajectory_path(path)
        case["source"] = {
            "format": "mini-swe-agent",
            "traj_relpath": path.relative_to(traj_root).as_posix(),
        }
        write_json(experiment_dir / "cases" / f"{case['instance_id']}.json", case)
        cases.append(case)
    return cases


def _extract_tool_calls(message: dict[str, Any], step_index: int) -> list[dict[str, Any]]:
    """提取 assistant 消息中的全部工具调用。"""
    raw_calls = message.get("tool_calls") or []
    if raw_calls:
        return [_tool_call(raw, step_index, position) for position, raw in enumerate(raw_calls)]
    return [
        {
            "id": str(raw.get("tool_call_id") or f"step-{step_index}-call-{position}"),
            "name": "bash",
            "arguments": {"command": raw["command"]},
            "observation": None,
            "returncode": None,
        }
        for position, raw in enumerate((message.get("extra") or {}).get("actions") or [])
        if isinstance(raw.get("command"), str)
    ]


def _tool_call(raw: dict[str, Any], step_index: int, position: int) -> dict[str, Any]:
    """归一化一个 OpenAI 兼容工具调用。"""
    function = raw.get("function") or {}
    return {
        "id": str(raw.get("id") or f"step-{step_index}-call-{position}"),
        "name": str(function.get("name") or "unknown"),
        "arguments": _arguments(function.get("arguments")),
        "observation": None,
        "returncode": None,
    }


def _arguments(value: Any) -> Any:
    """保留工具参数的 JSON 结构；无法解析时保留原字符串。"""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _thinking(message: dict[str, Any]) -> str | None:
    """提取模型的第一个非空推理字段。"""
    for field in _THINKING_FIELDS:
        value = message.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _observation(content: str) -> str:
    """去掉 mini-swe 输出包装。"""
    match = re.search(r"<output>(.*?)</output>", content, re.S)
    return match.group(1) if match else content


def _returncode(content: str) -> int | None:
    """提取命令退出码。"""
    match = re.search(r"<returncode>(-?\d+)</returncode>", content)
    return int(match.group(1)) if match else None
