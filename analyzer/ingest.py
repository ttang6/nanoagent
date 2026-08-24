"""摄入 mini-swe 轨迹并关联官方评测结果。"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from .adapters.mini_swe import iter_trajectory_paths, write_experiment
from .helpers import load_json, write_json


_SENSITIVE_CONFIG_KEY = re.compile(r"api[_-]?key|token|secret|password|credential|authorization", re.I)


def run_ingest(
    source_run_dir: Path,
    experiment_dir: Path,
    evaluation_report_path: Path,
    source_git_branch: str | None = None,
    benchmark_name: str = "",
) -> None:
    """摄入轨迹并将官方 grading 回填到索引。"""
    cases = write_experiment(source_run_dir, experiment_dir)
    grading_counts = _apply_grading(experiment_dir, evaluation_report_path)
    write_json(
        experiment_dir / "meta.json",
        _meta(source_run_dir, evaluation_report_path, source_git_branch, benchmark_name, len(cases)),
    )
    write_json(experiment_dir / "summary.json", _summary(cases, grading_counts))
    print(f"ingest: {len(cases)} cases -> {experiment_dir}")


def _apply_grading(experiment_dir: Path, evaluation_report_path: Path) -> dict[str, int]:
    """将评测报告的 outcome 写回每一条 case。"""
    report = load_json(evaluation_report_path)
    grading = {
        instance_id: outcome
        for report_field, outcome in (
            ("resolved_ids", "resolved"),
            ("unresolved_ids", "unresolved"),
            ("empty_patch_ids", "empty_patch"),
            ("error_ids", "error"),
        )
        for instance_id in report.get(report_field) or []
    }
    for case_path in sorted((experiment_dir / "cases").glob("*.json")):
        case = load_json(case_path)
        instance_id = case["instance_id"]
        if instance_id not in grading:
            raise ValueError(f"评测报告缺少 instance: {instance_id}")
        case["grading"] = grading[instance_id]
        write_json(case_path, case)
    return dict(Counter(grading.values()))


def _meta(
    source_run_dir: Path,
    evaluation_report_path: Path,
    source_git_branch: str | None,
    benchmark_name: str,
    case_count: int,
) -> dict[str, Any]:
    """整理外部实验环境、版本与配置快照。"""
    manifest_path = source_run_dir / "run_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.is_file() else None
    manifest_snapshot = deepcopy(manifest) if manifest else None
    if manifest_snapshot is not None:
        (manifest_snapshot.get("selection") or {}).pop("instance_ids", None)
    runtime = (manifest or {}).get("runtime") or {}
    return {
        "schema_version": 3,
        "ingested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark": {"name": benchmark_name or None, "case_count": case_count},
        "source": {
            "trajectory_dir": str(source_run_dir),
            "trajectory_format": "mini-swe-agent",
            "run_manifest": {
                "path": str(manifest_path),
                "sha256": sha256(manifest_path.read_bytes()).hexdigest() if manifest else None,
                "content": manifest_snapshot,
            },
            "git": {"sha": runtime.get("git_sha"), "branch": source_git_branch},
            "trajectory_configurations": _trajectory_configurations(source_run_dir),
        },
        "evaluation": _evaluation_meta(evaluation_report_path),
    }


def _trajectory_configurations(source_run_dir: Path) -> list[dict[str, Any]]:
    """按配置快照归并轨迹实际生效的运行配置。"""
    groups: dict[str, dict[str, Any]] = {}
    for trajectory_path in iter_trajectory_paths(source_run_dir):
        trajectory = load_json(trajectory_path)
        config = (trajectory.get("info") or {}).get("config")
        instance_id = trajectory.get("instance_id") or (trajectory.get("info") or {}).get("instance_id")
        if not isinstance(config, dict) or not isinstance(instance_id, str):
            continue
        snapshot = _redact_config(config)
        encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = sha256(encoded.encode("utf-8")).hexdigest()
        group = groups.setdefault(digest, {"sha256": digest, "config": snapshot, "instance_ids": []})
        group["instance_ids"].append(instance_id)
    return [
        {**group, "instance_ids": sorted(group["instance_ids"])}
        for _, group in sorted(groups.items())
    ]


def _redact_config(value: Any) -> Any:
    """递归脱敏配置中的常见凭据字段。"""
    if isinstance(value, dict):
        return {
            key: "<redacted>" if _SENSITIVE_CONFIG_KEY.search(str(key)) else _redact_config(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_config(item) for item in value]
    return value


def _evaluation_meta(evaluation_report_path: Path) -> dict[str, Any]:
    """摘录本次评测的来源、环境和输入预测文件。"""
    meta_path = evaluation_report_path.parent / "eval_meta.json"
    eval_meta = load_json(meta_path) if meta_path.is_file() else {}
    return {
        "report": _file_ref(evaluation_report_path),
        "meta": _file_ref(meta_path),
        "run_id": eval_meta.get("run_id"),
        "started_at": eval_meta.get("started_at"),
        "dataset": eval_meta.get("dataset"),
        "split": eval_meta.get("split"),
        "case_count": eval_meta.get("n_instances"),
        "max_workers": eval_meta.get("max_workers"),
        "timeout_s": eval_meta.get("timeout_s"),
        "harness_python": eval_meta.get("harness_python"),
        "predictions": {
            "source": _file_ref(Path(eval_meta["predictions_source"]))
            if eval_meta.get("predictions_source") else None,
            "used": _file_ref(Path(eval_meta["predictions_for_run"]))
            if eval_meta.get("predictions_for_run") else None,
        },
    }


def _file_ref(path: Path) -> dict[str, str | None]:
    """记录外部文件路径及其当前内容哈希。"""
    return {
        "path": str(path),
        "sha256": sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
    }


def _summary(cases: list[dict[str, Any]], grading_counts: dict[str, int]) -> dict[str, Any]:
    """写入不重复外部配置的批次级摄入结果。"""
    tool_calls = [call for case in cases for step in case["steps"] for call in step["tool_calls"]]
    return {
        "schema_version": 3,
        "ingest": {
            "case_count": len(cases),
            "tool_call_count": len(tool_calls),
            "tool_result_count": sum(call["observation"] is not None for call in tool_calls),
            "grading_counts": grading_counts,
        },
    }
