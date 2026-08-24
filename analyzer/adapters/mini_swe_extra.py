"""读取 mini-swe 对应的 SWE-bench 单题评测产物。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..helpers import iter_cases, load_json, write_json


_FILES = ("report.json", "test_output.txt", "run_instance.log", "patch.diff")
_EXCERPT_LINES = 24


def evaluation_case_dir(
    evaluation_report_path: str | Path, run_id: str, instance_id: str,
) -> Path:
    """根据聚合评测报告定位单题产物目录。"""
    report_path = Path(evaluation_report_path)
    provider = report_path.name.removesuffix(f".{run_id}.json")
    return report_path.parent / "logs" / "run_evaluation" / run_id / provider / instance_id


def related_sources(case_dir: str | Path) -> dict[str, str | None]:
    """返回单题评测产物的现有路径。"""
    root = Path(case_dir)
    names = {
        "evaluation_report": "report.json",
        "evaluation_output": "test_output.txt",
    }
    return {
        key: str(path) if (path := root / name).is_file() else None
        for key, name in names.items()
    }


def run_mini_swe_extra(experiment_dir: str | Path) -> None:
    """将单题评测材料路径写入全部 case。"""
    root = Path(experiment_dir)
    meta = load_json(root / "meta.json")
    evaluation = meta.get("evaluation") or {}
    report_path = (evaluation.get("report") or {}).get("path")
    run_id = evaluation.get("run_id")
    if not isinstance(report_path, str) or not isinstance(run_id, str):
        raise ValueError("meta.json 缺少 evaluation.report.path 或 evaluation.run_id")
    for case in iter_cases(root):
        case["related_sources"] = related_sources(
            evaluation_case_dir(report_path, run_id, case["instance_id"])
        )
        write_json(root / "cases" / f"{case['instance_id']}.json", case)


def load_evaluation_evidence(case_dir: str | Path, instance_id: str) -> dict[str, Any]:
    """读取单题 report 与失败测试附近的输出片段。"""
    root = Path(case_dir)
    paths = {name: root / name for name in _FILES}
    files = {name: "present" if path.is_file() else "missing" for name, path in paths.items()}
    evidence: dict[str, Any] = {
        "case_dir": str(root),
        "files": files,
        "report": None,
        "failure_output": None,
    }
    if files["report.json"] == "missing":
        return evidence

    report = load_json(paths["report.json"])
    instance_report = report.get(instance_id)
    if not isinstance(instance_report, dict):
        return evidence
    evidence["report"] = instance_report
    if files["test_output.txt"] == "present":
        evidence["failure_output"] = _failure_output(
            paths["test_output.txt"], instance_report,
        )
    return evidence


def _failure_output(output_path: Path, report: dict[str, Any]) -> str | None:
    """按 fail-to-pass 失败测试名截取局部输出。"""
    failures = (
        report.get("tests_status", {})
        .get("FAIL_TO_PASS", {})
        .get("failure", [])
    )
    if not isinstance(failures, list) or not failures:
        return None
    lines = output_path.read_text(encoding="utf-8").splitlines()
    excerpts = []
    for name in failures:
        if not isinstance(name, str):
            continue
        start = next((index for index, line in enumerate(lines) if f"FAIL: {name}" in line), None)
        if start is not None:
            excerpts.append("\n".join(lines[start:start + _EXCERPT_LINES]))
    return "\n\n".join(excerpts) or None
