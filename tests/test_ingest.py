"""摄入阶段的评测元数据契约测试。"""

import json
from pathlib import Path

from analyzer.ingest import _evaluation_meta, _trajectory_configurations


def test_evaluation_meta_keeps_selected_harness_fields_and_file_hashes(tmp_path: Path):
    """实验 meta 记录评测来源，但不复制题目列表。"""
    source_predictions = tmp_path / "source_predictions.json"
    used_predictions = tmp_path / "used_predictions.json"
    report = tmp_path / "report.json"
    source_predictions.write_text("{}", encoding="utf-8")
    used_predictions.write_text("{}", encoding="utf-8")
    report.write_text("{}", encoding="utf-8")
    (tmp_path / "eval_meta.json").write_text(json.dumps({
        "run_id": "eval-1",
        "started_at": "2026-08-12T00:00:00Z",
        "dataset": "dataset",
        "split": "test",
        "instance_ids": ["proj__one"],
        "n_instances": 1,
        "max_workers": 5,
        "timeout_s": 1800,
        "harness_python": "C:/harness/python.exe",
        "predictions_source": str(source_predictions),
        "predictions_for_run": str(used_predictions),
    }), encoding="utf-8")

    actual = _evaluation_meta(report)

    assert actual["run_id"] == "eval-1"
    assert actual["case_count"] == 1
    assert actual["report"]["sha256"] is not None
    assert actual["meta"]["sha256"] is not None
    assert actual["predictions"]["source"]["sha256"] is not None
    assert actual["predictions"]["used"]["sha256"] is not None
    assert "instance_ids" not in actual


def test_trajectory_configurations_preserves_unknown_fields_and_redacts_credentials(tmp_path: Path):
    """轨迹配置按快照归并，不需要为模型新字段添加接收逻辑。"""
    for instance_id in ("proj__one", "proj__two"):
        (tmp_path / f"{instance_id}.traj.json").write_text(json.dumps({
            "instance_id": instance_id,
            "info": {"config": {
                "model": {"model_kwargs": {"extra_body": {"thinking_budget": 8196}}},
                "api_key": "secret-value",
            }},
        }), encoding="utf-8")

    configurations = _trajectory_configurations(tmp_path)

    assert len(configurations) == 1
    assert configurations[0]["instance_ids"] == ["proj__one", "proj__two"]
    assert configurations[0]["config"]["model"]["model_kwargs"]["extra_body"] == {"thinking_budget": 8196}
    assert configurations[0]["config"]["api_key"] == "<redacted>"
