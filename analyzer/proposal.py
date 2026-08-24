"""基于整批诊断和机械统计生成一份工程 proposal。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from infra.core.tools import ToolRegistry
from infra.core.types import RunLimits
from infra.runtime.providers.openai_compat import OpenAICompatProvider
from infra.runtime.runner import run_agent

from .helpers import load_json, write_json


MODEL = "deepseek-v4-pro"
THINKING = "enabled"
MAX_TURNS = 10
REQUEST_TIMEOUT_S = 300
MAX_OUTPUT_TOKENS = 8192 * 2
CONTEXT_WINDOW_TOKENS = 384_000
OUTPUT_PATH = Path("proposal") / "proposal.json"
EVOLVE_SPEC_PATH = Path("proposal") / "evolve_spec.md"
SYSTEM_PROMPT_PATH = Path(__file__).parent / "assets" / "proposal" / "system.md"
TARGET_KINDS = ("static_asset", "tool", "harness")


def run_proposal(
    experiment_dir: Path,
    *,
    artifacts_root: Path,
    evidence_experiment_dirs: list[Path] | None = None,
) -> None:
    """为一个实验生成 proposal；已有产物时不重跑。"""
    output_path = experiment_dir / OUTPUT_PATH
    if output_path.is_file():
        evolve_spec_path = experiment_dir / EVOLVE_SPEC_PATH
        if not evolve_spec_path.is_file():
            evolve_spec_path.write_text(build_evolve_spec(load_json(output_path)), encoding="utf-8")
            print(f"evolve spec: {evolve_spec_path}")
        print(f"proposal exists: {output_path}")
        return

    evidence_dirs = evidence_experiment_dirs or [experiment_dir]
    request, used_materials = build_proposal_request(evidence_dirs)
    provider = OpenAICompatProvider(
        MODEL,
        timeout_s=REQUEST_TIMEOUT_S,
        default_max_tokens=MAX_OUTPUT_TOKENS,
        default_extra={"thinking": {"type": THINKING}},
    )
    try:
        result = run_agent(
            json.dumps(request, ensure_ascii=False, indent=2),
            provider=provider,
            tools=ToolRegistry(),
            workdir=str(artifacts_root.resolve()),
            artifacts_root=artifacts_root,
            artifact_namespace=f"analysis/{experiment_dir.name}/proposal",
            system_prompt=SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
            limits=RunLimits(
                max_turns=MAX_TURNS,
                context_window_tokens=CONTEXT_WINDOW_TOKENS,
            ),
            close_resources=False,
        )
        if result.finish_reason != "completed":
            raise RuntimeError(f"proposal 未完成: {result.finish_reason}: {result.error}")
        proposal = parse_proposal(result.final_text)
    finally:
        provider.close()

    payload = finalize_proposal(
        proposal,
        used_materials,
    )
    write_json(output_path, payload)
    evolve_spec_path = experiment_dir / EVOLVE_SPEC_PATH
    evolve_spec_path.write_text(build_evolve_spec(payload), encoding="utf-8")
    print(f"proposal: {output_path}")
    print(f"evolve spec: {evolve_spec_path}")


def build_proposal_request(experiment_dirs: list[Path]) -> tuple[dict[str, Any], list[str]]:
    """组装全量机械统计、所有诊断和目标说明。"""
    materials = [
        {
            "filename": f"{experiment_dir.name}/diagnoses/{path.name}",
            "content": load_json(path),
        }
        for experiment_dir in experiment_dirs
        for path in sorted((experiment_dir / "diagnoses").glob("*.json"))
    ]
    if not materials:
        names = ", ".join(str(path / "diagnoses") for path in experiment_dirs)
        raise ValueError(f"没有可供 proposal 使用的 diagnosis: {names}")
    reference_docs = [
        {
            "name": path.stem,
            "filename": path.name,
            "content": path.read_text(encoding="utf-8").strip(),
        }
        for path in sorted((Path(__file__).parent / "assets" / "targets").glob("*.md"))
    ]
    request = {
        "target_kinds": list(TARGET_KINDS),
        "reference_docs": reference_docs,
        "mechanical_overview": load_json(
            experiment_dirs[0] / "investigate" / "investigate.json"
        ),
        "evidence_source": "diagnoses",
        "evidence_materials": materials,
        "guardrails": [],
        "history": [],
    }
    return request, [material["filename"] for material in materials]


def parse_proposal(text: str) -> dict[str, Any]:
    """解析 proposal 的最小输出契约。"""
    value = json.loads(_unfence(text))
    if not isinstance(value, dict):
        raise ValueError("proposal 输出必须是 JSON 对象")
    for field in (
        "problem_statement",
        "evidence",
        "candidate_directions",
        "recommended_direction",
    ):
        if field not in value:
            raise ValueError(f"proposal 缺少字段: {field}")
    recommended = value["recommended_direction"]
    if not isinstance(recommended, dict):
        raise ValueError("recommended_direction 必须是对象")
    index = recommended.get("candidate_index")
    directions = value["candidate_directions"]
    if not isinstance(index, int) or isinstance(index, bool) or not isinstance(directions, list):
        raise ValueError("recommended_direction 必须引用一个候选方向")
    if not 1 <= index <= len(directions):
        raise ValueError("recommended_direction 引用的候选方向不存在")
    for direction_index, direction in enumerate(directions, start=1):
        if not isinstance(direction, dict) or not isinstance(direction.get("target_problem"), str):
            raise ValueError(f"第 {direction_index} 条 candidate_direction 缺少 target_problem")
        if not direction["target_problem"].strip():
            raise ValueError(f"第 {direction_index} 条 candidate_direction 的 target_problem 不能为空")
    _validate_item_count(value, "validation", minimum=1, maximum=3)
    _validate_item_count(value, "constraints", minimum=2, maximum=3)
    _validate_item_count(value, "risks", minimum=0, maximum=3)
    return value


def finalize_proposal(
    proposal: dict[str, Any],
    used_materials: list[str],
) -> dict[str, Any]:
    """过滤无效方向，并补上最小的阶段元信息。"""
    directions = proposal.get("candidate_directions")
    if isinstance(directions, list):
        filtered_directions = [
            direction
            for direction in directions
            if isinstance(direction, dict) and direction.get("asset_type") in TARGET_KINDS
        ]
        original_index = proposal["recommended_direction"]["candidate_index"]
        recommended = directions[original_index - 1]
        proposal["candidate_directions"] = filtered_directions
        if recommended not in filtered_directions:
            raise ValueError("recommended_direction 不能引用词表外方向")
        proposal["recommended_direction"]["candidate_index"] = (
            filtered_directions.index(recommended) + 1
        )
    proposal["meta"] = {
        "schema_version": 1,
        "stage": "proposal",
        "model": MODEL,
        "thinking": THINKING,
        "evidence_source": "diagnoses",
        "evidence_count": len(used_materials),
        "direction_count": len(proposal.get("candidate_directions") or []),
        "target_kinds": list(TARGET_KINDS),
    }
    return proposal


def build_evolve_spec(proposal: dict[str, Any]) -> str:
    """从推荐候选确定性生成 Evolve 勘测规格。"""
    recommendation = proposal["recommended_direction"]
    direction = proposal["candidate_directions"][recommendation["candidate_index"] - 1]
    sections = [
        "# 本轮 Evolve 勘测规格",
        "本文只说明本轮要处理哪个问题、依据是什么。承载位置、改动手段和验证命令由勘测阶段"
        "在读取目标仓库后确定，不在本文范围内。它不是直接实施的施工单。",
        "## 目标问题",
        str(direction["target_problem"]),
        "## 机制假设",
        str(direction.get("mechanism_hypothesis") or ""),
        "## 载体层级",
        str(direction.get("asset_type") or ""),
        "## 预期过程变化",
        str(direction.get("expected_effect") or ""),
        "## 为什么优先这一条",
        str(recommendation.get("reason") or ""),
    ]
    reference_basis = direction.get("reference_basis")
    if isinstance(reference_basis, list) and reference_basis:
        sections.extend(["## 相关设计依据"])
        sections.extend(
            f"- {item['filename']}：{item['relevance']}"
            for item in reference_basis
            if isinstance(item, dict) and isinstance(item.get("filename"), str)
            and isinstance(item.get("relevance"), str)
        )
    sections.extend([
        "## 判定为无效的条件",
        *_bullets(proposal["validation"]),
    ])
    if proposal["risks"]:
        sections.extend(["## 已知副作用与替代解释", *_bullets(proposal["risks"])])
    sections.extend([
        "## 固定边界",
        *_bullets(proposal["constraints"]),
        "不得修改评分器、评测、标准答案或冻结层。",
    ])
    return "\n\n".join(sections) + "\n"


def _unfence(text: str) -> str:
    """去除模型偶尔附带的一层 JSON 代码围栏。"""
    stripped = text.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        return stripped[len("```json"):-len("```")].strip()
    return stripped


def _validate_item_count(value: dict[str, Any], field: str, *, minimum: int, maximum: int) -> None:
    """校验 proposal 列表字段的收敛数量。"""
    items = value.get(field)
    if not isinstance(items, list) or not minimum <= len(items) <= maximum:
        raise ValueError(f"{field} 必须是 {minimum} 到 {maximum} 条的列表")


def _bullets(items: list[object]) -> list[str]:
    """把简短列表写成 Markdown 项。"""
    return [f"- {item}" for item in items]
