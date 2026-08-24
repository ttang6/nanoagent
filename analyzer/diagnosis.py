"""生成有轨迹证据的单题失败诊断。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from infra.core.tools import ToolRegistry
from infra.core.types import RunLimits
from infra.runtime.providers.openai_compat import OpenAICompatProvider
from infra.runtime.runner import run_agent

from .adapters.mini_swe_extra import load_evaluation_evidence
from .helpers import iter_cases, load_json, truncate_text, write_json


MODEL = "deepseek-v4-flash"
MAX_TURNS = 20
REQUEST_TIMEOUT_S = 300
MAX_OUTPUT_TOKENS = 8192
MAX_THINKING_CHARS = 8_000
MAX_OBSERVATION_CHARS = 8_000
CONTEXT_WINDOW_TOKENS = 384_000
OUTPUT_DIR = "diagnoses"
SYSTEM_PROMPT_PATH = Path(__file__).parent / "assets" / "diagnosis" / "system.md"
FAILED_OUTCOMES = frozenset({"unresolved", "empty_patch", "error"})


def run_diagnosis(
    experiment_dir: Path,
    *,
    artifacts_root: Path,
    limit: int | None = None,
) -> None:
    """为失败 case 生成诊断文件。"""
    investigate = load_json(experiment_dir / "investigate" / "investigate.json")
    provider = OpenAICompatProvider(
        MODEL,
        timeout_s=REQUEST_TIMEOUT_S,
        default_max_tokens=MAX_OUTPUT_TOKENS,
        default_extra={"thinking": {"type": "disabled"}},
    )
    completed = 0
    try:
        for case in iter_cases(experiment_dir):
            if case.get("grading") not in FAILED_OUTCOMES:
                continue
            if limit is not None and completed >= limit:
                break
            output_path = experiment_dir / OUTPUT_DIR / f"{case['instance_id']}.json"
            if output_path.is_file():
                print(f"diagnosis exists: {case['instance_id']}")
                continue
            request = build_diagnosis_request(case, investigate)
            result = run_agent(
                json.dumps(request, ensure_ascii=False, indent=2),
                provider=provider,
                tools=ToolRegistry(),
                workdir=str(artifacts_root.resolve()),
                artifacts_root=artifacts_root,
                artifact_namespace=(
                    f"analysis/{experiment_dir.name}/diagnosis/{case['instance_id']}"
                ),
                system_prompt=SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
                limits=RunLimits(
                    max_turns=MAX_TURNS,
                    context_window_tokens=CONTEXT_WINDOW_TOKENS,
                ),
                close_resources=False,
            )
            if result.finish_reason != "completed":
                raise RuntimeError(f"diagnosis 未完成: {result.finish_reason}: {result.error}")
            diagnosis = parse_diagnosis(result.final_text)
            diagnosis["meta"] = {
                "schema_version": 1,
                "instance_id": case["instance_id"],
                "grading": case["grading"],
            }
            write_json(output_path, diagnosis)
            completed += 1
            print(f"diagnosed: {case['instance_id']}")
    finally:
        provider.close()


def build_diagnosis_request(case: dict[str, Any], investigate: dict[str, Any]) -> dict[str, Any]:
    """组装单题诊断所需的过程与评测证据。"""
    instance_id = case["instance_id"]
    metrics = investigate["cases"][instance_id]
    evaluation = _load_related_evaluation_evidence(case)
    submission = case.get("submission") or ""
    return {
        "task": case["task"],
        "terminal_facts": {
            "grading": case["grading"],
            "exit_status": case.get("exit_status"),
            "submission_state": "present" if submission.strip() else "empty",
            "evaluation_sources": evaluation["availability"],
            "diagnosis_scope": _diagnosis_scope(submission, evaluation),
        },
        "submission": submission or None,
        "steps": _project_steps(case["steps"]),
        "mechanical_context": {
            "field_notes": investigate["field_notes"],
            "this_case": metrics,
            "resolved_baseline": investigate["by_outcome"].get("resolved"),
        },
        "evaluation_evidence": {
            "report": evaluation["report"],
            "failure_output": evaluation["failure_output"],
        },
    }


def _load_related_evaluation_evidence(case: dict[str, Any]) -> dict[str, Any]:
    """只通过 case 记录的相关来源读取单题评测材料。"""
    sources = case.get("related_sources") or {}
    report_path = sources.get("evaluation_report")
    output_path = sources.get("evaluation_output")
    availability = {
        "evaluation_report": "present" if isinstance(report_path, str) else "missing",
        "evaluation_output": "present" if isinstance(output_path, str) else "missing",
    }
    if not isinstance(report_path, str):
        return {"availability": availability, "report": None, "failure_output": None}

    evidence = load_evaluation_evidence(Path(report_path).parent, case["instance_id"])
    if evidence["report"] is None:
        availability["evaluation_report"] = "unreadable"
    if output_path is not None and evidence["failure_output"] is None:
        availability["evaluation_output"] = "no_relevant_excerpt"
    return {
        "availability": availability,
        "report": evidence["report"],
        "failure_output": evidence["failure_output"],
    }


def _diagnosis_scope(submission: str, evaluation: dict[str, Any]) -> str:
    """说明当前证据允许诊断到哪一层。"""
    if not submission.strip():
        return "未形成提交：只解释为何未得到可评测补丁，不归因于补丁逻辑或测试失败。"
    report = evaluation["report"]
    if not isinstance(report, dict):
        return "官方评测材料不可用：只分析任务过程与本地验证边界，不声称官方失败机制。"
    if report.get("patch_successfully_applied") is False:
        return "补丁未成功应用：只分析提交和应用阶段，不归因于测试断言或业务逻辑。"
    if evaluation["failure_output"] is not None:
        return "有官方评测报告和失败文本：可结合它们定位测试失败机制。"
    return "有官方评测报告但无相关失败文本：可描述报告事实，不把具体失败机制说成已证实。"


def parse_diagnosis(text: str) -> dict[str, Any]:
    """解析并校验 diagnosis 的最小输出契约。"""
    value = _parse_diagnosis_object(text)
    if not isinstance(value, dict):
        raise ValueError("diagnosis 输出必须是 JSON 对象")
    for field in ("summary", "observations", "hypotheses", "open_questions"):
        if field not in value:
            raise ValueError(f"diagnosis 缺少字段: {field}")
    questions = value["open_questions"]
    if not isinstance(questions, list) or len(questions) > 2:
        raise ValueError("open_questions 必须是至多两条的列表")
    return value


def _parse_diagnosis_object(text: str) -> object:
    """优先严格解析，失败后只提取唯一的 JSON 值。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError as strict_error:
        values = _top_level_json_values(text)
        if len(values) == 1 and isinstance(values[0], dict):
            return values[0]
        repaired = _repair_unescaped_string_quotes(text)
        values = _top_level_json_values(repaired)
        if len(values) == 1 and isinstance(values[0], dict):
            return values[0]
        try:
            value = json.loads(repaired)
        except json.JSONDecodeError:
            raise ValueError("diagnosis 输出必须是唯一的 JSON 对象") from strict_error
        if not isinstance(value, dict):
            raise ValueError("diagnosis 输出必须是唯一的 JSON 对象") from strict_error
        return value


def _repair_unescaped_string_quotes(text: str) -> str:
    """只修复 JSON 字符串内、不能构成分隔符的裸双引号。"""
    repaired = []
    in_string = False
    index = 0
    while index < len(text):
        character = text[index]
        if character == "\\" and in_string and index + 1 < len(text):
            repaired.extend((character, text[index + 1]))
            index += 2
            continue
        if character != '"':
            repaired.append(character)
            index += 1
            continue
        if not in_string:
            in_string = True
            repaired.append(character)
        elif _is_string_delimiter(text, index):
            in_string = False
            repaired.append(character)
        else:
            repaired.append('\\"')
        index += 1
    return "".join(repaired)


def _is_string_delimiter(text: str, index: int) -> bool:
    """判断字符串内的双引号是否是 JSON 结构分隔符。"""
    following = index + 1
    while following < len(text) and text[following].isspace():
        following += 1
    if following == len(text) or text[following] in ":}]":
        return True
    if text[following] != ",":
        return False
    following += 1
    while following < len(text) and text[following].isspace():
        following += 1
    if following >= len(text) or text[following] != '"':
        return False
    key_end = text.find('"', following + 1)
    if key_end == -1:
        return False
    key_end += 1
    while key_end < len(text) and text[key_end].isspace():
        key_end += 1
    return key_end < len(text) and text[key_end] == ":"


def _top_level_json_values(text: str) -> list[object]:
    """提取回复中彼此不重叠的 JSON 对象或数组。"""
    decoder = json.JSONDecoder()
    values = []
    index = 0
    while index < len(text):
        if text[index] not in "[{" or not _is_json_boundary(text, index):
            index += 1
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            index += 1
            continue
        values.append(value)
        index += end
    return values


def _is_json_boundary(text: str, index: int) -> bool:
    """排除代码或自然语言中的内联方括号。"""
    if index == 0:
        return True
    return text[index - 1] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.'\"}]"


def _project_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """折叠重复推理并截断请求侧长文本。"""
    previous_thinking: str | None = None
    previous_index: int | None = None
    projected = []
    for step in steps:
        thinking = step.get("thinking")
        if thinking and thinking == previous_thinking:
            projected_thinking: str | dict[str, int] = {"same_as_step": previous_index}
        elif isinstance(thinking, str):
            projected_thinking = truncate_text(thinking, MAX_THINKING_CHARS)
            previous_thinking = thinking
            previous_index = step["index"]
        else:
            projected_thinking = thinking
            previous_thinking = None
            previous_index = None
        projected.append({
            "index": step["index"],
            "thinking": projected_thinking,
            "tool_calls": [
                {
                    **call,
                    "observation": truncate_text(call["observation"], MAX_OBSERVATION_CHARS)
                    if isinstance(call.get("observation"), str) else call.get("observation"),
                }
                for call in step.get("tool_calls") or []
            ],
        })
    return projected
