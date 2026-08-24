"""从 case 生成可复核的机械行为字段。"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from .helpers import iter_cases, write_json


INVESTIGATE_SCHEMA_VERSION = 1
OUTPUT_PATH = Path("investigate") / "investigate.json"

# 提交协议本身会写 patch.txt 并 cat 它。它不是编辑，必须先排除，
# 否则每条轨迹的“最后一次编辑”都会变成提交流程。
_SUBMIT_COMMAND = re.compile(r"patch\.txt|COMPLETE_TASK_AND_SUBMIT", re.I)
# 自建的复现脚本、补丁脚本与调试脚本不是对目标源码的编辑。
_HELPER_NAME = r"(?:tests?_|test|repro|patch|fix|debug|check|tmp)"
_INPLACE_EDIT = re.compile(r"(?:sed\s+-i|perl\s+-\w*i|git\s+apply|apply_patch|\bpatch\s+-p\d)", re.I)
_SOURCE_WRITE = re.compile(
    # 重定向或 tee 写入 /testbed 下的非辅助文件
    rf"(?:>>?|tee\s+)\s*/testbed/(?!{_HELPER_NAME})\S+"
    # 脚本内以写模式打开 /testbed 下的非辅助文件
    rf"|open\(\s*['\"]/testbed/(?!{_HELPER_NAME})[^'\"]+['\"]\s*,\s*['\"][wa]",
    re.I,
)
# 以下签名一律锚定在片段的**首词**上。裸词全串匹配会把命令行参数误当成命令：
# 最初的编辑签名含裸 `patch`，于是提交用的 `git diff > patch.txt` 被判成编辑；
# 同理 `tail -100 .../runtests.py` 不是在跑测试，只是在读那个文件。
_TEST_RUN = re.compile(
    r"^(?:python[0-9.]*\s+(?:-m\s+)?)?(?:pytest|tox|nox)\b"
    r"|^python[0-9.]*\s+\S*runtests\.py\b"
    r"|^(?:python[0-9.]*\s+)?\S*manage\.py\s+test\b"
    r"|^python[0-9.]*\s+(?:-c\b|<<)",  # 一次性运行时探针，按 Verify 处理
    re.I,
)
_EXPLORE_RUN = re.compile(
    r"^(?:rg|grep|cat|head|tail|find|ls|wc|file|awk|sed\s+-n|git\s+(?:show|diff|log|status|blame))\b",
    re.I,
)
_NAVIGATE_ONLY = re.compile(r"^(?:cd|pwd|export)\b", re.I)
_UTILITY_RUN = re.compile(r"^(?:git|pip|echo|mkdir|cp|mv|rm|chmod|which|touch|source)\b", re.I)
# 写入重定向：命令名可能是 cat，但 `cat > f` 是写不是读。
_WRITE_REDIRECT = re.compile(r"(?<![0-9])>>?\s*(?P<path>[^\s|&;]+)")
_SEGMENT_SPLIT = re.compile(r"&&|\|\||;|\|")
# 类别优先级：一条复合命令里出现多种动作时，取语义最强的那个。
_CATEGORY_PRIORITY = ("modify", "test", "explore", "utility", "navigate")
_PATH_IN_PATCH = re.compile(r"^(?:\+\+\+ b/|--- a/)(.+)$", re.M)

FIELD_NOTES = {
    "total_steps": "assistant 消息总数。",
    "total_tool_calls": "所有 step 中的实际工具调用数。",
    "first_edit_step": "首次识别到编辑行为的 assistant step；从未编辑为 null。",
    "reads_before_first_edit": "首次编辑前，明确读取或检查最终修改文件的次数。",
    "edit_count": "识别到的编辑命令次数。",
    "files_modified": "最终 submission diff 中的去重文件数。",
    "validated_after_final_edit": "最后一次编辑后、trajectory 结束前是否至少发生一次识别为验证行为的命令。",
    "edit_to_first_validation_steps": "第一次编辑到其后第一次验证的 step 差；之后未验证为 null。",
    "unknown_action_ratio": "动作签名表未命中的调用占比。它是语义类字段的体检指标：比例偏高说明"
    "签名表对当前 target 已失效，first_edit_step、edit_count、validated_after_final_edit 等均不可信。",
    "longest_identical_run": "连续且（命令、returncode、observation）三者完全相同的调用的最大长度。",
    "distinct_observation_ratio": "去重后的 observation 数占工具调用数的比例；无调用为 null。",
    "calls_after_last_novel_observation": "最后一次出现前所未见的 observation 之后，还执行了多少次调用。",
    "max_consecutive_failure": "连续非零 returncode 的最大长度；缺少 returncode 会中断连续段。",
    "exact_action_repeat_count": "完全相同 tool_name + normalized_args 的额外出现次数。",
}

GLOBAL_NOTES = {
    "behavior_buckets": "把已有逐题字段重新分桶，展示每个行为状态的 case 数、比例和 outcome 分布。",
    "project_coverage": "各官方 outcome 覆盖的项目数和每个项目的 case 数，用于识别局部现象。",
}


def run_investigate(experiment_dir: Path) -> None:
    """计算并写入整批机械分析。"""
    report = build_investigate(experiment_dir)
    write_json(experiment_dir / OUTPUT_PATH, report)
    print(f"investigate: {len(report['cases'])} cases -> {experiment_dir / OUTPUT_PATH}")


def build_investigate(experiment_dir: Path) -> dict[str, Any]:
    """构建逐题字段及其分组汇总。"""
    records = [_record(case) for case in iter_cases(experiment_dir)]
    return {
        "schema_version": INVESTIGATE_SCHEMA_VERSION,
        "field_notes": FIELD_NOTES,
        "overall": _summarize(records),
        "by_outcome": _grouped_summary(records, "outcome"),
        "by_project": _grouped_summary(records, "project"),
        "global_notes": GLOBAL_NOTES,
        "behavior_buckets": _behavior_buckets(records),
        "project_coverage": _project_coverage(records),
        "cases": {record["instance_id"]: record["metrics"] for record in records},
    }


def _record(case: dict[str, Any]) -> dict[str, Any]:
    """计算一条 case 的机械字段。"""
    steps = case["steps"]
    metrics = _metrics(steps, str(case.get("submission") or ""))
    instance_id = str(case["instance_id"])
    return {
        "instance_id": instance_id,
        "outcome": str(case["grading"]),
        "project": instance_id.split("__", 1)[0],
        "metrics": metrics,
    }


def _metrics(steps: list[dict[str, Any]], submission: str) -> dict[str, Any]:
    """从步骤与最终 patch 计算当前字段集。"""
    calls = _calls(steps)
    edit_steps = [call for call in calls if _is_edit(call)]
    validation_steps = [call for call in calls if _is_validation(call)]
    modified_files = _modified_files(submission)
    first_edit = edit_steps[0]["step_index"] if edit_steps else None
    final_edit = edit_steps[-1]["step_index"] if edit_steps else None
    first_validation = next(
        (call["step_index"] for call in validation_steps if first_edit is not None and call["step_index"] > first_edit),
        None,
    )
    return {
        "total_steps": len(steps),
        "total_tool_calls": len(calls),
        "first_edit_step": first_edit,
        "reads_before_first_edit": _reads_before_first_edit(calls, modified_files, first_edit),
        "edit_count": len(edit_steps),
        "files_modified": len(modified_files),
        "validated_after_final_edit": any(
            final_edit is not None and call["step_index"] > final_edit for call in validation_steps
        ),
        "edit_to_first_validation_steps": (
            first_validation - first_edit if first_validation is not None and first_edit is not None else None
        ),
        "max_consecutive_failure": _max_consecutive_failure(calls),
        "exact_action_repeat_count": _exact_action_repeat_count(calls),
        "unknown_action_ratio": _unknown_action_ratio(calls),
        "longest_identical_run": _longest_identical_run(calls),
        "distinct_observation_ratio": _distinct_observation_ratio(calls),
        "calls_after_last_novel_observation": _calls_after_last_novel_observation(calls),
    }


def _calls(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """展开步骤中的工具调用，并保留所属 step。"""
    return [
        {**call, "step_index": step["index"]}
        for step in steps
        for call in step.get("tool_calls") or []
        if isinstance(call, dict)
    ]


def _modified_files(submission: str) -> set[str]:
    """从最终 patch 提取去重的修改文件。"""
    return {path for path in _PATH_IN_PATCH.findall(submission) if path != "/dev/null"}


def _reads_before_first_edit(
    calls: list[dict[str, Any]], modified_files: set[str], first_edit: int | None,
) -> int:
    """统计首次编辑前对最终修改文件的明确检查。"""
    if first_edit is None:
        return 0
    return sum(
        call["step_index"] < first_edit
        and _is_read_or_check(call)
        and _mentions_any_file(call, modified_files)
        for call in calls
    )


def _max_consecutive_failure(calls: list[dict[str, Any]]) -> int:
    """计算最长连续非零退出码。"""
    longest = current = 0
    for call in calls:
        if isinstance(call.get("returncode"), int) and call["returncode"] != 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _observation(call: dict[str, Any]) -> str:
    """取调用结果文本，供只比较相等性的结构指标使用。"""
    return str(call.get("observation") or "").strip()


def _longest_identical_run(calls: list[dict[str, Any]]) -> int:
    """计算连续且命令与结果完全相同的最长段。

    只比较相等性，不解释命令含义，因此换 target 仍然成立。
    """
    longest = current = 0
    previous = None
    for call in calls:
        identity = (_call_identity(call), call.get("returncode"), _observation(call))
        current = current + 1 if identity == previous else 1
        longest = max(longest, current)
        previous = identity
    return longest


def _distinct_observation_ratio(calls: list[dict[str, Any]]) -> float | None:
    """计算去重 observation 占全部调用的比例；越低说明重复获取的信息越多。"""
    if not calls:
        return None
    return round(len({_observation(call) for call in calls}) / len(calls), 3)


def _calls_after_last_novel_observation(calls: list[dict[str, Any]]) -> int:
    """计算最后一次出现新 observation 之后仍继续执行的调用数。"""
    seen: set[str] = set()
    last_novel = -1
    for index, call in enumerate(calls):
        observation = _observation(call)
        if observation not in seen:
            seen.add(observation)
            last_novel = index
    return len(calls) - 1 - last_novel if calls else 0


def _exact_action_repeat_count(calls: list[dict[str, Any]]) -> int:
    """统计重复的工具名与规范化参数。"""
    counts = Counter(_call_identity(call) for call in calls)
    return sum(count - 1 for count in counts.values() if count > 1)


def _action_category(call: dict[str, Any]) -> str:
    """把一次调用归入动作类别；无法判定时返回 unknown。

    不得让"未识别"退化成"确定不是"。两者混同会让签名表失配时无人察觉，
    而这正是本模块此前把提交协议误判为编辑却长期无人发现的原因。
    """
    if _tool_name(call) != "bash":
        return "unknown"
    command = _command(call)
    if _SUBMIT_COMMAND.search(command):
        return "utility"
    found = {_segment_category(segment) for segment in _segments(command)}
    for category in _CATEGORY_PRIORITY:
        if category in found:
            return category
    return "unknown"


def _segments(command: str) -> list[str]:
    """把复合命令拆成实际执行的片段。

    `cd /testbed && python -c ...` 的语义在第二段；只看首词会把它误判为目录切换。
    """
    parts = [part.strip() for part in _SEGMENT_SPLIT.split(command)]
    substantive = [part for part in parts if part and not _NAVIGATE_ONLY.match(part)]
    return substantive or [part for part in parts if part]


def _segment_category(segment: str) -> str:
    """判定单个片段的动作类别。"""
    if _INPLACE_EDIT.search(segment) or _SOURCE_WRITE.search(segment):
        return "modify"
    # 重定向只看 heredoc 标记之前的部分：正文是数据，里面的 `>` 不是重定向。
    redirect = _WRITE_REDIRECT.search(segment.split("<<")[0])
    if redirect:
        return "modify" if _is_target_source(redirect.group("path")) else "utility"
    if _TEST_RUN.match(segment):
        return "test"
    if _EXPLORE_RUN.match(segment):
        return "explore"
    if _NAVIGATE_ONLY.match(segment):
        return "navigate"
    if _UTILITY_RUN.match(segment):
        return "utility"
    return "unknown"


def _is_target_source(path: str) -> bool:
    """判断写入路径是否为目标仓库的源码，而非自建脚本或临时文件。"""
    return path.startswith("/testbed/") and not re.match(
        rf"/testbed/{_HELPER_NAME}", path, re.I
    )


def _unknown_action_ratio(calls: list[dict[str, Any]]) -> float | None:
    """计算签名表未命中的调用占比；无调用为 null。

    这是语义类字段的体检指标。比例偏高说明签名表对当前 target 已失效，
    此时 first_edit_step、edit_count、validated_after_final_edit 等均不可信。
    """
    if not calls:
        return None
    return round(sum(_action_category(call) == "unknown" for call in calls) / len(calls), 3)


def _is_edit(call: dict[str, Any]) -> bool:
    """判断 action 是否在编辑目标仓库的源码。"""
    return _action_category(call) == "modify"


def _is_validation(call: dict[str, Any]) -> bool:
    """判断 action 是否为当前支持的验证命令。"""
    return _action_category(call) == "test"


def _is_read_or_check(call: dict[str, Any]) -> bool:
    """判断调用是否明确读取或检查文件。"""
    return _action_category(call) == "explore"


def _mentions_any_file(call: dict[str, Any], paths: set[str]) -> bool:
    """判断 action 是否明确提到其中任一相对文件路径。"""
    command = _command(call)
    return any(path in command or f"/testbed/{path}" in command for path in paths)


def _call_identity(call: dict[str, Any]) -> str:
    """构造只规范化格式的工具调用身份。"""
    arguments = call.get("arguments")
    if isinstance(arguments, dict):
        normalized_args = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    elif isinstance(arguments, str):
        normalized_args = " ".join(arguments.split())
    else:
        normalized_args = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{_tool_name(call)}:{normalized_args}"


def _tool_name(call: dict[str, Any]) -> str:
    """返回工具名。"""
    return str(call.get("name") or "")


def _command(call: dict[str, Any]) -> str:
    """返回 bash 工具的 command 参数。"""
    arguments = call.get("arguments")
    if isinstance(arguments, dict) and isinstance(arguments.get("command"), str):
        return arguments["command"]
    return ""


def _grouped_summary(records: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    """按 outcome 或 project 汇总字段。"""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[field])].append(record)
    return {name: _summarize(group) for name, group in sorted(groups.items())}


def _behavior_buckets(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按已有字段汇总少量直观的行为状态。"""
    definitions = {
        "editing": lambda metrics: "edited" if metrics["edit_count"] else "not_edited",
        "validation_after_final_edit": _validation_bucket,
        "exact_action_repetition": lambda metrics: (
            "has_repeat" if metrics["exact_action_repeat_count"] else "no_repeat"
        ),
        "consecutive_failures": lambda metrics: (
            "has_failure" if metrics["max_consecutive_failure"] else "no_failure"
        ),
    }
    return {
        name: _bucket_counts(records, classify)
        for name, classify in definitions.items()
    }


def _validation_bucket(metrics: dict[str, Any]) -> str:
    """区分未编辑、编辑后验证和编辑后未验证。"""
    if not metrics["edit_count"]:
        return "not_edited"
    return "validated" if metrics["validated_after_final_edit"] else "not_validated"


def _bucket_counts(records: list[dict[str, Any]], classify: Any) -> dict[str, Any]:
    """统计一个行为分桶的数量、比例与 outcome 分布。"""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(classify(record["metrics"]))].append(record)
    total = len(records)
    return {
        "case_count": total,
        "buckets": {
            name: {
                "case_count": len(group),
                "case_ratio": round(len(group) / total, 3) if total else 0.0,
                "by_outcome": dict(sorted(Counter(record["outcome"] for record in group).items())),
            }
            for name, group in sorted(groups.items())
        },
    }


def _project_coverage(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按 outcome 列出项目覆盖。"""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["outcome"]].append(record)
    return {
        outcome: {
            "project_count": len(Counter(record["project"] for record in group)),
            "cases_by_project": dict(sorted(Counter(record["project"] for record in group).items())),
        }
        for outcome, group in sorted(groups.items())
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总一组 case 的字段分布。"""
    fields = tuple(FIELD_NOTES)
    return {
        "case_count": len(records),
        "metrics": {
            field: _summary([record["metrics"][field] for record in records])
            for field in fields
        },
    }


def _summary(values: list[Any]) -> dict[str, float | int | None]:
    """汇总数值或布尔字段。"""
    present = [value for value in values if value is not None]
    if not present:
        return {"min": None, "median": None, "mean": None, "max": None}
    numbers = [int(value) if isinstance(value, bool) else value for value in present]
    return {
        "min": min(numbers),
        "median": round(median(numbers), 3),
        "mean": round(sum(numbers) / len(numbers), 3),
        "max": max(numbers),
    }
