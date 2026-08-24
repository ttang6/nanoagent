"""动作分类与结构指标的回归测试。

分类用例全部来自 2026-08-10-verified50 的真实命令。本模块曾出现过三个同源 bug——
把整条命令当一个字符串糊着匹配，而不看实际执行的是什么——每个都在下面钉了用例。
"""

from __future__ import annotations

import pytest

from analyzer.investigate import _action_category, _metrics


def bash(command: str) -> dict[str, object]:
    """构造一次 bash 调用。"""
    return {"name": "bash", "arguments": {"command": command}}


def step(index: int, command: str, observation: str, returncode: int = 0) -> dict[str, object]:
    """构造一条只含单次 bash 调用的步骤。"""
    return {
        "index": index,
        "tool_calls": [{
            "name": "bash",
            "arguments": {"command": command},
            "observation": observation,
            "returncode": returncode,
        }],
    }


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # 提交协议不是编辑。裸 `patch` 曾让 `git diff > patch.txt` 被判成编辑，
        # 于是每条轨迹的“最后一次编辑”都变成提交流程，validated_after_final_edit 恒为 false。
        ("cd /testbed && git diff django/db/models/sql/query.py > patch.txt", "utility"),
        ("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt", "utility"),
        # 命令行参数不是命令。`runtests.py` 出现在路径里只是被读取。
        ("tail -100 /testbed/tests/runtests.py", "explore"),
        ("cd /testbed/tests && python runtests.py admin_docs.test_utils", "test"),
        # 复合命令的语义在 cd 之后的片段里。
        ("cd /testbed && python -c \"from django.db import connection\"", "test"),
        ("cd /testbed && git log --oneline -10", "explore"),
        # 重定向改变命令语义：cat 是读，`cat >` 是写。
        ("cat > /tmp/test_issue.py << 'EOF'\nimport os\nEOF", "utility"),
        ("cat > /testbed/django/db/models/base.py << 'EOF'\nx = 1\nEOF", "modify"),
        # heredoc 正文是数据，其中的 > 不是重定向。
        ("python3 << 'EOF'\nimport sphinx\nprint(1 > 0)\nEOF", "test"),
        # 写模式打开目标源码算编辑，写自建脚本不算。
        ("cat > /tmp/fix.py << 'EOF'\nopen('/testbed/sphinx/writers/latex.py', 'w')\nEOF", "modify"),
        ("sed -i 's/a/b/' /testbed/django/forms/models.py", "modify"),
        ("sed -n '100,200p' /testbed/django/db/models/deletion.py", "explore"),
    ],
)
def test_action_category_classifies_real_commands(command: str, expected: str) -> None:
    """签名表按片段首词与重定向判定，不被参数或 heredoc 正文误导。"""
    assert _action_category(bash(command)) == expected


def test_unrecognized_command_is_unknown_not_a_silent_negative() -> None:
    """未识别必须显式可见，否则签名表失配时无人察觉。"""
    assert _action_category(bash("frobnicate --wibble")) == "unknown"
    assert _metrics([step(1, "frobnicate --wibble", "out")], "")["unknown_action_ratio"] == 1.0


def test_identical_run_counts_only_adjacent_calls_with_identical_results() -> None:
    """结构指标只比较相等性：结果变化即中断连续段。"""
    same = [step(index, "grep -r foo /testbed", "") for index in range(1, 5)]
    assert _metrics(same, "")["longest_identical_run"] == 4
    assert _metrics(same, "")["calls_after_last_novel_observation"] == 3

    changed = [*same[:2], step(3, "grep -r foo /testbed", "hit"), *same[2:]]
    assert _metrics(changed, "")["longest_identical_run"] == 2


def test_metrics_stay_computable_without_any_recognized_edit() -> None:
    """全程只读的轨迹不应把“没有编辑”表示成 0 步编辑。"""
    metrics = _metrics([step(1, "ls /testbed", "a.py"), step(2, "cat /testbed/a.py", "x")], "")

    assert metrics["first_edit_step"] is None
    assert metrics["edit_count"] == 0
    assert metrics["validated_after_final_edit"] is False
    assert metrics["distinct_observation_ratio"] == 1.0
