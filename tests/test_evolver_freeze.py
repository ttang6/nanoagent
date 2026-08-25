"""Evolver freeze 和宿主候选封存测试。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from evolver.freeze import (
    build_evolve_context,
    load_freeze,
    prepare_freeze,
)
from evolver.patch_export import PatchExportPolicy, export_candidate
from infra.runtime.environments.base import CommandResult
from infra.core.types import RunResult, Usage
from run_evolve import EVOLVE_PROFILES, _thinking_config, _write_run_summary


def _git(directory: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(directory), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return result.stdout.strip()


def _target(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "target"
    source.mkdir()
    _git(source, "init", "--quiet")
    _git(source, "config", "user.name", "test")
    _git(source, "config", "user.email", "test@example.invalid")
    (source / "config.yaml").write_text("agent:\n  instance_template: before\n", encoding="utf-8")
    _git(source, "add", "config.yaml")
    _git(source, "commit", "--quiet", "-m", "base")
    return source, _git(source, "rev-parse", "HEAD")


def _experiment(tmp_path: Path) -> Path:
    experiment = tmp_path / "experiment"
    spec = experiment / "proposal" / "evolve_spec.md"
    spec.parent.mkdir(parents=True)
    spec.write_text("# spec\n", encoding="utf-8")
    return experiment


def _evolve_experiment(tmp_path: Path) -> Path:
    return tmp_path / "evolver" / "experiments" / "experiment"


def _harness_root(tmp_path: Path) -> Path:
    """创建 freeze 所需的稳定 Evolve 输入。"""
    root = tmp_path / "harness"
    (root / "evolver" / "assets").mkdir(parents=True)
    (root / "evolver" / "assets" / "CLAUDE.md").write_text("规则\n", encoding="utf-8")
    (root / "evolver" / "assets" / "system.md").write_text(
        "系统规则\n", encoding="utf-8"
    )
    return root


def test_prepare_freeze_captures_commit_bundle_and_inputs(tmp_path: Path) -> None:
    """准备阶段不修改 target，并留下可克隆 bundle。"""
    source, base = _target(tmp_path)
    repository_root = _harness_root(tmp_path)

    frozen = prepare_freeze(
        _experiment(tmp_path),
        _evolve_experiment(tmp_path),
        target_source=source,
        base_revision="HEAD",
        repository_root=repository_root,
    )

    assert frozen.source_commit == base
    assert frozen.directory.parent == _evolve_experiment(tmp_path) / "freezes"
    assert frozen.bundle_path.is_file()
    assert frozen.source_snapshot_path.is_file()
    assert frozen.system_path.read_text(encoding="utf-8") == "系统规则\n"
    assert load_freeze(frozen.directory) == frozen
    assert _git(source, "status", "--porcelain") == ""
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "--quiet", "--no-checkout", str(frozen.bundle_path), str(clone)], check=True)
    _git(clone, "checkout", "--quiet", "--detach", frozen.base_commit)
    assert _git(clone, "rev-parse", "HEAD") == frozen.base_commit
    assert _git(clone, "rev-parse", "HEAD^{tree}") == frozen.source_tree


def test_context_describes_container_not_host(tmp_path: Path) -> None:
    """Evolve 上下文只陈述容器副本与冻结输入。"""
    source, _ = _target(tmp_path)
    repository_root = _harness_root(tmp_path)
    frozen = prepare_freeze(
        _experiment(tmp_path), _evolve_experiment(tmp_path),
        target_source=source, base_revision="HEAD", repository_root=repository_root
    )

    messages = build_evolve_context(frozen).build(type("State", (), {"messages": []})())

    assert "工作目录是 /work" in messages[0].content
    assert "系统规则" in messages[0].content
    assert "# spec" in messages[0].content


def test_patch_export_policy_accepts_only_the_configured_agent_and_test_directories() -> None:
    """目录级策略只开放 agent 控制流与测试。"""
    policy = PatchExportPolicy(
        frozenset(),
        frozenset(),
        ("src/minisweagent/agents/", "tests/"),
    )

    assert policy.allows_path("src/minisweagent/agents/default.py")
    assert policy.allows_path("tests/test_default_agent.py")
    assert not policy.allows_path("src/minisweagent/environments/docker.py")


class _ExportEnvironment:
    """用本地 Git 工作目录模拟候选容器的最小接口。"""

    def __init__(self, worktree: Path, export_path: Path) -> None:
        self.worktree = worktree
        self.export_path = export_path

    def execute(self, command: str, *, timeout_s: float | None = None) -> CommandResult:
        if command.startswith("git diff --binary"):
            base = command.split()[4]
            diff = subprocess.run(
                ["git", "-C", str(self.worktree), "diff", "--binary", "--full-index", base, "--", "."],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            self.export_path.write_bytes(diff.stdout.encode("utf-8"))
            return CommandResult(exit_code=0)
        result = subprocess.run(
            command.split(), cwd=self.worktree, capture_output=True, text=True, encoding="utf-8"
        )
        return CommandResult(result.returncode, result.stdout, result.stderr)

    def copy_from_container(self, source: str, destination: Path) -> None:
        shutil.copyfile(self.export_path, destination)


def test_export_candidate_archives_patch_as_ref(tmp_path: Path) -> None:
    """导出以真实 diff 创建实验专用 candidate ref。"""
    source, _ = _target(tmp_path)
    repository_root = _harness_root(tmp_path)
    frozen = prepare_freeze(
        _experiment(tmp_path), _evolve_experiment(tmp_path),
        target_source=source, base_revision="HEAD", repository_root=repository_root
    )
    worktree = tmp_path / "worktree"
    subprocess.run(["git", "clone", "--quiet", "--no-checkout", str(frozen.bundle_path), str(worktree)], check=True)
    _git(worktree, "checkout", "--quiet", "--detach", frozen.base_commit)
    (worktree / "config.yaml").write_text("agent:\n  instance_template: after\n", encoding="utf-8")
    environment = _ExportEnvironment(worktree, tmp_path / "container.patch")

    result = export_candidate(  # type: ignore[arg-type]
        environment,
        frozen,
        policy=PatchExportPolicy(frozenset({"config.yaml"}), frozenset({"agent.instance_template"})),
    )

    assert result.patch_path is not None and result.patch_path.is_file()
    ref = f"refs/accrete/evolve/freeze-{frozen.freeze_id.removeprefix('freeze-')}"
    refs_repo = frozen.ref_store_path
    assert _git(refs_repo, "show", f"{ref}:config.yaml").endswith("after")


def test_run_summary_links_host_trace_and_export(tmp_path: Path) -> None:
    """摘要只索引宿主轨迹和封存事实，不依赖容器存活。"""
    path = tmp_path / "run_summary.json"
    exported = type("Export", (), {
        "status": "sealed_pending_evaluation",
        "patch_path": tmp_path / "candidate.patch",
        "candidate_ref": "refs/accrete/evolve/freeze-abc",
    })()

    _write_run_summary(
        path,
        frozen_id="freeze-abc",
        base_commit="a" * 40,
        profile="deepseek-v4-pro",
        model="model",
        image="image",
        max_turns=12,
        context_window_tokens=1_000_000,
        reserved_answer_tokens=100_000,
        max_output_tokens=8_192,
        wall_time_s=1_800,
        reasoning_effort="high",
        thinking=_thinking_config(EVOLVE_PROFILES["deepseek-v4-pro"], "high"),
        artifacts_root=tmp_path / "artifacts",
        started_at="2026-08-16T14:00:00Z",
        ended_at="2026-08-16T14:05:00Z",
        result=RunResult(None, "completed", 3, Usage(input=2, output=1)),
        exported=exported,
    )

    summary = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert summary["run"]["trace_root"] == str((tmp_path / "artifacts/evolve/freeze-abc").resolve())
    assert summary["run"]["history_token_limit"] == 900_000
    assert summary["run"]["profile"] == "deepseek-v4-pro"
    assert summary["run"]["reasoning_effort"] == "high"
    assert summary["run"]["thinking"] == {
        "reasoning_effort": "high",
        "provider_extra": {"thinking": {"type": "enabled"}},
    }
    assert summary["run"]["started_at"] == "2026-08-16T14:00:00Z"
    assert summary["run"]["ended_at"] == "2026-08-16T14:05:00Z"
    assert summary["export"]["candidate_ref"] == "refs/accrete/evolve/freeze-abc"


def test_deepseek_evolve_profiles_cover_reasoning_ablation() -> None:
    """DeepSeek 实验配置应明确区分低强度推理与关闭思考。"""
    assert EVOLVE_PROFILES["deepseek-v4-pro-low"] == {
        "model": "deepseek-v4-pro",
        "reasoning_effort": "low",
        "default_extra": {"thinking": {"type": "enabled"}},
    }
    assert EVOLVE_PROFILES["deepseek-v4-pro-thinking-off"] == {
        "model": "deepseek-v4-pro",
        "reasoning_effort": "none",
        "default_extra": {"thinking": {"type": "disabled"}},
    }
    assert EVOLVE_PROFILES["grok-4.6-medium"] == {
        "model": "grok-4.6",
        "reasoning_effort": "medium",
        "default_extra": {},
    }
