"""导出 Evolve 候选并按固定范围封存 Git ref。"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from infra.runtime.environments.docker import DockerEnvironment

from .freeze import FreezeError, FrozenEvolve, update_freeze_meta
from .helpers import write_json


@dataclass(frozen=True)
class PatchExportPolicy:
    """定义本轮候选允许修改的固定文件和 YAML 叶子键。"""

    allowed_paths: frozenset[str]
    allowed_yaml_keys: frozenset[str]
    allowed_path_prefixes: tuple[str, ...] = ()

    def allows_path(self, path: str) -> bool:
        """判断路径是否属于宿主批准的精确文件或目录范围。"""
        return path in self.allowed_paths or any(path.startswith(prefix) for prefix in self.allowed_path_prefixes)


@dataclass(frozen=True)
class ExportResult:
    """记录候选导出的终态；被拒绝时不返回 patch。"""

    status: str
    patch_path: Path | None
    candidate_ref: str | None


def export_candidate(
    environment: DockerEnvironment,
    frozen: FrozenEvolve,
    *,
    policy: PatchExportPolicy,
) -> ExportResult:
    """以真实 Git/YAML diff 检查候选，合格后创建独立 Git ref。"""
    patch_path = frozen.directory / "candidate.patch"
    ref: str | None = None
    commit: str | None = None
    try:
        _require_base_head(environment, frozen.base_commit)
        _require_clean_scope(environment, frozen, policy)
        _export_patch(environment, frozen, patch_path)
        if not patch_path.read_bytes():
            raise FreezeError("候选没有实际 Git 改动")
        ref, commit = _archive_ref(frozen, patch_path)
        _require_yaml_scope(frozen, commit, policy)
    except FreezeError as error:
        _record_rejection(frozen, error, patch_path, ref, commit)
        return ExportResult("rejected", patch_path if patch_path.is_file() else None, ref)

    record = {
        "schema_version": 1,
        "freeze_id": frozen.freeze_id,
        "base_commit": frozen.base_commit,
        "patch": "candidate.patch",
        "patch_sha256": _sha256(patch_path),
        "candidate_ref": ref,
        "candidate_commit": commit,
        "status": "sealed_pending_evaluation",
    }
    write_json(frozen.directory / "export.json", record)
    update_freeze_meta(frozen, status=record["status"], export=record)
    return ExportResult(record["status"], patch_path, ref)


def _require_base_head(environment: DockerEnvironment, base_commit: str) -> None:
    """拒绝 agent 修改 Git 历史后再导出候选。"""
    result = environment.execute("git rev-parse HEAD", timeout_s=30)
    if result.exit_code != 0 or result.stdout.strip() != base_commit:
        raise FreezeError("freeze 容器的 HEAD 已偏离 base_commit，拒绝导出")


def _require_clean_scope(environment: DockerEnvironment, frozen: FrozenEvolve, policy: PatchExportPolicy) -> None:
    """检查真实 Git 文件差异只落在宿主预先允许的路径。"""
    changed = environment.execute(f"git diff --name-only {frozen.base_commit} --", timeout_s=30)
    if changed.exit_code != 0:
        raise FreezeError(f"无法读取候选 Git diff: {changed.stderr.strip()}")
    paths = {line.strip() for line in changed.stdout.splitlines() if line.strip()}
    if not paths:
        raise FreezeError("候选没有实际 Git 改动")
    unexpected = sorted(path for path in paths if not policy.allows_path(path))
    if unexpected:
        raise FreezeError(f"候选修改了未授权路径: {', '.join(unexpected)}")
    status = environment.execute("git status --porcelain", timeout_s=30)
    if status.exit_code != 0:
        raise FreezeError(f"无法读取候选 Git 状态: {status.stderr.strip()}")
    extra_untracked = [
        line[3:]
        for line in status.stdout.splitlines()
        if line.startswith("?? ")
    ]
    if extra_untracked:
        raise FreezeError(f"候选留下了未授权未跟踪文件: {', '.join(extra_untracked)}")
    checked = environment.execute(f"git diff --check {frozen.base_commit}", timeout_s=60)
    if checked.exit_code != 0:
        raise FreezeError(f"候选 diff 检查失败: {checked.stdout}{checked.stderr}")


def _export_patch(environment: DockerEnvironment, frozen: FrozenEvolve, patch_path: Path) -> None:
    """从容器工作树导出二进制安全的真实 Git patch。"""
    made = environment.execute(
        f"git diff --binary --full-index {frozen.base_commit} -- . > /tmp/accrete-candidate.patch",
        timeout_s=60,
    )
    if made.exit_code != 0:
        raise FreezeError(f"无法生成候选 patch: {made.stderr.strip()}")
    environment.copy_from_container("/tmp/accrete-candidate.patch", patch_path)


def _archive_ref(frozen: FrozenEvolve, patch_path: Path) -> tuple[str, str]:
    """在实验专用 bare 仓库创建不受 agent 改写的候选 ref。"""
    refs_repo = frozen.ref_store_path
    if not refs_repo.exists():
        _run_git(frozen.directory, ["init", "--bare", str(refs_repo)])
    workspace = frozen.directory / ".archive-worktree"
    try:
        _run_git(frozen.directory, ["clone", "--quiet", "--no-checkout", str(frozen.bundle_path), str(workspace)])
        _run_git(workspace, ["checkout", "--quiet", "--detach", frozen.base_commit])
        _run_git(workspace, ["apply", "--cached", "--check", str(patch_path)])
        _run_git(workspace, ["apply", "--cached", str(patch_path)])
        tree = _run_git(workspace, ["write-tree"])
        commit = _run_git(
            workspace,
            ["commit-tree", tree, "-p", frozen.base_commit, "-m", f"Evolve candidate from {frozen.freeze_id}"],
        )
        ref = f"refs/accrete/evolve/freeze-{frozen.freeze_id.removeprefix('freeze-')}"
        _run_git(refs_repo, ["fetch", "--quiet", str(workspace), f"{commit}:{ref}"])
        return ref, commit
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _require_yaml_scope(frozen: FrozenEvolve, commit: str, policy: PatchExportPolicy) -> None:
    """确认封存 commit 只改变了允许的 YAML 叶子键。"""
    if not policy.allowed_yaml_keys:
        return
    for path in policy.allowed_paths:
        before = _load_yaml_at_ref(frozen.ref_store_path, frozen.base_commit, path)
        after = _load_yaml_at_ref(frozen.ref_store_path, commit, path)
        changed = _yaml_changes(before, after)
        unexpected = sorted(changed - policy.allowed_yaml_keys)
        if unexpected:
            raise FreezeError(f"候选修改了未授权 YAML 键: {', '.join(unexpected)}")


def _load_yaml_at_ref(repository: Path, revision: str, path: str) -> Any:
    """读取指定 commit 中的 YAML 文件。"""
    content = _run_git(repository, ["show", f"{revision}:{path}"])
    try:
        return yaml.safe_load(content)
    except yaml.YAMLError as error:
        raise FreezeError(f"YAML 无法解析: {path}") from error


def _yaml_changes(before: Any, after: Any, prefix: str = "") -> set[str]:
    """返回两个 YAML 值之间发生改变的叶子键路径。"""
    if isinstance(before, dict) and isinstance(after, dict):
        changed: set[str] = set()
        for key in set(before) | set(after):
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                changed.add(name)
            else:
                changed.update(_yaml_changes(before[key], after[key], name))
        return changed
    return {prefix} if before != after else set()


def _record_rejection(
    frozen: FrozenEvolve,
    error: FreezeError,
    patch_path: Path,
    ref: str | None,
    commit: str | None,
) -> None:
    """保留候选目录，并写入封存拒绝原因。"""
    record = {
        "schema_version": 1,
        "freeze_id": frozen.freeze_id,
        "base_commit": frozen.base_commit,
        "status": "rejected",
        "reason": str(error),
        "patch": "candidate.patch" if patch_path.is_file() else None,
        "candidate_ref": ref,
        "candidate_commit": commit,
    }
    write_json(frozen.directory / "export.json", record)
    update_freeze_meta(frozen, status="rejected", export=record)


def _run_git(directory: Path, args: list[str]) -> str:
    """以固定身份执行宿主 Git 命令。"""
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "accrete-evolve",
        "GIT_AUTHOR_EMAIL": "accrete-evolve@local.invalid",
        "GIT_COMMITTER_NAME": "accrete-evolve",
        "GIT_COMMITTER_EMAIL": "accrete-evolve@local.invalid",
    }
    result = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-C", str(directory), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    if result.returncode != 0:
        raise FreezeError(result.stderr.strip() or "git 命令失败")
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
