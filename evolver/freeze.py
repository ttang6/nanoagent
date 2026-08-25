"""Evolve freeze 的冻结输入与容器副本准备。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from infra.core.context import ContextBuilder
from infra.core.state import RunState
from infra.core.types import Message
from infra.runtime.environments.docker import DockerEnvironment

from .helpers import write_json


class FreezeError(RuntimeError):
    """freeze 准备或导出不满足版本边界。"""


@dataclass(frozen=True)
class FrozenEvolve:
    """一个已冻结但尚未运行的 Evolve 输入。"""

    directory: Path
    freeze_id: str
    base_commit: str
    source_commit: str
    source_tree: str
    bundle_path: Path
    source_snapshot_path: Path
    system_path: Path
    spec_path: Path
    claude_path: Path
    ref_store_path: Path


class EvolveContextBuilder(ContextBuilder):
    """为容器内 Evolve 提供不依赖宿主路径的环境上下文。"""

    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

    def build(self, state: RunState) -> list[Message]:
        """返回固定容器事实和已有对话。"""
        return [Message(role="system", content=self.system_prompt), *state.messages]


def prepare_freeze(
    analysis_experiment_dir: Path,
    evolve_experiment_dir: Path,
    *,
    target_source: Path,
    base_revision: str,
    repository_root: Path,
) -> FrozenEvolve:
    """冻结 target commit、bundle 与 Evolve 输入快照。"""
    analysis_experiment_dir = analysis_experiment_dir.resolve()
    evolve_experiment_dir = evolve_experiment_dir.resolve()
    source = target_source.resolve()
    spec_source = analysis_experiment_dir / "proposal" / "evolve_spec.md"
    claude_source = repository_root.resolve() / "evolver" / "assets" / "CLAUDE.md"
    system_source = repository_root.resolve() / "evolver" / "assets" / "system.md"
    if not spec_source.is_file():
        raise FreezeError(f"缺少 evolve_spec.md: {spec_source}")
    if not claude_source.is_file():
        raise FreezeError(f"缺少 CLAUDE.md: {claude_source}")
    if not system_source.is_file():
        raise FreezeError(f"缺少 Evolve system prompt: {system_source}")

    source_commit = _git(source, ["rev-parse", "--verify", f"{base_revision}^{{commit}}"])
    source_tree = _git(source, ["rev-parse", f"{source_commit}^{{tree}}"])
    freeze_id = "freeze-" + uuid.uuid4().hex[:12]
    directory = evolve_experiment_dir / "freezes" / freeze_id
    directory.mkdir(parents=True, exist_ok=False)
    bundle_path = directory / "base.bundle"
    source_snapshot_path = directory / "source_snapshot.tar"
    base_commit = _create_snapshot_bundle(
        source, source_commit, source_snapshot_path, bundle_path
    )

    input_dir = directory / "input"
    input_dir.mkdir()
    system_path = input_dir / "system.md"
    spec_path = input_dir / "evolve_spec.md"
    claude_path = input_dir / "CLAUDE.md"
    system_path.write_bytes(system_source.read_bytes())
    spec_path.write_bytes(spec_source.read_bytes())
    claude_path.write_bytes(claude_source.read_bytes())
    write_json(directory / "freeze_meta.json", {
        "schema_version": 1,
        "freeze_id": freeze_id,
        "status": "prepared",
        "base_commit": base_commit,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "target_source": str(source),
        "analysis_experiment": str(analysis_experiment_dir),
        "candidate_ref_store": str(repository_root.resolve() / ".evolve_candidates.git"),
        "base_bundle_sha256": _sha256(bundle_path),
        "source_snapshot_sha256": _sha256(source_snapshot_path),
        "inputs": {
            "system_sha256": _sha256(system_path),
            "evolve_spec_sha256": _sha256(spec_path),
            "claude_sha256": _sha256(claude_path),
        },
    })
    return FrozenEvolve(
        directory,
        freeze_id,
        base_commit,
        source_commit,
        source_tree,
        bundle_path,
        source_snapshot_path,
        system_path,
        spec_path,
        claude_path,
        repository_root.resolve() / ".evolve_candidates.git",
    )


def load_freeze(directory: Path) -> FrozenEvolve:
    """读取并校验已准备 freeze 的最小输入。"""
    directory = directory.resolve()
    meta_path = directory / "freeze_meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        freeze_id = str(meta["freeze_id"])
        base_commit = str(meta["base_commit"])
        source_commit = str(meta["source_commit"])
        source_tree = str(meta["source_tree"])
        ref_store_path = Path(str(meta["candidate_ref_store"]))
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise FreezeError(f"freeze 元数据无效: {meta_path}") from error
    frozen = FrozenEvolve(
        directory=directory,
        freeze_id=freeze_id,
        base_commit=base_commit,
        source_commit=source_commit,
        source_tree=source_tree,
        bundle_path=directory / "base.bundle",
        source_snapshot_path=directory / "source_snapshot.tar",
        system_path=directory / "input" / "system.md",
        spec_path=directory / "input" / "evolve_spec.md",
        claude_path=directory / "input" / "CLAUDE.md",
        ref_store_path=ref_store_path,
    )
    for path in (
        frozen.bundle_path,
        frozen.source_snapshot_path,
        frozen.system_path,
        frozen.spec_path,
        frozen.claude_path,
    ):
        if not path.is_file():
            raise FreezeError(f"freeze 输入缺失: {path}")
    return frozen


def bootstrap_container(environment: DockerEnvironment, frozen: FrozenEvolve) -> None:
    """在容器的 /work 中从冻结 bundle 创建完整 Git 副本。"""
    environment.copy_to_container(frozen.bundle_path, "/tmp/accrete-base.bundle")
    cloned = environment.execute("git clone --quiet --no-checkout /tmp/accrete-base.bundle .", timeout_s=120)
    if cloned.exit_code != 0:
        raise FreezeError(f"容器内 clone 失败: {cloned.stderr.strip()}")
    actual = environment.execute(f"git checkout --quiet --detach {frozen.base_commit} && git rev-parse HEAD", timeout_s=30)
    if actual.exit_code != 0 or actual.stdout.strip() != frozen.base_commit:
        raise FreezeError("容器副本的 HEAD 与冻结 base_commit 不一致")


def build_evolve_context(frozen: FrozenEvolve) -> EvolveContextBuilder:
    """构造包含冻结规格和容器事实的 Evolve 上下文。"""
    system = frozen.system_path.read_text(encoding="utf-8").strip()
    spec = frozen.spec_path.read_text(encoding="utf-8").strip()
    claude = frozen.claude_path.read_text(encoding="utf-8").strip()
    prompt = "\n\n".join([
        system,
        "# 容器事实\n工作目录是 /work，且它是一个已冻结基线的可写 Git 副本。"
        "不要访问或假定存在宿主文件、评测材料或 heldout 结果。",
        "# Evolve 规格\n" + spec,
        "# 通用实现约定\n" + claude,
    ])
    return EvolveContextBuilder(prompt)


def update_freeze_meta(frozen: FrozenEvolve, **changes: object) -> None:
    """更新 freeze 元数据。"""
    path = frozen.directory / "freeze_meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update(changes)
    write_json(path, meta)


def _git(repository: Path, args: list[str]) -> str:
    """在宿主 Git 仓库执行只读或准备命令。"""
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repository.resolve()}", "-C", str(repository), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise FreezeError(result.stderr.strip() or "git 命令失败")
    return result.stdout.strip()


def _create_snapshot_bundle(
    source: Path,
    source_commit: str,
    snapshot_path: Path,
    bundle_path: Path,
) -> str:
    """从本地 tracked 文件快照创建供容器 diff 的独立 Git 基线。"""
    _create_source_snapshot(source, source_commit, snapshot_path)
    temporary_repo = bundle_path.parent / ".snapshot-source"
    try:
        temporary_repo.mkdir()
        with tarfile.open(snapshot_path) as archive:
            archive.extractall(temporary_repo, filter="data")
        _git(temporary_repo, ["init", "--quiet"])
        _git(temporary_repo, ["config", "user.name", "accrete-evolve"])
        _git(temporary_repo, ["config", "user.email", "accrete-evolve@local.invalid"])
        _git(temporary_repo, ["add", "--all"])
        _git(temporary_repo, ["commit", "--quiet", "-m", "Frozen target snapshot"])
        base_commit = _git(temporary_repo, ["rev-parse", "HEAD"])
        _git(temporary_repo, ["bundle", "create", str(bundle_path), "HEAD"])
        return base_commit
    finally:
        shutil.rmtree(temporary_repo, ignore_errors=True)


def _create_source_snapshot(source: Path, source_commit: str, destination: Path) -> None:
    """把 target 指定 commit 的 tracked 文件导出为离线 tar 快照。"""
    with destination.open("wb") as archive:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={source.resolve()}", "-C", str(source), "archive", "--format=tar", source_commit],
            stdout=archive,
            stderr=subprocess.PIPE,
        )
    if result.returncode != 0:
        raise FreezeError(result.stderr.decode("utf-8", errors="replace").strip() or "无法创建 source snapshot")


def _sha256(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
