"""在独立 Docker 容器中运行一个已准备的 Evolve 候选。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evolver.freeze import (
    bootstrap_container,
    build_evolve_context,
    load_freeze,
    update_freeze_meta,
)
from evolver.helpers import write_json
from evolver.patch_export import ExportResult, PatchExportPolicy, export_candidate
from infra.core.tools import ToolRegistry
from infra.core.types import RunLimits, RunResult
from infra.runtime.environments import DockerEnvironment
from infra.runtime.providers.openai_compat import OpenAICompatProvider
from infra.runtime.runner import run_agent
from infra.runtime.tools import BashTool, EditTool, ReadTool, WriteTool


MINI_SWE_EXPORT_POLICY = PatchExportPolicy(
    allowed_paths=frozenset(),
    allowed_yaml_keys=frozenset(),
    allowed_path_prefixes=("src/minisweagent/agents/", "tests/"),
)

EVOLVE_DEFAULT_MODEL = "gpt-5.6-terra"
EVOLVE_DEFAULT_MAX_TURNS = 100
EVOLVE_CONTEXT_WINDOW_TOKENS = 1_000_000
EVOLVE_RESERVED_ANSWER_TOKENS = 100_000
EVOLVE_DEFAULT_MAX_OUTPUT_TOKENS = 8_192
EVOLVE_DEFAULT_WALL_TIME_S = 1_800
EVOLVE_DEFAULT_PROFILE = "deepseek-v4-pro"

EVOLVE_PROFILES: dict[str, dict[str, Any]] = {
    "terra": {
        "model": EVOLVE_DEFAULT_MODEL,
        "reasoning_effort": "none",
        "default_extra": {},
    },
    "grok-4.6-medium": {
        "model": "grok-4.6",
        "reasoning_effort": "medium",
        "default_extra": {},
    },
    "deepseek-v4-pro": {
        "model": "deepseek-v4-pro",
        "reasoning_effort": "high",
        "default_extra": {"thinking": {"type": "enabled"}},
    },
    "deepseek-v4-pro-low": {
        "model": "deepseek-v4-pro",
        "reasoning_effort": "low",
        "default_extra": {"thinking": {"type": "enabled"}},
    },
    "deepseek-v4-pro-thinking-off": {
        "model": "deepseek-v4-pro",
        "reasoning_effort": "none",
        "default_extra": {"thinking": {"type": "disabled"}},
    },
}


def main() -> None:
    """运行 freeze 并导出尚待 Manifest Gate 审核的 Git ref。"""
    parser = argparse.ArgumentParser(description="运行一个已准备的 Evolve freeze")
    parser.add_argument("freeze_dir", type=Path)
    parser.add_argument("--profile", choices=EVOLVE_PROFILES, default=EVOLVE_DEFAULT_PROFILE)
    parser.add_argument("--image", default="accrete-evolve:py311")
    parser.add_argument("--model")
    parser.add_argument("--max-turns", type=int, default=EVOLVE_DEFAULT_MAX_TURNS)
    parser.add_argument("--context-window-tokens", type=int, default=EVOLVE_CONTEXT_WINDOW_TOKENS)
    parser.add_argument("--reserved-answer-tokens", type=int, default=EVOLVE_RESERVED_ANSWER_TOKENS)
    parser.add_argument("--max-output-tokens", type=int, default=EVOLVE_DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--wall-time-s", type=float, default=EVOLVE_DEFAULT_WALL_TIME_S)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.max_turns <= 0 or args.context_window_tokens <= 0 or args.max_output_tokens <= 0 or args.wall_time_s <= 0:
        raise SystemExit("轮数、上下文窗口和最大输出必须为正整数")
    if not 0 <= args.reserved_answer_tokens < args.context_window_tokens:
        raise SystemExit("--reserved-answer-tokens 必须不小于零且小于上下文窗口")
    profile = EVOLVE_PROFILES[args.profile]
    model = args.model or profile["model"]
    reasoning_effort = args.reasoning_effort or profile["reasoning_effort"]

    load_dotenv()
    frozen = load_freeze(args.freeze_dir)
    environment = DockerEnvironment(args.image, workspace_root="/work", network="none")
    provider = OpenAICompatProvider(
        model,
        timeout_s=300,
        default_max_tokens=args.max_output_tokens,
        default_reasoning_effort=reasoning_effort,
        default_extra=profile["default_extra"],
    )
    try:
        started_at = _utc_now()
        bootstrap_container(environment, frozen)
        tools = _build_tools(environment)
        result = run_agent(
            "请按系统中的 Evolve 规格完成仓库探查、必要修改与可见测试。"
            "若仓库证据不支持该方向，明确记录否决依据后停止。",
            provider=provider,
            tools=tools,
            workdir="/work",
            artifacts_root=args.artifacts_root,
            artifact_namespace=f"evolve/{frozen.freeze_id}",
            limits=RunLimits(
                max_turns=args.max_turns,
                context_window_tokens=args.context_window_tokens,
                reserved_answer_tokens=args.reserved_answer_tokens,
                wall_time_s=args.wall_time_s,
            ),
            environment=environment,
            close_resources=False,
            context_builder=build_evolve_context(frozen),
        )
        update_freeze_meta(
            frozen,
            run={
                "finish_reason": result.finish_reason,
                "turns": result.turns,
                "error": result.error,
                "profile": args.profile,
                "model": model,
                "image": args.image,
                "context_window_tokens": args.context_window_tokens,
                "reserved_answer_tokens": args.reserved_answer_tokens,
                "max_output_tokens": args.max_output_tokens,
                "wall_time_s": args.wall_time_s,
                "reasoning_effort": reasoning_effort,
                "thinking": _thinking_config(profile, reasoning_effort),
            },
        )
        patch = export_candidate(
            environment,
            frozen,
            policy=MINI_SWE_EXPORT_POLICY,
        )
        ended_at = _utc_now()
        _write_run_summary(
            frozen.directory / "run_summary.json",
            frozen_id=frozen.freeze_id,
            base_commit=frozen.base_commit,
            profile=args.profile,
            model=model,
            image=args.image,
            max_turns=args.max_turns,
            context_window_tokens=args.context_window_tokens,
            reserved_answer_tokens=args.reserved_answer_tokens,
            max_output_tokens=args.max_output_tokens,
            wall_time_s=args.wall_time_s,
            reasoning_effort=reasoning_effort,
            thinking=_thinking_config(profile, reasoning_effort),
            artifacts_root=args.artifacts_root,
            started_at=started_at,
            ended_at=ended_at,
            result=result,
            exported=patch,
        )
        update_freeze_meta(frozen, run_summary="run_summary.json")
        print(f"candidate status: {patch.status}")
        if patch.patch_path is not None:
            print(f"candidate patch: {patch.patch_path}")
        if patch.candidate_ref is not None:
            print(f"candidate ref: {patch.candidate_ref}")
        print(f"candidate export: {frozen.directory / 'export.json'}")
    finally:
        provider.close()
        environment.close()


def _build_tools(environment: DockerEnvironment) -> ToolRegistry:
    """只注册 Evolve 所需的本地 workspace 工具。"""
    registry = ToolRegistry()
    for tool_type in (BashTool, ReadTool, WriteTool, EditTool):
        registry.register(tool_type(environment))
    return registry


def _write_run_summary(
    path: Path,
    *,
    frozen_id: str,
    base_commit: str,
    profile: str,
    model: str,
    image: str,
    max_turns: int,
    context_window_tokens: int,
    reserved_answer_tokens: int,
    max_output_tokens: int,
    wall_time_s: float,
    reasoning_effort: str,
    thinking: dict[str, Any],
    artifacts_root: Path,
    started_at: str,
    ended_at: str,
    result: RunResult,
    exported: ExportResult,
) -> None:
    """写入宿主侧可检索的 Evolve 运行和封存摘要。"""
    trace_root = artifacts_root / "evolve" / frozen_id
    write_json(path, {
        "schema_version": 1,
        "freeze_id": frozen_id,
        "base_commit": base_commit,
        "run": {
            "profile": profile,
            "model": model,
            "image": image,
            "max_turns": max_turns,
            "context_window_tokens": context_window_tokens,
            "reserved_answer_tokens": reserved_answer_tokens,
            "history_token_limit": context_window_tokens - reserved_answer_tokens,
            "max_output_tokens": max_output_tokens,
            "wall_time_s": wall_time_s,
            "reasoning_effort": reasoning_effort,
            "thinking": thinking,
            "started_at": started_at,
            "ended_at": ended_at,
            "finish_reason": result.finish_reason,
            "turns": result.turns,
            "usage": asdict(result.usage),
            "error": result.error,
            "trace_root": str(trace_root.resolve()),
        },
        "export": {
            "status": exported.status,
            "patch": str(exported.patch_path) if exported.patch_path is not None else None,
            "candidate_ref": exported.candidate_ref,
        },
    })


def _utc_now() -> str:
    """返回 UTC 的 ISO 8601 时间戳。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _thinking_config(profile: dict[str, Any], reasoning_effort: str) -> dict[str, Any]:
    """返回本次请求实际采用的思考配置。"""
    return {
        "reasoning_effort": reasoning_effort,
        "provider_extra": profile["default_extra"],
    }


if __name__ == "__main__":
    main()
