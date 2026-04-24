from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

AGENTICLAB_ROOT_ENV = "AGENTICLAB_ROOT"
AGENTICLAB_OUTPUT_DIR_ENV = "AGENTICLAB_OUTPUT_DIR"


@lru_cache
def repo_root() -> Path:
    env_root = os.getenv(AGENTICLAB_ROOT_ENV)
    if env_root:
        return Path(env_root).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (
            parent / "src" / "agenticlab"
        ).exists():
            return parent

    raise RuntimeError(
        "Could not locate AgenticLab repo root. "
        "Set AGENTICLAB_ROOT to the repository root."
    )


def resolve_repo_path(path: str | Path, *, must_exist: bool = False) -> Path:
    candidate = Path(path).expanduser()
    resolved = candidate if candidate.is_absolute() else repo_root() / candidate
    resolved = resolved.resolve()

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")

    return resolved


def config_path(*parts: str, must_exist: bool = True) -> Path:
    return resolve_repo_path(Path("configs", *parts), must_exist=must_exist)


def prompt_path(name: str, must_exist: bool = True) -> Path:
    return config_path("prompts", name, must_exist=must_exist)


def data_path(*parts: str, must_exist: bool = True) -> Path:
    return resolve_repo_path(Path("data", *parts), must_exist=must_exist)


def third_party_path(*parts: str, must_exist: bool = False) -> Path:
    return resolve_repo_path(Path("third_party", *parts), must_exist=must_exist)


def output_path(path: str | Path) -> Path:
    base = os.getenv(AGENTICLAB_OUTPUT_DIR_ENV)
    if base and not Path(path).is_absolute():
        resolved = Path(base).expanduser().resolve() / path
    else:
        resolved = resolve_repo_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
