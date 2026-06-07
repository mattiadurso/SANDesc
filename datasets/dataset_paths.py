"""Dataset path resolution from environment variables or known locations."""

import os
from pathlib import Path


def resolve_dataset_path(name: str, env_var: str, candidates: list[Path]) -> Path:
    """Resolve a dataset root directory.

    Checks the ``env_var`` environment variable first, then each candidate in
    order, returning the first path that exists.

    Args:
        name: human-readable dataset name, used in the error message.
        env_var: environment variable that may hold an override path.
        candidates: fallback paths to try, in priority order.

    Returns:
        The first existing path.

    Raises:
        FileNotFoundError: if none of the paths exist.
    """
    searched = []
    env_value = os.environ.get(env_var)
    if env_value:
        searched.append(Path(env_value))
    searched.extend(candidates)
    for path in searched:
        if path.exists():
            return path
    locations = ", ".join(str(p) for p in searched)
    raise FileNotFoundError(
        f"{name} dataset not found. Set the {env_var} environment variable or "
        f"place the data at one of: {locations}"
    )
