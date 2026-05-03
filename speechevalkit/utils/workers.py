from __future__ import annotations

import os


def resolve_num_workers(num_workers: int | None = None) -> int:
    """
    Resolve worker count based on CPU capacity.

    Parameters
    ----------
    num_workers:
        User-requested number of workers. If None or <= 0, auto-detect.

    Returns
    -------
    int
        Number of workers.
    """
    cpu_count = os.cpu_count() or 1

    if num_workers is None or num_workers <= 0:
        return max(1, cpu_count - 1)

    return max(1, min(num_workers, cpu_count))
