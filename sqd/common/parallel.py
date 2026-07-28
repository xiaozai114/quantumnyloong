"""线程级并行工具（numpy/jax 后端均适用；JAX 算子释放 GIL）。

- ``parallel_map(fn, items)``：保序并行映射，用 ThreadPoolExecutor。
- 每个工作线程经 ``threadpoolctl`` 限制 BLAS/OpenMP 线程数（默认 1），避免
  「Python 线程数 × BLAS 线程数」超订 CPU（8 核机器跑 8 碎片并行时尤为关键）。

典型用途：
- 碎片级并行（Q10/Q13/Q14：每个团簇的 LUCJ 采样+SQD 对角化相互独立）。
- 扫描级并行（Q11 λ-扫描、Q12 噪声扫描、Q13 η-扫描：每个扫描点独立）。
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

from common.backend import N_CPUS


def _blas_limiter(limit: int):
    """返回一个上下文管理器，限制当前线程 BLAS/OpenMP 线程数。"""
    try:
        import threadpoolctl
        return threadpoolctl.threadpool_limits(limits=limit)
    except Exception:
        return nullcontext()


def parallel_map(fn, items, max_workers: int | None = None,
                 blas_limit: int = 1):
    """保序并行映射 fn(items[i])。

    Parameters
    ----------
    fn        : 单参函数（每个 item 调用一次）
    items     : 可迭代输入
    max_workers : 最大并行度；None = min(len(items), N_CPUS)
    blas_limit  : 每工作线程 BLAS 线程数上限（1 避免超订）
    """
    items = list(items)
    n = len(items)
    if max_workers is None:
        max_workers = min(max(1, n), N_CPUS)
    if n <= 1 or max_workers <= 1:
        return [fn(x) for x in items]

    results = [None] * n

    def _worker(i):
        with _blas_limiter(blas_limit):
            results[i] = fn(items[i])

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # ex.map 保序；逐个提交保证 results[i] 对应 items[i]
        list(ex.map(_worker, range(n)))
    return results
