"""SQD 流程层：采样 -> 配置恢复 -> 子空间对角化。

子空间结构对齐 qiskit-addon-sqd：恢复后的 bitstring 分解为 α/β 自旋串集合，
子空间 = α串 × β串 笛卡尔积（闭壳层合并为同一集合）。

- config_recovery        : 修正违反粒子数的 bitstring
                           （method="max_dev" 默认 | "directed"）
- config_recovery_counts : 同上，保留恢复后计数
- bitstrings_to_ci_strs  : counts -> (α串集, β串集)，按权重降序
- run_sqd                : 给定行列式列表的 Slater-Condon 对角化
- run_sqd_product        : α×β 笛卡尔积子空间对角化（含 max_dets 截断）
- sqd_from_counts        : counts -> 恢复 -> 乘积子空间 SQD（端到端便捷入口）
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from common.ci import int_to_bitstring, solve_subspace
from common.circuits import sample_counts


def _avg_occupancies(counts: Dict[int, int], nq: int) -> np.ndarray:
    """按计数加权的逐比特平均占据 n̄_i。"""
    tot = sum(counts.values())
    if tot == 0:
        return np.zeros(nq)
    avg = np.zeros(nq)
    for key, cnt in counts.items():
        bs = int_to_bitstring(key, nq)
        for q in range(nq):
            if bs[q] == "1":
                avg[q] += cnt
    return avg / tot


def _recover_one_directed(bs: str, nq: int, n_alpha: int, n_beta: int,
                          avg: np.ndarray):
    """定向翻转（method="directed"，默认）：返回 (bitstring, converged=True)。

    对每个自旋分别处理：粒子偏多时在占据位中翻掉 n̄ 最小的（后验错误概率
    最大的错误占据），偏少时在空位中翻上 n̄ 最大的——每步使
    Φ = |n_α−n_α*| + |n_β−n_β*| 严格减 1，≤ Φ₀ 步必收敛，输出必在目标扇区。
    """
    bs = list(bs)
    for q0, target in ((0, n_alpha), (1, n_beta)):
        while True:
            occ = [q for q in range(q0, nq, 2) if bs[q] == "1"]
            if len(occ) == target:
                break
            if len(occ) > target:
                q = min(occ, key=lambda x: avg[x])
            else:
                vir = [q for q in range(q0, nq, 2) if bs[q] == "0"]
                q = max(vir, key=lambda x: avg[x])
            bs[q] = "1" if bs[q] == "0" else "0"
    return "".join(bs), True


def _recover_one_max_dev(bs: str, nq: int, n_alpha: int, n_beta: int,
                         avg: np.ndarray):
    """最大误差翻转（method="max_dev"）：返回 (bitstring, converged)。

    反复翻转 |b_i − n̄_i| 最大的比特（贪心最大似然）。势函数 Φ 无单调性
    保证——高噪声下 n̄ 平坦化时可能翻错方向（Φ 增 1），超过 nq+2 步仍未
    收敛时返回 converged=False（输出可能不在目标扇区）。
    """
    bs = list(bs)

    def nab():
        na = sum(1 for q in range(0, nq, 2) if bs[q] == "1")
        nb = sum(1 for q in range(1, nq, 2) if bs[q] == "1")
        return na, nb

    na, nb = nab()
    guard = 0
    while (na != n_alpha or nb != n_beta) and guard < nq + 2:
        guard += 1
        best_q, best_dev = None, -1.0
        for q in range(nq):
            dev = abs(int(bs[q]) - avg[q])
            if dev > best_dev:
                best_dev, best_q = dev, q
        if best_q is None:
            break
        bs[best_q] = "1" if bs[best_q] == "0" else "0"
        na, nb = nab()
    return "".join(bs), (na == n_alpha and nb == n_beta)


_RECOVERERS = {
    "directed": _recover_one_directed,
    "max_dev": _recover_one_max_dev,
}


def config_recovery_counts(counts: Dict[int, int], nq: int, n_alpha: int,
                           n_beta: int, method: str = "max_dev",
                           return_stats: bool = False):
    """配置恢复并保留计数：返回恢复后构型的 {int_key: count}。

    Args:
        method : 翻转策略——"max_dev"（最大误差翻转，贪心最大似然，默认，
                 与 qiskit-addon-sqd 的 recover_configurations 一致）或
                 "directed"（定向翻转，保证收敛）。原理对比见 q03/solution.md。
                 max_dev 高噪声下可能不收敛（输出落在错误扇区），下游
                 bitstrings_to_ci_strs 会过滤掉扇区外构型保证子空间合法。
        return_stats : 为 True 时额外返回统计 dict（收敛率等，供对比实验）。

    每个采样 bitstring 独立恢复到 (n_alpha, n_beta) 扇区后，把其计数累加到
    恢复后的构型上；合法构型的计数得以保留，供 α/β 串按权重排序。
    """
    recover = _RECOVERERS[method]
    avg = _avg_occupancies(counts, nq)
    out: Dict[int, int] = {}
    n_ok = n_tot = 0
    for key, cnt in counts.items():
        rec, ok = recover(int_to_bitstring(key, nq), nq, n_alpha, n_beta, avg)
        rkey = int(rec[::-1], 2)  # bitstring[q]=qubit q -> int（LSB=qubit0）
        out[rkey] = out.get(rkey, 0) + cnt
        n_ok += ok * cnt
        n_tot += cnt
    if return_stats:
        return out, dict(converge_rate=n_ok / n_tot if n_tot else 1.0)
    return out


def config_recovery(counts: Dict[int, int], nq: int, n_alpha: int,
                    n_beta: int, method: str = "max_dev"):
    """配置恢复：修正违反 (N_alpha, N_beta) 约束的 bitstring。

    Args:
        method : 翻转策略 "max_dev"（默认，贪心最大似然）或 "directed"
                 （定向翻转，保证收敛）。
    Returns:
        valid_bitstrings : 去重后的合法 JW bitstring 列表
    """
    rc = config_recovery_counts(counts, nq, n_alpha, n_beta, method=method)
    return [int_to_bitstring(k, nq) for k in rc]


def bitstrings_to_ci_strs(counts: Dict[int, int], nq: int,
                          open_shell: bool = False,
                          n_alpha: int = None, n_beta: int = None):
    """恢复后 counts -> (α串列表, β串列表)（空间轨道整数表示，按权重降序）。

    JW 约定：偶数 qubit=α、奇数 qubit=β（交错）。
    open_shell=False（闭壳层）：α、β 串合并为同一唯一集合用于两个自旋扇区
    （qiskit-addon-sqd 对单重态的做法），保持自旋交换对称、维度减半。
    n_alpha/n_beta：若给定，过滤掉不在目标扇区的构型（max_dev 不收敛时
    保证子空间合法性，与 qiskit recover_configurations 保证输出在扇区一致）。
    """
    norb = nq // 2

    def _popcount(x):
        c = 0
        while x:
            c += x & 1
            x >>= 1
        return c

    wa: Dict[int, int] = {}
    wb: Dict[int, int] = {}
    for key, cnt in counts.items():
        bs = int_to_bitstring(key, nq)
        a = b = 0
        for i in range(norb):
            if bs[2 * i] == "1":
                a |= 1 << i
            if bs[2 * i + 1] == "1":
                b |= 1 << i
        # 扇区过滤：max_dev 不收敛时丢弃落在错误扇区的构型
        if n_alpha is not None and _popcount(a) != n_alpha:
            continue
        if n_beta is not None and _popcount(b) != n_beta:
            continue
        wa[a] = wa.get(a, 0) + cnt
        wb[b] = wb.get(b, 0) + cnt
    if not open_shell:
        w: Dict[int, int] = dict(wa)
        for k, v in wb.items():
            w[k] = w.get(k, 0) + v
        u = sorted(w, key=lambda k: (-w[k], k))
        return u, u
    return (sorted(wa, key=lambda k: (-wa[k], k)),
            sorted(wb, key=lambda k: (-wb[k], k)))


def ci_strs_to_determinants(a_strs, b_strs, nq: int) -> List[str]:
    """α×β 笛卡尔积展开为 JW bitstring 行列式列表。"""
    norb = nq // 2
    dets = []
    for a in a_strs:
        for b in b_strs:
            key = 0
            for i in range(norb):
                if (a >> i) & 1:
                    key |= 1 << (2 * i)
                if (b >> i) & 1:
                    key |= 1 << (2 * i + 1)
            dets.append(int_to_bitstring(key, nq))
    return dets


def run_sqd(h1e, eri, n_spin: int, ecore: float, basis: List[str]):
    """在给定行列式子空间上运行 SQD（Slater-Condon 精确对角化）。"""
    E, evals, evecs = solve_subspace(basis, h1e, eri, n_spin, ecore)
    return dict(E_sqd=E, evals=evals, evecs=evecs, basis=basis, M=len(basis))


def run_sqd_product(h1e, eri, n_spin: int, ecore: float, a_strs, b_strs,
                    max_dets: int = 8000, include=None):
    """在 α×β 笛卡尔积子空间运行 SQD（qiskit-addon-sqd 的子空间结构）。

    Args:
        a_strs, b_strs : α/β 自旋串（空间轨道整数），已按权重降序。
        max_dets       : 子空间行列式数上限；超出时按权重截断 α/β 串
                         （对应 qiskit 的 max_dim 自旋扇区限制）。
        include        : 必须包含的 JW bitstring（如 HF），其 α/β 串强制入列。
    """
    a_strs, b_strs = list(a_strs), list(b_strs)
    if include:
        norb = n_spin // 2
        for bs in include:
            a = sum(1 << i for i in range(norb) if bs[2 * i] == "1")
            b = sum(1 << i for i in range(norb) if bs[2 * i + 1] == "1")
            if a not in a_strs:
                a_strs.insert(0, a)
            if b not in b_strs:
                b_strs.insert(0, b)
    if max_dets is not None and len(a_strs) * len(b_strs) > max_dets:
        ka = max(1, int(max_dets ** 0.5))
        a_strs = a_strs[:ka]
        kb = max(1, max_dets // len(a_strs))
        b_strs = b_strs[:kb]
    basis = ci_strs_to_determinants(a_strs, b_strs, n_spin)
    res = run_sqd(h1e, eri, n_spin, ecore, basis)
    res["n_ci_strs_a"] = len(a_strs)
    res["n_ci_strs_b"] = len(b_strs)
    return res


def sqd_from_counts(counts: Dict[int, int], nq: int, n_alpha: int, n_beta: int,
                    h1e, eri, ecore: float, hf_bs: str = None,
                    max_dets: int = 8000, method: str = "max_dev"):
    """counts -> 配置恢复 -> α/β 串 -> 笛卡尔积子空间 SQD（标准采样路径）。

    Args:
        method : 配置恢复翻转策略（默认 "max_dev"，与 qiskit 一致）。
    返回 dict 含 E_sqd、M（乘积行列式数）、n_recovered（恢复构型数）、
    n_ci_strs_a/b（α/β 串数）。
    """
    rc = config_recovery_counts(counts, nq, n_alpha, n_beta, method=method)
    # 扇区过滤保证子空间合法（max_dev 不收敛时丢弃错误扇区构型）
    a_strs, b_strs = bitstrings_to_ci_strs(rc, nq, n_alpha=n_alpha, n_beta=n_beta)
    include = [hf_bs] if hf_bs else None
    res = run_sqd_product(h1e, eri, nq, ecore, a_strs, b_strs,
                          max_dets=max_dets, include=include)
    res["n_recovered"] = len(rc)
    return res


def sampling_sqd(circuit, n_shots: int, h1e, eri, nq: int, n_alpha: int,
                 n_beta: int, ecore: float, augment_full: bool = False,
                 norb: int = None, seed=None, hf_bs: str = None,
                 max_dets: int = 8000):
    """端到端：采样 -> 配置恢复 -> α×β 乘积子空间 SQD 对角化。

    augment_full：若为真，把 FCI 全空间构型并入子空间（用于演示 HF 态下
    仅 1 构型 -> 退化为 HF，全空间 -> 复现 FCI）。
    """
    counts = sample_counts(circuit, n_shots, nq, seed=seed)
    rc = config_recovery_counts(counts, nq, n_alpha, n_beta)
    a_strs, b_strs = bitstrings_to_ci_strs(rc, nq)
    include = [hf_bs] if hf_bs else None
    res = run_sqd_product(h1e, eri, nq, ecore, a_strs, b_strs,
                          max_dets=max_dets, include=include)
    if augment_full and norb is not None:
        full = set(res["basis"])
        for bs in full_fci_basis_local(norb, n_alpha, n_beta):
            full.add(bs)
        res = run_sqd(h1e, eri, nq, ecore, list(full))
    res["counts"] = counts
    res["n_recovered"] = len(rc)
    return res


def full_fci_basis_local(norb, n_alpha, n_beta):
    from common.ci import full_fci_basis
    return full_fci_basis(norb, n_alpha, n_beta)
