"""SQD 费米子求解器：从 h1e/eri 用 Slater-Condon 规则构建 CI 矩阵。

依赖 numpy/scipy。JW 约定：
  - 偶数索引 qubit (0,2,4,...) -> alpha 自旋轨道
  - 奇数索引 qubit (1,3,5,...) -> beta  自旋轨道
  - bitstring[0] 对应 qubit 0（LSB-first）
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.linalg import eigh

# ====================================================================== #
# 1. JW bitstring 解析与整数键转换
# ====================================================================== #


def int_to_bitstring(key: int, nq: int) -> str:
    """TC 采样整数 key -> JW bitstring（qubit 0 在索引 0 / 最左）。

    TC 的 count_dict_int 中整数 key 的 LSB 是 qubit 0，故需反转。
    """
    return format(key, f"0{nq}b")[::-1]


def parse_jw_bitstring(bitstring: str, n_spin: int):
    bs = bitstring.zfill(n_spin)
    alpha_occ, beta_occ = [], []
    for q in range(n_spin):
        if bs[q] == '1':
            spatial = q // 2
            (alpha_occ if q % 2 == 0 else beta_occ).append(spatial)
    return (sorted(alpha_occ), sorted(beta_occ), len(alpha_occ), len(beta_occ))


# ====================================================================== #
# 2. Slater-Condon CI 矩阵构建
# ====================================================================== #


def build_ci_hamiltonian(basis: List[str], h1e, eri, n_spin, ecore=0.0):
    n_basis = len(basis)
    parsed = [parse_jw_bitstring(bs, n_spin) for bs in basis]
    H = np.zeros((n_basis, n_basis))
    for i in range(n_basis):
        a_i, b_i, na_i, nb_i = parsed[i]
        sa_i, sb_i = set(a_i), set(b_i)
        for j in range(i, n_basis):
            a_j, b_j, na_j, nb_j = parsed[j]
            sa_j, sb_j = set(a_j), set(b_j)
            if na_i != na_j or nb_i != nb_j:
                continue
            aexc = len(sa_i - sa_j)
            bexc = len(sb_i - sb_j)
            nexc = aexc + bexc
            if nexc > 2:
                continue
            if nexc == 0:
                h = _diag(sa_i, sb_i, h1e, eri)
            elif nexc == 1:
                h = _single(sa_i, sb_i, sa_j, sb_j, h1e, eri)
            else:
                h = _double(sa_i, sb_i, sa_j, sb_j, h1e, eri)
            H[i, j] = h
            H[j, i] = h
    H += ecore * np.eye(n_basis)
    return H


def _diag(alpha, beta, h1e, h2e):
    E = 0.0
    al, be = sorted(alpha), sorted(beta)
    for p in al:
        E += h1e[p, p]
    for p in be:
        E += h1e[p, p]
    for p in al:
        for q in al:
            E += 0.5 * (h2e[p, p, q, q] - h2e[p, q, q, p])
    for p in be:
        for q in be:
            E += 0.5 * (h2e[p, p, q, q] - h2e[p, q, q, p])
    for p in al:
        for q in be:
            E += h2e[p, p, q, q]
    return E


def _single(sa_i, sb_i, sa_j, sb_j, h1e, h2e):
    # 判断激发发生在哪条自旋
    alpha_same = (len(sa_i - sa_j) == 1 and len(sb_i - sb_j) == 0)
    spin = 'alpha' if alpha_same else 'beta'
    if spin == 'alpha':
        occ_i, occ_j, diff = sa_i, sa_j, sb_i
    else:
        occ_i, occ_j, diff = sb_i, sb_j, sa_i
    p = sorted(occ_i - occ_j)[0]
    q = sorted(occ_j - occ_i)[0]
    sign = _sign(occ_i, occ_j, p, q)
    E = sign * h1e[p, q]
    for r in occ_i:
        if r != p:
            E += sign * (h2e[p, q, r, r] - h2e[p, r, r, q])
    for r in diff:
        E += sign * h2e[p, q, r, r]
    return E


def _double(sa_i, sb_i, sa_j, sb_j, h1e, h2e):
    aexc = len(sa_i - sa_j)
    if aexc == 2:
        p_list, q_list, occ_i, occ_j = sa_i, sa_j, sa_i, sa_j
        p1, p2 = sorted(p_list - q_list)
        q1, q2 = sorted(q_list - p_list)
        s1 = _sign(occ_i, occ_j, p1, q1)
        s2 = _sign(occ_i, occ_j, p2, q2)
        return s1 * s2 * (h2e[p1, q1, p2, q2] - h2e[p1, q2, p2, q1])
    if aexc == 0:
        p_list, q_list = sb_i, sb_j
        p1, p2 = sorted(p_list - q_list)
        q1, q2 = sorted(q_list - p_list)
        s1 = _sign(p_list, q_list, p1, q1)
        s2 = _sign(p_list, q_list, p2, q2)
        return s1 * s2 * (h2e[p1, q1, p2, q2] - h2e[p1, q2, p2, q1])
    # alpha-beta
    p = sorted(sa_i - sa_j)[0]
    q = sorted(sa_j - sa_i)[0]
    r = sorted(sb_i - sb_j)[0]
    s = sorted(sb_j - sb_i)[0]
    sa = _sign(sa_i, sa_j, p, q)
    sb = _sign(sb_i, sb_j, r, s)
    return sa * sb * h2e[p, q, r, s]


def _sign(occ_i, occ_j, p, q):
    si, sj = sorted(occ_i), sorted(occ_j)
    return (-1) ** abs(si.index(p) - sj.index(q))


# ====================================================================== #
# 3. 一站式求解
# ====================================================================== #


def solve_subspace(basis: List[str], h1e, eri, n_spin, ecore=0.0):
    H = build_ci_hamiltonian(basis, h1e, eri, n_spin, ecore)
    evals, evecs = eigh(H)
    return float(evals[0]), evals, evecs


def full_fci_basis(norb, n_alpha, n_beta):
    """生成 FCI 全子空间所有 JW bitstring（按激发序），用作 SQD 全采样基准。"""
    from itertools import combinations
    nq = 2 * norb
    alpha_qubits = list(range(0, nq, 2))
    beta_qubits = list(range(1, nq, 2))
    basis = []
    for ac in combinations(alpha_qubits, n_alpha):
        for bc in combinations(beta_qubits, n_beta):
            occ = sorted(list(ac) + list(bc))
            key = 0
            for qb in occ:
                key |= (1 << qb)
            basis.append(int_to_bitstring(key, nq))
    return basis
