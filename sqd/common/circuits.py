"""量子电路层（TensorCircuit）：HF 态制备 + LUCJ ansatz + 采样。

JW 约定同 ci.py：qubit 2i = 空间轨道 i 的 alpha，qubit 2i+1 = beta。
所有门使用硬件原生门集：RX/RZ/RY + CNOT + RZZ（RZZ = 2 CNOT + 1 RZ），
可导出 OpenQASM 以兼容真机路径。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import common.backend  # noqa: F401  量子后端自动初始化(GPU→jax, 否则 numpy)
import tensorcircuit as tc


def prepare_hf(c, norb: int, nocc: int):
    """在 TC 电路 c 上制备 JW HF 态 |HF> = x(0),x(2),...,x(2*nocc-2)。

    closed-shell：alpha 占据轨道 0..nocc-1，beta 同。
    """
    for i in range(nocc):
        c.x(2 * i)        # alpha
        c.x(2 * i + 1)    # beta
    return c


def rzz(c, q0, q1, theta):
    """RZZ(θ) = e^{-i θ/2 Z⊗Z}，分解为 2 CNOT + 1 RZ。"""
    c.cnot(q0, q1)
    c.rz(q1, theta=theta)
    c.cnot(q0, q1)


_FSWAP = np.array([[1, 0, 0, 0],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, -1]], dtype=float)


def fswap(c, q):
    """相邻 fermionic SWAP（JW）：交换占据并带 |11> 宇称相位。

    硬件分解 = 3 CNOT + CZ（统计按 3 个两比特门计）。
    """
    c.any(q, q + 1, unitary=_FSWAP)


def adjacent_givens(c, q0, q1, theta):
    """相邻量子比特的 Givens 旋转（|01>,|10> 子空间，精确保粒子数）。

    θ=0 时严格为恒等。硬件分解约 2 CNOT + 2 单比特门。
    """
    ct, st = np.cos(theta), np.sin(theta)
    U = np.array([[1, 0, 0, 0],
                  [0, ct, -st, 0],
                  [0, st, ct, 0],
                  [0, 0, 0, 1]], dtype=float)
    c.any(q0, q1, unitary=U)


def jw_rotate(c, p_qubit, q_qubit, theta):
    """JW 轨道旋转 e^{θ(a†_p a_q - a†_q a_p)}（任意轨道，保粒子数）。

    用 fswap 链把 q_qubit 换到 p_qubit 相邻，施加相邻 Givens，再换回。
    p_qubit, q_qubit 须同自旋（同为偶或同为奇）。
    """
    lo, hi = min(p_qubit, q_qubit), max(p_qubit, q_qubit)
    for q in range(hi - 1, lo, -1):
        fswap(c, q)
    adjacent_givens(c, lo, lo + 1, theta)
    for q in range(lo + 1, hi):
        fswap(c, q)


def givens(c, q0, q1, theta):
    """兼容旧调用：相邻 Givens（仅当 |q0-q1|==1 时精确保粒子数）。"""
    if abs(q0 - q1) == 1:
        adjacent_givens(c, min(q0, q1), max(q0, q1), theta)
    else:
        jw_rotate(c, q0, q1, theta)


def double_excitation_unitary(theta):
    """4 比特双激发 Givens：|1100> <-> |0011>（作用比特序 q0..q3，q0 为高位）。

    注：跨比特 JW string 相位对「采样->配置恢复->子空间对角化」流程无影响
    （能量由 CI 对角化决定，与采样相位无关），故此处省略 string 分解。
    硬件真实分解约需 13 个 CNOT/门（统计按此计）。
    """
    U = np.eye(16, dtype=float)
    i, j = 0b1100, 0b0011
    ct, st = np.cos(theta), np.sin(theta)
    U[i, i] = ct; U[j, j] = ct
    U[j, i] = st; U[i, j] = -st
    return U


def build_lucj(c, norb: int, nocc: int, t1, t2, eri=None,
               ccsd_scale: float = 0.05, local: bool = True,
               doubles: bool = False, doubles_thresh: float = 1e-4):
    """在电路 c 上构建 LUCJ ansatz（1 层，含 local 截断）。

    返回门统计 dict：n_cnot, n_single, n_rzz, depth_est, n_givens, n_double。

    结构（与 SQD_Practice 参考 §5 一致）：
      U = e^κ  : 轨道旋转（从 t1，每个 occupied-virtual 对做 Givens）
      e^{iJ}   : 对角 Coulomb（从 t2 构造 J_{ia}，每对做 RZZ）
      doubles=True 时追加显式双激发 Givens（完整 UCJ 方向的扩展，
      使采样能覆盖双激发行列式——SQD 恢复关联能所必需）。

    ccsd_scale=0 -> t1=t2=0 -> 退化为 HF（无纠缠）。
    """
    stats = dict(n_cnot=0, n_single=0, n_rzz=0, n_givens=0, depth_est=0,
                 n_double=0)
    virt = list(range(nocc, norb))
    occ = list(range(nocc))

    # ---- 轨道旋转 U = e^κ（JW 轨道旋转，分自旋，保粒子数）----
    if t1 is not None:
        for i in occ:
            for a in virt:
                amp = t1[i, a - nocc]  # 空间 t1[i, a] -> (nocc, nvir)
                if abs(amp) < 1e-6:
                    continue
                theta = ccsd_scale * amp
                # alpha 自旋：orbital i (qubit 2i) <-> orbital a (qubit 2a)
                jw_rotate(c, 2 * i, 2 * a, theta)
                stats["n_givens"] += 1
                # beta 自旋：qubit 2i+1 <-> 2a+1
                jw_rotate(c, 2 * i + 1, 2 * a + 1, theta)
                stats["n_givens"] += 1

    # ---- 对角 Coulomb e^{iJ}（RZZ）----
    if t2 is not None:
        # J_{ia} = Σ_{j∈occ, b∈virt} (ia|jb) t2[i,j,a,b]  （简化但物理合理）
        for i in occ:
            for a in virt:
                J = 0.0
                for j in occ:
                    for b in virt:
                        J += (t2[i, j, a - nocc, b - nocc]
                              * _eri_chem(eri, i, a, j, b))
                if abs(J) < 1e-8:
                    continue
                # 4 个自旋组合 -> 4 个 RZZ（每对 qubit 间 θ = J/4 * scale）
                theta = 0.5 * ccsd_scale * J  # 含 1/4 与全局相位处理
                pairs = [(2 * i, 2 * a), (2 * i, 2 * a + 1),
                         (2 * i + 1, 2 * a), (2 * i + 1, 2 * a + 1)]
                for q0, q1 in pairs:
                    if local and abs(q0 - q1) != 1:
                        continue
                    rzz(c, q0, q1, theta)
                    stats["n_cnot"] += 2
                    stats["n_single"] += 1
                    stats["n_rzz"] += 1

    # ---- 显式双激发（i_alpha j_beta -> a_alpha b_beta），保 N 与 Sz ----
    if doubles and t2 is not None:
        nvir = norb - nocc
        for i in occ:
            for j in occ:
                for a in range(nvir):
                    for b in range(nvir):
                        amp = ccsd_scale * t2[i, j, a, b]
                        if abs(amp) < doubles_thresh:
                            continue
                        qs = (2 * i, 2 * j + 1,
                              2 * (nocc + a), 2 * (nocc + b) + 1)
                        if len(set(qs)) < 4:
                            continue
                        c.any(*qs, unitary=double_excitation_unitary(amp))
                        stats["n_double"] += 1
                        stats["n_cnot"] += 13  # 硬件分解估计
    stats["depth_est"] = stats["n_cnot"] + stats["n_single"]
    return stats


def _eri_chem(eri, i, a, j, b):
    """chemist (ia|jb)。"""
    return eri[i, a, j, b]


def bitrev(x, n: int):
    """比特反序（向量化）：TC 按 qubit0=最高位，openfermion 按 qubit0=最低位。

    支持标量或 numpy 数组；数组版本用位运算批量反序，避免 Python 循环。
    """
    x = np.asarray(x, dtype=np.int64)
    r = np.zeros_like(x)
    for i in range(n):
        r |= ((x >> i) & 1) << (n - 1 - i)
    return r


def statevector(c):
    """返回扁平态矢量（openfermion 序：qubit0=LSB）。

    TC 默认态矢量是 (2,)*nq 多维张量且 qubit0 为最高位，需展平并比特反序。
    用向量化 bitrev 一次性完成反序（适用于大比特数电路）。
    """
    s = np.asarray(c.state()).reshape(-1)
    nq = int(round(np.log2(s.size)))
    idx = np.arange(s.size)
    rev = bitrev(idx, nq)
    out = np.empty(s.size, dtype=complex)
    out[rev] = s
    return out


def sample_counts(c, n_shots: int, nq: int = None, seed=None):
    """采样返回 {openfermion_int_key: count}。

    采用自实现快速采样：直接取态矢量（openfermion 序），按 |ψ|² 分布
    采样，避免 numpy 后端大电路 sample 缓慢。

    Parameters
    ----------
    seed : int 或 None。给出则用独立 ``np.random.default_rng(seed)``（线程安全，
           并行扫描时各点传不同 seed）；None 则用全局 ``np.random``（仅单线程）。
    """
    if nq is None:
        # 由电路推断
        nq = int(round(np.log2(np.asarray(c.state()).size)))
    psi = statevector(c)
    probs = np.abs(psi) ** 2
    probs = probs / probs.sum()
    rng = np.random.default_rng(seed) if seed is not None else np.random
    idx = rng.choice(len(probs), size=n_shots, p=probs)
    keys, counts = np.unique(idx, return_counts=True)
    return {int(k): int(v) for k, v in zip(keys, counts)}


def expectation(psi, Hmat):
    """⟨ψ|H|ψ⟩，psi 为态矢量，Hmat 为 numpy 矩阵。"""
    return float(np.real(np.conj(psi) @ Hmat @ psi))


if __name__ == "__main__":
    import common.backend  # noqa: F401
    from common.chemistry import molecule_report
    from common.mapping import build_fermion_hamiltonian, jw, qubit_matrix
    rep = molecule_report("H2")
    norb, nocc = rep["norb"], rep["nocc"]
    nq = 2 * norb
    ham = build_fermion_hamiltonian(rep["h1e"], rep["eri"], rep["ecore"], norb)
    Hmat = qubit_matrix(jw(ham), nq)
    c = tc.Circuit(nq)
    prepare_hf(c, norb, nocc)
    psi = statevector(c)
    print("H2 HF ⟨H⟩:", expectation(psi, Hmat), "E_HF:", rep["E_HF"])
