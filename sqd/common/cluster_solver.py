"""团簇量子求解器（EWF-SQD 管线共用，Q10/Q13/Q14）。

- solve_cluster_sqd: 小团簇全 FCI 子空间（精确 SQD）；大团簇
  LUCJ+doubles(MP2 t2) 采样 + 配置恢复 + SQD 自洽扩展一轮。
- expand_configs   : S-CORE 式子空间扩展（重要构型的保 Sz 单+双激发）。
"""
from math import comb
import itertools

import numpy as np
import common.backend  # noqa: F401  量子后端自动初始化
import tensorcircuit as tc

from common.dmet import cluster_scf, cluster_mp2_t2
from common.circuits import prepare_hf, build_lucj, sample_counts
from common.sqd import (config_recovery_counts, bitstrings_to_ci_strs,
                        run_sqd, run_sqd_product)
from common.ci import full_fci_basis


def mo_integrals(frag, scf_res):
    """团簇积分旋转到正则 MO 基（HF 行列式 = 单构型，供 SQD/LUCJ 用）。"""
    C = scf_res["mo_coeff"]
    h_mo = C.T @ frag["heff"] @ C
    eri_mo = np.einsum("pqrs,pi,qj,rk,sl->ijkl", frag["eris"], C, C, C, C,
                       optimize=True)
    return h_mo, eri_mo


def solve_cluster_sqd(frag, n_shots=20000, lam=5.0, seed=0):
    """量子路径求解团簇：返回 (E_SQD_elec, E_HF_elec, M, 电路统计)。

    lam>1 为「采样偏置放大」：SQD 能量来自子空间对角化，不依赖电路振幅
    精度，放大 t2 仅为提高双激发构型的采样覆盖率（SQD 的核心鲁棒性）。
    """
    norb, nocc = frag["norb"], frag["nocc"]
    nq = 2 * norb
    scf_res = cluster_scf(frag if "h1e" in frag else
                          dict(h1e=frag["heff"], eri=frag["eris"],
                               norb=norb, nocc=nocc))
    h_mo, eri_mo = mo_integrals(frag, scf_res)
    e_hf = scf_res["E_elec"]

    if comb(norb, nocc) ** 2 <= 1000:  # 小团簇：全 FCI 子空间 = 精确 SQD
        basis = full_fci_basis(norb, nocc, nocc)
        res = run_sqd(h_mo, eri_mo, nq, 0.0, basis)
        return float(res["E_sqd"]), float(e_hf), len(basis), dict(mode="full")

    # 大团簇：LUCJ+doubles(MP2 t2) 采样
    t2 = cluster_mp2_t2(dict(norb=norb, nocc=nocc, eri=frag["eris"]),
                        scf_res)
    t1 = np.zeros((nocc, norb - nocc))
    c = tc.Circuit(nq)
    prepare_hf(c, norb, nocc)
    stats = build_lucj(c, norb, nocc, t1, t2, eri=eri_mo, ccsd_scale=lam,
                       doubles=True, doubles_thresh=3e-4)
    # seed 传给 sample_counts 用独立 RNG（线程安全，碎片级并行可复现）
    counts = sample_counts(c, n_shots, nq, seed=seed)
    # 配置恢复 -> α/β 串 -> 笛卡尔积子空间（qiskit-addon-sqd 结构），HF 强制入列
    rc = config_recovery_counts(counts, nq, nocc, nocc)
    a_strs, b_strs = bitstrings_to_ci_strs(rc, nq)
    hf_bs = "".join("1" if q // 2 < nocc else "0" for q in range(nq))
    res = run_sqd_product(h_mo, eri_mo, nq, 0.0, a_strs, b_strs,
                          max_dets=8000, include=[hf_bs])
    # ---- SQD 自洽配置扩展 1 轮：对基态中权重大的构型做保 Sz 双激发 ----
    basis = expand_configs(res["basis"], res["evecs"][:, 0], norb, nocc,
                           weight_cut=0.02, max_M=1500)
    res = run_sqd(h_mo, eri_mo, nq, 0.0, basis)
    stats["mode"] = f"lucj_sampling(S={n_shots},λ={lam})+product+expand"
    return float(res["E_sqd"]), float(e_hf), len(basis), stats


def solve_clusters(frags, n_shots=20000, lam=5.0, parallel=True,
                   max_workers=None):
    """并行求解多个独立团簇（碎片级并行）。

    每个团簇的 LUCJ 采样 + SQD 对角化相互独立，用 parallel_map 线程并行
    （JAX/numpy 均释放 GIL；threadpoolctl 限制每线程 BLAS 线程避免超订）。
    返回 [(E_SQD_elec, E_HF_elec, M, stats), ...]，与 frags 同序。
    """
    fn = lambda f: solve_cluster_sqd(f, n_shots=n_shots, lam=lam)
    if not parallel:
        return [fn(f) for f in frags]
    from common.parallel import parallel_map
    return parallel_map(fn, frags, max_workers=max_workers)


def expand_configs(basis, vec, norb, nocc, weight_cut=0.02, max_M=1500):
    """对权重 > weight_cut 的构型生成全部保 Sz 单+双激发，扩展子空间。

    对应真实 SQD 的自洽配置恢复迭代（S-CORE）。
    """
    nq = 2 * norb
    seen = set(basis)
    order = np.argsort(-np.abs(vec))
    for k in order:
        if abs(vec[k]) < weight_cut or len(seen) >= max_M:
            break
        bs = basis[k]
        for spin in (0, 1):  # 同自旋内的单激发（两次叠加即双激发路径）
            occ_s = [q for q in range(spin, nq, 2) if bs[q] == "1"]
            vir_s = [q for q in range(spin, nq, 2) if bs[q] == "0"]
            for o, v in itertools.product(occ_s, vir_s):
                nb = list(bs); nb[o], nb[v] = "0", "1"
                nb = "".join(nb)
                if nb not in seen:
                    seen.add(nb)
        # 异自旋对双激发
        occ_a = [q for q in range(0, nq, 2) if bs[q] == "1"]
        vir_a = [q for q in range(0, nq, 2) if bs[q] == "0"]
        occ_b = [q for q in range(1, nq, 2) if bs[q] == "1"]
        vir_b = [q for q in range(1, nq, 2) if bs[q] == "0"]
        for oa, va, ob, vb in itertools.product(occ_a, vir_a, occ_b, vir_b):
            if len(seen) >= max_M:
                break
            nb = list(bs)
            nb[oa], nb[va], nb[ob], nb[vb] = "0", "1", "0", "1"
            nb = "".join(nb)
            if nb not in seen:
                seen.add(nb)
    return sorted(seen)
