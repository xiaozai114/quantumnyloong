"""DMET/EWF 碎片化层。

依赖 numpy/scipy/pyscf。

流程：
  1. Löwdin 正交化 AO；HF 投影矩阵 P = S^{1/2} C_occ C_occ^T S^{1/2}。
  2. 杂质 = 指定原子的 AO；bath = SVD(P[env, imp]) 奇异值 > tol 的环境轨道
     （数目 <= n_imp，Schmidt 秩上限）。
  3. 团簇 = 杂质 + bath；core 环境轨道的 HF 平均场（J-K/2）并入 h_emb。
  4. 能量重构：democratic partitioning（团簇 RDM 杂质首指标投影）。
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, fractional_matrix_power

import pyscf.ao2mo
from pyscf.fci import cistring


# ====================================================================== #
# 1. 团簇构建（Löwdin + SVD bath）
# ====================================================================== #


def build_clusters(mf, frag_atom_lists, tol=1e-8):
    """DMET 原子碎片化。

    Args:
        mf   : 收敛的 PySCF RHF 对象
        frag_atom_lists : [[原子编号,...], ...]，每个子列表定义一个碎片
        tol  : bath 奇异值截断（DMET 最小 bath）
    Returns:
        clusters: list[dict]，每个含
            h1e (含 core 嵌入势), eri, norb, nocc, n_imp,
            schmidt (bath 奇异值), C_cl (AO 系数)
    """
    mol = mf.mol
    S = mol.intor_symmetric("int1e_ovlp")
    Sh = fractional_matrix_power(S, 0.5).real
    Sih = np.linalg.inv(Sh)
    C_occ = mf.mo_coeff[:, mf.mo_occ > 0]
    P = Sh @ C_occ @ C_occ.T @ Sh          # 每自旋投影矩阵，P^2 = P
    nao = S.shape[0]
    slices = mol.aoslice_by_atom()

    clusters = []
    for atoms in frag_atom_lists:
        imp = []
        for a in atoms:
            imp.extend(range(slices[a][2], slices[a][3]))
        env = [i for i in range(nao) if i not in imp]
        n_imp = len(imp)

        # ---- Schmidt/bath：SVD(P[env, imp]) ----
        if env:
            U, sv, _ = np.linalg.svd(P[np.ix_(env, imp)], full_matrices=False)
            nbath = int(np.sum(sv > tol))
        else:
            U, sv, nbath = np.zeros((0, 0)), np.array([]), 0

        ncl = n_imp + nbath
        T = np.zeros((nao, ncl))            # 团簇轨道（Löwdin 基）
        for k, i in enumerate(imp):
            T[i, k] = 1.0
        for k in range(nbath):
            T[env, n_imp + k] = U[:, k]

        # ---- core：环境中与 bath 正交的占据轨道 ----
        if env:
            Ub = U[:, :nbath]
            Q = np.eye(len(env)) - Ub @ Ub.T
            w, V = np.linalg.eigh(Q @ P[np.ix_(env, env)] @ Q)
            core_cols = V[:, w > 0.5]
        else:
            core_cols = np.zeros((0, 0))
        C_core_L = np.zeros((nao, core_cols.shape[1]))
        if env:
            C_core_L[env, :] = core_cols

        # ---- 团簇电子数（HF 水平精确为整数，Q8(2)）----
        n_per_spin = float(np.trace(T.T @ P @ T))
        nocc = int(round(n_per_spin))
        assert abs(n_per_spin - nocc) < 1e-6, \
            f"团簇电子数非整数: {n_per_spin}（DMET HF 精确性被破坏）"

        # ---- 团簇积分（AO 基回转 + core 嵌入势）----
        C_cl = Sih @ T
        C_core = Sih @ C_core_L
        D_core = 2.0 * C_core @ C_core.T
        if D_core.size:
            vj, vk = mf.get_jk(dm=D_core)
            v_emb = vj - 0.5 * vk
        else:
            v_emb = 0.0
        h1e = C_cl.T @ (mf.get_hcore() + v_emb) @ C_cl
        eri = pyscf.ao2mo.restore(1, pyscf.ao2mo.kernel(mol, C_cl), ncl)

        clusters.append(dict(
            h1e=h1e, eri=eri, norb=ncl, nocc=nocc, n_imp=n_imp,
            nbath=nbath, schmidt=sv, C_cl=C_cl, atoms=list(atoms),
        ))
    return clusters


# ====================================================================== #
# 2. 团簇 HF（正则轨道 + 电子能量）
# ====================================================================== #


def cluster_scf(cl, maxiter=100, conv=1e-10):
    """团簇内 RHF（固定点 + 阻尼）。返回 E_elec、正则轨道、密度矩阵。"""
    h, eri, nocc = cl["h1e"], cl["eri"], cl["nocc"]
    _, C = eigh(h)
    D = 2.0 * C[:, :nocc] @ C[:, :nocc].T
    E_old = 0.0
    for it in range(maxiter):
        J = np.einsum("pqrs,rs->pq", eri, D)
        K = np.einsum("prqs,rs->pq", eri, D)
        F = h + J - 0.5 * K
        E = 0.5 * np.einsum("pq,qp->", h + F, D)
        e_mo, C = eigh(F)
        D_new = 2.0 * C[:, :nocc] @ C[:, :nocc].T
        D = 0.7 * D_new + 0.3 * D            # 阻尼防振荡
        if abs(E - E_old) < conv:
            break
        E_old = E
    D = 2.0 * C[:, :nocc] @ C[:, :nocc].T     # 收敛后取纯净密度
    J = np.einsum("pqrs,rs->pq", eri, D)
    K = np.einsum("prqs,rs->pq", eri, D)
    F = h + J - 0.5 * K
    E = 0.5 * np.einsum("pq,qp->", h + F, D)
    return dict(E_elec=float(E), mo_energy=eigh(F)[0], mo_coeff=C, dm1=D)


def cluster_mp2_t2(cl, scf_res):
    """团簇 MP2 t2 振幅（正则 MO 基），供 LUCJ 采样电路使用。"""
    norb, nocc = cl["norb"], cl["nocc"]
    nvir = norb - nocc
    C, e = scf_res["mo_coeff"], scf_res["mo_energy"]
    eri_mo = np.einsum("pqrs,pi,qj,rk,sl->ijkl", cl["eri"], C, C, C, C,
                       optimize=True)
    t2 = np.zeros((nocc, nocc, nvir, nvir))
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    denom = e[i] + e[j] - e[nocc + a] - e[nocc + b]
                    t2[i, j, a, b] = eri_mo[i, nocc + a, j, nocc + b] / denom
    return t2


# ====================================================================== #
# 3. SQD CI 向量 -> pyscf fcivec -> RDM -> 民主分配能量
# ====================================================================== #


def sqd_vec_to_fcivec(basis, vec, norb, na, nb):
    """把 SQD 子空间 CI 向量嵌入 pyscf FCI 向量（未采样构型系数为 0）。"""
    ndet_a = cistring.num_strings(norb, na)
    ndet_b = cistring.num_strings(norb, nb)
    fcivec = np.zeros((ndet_a, ndet_b))
    for coeff, bs in zip(vec, basis):
        occ_a = [q // 2 for q in range(len(bs)) if bs[q] == "1" and q % 2 == 0]
        occ_b = [q // 2 for q in range(len(bs)) if bs[q] == "1" and q % 2 == 1]
        addr_a = cistring.str2addr(norb, na, sum(1 << o for o in occ_a))
        addr_b = cistring.str2addr(norb, nb, sum(1 << o for o in occ_b))
        fcivec[addr_a, addr_b] = coeff
    return fcivec


def rdms_from_sqd(basis, vec, norb, na, nb):
    """由 SQD 解得到自旋求和 1-/2-RDM（chemist 约定，与 eri 同序）。"""
    from pyscf.fci import direct_spin1
    fcivec = sqd_vec_to_fcivec(basis, vec, norb, na, nb)
    dm1, dm2 = direct_spin1.make_rdm12(fcivec, norb, (na, nb))
    return dm1, dm2


def hf_rdms(cl, scf_res):
    """团簇 HF 的 1-/2-RDM（闭壳层解析式）。"""
    D = scf_res["dm1"]
    dm2 = np.einsum("pq,rs->pqrs", D, D) - 0.5 * np.einsum(
        "ps,rq->pqrs", D, D)
    return D, dm2


def frag_projected_energy(cl, dm1, dm2):
    """民主分配的碎片电子能量：首指标限制在杂质轨道（前 n_imp 列）。

    E_f = sum_{p in imp, q} h_pq dm1_pq
        + 1/2 sum_{p in imp, qrs} (pq|rs) dm2[p,q,r,s]
    （dm1/dm2 与 h1e/eri 同在团簇局域基；对称化由 RDM 的厄米性保证。）
    """
    ni = cl["n_imp"]
    h, eri = cl["h1e"], cl["eri"]
    e1 = np.einsum("pq,pq->", h[:ni, :], dm1[:ni, :])
    e2 = 0.5 * np.einsum("pqrs,pqrs->", eri[:ni], dm2[:ni], optimize=True)
    return float(e1 + e2)


def rotate_rdms(dm1_mo, dm2_mo, C):
    """RDM 从正则 MO 基旋转回团簇局域基（C: 局域 -> MO 系数矩阵）。"""
    dm1 = C @ dm1_mo @ C.T
    dm2 = np.einsum("ijkl,pi,qj,rk,sl->pqrs", dm2_mo, C, C, C, C,
                    optimize=True)
    return dm1, dm2
