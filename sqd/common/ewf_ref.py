"""EWF 参考层：碎片提取与能量重构。

依赖 numpy/scipy/pyscf 与 common.dmet。

提供两种能量评估方式（对应 Vayesta 的 e_tot / get_dm_energy）：
  - DM 民主式（``e_tot_ewf``，默认）：团簇 RDM → AO 基 + 全局 H（无 v_emb）
    + 杂质首指标归属。变分安全。
  - WF 投影式（``e_tot_ewf_wf``）：团簇 H（含 v_emb）+ 杂质首指标投影。
    与 DM 式都按杂质分片避免双计数，区别在 H 是否含 v_emb。

API
---
- ewf_reference(name, bath_type, threshold, solver, frag_atoms)
      -> dict(e_mf, e_tot_ewf, e_tot_ewf_wf, fragments[...], ...)
- cluster_fci(heff, eris, norb, nelec) -> float
      团簇 FCI 电子能量（pyscf direct_spin1）。
"""
from __future__ import annotations

import numpy as np

from common.chemistry import build_mol, rhf
from common.dmet import build_clusters, cluster_scf
import pyscf
from pyscf.fci import direct_spin1


def cluster_fci(heff: np.ndarray, eris: np.ndarray, norb: int, nelec):
    """团簇 FCI 电子能量（不含核排斥常数），经典精确参考（供 SQD 对比）。"""
    e, _ = direct_spin1.kernel(np.asarray(heff), np.asarray(eris), int(norb), tuple(nelec))
    return float(e)


def _impurity_ao(mol, atoms):
    """碎片原子组的杂质 AO 指标列表。"""
    slices = mol.aoslice_by_atom()
    imp = []
    for a in atoms:
        imp.extend(range(slices[a][2], slices[a][3]))
    return imp


def _rdm_to_ao(dm1, dm2, C):
    """把团簇局域基的 1-/2-RDM 经系数矩阵 C（AO->团簇）变换到 AO 基。"""
    d1 = C @ dm1 @ C.T
    t = np.einsum("ijkl,pi->pjkl", dm2, C)
    t = np.einsum("pjkl,qj->pqkl", t, C)
    t = np.einsum("pqkl,rk->pqrl", t, C)
    d2 = np.einsum("pqrl,sl->pqrs", t, C)
    return d1, d2


def _cluster_rdms_cl(cl):
    """团簇 FCI 与 HF 的 1-/2-RDM（团簇局域基，未变换到 AO）。"""
    norb, nocc = cl["norb"], cl["nocc"]
    nelec = (nocc, nocc)
    _, fc = direct_spin1.kernel(cl["h1e"], cl["eri"], norb, nelec)
    dm1f = direct_spin1.make_rdm1(fc, norb, nelec)
    dm2f = direct_spin1.make_rdm12(fc, norb, nelec)[1]
    sc = cluster_scf(cl)
    dm1h = sc["dm1"]
    dm2h = (np.einsum("pq,rs->pqrs", dm1h, dm1h)
            - 0.5 * np.einsum("ps,rq->pqrs", dm1h, dm1h))
    return dm1f, dm2f, dm1h, dm2h


def _cluster_rdms_ao(cl):
    """团簇 FCI 与 HF 的 1-/2-RDM，经 C_cl 变换到 AO 基。"""
    C = cl["C_cl"]
    d1f_cl, d2f_cl, d1h_cl, d2h_cl = _cluster_rdms_cl(cl)
    d1f, d2f = _rdm_to_ao(d1f_cl, d2f_cl, C)
    d1h, d2h = _rdm_to_ao(d1h_cl, d2h_cl, C)
    return d1f, d2f, d1h, d2h


def _imp_energy(d1, d2, h, eri, imp):
    """DM 式杂质归属能量：能量首指标限制在杂质 AO（全局 Hamiltonian，无 v_emb）。"""
    nao = h.shape[0]
    S = np.zeros(nao, dtype=bool)
    S[imp] = True
    e1 = np.einsum("pq,pq->", h[S, :], d1[S, :])
    e2 = 0.5 * np.einsum("pqrs,pqrs->", eri[S, :, :, :], d2[S, :, :, :])
    return e1 + e2


def _imp_energy_cluster(cl, dm1, dm2):
    """WF 投影式杂质能量：团簇 H（含 v_emb）+ 杂质首指标（团簇基前 n_imp）。

    与 DM 式的区别：WF 式用团簇哈密顿（含嵌入势 v_emb）在团簇局域基计算，
    DM 式用全局 H（不含 v_emb）在 AO 基计算。两者都按杂质首指标分片，
    避免了整团簇本征值差值的双计数；差异来自 v_emb 进入能量。
    """
    n_imp = cl["n_imp"]
    h1e, eri = cl["h1e"], cl["eri"]
    e1 = np.einsum("pq,pq->", h1e[:n_imp, :], dm1[:n_imp, :])
    e2 = 0.5 * np.einsum("pqrs,pqrs->", eri[:n_imp, :, :, :], dm2[:n_imp, :, :, :])
    return e1 + e2


def ewf_reference(name: str, bath_type: str = "dmet", threshold: float = 1e-6,
                  solver: str = "FCI", frag_atoms=None) -> dict:
    """自包含 EWF 碎片提取 + RDM 杂质归属能量重构。

    Parameters
    ----------
    name        : 分子名（见 common.chemistry.molecule_report 支持的集合）
    bath_type   : 仅作接口兼容保留（当前统一使用 SVD Schmidt bath）。
    threshold   : bath 奇异值截断阈值 η
    solver      : 团簇相关能求解器；当前统一以精确 FCI 给出参考（自包含）。
    frag_atoms  : 自定义碎片原子分组；None 表示按原子碎片化。
    """
    mol = build_mol(name)
    mf = rhf(mol)
    e_mf = float(mf.e_tot)

    if frag_atoms is None:
        frag_atoms = [[a] for a in range(mol.natm)]

    clusters = build_clusters(mf, frag_atoms, tol=threshold)

    # 全局 Hamiltonian（RDM 归属重构用；v_emb 仅用于团簇求解，不混入能量）
    h_glob = mf.get_hcore()
    eri_glob = pyscf.ao2mo.restore(1, mol.intor("int2e_sph"), mol.nao_nr())

    fragments = []
    e_corr_ewf = 0.0      # DM 民主式
    e_corr_ewf_wf = 0.0   # WF 投影式（团簇 H + 杂质首指标）
    for i, cl in enumerate(clusters):
        norb, nocc = cl["norb"], cl["nocc"]
        nelec = (nocc, nocc)
        imp = _impurity_ao(mol, cl["atoms"])
        C = cl["C_cl"]
        d1f_cl, d2f_cl, d1h_cl, d2h_cl = _cluster_rdms_cl(cl)
        # DM 式：AO 基 + 全局 H（无 v_emb）+ AO 杂质首指标（变分安全）
        d1f, d2f = _rdm_to_ao(d1f_cl, d2f_cl, C)
        d1h, d2h = _rdm_to_ao(d1h_cl, d2h_cl, C)
        e_fci_imp = _imp_energy(d1f, d2f, h_glob, eri_glob, imp)
        e_hf_imp = _imp_energy(d1h, d2h, h_glob, eri_glob, imp)
        e_corr_proj = float(e_fci_imp - e_hf_imp)
        e_corr_ewf += e_corr_proj
        # WF 式：团簇基 + 团簇 H（含 v_emb）+ 团簇杂质首指标（投影式）
        e_fci_wf = _imp_energy_cluster(cl, d1f_cl, d2f_cl)
        e_hf_wf = _imp_energy_cluster(cl, d1h_cl, d2h_cl)
        e_corr_wf = float(e_fci_wf - e_hf_wf)
        e_corr_ewf_wf += e_corr_wf
        fragments.append(dict(
            heff=cl["h1e"], eris=cl["eri"], norb=norb, nocc=nocc,
            nelec=nelec, e_corr_proj=e_corr_proj, e_corr_wf=e_corr_wf,
            label=i,
        ))

    # 两式总能量
    e_tot_ewf = e_mf + e_corr_ewf          # DM 民主式（全局 H，无 v_emb）
    e_tot_ewf_wf = e_mf + e_corr_ewf_wf    # WF 投影式（团簇 H，含 v_emb）

    return dict(
        mf=mf, e_mf=e_mf, e_tot_ewf=float(e_tot_ewf),
        e_tot_ewf_wf=float(e_tot_ewf_wf),
        e_corr_ewf=float(e_corr_ewf),
        e_corr_ewf_wf=float(e_corr_ewf_wf),
        fragments=fragments, emb=None,
    )
