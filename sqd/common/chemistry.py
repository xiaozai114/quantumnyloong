"""化学接口层：PySCF 取 RHF 积分、FCI/CCSD/MP2 参考解。

所有积分采用 chemist 记号：
    h_{pq}    : (norb, norb)        单体积分（分子轨道基）
    (pq|rs)   : (norb, norb, norb, norb)  双体积分，chemist 记号
    E_core    : 核排斥能（不含冻芯；小分子无冻芯）
总能量 = <D|H|D> + E_core。
"""
from __future__ import annotations

import numpy as np
import pyscf.gto
import pyscf.scf
import pyscf.fci
import pyscf.cc
import pyscf.mp
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator


# --------------------------------------------------------------------------- #
# 分子定义
# --------------------------------------------------------------------------- #
# 与 docs/sqd_practice/SQD_Practice_CN.md 「测试分子」表一致。
MOLECULES = {
    "H2": dict(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", charge=0, spin=0),
    "LiH": dict(atom="Li 0 0 0; H 0 0 1.55", basis="sto-3g", charge=0, spin=0),
    "H2O": dict(atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
                basis="sto-3g", charge=0, spin=0),
    "N2": dict(atom="N 0 0 0; N 0 0 1.10", basis="sto-3g", charge=0, spin=0),
    "C2H4": dict(atom="C 0 0 0; C 0 0 1.33; H 0 0.94 -0.51; H 0 -0.94 -0.51; "
                     "H 0 0.94 1.84; H 0 -0.94 1.84",
                 basis="sto-3g", charge=0, spin=0),
}


def build_mol(name: str, verbose: int = 0):
    """构建并返回 PySCF molecule 对象（未运行 SCF）。"""
    cfg = MOLECULES[name]
    mol = pyscf.gto.M(
        atom=cfg["atom"], basis=cfg["basis"],
        charge=cfg.get("charge", 0), spin=cfg.get("spin", 0),
        verbose=verbose,
    )
    mol.build()
    return mol


def rhf(mol):
    """运行 RHF，返回 mf 对象。"""
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    return mf


def integrals_from_mf(mf):
    """从 RHF 结果提取 chemist 记号积分。

    返回 dict: h1e (norb,norb), eri (norb,norb,norb,norb), ecore, norb, nocc。
    norb  = 空间轨道数 = 自旋轨道数 / 2
    nocc  = 占据(空间)轨道数 = 电子数 / 2（闭壳层）
    """
    mol = mf.mol
    mo = mf.mo_coeff
    norb = mo.shape[1]
    # 单体积分（含动能 + 核吸引）
    h1e = mo.T @ mf.get_hcore() @ mo
    # 双体积分 chemist 记号 (pq|rs)
    eri = pyscf.ao2mo.restore(1, pyscf.ao2mo.kernel(mol, mo), norb)
    # pyscf 的 restore(1) 给出完整 (norb,norb,norb,norb)，chemist 记号
    ecore = mol.energy_nuc()
    nocc = int(mol.nelectron // 2)
    return dict(h1e=h1e, eri=eri, ecore=ecore, norb=norb, nocc=nocc,
                nelec=mol.nelectron)


def fci_energy(mf):
    """返回 PySCF FCI 基态能量（Ha）。"""
    cisolver = pyscf.fci.FCI(mf)
    e, _ = cisolver.kernel()
    return float(e)


def ccsd_energy(mf):
    """返回 CCSD 能量（Ha）与振幅 t1, t2。"""
    cc = pyscf.cc.CCSD(mf).run()
    return float(cc.e_tot), cc.t1, cc.t2


def mp2_energy(mf):
    """返回 MP2 能量（Ha）与密度矩阵修正（用于 EWF MP2 bath，可选）。"""
    mp = pyscf.mp.MP2(mf).run()
    return float(mp.e_tot)


def hf_energy(mf):
    return float(mf.e_tot)


def of_molecular_data(mol):
    """用 openfermionpyscf 构造 openfermion MolecularData（约定正确）。

    返回的 md 提供 one_body_integrals / two_body_integrals（openfermion 约定）
    与 nuclear_repulsion，可直接 get_molecular_hamiltonian() -> InteractionOperator。
    """
    geometry = [(a, tuple(map(float, xyz.split())))
                for a, xyz in (line.split(maxsplit=1)
                               for line in mol.atom.split(";"))]
    md = MolecularData(geometry=[(a, tuple(xyz)) for a, xyz in geometry],
                       basis=mol.basis, multiplicity=mol.spin + 1,
                       charge=mol.charge)
    md = run_pyscf(md)
    return md


def of_interaction_operator(mol):
    """返回正确约定下的 openfermion InteractionOperator（含核排斥常数项）。"""
    md = of_molecular_data(mol)
    return md, md.get_molecular_hamiltonian()


def molecule_report(name: str, do_fci: bool = True, do_of: bool = True):
    """一键给出某分子的全部参考量，便于各题调用。

    do_fci=False 跳过 FCI（大分子如 C2H4 的 FCI 空间 ~3e7，代价过高）；
    do_of=False 跳过 openfermion 算符构建（仅需积分/能量时提速）。
    """
    mol = build_mol(name)
    mf = rhf(mol)
    integ = integrals_from_mf(mf)
    integ["E_HF"] = hf_energy(mf)
    integ["E_FCI"] = fci_energy(mf) if do_fci else None
    try:
        integ["E_CCSD"], integ["t1"], integ["t2"] = ccsd_energy(mf)
    except Exception as exc:  # noqa: BLE001
        integ["E_CCSD"], integ["t1"], integ["t2"] = None, None, None
        integ["ccsd_error"] = str(exc)
    # openfermion 交互算符（映射层用，约定正确）
    if do_of:
        md_of, inter = of_interaction_operator(mol)
        integ["md_of"] = md_of
        integ["inter"] = inter
        integ["fermion_hamiltonian"] = get_fermion_operator(inter)
    integ["mol"] = mol
    integ["mf"] = mf
    return integ


if __name__ == "__main__":
    for nm in ["H2", "LiH"]:
        rep = molecule_report(nm)
        print(f"{nm}: norb={rep['norb']} nq={2*rep['norb']} "
              f"E_HF={rep['E_HF']:.6f} E_FCI={rep['E_FCI']:.6f}")
