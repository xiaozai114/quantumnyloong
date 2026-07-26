"""
hamiltonian.py — 构建 LiH 哈密顿量（active space 约化到题目 N=8）。

题目设定：LiH，N=8 自旋轨道（= 4 空间轨道）。
STO-3G 下 LiH 原生是 6 空间轨道 (12 qubit)，这里用 active space：
  - 冻结 Li 1s 芯轨道（occupied_indices=[0]）
  - 取 4 个 active 空间轨道 (active_indices=[1,2,3,4]) = 8 qubit
返回：
  - qubit 哈密顿量（JW 映射）的稀疏矩阵
  - n_qubits, n_electrons(active)
  - E_FCI（active space 内精确基态，作为三明治不等式下界 E_0）
  - E_HF
"""
import numpy as np
from openfermion import MolecularData, jordan_wigner, get_sparse_operator
from openfermionpyscf import run_pyscf


def build_lih_hamiltonian(bond_length: float = 1.595):
    """构建 active-space 约化后的 LiH 哈密顿量。"""
    geometry = [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, bond_length))]
    mol = MolecularData(geometry, "sto-3g", multiplicity=1, charge=0)
    mol = run_pyscf(mol, run_scf=True, run_fci=True)

    # active space：冻 Li 1s，取 4 个活性空间轨道 -> 8 qubit
    occupied_indices = [0]           # 冻结的芯轨道
    active_indices = [1, 2, 3, 4]    # 4 空间轨道 = 8 自旋轨道 = 8 qubit

    core_ham = mol.get_molecular_hamiltonian(
        occupied_indices=occupied_indices,
        active_indices=active_indices,
    )
    qop = jordan_wigner(core_ham)
    n_qubits = 2 * len(active_indices)                 # = 8
    # active 电子数 = 总电子数 - 2 * 冻结轨道数
    n_electrons = mol.n_electrons - 2 * len(occupied_indices)  # 4 - 2 = 2

    H = get_sparse_operator(qop, n_qubits=n_qubits).toarray()
    H = np.real_if_close(H)

    # active-space 内精确基态能量（下界 E_0），限制在正确电子数扇区
    E_fci_active = _fci_in_sector(H, n_qubits, n_electrons)

    return {
        "H": H,
        "n_qubits": n_qubits,
        "n_electrons": n_electrons,
        "E0": E_fci_active,           # active space FCI = 三明治不等式的下界
        "E_HF": mol.hf_energy,
        "E_FCI_full": mol.fci_energy, # 全空间 FCI（仅供参考）
        "qop": qop,
    }


def bitstring_particle_number(idx: int, n_qubits: int) -> int:
    """计算基态 |idx> 的粒子数（1 的个数）。JW 下比特=占据数。"""
    return bin(idx).count("1")


def _fci_in_sector(H: np.ndarray, n_qubits: int, n_electrons: int) -> float:
    """在固定粒子数扇区内求 H 的最小本征值。"""
    dim = H.shape[0]
    sector = [i for i in range(dim)
              if bitstring_particle_number(i, n_qubits) == n_electrons]
    Hs = H[np.ix_(sector, sector)]
    w = np.linalg.eigvalsh(Hs)
    return float(w[0])


if __name__ == "__main__":
    data = build_lih_hamiltonian()
    print(f"n_qubits    = {data['n_qubits']}")
    print(f"n_electrons = {data['n_electrons']} (active)")
    print(f"E_HF        = {data['E_HF']:.6f} Ha")
    print(f"E0 (active FCI, 下界) = {data['E0']:.6f} Ha")
    print(f"E_FCI_full  = {data['E_FCI_full']:.6f} Ha")
