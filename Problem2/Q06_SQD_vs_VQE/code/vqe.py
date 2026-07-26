"""
vqe.py — TensorCircuit 实现 UCCSD 截断的 VQE 浅线路制备。

后端：numpy（纯态矢量模拟，无需 GPU/tensorflow）。
优化器：scipy L-BFGS-B（对小体系稳定高效）。

物理思路（UCCSD 截断）：
  从 HF 参考态 |11000000> 出发，作用「保粒子数」的 Givens 旋转激发算符：
    - single excitation:  Givens (i->a)，把电子从占据轨道搬到虚轨道
    - double excitation:  成对激发 (ij->ab)，两个 Givens 串联近似
  这些门保持总粒子数不变，是 UCCSD 的核心；截断=只保留 single+double，故线路浅。

优化目标：E_VQE = <Psi(theta)|H|Psi(theta)>。
输出：E_VQE、优化后态的全振幅向量 psi（供后续采样/子空间对角化）、params。
"""
import numpy as np
import tensorcircuit as tc
from scipy.optimize import minimize

tc.set_backend("numpy")
tc.set_dtype("complex128")


def _hf_bitstring(n_qubits, n_electrons):
    """HF 参考：前 n_electrons 个自旋轨道占据（JW 约定）。"""
    return [1] * n_electrons + [0] * (n_qubits - n_electrons)


def _givens(circuit, theta, i, j):
    """保粒子数的 Givens 旋转（单激发 i<->j），作用在 qubit i,j。"""
    circuit.cnot(i, j)
    circuit.cry(j, i, theta=2 * theta)
    circuit.cnot(i, j)


def build_uccsd_ansatz(n_qubits, n_electrons, params):
    """构建 UCCSD 截断 ansatz 线路。"""
    c = tc.Circuit(n_qubits)
    for q, b in enumerate(_hf_bitstring(n_qubits, n_electrons)):
        if b == 1:
            c.x(q)

    occ = list(range(n_electrons))
    vir = list(range(n_electrons, n_qubits))

    idx = 0
    # single excitations: occ -> vir
    for i in occ:
        for a in vir:
            _givens(c, params[idx], i, a)
            idx += 1
    # double excitations: (i,j)->(a,b)
    for ii in range(len(occ)):
        for jj in range(ii + 1, len(occ)):
            for aa in range(len(vir)):
                for bb in range(aa + 1, len(vir)):
                    i, j = occ[ii], occ[jj]
                    a, b = vir[aa], vir[bb]
                    _givens(c, params[idx], i, a)
                    _givens(c, params[idx], j, b)
                    idx += 1
    return c


def count_params(n_qubits, n_electrons):
    occ = n_electrons
    vir = n_qubits - n_electrons
    n_single = occ * vir
    n_double = (occ * (occ - 1) // 2) * (vir * (vir - 1) // 2)
    return n_single + n_double


def get_state(n_qubits, n_electrons, params):
    """返回给定参数下的全振幅向量。"""
    c = build_uccsd_ansatz(n_qubits, n_electrons, params)
    return np.array(c.state()).flatten()


def run_vqe(H, n_qubits, n_electrons, maxiter=300, seed=42):
    """优化 VQE 参数，返回 (E_VQE, psi_amplitudes, params)。"""
    np.random.seed(seed)
    n_params = count_params(n_qubits, n_electrons)

    def energy(params):
        psi = get_state(n_qubits, n_electrons, params)
        return float(np.real(np.conj(psi) @ (H @ psi)))

    x0 = np.random.uniform(-0.1, 0.1, n_params)
    res = minimize(energy, x0, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-12})
    psi = get_state(n_qubits, n_electrons, res.x)
    return float(res.fun), psi, res.x


if __name__ == "__main__":
    from hamiltonian import build_lih_hamiltonian
    data = build_lih_hamiltonian()
    print(f"n_params = {count_params(data['n_qubits'], data['n_electrons'])}")
    E_vqe, psi, p = run_vqe(data["H"], data["n_qubits"], data["n_electrons"])
    print(f"E_VQE = {E_vqe:.6f} Ha")
    print(f"E0    = {data['E0']:.6f} Ha (下界)")
    print(f"gap E_VQE - E0 = {E_vqe - data['E0']:.6f} Ha")
