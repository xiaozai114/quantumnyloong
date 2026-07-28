"""费米子 -> 量子比特映射层（基于 openfermion）。

提供：
  - build_fermion_hamiltonian : 由 h1e/eri 构造 openfermion FermionOperator
  - jw / parity / bk          : 三种映射 -> QubitOperator
  - pauli_stats               : 统计 Pauli 项数与最大权重
  - reduce_2q                 : 闭壳层 Z2 粒子数约化（4q -> 2q 显式 2 比特算符）
  - qubit_matrix / diagonalize: 显式矩阵与对角化（自实现，避免依赖 openfermion 内部路径）
"""
from __future__ import annotations

import numpy as np
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import jordan_wigner, bravyi_kitaev, parity_code
from openfermion.chem import MolecularData
from openfermion.transforms import get_fermion_operator

# Pauli 矩阵
_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _ensure_python_numeric(op):
    """openfermion 系数转 Python float/complex（numpy 2.x 类型检查兼容）。

    openfermion 的 SymbolicOperator 在 numpy 2.x 下拒绝 numpy 标量系数
   （``Coefficient must be a numeric type. Got numpy.int64``），需转原生类型。
    """
    if hasattr(op, "terms"):
        op.terms = {term: (complex(c) if isinstance(c, complex) else float(c))
                    for term, c in op.terms.items()}
    return op


def build_fermion_hamiltonian(fermion_hamiltonian=None, h1e=None, eri=None,
                               ecore=None, norb=None, inter=None):
    """构造 openfermion FermionOperator。

    优先使用由 openfermionpyscf 产出的 InteractionOperator（inter）转换，
    保证与 PySCF 的化学积分约定一致；亦兼容手动传入 FermionOperator。
    """
    if inter is not None:
        return _ensure_python_numeric(get_fermion_operator(inter))
    if fermion_hamiltonian is not None:
        return fermion_hamiltonian
    # 手动构造（需 openfermion 约定下的空间积分，一般不直接使用）
    md = MolecularData(geometry=[("H", (0, 0, 0))], basis="sto-3g",
                        multiplicity=1, description="sqd_tmp")
    md.n_orbitals = norb
    md.one_body_integrals = np.asarray(h1e, dtype=float)
    md.two_body_integrals = 0.5 * np.asarray(eri, dtype=float)
    md.nuclear_repulsion = float(ecore)
    return _ensure_python_numeric(get_fermion_operator(md.get_molecular_hamiltonian()))


def _n_modes(fop):
    n = 0
    for term in fop.terms:
        for (idx, _p) in term:
            n = max(n, idx + 1)
    return n


def jw(ham):
    return jordan_wigner(ham)


def bk(ham):
    return bravyi_kitaev(ham, n_qubits=_n_modes(ham))


def parity(ham):
    from openfermion.transforms import binary_code_transform
    code = parity_code(_n_modes(ham))
    return binary_code_transform(ham, code)


def pauli_stats(qop: QubitOperator):
    """返回 (项数 n_terms, 最大 Pauli 权重)。"""
    n_terms = 0
    max_weight = 0
    for term in qop.terms:
        n_terms += 1
        w = sum(1 for (_idx, pauli) in term if pauli != "I")
        max_weight = max(max_weight, w)
    return n_terms, max_weight


# --------------------------------------------------------------------------- #
# 显式矩阵构造（自实现，避免依赖 openfermion 内部路径）
# --------------------------------------------------------------------------- #
def _term_matrix(term, nqubits):
    """把单个 Pauli string 作用为 2^nqubits 矩阵。"""
    if not term:
        return np.eye(2 ** nqubits, dtype=complex)
    mat = np.array([1], dtype=complex)
    for idx in range(nqubits - 1, -1, -1):
        p = "I"
        for (i, pauli) in term:
            if i == idx:
                p = pauli
                break
        mat = np.kron(mat, _PAULI[p])
    return mat


def qubit_matrix(qop: QubitOperator, nqubits=None):
    if nqubits is None:
        nqubits = 0
        for term in qop.terms:
            for (idx, _p) in term:
                nqubits = max(nqubits, idx + 1)
    dim = 2 ** nqubits
    mat = np.zeros((dim, dim), dtype=complex)
    for term, coeff in qop.terms.items():
        mat += coeff * _term_matrix(term, nqubits)
    return mat


def _term_matrix_sparse(term, nqubits):
    """稀疏版 _term_matrix（scipy.sparse.kron）。"""
    from scipy import sparse
    pauli = {k: sparse.csr_matrix(v) for k, v in _PAULI.items()}
    if not term:
        return sparse.eye(2 ** nqubits, dtype=complex, format="csr")
    mat = sparse.csr_matrix([[1]], dtype=complex)
    for idx in range(nqubits - 1, -1, -1):
        p = "I"
        for (i, pj) in term:
            if i == idx:
                p = pj
                break
        mat = sparse.kron(mat, pauli[p], format="csr")
    return mat


def qubit_matrix_sparse(qop: QubitOperator, nqubits=None):
    """稀疏 QubitOperator 矩阵（大体系用，内存 O(nnz) 而非 O(4^n)）。"""
    from scipy import sparse
    if nqubits is None:
        nqubits = 0
        for term in qop.terms:
            for (idx, _p) in term:
                nqubits = max(nqubits, idx + 1)
    dim = 2 ** nqubits
    mat = sparse.csr_matrix((dim, dim), dtype=complex)
    for term, coeff in qop.terms.items():
        mat = mat + coeff * _term_matrix_sparse(term, nqubits)
    return mat


def ground_energy(qop: QubitOperator, nqubits=None, tol=1e-12):
    """稀疏迭代求基态能量（只要最低本征值，避免稠密全谱）。

    大体系（nqubits ≳ 10）下比 diagonalize(...)[0] 快数个量级。
    """
    from scipy.sparse.linalg import eigsh
    mat = qubit_matrix_sparse(qop, nqubits)
    ev = eigsh(mat, k=1, which="SA", tol=tol, return_eigenvectors=False)
    return float(np.real(ev[0]))


def diagonalize(qop: QubitOperator, nqubits=None):
    mat = qubit_matrix(qop, nqubits)
    evals = np.linalg.eigvalsh(mat)
    return np.sort(np.real(evals))


def qubit_operator_from_matrix(mat, nqubits):
    """把一个 2^nqubits 厄米矩阵按 Pauli 基分解回 QubitOperator（数值精确）。"""
    from itertools import product
    dim = 2 ** nqubits
    basis_paulis = ["I", "X", "Y", "Z"]
    qop = QubitOperator()
    for combo in product(basis_paulis, repeat=nqubits):
        if all(p == "I" for p in combo):
            continue
        term = tuple((idx, p) for idx, p in enumerate(combo) if p != "I")
        pmat = _term_matrix(term, nqubits)
        coeff = np.real(np.trace(pmat.conj().T @ mat)) / dim
        if abs(coeff) > 1e-9:
            qop += coeff * QubitOperator(term)
    # 常数项
    c = np.real(np.trace(mat)) / dim
    if abs(c) > 1e-9:
        qop += c * QubitOperator(())
    return qop


def reduce_2q(ham, norb, n_alpha, n_beta):
    """闭壳层：把 JW Hamiltonian 投影到 (N_alpha, N_beta) 扇区 -> 2 比特显式算符。

    返回 (QubitOperator_2q, list_of_jw_bitstrings)。
    """
    nq = 2 * norb
    Hfull = qubit_matrix(jw(ham), nq)
    # 选出 N_alpha / N_beta 正确的 JW 基态（qubit 0,2,...=alpha; 1,3,...=beta）
    basis = []
    for key in range(2 ** nq):
        a = sum((key >> q) & 1 for q in range(0, nq, 2))
        b = sum((key >> q) & 1 for q in range(1, nq, 2))
        if a == n_alpha and b == n_beta:
            basis.append(key)
    idx = {k: i for i, k in enumerate(basis)}
    block = np.zeros((len(basis), len(basis)), dtype=complex)
    for i, ki in enumerate(basis):
        for j, kj in enumerate(basis):
            block[i, j] = Hfull[ki, kj]
    qop2 = qubit_operator_from_matrix(block, 2)
    # 把 2 比特算符的基态顺序对齐：2 比特的 |b1 b0> 对应 4 态排序
    return qop2, basis


if __name__ == "__main__":
    from common.chemistry import molecule_report
    rep = molecule_report("H2")
    ham = build_fermion_hamiltonian(rep["h1e"], rep["eri"], rep["ecore"], rep["norb"])
    qjw = jw(ham)
    n, w = pauli_stats(qjw)
    print(f"H2 JW: terms={n} max_weight={w}")
    ev = diagonalize(qjw, nqubits=4)
    print("JW ground:", ev[0], "FCI:", rep["E_FCI"])
    q2, basis = reduce_2q(ham, rep["norb"], 1, 1)
    nv, vw = pauli_stats(q2)
    ev2 = diagonalize(q2)
    print(f"reduced terms={nv} max_weight={vw} ground={ev2[0]}")
    print("reduced basis keys:", basis)
