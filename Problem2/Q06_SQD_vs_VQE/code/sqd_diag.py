"""
sqd_diag.py — SQD 后处理：采样 + 子空间投影对角化。

对应理论（进阶挑战三步的第 2、3 步）：
  第2步 采样：对 VQE 态 |Psi> 在计算基下采样 S 次，收集出现过的组态，
            张成子空间 S = span{|x> : 被采到}。
  第3步 对角化：在 S 内把 H 投影成小矩阵 H_S = P_S H P_S，经典对角化取最小本征值：
            E_hybrid = lambda_min(H_S)。

关键：只保留粒子数正确的组态（物理有效扇区），对应 SQD 的组态恢复思想的简化版
     （这里 UCCSD ansatz 天然保粒子数，采到的组态已在正确扇区）。
"""
import numpy as np


def sample_configs(psi, n_shots, n_qubits, n_electrons, seed=0):
    """对态 psi 采样 n_shots 次，返回被采到的（去重）基态索引集合。

    只保留粒子数 == n_electrons 的组态（物理有效）。
    """
    rng = np.random.default_rng(seed)
    probs = np.abs(psi) ** 2
    probs = probs / probs.sum()
    samples = rng.choice(len(psi), size=n_shots, p=probs)
    unique = np.unique(samples)
    # 过滤到正确粒子数扇区
    valid = [int(i) for i in unique
             if bin(int(i)).count("1") == n_electrons]
    return sorted(valid)


def subspace_diagonalize(H, config_indices):
    """在给定组态张成的子空间内对角化 H，返回最小本征值 E_hybrid。"""
    idx = np.array(config_indices, dtype=int)
    H_sub = H[np.ix_(idx, idx)]
    # H_sub 可能非厄米的微小数值误差，取厄米化
    H_sub = 0.5 * (H_sub + H_sub.conj().T)
    w = np.linalg.eigvalsh(H_sub)
    return float(w[0])


def hybrid_energy(psi, H, n_shots, n_qubits, n_electrons, seed=0):
    """完整混合后处理：采样 -> 子空间对角化 -> E_hybrid。

    返回 (E_hybrid, 子空间维数)。
    """
    configs = sample_configs(psi, n_shots, n_qubits, n_electrons, seed=seed)
    if len(configs) == 0:
        return np.nan, 0
    E_hyb = subspace_diagonalize(H, configs)
    return E_hyb, len(configs)


def vqe_energy_in_subspace(psi, H, config_indices):
    """VQE 态投影到采样子空间 S 后的能量 <Psi_S|H|Psi_S>/<Psi_S|Psi_S>。

    这是与 E_hybrid 严格可比的上界：因为采样会丢弃 |Psi> 的小振幅分量，
    严格来说 |Psi> ∉ S，直接比 <Psi|H|Psi> 会有采样截断误差。
    投影到 S 后，|Psi_S> ∈ S 是 S 中一个特定态，故必有 E_hybrid <= E_VQE^S。
    """
    idx = np.array(config_indices, dtype=int)
    psi_s = np.zeros_like(psi)
    psi_s[idx] = psi[idx]
    norm2 = np.real(np.conj(psi_s) @ psi_s)
    if norm2 < 1e-14:
        return np.nan
    e = np.real(np.conj(psi_s) @ (H @ psi_s)) / norm2
    return float(e)
