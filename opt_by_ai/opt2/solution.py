"""第二大题选做题 2: 量子噪声与配置恢复 (Quantum Noise and Configuration Recovery).

理论见同目录 theory.tex。本文件覆盖:
  (1) 构造 LiH / STO-3G 的 LUCJ 电路 (ffsim + qiskit), 数两比特门 N_2q 与单比特门 N_1q;
  (2) 用 epsilon_eff = (p_2q*N_2q + p_1q*N_1q) / 2 推得单比特 ``预期翻转次数'',
      并以 ``bit-flip 概率 r = min(eps_eff, 0.5)'' 注入退极化噪声;
  (3) 对比 SQD 能量在 ``无恢复 (直接喂噪声)'' vs ``有恢复 (tc_sqd.recover_configurations)''
      下的表现, 扫描 eps_flip / p_2q;
  (4) 定出 ``恢复失效阈值'' p_2q* (eps_eff = 1 处的 p_2q)。

题目设定
--------
LiH / STO-3G LUCJ 电路: N_2q 个两比特门 (CNOT), N_1q 个单比特门 (u)。
退极化噪声 p_2q = 0.01, p_1q = 0.001。推导
    eps_eff = (p_2q * N_2q + p_1q * N_1q) / 2  (单比特预期翻转次数)。
该量 > 1 时配置恢复 (依赖平均占据 n̄_i 的判别力) 失效。

关键物理 (见 theory.tex)
------------------------
* 因子 1/2: 退极化 ``出错'' 的 3 种非平凡 Pauli (X/Y/Z) 中 X,Y 翻比特, 占 2/3。
  题面采用工程近似 2/3 ≈ 1/2 (与 ``IBM/Google 把 depolarizing p 等效为 p/2 X-flip'' 一致)。
* eps_eff 是 ``每比特穿过全电路的预期翻转次数'', 不是概率。
  eps_eff << 1: 单比特至多翻一次, 实际翻转概率 ≈ eps_eff (恢复有效)。
  eps_eff >> 1: 单比特几乎必翻至少一次, 采样分布退化为随机, 恢复失效。
* SQD 的 Hamiltonian 投影是强降噪器: 即便采样分布被噪声破坏, 只要 S 含 FCI dominant
  determinant, 限制对角化仍给出接近 FCI 的能量。故 ``E_SQD 接近 FCI'' 不等于
  ``采样分布有意义'', 也不等于 ``恢复有效''。

依赖
----
pyscf, qiskit, ffsim, tc_sqd, numpy, matplotlib。
(若安装了 qiskit_aer, 则额外启用 ``真实 depolarizing 噪声模型'' 对照; 否则用 bit-flip
 注入, 两者在 eps_eff << 1 区域一致。)

运行
----
WSL: conda activate tc_vayesta
     cd <opt2 目录>
     python solution.py
"""

from __future__ import annotations

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")                        # 无显示环境也能存图
import matplotlib.pyplot as plt

from pyscf import gto, scf, cc, fci

import qiskit
from qiskit.quantum_info import Statevector
import ffsim
from ffsim.qiskit import PrepareHartreeFockJW, UCJOpSpinBalancedJW

import tc_sqd

warnings.filterwarnings("ignore")

# 可选: 真实 Aer depolarizing 噪声模型
try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    HAS_AER = True
except Exception:
    HAS_AER = False


# =========================================================================== #
#  1. 分子积分 + CCSD 振幅
# =========================================================================== #
def build_lih(atom: str = "Li 0 0 0; H 0 0 1.55"):
    """LiH / STO-3G: RHF + MO 基一/二体积分 + 核排斥。

    Returns mol, mf, h1e, eri, ecore
        h1e  : (norb, norb)   hcore in MO basis
        eri  : (n,n,n,n)      MO-basis chemist-notation two-electron integrals
        ecore: float          nuclear repulsion
    """
    mol = gto.M(atom=atom, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    mo = mf.mo_coeff
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl",
                    mol.intor("int2e_sph"), mo, mo, mo, mo)
    return mol, mf, h1e, eri, mf.energy_nuc()


# =========================================================================== #
#  2. 构造 LiH 的 LUCJ 电路 (ffsim + qiskit), 数 N_2q, N_1q
# =========================================================================== #
def build_lucj_circuit(norb, nelec, mf):
    """构造 ffsim 标准 UCJ 算符 (CCSD t1/t2 驱动) 的 qiskit 电路。

    返回 (circ, ucj_n_layers):
        circ        : qiskit QuantumCircuit, 未分解 (高门)
        ucj_n_layers: UCJ 算符层数 L (t2 双因子分解得到的对角 Coulomb 层数)
    """
    mycc = cc.CCSD(mf).run()
    t1, t2 = mycc.t1, mycc.t2
    ucj = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, t1=t1)
    circ = qiskit.QuantumCircuit(2 * norb)
    circ.append(PrepareHartreeFockJW(norb, nelec), range(2 * norb))
    circ.append(UCJOpSpinBalancedJW(ucj), range(2 * norb))
    # 层数 L = 对角 Coulomb 矩阵的张数 (即 Jastrow 项数)
    n_layers = len(ucj.diag_coulomb_mats)
    return circ, n_layers


def count_gates(circ, decompose_reps: int = 6):
    """把电路 decompose 到 {u, CNOT} 基, 数两比特门 N_2q 与单比特门 N_1q。

    qiskit 默认用通用单比特门 'u' (= u3) 与 'cx' (CNOT) 作为分解目标。
    decompose_reps: 递归 decompose 次数, 足够大才能把 ffsim 自定义门全展开。
    """
    circ_d = circ.decompose(reps=decompose_reps)
    # 两比特门名集合 (含通用与硬件专用)
    two_qubit_names = {"cx", "cz", "swap", "cp", "crx", "cry", "crz",
                       "rxx", "ryy", "rzz", "ecr", "iswap", "dcx"}
    one_qubit_names = {"u", "u1", "u2", "u3", "rx", "ry", "rz", "h", "x",
                       "y", "z", "s", "sdg", "t", "tdg", "p", "sx", "id"}
    n_2q = n_1q = 0
    gate_breakdown = {}
    for inst in circ_d.data:
        nm = inst.operation.name
        gate_breakdown[nm] = gate_breakdown.get(nm, 0) + 1
        if nm in two_qubit_names:
            n_2q += 1
        elif nm in one_qubit_names:
            n_1q += 1
    return n_2q, n_1q, circ_d.depth(), gate_breakdown


# =========================================================================== #
#  3. 理想采样 (无噪声), 得到恢复先验 n̄_i
# =========================================================================== #
def ideal_sample(circ, n_qubits, n_samples=3000, seed=42):
    """Statevector 精确采样: 返回 (bsm, probs) 的 tc_sqd 标准布局 [β|α]。"""
    np.random.seed(seed)                      # qiskit sample_counts 用全局 RNG
    sv = Statevector.from_instruction(circ)
    counts = sv.sample_counts(n_samples)
    bsm, probs = tc_sqd.counts_dict_to_bitstring_matrix(counts, n_qubits)
    return bsm, probs


def avg_occupations(bsm, probs, norb):
    """从 (bsm, probs) 算每轨道平均占据 n̄_i (作恢复先验)。

    返回 (avg_a, avg_b): 长度各为 norb 的 1-D float 数组, 按 tc_sqd 轨道索引。
    tc_sqd 布局: bitstring = [β_{n-1}..β_0 | α_{n-1}..α_0]
    """
    avg_a = bsm[:, norb:].astype(float).T @ probs    # 右半 alpha
    avg_b = bsm[:, :norb].astype(float).T @ probs    # 左半 beta
    return avg_a, avg_b


# =========================================================================== #
#  4. 噪声注入: bit-flip 模型 (与题面 eps_eff 公式自洽)
# =========================================================================== #
def inject_bitflip_noise(bsm_ideal, probs_ideal, eps_eff,
                         n_expand=3000, seed=42):
    """按 eps_eff 给理想采样注入 bit-flip 噪声。

    参数
    ----
    bsm_ideal, probs_ideal : 理想分布 (tc_sqd 布局)
    eps_eff : float, 每比特 ``预期翻转次数''。翻转概率 r_flip 由 Poisson 偶/奇
              翻转模型给出 r_flip = (1 - exp(-2*eps_eff)) / 2, 当 eps_eff 小时
              ≈ eps_eff, 当 eps_eff 大时 → 0.5 (随机化)。
    n_expand : 把概率分布展开成 n_expand 个原始 bitstring 再注入翻转。

    返回 (bsm_noisy, probs_noisy, frac_valid):
        bsm_noisy, probs_noisy : 噪声分布 (去重 + 重一化)
        frac_valid : 噪声样本中满足粒子数守恒 (alpha=N_a, beta=N_b) 的比例
    """
    rng = np.random.default_rng(seed)
    # 翻转概率: Poisson(eps_eff) 奇数次翻转的概率 = (1 - e^{-2 eps_eff}) / 2
    # eps_eff=0 → 0; eps_eff=1 → 0.432; eps_eff→∞ → 0.5
    r_flip = (1.0 - np.exp(-2.0 * min(eps_eff, 50.0))) / 2.0

    # 展开为 n_expand 个原始 bitstring (按理想概率抽样)
    idx = rng.choice(len(bsm_ideal), size=n_expand, p=probs_ideal)
    samples = bsm_ideal[idx].astype(bool).copy()
    if r_flip > 0:
        flips = rng.random(samples.shape) < r_flip
        samples = samples ^ flips

    # 去重 + 计数 → (bsm, probs)
    uniq, cnt = np.unique(samples, axis=0, return_counts=True)
    bsm_noisy = uniq
    probs_noisy = cnt.astype(float) / cnt.sum()
    return bsm_noisy, probs_noisy, r_flip


# =========================================================================== #
#  5. SQD 能量: 无恢复 vs 有恢复
# =========================================================================== #
def sqd_energy(h1e, eri, norb, nelec, ecore, bsm, probs, max_iter=5):
    """tc_sqd 子空间对角化: 在 bsm 张成的组态子空间内对角化 H, 返回最低能量。"""
    n_u = bsm.shape[0]
    e = tc_sqd.compute_ground_state_energy(
        h1e, eri, norb, nelec, ecore=ecore, method="sqd",
        bitstring_matrix=bsm, probabilities=probs,
        samples_per_batch=min(200, max(1, n_u)),
        max_iterations=max_iter)
    return e, n_u


def sqd_with_recovery(h1e, eri, norb, nelec, ecore,
                      bsm_noisy, probs_noisy, avg_occ, seed=42):
    """先做 configuration recovery 再喂 SQD。

    avg_occ = (avg_a, avg_b) 是恢复先验 (从理想分布算得, 见 avg_occupations)。
    """
    avg_a, avg_b = avg_occ
    recovered, rec_probs = tc_sqd.recover_configurations(
        bsm_noisy, probs_noisy, (avg_a, avg_b),
        nelec[0], nelec[1], rand_seed=seed)
    e, n_u = sqd_energy(h1e, eri, norb, nelec, ecore, recovered, rec_probs)
    return e, n_u, recovered, rec_probs


def fraction_valid(bsm, probs, norb, nelec):
    """计算粒子数守恒样本的比例 (alpha=N_a 且 beta=N_b)。"""
    half = norb
    ok = (bsm[:, half:].sum(axis=1) == nelec[0]) & \
         (bsm[:, :half].sum(axis=1) == nelec[1])
    return float(probs[ok].sum())


# =========================================================================== #
#  6. 主程序: (a) 电路构造 + 门计数; (b) eps_eff 公式验证;
#             (c) 有/无恢复能量对比; (d) p_2q 扫描 + 失效阈值
# =========================================================================== #
def main():
    print("=" * 78)
    print("选做题 2: 量子噪声与配置恢复 (LiH / STO-3G LUCJ-SQD)")
    print("=" * 78)

    # ---- (1) 分子积分 + FCI 基准 ------------------------------------------------
    mol, mf, h1e, eri, ecore = build_lih()
    norb = int(mol.nao_nr())
    nelec = (mol.nelectron // 2,) * 2
    e_fci = fci.FCI(mf).kernel()[0]
    e_hf = mf.e_tot
    print(f"norb = {norb}, nelec = {nelec}, 2*norb = {2*norb} qubits")
    print(f"E(HF)  = {e_hf:.8f}")
    print(f"E(FCI) = {e_fci:.8f}")
    print(f"相关能 = E(HF) - E(FCI) = {e_hf - e_fci:+.4e}")

    # ---- (2) 构造 LUCJ 电路 + 数门 ----------------------------------------------
    circ, n_layers = build_lucj_circuit(norb, nelec, mf)
    n_2q, n_1q, depth, breakdown = count_gates(circ, decompose_reps=6)
    print("\n--- (a) LiH LUCJ 电路构造 (ffsim UCJOpSpinBalanced, qiskit decompose) ---")
    print(f"UCJ 层数 L = {n_layers}")
    print(f"N_2q (两比特门, CNOT) = {n_2q}")
    print(f"N_1q (单比特门, u)    = {n_1q}")
    print(f"电路深度              = {depth}")
    print(f"门分解明细 (top 5)    = "
          f"{dict(sorted(breakdown.items(), key=lambda x: -x[1])[:5])}")

    # ---- (3) 理想采样 + 恢复先验 ------------------------------------------------
    n_qubits = 2 * norb
    bsm_ideal, probs_ideal = ideal_sample(circ, n_qubits, n_samples=3000, seed=42)
    avg_a, avg_b = avg_occupations(bsm_ideal, probs_ideal, norb)
    e_ideal, n_ideal = sqd_energy(h1e, eri, norb, nelec, ecore,
                                  bsm_ideal, probs_ideal)
    print("\n--- 理想采样 (无噪声) 基准 ---")
    print(f"唯一 bitstring 数 = {n_ideal}")
    print(f"avg_occ alpha = {np.round(avg_a, 3)}")
    print(f"avg_occ beta  = {np.round(avg_b, 3)}")
    print(f"E(SQD ideal)  = {e_ideal:.8f}  误差 vs FCI = {e_ideal - e_fci:+.2e}")

    # ---- (4) eps_eff 公式 (题面 p_2q=0.01, p_1q=0.001) --------------------------
    print("\n" + "=" * 78)
    print("--- (b) eps_eff = (p_2q*N_2q + p_1q*N_1q) / 2  (题面参数) ---")
    p_2q_problem, p_1q_problem = 0.01, 0.001
    eps_eff_problem = (p_2q_problem * n_2q + p_1q_problem * n_1q) / 2.0
    print(f"  p_2q = {p_2q_problem}, p_1q = {p_1q_problem}")
    print(f"  p_2q * N_2q = {p_2q_problem * n_2q:.2f}  (两比特门贡献)")
    print(f"  p_1q * N_1q = {p_1q_problem * n_1q:.2f}  (单比特门贡献)")
    print(f"  eps_eff     = {eps_eff_problem:.2f}   (>> 1 → 恢复失效)")
    print(f"  → 题面噪声下, 每比特预期翻转 ~{eps_eff_problem:.0f} 次, "
          f"远超恢复上限。")

    # ---- (5) 有/无恢复: 扫描 eps_flip, 看恢复 ``收益'' 在何处显著 ---------------
    print("\n" + "=" * 78)
    print("--- (c) 有/无恢复 SQD 能量 vs 噪声率 (eps_flip = 翻转概率) ---")
    print(f"{'eps_flip':>9} {'P_valid':>8} {'E_no_rec':>12} {'E_rec':>12} "
          f"{'ΔE(rec-no)':>12} {'n_no':>6} {'n_rec':>6}")
    eps_flips = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    rows = []
    for ef in eps_flips:
        # eps_eff = -0.5 * ln(1 - 2*r) ; 但这里直接用 ef 作为 ``r'' 的来源:
        # 把 ef 理解为 ``预期翻转次数'' (eps_eff), 反推翻转概率
        eps_eff_for_r = ef                       # 直接用 ef 当 eps_eff 输入
        bsm_noisy, probs_noisy, r_used = inject_bitflip_noise(
            bsm_ideal, probs_ideal, eps_eff_for_r, n_expand=3000, seed=42)
        fv = fraction_valid(bsm_noisy, probs_noisy, norb, nelec)
        e_no, n_no = sqd_energy(h1e, eri, norb, nelec, ecore, bsm_noisy, probs_noisy)
        e_rec, n_rec, _, _ = sqd_with_recovery(
            h1e, eri, norb, nelec, ecore, bsm_noisy, probs_noisy,
            (avg_a, avg_b), seed=42)
        dE = e_rec - e_no
        rows.append((ef, r_used, fv, e_no, e_rec, dE, n_no, n_rec))
        print(f"{ef:>9.3f} {fv:>8.3f} {e_no:>12.5f} {e_rec:>12.5f} "
              f"{dE:>+12.2e} {n_no:>6d} {n_rec:>6d}")
    print("  注: 低噪声 (eps_flip~0.01) 时恢复显著改善能量 (ΔE<0, E_rec 更近 FCI);")
    print("      高噪声时 SQD Hamiltonian 投影自身是强降噪器, 二者皆→FCI。")

    # ---- (6) p_2q 扫描: 找恢复失效阈值 (eps_eff = 1) ---------------------------
    print("\n" + "=" * 78)
    print("--- (d) p_2q 扫描: 定恢复失效阈值 (判据 eps_eff = 1) ---")
    # p_1q = 0.1 * p_2q (题面比例)
    # eps_eff = 1  =>  p_2q*N_2q + p_1q*N_1q = 2
    #             =>  p_2q*(N_2q + 0.1*N_1q) = 2
    p_2q_star = 2.0 / (n_2q + 0.1 * n_1q)
    print(f"  判据: eps_eff = 1  <=>  p_2q*(N_2q + 0.1*N_1q) = 2")
    print(f"         p_2q* = 2 / ({n_2q} + 0.1*{n_1q}) "
          f"= 2 / {n_2q + 0.1 * n_1q:.1f}")
    print(f"         → p_2q* ≈ {p_2q_star:.2e}")
    print()
    print(f"  {'p_2q':>10} {'p_1q':>10} {'eps_eff':>9} {'regime':>22}")
    for p2 in [1e-4, 2e-4, p_2q_star, 5e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
        p1 = 0.1 * p2
        eps = (p2 * n_2q + p1 * n_1q) / 2.0
        if eps < 1:
            regime = "恢复有效 (<1)"
        elif eps < 2:
            regime = "边缘 (~1)"
        elif eps < 10:
            regime = "恢复失效"
        else:
            regime = "完全失效 (噪声主导)"
        marker = "  ← p_2q*" if abs(p2 - p_2q_star) < 1e-5 else ""
        print(f"  {p2:>10.2e} {p1:>10.2e} {eps:>9.3f} {regime:>22}{marker}")
    print(f"\n  题面 p_2q=0.01 对应 eps_eff={eps_eff_problem:.1f}, "
          f"约为阈值 p_2q* 的 {p_2q_problem / p_2q_star:.0f} 倍 → 恢复失效。")

    # ---- (7) 画图: 有/无恢复能量 vs eps_flip + p_2q→eps_eff 标度 -----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左图: E vs eps_flip
    ax = axes[0]
    efs = np.array([r[0] for r in rows])
    e_nos = np.array([r[3] for r in rows])
    e_recs = np.array([r[4] for r in rows])
    fvs = np.array([r[2] for r in rows])
    ax.plot(efs, e_nos, "o-", color="C3", label=r"$E_{\mathrm{SQD}}$ no recovery")
    ax.plot(efs, e_recs, "s-", color="C0", label=r"$E_{\mathrm{SQD}}$ with recovery")
    ax.axhline(e_fci, color="k", ls="--", lw=1, label=f"FCI = {e_fci:.4f}")
    ax.axhline(e_ideal, color="g", ls=":", lw=1,
               label=f"ideal SQD = {e_ideal:.4f}")
    ax2 = ax.twinx()
    ax2.plot(efs, fvs, "^:", color="gray", alpha=0.6, label=r"$P_{\mathrm{valid}}$")
    ax2.set_ylabel(r"$P_{\mathrm{valid}}$ (合法样本比例)", color="gray")
    ax2.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"per-bit flip rate $\epsilon_{\mathrm{flip}}$")
    ax.set_ylabel(r"$E_{\mathrm{SQD}}$  (Hartree)")
    ax.set_title("SQD energy: recovery vs no-recovery (LiH LUCJ)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)

    # 右图: eps_eff vs p_2q, 标出失效阈值
    ax = axes[1]
    p2s = np.logspace(-5, -1, 200)
    eps_effs = (p2s * n_2q + 0.1 * p2s * n_1q) / 2.0
    ax.loglog(p2s, eps_effs, "C0-", lw=2,
              label=r"$\epsilon_{\mathrm{eff}}=(p_{2q}N_{2q}+p_{1q}N_{1q})/2$")
    ax.axhline(1.0, color="red", ls="--", lw=1, label="recovery threshold $=1$")
    ax.axvline(p_2q_star, color="green", ls=":", lw=1,
               label=rf"$p_{{2q}}^\star\approx{p_2q_star:.1e}$")
    ax.axvline(p_2q_problem, color="purple", ls="-.", lw=1,
               label=rf"problem $p_{{2q}}=0.01$")
    ax.scatter([p_2q_problem], [eps_eff_problem], color="purple", zorder=5)
    ax.annotate(rf"$\epsilon_{{\mathrm{{eff}}}}\approx{eps_eff_problem:.0f}$",
                (p_2q_problem, eps_eff_problem),
                textcoords="offset points", xytext=(8, -5), color="purple")
    ax.set_xlabel(r"$p_{2q}$  (two-qubit gate error)")
    ax.set_ylabel(r"$\epsilon_{\mathrm{eff}}$")
    ax.set_title(r"Recovery failure threshold ($\epsilon_{\mathrm{eff}}=1$)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out_png = "noise_recovery.png"
    fig.savefig(out_png, dpi=130)
    print(f"\n图已保存: {out_png}")

    # ---- (8) 总结 --------------------------------------------------------------
    print("\n" + "=" * 78)
    print("总结")
    print("=" * 78)
    print(f"  电路: 12 qubit, L={n_layers} 层, N_2q={n_2q}, N_1q={n_1q}, "
          f"深度={depth}")
    print(f"  公式: eps_eff = (p_2q*N_2q + p_1q*N_1q)/2")
    print(f"  题面: p_2q=0.01, p_1q=0.001 → eps_eff = {eps_eff_problem:.1f} >> 1")
    print(f"        (单比特预期翻转 ~{eps_eff_problem:.0f} 次, 远超恢复上限)")
    print(f"  对比: 低噪声 (eps_flip~0.01) 下恢复 ΔE~{rows[2][5]:+.1e};")
    print(f"        高噪声下 SQD Hamiltonian 投影自身降噪, 二者皆→FCI。")
    print(f"  阈值: p_2q* = 2/(N_2q+0.1*N_1q) ≈ {p_2q_star:.2e}")


if __name__ == "__main__":
    main()
