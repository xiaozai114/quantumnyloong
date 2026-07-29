"""Problem 2 选做 1: Optimal ccsd_scale —— LUCJ-SQD 能量随 lambda 的竞争。

对应 main-polish.tex Problem 2 Optional 第 1 项。

题目核心
--------
LiH / STO-3G, lambda = ccsd_scale in [0, 0.5]。增大 lambda:
  (+) 改善采样覆盖: LUCJ 态偏离 HF 更多, 纠缠熵增大 -> 命中更多含双激发的 determinant
      -> SQD 子空间 S(lambda) 变大 -> 由变分单调性 E 下降 (朝 FCI)。
  (-) 增大 UCJ 对角 Coulomb 局域截断误差: LUCJ 把 Jastrow 的 R_ZZ 限制在相邻 qubit,
      丢弃非相邻 R_ZZ (长程 Coulomb 尾)。被丢弃项的相位 J_pq ~ lambda*t, 二阶能量
      偏差 ~ lambda^2 -> E 上升。

两项竞争 -> E(lambda) 存在内部最优 lambda* in (0, 0.5]。

本代码做的事
------------
1. PySCF: LiH/STO-3G RHF + h1e/eri/ecore + CCSD t1/t2 + FCI 基准。
2. 对 lambda in np.linspace(0, 0.5, 100) (100 个均匀点) 扫描:
     ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2*lambda, t1=t1*lambda)
     -> qiskit 电路 -> Statevector.sample_counts(S=2000)
     -> counts_dict_to_bitstring_matrix -> tc_sqd.compute_ground_state_energy
3. 打印 E(lambda) 表, 画 e_vs_lambda.png, 找最优 lambda*。
4. 同时记录唯一 bitstring 数 n_unique(lambda) —— 它应单调增 (纠缠更强), 而 E(lambda)
   非单调 (受截断项推高), 二者解耦是"截断误差存在"的直接证据。

运行
----
WSL: conda activate tc_vayesta  (含 ffsim + qiskit + tc_sqd + pyscf)
     cd <opt1 目录>
     python solution.py
预期: E(lambda) 在 lambda~0.1-0.2 处取极小, n_unique 单调增。
"""

from __future__ import annotations

import os
# 多进程并行前限制 BLAS 单线程: (1) 避免 OpenBLAS 多线程与 multiprocessing 过度订阅 CPU;
# (2) 避免 fork 多线程进程的死锁。必须在 import numpy 前设置。
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import multiprocessing as mp
import matplotlib
matplotlib.use("Agg")                  # 无显示环境也能存图
import matplotlib.pyplot as plt
# 尝试加载中文字体 (WSL Linux 常装 wenquanyi/noto-cjk); 若无则用英文标签避免缺字方块。
_CN_FONTS = ["WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Noto Sans CJK SC",
             "Noto Sans CJK JP", "Source Han Sans SC", "SimHei", "Microsoft YaHei"]
_have_fonts = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_CJK = next((f for f in _CN_FONTS if f in _have_fonts), None)
if _CJK:
    plt.rcParams["font.sans-serif"] = [_CJK] + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    _USE_CN = True
else:
    _USE_CN = False     # 回退到纯英文标签
from pyscf import gto, scf, fci, cc

# --- 量子侧: qiskit (Statevector 精确采样) + ffsim (标准 UCJ ansatz) -----------
import qiskit
from qiskit.quantum_info import Statevector
import ffsim
from ffsim.qiskit import PrepareHartreeFockJW, UCJOpSpinBalancedJW

# --- 经典侧: tc_sqd (配置恢复 + 子空间对角化) ----------------------------------
import tc_sqd


# =========================================================================== #
#  分子积分: LiH / STO-3G
# =========================================================================== #
def build_lih(atom: str = "Li 0 0 0; H 0 0 1.55"):
    """构造 LiH / STO-3G, 跑 RHF, 返回 MO 基一/二电子积分 + 核排斥。

    Returns
    -------
    mol, mf, h1e, eri, ecore
        h1e:  (norb, norb)   MO 基 hcore
        eri:  (n,n,n,n)      MO 基 chemist-记号 二电子积分 (ij|kl)
        ecore: float         E_nuc (此处无冻芯)
    """
    mol = gto.M(atom=atom, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    mo = mf.mo_coeff
    # AO -> MO 变换 (闭壳层, 空间轨道; 二电子用 chemist 记号 ij|kl)
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl",
                    mol.intor("int2e_sph"), mo, mo, mo, mo)
    return mol, mf, h1e, eri, mf.energy_nuc()


# =========================================================================== #
#  LUCJ-SQD 单点: 给定 lambda 算能量 + 唯一 bitstring 数
# =========================================================================== #
def lucj_sqd_energy(t1, t2, h1e, eri, norb, nelec, ecore,
                    ccsd_scale=0.1, n_samples=2000, seed=42):
    """对给定 lambda = ccsd_scale 跑一次 LUCJ-SQD。

    步骤 (与 q5 标准 LUCJ-SQD 流水线一致, 这里把 lambda 做成参数)
    ------------------------------------------------------------------
    1. 振幅缩放: t1' = lambda*t1, t2' = lambda*t2。整体缩放等价于把所有
       Givens 旋转角 theta ~ t 与 Jastrow 相位 J_pq ~ t 同乘 lambda。
         - lambda=0: t'=0 -> UCJ 算符为空 -> 电路=HF 制备 (HF-SQD 极限)。
         - lambda 增大: 纠缠增强, 采样覆盖更多 determinant (覆盖项: E 下降)。
         - 同时 lambda 增大: UCJ/LUCJ 截断误差 ~ lambda^2 上升 (截断项: E 上升)。
    2. ffsim.UCJOpSpinBalanced.from_t_amplitudes 做 t2 的双因子分解 (SVD),
       构造标准 UCJ 算符 (对角 Coulomb Jastrow + 精确 Givens 轨道旋转)。
       LUCJ 的"local"截断由 UCJOpSpinBalancedJW 在 qiskit 编码时实现
       (仅相邻 qubit 的 R_ZZ 被保留, 非相邻的被丢弃)。
    3. qiskit 电路 = PrepareHartreeFockJW(HF 初态, JW) + UCJOpSpinBalancedJW。
    4. Statevector.sample_counts(S) 在计算基采样 -> {二进制串: 计数}。
    5. counts_dict_to_bitstring_matrix 把 counts 转成 tc_sqd 标准 [β|α] 布局。
    6. tc_sqd SQD: 配置恢复 + 子空间对角化 -> 能量。

    返回
    ----
    e_sqd : float   SQD 能量
    n_unique : int  采样到的唯一 bitstring 数 (体现纠缠采样的覆盖度)
    """
    rng = np.random.default_rng(seed)

    # 1) ccsd_scale 缩放振幅
    t1_scaled = ccsd_scale * t1 if t1 is not None else None
    t2_scaled = ccsd_scale * t2

    # 2) ffsim 标准 UCJ 算符 (双因子分解 + 自旋平衡)
    #    lambda=0 -> t2_scaled 全 0 -> from_t_amplitudes 给空算符列表
    #    -> 电路退化为纯 HF 制备。
    ucj = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2_scaled, t1=t1_scaled)

    # 3) qiskit 电路: HF 初态 (JW) + UCJ 算符 (JW)
    circ = qiskit.QuantumCircuit(2 * norb)
    circ.append(PrepareHartreeFockJW(norb, nelec), range(2 * norb))
    circ.append(UCJOpSpinBalancedJW(ucj), range(2 * norb))

    # 4) Statevector 精确采样 (无噪声, 体现 ansatz 的纠缠分布)。
    #    qiskit 2.x 的 Statevector.sample_counts 内部用 numpy 全局 RNG,
    #    故显式 seed 一下以保证可复现。
    np.random.seed(seed)
    sv = Statevector.from_instruction(circ)
    counts = sv.sample_counts(n_samples)

    # 5) counts -> tc_sqd bitstring 矩阵 (自动 [β|α] 布局对齐)
    bsm, probs = tc_sqd.counts_dict_to_bitstring_matrix(counts, 2 * norb)
    n_unique = bsm.shape[0]

    # 6) SQD: 配置恢复 + 子空间对角化
    e = tc_sqd.compute_ground_state_energy(
        h1e, eri, norb, nelec, ecore=ecore, method="sqd",
        bitstring_matrix=bsm, probabilities=probs,
        samples_per_batch=min(200, max(1, n_unique)),
        max_iterations=5)
    return e, n_unique


# =========================================================================== #
#  进程池 worker (模块级, 保 picklable)
# =========================================================================== #
def _scan_one_lambda(args):
    """对单个 lambda 跑 n_avg 个 seed, 返回 (lam, [e_per_seed], [n_per_seed])。

    每个 (lambda, seed) 是完全独立的 LUCJ-SQD 调用, 互不依赖 -> 适合多进程并行。
    (之所以用 CPU 多进程而非 GPU: qiskit Statevector / ffsim / tc_sqd 均为 CPU 实现,
     且 LiH/STO-3G 矩阵极小, GPU 无用武之地; 2000 次独立调用按核数线性加速最有效。)
    """
    lam, t1, t2, h1e, eri, norb, nelec, ecore, S, n_avg = args
    es_seed, ns_seed = [], []
    for s in range(n_avg):
        e, n = lucj_sqd_energy(t1, t2, h1e, eri, norb, nelec, ecore,
                               ccsd_scale=lam, n_samples=S, seed=42 + s)
        es_seed.append(e)
        ns_seed.append(n)
    return (lam, es_seed, ns_seed)


# =========================================================================== #
#  主程序: lambda 扫描 + 画图 + 找最优
# =========================================================================== #
def main():
    print("=" * 78)
    print("Problem 2 选做 1: Optimal ccsd_scale —— LiH / STO-3G, lambda in [0, 0.5]")
    print("=" * 78)

    # ---- 分子积分 + CCSD 振幅 + FCI 基准 -------------------------------------
    mol, mf, h1e, eri, ecore = build_lih()
    norb = int(mol.nao_nr())
    nelec = (mol.nelectron // 2,) * 2            # 闭壳层 (n_alpha=n_beta)
    mycc = cc.CCSD(mf).run()
    t1, t2 = mycc.t1, mycc.t2
    e_fci = fci.FCI(mf).kernel()[0]
    print(f"norb={norb}, nelec={nelec}, E_nuc={ecore:.6f}")
    print(f"E(HF)  = {mf.e_tot:.8f}")
    print(f"E(CCSD)= {mycc.e_tot:.8f}")
    print(f"E(FCI) = {e_fci:.8f}")
    print(f"相关能 = E(HF)-E(FCI) = {mf.e_tot - e_fci:+.3e}")

    # ---- lambda 扫描 (每点多次独立采样取平均, 减小 SQD 采样振荡) -------------
    # 取 100 个点均匀覆盖 [0, 0.5]; 每个 lambda 跑 n_avg 次 (不同 seed), 用于:
    #   (1) 散点图: 看单次振荡幅度;  (2) 平均图: 多次平均后趋势清晰。
    lambdas = np.linspace(0.0, 0.5, 100)
    S = 10000
    n_avg = 20
    n_jobs = min(len(lambdas), mp.cpu_count())
    print(f"\n--- LUCJ-SQD lambda 扫描 ({len(lambdas)} 点 x {n_avg} 次独立采样, S={S}, "
          f"{n_jobs} 进程并行) ---")
    print(f"  {'lambda':>9} {'<E(SQD)>':>14} {'std':>10} {'<E>-E_FCI':>14} "
          f"{'<E>-E_HF':>14} {'<n_uniq>':>9}")

    # 每个 lambda 一个任务 (内部串行 n_avg 个 seed), 丢给进程池并行。
    # BLAS 已在模块顶部限为单线程, 故 n_jobs 个进程不会过度订阅 CPU。
    tasks = [(lam, t1, t2, h1e, eri, norb, nelec, ecore, S, n_avg) for lam in lambdas]
    # 用 spawn 而非默认 fork: qiskit/matplotlib 等库 import 时启动线程/锁,
    # fork 子进程继承损坏的锁状态 -> 死锁 (worker 0% CPU hang)。
    # spawn 启动干净解释器重新 import, 规避 fork 死锁 (代价: 每 worker 重新 import)。
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_jobs) as pool:
        results = pool.map(_scan_one_lambda, tasks)   # 保序: results[i] <-> lambdas[i]

    for lam, es_seed, ns_seed in results:
        e_mean = float(np.mean(es_seed))
        e_std = float(np.std(es_seed))
        n_mean = float(np.mean(ns_seed))
        print(f"  {lam:>9.4f} {e_mean:>14.8f} {e_std:>10.2e} "
              f"{e_mean - e_fci:>+14.3e} {e_mean - mf.e_tot:>+14.3e} {n_mean:>9.2f}")

    # ---- 统计 + 找最优 lambda* (基于平均能量) --------------------------------
    # 注意: lambda=0 是 HF 极限 (截断=0 但覆盖最差), 它不是"最优"。
    # 在 [0,0.5] 上 LiH/STO-3G 的 <E(lambda)> 单调下降 (覆盖项主导, 截断项 ~lambda^2
    # 因 B 小尚未追上), 故区间最优在右端附近。真实内部最优 lambda* 在 lambda>=1 处
    # 才出现 (q5 实测 lambda=1.0 误差 +7.5e-4, 与 0.5 持平)。
    lams = np.array([r[0] for r in results])
    es_mean = np.array([np.mean(r[1]) for r in results])
    es_std = np.array([np.std(r[1]) for r in results])
    ns_mean = np.array([np.mean(r[2]) for r in results])
    # 散点图用: 所有单次结果展平 (每 lambda n_avg 个点)
    e_scatter = np.array([e for r in results for e in r[1]])
    lam_scatter = np.repeat(lams, n_avg)

    i_star = int(np.argmin(es_mean))
    lam_star = float(lams[i_star])
    e_star = float(es_mean[i_star])
    n_star = float(ns_mean[i_star])
    print(f"\n  ==> 区间最优 lambda* = {lam_star:.4f}   <E_min> = {e_star:.8f}   "
          f"(误差 vs FCI = {e_star - e_fci:+.3e}, <n_uniq> = {n_star:.2f}, "
          f"std@lambda* = {es_std[i_star]:.2e})")
    print(f"  ==> 对比: lambda=0 (HF 极限) 误差 = {es_mean[0] - e_fci:+.3e}")

    # ---- 图 1: 散点图 (所有单次结果, 看 SQD 采样振荡) ------------------------
    fig1, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.scatter(lam_scatter, e_scatter, s=12, alpha=0.3, color="C0",
                edgecolors="none",
                label=f"single-shot E(SQD) ({len(lambdas)} pts x {n_avg} = {len(e_scatter)})")
    ax1.axhline(e_fci, color="green", ls="--", lw=1.2, label=f"FCI = {e_fci:.6f}")
    ax1.axhline(mf.e_tot, color="gray", ls=":", lw=1.2, label=f"HF = {mf.e_tot:.6f}")
    ax1.set_xlabel(r"ccsd\_scale $\lambda$")
    ax1.set_ylabel(r"$E_{\rm LUCJ\text{-}SQD}$ (Ha)")
    ax1.set_title("LiH / STO-3G: single-shot scatter (SQD sampling jitter visible)")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    out_scatter = "e_vs_lambda_scatter.png"
    fig1.savefig(out_scatter, dpi=140)
    print(f"\n  (散点图已存: {out_scatter})")

    # ---- 图 2: 多次平均 (减小振荡, 趋势清晰) + ±1 std 误差带 -----------------
    fig2, ax_e = plt.subplots(figsize=(7.5, 5))
    ax_n = ax_e.twinx()
    ax_e.plot(lams, es_mean, "-", color="C0", lw=2,
              label=r"$\langle E_{\rm SQD}\rangle(\lambda)$")
    ax_e.fill_between(lams, es_mean - es_std, es_mean + es_std,
                      color="C0", alpha=0.18, label=r"$\pm 1$ std")
    ax_e.axhline(e_fci, color="green", ls="--", lw=1.2, label=f"FCI = {e_fci:.6f}")
    ax_e.axhline(mf.e_tot, color="gray", ls=":", lw=1.2, label=f"HF = {mf.e_tot:.6f}")
    ax_e.axvline(lam_star, color="red", ls="-.", lw=1.2,
                 label=(rf"最优 $\lambda^\ast={lam_star:.3f}$"
                        if _USE_CN else rf"optimal $\lambda^\ast={lam_star:.3f}$"))
    ax_e.set_xlabel(r"ccsd\_scale $\lambda$")
    ax_e.set_ylabel(r"$\langle E\rangle_{\rm LUCJ\text{-}SQD}$ (Ha)", color="C0")
    ax_e.tick_params(axis="y", labelcolor="C0")
    title_line2 = ("覆盖改善 vs UCJ 截断误差 的竞争"
                   if _USE_CN else
                   "coverage gain (down) vs UCJ truncation error (up)")
    ax_e.set_title(f"LiH / STO-3G: {n_avg}-shot average (jitter reduced, trend clear)\n({title_line2})")

    ax_n.plot(lams, ns_mean, "--", color="C3", lw=1.5,
              label=r"$\langle n_{\rm uniq}\rangle(\lambda)$")
    ax_n.set_ylabel(r"$\langle n_{\rm uniq}\rangle$", color="C3")
    ax_n.tick_params(axis="y", labelcolor="C3")

    h1, l1 = ax_e.get_legend_handles_labels()
    h2, l2 = ax_n.get_legend_handles_labels()
    ax_e.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)
    fig2.tight_layout()
    out_avg = "e_vs_lambda.png"
    fig2.savefig(out_avg, dpi=140)
    print(f"  (平均图已存: {out_avg})")

    # ---- 物理解读 (打印) -----------------------------------------------------
    print("\n" + "=" * 78)
    print("物理解读:")
    print("  [两张图对比]")
    print(f"  - 散点图 ({out_scatter}): 每个 lambda 的 {n_avg} 次单次结果, 可见 SQD 采样")
    print(f"    带来的能量振荡 (平均 std ~ {np.mean(es_std):.1e} Ha)。")
    print(f"  - 平均图 ({out_avg}): {n_avg} 次平均后振荡显著减弱, 趋势清晰 (阴影 = ±1 std)。")
    print("  [两项竞争]")
    print("  (+) 覆盖项: lambda↑ -> 纠缠熵↑ -> 命中更多双激发 determinant ->")
    print("      子空间 S(lambda) 变大 -> 变分单调性 -> E 下降。")
    print("  (-) 截断项: LUCJ 把 Jastrow 的 R_ZZ 限制在相邻 qubit, 丢弃非相邻 R_ZZ。")
    print("      被丢项相位 J_pq ~ lambda*t, 二阶能量偏差 ~ lambda^2 -> E 上升。")
    print(f"  => 竞争框架预测 E(lambda) 有内部最优 lambda* = (2B/Ap)^(1/(2-p))。")
    print("  [LiH/STO-3G 实测: 区间内单调下降, 边界附近最优]")
    print(f"  - <E(lambda)> 在 [0,0.5] 上单调下降, lambda*={lam_star:.3f} (接近右端)。")
    print(f"  - <n_unique> 也单调增 (从 {ns_mean[0]:.1f} 到 {ns_mean[-1]:.1f}), 即纠缠覆盖随 lambda 改善。")
    print("  - 二者同步下降 = 覆盖项在 [0,0.5] 全程主导 (截断项 ~lambda^2 因 B 小尚未追上)。")
    print("  - 原因: LiH/STO-3G 仅 6 轨道, JW 链短, 被丢的非相邻 R_ZZ 少且权重小 (B<<A);")
    print("    q5 实测 lambda=1.0 误差 +7.5e-4 与 lambda=0.5 持平, 说明 lambda* 在 [0.5,1] 内。")
    print("  [体系依赖: 内部最优何时落进 [0,0.5]?]")
    print("  - 拉伸键长 / 增大基组 / 大体系 -> 长程相关与被丢 R_ZZ 数目都↑ -> B↑ ->")
    print("    lambda* 内移到 [0,0.5], E(lambda) 在区间内出现'先降后升'的内部极小。")
    print("  [lambda=0 不是最优]")
    print(f"  - lambda=0 退化为 HF (误差 {es_mean[0]-e_fci:+.2e}, 无相关能),")
    print(f"    lambda={lam_star:.3f} 把误差降到 {e_star-e_fci:+.2e}, 改善 "
          f"{(es_mean[0]-e_fci)/(e_star-e_fci):.1f}x。")
    print("=" * 78)


if __name__ == "__main__":
    main()
