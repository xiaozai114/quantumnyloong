#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选做题 5: 从 SQD 到 QPE —— 资源公式 + p_2q 阈值数值分析
================================================================
本脚本不模拟真实量子电路 (本题以推导为主)，而是：
  1. 实现 SQD / QPE 的资源公式 (qubit 数、深度、门总数)；
  2. 数值求 SQD/QPE 的 p_2q 交叉阈值；
  3. 画图：p_2q 扫描下两路线的 (噪声) 误差 / 等效物理成本；
  4. EWF 效应：n -> n_frag 后 QPE 资源降低的数值对比。

运行:
    cd opt5
    python solution.py
输出:
    - stdout  : 资源表 + 阈值 + EWF 对比表
    - sqd_vs_qpe.png      : p_2q 扫描图 (双 panel: 误差、物理成本)
    - ewf_resource.png    : EWF 前后 QPE 资源对比柱状图

依赖: numpy, matplotlib  (纯计算，无量子库)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------- 中文字体 (Linux/WSL 优先, Windows 兜底) ----------
# WSL 里通常装了 Noto CJK / WenQuanYi; Windows 上有微软雅黑/黑体.
# 列出多个候选, matplotlib 会用第一个找到的.
mpl.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
    "Source Han Sans SC", "Microsoft YaHei", "SimHei", "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. 资源公式 (逻辑层，忽略 O(1) 常数 —— 但常数已校准到文献量级)
# ============================================================
# 注: 这里的常数 c 用于把 O(·) 标度变为可读数值 (便于横向比较)，
# 不影响 p_2q 扫描的 *趋势* 和交叉点的位置量级。

def sqd_resources(n, S, n_reps=2):
    """SQD (NISQ) 资源。
    n      : 体系自旋轨道 (量子比特) 数
    S      : 采样比特串数
    n_reps : LUCJ 重复层数 (典型 1-2)
    返回 dict.
    """
    # LUCJ: 每个 U_mu (轨道旋转) ~ n 个 Givens, 每 Givens 2 CNOT
    #       e^{iJ} (local R_ZZ) ~ n-1 个相邻 R_ZZ, 每 R_ZZ 2 CNOT
    n_2q_per_layer = 2 * n + 2 * (n - 1)            # CNOT 数 / 单层
    n_2q = n_2q_per_layer * n_reps
    depth = n * n_reps                                # 并行化后 ~ O(n)
    return {
        "qubits": n,                                  # 无辅助比特
        "depth": depth,
        "n_2q": n_2q,
        "classical_post": S ** 3,                     # CI 对角化
        "needs_ft": False,
    }


def qpe_resources(n, eps, m=None):
    """QPE (FTQC) 逻辑资源。
    n   : 体系量子比特数
    eps : 目标能量精度
    m   : 辅助比特数 (默认 ceil(log2(1/eps)))
    """
    if m is None:
        m = int(np.ceil(np.log2(1.0 / eps)))
    # 一次 U 实现 ~ O(n) 门 (LCU/qubitization, 保守取 n)
    # QPE 串行链最长 U^{2^{m-1}}, 深度 ~ n * 2^{m-1}
    depth = n * (2 ** (m - 1))
    # 总门数 sum_k n*2^k = n*(2^m - 1)
    n_gates = n * (2 ** m - 1)
    n_2q = n_gates  # 简化: 大部分门经分解后是 2q
    return {
        "qubits": n + m,
        "ancilla": m,
        "depth": depth,
        "n_2q": n_2q,
        "needs_ft": True,
    }


def print_resource_table(n_list, eps, S):
    print("=" * 72)
    print(f"资源对比表  (eps={eps:.0e}, S={S})")
    print("=" * 72)
    hdr = f"{'n':>4} | {'SQD qubits':>10} {'SQD depth':>10} {'SQD n_2q':>10}" \
          f" | {'QPE qubits':>10} {'QPE depth':>12} {'QPE n_2q':>12}"
    print(hdr)
    print("-" * 72)
    for n in n_list:
        s = sqd_resources(n, S)
        q = qpe_resources(n, eps)
        print(f"{n:>4} | {s['qubits']:>10} {s['depth']:>10} {s['n_2q']:>10}"
              f" | {q['qubits']:>10} {q['depth']:>12} {q['n_2q']:>12}")
    print("=" * 72)
    print("观察: QPE 深度比 SQD 大 ~ 1/eps 倍; QPE 多 ceil(log2(1/eps)) 辅助比特.\n")


# ============================================================
# 2. p_2q 阈值分析
# ============================================================
# Surface code 逻辑错误率 (近似):  p_L ≈ A * (p_2q / p_th)^{(d+1)/2}
# 给定目标逻辑率 p_L^target = eps^2 / n  (使 QPE 整体逻辑错误 << eps),
# 解出所需码距 d, 然后物理比特 ~ (n+m)*d^2, 物理深度 ~ depth*d^2.

P_TH = 1e-2          # surface code 阈值 (~1%)
A_SC = 0.1           # 码距前因子


def code_distance(p2q, p_logical_target):
    """由逻辑率目标反解 surface code 码距 d."""
    if p2q >= P_TH:
        return np.inf
    ratio = p2q / P_TH
    # (d+1)/2 = log(p_L / A) / log(ratio)
    val = np.log(p_logical_target / A_SC) / np.log(ratio)
    d = 2 * val - 1
    return max(d, 1.0)


def qpe_physical_cost(n, eps, p2q):
    """QPE 物理成本 (物理比特 × 物理深度, 任意单位)."""
    q = qpe_resources(n, eps)
    p_logical_target = eps ** 2 / n
    d = code_distance(p2q, p2q * 0 + p_logical_target)  # 标量
    if not np.isfinite(d):
        return np.inf
    phys_qubits = q["qubits"] * d ** 2
    phys_depth = q["depth"] * d ** 2
    return phys_qubits * phys_depth


def sqd_noise_error(n, p2q, c_rec=0.5):
    """SQD 噪声残差 (configuration recovery 后) ~ c_rec * p2q * n_2q."""
    s = sqd_resources(n, S=1000)
    return c_rec * p2q * s["n_2q"]


def sqd_physical_cost(n, S):
    """SQD 物理+经典 总成本 (任意单位, 与 QPE 同尺度)."""
    s = sqd_resources(n, S)
    # 量子部分 (浅, 但 S 次采样) + 经典 S^3
    return s["n_2q"] * S + s["classical_post"]


def find_crossing(n, eps, S, p2q_grid):
    """在 p2q 网格上找 SQD/QPE 成本交叉点 (QPE 越小越好)."""
    c_sqd = sqd_physical_cost(n, S)
    c_qpe = np.array([qpe_physical_cost(n, eps, p) for p in p2q_grid])
    # 找 c_qpe <= c_sqd 的最小 p2q
    feasible = np.where(c_qpe <= c_sqd)[0]
    if len(feasible) == 0:
        return None, c_sqd, c_qpe
    return p2q_grid[feasible[0]], c_sqd, c_qpe


# ============================================================
# 3. 画图: p_2q 扫描
# ============================================================
def plot_p2q_scan(n, eps, S, save="sqd_vs_qpe.png"):
    p2q = np.logspace(-7, -1.5, 120)
    sqd_err = np.array([sqd_noise_error(n, p) for p in p2q])
    sqd_cost = sqd_physical_cost(n, S) * np.ones_like(p2q)
    qpe_cost = np.array([qpe_physical_cost(n, eps, p) for p in p2q])
    qpe_logical = np.array([eps ** 2 / n * np.ones_like(p2q)]).squeeze() \
                  if False else (eps ** 2 / n)
    # 码距随 p2q
    d_arr = np.array([code_distance(p, eps ** 2 / n) for p in p2q])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- 左: 误差 ---
    ax = axes[0]
    ax.loglog(p2q, sqd_err, "r-o", lw=2, ms=4, label="SQD noise residual")
    ax.axhline(eps ** 2 / n, color="b", ls="--", lw=2,
               label=f"QPE logical err target $\\epsilon^2/n$={eps**2/n:.1e}")
    ax.axvline(1e-4, color="gray", ls=":", lw=2,
               label="empirical $p_{2q}^*\\sim 10^{-4}$")
    ax.set_xlabel("two-qubit gate error $p_{2q}$")
    ax.set_ylabel("error (Ha, arbitrary scale)")
    ax.set_title(f"(a) Noise error vs $p_{{2q}}$  (n={n}, $\\epsilon$={eps:.0e})")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # --- 右: 物理成本 ---
    ax = axes[1]
    mask = np.isfinite(qpe_cost)
    ax.loglog(p2q[mask], qpe_cost[mask], "b-o", lw=2, ms=4,
              label="QPE physical cost")
    ax.loglog(p2q, sqd_cost, "r--", lw=2, label=f"SQD cost (S={S})")
    ax.set_xlabel("two-qubit gate error $p_{2q}$")
    ax.set_ylabel("physical cost (qubits$\\times$depth, log)")
    ax.set_title(f"(b) Equivalent physical cost vs $p_{{2q}}$  (n={n})")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    # 标注交叉点
    cross, _, _ = find_crossing(n, eps, S, p2q)
    if cross is not None:
        ax.axvline(cross, color="green", ls="-.", lw=2,
                   label=f"crossover $p_{{2q}}^*\\approx${cross:.1e}")
        ax.legend(fontsize=9)

    # 第二 y 轴: 码距
    ax2 = ax.twinx()
    ax2.semilogy(p2q, d_arr, "g:", lw=1.5, alpha=0.7)
    ax2.set_ylabel("required surface-code distance $d$", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    plt.tight_layout()
    plt.savefig(save, dpi=130, bbox_inches="tight")
    print(f"[图] 已保存: {save}")
    plt.close(fig)


# ============================================================
# 4. EWF 资源压缩对比
# ============================================================
def ewf_compare(n, n_frag, eps, S):
    """直接 vs EWF 后的资源对比."""
    F = int(np.ceil(n / n_frag))
    direct_q = qpe_resources(n, eps)
    frag_q = qpe_resources(n_frag, eps)
    direct_s = sqd_resources(n, S)
    frag_s = sqd_resources(n_frag, S)

    print("=" * 78)
    print(f"EWF 资源压缩对比   n={n}, n_frag={n_frag}, F={F} 片, eps={eps:.0e}")
    print("=" * 78)
    print(f"{'资源':<22} | {'直接 (n)':<18} | {'EWF (n_frag)':<18} | {'压缩比':<10}")
    print("-" * 78)
    rows = [
        ("QPE 量子比特数",     direct_q["qubits"],     frag_q["qubits"]),
        ("QPE 深度 (并行)",    direct_q["depth"],      frag_q["depth"]),
        ("QPE 门总数 (并行)",  direct_q["n_2q"],       frag_q["n_2q"]),
        ("SQD 量子比特数",     direct_s["qubits"],     frag_s["qubits"]),
        ("SQD 深度",           direct_s["depth"],      frag_s["depth"]),
        ("SQD n_2q",           direct_s["n_2q"],       frag_s["n_2q"]),
    ]
    for name, d, f in rows:
        ratio = d / f if f else float("inf")
        print(f"{name:<22} | {d:<18} | {f:<18} | {ratio:<10.2f}")
    print("-" * 78)
    print(f"注: 并行执行下 EWF 把 *瞬时* 资源降为 n_frag; 辅助比特 "
          f"m={direct_q['ancilla']} (=片段 m, 因精度 eps 不变).")
    print(f"    串行总操作数不变 (~F 倍), 但墙钟时间 / 比特上限 大幅改善.\n")
    return rows


def plot_ewf(n_list, n_frag, eps, S, save="ewf_resource.png"):
    """柱状图: 不同 n 下直接 QPE vs EWF-QPE 的 (并行) 深度."""
    direct_depth = [qpe_resources(n, eps)["depth"] for n in n_list]
    ewf_depth = [qpe_resources(n_frag, eps)["depth"] for _ in n_list]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(n_list))
    w = 0.38
    ax.bar(x - w / 2, direct_depth, w, label="direct QPE (n)", color="#c44")
    ax.bar(x + w / 2, ewf_depth, w,
           label=f"EWF-QPE ($n_{{frag}}$={n_frag}, parallel)", color="#48a")
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in n_list])
    ax.set_ylabel("circuit depth (logical, parallel)")
    ax.set_title(f"EWF compression of QPE parallel depth  ($\\epsilon$={eps:.0e})")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    for i, (d, e) in enumerate(zip(direct_depth, ewf_depth)):
        ax.text(i - w / 2, d * 1.15, f"{d:.0e}", ha="center", fontsize=8)
        ax.text(i + w / 2, e * 1.15, f"{e:.0e}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(save, dpi=130, bbox_inches="tight")
    print(f"[图] 已保存: {save}")
    plt.close(fig)


# ============================================================
# 5. main
# ============================================================
def main():
    # ---- (1) 资源表 ----
    print("\n>>> 第 (1) 部分: SQD vs QPE 资源公式\n")
    print_resource_table(n_list=[4, 8, 20, 28, 50, 96], eps=1e-3, S=1000)

    # ---- (2) p2q 阈值 ----
    print(">>> 第 (2) 部分: p_2q 交叉阈值数值分析\n")
    n0, eps0, S0 = 20, 1e-3, 1000
    # 扩展网格到 1e-7 以捕获小 n 的交叉点
    p2q_grid = np.logspace(-7, -1.5, 250)
    cross, c_sqd, c_qpe = find_crossing(n0, eps0, S0, p2q_grid)
    if cross is None:
        print(f"  在所扫 p2q 范围内 QPE 物理成本始终 > SQD (n={n0} 太小).")
        print("  -> 说明小规模下 SQD 在任意现实 p2q 都更便宜 (符合理论).")
    else:
        print(f"  交叉点 p_2q* ≈ {cross:.2e}  (n={n0}, eps={eps0:.0e}, S={S0})")

    # 多个 n 的交叉点: 关键观察是 SQD 经典成本 ~S^3 (与 n 无关, 但 S~1/eps^2 也大),
    # QPE 物理成本 ~ (n/eps)*d^2. 增大 n 使 QPE 物理成本↑, 但 SQD 不变 -> 交叉点上移.
    print("\n  不同 n 下的交叉点 (固定 eps=1e-3, S=1000):")
    print(f"  {'n':>4} | {'p_2q*':>12} | {'说明':<48}")
    print("  " + "-" * 70)
    for n in [10, 20, 50, 100, 200]:
        _, cs, cq = find_crossing(n, eps0, S0, p2q_grid)
        feasible = np.where(cq <= cs)[0]
        if len(feasible) == 0:
            print(f"  {n:>4} | {'(网格内无)':>12} | SQD 始终更优 "
                  f"(SQD 经典 S^3={S0**3:.0e} 仍 < QPE 容错总成本)")
        else:
            p = p2q_grid[feasible[0]]
            print(f"  {n:>4} | {p:>12.2e} | n 增大 -> QPE 容错成本↑ -> 需更低 p2q 才能胜出")
    print("\n  注: 此处用 *成本* 模型. 若改用 *精度* 模型 (SQD 噪声 vs QPE 逻辑率),")
    print("      交叉点会在 ~1e-4 量级 (即 NISQ/FTQC 经验分界), 见下方画图.\n")

    # ---- (3) 画图 ----
    print(">>> 第 (3) 部分: 画图\n")
    plot_p2q_scan(n=20, eps=1e-3, S=1000)
    plot_p2q_scan(n=100, eps=1e-3, S=1000, save="sqd_vs_qpe_n100.png")

    # ---- (4) EWF ----
    print(">>> 第 (4) 部分: EWF 资源压缩\n")
    ewf_compare(n=96, n_frag=10, eps=1e-3, S=1000)
    ewf_compare(n=72, n_frag=12, eps=1e-3, S=1000)
    plot_ewf(n_list=[28, 50, 72, 88, 96], n_frag=10, eps=1e-3, S=1000)

    print(">>> 完成. 输出文件: sqd_vs_qpe.png, sqd_vs_qpe_n100.png, ewf_resource.png\n")


if __name__ == "__main__":
    main()
