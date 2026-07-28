"""QAOA MaxCut 真机/模拟器参考案例（4 节点环 C₄）。

演示 QAOA 端到端流程与后处理：

    电路构造（H 初始化 + 成本层 RZZ(γ) + 混合层 RX(β)）
      -> sample(backend="sim"|"qpu", dry_run=True|False)
      -> 后处理：
         (a) 期望值 <C> = Σ_z P(z) C(z)（MaxCut 切割边数期望）
         (b) top-k 构型（采样中最优的切割方案）
         (c) 近似比 = <C> / C_opt（C_opt 为经典 brute-force 最优）

真机提交同 sqd_qpu_example：设置 QPU_TOKEN 并 dry_run=False。
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common.backend  # noqa: F401
import tensorcircuit as tc

from common.circuits import rzz
from common.hardware import sample, circuit_resource_summary


def maxcut_value(z, edges):
    """构型 z（int，bit i = 节点 i 的归属）的 MaxCut 切割边数。"""
    return sum(((z >> i) & 1) ^ ((z >> j) & 1) for (i, j) in edges)


def brute_force_max(n, edges):
    """经典穷举最优 MaxCut 值。"""
    return max(maxcut_value(z, edges) for z in range(2 ** n))


def build_qaoa(n, edges, gamma, beta):
    """p=1 QAOA 电路：|+>^n -> U_C(γ)=∏ RZZ(2γ) -> U_M(β)=∏ RX(2β)。"""
    c = tc.Circuit(n)
    for i in range(n):
        c.h(i)
    for (i, j) in edges:
        rzz(c, i, j, 2.0 * gamma)   # exp(-i γ Z_i Z_j)（rzz 角度=2γ）
    for i in range(n):
        c.rx(i, theta=2.0 * beta)
    return c


def expectation_and_topk(counts, n, edges, k=3):
    """从 counts 计算 <C>、top-k 构型、近似比。"""
    total = sum(counts.values())
    exp_c = 0.0
    scored = []
    for z, cnt in counts.items():
        cv = maxcut_value(z, edges)
        p = cnt / total
        exp_c += p * cv
        scored.append((z, cv, cnt))
    scored.sort(key=lambda x: (-x[1], -x[2]))
    return exp_c, scored[:k]


def expectation_from_statevector(c, n, edges):
    """用态矢量精确计算 <C>（参数搜索用，避免重复采样）。"""
    from common.circuits import statevector
    psi = statevector(c)
    probs = np.abs(psi) ** 2
    exp_c = 0.0
    for z in range(2 ** n):
        exp_c += probs[z] * maxcut_value(z, edges)
    return float(exp_c)


def grid_search_params(n, edges, n_grid=9):
    """γ,β 网格搜索（经典外环），返回最优 (gamma, beta, <C>)。"""
    best = (0.0, 0.0, -1.0)
    for g in np.linspace(0, np.pi / 2, n_grid):
        for b in np.linspace(0, np.pi / 2, n_grid):
            c = build_qaoa(n, edges, g, b)
            ec = expectation_from_statevector(c, n, edges)
            if ec > best[2]:
                best = (float(g), float(b), ec)
    return best


def run_qaoa_example(backend: str = "sim", device: str = "",
                     dry_run: bool = True, n_shots: int = 4000,
                     gamma: float = None, beta: float = None):
    """C₄ 环 MaxCut QAOA（p=1）。"""
    n = 4
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]   # 4 节点环
    c_opt = brute_force_max(n, edges)

    # 经典外环：γ,β 网格搜索找最优参数（QAOA 的经典-量子混合优化）
    gamma, beta, ec_sv = grid_search_params(n, edges, n_grid=9)
    print(f"[QAOA-example] 网格搜索最优 γ={gamma:.3f} β={beta:.3f} "
          f"-> <C>_sv={ec_sv:.4f}")

    c = build_qaoa(n, edges, gamma, beta)
    res = circuit_resource_summary(c)
    print(f"[QAOA-example] C4 MaxCut | nq={res['nq']} 1q={res['n_1q']} "
          f"2q={res['n_2q']} | backend={backend} device={device or '-'} "
          f"dry_run={dry_run}")

    counts = sample(c, n_shots, n, backend=backend, device=device,
                    dry_run=dry_run, task_label="qaoa_c4_example")

    # 后处理：期望值 / top-k / 近似比
    exp_c, topk = expectation_and_topk(counts, n, edges, k=3)
    ratio = exp_c / c_opt
    print(f"[QAOA-example] <C> = {exp_c:.4f}  C_opt = {c_opt}  "
          f"近似比 = {ratio:.4f}")
    print(f"[QAOA-example] top-3 构型：")
    for z, cv, cnt in topk:
        bitstr = format(z, f"0{n}b")
        print(f"    z={bitstr} (int={z})  cut={cv}/{c_opt}  counts={cnt}")
    # 后处理提示：readout mitigation（如对称化、最小二乘去噪）可进一步提升 <C> 估计
    print("[QAOA-example] 后处理提示：readout mitigation（对称化测量、最小二乘")
    print("    去噪）可进一步修正 <C> 估计；本例为理想/模拟器无需 mitigation。")
    return ratio


if __name__ == "__main__":
    # 默认本地模拟器
    run_qaoa_example(backend="sim")
    print("\n--- 真机提交示例（需 QPU_TOKEN，此处仅 dry-run）---")
    run_qaoa_example(backend="qpu", device="tianji-s2", dry_run=True)
    # 真提交：
    #   export QPU_TOKEN=<your_token>
    #   run_qaoa_example(backend="qpu", device="tianji-s2", dry_run=False)
