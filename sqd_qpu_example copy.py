"""SQD λ 扫描案例（H₂ 关联能 vs LUCJ 振幅缩放 λ）。

在 sqd_qpu_example.py 的端到端流程基础上，扫描 LUCJ ansatz 的 CCSD 振幅
缩放因子 λ（``build_lucj(..., ccsd_scale=lam)``），观察 SQD 能量随 λ 的变化：

    for λ in scan:
        HF 制备 + LUCJ(ccsd_scale=λ)
          -> sample(backend="sim")              # common.hardware
          -> config_recovery_counts(directed)   # common.sqd
          -> run_sqd_product (α×β 笛卡尔积子空间对角化)
          -> E_SQD(λ)

最后绘制 E_SQD vs λ 曲线，并叠加 E_HF / E_FCI 参考线，输出 PNG。

说明：molecule_report("H2") 只依赖分子本身、与 λ 无关，故提到循环外只算
一次，既省时也保证所有 λ 共享同一套 (h1e, eri, ecore) 与参考能量。
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无显示环境下也能出图
import matplotlib.pyplot as plt
from matplotlib import font_manager


def _setup_cjk_font():
    """注册可用的中文字体，避免中文显示为方块。"""
    candidates = [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
                name = font_manager.FontProperties(fname=path).get_name()
                plt.rcParams["font.sans-serif"] = [name]
                plt.rcParams["axes.unicode_minus"] = False
                return name
            except Exception:
                continue
    return None


_setup_cjk_font()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common.backend  # noqa: F401
import tensorcircuit as tc

from common.chemistry import molecule_report
from common.circuits import prepare_hf, build_lucj
from common.hardware import sample, circuit_resource_summary
from common.sqd import (config_recovery_counts, bitstrings_to_ci_strs,
                        run_sqd_product)


def sqd_energy_at_lambda(rep, lam, *, backend="sim", device="",
                         dry_run=True, n_shots=4000,
                         recover_method="directed", verbose=False):
    """给定分子报告 rep 与 λ，返回该 λ 下的 SQD 能量 E_SQD。

    复用外部已计算好的 rep（molecule_report 结果），只重建 LUCJ 电路并跑
    一遍「采样 -> 配置恢复 -> 子空间对角化」。
    """
    norb, nocc = rep["norb"], rep["nocc"]
    nq = 2 * norb
    h1e, eri, ecore = rep["h1e"], rep["eri"], rep["ecore"]
    hf_bs = "".join("1" if q < 2 * nocc else "0" for q in range(nq))

    # 1) 电路构造：HF 制备 + LUCJ（CCSD 振幅缩放 λ）
    c = tc.Circuit(nq)
    prepare_hf(c, norb, nocc)
    build_lucj(c, norb, nocc, rep["t1"], rep["t2"], eri=rep["eri"],
               ccsd_scale=lam, local=True, doubles=True, doubles_thresh=1e-5)

    if verbose:
        res = circuit_resource_summary(c)
        print(f"  [λ={lam:.3f}] circuit: nq={res['nq']} "
              f"1q={res['n_1q']} 2q={res['n_2q']}")

    # 2) 采样
    counts = sample(c, n_shots, nq, backend=backend, device=device,
                    dry_run=dry_run, task_label=f"sqd_h2_lambda_{lam:.3f}")

    # 3) 配置恢复 + α×β 笛卡尔积子空间 SQD
    rc = config_recovery_counts(counts, nq, nocc, nocc, method=recover_method)
    a_strs, b_strs = bitstrings_to_ci_strs(rc, nq)
    sqd_res = run_sqd_product(h1e, eri, nq, ecore, a_strs, b_strs,
                              include=[hf_bs])
    return sqd_res["E_sqd"]


def scan_lambda(lambdas, *, backend="sim", device="", dry_run=True,
                n_shots=4000, n_reps=1, recover_method="directed",
                out_png="sqd_lambda_scan.png"):
    """扫描 λ 列表，计算各 λ 下 E_SQD，绘制曲线并保存 PNG。

    n_reps>1 时对每个 λ 重复采样-SQD 多次取平均，误差带为 ±1σ。这是应对
    有限采样涨落的正确做法：小 λ 下双激发组态振幅 ∝(λ·t2)²，被采到的概率
    低，单次能量在 HF 与 FCI 间随机跳变；多次平均后得到平滑的「期望 SQD
    能量 vs λ」，刻画「捕获关联组态的概率随 λ 上升」。

    Returns:
        (lambdas, mean_e, std_e, rep): λ、平均能量、标准差、分子报告。
    """
    # 分子报告只算一次（与 λ 无关），所有 λ 共享参考能量
    rep = molecule_report("H2")
    e_hf, e_fci = rep["E_HF"], rep["E_FCI"]
    e_ccsd = rep.get("E_CCSD", None)

    print(f"[scan] H2 参考能量: E_HF={e_hf:.6f}  E_FCI={e_fci:.6f}"
          + (f"  E_CCSD={e_ccsd:.6f}" if e_ccsd is not None else ""))
    print(f"[scan] 扫描 λ ∈ [{lambdas[0]:.3f}, {lambdas[-1]:.3f}]，"
          f"共 {len(lambdas)} 个点，n_shots={n_shots}，n_reps={n_reps}")

    mean_e, std_e = [], []
    for lam in lambdas:
        es = [sqd_energy_at_lambda(rep, lam, backend=backend, device=device,
                                   dry_run=dry_run, n_shots=n_shots,
                                   recover_method=recover_method,
                                   verbose=(r == 0))
              for r in range(n_reps)]
        es = np.asarray(es, dtype=float)
        m, s = float(es.mean()), float(es.std())
        corr = ((m - e_hf) / (e_fci - e_hf)) if (e_fci - e_hf) != 0 else float("nan")
        print(f"  [λ={lam:.3f}] <E_SQD>={m:.6f}±{s:.6f} Ha  平均关联能捕获={corr:.4f}")
        mean_e.append(m)
        std_e.append(s)

    mean_e = np.asarray(mean_e, dtype=float)
    std_e = np.asarray(std_e, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    if n_reps > 1:
        ax.fill_between(lambdas, mean_e - std_e, mean_e + std_e,
                        color="#1f77b4", alpha=0.18, zorder=1,
                        label=r"$\pm 1\sigma$（采样涨落）")
    ax.plot(lambdas, mean_e, "o-", color="#1f77b4", lw=2, ms=6,
            label=(r"$\langle E_{\mathrm{SQD}}\rangle(\lambda)$"
                   if n_reps > 1 else r"$E_{\mathrm{SQD}}(\lambda)$"),
            zorder=3)
    ax.axhline(e_fci, color="#d62728", ls="--", lw=1.6,
               label=f"E_FCI = {e_fci:.5f} Ha", zorder=2)
    ax.axhline(e_hf, color="#7f7f7f", ls=":", lw=1.6,
               label=f"E_HF = {e_hf:.5f} Ha", zorder=2)
    if e_ccsd is not None:
        ax.axhline(e_ccsd, color="#2ca02c", ls="-.", lw=1.2,
                   label=f"E_CCSD = {e_ccsd:.5f} Ha", zorder=2)

    # 标注最优 λ（平均能量最低）
    i_best = int(np.argmin(mean_e))
    ax.scatter([lambdas[i_best]], [mean_e[i_best]], s=140,
               facecolors="none", edgecolors="#ff7f0e", lw=2.2, zorder=4,
               label=f"最优 λ={lambdas[i_best]:.3f} → {mean_e[i_best]:.6f} Ha")

    ax.set_xlabel(r"LUCJ 振幅缩放 $\lambda$ (ccsd_scale)", fontsize=12)
    ax.set_ylabel("能量 (Ha)", fontsize=12)
    ax.set_title(r"H$_2$ SQD 能量随 LUCJ $\lambda$ 的变化", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_png)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[scan] 曲线已保存: {out_path}")
    print(f"[scan] 最优 λ={lambdas[i_best]:.3f}，<E_SQD>={mean_e[i_best]:.6f} Ha"
          f"（与 E_FCI 差 {mean_e[i_best]-e_fci:+.2e} Ha）")
    return lambdas, mean_e, std_e, rep


if __name__ == "__main__":
    # 本地模拟器扫描 λ（无需凭据）。在小 λ 过渡区（0~0.25）加密采样，
    # 以展示 SQD 从 HF 爬升到 FCI 的过程；大 λ 稀疏采样确认平台稳定。
    # n_reps>1：每个 λ 重复多次取平均+误差带，抹平有限采样涨落。
    lambdas = np.unique(np.concatenate([
        np.linspace(0.0, 0.25, 11),   # 过渡区加密
        np.linspace(0.3, 1.5, 7),     # 平台区
    ]))
    scan_lambda(lambdas, backend="sim", n_shots=4000, n_reps=20,
                out_png="sqd_lambda_scan.png")
