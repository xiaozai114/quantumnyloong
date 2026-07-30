"""在 sqd_qpu_example.py 基础上加 matplotlib 画图：研究 lam 与 n_shots 的影响。

保持源示例的 SQD 流程不变（电路 -> 采样 -> 配置恢复 -> 乘积子空间对角化），
仅：(1) 把单次调用封装成 sqd_energy()；(2) 加两组扫描并画图。lam 限定在 [0,1]。
每个数据点跑 50 次（不同随机种子）取平均，阴影为这 50 次的最小~最大范围。
左图固定 n_shots=8000 扫 lam，右图固定 lam=0.5 扫 n_shots。
对小分子 LiH 与稍大分子 H2O 各跑一次。
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, "/Users/jinxule/quantumnyloong-main/sqd")

import common.backend  # noqa: F401
import tensorcircuit as tc

from common.chemistry import molecule_report
from common.circuits import prepare_hf, build_lucj, statevector
from common.sqd import config_recovery_counts, bitstrings_to_ci_strs, run_sqd_product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
# 字体加大、加粗、文字颜色加深（近纯黑）
_DARK = "#111111"
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
    "text.color": _DARK, "axes.labelcolor": _DARK,
    "xtick.color": _DARK, "ytick.color": _DARK, "axes.edgecolor": _DARK,
})

OUT = os.path.dirname(os.path.abspath(__file__))


def sqd_energy(rep, lam, n_shots, seed=0):
    """源示例的一次 SQD：返回 E_SQD（Ha）。"""
    norb, nocc = rep["norb"], rep["nocc"]
    nq = 2 * norb
    h1e, eri, ecore = rep["h1e"], rep["eri"], rep["ecore"]
    hf_bs = "".join("1" if q < 2 * nocc else "0" for q in range(nq))

    c = tc.Circuit(nq)
    prepare_hf(c, norb, nocc)
    build_lucj(c, norb, nocc, rep["t1"], rep["t2"], eri=eri,
               ccsd_scale=lam, local=True, doubles=True, doubles_thresh=1e-5)

    psi = statevector(c)
    p = np.abs(psi) ** 2; p /= p.sum()
    idx = np.random.default_rng(seed).choice(len(p), size=n_shots, p=p)
    ks, cs = np.unique(idx, return_counts=True)
    counts = {int(k): int(v) for k, v in zip(ks, cs)}

    rc = config_recovery_counts(counts, nq, nocc, nocc, method="directed")
    a, b = bitstrings_to_ci_strs(rc, nq, n_alpha=nocc, n_beta=nocc)
    return run_sqd_product(h1e, eri, nq, ecore, a, b, include=[hf_bs])["E_sqd"]


def sqd_stats(rep, lam, n_shots, n_rep=50):
    """跑 n_rep 次（不同 seed），返回 ΔE (mHa) 的 (均值, 最小值, 最大值)。"""
    e_fci = rep["E_FCI"]
    dE = [(sqd_energy(rep, lam, n_shots, seed=s) - e_fci) * 1000
          for s in range(n_rep)]
    return float(np.mean(dE)), float(np.min(dE)), float(np.max(dE))


def study(name, n_rep=50):
    rep = molecule_report(name)
    print(f"{name}: nq={2*rep['norb']} E_FCI={rep['E_FCI']:.6f}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # 左：固定 n_shots=8000，扫 lam ∈ [0,1]（步长 0.05）
    lam_grid = np.linspace(0.0, 1.0, 21)
    st = np.array([sqd_stats(rep, lam, 8000, n_rep) for lam in lam_grid])
    m, lo, hi = st[:, 0], st[:, 1], st[:, 2]
    axL.plot(lam_grid, m, "o-", color="C0", label="均值 (50 次)")
    axL.fill_between(lam_grid, lo, hi, color="C0", alpha=.25,
                     label="最小~最大范围")
    axL.axhline(0, color="crimson", ls="--", lw=1)
    axL.set_xlabel("λ (CCSD 缩放因子)"); axL.set_ylabel("ΔE = E$_{SQD}$−E$_{FCI}$ (mHa)")
    axL.set_title(f"{name}：n_shots=8000，λ 的影响"); axL.legend(); axL.grid(alpha=.3)

    # 右：固定 lam=0.5，扫 n_shots（对数网格，加密取点）
    shots_grid = [50, 100, 200, 400, 800, 1500, 3000, 6000, 12000, 24000, 48000, 96000]
    st = np.array([sqd_stats(rep, 0.5, ns, n_rep) for ns in shots_grid])
    m, lo, hi = st[:, 0], st[:, 1], st[:, 2]
    axR.plot(shots_grid, m, "s-", color="C1", label="均值 (50 次)")
    axR.fill_between(shots_grid, lo, hi, color="C1", alpha=.25,
                     label="最小~最大范围")
    axR.axhline(0, color="crimson", ls="--", lw=1)
    axR.set_xscale("log")
    axR.set_xlabel("n_shots (对数轴)"); axR.set_ylabel("ΔE = E$_{SQD}$−E$_{FCI}$ (mHa)")
    axR.set_title(f"{name}：λ=0.5，n_shots 的影响"); axR.legend(); axR.grid(alpha=.3, which="both")

    fig.tight_layout()
    path = os.path.join(OUT, f"sqd_study_{name}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  -> {path}")

    return dict(
        name=name, nq=2 * rep["norb"], E_HF=rep["E_HF"], E_FCI=rep["E_FCI"],
        corr_mHa=(rep["E_FCI"] - rep["E_HF"]) * 1000,
        lam_grid=lam_grid.tolist(),
        lam_scan=np.array([sqd_stats(rep, lam, 8000, n_rep) for lam in lam_grid]).tolist(),
        shots_grid=shots_grid,
        shots_scan=np.array([sqd_stats(rep, 0.5, ns, n_rep) for ns in shots_grid]).tolist(),
    )


if __name__ == "__main__":
    results = {name: study(name) for name in ["LiH", "H2O"]}
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("results.json saved")
