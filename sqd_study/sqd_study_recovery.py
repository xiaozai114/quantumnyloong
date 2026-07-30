"""SQD 配置恢复策略对比：directed / max_dev / none 在 bitflip 噪声下的表现。

基于 sqd_qpu_example.py 的 SQD 流程：电路 -> 采样 -> (注入 bitflip 噪声) ->
配置恢复 -> α×β 乘积子空间对角化。

对比三种配置恢复方法（common.sqd 的 method）：
  - directed : 定向翻转，保证把样本拉回目标 (nα,nβ) 扇区（必收敛）。
  - max_dev  : 最大误差贪心翻转（与 qiskit-addon-sqd 一致），高噪声下可能翻错。
  - none     : 不做修复，只保留本就落在正确扇区的样本（其余丢弃）。

每张图左=固定 n_shots=8000 扫 λ∈[0,1]，右=固定 λ=0.5 扫 n_shots。
每个数据点跑 50 次（不同随机种子）取平均，只画均值曲线（不画阴影）。
噪声固定 bitflip 概率（每比特独立），对 LiH 与 H2O 各出一张图。
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
# 字体加大、文字颜色加深（近纯黑）
_DARK = "#111111"
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "text.color": _DARK,
    "axes.labelcolor": _DARK,
    "xtick.color": _DARK,
    "ytick.color": _DARK,
    "axes.edgecolor": _DARK,
})

OUT = os.path.dirname(os.path.abspath(__file__))

METHODS = ["directed", "max_dev", "none"]
COLORS = {"directed": "C0", "max_dev": "C1", "none": "C2"}
MARKERS = {"directed": "o", "max_dev": "s", "none": "^"}
BITFLIP = 0.06  # 每比特独立翻转概率（模拟读出噪声）


def sqd_energy(rep, lam, n_shots, method, seed=0, bitflip_noise=BITFLIP):
    """一次 SQD：制备 -> 采样 -> bitflip 噪声 -> 配置恢复(method) -> 对角化。

    返回 E_SQD（Ha）。噪声 rng 由 seed 派生（每次重复噪声不同，保证统计意义）。
    """
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
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(p), size=n_shots, p=p).astype(np.int64)

    # bitflip 噪声：每个采样比特以 bitflip_noise 概率翻转（向量化）
    if bitflip_noise:
        flips = rng.random((n_shots, nq)) < bitflip_noise
        mask = (flips * (1 << np.arange(nq))).sum(axis=1).astype(np.int64)
        idx ^= mask

    ks, cs = np.unique(idx, return_counts=True)
    counts = {int(k): int(v) for k, v in zip(ks, cs)}

    rc = config_recovery_counts(counts, nq, nocc, nocc, method=method)
    a, b = bitstrings_to_ci_strs(rc, nq, n_alpha=nocc, n_beta=nocc)
    if not a or not b:  # none 在高噪/小样本下可能全被丢弃 -> 退化为 HF
        a, b = [], []
    return run_sqd_product(h1e, eri, nq, ecore, a, b, include=[hf_bs])["E_sqd"]


def sqd_stats(rep, lam, n_shots, method, noise, n_rep=50):
    """跑 n_rep 次，返回 ΔE (mHa) 的 (均值, 最小值, 最大值)。"""
    e_fci = rep["E_FCI"]
    dE = [(sqd_energy(rep, lam, n_shots, method, seed=s, bitflip_noise=noise) - e_fci) * 1000
          for s in range(n_rep)]
    return float(np.mean(dE)), float(np.min(dE)), float(np.max(dE))


def study(name, noise, n_rep=50):
    rep = molecule_report(name)
    print(f"{name}: nq={2*rep['norb']} E_FCI={rep['E_FCI']:.6f} "
          f"(bitflip={noise})")

    lam_grid = np.linspace(0.0, 1.0, 11)
    shots_grid = [50, 100, 200, 400, 800, 1500, 3000, 6000, 12000, 24000]

    lam_data, shots_data = {}, {}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4))

    for method in METHODS:
        # 左：固定 n_shots=8000，扫 lam
        st = np.array([sqd_stats(rep, lam, 8000, method, noise, n_rep) for lam in lam_grid])
        lam_data[method] = st.tolist()
        m = st[:, 0]
        axL.plot(lam_grid, m, MARKERS[method] + "-", color=COLORS[method],
                 label=f"{method}")
        print(f"  [lam扫描] {method}: ΔE(λ=0)={m[0]:.2f}  ΔE(λ=1)={m[-1]:.2f} mHa")

        # 右：固定 lam=0.5，扫 n_shots
        st = np.array([sqd_stats(rep, 0.5, ns, method, noise, n_rep) for ns in shots_grid])
        shots_data[method] = st.tolist()
        m = st[:, 0]
        axR.plot(shots_grid, m, MARKERS[method] + "-", color=COLORS[method],
                 label=f"{method}")
        print(f"  [shots扫描] {method}: ΔE(50)={m[0]:.2f}  ΔE(24000)={m[-1]:.2f} mHa")

    axL.axhline(0, color="crimson", ls="--", lw=1)
    axL.set_xlabel("λ (CCSD 缩放因子)")
    axL.set_ylabel("ΔE = E$_{SQD}$−E$_{FCI}$ (mHa)")
    axL.set_title(f"{name}：n_shots=8000, bitflip={noise}\n配置恢复方法对比（扫 λ）")
    axL.legend(); axL.grid(alpha=.3)

    axR.axhline(0, color="crimson", ls="--", lw=1)
    axR.set_xscale("log")
    axR.set_xlabel("n_shots (对数轴)")
    axR.set_ylabel("ΔE = E$_{SQD}$−E$_{FCI}$ (mHa)")
    axR.set_title(f"{name}：λ=0.5, bitflip={noise}\n配置恢复方法对比（扫 n_shots）")
    axR.legend(); axR.grid(alpha=.3, which="both")

    fig.tight_layout()
    path = os.path.join(OUT, f"sqd_recovery_{name}_noise{noise}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  -> {path}")

    return dict(name=name, nq=2 * rep["norb"], E_HF=rep["E_HF"],
                E_FCI=rep["E_FCI"], corr_mHa=(rep["E_FCI"] - rep["E_HF"]) * 1000,
                bitflip=noise, lam_grid=lam_grid.tolist(),
                shots_grid=shots_grid, lam_data=lam_data, shots_data=shots_data)


if __name__ == "__main__":
    results = {}
    for noise in [0.01, 0.05]:
        for name in ["H2O"]:
            results[f"{name}_noise{noise}"] = study(name, noise)
    with open(os.path.join(OUT, "recovery_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("recovery_results.json saved")
