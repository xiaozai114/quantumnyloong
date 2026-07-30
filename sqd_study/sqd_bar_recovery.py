"""SQD 配置恢复策略柱状图对比：两种噪声 × 三种典型 (λ, n_shots) 组合。

基于 sqd_qpu_example.py 的 SQD 流程（复用 sqd_study_recovery.sqd_energy）。

对每种分子（LiH, H2O）画两张柱状图，分别对应两种 bitflip 噪声：
  - 小噪声 0.01
  - 大噪声 0.05
每张图 x 轴为 3 个典型 (λ, n_shots) 组合，每组 3 根柱 = 三种配置恢复方法
（directed / max_dev / none）的 ΔE = E_SQD − E_FCI（mHa），50 次重复取均值。
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, "/Users/jinxule/quantumnyloong-main/sqd")

import common.backend  # noqa: F401
from common.chemistry import molecule_report

# 复用已写好的单次 SQD（含 bitflip 噪声、method 选择）
from sqd_study_recovery import sqd_energy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.dirname(os.path.abspath(__file__))

METHODS = ["directed", "max_dev", "none"]
COLORS = {"directed": "C0", "max_dev": "C1", "none": "C2"}
NOISES = [0.01, 0.05]

# 3 个典型 (λ, n_shots) 组合：少采样 / 中等 / 充分
COMBOS = [(0.5, 200), (0.5, 800), (1.0, 4000)]
N_REP = 50


def mean_dE(rep, lam, n_shots, method, noise, n_rep=N_REP):
    """50 次重复的 ΔE (mHa) 均值。"""
    e_fci = rep["E_FCI"]
    vals = [(sqd_energy(rep, lam, n_shots, method, seed=s, bitflip_noise=noise)
             - e_fci) * 1000 for s in range(n_rep)]
    return float(np.mean(vals))


def plot_molecule_noise(name, rep, noise, data):
    """一张柱状图：3 组合 × 3 方法。data[combo_idx][method] = ΔE 均值。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(COMBOS))
    width = 0.25
    for i, method in enumerate(METHODS):
        vals = [data[c][method] for c in range(len(COMBOS))]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      color=COLORS[method], label=method)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    labels = [f"λ={lam}\nshots={ns}" for lam, ns in COMBOS]
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("ΔE = E$_{SQD}$ − E$_{FCI}$ (mHa)")
    ax.set_title(f"{name}（nq={2*rep['norb']}）配置恢复方法对比  bitflip={noise}")
    ax.legend(title="recovery")
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    path = os.path.join(OUT, f"sqd_bar_{name}_noise{noise}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  -> {path}")
    return path


def main():
    results = {}
    for name in ["LiH", "H2O"]:
        rep = molecule_report(name)
        print(f"{name}: nq={2*rep['norb']} E_FCI={rep['E_FCI']:.6f}")
        results[name] = {}
        for noise in NOISES:
            data = []
            for lam, ns in COMBOS:
                row = {m: mean_dE(rep, lam, ns, m, noise) for m in METHODS}
                data.append(row)
                print(f"  noise={noise} λ={lam} shots={ns}: " +
                      "  ".join(f"{m}={row[m]:.2f}" for m in METHODS))
            plot_molecule_noise(name, rep, noise, data)
            results[name][str(noise)] = data

    with open(os.path.join(OUT, "bar_results.json"), "w") as f:
        json.dump(dict(combos=COMBOS, methods=METHODS, results=results),
                  f, indent=2)
    print("bar_results.json saved")


if __name__ == "__main__":
    main()
