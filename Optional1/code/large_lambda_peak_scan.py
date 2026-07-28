#!/usr/bin/env python
"""
Large-range lambda scan for theoretically predicted SQD spikes.

Two complementary experiments:

1) H2, lambda in [0, 1000]: the state has only HF and one double excitation,
   so the finite-shot expected SQD error is available analytically.

2) H2O, n_reps=3, lambda in [0.1, 100]: first scan a cheap theoretical
   missing-key-weight proxy, automatically locate candidate probability nodes,
   then validate each candidate with no-K SQD over several nearby lambda values,
   shot counts, and random seeds.

All paths are relative to this file's directory.  Edit CONTROL to tune the scan.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from pyscf import fci
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from lucj_sqd_framework import SQDConfig, LUCJSQDExperiment


CONTROL = {
    "h2": {
        "lambda_min": 0.0,
        "lambda_max": 1000.0,
        "lambda_step": 0.1,
        "shots": [500, 5000],
        "peak_distance": 500,       # grid points, i.e. ~50 lambda
        "n_report": 12,
    },
    "h2o": {
        "n_reps": 3,
        "lambda_min": 0.1,
        "lambda_max": 100.0,
        "lambda_step": 0.05,
        "key_count": 10,
        "proxy_shots": [100_000, 1_000_000, 10_000_000],
        "candidate_shots": [100_000, 1_000_000, 10_000_000],
        "candidate_seeds": 20,
        "candidate_offsets": [-0.30, -0.15, -0.05, 0.0, 0.05, 0.15, 0.30],
        "candidate_peak_distance": 20,
        "candidate_count": 2,
    },
    "seed": 20260728,
    "data_file": "large_lambda_peaks.json",
    "figure_file": "large_lambda_peaks.png",
}


def hamiltonian_and_fci(exp):
    dim = exp._linop.shape[0]
    hmat = np.asarray(exp._linop @ np.eye(dim), dtype=complex)
    hmat = (hmat + hmat.conj().T) / 2
    e_fci, civec = fci.direct_spin1.FCI().kernel(
        exp.h1e, exp.eri, exp.norb, exp.nelec, ecore=exp.ecore)
    vec = np.asarray(civec).reshape(-1).astype(complex)
    vec /= np.linalg.norm(vec)
    return hmat, float(e_fci), np.abs(vec) ** 2


def subspace_energy(hmat, addresses):
    idx = np.unique(np.asarray(addresses, dtype=int))
    return float(np.linalg.eigvalsh(hmat[np.ix_(idx, idx)])[0].real)


# -----------------------------------------------------------------------------
# H2: exact finite-shot expectation
# -----------------------------------------------------------------------------
def run_h2():
    cfg = CONTROL["h2"]
    exp = LUCJSQDExperiment(SQDConfig(
        molecule="H2", n_reps=1, local_truncation=True,
        noise_model="none", shots=10, n_seeds=1))
    hmat, e_fci, fci_prob = hamiltonian_and_fci(exp)
    key = np.argsort(fci_prob)[::-1]
    hf, double = int(key[0]), int(key[1])
    delta_hf = float((hmat[hf, hf].real - e_fci) * 1000)
    delta_double = float((hmat[double, double].real - e_fci) * 1000)

    lambdas = np.arange(cfg["lambda_min"], cfg["lambda_max"] + cfg["lambda_step"] / 2,
                        cfg["lambda_step"])
    p_hf = np.empty(len(lambdas)); p_double = np.empty(len(lambdas))
    errors = {str(s): np.empty(len(lambdas)) for s in cfg["shots"]}
    for i, lam in enumerate(lambdas):
        prob = np.abs(exp._lucj_state(float(lam))) ** 2
        p_hf[i], p_double[i] = prob[hf], prob[double]
        for shots in cfg["shots"]:
            errors[str(shots)][i] = (
                p_hf[i] ** shots * delta_hf +
                p_double[i] ** shots * delta_double)

    # The peaks are deterministic probability nodes, not Monte-Carlo accidents.
    reference = errors[str(cfg["shots"][0])]
    peak_idx, _ = find_peaks(reference, distance=cfg["peak_distance"], prominence=1.0)
    peak_idx = peak_idx[np.argsort(reference[peak_idx])[::-1][:cfg["n_report"]]]
    peaks = []
    for i in sorted(peak_idx, key=lambda j:lambdas[j]):
        row = {"lambda": float(lambdas[i]), "p_hf": float(p_hf[i]),
               "p_double": float(p_double[i])}
        for shots in cfg["shots"]:
            row[f"expected_err_S{shots}_mha"] = float(errors[str(shots)][i])
        peaks.append(row)

    return {
        "lambda": lambdas.tolist(), "p_hf": p_hf.tolist(),
        "p_double": p_double.tolist(),
        "expected_error_mha": {k:v.tolist() for k,v in errors.items()},
        "delta_hf_mha": delta_hf, "delta_double_mha": delta_double,
        "peaks": peaks,
    }


# -----------------------------------------------------------------------------
# H2O: broad proxy scan + finite-shot SQD validation at candidate nodes
# -----------------------------------------------------------------------------
def run_h2o():
    cfg = CONTROL["h2o"]
    exp = LUCJSQDExperiment(SQDConfig(
        molecule="H2O", n_reps=cfg["n_reps"], local_truncation=True,
        noise_model="none", shots=10, n_seeds=1))
    hmat, e_fci, fci_prob = hamiltonian_and_fci(exp)
    key = np.argsort(fci_prob)[::-1][:cfg["key_count"]]
    key_weights = fci_prob[key]

    lambdas = np.arange(cfg["lambda_min"], cfg["lambda_max"] + cfg["lambda_step"] / 2,
                        cfg["lambda_step"])
    key_prob = np.empty((len(lambdas), len(key)))
    support_floor = np.empty(len(lambdas))
    for i, lam in enumerate(lambdas):
        prob = np.abs(exp._lucj_state(float(lam))) ** 2
        key_prob[i] = prob[key]
        practical = np.flatnonzero(prob > 1e-12)
        support_floor[i] = (subspace_energy(hmat, practical) - e_fci) * 1000

    # Missing-key FCI-weight proxy:
    # sum_i w_i * P(key i is absent) = sum_i w_i*(1-p_i)^S.
    proxy = {}
    for shots in cfg["proxy_shots"]:
        proxy[str(shots)] = (
            np.power(1 - key_prob, shots) * key_weights[None, :]).sum(axis=1)

    # Locate candidates using the largest-shot proxy. These are the narrowest,
    # most unambiguous nodes; lower-shot curves contain the same nodes more broadly.
    target_s = cfg["proxy_shots"][-1]
    y = proxy[str(target_s)]
    peak_idx, _ = find_peaks(
        y, distance=cfg["candidate_peak_distance"],
        prominence=max(y) * 1e-4)
    peak_idx = peak_idx[np.argsort(y[peak_idx])[::-1][:cfg["candidate_count"]]]

    # Refine each coarse candidate by minimizing the particular key probability
    # responsible for the proxy peak. At a genuine unitary node this probability
    # should approach zero quadratically.
    candidates = []
    for i in peak_idx[np.argsort(lambdas[peak_idx])]:
        coarse = float(lambdas[i])
        rank = int(np.argmin(key_prob[i]))
        address = int(key[rank])
        probability = lambda x: float(abs(exp._lucj_state(float(x))[address]) ** 2)
        refined = minimize_scalar(
            probability,
            bounds=(coarse - 2 * cfg["lambda_step"],
                    coarse + 2 * cfg["lambda_step"]),
            method="bounded", options={"xatol": 1e-13})
        center = float(refined.x)
        p0 = float(refined.fun)
        h = 1e-3
        curvature = float((probability(center + h) + probability(center - h) - 2 * p0)
                          / (2 * h * h))
        prob_center = np.abs(exp._lucj_state(center)) ** 2
        key_center = prob_center[key]
        proxy_center = {
            str(shots): float((np.power(1 - key_center, shots) * key_weights).sum())
            for shots in cfg["proxy_shots"]}
        widths = {}
        for shots in cfg["candidate_shots"]:
            target = 1.0 / shots
            half = np.sqrt(max(0.0, target - p0) / curvature) if curvature > 0 else np.nan
            widths[str(shots)] = {
                "half_width_Sp_lt_1": float(half),
                "full_width_Sp_lt_1": float(2 * half)}
        practical = np.flatnonzero(prob_center > 1e-12)
        candidates.append({
            "coarse_lambda": coarse, "lambda": center,
            "min_key_probability": p0, "min_key_rank": rank,
            "quadratic_curvature_a": curvature,
            "proxy": proxy_center, "dangerous_width": widths,
            "support_floor_err_mha": float(
                (subspace_energy(hmat, practical) - e_fci) * 1000),
        })

    centers = [r["lambda"] for r in candidates]
    validations = []
    for center_id, center in enumerate(centers):
        for offset in cfg["candidate_offsets"]:
            lam = float(center + offset)
            prob = np.abs(exp._lucj_state(lam)) ** 2
            key_p = prob[key]
            for shots in cfg["candidate_shots"]:
                metrics = []
                for seed in range(cfg["candidate_seeds"]):
                    rng = np.random.default_rng(
                        CONTROL["seed"] + center_id * 1_000_000 +
                        int(round((offset + 1) * 1000)) + seed)
                    counts = rng.multinomial(shots, prob)
                    idx = np.flatnonzero(counts)
                    metrics.append((
                        (subspace_energy(hmat, idx) - e_fci) * 1000,
                        len(set(idx).intersection(set(key))) / len(key),
                        fci_prob[idx].sum(), len(idx),
                    ))
                arr = np.asarray(metrics, dtype=float)
                validations.append({
                    "center": center, "offset": offset, "lambda": lam,
                    "shots": shots,
                    "min_key_probability": float(key_p.min()),
                    "min_key_rank": int(np.argmin(key_p)),
                    "error_mean_mha": float(arr[:,0].mean()),
                    "error_std_mha": float(arr[:,0].std(ddof=1)),
                    "top10_recall_mean": float(arr[:,1].mean()),
                    "fci_mass_mean": float(arr[:,2].mean()),
                    "subspace_dim_mean": float(arr[:,3].mean()),
                })

    return {
        "lambda": lambdas.tolist(),
        "key_probability": key_prob.tolist(),
        "key_fci_weight": key_weights.tolist(),
        "proxy_missing_fci_weight": {k:v.tolist() for k,v in proxy.items()},
        "support_floor_err_mha": support_floor.tolist(),
        "candidate_centers": centers,
        "candidates": candidates,
        "validations": validations,
    }


def make_figure(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    h2 = data["h2"]
    x = np.asarray(h2["lambda"])
    ax = axes[0,0]
    ax.plot(x, h2["p_hf"], label="p(HF)")
    ax.plot(x, h2["p_double"], label="p(double)")
    ax.set_xlabel("lambda"); ax.set_ylabel("probability")
    ax.set_title("H2: periodic probability exchange")
    ax.grid(alpha=.3); ax.legend()

    ax = axes[0,1]
    for shots, err in h2["expected_error_mha"].items():
        ax.plot(x, np.maximum(np.asarray(err), 1e-12), lw=1, label=f"S={shots}")
    ax.set_yscale("log"); ax.set_xlabel("lambda")
    ax.set_ylabel("exact expected SQD error (mHa)")
    ax.set_title("H2: narrow periodic spikes")
    ax.grid(alpha=.3); ax.legend()

    h2o = data["h2o"]
    x2 = np.asarray(h2o["lambda"])
    ax = axes[1,0]
    for shots, proxy in h2o["proxy_missing_fci_weight"].items():
        ax.plot(x2, np.maximum(np.asarray(proxy), 1e-15), label=f"S={shots}")
    for center in h2o["candidate_centers"]:
        ax.axvline(center, color="red", ls=":", lw=1)
    ax.set_yscale("log"); ax.set_xlabel("lambda")
    ax.set_ylabel("missing-key FCI-weight proxy")
    ax.set_title("H2O n_reps=3: broad scan and detected nodes")
    ax.grid(alpha=.3); ax.legend()

    ax = axes[1,1]
    colors = {100_000:"#c0392b", 1_000_000:"#e67e22", 10_000_000:"#2980b9"}
    for center in h2o["candidate_centers"]:
        for shots in CONTROL["h2o"]["candidate_shots"]:
            rows = [r for r in h2o["validations"]
                    if r["center"] == center and r["shots"] == shots]
            ax.errorbar([r["lambda"] for r in rows], [r["error_mean_mha"] for r in rows],
                        yerr=[r["error_std_mha"] for r in rows], marker="o", capsize=2,
                        color=colors[shots], alpha=.8,
                        label=f"node {center:.2f}, S={shots:g}")
    ax.set_xlabel("lambda"); ax.set_ylabel("no-K SQD error (mHa)")
    ax.set_title("H2O: finite-shot validation around predicted spikes")
    ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(HERE / CONTROL["figure_file"], dpi=160)


def main():
    data = {"control": CONTROL, "h2": run_h2(), "h2o": run_h2o()}
    with open(HERE / CONTROL["data_file"], "w") as f:
        json.dump(data, f, indent=2)
    make_figure(data)
    print("H2 detected peaks:")
    for r in data["h2"]["peaks"]:
        print(r)
    print("H2O candidate nodes:")
    for r in data["h2o"]["candidates"]:
        print(r)
    print(f"Saved {HERE / CONTROL['data_file']}")
    print(f"Saved {HERE / CONTROL['figure_file']}")


if __name__ == "__main__":
    main()
