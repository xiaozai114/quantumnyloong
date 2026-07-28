#!/usr/bin/env python
"""
Study the key-configuration coverage-saturated regime (no fixed K)
=================================================================

Definition used here
--------------------
A finite-shot point is called "key-coverage sufficient" when all sampled legal
configurations are retained and

  1. mean recall of the 10 largest-FCI-weight determinants >= 0.95, and
  2. the sampled determinants carry >= 99% of the exact FCI probability mass.

The second condition alone is not enough: a state can cover >99% FCI mass while
missing several individually important determinants because the HF determinant
carries most of the weight.

We compare one- and three-repetition local LUCJ.  This is crucial because a
one-repetition ansatz may assign exactly zero probability to important FCI
configurations; no number of shots can recover a zero-probability determinant.

Unlike controlled_optional1_experiments.py, this script does NOT impose a maximum
subspace dimension K. Every unique sampled determinant is diagonalized.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from pyscf import fci

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from lucj_sqd_framework import SQDConfig, LUCJSQDExperiment


CONTROL = {
    "system": {"molecule": "H2O", "bond": None},
    "n_reps": [1, 3],
    "lambdas": [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    "shots": [1000, 10_000, 100_000, 1_000_000, 10_000_000],
    "seeds": 20,
    "top_key_count": 10,
    "key_recall_threshold": 0.95,
    "fci_mass_threshold": 0.99,
    # Configurations below this probability require >~1e12 samples and are
    # classified as practically inaccessible for the support-limit diagnostic.
    "practical_support_tol": 1e-12,
    "seed": 20260728,
    "data_file": "coverage_saturated.json",
    "figure_file": "coverage_saturated.png",
}


def make_problem(n_reps: int):
    cfg = SQDConfig(
        molecule=CONTROL["system"]["molecule"],
        bond=CONTROL["system"]["bond"],
        n_reps=n_reps, local_truncation=True,
        noise_model="none", shots=10, n_seeds=1)
    exp = LUCJSQDExperiment(cfg)
    dim = exp._linop.shape[0]
    hmat = np.asarray(exp._linop @ np.eye(dim), dtype=complex)
    hmat = (hmat + hmat.conj().T) / 2

    e_fci, civec = fci.direct_spin1.FCI().kernel(
        exp.h1e, exp.eri, exp.norb, exp.nelec, ecore=exp.ecore)
    fci_vec = np.asarray(civec).reshape(-1).astype(complex)
    fci_vec /= np.linalg.norm(fci_vec)
    fci_prob = np.abs(fci_vec) ** 2
    key = np.argsort(fci_prob)[::-1][:CONTROL["top_key_count"]]
    return exp, hmat, fci_prob, key


def subspace_energy(hmat, idx):
    idx = np.unique(np.asarray(idx, dtype=int))
    return float(np.linalg.eigvalsh(hmat[np.ix_(idx, idx)])[0].real)


def exact_diagnostics(exp, hmat, fci_prob, key, lam):
    psi = np.asarray(exp._lucj_state(lam), dtype=complex)
    p = np.abs(psi) ** 2
    practical = np.flatnonzero(p > CONTROL["practical_support_tol"])
    mathematical = np.flatnonzero(p > 1e-15)

    # Direct variational energy keeps the LUCJ amplitudes.
    e_var = float(np.real(np.vdot(psi, hmat @ psi)))
    # Support-floor SQD discards amplitudes and diagonalizes over all practically
    # accessible configurations. This is the high-shot limit for this ansatz.
    e_support = subspace_energy(hmat, practical)

    key_probs = p[key]
    seen_at_max = 1 - np.power(1 - key_probs, max(CONTROL["shots"]))
    return {
        "lambda": lam,
        "direct_var_err_mha": (e_var - exp.e_fci) * 1000,
        "practical_support_dim": int(len(practical)),
        "mathematical_support_dim": int(len(mathematical)),
        "support_fci_mass": float(fci_prob[practical].sum()),
        "support_top10_recall": float(len(set(practical).intersection(set(key))) / len(key)),
        "support_floor_err_mha": (e_support - exp.e_fci) * 1000,
        "min_top10_sampling_prob": float(key_probs.min()),
        "expected_top10_recall_at_max_shots": float(seen_at_max.mean()),
    }, psi


def sampled_diagnostics(exp, hmat, fci_prob, key, p, lam, shots, n_reps):
    metrics = []
    for seed in range(CONTROL["seeds"]):
        rng = np.random.default_rng(
            CONTROL["seed"] + n_reps * 10_000_000 + int(100 * lam) * 10_000 + seed)
        counts = rng.multinomial(shots, p)
        idx = np.flatnonzero(counts)
        key_recall = len(set(idx).intersection(set(key))) / len(key)
        fci_mass = float(fci_prob[idx].sum())
        metrics.append({
            "energy_err_mha": (subspace_energy(hmat, idx) - exp.e_fci) * 1000,
            "subspace_dim": len(idx),
            "fci_mass": fci_mass,
            "top10_recall": key_recall,
        })

    row = {"lambda": lam, "shots": shots}
    for field in metrics[0]:
        values = np.array([m[field] for m in metrics], dtype=float)
        row[f"{field}_mean"] = float(values.mean())
        row[f"{field}_std"] = float(values.std(ddof=1))
    row["coverage_sufficient"] = bool(
        row["top10_recall_mean"] >= CONTROL["key_recall_threshold"] and
        row["fci_mass_mean"] >= CONTROL["fci_mass_threshold"])
    return row


def run():
    output = {"control": CONTROL, "repetitions": {}}
    for reps in CONTROL["n_reps"]:
        print(f"Running n_reps={reps}")
        exp, hmat, fci_prob, key = make_problem(reps)
        exact_rows, sample_rows = [], []
        for lam in CONTROL["lambdas"]:
            exact, psi = exact_diagnostics(exp, hmat, fci_prob, key, lam)
            exact_rows.append(exact)
            p = np.abs(psi) ** 2
            for shots in CONTROL["shots"]:
                sample_rows.append(sampled_diagnostics(
                    exp, hmat, fci_prob, key, p, lam, shots, reps))
        output["repetitions"][str(reps)] = {
            "dimension": len(fci_prob),
            "e_hf": exp.e_hf, "e_fci": exp.e_fci,
            "top10_fci_mass": float(fci_prob[key].sum()),
            "exact": exact_rows, "samples": sample_rows,
        }

    with open(HERE / CONTROL["data_file"], "w") as f:
        json.dump(output, f, indent=2)
    make_figure(output)
    print(f"Saved {HERE / CONTROL['data_file']}")
    print(f"Saved {HERE / CONTROL['figure_file']}")
    return output


def make_figure(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {1: "#c0392b", 3: "#2980b9"}

    # Exact direct variational energy vs support-floor SQD
    ax = axes[0, 0]
    for reps in CONTROL["n_reps"]:
        rows = data["repetitions"][str(reps)]["exact"]
        l = [r["lambda"] for r in rows]
        ax.plot(l, [r["direct_var_err_mha"] for r in rows], marker="o",
                color=colors[reps], label=f"reps={reps}: direct state")
        ax.plot(l, [r["support_floor_err_mha"] for r in rows], marker=".", ls="--",
                color=colors[reps], label=f"reps={reps}: all accessible support")
    ax.set_xlabel("lambda"); ax.set_ylabel("error vs FCI (mHa)")
    ax.set_title("Exact state quality vs high-shot SQD floor")
    ax.grid(alpha=.3); ax.legend(fontsize=8)

    # Structural support limit
    ax = axes[0, 1]
    for reps in CONTROL["n_reps"]:
        rows = data["repetitions"][str(reps)]["exact"]
        ax.plot([r["lambda"] for r in rows], [r["support_top10_recall"] for r in rows],
                marker="o", color=colors[reps], label=f"reps={reps}: top10 recall")
        ax.plot([r["lambda"] for r in rows], [r["support_fci_mass"] for r in rows],
                marker=".", ls="--", color=colors[reps], label=f"reps={reps}: FCI mass")
    ax.axhline(CONTROL["key_recall_threshold"], color="gray", ls=":")
    ax.set_ylim(0, 1.02); ax.set_xlabel("lambda"); ax.set_ylabel("accessible coverage")
    ax.set_title("Ansatz support: structural coverage ceiling")
    ax.grid(alpha=.3); ax.legend(fontsize=8)

    # Recall vs shots for representative lambdas
    ax = axes[1, 0]
    for reps in CONTROL["n_reps"]:
        rows_all = data["repetitions"][str(reps)]["samples"]
        for lam in (1.0, 2.0, 3.0, 5.0):
            rows = [r for r in rows_all if r["lambda"] == lam]
            ax.plot([r["shots"] for r in rows], [r["top10_recall_mean"] for r in rows],
                    marker="o", color=colors[reps], alpha=.45 + .1 * (lam / 5),
                    label=f"r={reps}, λ={lam}")
    ax.set_xscale("log"); ax.axhline(CONTROL["key_recall_threshold"], color="gray", ls=":")
    ax.set_xlabel("shots"); ax.set_ylabel("mean top-10 FCI recall")
    ax.set_title("When does key coverage become sufficient?")
    ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)

    # No-K SQD energy vs lambda, several shots for n_reps=3
    ax = axes[1, 1]
    rows_all = data["repetitions"]["3"]["samples"]
    for shots in CONTROL["shots"]:
        rows = [r for r in rows_all if r["shots"] == shots]
        ax.errorbar([r["lambda"] for r in rows], [r["energy_err_mha_mean"] for r in rows],
                    yerr=[r["energy_err_mha_std"] for r in rows], marker="o", capsize=2,
                    label=f"S={shots:g}")
    ax.set_xlabel("lambda"); ax.set_ylabel("SQD error, no K cap (mHa)")
    ax.set_title("n_reps=3: coverage-saturated SQD")
    ax.grid(alpha=.3); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(HERE / CONTROL["figure_file"], dpi=160)


if __name__ == "__main__":
    run()
