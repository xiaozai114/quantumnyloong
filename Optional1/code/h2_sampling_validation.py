#!/usr/bin/env python
"""Explicit finite-shot H2 sampling validation of the analytic spike curve.

This is intentionally separate from the analytic expectation used in
large_lambda_peak_scan.py. For every lambda/shot/seed point we:

1. prepare the H2 LUCJ state with ffsim;
2. draw actual computational-basis samples with ffsim.sample_state_vector;
3. convert sampled bitstrings to determinant addresses;
4. diagonalize the molecular Hamiltonian in that sampled subspace.

The empirical mean is compared with the exact two-configuration expectation.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import ffsim

from lucj_sqd_framework import SQDConfig, LUCJSQDExperiment

HERE = Path(__file__).resolve().parent
CONTROL = {
    "lambdas": [50.0, 55.5, 60.0, 160.0, 166.4, 172.0,
                215.0, 221.9, 228.0, 493.0, 499.3, 505.0],
    "shots": [500, 5000],
    "seeds": 1000,
    "seed": 20260728,
    "data_file": "h2_sampling_validation.json",
    "figure_file": "h2_sampling_validation.png",
}


def main():
    exp = LUCJSQDExperiment(SQDConfig(
        molecule="H2", n_reps=1, local_truncation=True,
        noise_model="none", shots=10, n_seeds=1))
    dim = exp._linop.shape[0]
    hmat = np.asarray(exp._linop @ np.eye(dim), dtype=complex)
    hmat = (hmat + hmat.conj().T) / 2
    e_fci = exp.e_fci

    # Identify the only two nonzero configurations at a generic lambda.
    p_ref = np.abs(exp._lucj_state(5.0)) ** 2
    support = np.flatnonzero(p_ref > 1e-14)
    if len(support) != 2:
        raise RuntimeError(f"Expected two H2 configurations, got {support}")
    hf = int(np.argmax(np.abs(exp._lucj_state(0.0)) ** 2))
    double = int(support[support != hf][0])
    delta_hf = float((hmat[hf, hf].real - e_fci) * 1000)
    delta_double = float((hmat[double, double].real - e_fci) * 1000)

    rows = []
    for ilam, lam in enumerate(CONTROL["lambdas"]):
        psi = np.asarray(exp._lucj_state(lam), dtype=complex)
        p = np.abs(psi) ** 2
        ph, pd = float(p[hf]), float(p[double])
        for shots in CONTROL["shots"]:
            errors = []
            hit_both = 0
            only_hf = 0
            only_double = 0
            for seed in range(CONTROL["seeds"]):
                rng = np.random.default_rng(
                    CONTROL["seed"] + ilam * 100000 + shots + seed)
                samples = ffsim.sample_state_vector(
                    psi, norb=exp.norb, nelec=exp.nelec, shots=shots,
                    bitstring_type=ffsim.BitstringType.STRING, seed=rng)
                addresses = np.unique(ffsim.strings_to_addresses(
                    samples, norb=exp.norb, nelec=exp.nelec))
                energy = float(np.linalg.eigvalsh(
                    hmat[np.ix_(addresses, addresses)])[0].real)
                errors.append((energy - e_fci) * 1000)
                aset = set(map(int, addresses))
                if hf in aset and double in aset:
                    hit_both += 1
                elif hf in aset:
                    only_hf += 1
                elif double in aset:
                    only_double += 1
            errors = np.asarray(errors)
            exact = ph ** shots * delta_hf + pd ** shots * delta_double
            rows.append({
                "lambda": lam, "shots": shots,
                "p_hf": ph, "p_double": pd,
                "analytic_expected_err_mha": float(exact),
                "sampled_mean_err_mha": float(errors.mean()),
                "sampled_std_err_mha": float(errors.std(ddof=1)),
                "sampled_sem_mha": float(errors.std(ddof=1) / np.sqrt(len(errors))),
                "both_fraction": hit_both / CONTROL["seeds"],
                "only_hf_fraction": only_hf / CONTROL["seeds"],
                "only_double_fraction": only_double / CONTROL["seeds"],
            })
            print(
                f"lambda={lam:7.1f} S={shots:5d}: "
                f"sample={errors.mean():9.3f} +/- {errors.std(ddof=1)/np.sqrt(len(errors)):.3f} mHa, "
                f"analytic={exact:9.3f}, both={hit_both/CONTROL['seeds']:.3f}")

    out = {
        "control": CONTROL,
        "delta_hf_mha": delta_hf,
        "delta_double_mha": delta_double,
        "rows": rows,
    }
    with open(HERE / CONTROL["data_file"], "w") as f:
        json.dump(out, f, indent=2)
    make_figure(out)


def make_figure(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, shots in zip(axes, CONTROL["shots"]):
        rows = [r for r in data["rows"] if r["shots"] == shots]
        x = [r["lambda"] for r in rows]
        analytic = [r["analytic_expected_err_mha"] for r in rows]
        sampled = [r["sampled_mean_err_mha"] for r in rows]
        sem = [r["sampled_sem_mha"] for r in rows]
        ax.plot(x, analytic, color="#c0392b", marker="o", label="analytic expectation")
        ax.errorbar(x, sampled, yerr=sem, color="#2980b9", marker="s",
                    ls="--", capsize=3, label="explicit sampled SQD mean")
        ax.set_xlabel("lambda"); ax.set_ylabel("SQD error (mHa)")
        ax.set_title(f"H2 explicit sampling, S={shots}, {CONTROL['seeds']} seeds")
        ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / CONTROL["figure_file"], dpi=160)


if __name__ == "__main__":
    main()
