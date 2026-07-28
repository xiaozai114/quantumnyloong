#!/usr/bin/env python
"""
Controlled experiments for Problem 2 / Optional 1
==================================================

This script separates effects that were mixed together in the original sweeps:

1. Exact-state diagnostics (no finite-shot noise):
   local LUCJ vs full UCJ, state infidelity, total-variation distance, direct
   variational energy, and a fixed-K oracle SQD subspace.

2. Finite-shot diagnostics (no hardware noise):
   compare several shot counts at the same fixed maximum subspace size K.
   Report FCI probability mass covered and recall of the important FCI
   determinants, instead of using raw unique-bitstring count alone.

3. Controlled noise diagnostics:
   postselect the correct particle-number sector and cap the final subspace at K.
   This prevents noise from looking artificially good merely because it creates an
   unlimited number of bitstrings.  We report valid-sample fraction, FCI mass, and
   fixed-K energy.

Edit only CONTROL_PANEL if you want a different experiment.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import ffsim
from pyscf import fci

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lucj_sqd_framework import SQDConfig, LUCJSQDExperiment


CONTROL_PANEL = {
    "lambdas": [0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0],
    "systems": [
        # Strict Problem-2 LiH active space: 4 spatial orbitals, 2 active electrons.
        {"name": "LiH_eq", "molecule": "LiH", "bond": None, "fixed_k": 8},
        {"name": "LiH_2.5A", "molecule": "LiH", "bond": 2.5, "fixed_k": 8},
        # Larger nontrivial test: 7 spatial orbitals / 10 electrons (14 qubits).
        {"name": "H2O_eq", "molecule": "H2O", "bond": None, "fixed_k": 32},
    ],
    "shots": [100, 500, 2000, 10000],
    "finite_shot_seeds": 20,
    "noise_system": "H2O_eq",
    "noise_shots": 500,
    "noise_seeds": 20,
    "noise_cases": [
        {"name": "none", "kind": "none", "p": 0.0},
        {"name": "bitflip_1pct", "kind": "bitflip", "p": 0.01},
        {"name": "bitflip_5pct", "kind": "bitflip", "p": 0.05},
        {"name": "readout_2pct", "kind": "readout", "p": 0.02},
        # A particle-number-preserving random orbital swap. This is not a hardware
        # channel; it is a useful control that isolates wrong-determinant selection
        # without relying on configuration recovery.
        {"name": "sector_swap_2pct", "kind": "sector_swap", "p": 0.02},
        {"name": "sector_swap_10pct", "kind": "sector_swap", "p": 0.10},
    ],
    "top_fci_count": 10,
    "seed": 20260728,
    "data_file": "controlled_experiments.json",
    "figure_file": "controlled_experiments.png",
}


@dataclass
class SystemData:
    spec: dict
    exp: LUCJSQDExperiment
    hmat: np.ndarray
    fci_vec: np.ndarray
    fci_prob: np.ndarray
    top_fci: np.ndarray


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def make_system(spec: dict, top_fci_count: int) -> SystemData:
    cfg = SQDConfig(
        molecule=spec["molecule"], bond=spec["bond"],
        ccsd_scale=0.0, local_truncation=True,
        noise_model="none", shots=10, n_seeds=1,
    )
    exp = LUCJSQDExperiment(cfg)

    # Exact Hamiltonian matrix in the fixed (N_alpha, N_beta) determinant sector.
    dim = exp._linop.shape[0]
    hmat = np.asarray(exp._linop @ np.eye(dim), dtype=complex)
    hmat = (hmat + hmat.conj().T) / 2

    e_fci, civec = fci.direct_spin1.FCI().kernel(
        exp.h1e, exp.eri, exp.norb, exp.nelec, ecore=exp.ecore)
    fci_vec = np.asarray(civec).reshape(-1).astype(complex)
    fci_vec /= np.linalg.norm(fci_vec)
    check = float(np.real(np.vdot(fci_vec, hmat @ fci_vec)))
    if abs(check - e_fci) > 1e-8:
        raise RuntimeError(
            f"FCI/ffsim determinant ordering mismatch: {check} vs {e_fci}")
    fci_prob = np.abs(fci_vec) ** 2
    top_fci = np.argsort(fci_prob)[::-1][:min(top_fci_count, dim)]
    return SystemData(spec, exp, hmat, fci_vec, fci_prob, top_fci)


def state(exp: LUCJSQDExperiment, lam: float, local: bool) -> np.ndarray:
    old = exp.cfg.local_truncation
    exp.cfg.local_truncation = local
    try:
        return np.asarray(exp._lucj_state(lam), dtype=complex)
    finally:
        exp.cfg.local_truncation = old


def subspace_energy(hmat: np.ndarray, addresses: np.ndarray) -> float:
    idx = np.unique(np.asarray(addresses, dtype=int))
    if len(idx) == 0:
        return float("nan")
    return float(np.linalg.eigvalsh(hmat[np.ix_(idx, idx)])[0].real)


def top_k_indices(weights: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(weights))
    return np.argsort(weights)[::-1][:k]


def coverage_metrics(addresses: np.ndarray, data: SystemData) -> dict:
    idx = np.unique(np.asarray(addresses, dtype=int))
    covered_mass = float(data.fci_prob[idx].sum()) if len(idx) else 0.0
    recall = float(len(set(idx).intersection(set(data.top_fci))) / len(data.top_fci))
    return {"effective_dim": int(len(idx)),
            "fci_mass": covered_mass,
            "top_fci_recall": recall}


def valid_particle_number(s: str, norb: int, nelec: tuple[int, int]) -> bool:
    s = s.replace(" ", "")
    if len(s) != 2 * norb:
        return False
    # All systems here are closed shell, so Qiskit/ffsim alpha-beta half ordering
    # does not affect this test. Keep the explicit halves for readability.
    left, right = s[:norb], s[norb:]
    return left.count("1") == nelec[0] and right.count("1") == nelec[1]


def samples_to_fixed_k(samples: list[str], data: SystemData, k: int) -> tuple[np.ndarray, dict]:
    valid = [s.replace(" ", "") for s in samples
             if valid_particle_number(s, data.exp.norb, data.exp.nelec)]
    valid_fraction = len(valid) / max(1, len(samples))
    if not valid:
        return np.array([], dtype=int), {
            "valid_fraction": valid_fraction, "raw_unique": len(set(samples)),
            "valid_unique": 0}

    counts = Counter(valid)
    # Rank valid determinants by observed frequency. Deterministic tie-break by string.
    selected_strings = [s for s, _ in sorted(
        counts.items(), key=lambda item: (-item[1], item[0]))[:k]]
    addresses = ffsim.strings_to_addresses(
        selected_strings, norb=data.exp.norb, nelec=data.exp.nelec)
    return np.asarray(addresses, dtype=int), {
        "valid_fraction": valid_fraction,
        "raw_unique": len(set(samples)), "valid_unique": len(counts)}


def apply_controlled_noise(samples: list[str], case: dict, rng: np.random.Generator,
                           norb: int) -> list[str]:
    kind, p = case["kind"], case["p"]
    if kind == "none":
        return list(samples)
    arr = np.array([[int(b) for b in s] for s in samples], dtype=np.int8)

    if kind in {"bitflip", "readout"}:
        # Symmetric readout and post-measurement bit flip are equivalent on bitstrings;
        # separate labels are retained because their physical interpretations differ.
        mask = rng.random(arr.shape) < p
        arr[mask] = 1 - arr[mask]
    elif kind == "sector_swap":
        for row in arr:
            for start in (0, norb):
                if rng.random() < p:
                    block = row[start:start + norb]
                    occ, vir = np.flatnonzero(block), np.flatnonzero(1 - block)
                    if len(occ) and len(vir):
                        i = int(rng.choice(occ)); a = int(rng.choice(vir))
                        block[i], block[a] = 0, 1
    else:
        raise ValueError(f"Unknown controlled noise kind: {kind}")
    return ["".join(map(str, row)) for row in arr]


# -----------------------------------------------------------------------------
# Experiment 1: exact local/full diagnostics + fixed-K oracle subspace
# -----------------------------------------------------------------------------
def exact_experiment(data: SystemData, cp: dict) -> list[dict]:
    rows = []
    k = data.spec["fixed_k"]
    for lam in cp["lambdas"]:
        psi_l = state(data.exp, lam, local=True)
        psi_f = state(data.exp, lam, local=False)
        p_l, p_f = np.abs(psi_l) ** 2, np.abs(psi_f) ** 2
        idx_l, idx_f = top_k_indices(p_l, k), top_k_indices(p_f, k)
        met_l, met_f = coverage_metrics(idx_l, data), coverage_metrics(idx_f, data)
        rows.append({
            "lambda": lam,
            "trunc_infid": float(1 - abs(np.vdot(psi_l, psi_f)) ** 2),
            "probability_tv": float(0.5 * np.abs(p_l - p_f).sum()),
            "fci_infid_local": float(1 - abs(np.vdot(psi_l, data.fci_vec)) ** 2),
            "fci_infid_full": float(1 - abs(np.vdot(psi_f, data.fci_vec)) ** 2),
            "var_err_local_mha": float((np.real(np.vdot(psi_l, data.hmat @ psi_l))
                                         - data.exp.e_fci) * 1000),
            "var_err_full_mha": float((np.real(np.vdot(psi_f, data.hmat @ psi_f))
                                        - data.exp.e_fci) * 1000),
            "fixed_k": k,
            "oracle_err_local_mha": float((subspace_energy(data.hmat, idx_l)
                                            - data.exp.e_fci) * 1000),
            "oracle_err_full_mha": float((subspace_energy(data.hmat, idx_f)
                                           - data.exp.e_fci) * 1000),
            "oracle_fci_mass_local": met_l["fci_mass"],
            "oracle_fci_mass_full": met_f["fci_mass"],
            "oracle_recall_local": met_l["top_fci_recall"],
            "oracle_recall_full": met_f["top_fci_recall"],
        })
    return rows


# -----------------------------------------------------------------------------
# Experiment 2: finite shots with a fixed maximum subspace dimension
# -----------------------------------------------------------------------------
def finite_shot_experiment(data: SystemData, cp: dict) -> list[dict]:
    rows = []
    k = data.spec["fixed_k"]
    for shots in cp["shots"]:
        for lam in cp["lambdas"]:
            psi = state(data.exp, lam, local=True)
            metrics = []
            for seed in range(cp["finite_shot_seeds"]):
                rng = np.random.default_rng(cp["seed"] + 10000 * shots + 100 * seed)
                samples = ffsim.sample_state_vector(
                    psi, norb=data.exp.norb, nelec=data.exp.nelec, shots=shots,
                    bitstring_type=ffsim.BitstringType.STRING, seed=rng)
                idx, raw = samples_to_fixed_k(samples, data, k)
                cov = coverage_metrics(idx, data)
                metrics.append({
                    **raw, **cov,
                    "energy_err_mha": (subspace_energy(data.hmat, idx)
                                       - data.exp.e_fci) * 1000,
                })
            rows.append(aggregate(metrics, {
                "lambda": lam, "shots": shots, "fixed_k": k,
            }))
    return rows


# -----------------------------------------------------------------------------
# Experiment 3: noise, but fixed K and particle-number postselection
# -----------------------------------------------------------------------------
def noise_experiment(data: SystemData, cp: dict) -> list[dict]:
    rows = []
    k, shots = data.spec["fixed_k"], cp["noise_shots"]
    for case in cp["noise_cases"]:
        for lam in cp["lambdas"]:
            psi = state(data.exp, lam, local=True)
            metrics = []
            for seed in range(cp["noise_seeds"]):
                rng = np.random.default_rng(cp["seed"] + 100000 * seed)
                ideal = ffsim.sample_state_vector(
                    psi, norb=data.exp.norb, nelec=data.exp.nelec, shots=shots,
                    bitstring_type=ffsim.BitstringType.STRING, seed=rng)
                noisy = apply_controlled_noise(ideal, case, rng, data.exp.norb)
                idx, raw = samples_to_fixed_k(noisy, data, k)
                cov = coverage_metrics(idx, data)
                metrics.append({
                    **raw, **cov,
                    "energy_err_mha": (subspace_energy(data.hmat, idx)
                                       - data.exp.e_fci) * 1000,
                })
            rows.append(aggregate(metrics, {
                "lambda": lam, "noise": case["name"], "noise_p": case["p"],
                "shots": shots, "fixed_k": k,
            }))
    return rows


def aggregate(metrics: list[dict], fixed: dict) -> dict:
    result = dict(fixed)
    for key in metrics[0]:
        vals = np.array([m[key] for m in metrics], dtype=float)
        result[f"{key}_mean"] = float(np.nanmean(vals))
        result[f"{key}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
    return result


# -----------------------------------------------------------------------------
# Plot and text summary
# -----------------------------------------------------------------------------
def make_figure(results: dict, cp: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Exact local/full truncation diagnostics
    ax = axes[0, 0]
    for name, block in results["systems"].items():
        rows = block["exact"]
        ax.plot([r["lambda"] for r in rows], [r["trunc_infid"] for r in rows],
                marker="o", label=f"{name}: infidelity")
        ax.plot([r["lambda"] for r in rows], [r["probability_tv"] for r in rows],
                ls="--", marker=".", label=f"{name}: TV")
    ax.set_yscale("log"); ax.set_xlabel("lambda"); ax.set_ylabel("distance")
    ax.set_title("Local vs full UCJ: state/probability distance")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # Fixed-K exact oracle energy
    ax = axes[0, 1]
    for name, block in results["systems"].items():
        rows = block["exact"]
        ax.plot([r["lambda"] for r in rows], [r["oracle_err_local_mha"] for r in rows],
                marker="o", label=f"{name}: local")
        ax.plot([r["lambda"] for r in rows], [r["oracle_err_full_mha"] for r in rows],
                ls="--", marker=".", label=f"{name}: full")
    ax.set_xlabel("lambda"); ax.set_ylabel("fixed-K SQD error (mHa)")
    ax.set_title("Exact probabilities, same subspace budget K")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # Finite-shot fixed-K (H2O only for readability)
    ax = axes[1, 0]
    h2o = results["systems"]["H2O_eq"]["finite_shots"]
    for shots in cp["shots"]:
        rows = [r for r in h2o if r["shots"] == shots]
        ax.errorbar([r["lambda"] for r in rows], [r["energy_err_mha_mean"] for r in rows],
                    yerr=[r["energy_err_mha_std"] for r in rows], marker="o", capsize=2,
                    label=f"S={shots}")
    ax.set_xlabel("lambda"); ax.set_ylabel("fixed-K SQD error (mHa)")
    ax.set_title("H2O: finite shots, fixed K=32")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # Controlled noise with fixed K and postselection
    ax = axes[1, 1]
    rows_all = results["noise_experiment"]
    for case in cp["noise_cases"]:
        rows = [r for r in rows_all if r["noise"] == case["name"]]
        ax.errorbar([r["lambda"] for r in rows], [r["energy_err_mha_mean"] for r in rows],
                    yerr=[r["energy_err_mha_std"] for r in rows], marker="o", capsize=2,
                    label=case["name"])
    ax.set_xlabel("lambda"); ax.set_ylabel("fixed-K SQD error (mHa)")
    ax.set_title("H2O: noise with postselection and fixed K=32")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(HERE / cp["figure_file"], dpi=160)


def main():
    cp = CONTROL_PANEL
    systems = {s["name"]: make_system(s, cp["top_fci_count"])
               for s in cp["systems"]}

    results = {"control_panel": cp, "systems": {}}
    for name, data in systems.items():
        print(f"Running exact + finite-shot experiments: {name} "
              f"(dim={len(data.fci_vec)}, K={data.spec['fixed_k']})")
        results["systems"][name] = {
            "dimension": len(data.fci_vec),
            "e_hf": data.exp.e_hf, "e_fci": data.exp.e_fci,
            "fixed_k": data.spec["fixed_k"],
            "exact": exact_experiment(data, cp),
            "finite_shots": finite_shot_experiment(data, cp),
        }

    noise_data = systems[cp["noise_system"]]
    print(f"Running controlled noise experiment: {cp['noise_system']}")
    results["noise_experiment"] = noise_experiment(noise_data, cp)

    with open(HERE / cp["data_file"], "w") as f:
        json.dump(results, f, indent=2)
    make_figure(results, cp)

    print(f"Saved: {HERE / cp['data_file']}")
    print(f"Saved: {HERE / cp['figure_file']}")


if __name__ == "__main__":
    main()
