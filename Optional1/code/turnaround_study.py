#!/usr/bin/env python
"""
turnaround_study.py
===================
Control script to hunt for the LUCJ-SQD "turn-around" (回升) of the energy error
as a function of the CCSD scale lambda, while sweeping the experimental knobs:

    * sampling count      (shots)
    * molecule type       (molecule)
    * noise on/off & type (noise)
    * bond stretch        (bond)  -- diatomics only
    * lambda range        (lambdas, may exceed 1.0)

HOW TO USE
----------
Edit ONLY the `CONTROL_PANEL` block below, then run:

    /path/to/python turnaround_study.py

Everything else is machinery.  For each scenario the script performs a full
lambda-sweep, prints a per-lambda table, and reports whether a turn-around was
detected (error decreases then rises again by more than `RISE_THRESHOLD_MHA`).
Optionally saves a PNG with one error-vs-lambda curve per scenario.

Depends on the local framework `lucj_sqd_framework.py`.
"""

from __future__ import annotations
import itertools
import json
import numpy as np

from lucj_sqd_framework import SQDConfig, LUCJSQDExperiment


# ======================================================================
#  CONTROL PANEL  --  edit these to design your experiment
# ======================================================================
CONTROL_PANEL = dict(

    # -- lambda axis (the x-axis of every curve). May exceed 1.0. --
    lambdas=[0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0, 10.0],

    # -- Each knob below is a LIST. The script runs the Cartesian product --
    # -- of all knobs, i.e. one scenario per combination.                --

    molecules=["LiH"],                 # e.g. ["H2", "LiH", "H2O", "N2"]
    shots_list=[10000, 50000, 100000, 1000000],            # sampling counts to compare
    bonds=[None],                 # None = equilibrium; numbers = stretch (A).
                                       # (ignored for non-diatomics like H2O/C2H4)

    # noise scenarios: list of dicts. {} means noiseless.
    # keys map directly onto SQDConfig fields.
    noise_scenarios=[
    #    {"noise_model": "none"},
        {"noise_model": "bitflip", "p_flip": 0.02},
    #    {"noise_model": "aer_depolarizing", "p_2q": 0.005, "p_1q": 0.0005},
    ],

    # -- statistics / solver --
    n_seeds=15,                        # samplings averaged per data point
    config_recovery=True,              # SQD configuration recovery on/off
    n_reps=1,                          # LUCJ repetitions
    local_truncation=True,             # keep only adjacent RZZ

    # -- analysis / output --
    rise_threshold_mha=0.5,            # min error rise to call it a "turn-around"
    make_plot=True,
    plot_path="huge_lambda2.png",
    data_path="huge_lambda2.json",
)
# ======================================================================


DIATOMIC = {"H2", "LiH", "N2"}         # molecules for which `bond` is meaningful


def build_scenarios(cp) -> list[dict]:
    """Cartesian product of the knob lists -> a list of scenario dicts."""
    scenarios = []
    for mol, shots, bond, noise in itertools.product(
            cp["molecules"], cp["shots_list"], cp["bonds"], cp["noise_scenarios"]):
        # skip bond stretch for non-diatomic molecules (keep only the None entry)
        if mol not in DIATOMIC and bond is not None:
            continue
        scenarios.append(dict(molecule=mol, shots=shots, bond=bond, noise=noise))
    return scenarios


def label(sc) -> str:
    b = "eq" if sc["bond"] is None else f"{sc['bond']}A"
    nm = sc["noise"].get("noise_model", "none")
    extra = ""
    if nm == "bitflip":
        extra = f"(p={sc['noise'].get('p_flip')})"
    elif nm.startswith("aer") or nm == "depolarizing":
        extra = f"(p2q={sc['noise'].get('p_2q')})"
    return f"{sc['molecule']}|{b}|S={sc['shots']}|{nm}{extra}"


def run_scenario(sc, cp):
    """Full lambda-sweep for one scenario. Returns dict with arrays + verdict."""
    # base config shared by every lambda point in this scenario
    base = SQDConfig(
        molecule=sc["molecule"],
        bond=sc["bond"],
        shots=sc["shots"],
        n_seeds=cp["n_seeds"],
        config_recovery=cp["config_recovery"],
        n_reps=cp["n_reps"],
        local_truncation=cp["local_truncation"],
        **sc["noise"],
    )
    exp = LUCJSQDExperiment(base)         # builds molecule once
    results = exp.sweep("ccsd_scale", cp["lambdas"])

    lam = np.array(cp["lambdas"], float)
    err = np.array([r.err_mean_mha for r in results], float)
    M = np.array([r.n_configs_mean for r in results], float)
    e_fci = results[0].e_fci
    e_hf = results[0].e_hf

    # turn-around detection: interior minimum followed by a rise
    imin = int(np.nanargmin(err))
    rise_after = float(np.nanmax(err[imin:]) - err[imin]) if imin < len(err) - 1 else 0.0
    turn_around = (0 < imin < len(err) - 1) and (rise_after > cp["rise_threshold_mha"])

    return dict(
        label=label(sc), scenario=sc,
        lambdas=lam.tolist(), err_mha=err.tolist(), M=M.tolist(),
        e_hf=e_hf, e_fci=e_fci,
        opt_lambda=float(lam[imin]), opt_err_mha=float(err[imin]),
        rise_after_mha=rise_after, turn_around=bool(turn_around),
    )


def print_scenario(res):
    print(f"\n### {res['label']}   "
          f"(E_HF={res['e_hf']:.5f}, E_FCI={res['e_fci']:.5f})")
    print("   lambda    err(mHa)     M")
    for lam, err, m in zip(res["lambdas"], res["err_mha"], res["M"]):
        print(f"   {lam:5.2f}   {err:+9.3f}   {m:6.1f}")
    verdict = "YES  <-- turn-around" if res["turn_around"] else "no"
    print(f"   => optimal lambda={res['opt_lambda']:.2f} "
          f"(err={res['opt_err_mha']:+.3f} mHa); rise after min="
          f"{res['rise_after_mha']:+.3f} mHa; turn-around: {verdict}")


def make_plot(all_res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    for res in all_res:
        marker = "o" if res["turn_around"] else "."
        ax.plot(res["lambdas"], res["err_mha"], marker=marker, lw=1.4,
                label=res["label"] + (" *" if res["turn_around"] else ""))
    ax.axhline(1.6, color="green", ls=":", lw=1, label="chemical accuracy")
    ax.axvline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("ccsd_scale  lambda")
    ax.set_ylabel("SQD energy error vs FCI (mHa)")
    ax.set_title("LUCJ-SQD turn-around study  (* = turn-around detected)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nSaved plot -> {path}")


def main():
    cp = CONTROL_PANEL
    scenarios = build_scenarios(cp)
    print(f"Running {len(scenarios)} scenario(s), "
          f"{len(cp['lambdas'])} lambda points each, "
          f"{cp['n_seeds']} seeds per point.\n" + "=" * 60)

    all_res = []
    for sc in scenarios:
        res = run_scenario(sc, cp)
        print_scenario(res)
        all_res.append(res)

    # summary
    print("\n" + "=" * 60 + "\nSUMMARY")
    n_turn = sum(r["turn_around"] for r in all_res)
    for r in all_res:
        flag = "TURN-AROUND" if r["turn_around"] else "monotone/edge"
        print(f"  [{flag:12s}] {r['label']:45s} "
              f"opt_lambda={r['opt_lambda']:.2f} min_err={r['opt_err_mha']:+.3f}mHa")
    print(f"\n{n_turn}/{len(all_res)} scenario(s) showed a turn-around "
          f"(rise > {cp['rise_threshold_mha']} mHa after the minimum).")

    with open(cp["data_path"], "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"Saved data -> {cp['data_path']}")

    if cp["make_plot"]:
        make_plot(all_res, cp["plot_path"])


if __name__ == "__main__":
    main()
