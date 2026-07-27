"""
experiment2.py -- Multi-layer (double-factorized) LUCJ + SQD for LiH
=====================================================================

Follow-up study to experiment.py, prompted by the cross-worker's finding
(code/2.5-review2): a SINGLE-layer LUCJ with a diagonal Jastrow cannot place
probability on the paired doubles at all -- the correlation energy recovered
by SQD is then capped by whatever determinants the hand-built paired-Givens
layer injects (local truncation plateau: 1.587 mHa above FCI).

The multi-layer double-factorized (DF) LUCJ

    |Psi(lam)> = prod_{k=1..L} U_k e^{i J_k} U_k^dagger |HF>

fixes this structurally:
  * layers  <- eigendecomposition T[a,b] = t2[0,0,a,b] = sum_k g_k v_k v_k^T
  * U_k rotates occupied orbital 0 into the collective virtual mode v_k by a
    FIXED angle theta = pi/4 (amplitude-in-phase trick);
  * the Jastrow phase phi_k = g_k * lam breaks the U_k/U_k^dagger cancellation
    and leaves a paired-double amplitude ~ sin^2(theta) * phi_k  (linear in
    lam, so probability ~ lam^2 with an O(g_k^2) prefactor instead of the
    hopelessly small O(t2^4) one would get from theta ~ t2).

Why this is sound for SQD: the theta=pi/4 state is a POOR wavefunction, but
SQD only needs the right determinants to APPEAR in the samples -- the
subspace diagonalisation re-optimises all amplitudes ("support over
fidelity").

Parts
-----
  1. DF layer table + variational floors (HF+paired subspace, full support).
  2. Scaling: p_paired(lam) for L = 1,2,3 layers vs. the single-layer local
     LUCJ of experiment.py; log-log slope fits.
  3. Energies vs lam: infinite-sample SQD (exact support) for L = 1,2,3 and
     sampled SQD (S = 1000, S = 100000) for L = 3; HF / FCI / old plateau
     reference lines.
  4. Sample requirement S_req = ln(0.05)/ln(1-p_paired) vs lam, multi-layer
     vs single-layer; lambda threshold where S = 1000 suffices.

Outputs
-------
  multilayer_energy_vs_lambda.png
  multilayer_scaling.png
  multilayer_data.npz

Run:  python experiment2.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorcircuit as tc

tc.set_backend("tensorflow")
tc.set_dtype("complex128")

from definitions import (
    build_lih_active_space,
    ccsd_amplitudes,
    ccsd_t2_active,
    df_layers,
    build_multilayer_circuit,
    lucj_generators,
    build_circuit,
    paired_double_probability,
    support_bits,
    dets_to_bits,
    sample_bitstrings,
    sqd_energy,
)

HARTREE_TO_MHA = 1000.0
CONF = 0.95  # confidence level for the coupon-collector sample requirement


def state_of(c):
    return np.asarray(tc.backend.numpy(c.state()))


def sreq(p):
    """Samples needed to see an event of probability p at confidence CONF.
    Uses log1p to stay accurate (and finite) for tiny p."""
    if p <= 0.0:
        return np.inf
    if p >= 1.0:
        return 1.0
    denom = np.log1p(-p)
    if denom == 0.0:  # p below double-precision resolution
        return np.inf
    return np.log(1.0 - CONF) / denom


def main():
    # ------------------------------------------------------------------
    # Part 0/1 -- setup, DF layers, variational floors
    # ------------------------------------------------------------------
    print("=" * 72)
    print("PART 1: setup, DF layers, variational floors")
    print("=" * 72)

    data = build_lih_active_space()
    nq = 2 * data["nact"]
    nocc = data["nelec_act"] // 2
    e_hf, e_fci = data["e_hf"], data["e_fci"]
    print(f"E_HF  = {e_hf:.8f} Ha")
    print(f"E_FCI = {e_fci:.8f} Ha   (dE_corr = "
          f"{(e_hf - e_fci) * HARTREE_TO_MHA:.3f} mHa)")

    # active-window CCSD -> paired channel -> DF layers
    _, t2_act, e_corr_act = ccsd_t2_active(data)
    layers = df_layers(data, t2_act)
    L = len(layers)
    print(f"\nDF layers of T[a,b] = t2[0,0,a,b]  (L = {L}):")
    for k, (g, _) in enumerate(layers):
        print(f"  layer {k + 1}:  g_k = {g:+.6f}")

    # variational floor of the {HF + paired doubles} subspace
    dets_floor = [(tuple(range(nocc)), tuple(range(nocc)))]
    dets_floor += [((a,), (a,)) for a in range(nocc, data["nact"])]
    e_floor = sqd_energy(dets_to_bits(data, dets_floor), data)["energy"]
    print(f"\nVariational floor of {{HF + 3 paired doubles}}: "
          f"{e_floor:.8f} Ha  "
          f"({(e_floor - e_fci) * HARTREE_TO_MHA:.3f} mHa above FCI)")

    # single-layer local-LUCJ plateau (experiment.py reference)
    t1_full, t2_full = ccsd_amplitudes(data)
    tp, kap, Jm = lucj_generators(data, t1_full, t2_full, 1.0)
    psi_old = state_of(build_circuit(data, tp, kap, Jm, local=True))
    e_old_inf = sqd_energy(support_bits(psi_old, nq), data)["energy"]
    print(f"Single-layer local LUCJ plateau (inf. samples): "
          f"{e_old_inf:.8f} Ha  "
          f"({(e_old_inf - e_fci) * HARTREE_TO_MHA:.3f} mHa above FCI)")

    # ------------------------------------------------------------------
    # Part 2 -- p_paired scaling
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("PART 2: paired-double probability scaling")
    print("=" * 72)

    lam_scan = np.linspace(0.0, 1.0, 51)
    p_paired_ml = np.zeros((3, len(lam_scan)))   # rows: L = 1, 2, 3
    p_paired_old = np.zeros(len(lam_scan))

    for j, lam in enumerate(lam_scan):
        for nl in (1, 2, 3):
            if nl > L:
                continue
            psi = state_of(build_multilayer_circuit(data, layers, lam,
                                                    n_layers=nl))
            _, p_paired_ml[nl - 1, j] = paired_double_probability(data, psi)
        tp, kap, Jm = lucj_generators(data, t1_full, t2_full, lam)
        psi = state_of(build_circuit(data, tp, kap, Jm, local=True))
        _, p_paired_old[j] = paired_double_probability(data, psi)

    # log-log slope fits on lam in [0.1, 1]
    fit_mask = lam_scan >= 0.1
    slopes = {}
    for label, curve in [("L=1", p_paired_ml[0]), ("L=2", p_paired_ml[1]),
                         ("L=3", p_paired_ml[2]),
                         ("single-layer", p_paired_old)]:
        y = curve[fit_mask]
        ok = y > 0
        if ok.sum() >= 3:
            s = np.polyfit(np.log(lam_scan[fit_mask][ok]), np.log(y[ok]), 1)[0]
            slopes[label] = s
            print(f"  {label:>13s}:  p_paired(lam=1) = {curve[-1]:.3e}   "
                  f"log-log slope = {s:.2f}")
        else:
            slopes[label] = np.nan
            print(f"  {label:>13s}:  p_paired identically ~ 0 "
                  "(no paired support)")

    # ------------------------------------------------------------------
    # Part 3 -- energies vs lambda
    #
    # Each lambda point uses R_REP INDEPENDENT sampling repetitions (the
    # state vector is computed once, then re-sampled): a single multinomial
    # draw makes "did we catch the rare paired doubles?" a Bernoulli event,
    # which shows up as zig-zag noise between adjacent lambda points.  We
    # therefore report mean +/- std over repetitions.
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"PART 3: SQD energies vs lambda  (R = {20} repetitions/point)")
    print("=" * 72)

    R_REP = 20
    lam_E = np.linspace(0.0, 1.0, 41)
    E_inf = np.zeros((3, len(lam_E)))    # infinite-sample, L = 1, 2, 3
    E_s1k = np.zeros((len(lam_E), R_REP))     # S = 1000,   L = 3
    E_s100k = np.zeros((len(lam_E), R_REP))   # S = 100000, L = 3

    def draw_bits(probs, n_shots, rng):
        idx = rng.choice(probs.shape[0], size=n_shots, p=probs)
        shifts = (nq - 1 - np.arange(nq))[None, :]
        return ((idx[:, None] >> shifts) & 1).astype(int)

    print(f"{'lam':>6s} {'E_inf(L=3)':>14s} {'<E_S=1e3>':>14s} "
          f"{'<E_S=1e5>':>14s} {'std_1e5(mHa)':>13s}")
    for j, lam in enumerate(lam_E):
        for nl in (1, 2, 3):
            psi = state_of(build_multilayer_circuit(data, layers, lam,
                                                    n_layers=nl))
            E_inf[nl - 1, j] = sqd_energy(support_bits(psi, nq),
                                          data)["energy"]
            if nl == 3:
                probs = np.abs(psi) ** 2
                probs = probs / probs.sum()
                for r in range(R_REP):
                    rng = np.random.default_rng(1000 * j + r)
                    E_s1k[j, r] = sqd_energy(draw_bits(probs, 1000, rng),
                                             data)["energy"]
                    E_s100k[j, r] = sqd_energy(draw_bits(probs, 100000, rng),
                                               data)["energy"]
        print(f"{lam:6.2f} {E_inf[2, j]:14.8f} {E_s1k[j].mean():14.8f} "
              f"{E_s100k[j].mean():14.8f} "
              f"{E_s100k[j].std() * HARTREE_TO_MHA:13.4f}")

    # ------------------------------------------------------------------
    # Part 4 -- sample requirement
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("PART 4: sample requirement S_req (coupon-collector, 95% conf.)")
    print("=" * 72)

    S_req_ml = np.array([sreq(p) for p in p_paired_ml[2]])
    S_req_old = np.array([sreq(p) for p in p_paired_old])

    def lam_threshold(S_req, budget):
        for j in range(len(lam_scan)):
            if S_req[j] <= budget:
                return lam_scan[j]
        return None

    for budget in (1000.0, 1e5):
        th_ml = lam_threshold(S_req_ml, budget)
        th_old = lam_threshold(S_req_old, budget)
        print(f"  budget S = {budget:>8.0f}:  multilayer lam >= "
              f"{th_ml if th_ml is not None else 'never':>6}   "
              f"single-layer lam >= "
              f"{th_old if th_old is not None else 'never'}")
    print(f"  S_req(lam=1): multilayer = {S_req_ml[-1]:.0f}, "
          f"single-layer = {S_req_old[-1]:.0f}")
    lam_1k_ml = lam_threshold(S_req_ml, 1e5)
    lam_1k_old = lam_threshold(S_req_old, 1e5)

    # ------------------------------------------------------------------
    # Figure 1: energies
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    m_s1k, sd_s1k = E_s1k.mean(axis=1), E_s1k.std(axis=1)
    m_s100k, sd_s100k = E_s100k.mean(axis=1), E_s100k.std(axis=1)

    ax1.axhline(e_hf, color="gray", ls=":", lw=1.2, label="HF")
    ax1.axhline(e_fci, color="k", ls="--", lw=1.2, label="FCI")
    ax1.axhline(e_old_inf, color="tab:red", ls="-.", lw=1.2,
                label="single-layer local plateau")
    ax1.plot(lam_E, E_inf[2], "o-", color="tab:blue", ms=4,
             label=r"multilayer, $S\to\infty$")
    ax1.fill_between(lam_E, m_s100k - sd_s100k, m_s100k + sd_s100k,
                     color="tab:green", alpha=0.20)
    ax1.plot(lam_E, m_s100k, "s--", color="tab:green", ms=4,
             label=r"multilayer, $S=10^5$ (mean$\pm$sd)")
    ax1.fill_between(lam_E, m_s1k - sd_s1k, m_s1k + sd_s1k,
                     color="tab:orange", alpha=0.20)
    ax1.plot(lam_E, m_s1k, "^--", color="tab:orange", ms=4,
             label=r"multilayer, $S=10^3$ (mean$\pm$sd)")
    ax1.set_xlabel(r"$\lambda$")
    ax1.set_ylabel(r"$E_{\mathrm{SQD}}$ (Ha)")
    ax1.set_title("Multi-layer DF-LUCJ + SQD energies")
    ax1.legend(fontsize=8, loc="center right")
    ax1.grid(alpha=0.3)

    for nl, col in [(1, "tab:purple"), (2, "tab:cyan"), (3, "tab:blue")]:
        err = np.maximum((E_inf[nl - 1] - e_fci) * HARTREE_TO_MHA, 1e-8)
        ax2.semilogy(lam_E, err, "o-", ms=4, color=col, label=f"L = {nl}")
    err_old = (e_old_inf - e_fci) * HARTREE_TO_MHA
    ax2.axhline(err_old, color="tab:red", ls="-.", lw=1.2,
                label="single-layer plateau")
    ax2.axhline((e_floor - e_fci) * HARTREE_TO_MHA, color="k", ls=":",
                lw=1.2, label="{HF+paired} floor")
    ax2.set_xlabel(r"$\lambda$")
    ax2.set_ylabel(r"$E_{\mathrm{SQD}} - E_{\mathrm{FCI}}$ (mHa)")
    ax2.set_title(r"Layer ablation ($S\to\infty$)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig("multilayer_energy_vs_lambda.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: scaling + sample requirement
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for nl, col in [(1, "tab:purple"), (2, "tab:cyan"), (3, "tab:blue")]:
        lab = f"multilayer L={nl}"
        if not np.isnan(slopes.get(f"L={nl}", np.nan)):
            lab += f"  (slope {slopes[f'L={nl}']:.2f})"
        ax1.loglog(lam_scan[1:], np.maximum(p_paired_ml[nl - 1, 1:], 1e-18),
                   "-", color=col, lw=1.6, label=lab)
    lab = "single-layer local"
    if not np.isnan(slopes.get("single-layer", np.nan)):
        lab += f"  (slope {slopes['single-layer']:.2f})"
    ax1.loglog(lam_scan[1:], np.maximum(p_paired_old[1:], 1e-18), "-",
               color="tab:red", lw=1.6, label=lab)
    ax1.set_xlabel(r"$\lambda$")
    ax1.set_ylabel(r"$p_{\mathrm{paired}}$")
    ax1.set_ylim(1e-10, 1.0)
    ax1.set_title("Paired-double probability scaling")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    ax2.semilogy(lam_scan[1:], S_req_ml[1:], "-", color="tab:blue", lw=1.8,
                 label="multilayer (L=3)")
    ok = np.isfinite(S_req_old[1:])
    ax2.semilogy(lam_scan[1:][ok], S_req_old[1:][ok], "-", color="tab:red",
                 lw=1.8, label="single-layer local")
    ax2.axhline(1000, color="gray", ls=":", lw=1.2, label=r"$S = 10^3$")
    ax2.axhline(1e5, color="gray", ls="--", lw=1.2, label=r"$S = 10^5$")
    if lam_1k_ml is not None:
        ax2.axvline(lam_1k_ml, color="tab:blue", ls=":", lw=1.0)
    if lam_1k_old is not None:
        ax2.axvline(lam_1k_old, color="tab:red", ls=":", lw=1.0)
    ax2.set_xlabel(r"$\lambda$")
    ax2.set_ylabel(r"$S_{\mathrm{req}}$ (95% conf.)")
    ax2.set_title("Samples needed to observe a paired double")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig("multilayer_scaling.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # save raw data
    # ------------------------------------------------------------------
    np.savez(
        "multilayer_data.npz",
        e_hf=e_hf, e_fci=e_fci, e_floor=e_floor, e_old_inf=e_old_inf,
        g_layers=np.array([g for g, _ in layers]),
        lam_scan=lam_scan, p_paired_ml=p_paired_ml, p_paired_old=p_paired_old,
        S_req_ml=S_req_ml, S_req_old=S_req_old,
        lam_E=lam_E, E_inf=E_inf,
        E_s1k=E_s1k, E_s100k=E_s100k,       # shape (nlam, R_REP)
        E_s1k_mean=E_s1k.mean(axis=1), E_s1k_std=E_s1k.std(axis=1),
        E_s100k_mean=E_s100k.mean(axis=1), E_s100k_std=E_s100k.std(axis=1),
        R_REP=R_REP,
    )

    print("\nDone.  Wrote multilayer_energy_vs_lambda.png, "
          "multilayer_scaling.png, multilayer_data.npz")


if __name__ == "__main__":
    main()
