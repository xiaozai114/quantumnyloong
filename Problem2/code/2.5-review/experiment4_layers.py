"""
experiment4_layers.py -- role of the number of LUCJ layers L.

Two complementary views of how the SQD energy depends on L:

  PART 1  E_SQD(lambda) for L = 1, 2, 3.
          * S -> infinity : exact support (deterministic, no averaging).
          * given S (S = 1e3, 1e5) : R repetitions per (lambda, L) point,
            reported as mean +/- std to expose how layer count and shot
            budget together determine the recovered energy.

  PART 2  E_SQD(S) at fixed lambda, for L = 1, 2, 3.
          For each L we scan S and plot the error (median + 10/90 pct band)
          against log S, with the coupon-collector threshold S_req(L) marked.
          More layers => larger p_paired => smaller S_req => faster
          convergence, which is the whole point of multi-layer LUCJ.

Run:  python experiment4_layers.py   (use the 'tc' env: numpy 1.26)
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
    ccsd_t2_active,
    df_layers,
    build_multilayer_circuit,
    paired_double_probability,
    sqd_energy,
    support_bits,
)

HARTREE_TO_MHA = 1000.0
CONF = 0.95
EPS_HIT = 1e-3  # Ha; |E - E_FCI| < EPS_HIT counts as "hit FCI"


def state_of(c):
    return np.asarray(tc.backend.numpy(c.state()))


def draw_bits(probs, n_shots, nq, rng):
    idx = rng.choice(probs.shape[0], size=n_shots, p=probs)
    shifts = (nq - 1 - np.arange(nq))[None, :]
    return ((idx[:, None] >> shifts) & 1).astype(int)


def sreq(p):
    if p <= 0.0 or p >= 1.0:
        return np.inf
    d = np.log1p(-p)
    return np.log(1.0 - CONF) / d if d != 0.0 else np.inf


def main():
    data = build_lih_active_space()
    nq = 2 * data["nact"]
    e_hf, e_fci = data["e_hf"], data["e_fci"]
    _, t2, _ = ccsd_t2_active(data)
    layers = df_layers(data, t2)
    Lmax = len(layers)
    print(f"E_HF = {e_hf:.8f}  E_FCI = {e_fci:.8f}  (Lmax = {Lmax})")

    # =================================================================
    # PART 1 -- E_SQD(lambda) for L = 1, 2, 3  (inf + finite-S averaged)
    # =================================================================
    lam_E = np.linspace(0.0, 1.0, 21)
    S_vals = [1000, 100000]
    R1 = 15
    E_inf = np.zeros((Lmax, len(lam_E)))
    E_fin = np.zeros((Lmax, len(S_vals), len(lam_E), R1))

    print("\nPART 1:  E_SQD(lambda) for L = 1..%d" % Lmax)
    for j, lam in enumerate(lam_E):
        for li in range(Lmax):
            psi = state_of(build_multilayer_circuit(data, layers, lam,
                                                     n_layers=li + 1))
            E_inf[li, j] = sqd_energy(support_bits(psi, nq), data)["energy"]
            probs = np.abs(psi) ** 2
            probs /= probs.sum()
            for si, S in enumerate(S_vals):
                for r in range(R1):
                    rng = np.random.default_rng(1000 * j + 100 * li
                                                + 10 * si + r)
                    E_fin[li, si, j, r] = sqd_energy(
                        draw_bits(probs, int(S), nq, rng), data)["energy"]

    # =================================================================
    # PART 2 -- fixed lambda, scan S, compare L = 1, 2, 3
    # =================================================================
    lam_fix = [1.0, 0.5]
    S_scan = np.array([200, 500, 1000, 2000, 5000, 1e4, 2e4, 5e4, 1e5])
    R2 = 30
    err_fix = {}      # key (lam, L) -> (len(S_scan), R2) in mHa
    pp_L = {}         # key (lam, L) -> p_paired
    sreq_L = {}       # key (lam, L) -> S_req

    print("\nPART 2:  E_SQD(S) at fixed lambda, per layer count")
    for lam in lam_fix:
        for li in range(Lmax):
            psi = state_of(build_multilayer_circuit(data, layers, lam,
                                                     n_layers=li + 1))
            probs = np.abs(psi) ** 2
            probs /= probs.sum()
            _, pp = paired_double_probability(data, psi)
            pp_L[(lam, li + 1)] = pp
            sreq_L[(lam, li + 1)] = sreq(pp)
            e = np.zeros((len(S_scan), R2))
            for i, S in enumerate(S_scan):
                for r in range(R2):
                    rng = np.random.default_rng(int(lam * 1e6) + 1000 * li
                                                + 100 * i + r)
                    e[i, r] = (sqd_energy(draw_bits(probs, int(S), nq, rng),
                                          data)["energy"] - e_fci) * HARTREE_TO_MHA
            err_fix[(lam, li + 1)] = e
            print(f"  lam={lam}: L={li+1}  p_paired={pp:.2e}  "
                  f"S_req={sreq(pp):.0f}")

    # -----------------------------------------------------------------
    # Figure 1: E_SQD(lambda) per layer count
    # -----------------------------------------------------------------
    fig1, axes = plt.subplots(1, Lmax, figsize=(4.2 * Lmax, 4.6),
                              sharey=True)
    if Lmax == 1:
        axes = [axes]
    colors = {0: "tab:orange", 1: "tab:green"}  # S indices
    for li in range(Lmax):
        ax = axes[li]
        ax.axhline(e_hf, color="gray", ls=":", lw=1.0, label="HF")
        ax.axhline(e_fci, color="k", ls="--", lw=1.0, label="FCI")
        ax.plot(lam_E, E_inf[li], "o-", color="tab:blue", ms=3, lw=1.2,
                label=r"$S\to\infty$")
        for si, S in enumerate(S_vals):
            m = E_fin[li, si].mean(axis=-1)
            sd = E_fin[li, si].std(axis=-1)
            ax.fill_between(lam_E, m - sd, m + sd,
                            color=colors[si], alpha=0.18)
            ax.plot(lam_E, m, "s--", color=colors[si], ms=3, lw=1.2,
                    label=rf"$S={S:.0e}$ (mean$\pm$sd)")
        ax.set_title(f"L = {li + 1} layer(s)")
        ax.set_xlabel(r"$\lambda$")
        if li == 0:
            ax.set_ylabel(r"$E_{\mathrm{SQD}}$ (Ha)")
        ax.legend(fontsize=7, loc="center right")
        ax.grid(alpha=0.3)
    fig1.suptitle("Multi-layer DF-LUCJ + SQD: energy vs lambda (averaged)")
    fig1.tight_layout()
    fig1.savefig("layers_energy_vs_lambda.png", dpi=160)
    plt.close(fig1)

    # -----------------------------------------------------------------
    # Figure 2: E_SQD(S) at fixed lambda, per layer count
    # -----------------------------------------------------------------
    fig2, axes = plt.subplots(1, len(lam_fix), figsize=(6.2 * len(lam_fix),
                                                        4.8))
    if len(lam_fix) == 1:
        axes = [axes]
    layer_colors = {1: "tab:red", 2: "tab:blue", 3: "tab:green"}
    for ai, lam in enumerate(lam_fix):
        ax = axes[ai]
        for li in range(Lmax):
            L = li + 1
            e = err_fix[(lam, L)]
            med = np.median(e, axis=1)
            lo = np.percentile(e, 10, axis=1)
            hi = np.percentile(e, 90, axis=1)
            c = layer_colors.get(L, None)
            ax.fill_between(S_scan, lo, hi, color=c, alpha=0.15)
            ax.semilogx(S_scan, med, "o-", color=c, ms=4, lw=1.4,
                        label=f"L = {L} (S_req={sreq_L[(lam,L)]:.0f})")
            ax.axvline(sreq_L[(lam, L)], color=c, ls=":", lw=1.0)
        ax.axhline(EPS_HIT * HARTREE_TO_MHA, color="k", ls="--", lw=0.8,
                   label="hit threshold")
        ax.set_xlabel("measurements S (shots)")
        ax.set_ylabel(r"$|E_{\mathrm{SQD}}-E_{\mathrm{FCI}}|$ (mHa)")
        ax.set_title(f"fixed $\\lambda$ = {lam}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig2.suptitle("Convergence with shots: role of layer count")
    fig2.tight_layout()
    fig2.savefig("layers_shots_scaling.png", dpi=160)
    plt.close(fig2)

    # -----------------------------------------------------------------
    # Figure 3: p_paired and S_req vs L (the mechanism)
    # -----------------------------------------------------------------
    fig3, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    Ls = np.arange(1, Lmax + 1)
    for lam in lam_fix:
        pp = [pp_L[(lam, L)] for L in Ls]
        sr = [sreq_L[(lam, L)] for L in Ls]
        ax[0].loglog(Ls, pp, "o-", label=f"lam = {lam}")
        ax[1].loglog(Ls, sr, "s-", label=f"lam = {lam}")
    ax[0].set_xlabel("number of layers L")
    ax[0].set_ylabel(r"$p_{\mathrm{paired}}$")
    ax[0].set_title("Paired-double probability vs L")
    ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
    ax[1].set_xlabel("number of layers L")
    ax[1].set_ylabel(r"$S_{\mathrm{req}}$ (95%)")
    ax[1].set_title("Samples needed vs L")
    ax[1].legend(); ax[1].grid(alpha=0.3, which="both")
    fig3.tight_layout()
    fig3.savefig("layers_sample_requirement.png", dpi=160)
    plt.close(fig3)

    # -----------------------------------------------------------------
    # save
    # -----------------------------------------------------------------
    np.savez("layers_data.npz",
             lam_E=lam_E, S_vals=S_vals, R1=R1, E_inf=E_inf, E_fin=E_fin,
             lam_fix=lam_fix, S_scan=S_scan, R2=R2,
             err_fix={f"{lam}_L{L}": err_fix[(lam, L)]
                      for lam in lam_fix for L in Ls},
             pp_L={f"{lam}_L{L}": pp_L[(lam, L)]
                   for lam in lam_fix for L in Ls},
             sreq_L={f"{lam}_L{L}": sreq_L[(lam, L)]
                     for lam in lam_fix for L in Ls})

    print("\nDone. Wrote layers_energy_vs_lambda.png, "
          "layers_shots_scaling.png, layers_sample_requirement.png, "
          "layers_data.npz")


if __name__ == "__main__":
    main()
