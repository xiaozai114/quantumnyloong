"""
experiment3_shots.py -- How does the SQD energy depend on the number of
measurements S (shots)?

Theory
------
SQD projects the Hamiltonian onto the subspace of determinants that appear in
the *samples*.  Its accuracy has TWO independent error sources:

  (A) SUPPORT error (subspace incomplete).  To recover the FCI energy the
      sampled pool must contain every physically relevant determinant -- here
      the HF reference plus the 3 paired doubles.  Each paired double has
      probability p_paired, so catching all of them requires
            S_req ~ 3 / p_paired        (coupon-collector, 95% conf)
      For S << S_req the pool is missing at least one paired double; the
      variational energy stays HIGH (close to the "HF + partial support"
      level).  Whether a given draw *hits* FCI is a Bernoulli event, so the
      error distribution is bimodal: either ~0 (hit) or large (miss).

  (B) SAMPLING (statistical) error, dominant for S >> S_req.  Even with full
      support, each determinant's amplitude is estimated from its empirical
      frequency, with relative error ~ 1/sqrt(S * p_d).  The perturbed
      subspace Hamiltonian gives an energy error ~ O(1/sqrt(S)).

Hence E_SQD(S) versus log S looks like:
    * a high plateau for S << S_req,
    * a sharp drop (the hit-rate sigmoid) around S ~ S_req,
    * a slow ~ 1/sqrt(S) statistical tail for S >> S_req.

We verify this by scanning S for two lambda values (lam = 1.0, easy;
lam = 0.3, hard) with many independent repetitions per S.

Run:  python experiment3_shots.py
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
)

HARTREE_TO_MHA = 1000.0
CONF = 0.95
EPS_HIT = 1e-3  # Ha; |E_SQD - E_FCI| < EPS_HIT counts as "hit FCI"


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
    _, t2_act, _ = ccsd_t2_active(data)
    layers = df_layers(data, t2_act)
    print(f"E_HF  = {e_hf:.8f} Ha   E_FCI = {e_fci:.8f} Ha")

    # measurement counts to scan (log-spaced)
    S_list = np.array([200, 500, 1000, 2000, 5000, 1e4, 2e4, 5e4, 1e5],
                      dtype=float)
    R = 40  # repetitions per S point

    results = {}
    for lam in (1.0, 0.3):
        psi = state_of(build_multilayer_circuit(data, layers, lam, n_layers=3))
        probs = np.abs(psi) ** 2
        probs /= probs.sum()
        p_hf, p_paired = paired_double_probability(data, psi)
        s_req = sreq(p_paired)
        print(f"\nlam = {lam}:  p_paired = {p_paired:.3e}   "
              f"S_req(95%) = {s_req:.0f}")

        err = np.zeros((len(S_list), R))   # (E - E_FCI) in mHa
        for i, S in enumerate(S_list):
            for r in range(R):
                rng = np.random.default_rng(int(lam * 1e6) + i * R + r)
                bits = draw_bits(probs, int(S), nq, rng)
                E = sqd_energy(bits, data)["energy"]
                err[i, r] = (E - e_fci) * HARTREE_TO_MHA
        results[lam] = dict(err=err, p_paired=p_paired, s_req=s_req)

    # -------------------------------------------------------------- figure
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    colors = {1.0: "tab:blue", 0.3: "tab:orange"}
    for lam in (1.0, 0.3):
        res = results[lam]
        err = res["err"]
        med = np.median(err, axis=1)
        lo = np.percentile(err, 10, axis=1)
        hi = np.percentile(err, 90, axis=1)
        hit = np.mean(np.abs(err) < EPS_HIT * HARTREE_TO_MHA, axis=1) * 100.0
        c = colors[lam]

        # (0,0): error vs S (log-log) with percentile band
        ax = axes[0, 0]
        ax.fill_between(S_list, lo, hi, color=c, alpha=0.20)
        ax.semilogx(S_list, med, "o-", color=c, label=f"lam = {lam}")
        ax.axvline(res["s_req"], color=c, ls=":", lw=1.2,
                   label=f"S_req(lam={lam}) = {res['s_req']:.0f}")
        ax.set_xlabel("measurements S (shots)")
        ax.set_ylabel(r"$|E_{\mathrm{SQD}}-E_{\mathrm{FCI}}|$ (mHa)")
        ax.set_title("SQD error vs S  (median + 10/90 pct)")
        ax.grid(alpha=0.3, which="both")

        # (0,1): hit rate vs S
        ax = axes[0, 1]
        ax.semilogx(S_list, hit, "s-", color=c, label=f"lam = {lam}")
        ax.axvline(res["s_req"], color=c, ls=":", lw=1.2)
        ax.set_xlabel("measurements S (shots)")
        ax.set_ylabel("hit-FCI rate (%)")
        ax.set_title("Probability of recovering FCI")
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3, which="both")

    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)

    # (1,0): error histograms at three S for lam = 1.0
    ax = axes[1, 0]
    lam = 1.0
    err = results[lam]["err"]
    for S, c in [(1e3, "tab:red"), (1e4, "tab:green"), (5e4, "tab:purple")]:
        i = np.argmin(np.abs(S_list - S))
        ax.hist(err[i], bins=20, alpha=0.55, density=True,
                label=f"S = {int(S):.0f}")
    ax.axvline(EPS_HIT * HARTREE_TO_MHA, color="k", ls="--", lw=1.0,
               label="hit threshold")
    ax.set_xlabel(r"$E_{\mathrm{SQD}}-E_{\mathrm{FCI}}$ (mHa)")
    ax.set_ylabel("density")
    ax.set_title("Error distribution (lam = 1.0)")
    ax.legend(fontsize=8)

    # (1,1): empirical tail vs 1/sqrt(S) for the large-S regime
    ax = axes[1, 1]
    for lam in (1.0, 0.3):
        res = results[lam]
        err = res["err"]
        # use the 90th percentile as a robust "worst typical" error
        p90 = np.percentile(np.abs(err), 90, axis=1)
        ax.loglog(S_list, p90, "o-", color=colors[lam], label=f"lam = {lam}")
        # reference slope -1/2
    ss = np.array([S_list[0], S_list[-1]])
    ax.loglog(ss, p90[-1] * (ss / S_list[-1]) ** 0.5, "k--", lw=1.0,
              label=r"$\propto S^{-1/2}$")
    ax.set_xlabel("measurements S (shots)")
    ax.set_ylabel(r"90th-pct $|E_{\mathrm{SQD}}-E_{\mathrm{FCI}}|$ (mHa)")
    ax.set_title("Statistical tail scaling")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig("shots_scaling.png", dpi=160)
    plt.close(fig)

    np.savez("shots_scaling_data.npz",
             S_list=S_list, R=R, eps_hit=EPS_HIT,
             lam1_err=results[1.0]["err"],
             lam1_p_paired=results[1.0]["p_paired"],
             lam1_s_req=results[1.0]["s_req"],
             lam03_err=results[0.3]["err"],
             lam03_p_paired=results[0.3]["p_paired"],
             lam03_s_req=results[0.3]["s_req"])

    # -------------------------------------------------------------- print
    print("\n" + "=" * 60)
    print("SUMMARY  (error in mHa; hit = |err| < 1 mHa)")
    print("=" * 60)
    print(f"{'S':>8s} | {'lam=1.0 med':>11s} {'hit%':>6s} "
          f"| {'lam=0.3 med':>11s} {'hit%':>6s}")
    for i, S in enumerate(S_list):
        m1 = np.median(results[1.0]["err"][i]) * 1.0
        h1 = np.mean(np.abs(results[1.0]["err"][i]) < EPS_HIT * HARTREE_TO_MHA) * 100
        m3 = np.median(results[0.3]["err"][i])
        h3 = np.mean(np.abs(results[0.3]["err"][i]) < EPS_HIT * HARTREE_TO_MHA) * 100
        print(f"{int(S):>8d} | {m1:>11.3f} {h1:>5.0f}% "
              f"| {m3:>11.3f} {h3:>5.0f}%")

    print("\nDone. Wrote shots_scaling.png, shots_scaling_data.npz")


if __name__ == "__main__":
    main()
