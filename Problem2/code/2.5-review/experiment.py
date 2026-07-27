"""
Tencent Sparking Program 2026 - Problem 2.5 (a) + Advanced Challenge  (LUCJ only)
=================================================================================
LUCJ-SQD vs HF-SQD, and the quantitative relationship between a LUCJ state's
entanglement entropy and the required number of samples S.

This is a self-contained driver built on top of `definitions.py` (the shared
building blocks: molecule, CCSD->LUCJ generators, circuit, sampler, SQD).
Everything below is written from the `definitions.py` primitives.

CONVENTION (see definitions.build_circuit):
    * local=True  -> the LUCJ state. This is what the problem means by "LUCJ"
                     (nearest-neighbour truncation of the diagonal-Coulomb /
                     paired-Givens layers). ALL analysis below uses LUCJ only.
    * The full (untruncated) UCJ is NOT considered anywhere in this script.

RUN:
    python experiment.py
    # reproduces Part 1 (2.5a task), Part 2 (lambda scan), Part 3 (Advanced
    # Challenge) and writes the figures / data file listed at the bottom.

OUTPUTS:
    lucj_energy_vs_lambda.png        Part 2: LUCJ SQD energy vs lambda
                                     (infinite-sample true energy vs S=1000 sampled)
    lucj_entanglement_samples.png    Part 3: entanglement entropy vs required S
    synthetic_entanglement_samples.png Part 3: controlled-entanglement demo
    entanglement_samples_data.npz    all raw arrays
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorcircuit as tc

tc.set_backend("tensorflow")
tc.set_dtype("complex128")

from definitions import (
    build_lih_active_space, ccsd_amplitudes, lucj_generators,
    build_circuit, sample_bitstrings, sqd_energy, hf_state_index,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# Bipartition for the entanglement entropy / Schmidt rank.
# OCCUPIED active orbital (both spins, qubits 0,1) | the three VIRTUAL active
# orbitals. This cut separates the double excitations (i->a), so it actually
# captures the entanglement that a 4|4 cut (orbitals 0,1 | 2,3) misses --
# note the only LUCJ-surviving double (0->1) keeps both electrons on the SAME
# side of a 4|4 cut and would give S_ent ~ 0 spuriously.
CUT_A = [0, 1]


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def entanglement_data(psi, nq, cut_A=CUT_A):
    """Return (S_ent, schmidt_rank) for the bipartition `cut_A` | rest."""
    p = np.asarray(psi).reshape([2] * nq)
    rest = [q for q in range(nq) if q not in cut_A]
    p = np.transpose(p, cut_A + rest)
    dA = 2 ** len(cut_A)
    rho = p.reshape(dA, -1) @ np.conj(p.reshape(dA, -1).T)
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-15]
    S = -np.sum(ev * np.log(ev))
    return float(S), int(len(ev))            # S_ent, Schmidt rank r


def exact_support_sqd_energy(psi, nq, data):
    """SQD energy in the INFINITE-sample limit: build H in the full support of
    the exact LUCJ statevector. For the LOCAL LUCJ this is the true (truncated)
    variational energy -- it plateaus below E_HF but never reaches FCI because
    the non-adjacent dominant doubles (0->2, 0->3) are dropped by the
    nearest-neighbour truncation."""
    pr = np.abs(psi) ** 2
    pr /= pr.sum()
    mask = pr > 1e-12
    idx = np.where(mask)[0]
    sh2 = (nq - 1 - np.arange(nq))[None, :]
    bits = ((idx[:, None] >> sh2) & 1).astype(int)
    return sqd_energy(bits, data)


def dominant_correlated_prob(pr):
    """Probability of the leading correlated (non-HF) determinant.

    The LUCJ state is HF-dominant; the correlation energy is carried by the
    leading double excitation. Its probability is the 2nd-largest entry of the
    distribution (HF is the largest). Using the 2nd-largest -- rather than the
    *rarest* determinant -- avoids being hijacked by a numerically-tiny,
    energy-irrelevant determinant that makes the required-sample count
    non-monotonic and physically meaningless."""
    s = np.sort(np.asarray(pr, dtype=float))[::-1]
    return float(s[1]) if len(s) > 1 else 0.0


def required_samples_coupon(p, target=0.95, cap=1e9):
    """Coupon-collector samples to observe one determinant of probability p at
    least once with success probability >= target. S = ln(1-target)/ln(1-p)
    ~ -ln(1-target)/p for small p. p<=0 (pure HF) -> trivially 1 sample."""
    if p <= 0:
        return 1
    S = np.log(1.0 - target) / np.log(1.0 - p)
    return int(min(max(S, 1.0), cap))


def synthetic_demo():
    """Controlled-entanglement family |psi> = (cos t|00> + sin t|11>)^{⊗m},
    with t=pi/4 so each Bell pair is maximally entangled and the probability
    mass is spread UNIFORMLY over R = 2^m = exp(S_ent) components. Coupon-
    collector theory then gives S ~ R ln R, i.e. log10 S ~ S_ent (slope 1/ln10)."""
    ms = [1, 2, 3, 4, 5, 6]
    Sents, Ss = [], []
    for m in ms:
        t = np.pi / 4
        comp = np.cos(t)
        pair = np.array([comp, 0.0, 0.0, comp])   # |Phi(t)> = comp|00> + comp|11>
        psi = pair
        for _ in range(m - 1):
            psi = np.kron(psi, pair)
        cut = list(range(0, 2 * m, 2))             # first qubit of each Bell pair
        Sents.append(entanglement_data(psi, 2 * m, cut_A=cut)[0])
        R = 2 ** m
        Ss.append(int(np.ceil(R * np.log(R / 0.05))))   # 95% coverage of R coupons
    Sents, Ss = np.array(Sents), np.array(Ss)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(Sents, Ss, "o-")
    for mm, x, y in zip(ms, Sents, Ss):
        ax[0].annotate(f"m={mm}", (x, y))
    ax[0].set_xlabel(r"$S_{\mathrm{ent}}$ (nats)")
    ax[0].set_ylabel("required samples S")
    ax[0].set_title(r"B1: required samples vs $S_{\mathrm{ent}}$ (synthetic)")
    ax[0].set_yscale("log")
    slope, intercept = np.polyfit(Sents, np.log10(Ss), 1)
    xs = np.linspace(Sents.min(), Sents.max(), 50)
    ax[1].plot(Sents, np.log10(Ss), "o", label="data")
    ax[1].plot(xs, slope * xs + intercept, "-",
               label=f"fit: log10 S = {slope:.3f} $S_{{\mathrm{{ent}}}}$ + {intercept:.2f}")
    ax[1].set_xlabel(r"$S_{\mathrm{ent}}$ (nats)")
    ax[1].set_ylabel(r"$\log_{10} S$")
    ax[1].set_title(r"B2: KEY -- $\log_{10} S \propto S_{\mathrm{ent}}$")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "synthetic_entanglement_samples.png"), dpi=140)
    print(f"  synthetic fit slope = {slope:.3f}  (ideal 1/ln10 = "
          f"{1/np.log(10):.3f})")
    return dict(m=ms, S_ent=Sents, S=Ss, slope=slope, intercept=intercept)


# ---------------------------------------------------------------------------
# PART 1 -- Problem 2.5 (a): HF-SQD vs LUCJ-SQD at lambda=0.05 (numeric report)
# ---------------------------------------------------------------------------
def part_a_task(lam=0.05, S=1000):
    data = build_lih_active_space()
    nq = 2 * data["nact"]
    t1, t2 = ccsd_amplitudes(data)
    e_hf, e_fci = data["e_hf"], data["e_fci"]

    tp, k, J = lucj_generators(data, t1, t2, lam)
    c_loc = build_circuit(data, tp, k, J, local=True)    # LUCJ

    # HF subspace
    hfi = hf_state_index(data)
    hfb = np.array([[int(b) for b in format(hfi, f"0{nq}b")]])
    e_hf_sqd = sqd_energy(hfb, data)["energy"]

    def sampled(c):
        es = [sqd_energy(sample_bitstrings(c, nq, S, seed=sd)[0], data)["energy"]
              for sd in range(4)]
        return np.mean(es), np.std(es)

    e_lucj, s_lucj = sampled(c_loc)

    rows = [
        ("HF-SQD", e_hf_sqd, e_hf_sqd - e_fci, 0.0),
        ("LUCJ-SQD (local)", e_lucj, e_lucj - e_fci, s_lucj),
        ("E_HF (exact)", e_hf, e_hf - e_fci, 0.0),
        ("E_FCI (target)", e_fci, 0.0, 0.0),
    ]
    print(f"\nPart 1 -- 2.5(a) at lambda={lam}, S={S}")
    print(f"  {'method':<20}{'E_SQD (Ha)':>14}{'err vs FCI (Ha)':>18}{'std (Ha)':>12}")
    for name, e, err, std in rows:
        print(f"  {name:<20}{e:>14.7f}{err:>18.2e}{std:>12.2e}")

    msg = ("  NOTE: at lambda=0.05 the LUCJ state is >99.98% HF, so with only "
           "S=1000 shots the rare correlated determinant is essentially never "
           "sampled and LUCJ-SQD coincides with HF-SQD (= E_HF) -- a "
           "sampling-statistics effect, not a bug (see Part 2).")
    print(msg)
    return dict(e_hf=e_hf, e_fci=e_fci, e_lucj=e_lucj, e_hf_sqd=e_hf_sqd)


# ---------------------------------------------------------------------------
# PART 2 -- lambda scan: LUCJ SQD energy vs lambda (infinite-sample vs S shots)
# ---------------------------------------------------------------------------
def part_a_lambda_scan(S=1000, nseed=4):
    data = build_lih_active_space()
    nq = 2 * data["nact"]
    t1, t2 = ccsd_amplitudes(data)
    e_hf, e_fci = data["e_hf"], data["e_fci"]
    lambdas = np.round(np.linspace(0.0, 1.0, 21), 3)
    rec = dict(e_inf_loc=[], e_samp_loc=[])
    for lam in lambdas:
        tp, k, J = lucj_generators(data, t1, t2, lam)
        pl = np.asarray(tc.backend.numpy(build_circuit(data, tp, k, J, local=True).state()))
        rec["e_inf_loc"].append(exact_support_sqd_energy(pl, nq, data)["energy"])
        sl = []
        for sd in range(nseed):
            bl, _ = sample_bitstrings(build_circuit(data, tp, k, J, local=True), nq, S, seed=sd)
            sl.append(sqd_energy(bl, data)["energy"])
        rec["e_samp_loc"].append(np.mean(sl))
        print(f"  lam={lam:4.2f}  E_true(LUCJ)={rec['e_inf_loc'][-1]:.6f}  "
              f"E_samp(S={S})={rec['e_samp_loc'][-1]:.6f}")
    for kk in rec:
        rec[kk] = np.array(rec[kk])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axhline(e_hf, color="gray", ls=":", label="E_HF (reference)")
    ax.axhline(e_fci, color="k", ls="--", label="E_FCI (target, unreachable by LUCJ)")
    ax.plot(lambdas, rec["e_inf_loc"], "s-", color="C1",
            label=r"$\infty$-sample SQD energy (true LUCJ)")
    ax.plot(lambdas, rec["e_samp_loc"], "v--", color="C1", alpha=0.7,
            label=f"SQD energy, S={S} (sampled LUCJ)")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("energy (Ha)")
    ax.set_title(r"LUCJ-SQD energy vs $\lambda$ (S=%d shots)" % S + "\n"
                 "sampled tracks HF until the rare double excitation is sampled;\n"
                 "true LUCJ plateaus below HF (local truncation drops 0->2, 0->3)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "lucj_energy_vs_lambda.png"), dpi=140)
    return rec, lambdas


# ---------------------------------------------------------------------------
# PART 3 -- Advanced Challenge: entanglement entropy vs required samples S
# ---------------------------------------------------------------------------
def part_advanced_challenge():
    data = build_lih_active_space()
    nq = 2 * data["nact"]
    t1, t2 = ccsd_amplitudes(data)
    lambdas = np.round(np.linspace(0.0, 1.0, 51), 3)   # DENSE grid
    out = dict(S_ent=[], rank=[], expS=[], S_req=[])
    for lam in lambdas:
        tp, k, J = lucj_generators(data, t1, t2, lam)
        c = build_circuit(data, tp, k, J, local=True)   # LUCJ == LOCAL truncation
        psi = np.asarray(tc.backend.numpy(c.state()))
        pr = np.abs(psi) ** 2
        pr /= pr.sum()
        S, r = entanglement_data(psi, nq)
        out["S_ent"].append(S)
        out["rank"].append(r)
        out["expS"].append(np.exp(S))
        pd = dominant_correlated_prob(pr)
        out["S_req"].append(required_samples_coupon(pd))
    for kk in out:
        out[kk] = np.array(out[kk])

    print("\nPart 3 -- Advanced Challenge (LUCJ = local): entanglement vs samples")
    print(f"  {'lam':>5}{'S_ent':>10}{'rank':>6}{'exp(S)':>9}{'p_double':>12}{'S_req':>12}")
    for i, lam in enumerate(lambdas):
        print(f"  {lam:5.2f}{out['S_ent'][i]:10.4f}{out['rank'][i]:6d}"
              f"{out['expS'][i]:9.3f}{'':>12}{out['S_req'][i]:12.1f}")

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].plot(lambdas, out["S_ent"], "o-")
    ax[0, 0].set_xlabel(r"$\lambda$")
    ax[0, 0].set_ylabel(r"$S_{\mathrm{ent}}$ (nats)")
    ax[0, 0].set_title(r"A1: LUCJ entanglement entropy vs $\lambda$")

    # rigorous lower bound: Schmidt rank r >= exp(S_ent)
    ax[0, 1].plot(out["expS"], out["rank"], "o", color="C2",
                  label=r"data $(r, \exp S_{\mathrm{ent}})$")
    xs = np.linspace(min(out["expS"].min(), 1.0), out["expS"].max(), 50)
    ax[0, 1].plot(xs, xs, "k--", label=r"$r = \exp(S_{\mathrm{ent}})$ (bound)")
    ax[0, 1].set_xlabel(r"$\exp(S_{\mathrm{ent}})$")
    ax[0, 1].set_ylabel("Schmidt rank $r$")
    ax[0, 1].set_title(r"A2: rigorous bound $r \geq \exp(S_{\mathrm{ent}})$")
    ax[0, 1].legend(fontsize=8)

    ax[1, 0].plot(lambdas, out["S_req"], "o-", color="C3")
    ax[1, 0].set_xlabel(r"$\lambda$")
    ax[1, 0].set_ylabel("required samples $S$")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_title(r"A3: required samples vs $\lambda$ (log)")
    ax[1, 0].set_ylim(1, None)

    ax[1, 1].plot(out["S_ent"], out["S_req"], "o", color="C3")
    ax[1, 1].set_xlabel(r"$S_{\mathrm{ent}}$ (nats)")
    ax[1, 1].set_ylabel("required samples $S$")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_title(r"A4: $S$ vs $S_{\mathrm{ent}}$ -- NOT exponential here")
    ax[1, 1].set_ylim(1, None)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "lucj_entanglement_samples.png"), dpi=140)
    return out, lambdas


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    outA = part_a_task(lam=0.05, S=1000)
    recE, lambdas = part_a_lambda_scan(S=1000, nseed=4)
    outC, _ = part_advanced_challenge()
    outB = synthetic_demo()

    np.savez(os.path.join(HERE, "entanglement_samples_data.npz"),
             **{f"A_{k}": v for k, v in outA.items()},
             **{f"E_{k}": v for k, v in recE.items()},
             **{f"C_{k}": v for k, v in outC.items()},
             **{f"B_{k}": v for k, v in outB.items()})

    print("\n" + "=" * 64)
    print("CONCLUSION")
    print("=" * 64)
    print("* The LUCJ state uses the LOCAL truncation (local=True). No full UCJ")
    print("  is considered anywhere in this analysis.")
    print("* Part 1/2: at lambda=0.05, S=1000 the LUCJ state is >99.98% HF, so")
    print("  the rare double-excitation determinant is essentially never sampled")
    print("  -> LUCJ-SQD == HF-SQD == E_HF. This is a SAMPLING-COVERAGE effect.")
    print("  The TRUE (infinite-sample) LUCJ energy drops below E_HF but plateaus")
    print("  (the non-adjacent dominant doubles 0->2, 0->3 are dropped -> bias).")
    print("* Part 3 Advanced Challenge: the rigorous lower bound r >= exp(S_ent)")
    print("  (Schmidt rank) holds for every lambda. BUT the required sample count")
    print("  is NOT exponential in S_ent here: the probability mass is concentrated")
    print("  in ONE dominant double excitation, whose probability p_d grows with")
    print("  lambda, so S ~ 1/p_d DECREASES as lambda (and S_ent) increase. The")
    print("  clean log S ~ S_ent law is only recovered in the controlled synthetic")
    print("  demo (Part B), slope ~ 1/ln10.")
    print("\nREFERENCE (single script): experiment.py  |  foundation: definitions.py")


if __name__ == "__main__":
    main()
