#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnostic for the MULTI-LAYER (double-factorized) LUCJ ansatz.

    |Psi_LUCJ> = ( prod_k U_k e^{i J_k} U_k^dagger ) |HF>

We build the layers from the CCSD t2 double factorization, then sweep lambda
in (0, 1] and report, for each lambda:
  * number of unique determinants sampled (raw, before recovery)
  * whether the PAIRED double excitations |a,abar> appear in the samples
  * HF-SQD and LUCJ-SQD energies and their errors vs FCI (CASCI)
  * probability weight the exact state puts on paired doubles

This directly checks the two claims:
  (a) at lambda = 0.05 the LUCJ result is still indistinguishable from HF
  (b) as lambda -> 1 the multi-layer LUCJ injects paired doubles and the
      LUCJ-SQD energy approaches FCI.
"""

import numpy as np
import tensorcircuit as tc
from pyscf import gto, scf, cc, mcscf, ao2mo

import problem_2_5_lucj_sqd as P

tc.set_backend("numpy")
tc.set_dtype("complex128")
np.random.seed(1234)

BOND = 1.55
NCAS = 4
NELECAS = (1, 1)
N_SAMPLES = 200000


def paired_double_prob(c, ncas, nelecas):
    """Total probability the exact statevector puts on the PAIRED double
    excitations |a, abar> (both electrons promoted to the same virtual a),
    and on the HF reference |0, 0bar>."""
    nq = 2 * ncas
    psi = np.asarray(c.state())
    p = np.abs(psi) ** 2
    nocc = nelecas[0]

    def occ_index(alpha_occ, beta_occ):
        # q0 = MSB;  alpha block qubits 0..ncas-1, beta block ncas..2ncas-1
        bits = [0] * nq
        for o in alpha_occ:
            bits[o] = 1
        for o in beta_occ:
            bits[ncas + o] = 1
        idx = 0
        for b in bits:
            idx = (idx << 1) | b
        return idx

    p_hf = p[occ_index((0,), (0,))]
    p_paired = 0.0
    for a in range(nocc, ncas):
        p_paired += p[occ_index((a,), (a,))]
    return p_hf, p_paired, p.sum()


def main():
    mol = gto.M(atom=f"Li 0 0 0; H 0 0 {BOND}", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    e_hf = mf.e_tot

    h1e, eri, ecore, ncore, mc = P.active_space_integrals(mol, mf, NCAS, NELECAS)
    e_cas = mc.kernel()[0]

    # CCSD confined to the exact active space, then DF into layers
    mycc = P.ccsd_active(mf, NCAS, NELECAS, ncore)
    layers = P.double_factorization_layers(mycc, NCAS, NELECAS)

    print("=" * 78)
    print("MULTI-LAYER (double-factorized) LUCJ diagnostic  --  LiH STO-3G")
    print("=" * 78)
    print(f"E(HF)   = {e_hf:.8f} Ha")
    print(f"E(FCI)  = {e_cas:.8f} Ha   (CASCI in the {NCAS}-orbital space)")
    print(f"E_corr(CCSD, active) = {mycc.e_corr:.8f} Ha")
    print(f"Number of DF layers L = {len(layers)}")
    for k, (g, kappa) in enumerate(layers):
        occv = kappa[0, NELECAS[0]:]      # occ-vir coupling row
        print(f"  layer {k}: g_k = {g:+.6f}   occ-vir direction = "
              f"{np.round(occv, 4)}")
    print("-" * 78)

    # HF-SQD baseline (sample the plain HF circuit)
    nq = 2 * NCAS
    c_hf = P.hf_circuit(NCAS, NELECAS)
    s_hf = P.sample_bitstrings(c_hf, nq, N_SAMPLES)
    s_hf = P.configuration_recovery(s_hf, NCAS, NELECAS)
    dets_hf = P.unique_determinants(s_hf, NCAS)
    e_sqd_hf, m_hf = P.sqd_energy(dets_hf, h1e, eri, ecore, NCAS, NELECAS)

    print(f"HF-SQD:  M={m_hf}  E={e_sqd_hf:.8f}  "
          f"err_vs_FCI={abs(e_sqd_hf - e_cas)*1e3:.4f} mHa")
    print("-" * 78)
    # Variational FLOOR: SQD energy of the subspace {HF} + all paired doubles
    # {|a,abar>}, i.e. the best the multi-layer ansatz can do once it injects
    # any paired-double support at all (independent of sampling luck).
    floor_dets = [((0,), (0,))] + [((a,), (a,)) for a in range(NELECAS[0], NCAS)]
    e_floor, m_floor = P.sqd_energy(floor_dets, h1e, eri, ecore, NCAS, NELECAS)
    print(f"Variational floor  (HF + all paired doubles, M={m_floor}): "
          f"E={e_floor:.8f}  err_vs_FCI={abs(e_floor - e_cas)*1e3:.4f} mHa")
    print("-" * 78)

    print(f"{'lambda':>8} {'L_used':>6} {'p_paired':>12} {'#uniq_raw':>10} "
          f"{'paired?':>8} {'M':>4} {'E_LUCJ-SQD':>14} {'err(mHa)':>10} "
          f"{'dE_vs_HF(uHa)':>14}")

    lambdas = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]
    rows = []
    for lam in lambdas:
        c = P.lucj_circuit_multilayer(NCAS, NELECAS, layers, lam)
        p_hf, p_paired, _ = paired_double_prob(c, NCAS, NELECAS)

        s = P.sample_bitstrings(c, nq, N_SAMPLES)
        raw = P.unique_determinants(s, NCAS)
        # detect paired doubles among raw samples
        paired_seen = any(
            (len(a) == 1 and len(b) == 1 and a[0] == b[0] and a[0] >= NELECAS[0])
            for (a, b) in raw
        )
        s_fix = P.configuration_recovery(s, NCAS, NELECAS)
        dets = P.unique_determinants(s_fix, NCAS)
        e_lucj, m = P.sqd_energy(dets, h1e, eri, ecore, NCAS, NELECAS)

        err = abs(e_lucj - e_cas) * 1e3
        dvs = abs(e_lucj - e_sqd_hf) * 1e6
        print(f"{lam:8.3f} {len(layers):6d} {p_paired:12.3e} {len(raw):10d} "
              f"{str(paired_seen):>8} {m:4d} {e_lucj:14.8f} {err:10.4f} "
              f"{dvs:14.2f}")
        rows.append(dict(lam=lam, p_paired=p_paired, n_uniq=len(raw),
                         paired_seen=paired_seen, M=m, e_lucj=e_lucj,
                         err_mHa=err, dE_vs_hf_uHa=dvs))

    print("=" * 78)
    import json
    with open("multilayer_scan.json", "w") as f:
        json.dump(dict(e_hf=e_hf, e_cas=e_cas, e_sqd_hf=e_sqd_hf,
                       e_corr=mycc.e_corr, L=len(layers), rows=rows), f, indent=2)
    print("Saved -> multilayer_scan.json")


if __name__ == "__main__":
    main()
