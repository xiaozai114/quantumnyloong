#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Problem 2.5 -- LUCJ-SQD vs HF-SQD for LiH (STO-3G).  Clean version.

Pipeline
--------
  1. Solve LiH HF; define an active space of 4 spatial orbitals / 2 electrons
     (freeze the Li-1s core), i.e. 8 qubits under Jordan-Wigner.
  2. Run CCSD in that same active space and double-factorize its paired-double
     t2 channel into L layers  (g_k, v_k).
  3. Build the multi-layer LUCJ state
         |Psi> = ( prod_k  U_k e^{i J_k} U_k^dagger ) |HF>
     with the CCSD physics carried by the Jastrow phase (phi_k = g_k * lambda)
     and a FIXED O(1) orbital-rotation angle.  This is the one construction that
     can actually inject paired double excitations into |HF>.
  4. Sample both circuits, restore particle number, diagonalize in the sampled
     determinant subspace (SQD), and compare to the exact FCI energy.

Key facts that make this work (learned the hard way):
  * A DIAGONAL Jastrow alone (single layer U e^{iJ} U^dag) cannot create a
    paired double from HF -- you need the multi-layer DF structure below.
  * The per-layer paired amplitude ~ sin^2(theta) * phi_k is LINEAR in the
    phase.  So keep theta fixed (pi/4) and put g_k*lambda into the phase;
    scaling theta with the tiny t2 weight instead would suppress it as t2^2.

Qubit layout (JW): q[p] = alpha orbital p, q[ncas+p] = beta orbital p.
Bit 1 = occupied.  q0 is the most-significant bit.
"""

import numpy as np
from scipy.linalg import expm
import tensorcircuit as tc
from pyscf import gto, scf, cc, mcscf, ao2mo
from pyscf.fci import cistring, direct_spin1

tc.set_backend("numpy")
tc.set_dtype("complex128")
np.random.seed(1234)

# --- problem settings ---
BOND = 1.55           # Angstrom
NCAS = 4              # active spatial orbitals -> 8 qubits
NELECAS = (1, 1)      # (Nalpha, Nbeta) active electrons (Li-1s core frozen)
LAMBDA = 0.05         # ccsd_scale
N_SAMPLES = 1000
THETA = np.pi / 4     # fixed per-layer orbital-rotation angle


# ============================================================================
# 1. Molecule, active space, exact reference
# ============================================================================
def solve_molecule():
    """RHF + active-space integrals + exact (CASCI=FCI) reference energy."""
    mol = gto.M(atom=f"Li 0 0 0; H 0 0 {BOND}", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()

    ncore = (sum(mol.nelec) - sum(NELECAS)) // 2   # frozen doubly-occ core = 1
    mc = mcscf.CASCI(mf, NCAS, NELECAS)
    mc.verbose = 0
    h1e, ecore = mc.get_h1eff()                    # active 1e ints + core energy
    eri = ao2mo.restore(1, mc.get_h2eff(), NCAS)   # active 2e ints (chemist)
    e_fci = mc.kernel()[0]                          # CASCI = FCI in active space
    return mf, ncore, h1e, eri, ecore, e_fci


# ============================================================================
# 2. CCSD t2 -> double-factorized layers
# ============================================================================
def df_layers(mf, ncore):
    """Run CCSD in exactly the NCAS-orbital active space and double-factorize
    its paired-double t2 channel.

    For a single active occupied orbital (LiH: nocc=1) the paired amplitudes
    form the symmetric virtual-space matrix  T[a,b] = t2[0,0,a,b].  Its
    eigen-decomposition  T = sum_k g_k v_k v_k^T  gives one layer per eigenpair:
        g_k -> Jastrow phase strength,
        v_k -> virtual direction that U_k rotates the occupied orbital into.
    Returns a list of (g_k, kappa_k); kappa_k is the antisymmetric generator so
    U_k = exp(theta * kappa_k) mixes orbital 0 with virtual mode v_k.
    """
    nmo = mf.mo_coeff.shape[1]
    # freeze the core AND every virtual above the active window, so CCSD's
    # occ/vir dims coincide with the 8-qubit active space (nocc=1, nvir=3).
    frozen = list(range(ncore)) + list(range(ncore + NCAS, nmo))
    mycc = cc.CCSD(mf, frozen=frozen).run()

    nocc = NELECAS[0]
    nvir = NCAS - nocc
    T = np.asarray(mycc.t2[0, 0, :nvir, :nvir])
    T = 0.5 * (T + T.T)
    g_vals, vecs = np.linalg.eigh(T)

    layers = []
    for k in range(nvir):
        g = float(g_vals[k])
        if abs(g) < 1e-8:
            continue
        v = vecs[:, k]                       # virtual direction
        kappa = np.zeros((NCAS, NCAS))       # rotate orbital 0 <-> mode v
        for a in range(nvir):
            kappa[0, nocc + a] = v[a]
            kappa[nocc + a, 0] = -v[a]
        layers.append((g, kappa))
    return layers, mycc.e_corr


# ============================================================================
# 3. Gate primitives (Problem 2.4 decompositions) and circuits
# ============================================================================
def givens(c, qi, qj, theta):
    """Givens rotation on {|01>,|10>} of (qi,qj), realized with 2 CNOTs."""
    c.cnot(qj, qi)
    c.ry(qj, theta=theta)
    c.cnot(qi, qj)
    c.ry(qj, theta=-theta)
    c.cnot(qi, qj)
    c.cnot(qj, qi)


def jastrow_pair(c, qi, qj, phi):
    """e^{i phi n_i n_j} = RZ_i(phi/2) RZ_j(phi/2) RZZ_ij(-phi/2), up to phase.
    RZZ_ij(a) = CNOT_ij RZ_j(a) CNOT_ij  (Problem 2.4)."""
    c.rz(qi, theta=phi / 2)
    c.rz(qj, theta=phi / 2)
    c.cnot(qi, qj)
    c.rz(qj, theta=-phi / 2)
    c.cnot(qi, qj)


def givens_network(kappa):
    """Decompose U = exp(kappa) (real antisymmetric) into a list of Givens
    rotations (p, q, theta) via QR-style elimination of Q = exp(kappa)."""
    Q = expm(kappa)
    n = Q.shape[0]
    M = Q.copy()
    rots = []
    for col in range(n):
        for row in range(n - 1, col, -1):
            a, b = M[row - 1, col], M[row, col]
            if abs(b) < 1e-14:
                continue
            theta = np.arctan2(b, a)
            crot = np.eye(n)
            cc_, ss_ = np.cos(theta), np.sin(theta)
            crot[row - 1, row - 1] = cc_; crot[row - 1, row] = ss_
            crot[row, row - 1] = -ss_;    crot[row, row] = cc_
            M = crot @ M
            rots.append((row - 1, row, theta))
    return [(p, q, -th) for (p, q, th) in reversed(rots)]


def hf_circuit():
    """|HF>: fill the lowest Nalpha alpha and Nbeta beta orbitals."""
    c = tc.Circuit(2 * NCAS)
    na, nb = NELECAS
    for p in range(na):
        c.x(p)
    for p in range(nb):
        c.x(NCAS + p)
    return c


def lucj_circuit(layers, lam, theta=THETA):
    """Multi-layer double-factorized LUCJ:
        |Psi> = ( prod_k  U_k e^{i J_k} U_k^dagger ) |HF>.

    Per layer: rotate occ->virtual mode by a FIXED angle theta in both spin
    blocks (U_k^dagger), stamp the opposite-spin Jastrow phase phi_k=g_k*lam on
    the doubly-occupied-virtual component, rotate back (U_k).  The phase breaks
    the otherwise-perfect U/U^dagger cancellation, leaving a paired double whose
    amplitude ~ sin^2(theta) * phi_k is linear in g_k*lam.
    """
    c = hf_circuit()
    for (g, kappa) in layers:
        phi = g * lam
        net = givens_network(theta * kappa)      # U_k gates per spin block

        def U_dag(off):
            for (p, q, th) in reversed(net):
                givens(c, off + p, off + q, -th)

        def U(off):
            for (p, q, th) in net:
                givens(c, off + p, off + q, th)

        U_dag(0); U_dag(NCAS)                     # U_k^dagger (alpha, beta)
        jastrow_pair(c, 0, NCAS + 0, phi)         # e^{i J_k} (opp-spin, occ-occ)
        U(0); U(NCAS)                             # U_k       (alpha, beta)
    return c


# ============================================================================
# 4. Sampling + configuration recovery + SQD
# ============================================================================
def sample(c, n_samples):
    """Draw bitstrings from the exact state-vector distribution |<x|psi>|^2."""
    nq = 2 * NCAS
    p = np.abs(np.asarray(c.state())) ** 2
    p /= p.sum()
    idx = np.random.choice(p.shape[0], size=n_samples, p=p)
    bits = np.zeros((n_samples, nq), dtype=int)
    for k in range(nq):
        bits[:, k] = (idx >> (nq - 1 - k)) & 1
    return bits


def recover(samples):
    """Restore Nalpha/Nbeta per spin block by flipping the bits that deviate
    most from the average occupancy (add high-occupancy 0s, drop low-occ 1s)."""
    na, nb = NELECAS
    avg = samples.mean(axis=0)
    out = samples.copy()

    def fix(bits, a_blk, target):
        bits = bits.copy()
        while bits.sum() > target:                # too many -> drop a 1
            ones = np.where(bits == 1)[0]
            bits[ones[np.argmin(a_blk[ones])]] = 0
        while bits.sum() < target:                # too few -> add a 1
            zeros = np.where(bits == 0)[0]
            bits[zeros[np.argmax(a_blk[zeros])]] = 1
        return bits

    for s in range(out.shape[0]):
        out[s, :NCAS] = fix(out[s, :NCAS], avg[:NCAS], na)
        out[s, NCAS:] = fix(out[s, NCAS:], avg[NCAS:], nb)
    return out


def unique_dets(samples):
    """Collect unique (alpha_string, beta_string) occupied-index tuples."""
    dets = set()
    for s in samples:
        a = tuple(np.where(s[:NCAS] == 1)[0])
        b = tuple(np.where(s[NCAS:] == 1)[0])
        dets.add((a, b))
    return sorted(dets)


def sqd_energy(dets, h1e, eri, ecore):
    """Build H restricted to the sampled determinant subspace and diagonalize."""
    na, nb = NELECAS
    stra = cistring.make_strings(range(NCAS), na)
    strb = cistring.make_strings(range(NCAS), nb)
    addr_a = {s: i for i, s in enumerate(stra)}
    addr_b = {s: i for i, s in enumerate(strb)}

    def to_str(occ):
        s = 0
        for o in occ:
            s |= (1 << o)
        return s

    idx = [(addr_a[to_str(a)], addr_b[to_str(b)]) for (a, b) in dets
           if to_str(a) in addr_a and to_str(b) in addr_b]
    if not idx:
        return None, 0

    dimA, dimB = len(stra), len(strb)
    ham = direct_spin1.absorb_h1e(h1e, eri, NCAS, NELECAS, 0.5)
    M = len(idx)
    H = np.zeros((M, M))
    for col, (ia, ib) in enumerate(idx):
        vec = np.zeros((dimA, dimB)); vec[ia, ib] = 1.0
        hc = direct_spin1.contract_2e(ham, vec, NCAS, NELECAS)
        for row, (ja, jb) in enumerate(idx):
            H[row, col] = hc[ja, jb]
    H = 0.5 * (H + H.T)
    return np.linalg.eigvalsh(H)[0] + ecore, M


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 64)
    print("Problem 2.5  --  multi-layer LUCJ-SQD vs HF-SQD  (LiH STO-3G)")
    print("=" * 64)

    mf, ncore, h1e, eri, ecore, e_fci = solve_molecule()
    print(f"[HF ] RHF total energy      = {mf.e_tot:.8f} Ha")
    print(f"[CAS] ncas={NCAS} nelecas={NELECAS} core={ncore} qubits={2*NCAS}")
    print(f"[FCI] CASCI (exact)         = {e_fci:.8f} Ha")

    layers, e_corr = df_layers(mf, ncore)
    print(f"[CC ] CCSD corr energy      = {e_corr:.8f} Ha")
    print(f"[DF ] #layers L = {len(layers)}  "
          f"g_k = {[round(g, 5) for g, _ in layers]}")
    print(f"[LUCJ] lambda = {LAMBDA}, theta = pi/4")

    c_hf = hf_circuit()
    c_lucj = lucj_circuit(layers, LAMBDA)

    dets_hf = unique_dets(recover(sample(c_hf, N_SAMPLES)))
    dets_lucj = unique_dets(recover(sample(c_lucj, N_SAMPLES)))

    e_hf_sqd, m_hf = sqd_energy(dets_hf, h1e, eri, ecore)
    e_lucj_sqd, m_lucj = sqd_energy(dets_lucj, h1e, eri, ecore)

    print("-" * 64)
    print(f"  E(HF-SQD)   = {e_hf_sqd:.8f}   (M={m_hf})")
    print(f"  E(LUCJ-SQD) = {e_lucj_sqd:.8f}   (M={m_lucj})")
    print(f"  E(FCI)      = {e_fci:.8f}")
    print(f"  err HF-SQD   vs FCI = {abs(e_hf_sqd - e_fci)*1e3:8.4f} mHa")
    print(f"  err LUCJ-SQD vs FCI = {abs(e_lucj_sqd - e_fci)*1e3:8.4f} mHa")
    print(f"  |LUCJ-SQD - HF-SQD| = {abs(e_lucj_sqd - e_hf_sqd)*1e6:8.4f} uHa")
    print("=" * 64)


if __name__ == "__main__":
    main()
