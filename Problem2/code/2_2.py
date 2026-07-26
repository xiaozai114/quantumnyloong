"""
Problem 2.2: HF State and SQD Subspace Diagonalization
=======================================================
H₂/STO-3G, 4 spin orbitals (0–3), 2 electrons.

(a) Determine HF state bitstring under JW; write preparation circuit.
(b) List all 4-choose-2 = 6 configurations; build 6×6 CI matrix via
    Slater–Condon rules; exactly diagonalize; compare with PySCF FCI.
(c) HF + singles subspace → Brillouin's theorem → equals RHF energy.

Tencent Sparking Program 2026 — Quantum Computing
"""

import numpy as np
from pyscf import gto, scf, fci
from pyscf import ao2mo


# =============================================================================
# 1. H₂/STO-3G setup & MO integrals
# =============================================================================
print("=" * 64)
print("2.2  H₂/STO-3G — SQD Subspace Diagonalization")
print("=" * 64)

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, verbose=0)
mf = scf.RHF(mol).run()
norb = mol.nao                         # 2 spatial
nspin = 2 * norb                       # 4 spin orbitals
E_nuc = mol.energy_nuc()

# AO → MO one-body integrals: h_pq = <p|h|q>
h_mo = mf.mo_coeff.T @ (mol.intor("int1e_kin") + mol.intor("int1e_nuc")) @ mf.mo_coeff

# AO → MO two-body integrals (chemist's notation): (pq|rs)
eri_mo = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)

print(f"\n  Spatial orbs = {norb},  Spin orbs = {nspin}")
print(f"  E_nuc  = {E_nuc:+.10f} Ha")
print(f"  E_RHF  = {mf.e_tot:+.10f} Ha")
print(f"  ε₀(σ_g)  = {mf.mo_energy[0]:+.8f},  ε₁(σ_u*) = {mf.mo_energy[1]:+.8f}")
print(f"  h[0,0] = {h_mo[0,0]:+.8f},  h[1,1] = {h_mo[1,1]:+.8f}")
print(f"  (00|00) = {eri_mo[0,0,0,0]:+.8f},  (00|11) = {eri_mo[0,0,1,1]:+.8f}")
print(f"  (01|01) = {eri_mo[0,1,0,1]:+.8f},  (01|10) = {eri_mo[0,1,1,0]:+.8f}")

# Verify HF energy: 2*h₀₀ + (00|00) + E_nuc
E_hf_check = 2 * h_mo[0, 0] + eri_mo[0, 0, 0, 0] + E_nuc
print(f"  2·h₀₀ + (00|00) + E_nuc = {E_hf_check:+.10f} Ha  "
      f"(Δ = {abs(E_hf_check - mf.e_tot):.2e}) ✓")


# =============================================================================
# 2. Spin-orbital integral helpers (chemist's notation)
# =============================================================================
# Chemist's notation for spin orbitals: ⟨pσ_p, qσ_q|rσ_r, sσ_s⟩
# = δ(σ_p, σ_q) · δ(σ_r, σ_s) · (P Q | R S)_spatial
# (not δ(σ_p,σ_r)·δ(σ_q,σ_s) — that's the physicist's convention!)

def h_so(p, q):
    """One-body integral ⟨p|h|q⟩ for spin orbitals p,q ∈ {0..3}."""
    if (p % 2) != (q % 2):          # different spin → zero
        return 0.0
    return h_mo[p // 2, q // 2]


def v_chem(p, q, r, s):
    """Two-body integral in chemist's notation: (pq|rs) for spin orbitals.
    Non-zero only when spin(p)=spin(q) AND spin(r)=spin(s)."""
    sp, sq, sr, ss = p % 2, q % 2, r % 2, s % 2
    if sp != sq or sr != ss:
        return 0.0
    return eri_mo[p // 2, q // 2, r // 2, s // 2]


# =============================================================================
# 3. Slater–Condon matrix elements
# =============================================================================
def _sc_sign(occ_a, occ_b, diff_a, diff_b):
    """Phase for ⟨A|H|B⟩: (−1)^{#crossings} for aligned determinants."""
    # The sign equals (−1)^{n_cross} where n_cross = number of electrons
    # that lie between the positions of the differing SOs
    common = sorted(set(occ_a) & set(occ_b))
    a_sorted = sorted(occ_a)
    b_sorted = sorted(occ_b)

    # Simpler approach: compute parity by reconstructing the transformation
    # Build occ_b in order, remove diff_b, insert diff_a → check parity
    result = list(b_sorted)
    # Remove diff_b in descending order (standard normal-ordering convention)
    for d in sorted(diff_b, reverse=True):
        result.remove(d)
    # Insert diff_a in ascending order (standard convention)
    for a in sorted(diff_a):
        # Find position maintaining sorted order
        for pos in range(len(result) + 1):
            if pos == len(result) or result[pos] > a:
                result.insert(pos, a)
                break
    # Now we need to count the parity from original b_sorted → result
    # This is the same as counting inversions in the combined mapping
    mapping = list(b_sorted)
    for d in sorted(diff_b, reverse=True):
        mapping.remove(d)
    mapping = sorted(diff_a) + mapping
    # Count inversions relative to b_sorted
    # b_sorted: [... common ..., ... diff_b ...]
    # result:   [... diff_a, ... common ...]
    # Parity = (#crossings between common and diff_a) + (#crossings between common and diff_b)
    n_cross = 0
    for c in common:
        for d in diff_b:
            if c > d:
                n_cross += 1
        for d in diff_a:
            if c > d:
                n_cross += 1
    return (-1) ** n_cross


def sc_diagonal(occ):
    """⟨D|H|D⟩ = Σ_i h_ii + ½ Σ_ij [(ii|jj) − (ij|ji)] + E_nuc
    where i,j run over occupied spin-orbitals."""
    E = sum(h_so(i, i) for i in occ)
    for i in occ:
        for j in occ:
            E += 0.5 * (v_chem(i, i, j, j) - v_chem(i, j, j, i))
    return E + E_nuc


def sc_offdiag(occ_a, occ_b):
    """⟨D_A|H|D_B⟩ for A ≠ B using Slater–Condon rules."""
    sa, sb = set(occ_a), set(occ_b)
    diff_a = sorted(sa - sb)      # SOs in A but not B (electrons added)
    diff_b = sorted(sb - sa)      # SOs in B but not A (electrons removed)
    nd = len(diff_a)

    if nd == 0:
        return sc_diagonal(occ_a)
    if nd > 2:
        return 0.0

    phase = _sc_sign(occ_a, occ_b, diff_a, diff_b)

    if nd == 1:
        # Single excitation: i → a  (i in B, a in A)
        i, a = diff_b[0], diff_a[0]
        common = sorted(sa - {a})
        me = h_so(i, a)
        for j in common:
            me += v_chem(i, a, j, j) - v_chem(i, j, j, a)
        return phase * me

    # nd == 2: Double excitation: i,j → a,b
    i, j = diff_b
    a, b = diff_a
    me = v_chem(i, a, j, b) - v_chem(i, b, j, a)
    return phase * me


# =============================================================================
# 4. HF state under JW
# =============================================================================
print(f"\n{'─' * 64}")
print("2.2(a)  HF state under Jordan-Wigner mapping")
print("─" * 64)

# Spin-orbital assignment (JW: qubit i ↔ spin orbital i)
#   SO 0 = σ_g(α)   SO 1 = σ_g(β)   SO 2 = σ_u*(α)   SO 3 = σ_u*(β)
# RHF: doubly occupy σ_g → spin orbitals 0, 1 occupied
# JW: |b₃b₂b₁b₀⟩ = |0,0,1,1⟩ → integer 3

print(f"  Spin orbitals: 0=σ_g(α), 1=σ_g(β), 2=σ_u*(α), 3=σ_u*(β)")
print(f"  HF state    : |0,0,1,1⟩  (SO 0,1 occupied)")
print(f"  JW integer  : 3")
print(f"\n  Preparation circuit (JW, 4 qubits, via tensorcircuit):")
print(f"    c = tc.Circuit(4);  c.x(0);  c.x(1)")
print(f"    → |0000⟩  ↦  |0011⟩  ≡ |HF⟩")

# Verify HF diagonal
E_hf_sc = sc_diagonal((0, 1))
assert abs(E_hf_sc - mf.e_tot) < 1e-10
print(f"\n  ⟨HF|H|HF⟩ (Slater–Condon) = {E_hf_sc:+.10f} Ha  = E_RHF ✓")


# =============================================================================
# 5. All 6 configurations
# =============================================================================
print(f"\n{'─' * 64}")
print("2.2(b)  6 two-electron configurations")
print("─" * 64)

# All 6 states with exactly 2 electrons (C(4,2)=6)
configs = sorted([(i, tuple(sorted(j for j in range(4) if (i >> j) & 1)))
                  for i in range(16) if bin(i).count("1") == 2])
assert len(configs) == 6


def exc_label(val, occ):
    """Label relative to HF = (0,1)."""
    hf = {0, 1}
    s = set(occ)
    r = hf - s
    a = s - hf
    n = len(r)
    if n == 0:
        return "HF"
    if n == 1:
        return f"single  {r} → {a}"
    return "double  (0,1)→(2,3)"


print(f"\n  {'k':>3}  {'int':>4}  {'|b₃b₂b₁b₀⟩':>12}  {'occ SOs':>10}  {'excitation':>26}")
print(f"  {'─' * 60}")
for k, (val, occ) in enumerate(configs):
    print(f"  {k:>3}  {val:>4}  {'|' + f'{val:04b}' + '⟩':>12}  {str(occ):>10}  "
          f"{exc_label(val, occ):>26}")


# =============================================================================
# 6. Build 6×6 CI matrix & diagonalize
# =============================================================================
print(f"\n{'─' * 64}")
print("2.2(b)  6×6 CI matrix (Slater–Condon) & diagonalization")
print("─" * 64)

H_ci = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        H_ci[i, j] = sc_offdiag(configs[i][1], configs[j][1])

assert np.allclose(H_ci, H_ci.conj().T), "CI matrix not Hermitian!"

print(f"\n  CI matrix H_kl = ⟨D_k|H|D_l⟩  (Ha):")
print(f"  {'k:':>3}  " + "".join(f"{'D'+str(j):>12}" for j in range(6)))
print(f"  {'─' * 78}")
for i in range(6):
    print(f"  {'D'+str(i):>3}  " + "".join(f"{H_ci[i, j].real:>12.8f}" for j in range(6)))

# Diagonalize
eigvals, eigvecs = np.linalg.eigh(H_ci)
E_sqd = eigvals[0].real

# FCI reference
E_fci, _ = fci.FCI(mf).kernel()

print(f"\n  6×6 eigenvalues:")
for k, ev in enumerate(eigvals):
    tag = " ← ground (SQD)" if k == 0 else ""
    print(f"    λ{k} = {ev.real:+.10f} Ha{tag}")

print(f"\n  E_SQD (6×6 subspace) = {E_sqd:+.10f} Ha")
print(f"  E_FCI (PySCF exact)  = {E_fci:+.10f} Ha")
print(f"  Δ = {abs(E_sqd - E_fci):.2e} Ha  ← full-CI in 2-electron subspace ✓")

# Ground-state vector
gs = eigvecs[:, 0]
print(f"\n  Ground state |Ψ₀⟩ = Σ c_k |D_k⟩:")
nominal_contrib = [(abs(c)**2, k, c) for k, c in enumerate(gs)]
nominal_contrib.sort(reverse=True)
for weight, k, c in nominal_contrib:
    val, occ = configs[k]
    bits = f"|{val:04b}⟩"
    print(f"    c{k} = {c.real:+9.6f}  {bits}  {exc_label(val, occ)}"
          f"   (|c|² = {weight:.5f})")


# =============================================================================
# 7. HF + singles subspace (5×5) — Brillouin's theorem
# =============================================================================
print(f"\n{'─' * 64}")
print("2.2(c)  HF + singles (5×5, no doubles) — Brillouin's theorem")
print("─" * 64)

idx5 = [0, 1, 2, 3, 4]  # exclude D5 = double
H5 = H_ci[np.ix_(idx5, idx5)]
ev5, _ = np.linalg.eigh(H5)

print(f"\n  5×5 CI matrix (HF + 4 singles):")
print(f"  {'k:':>3}  " + "".join(f"{'D'+str(j):>12}" for j in range(5)))
print(f"  {'─' * 66}")
for i in range(5):
    print(f"  {'D'+str(i):>3}  " + "".join(f"{H5[i, j].real:>12.8f}" for j in range(5)))

print(f"\n  HF row ⟨HF|H|D_k⟩:")
for j in range(5):
    val, occ = configs[j]
    print(f"    ⟨HF|H|D{j}(|{val:04b}⟩)⟩ = {H_ci[0, j].real:+.10f} Ha   "
          f"({exc_label(val, occ)})")

print(f"\n  5×5 eigenvalues:")
for k, ev in enumerate(ev5):
    tag = " ← ground" if k == 0 else ""
    print(f"    λ{k} = {ev.real:+.10f} Ha{tag}")

print(f"\n  E(HF + singles) = {ev5[0].real:+.10f} Ha")
print(f"  E_RHF           = {mf.e_tot:+.10f} Ha")
print(f"  Δ               = {abs(ev5[0].real - mf.e_tot):.2e} Ha")
print(f"\n  Brillouin's theorem summary:")
print(f"  ─────────────────────────────────────────────────────────────")
print(f"  ⟨HF|H|single⟩ = 0 for all 4 single excitations (confirmed above).")
print(f"  Reason: Fock matrix is diagonal in canonical MO basis →")
print(f"  the occupied-virtual block F_{{ia}} = 0.")
print(f"  Singles couple only indirectly (via doubles). Without doubles,")
print(f"  the HF decouples → lowest eigenvalue = E_RHF exactly.")
print(f"  The HF self-consistency condition makes singles vanish at first")
print(f"  order in CI expansion.")


# =============================================================================
# Summary
# =============================================================================
print(f"\n{'=' * 64}")
print("Summary")
print("=" * 64)
print(f"  Setup       : H₂/STO-3G  (R=0.74 Å, 4 spin orbs, 2 electrons)")
print(f"  HF state    : |0011⟩  (SO 0,1 occupied)")
print(f"  ⟨HF|H|HF⟩  : {E_hf_sc:+.10f} Ha")
print(f"  6×6 CI (FCI): E₀ = {E_sqd:+.10f} Ha  (Δ_FCI = {abs(E_sqd - E_fci):.2e})")
print(f"  5×5 CI      : E₀ = {ev5[0].real:+.10f} Ha  (= E_RHF, Brillouin)")
print(f"  FCI ref     : E₀ = {E_fci:+.10f} Ha")
print(f"\n  The 6-determinant subspace in the 2-electron sector spans the full")
print(f"  FCI space. SQD exact diagonalization recovers the FCI ground state.")
