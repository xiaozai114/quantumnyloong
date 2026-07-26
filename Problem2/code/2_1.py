"""
Problem 2.1: Hamiltonian Construction & Fermion-to-Qubit Mappings
=================================================================
(a) Build H₂/STO-3G second-quantized Hamiltonian
(b) Jordan-Wigner mapping verification (full derivation: 2.1b-JW-derivation.tex)
(c) Pauli term statistics — JW / Parity / BK
(d) Z₂ symmetry tapering (4→2 qubits)
(e) H₂ vs N₂ comparison + Advanced Challenge:
    For N₂ (n=10 spatial, 20 spin orbs), what is BK's max weight?
    How many times lower is the circuit depth vs JW?

Tencent Sparking Program 2026 — Quantum Computing
"""

import numpy as np
from pyscf import gto, scf, fci
from openfermion import (FermionOperator, QubitOperator, jordan_wigner,
                         count_qubits, get_sparse_operator, get_fermion_operator)
from openfermion.transforms import (binary_code_transform, parity_code,
                                    bravyi_kitaev, taper_off_qubits)
from openfermion.utils import commutator
from openfermionpyscf import generate_molecular_hamiltonian


# ── helper ──────────────────────────────────────────────────────────────────
def _stats(qubit_op):
    """(n_terms, max_weight, weight_distribution)."""
    wd = {}
    for t in qubit_op.terms:
        w = len(t)
        wd[w] = wd.get(w, 0) + 1
    return len(qubit_op.terms), max(wd.keys()) if wd else 0, dict(sorted(wd.items()))


def _print_stats(label, op):
    n, wmax, wd = _stats(op)
    print(f"  {label:>8}: {n:>5} Pauli terms  |  max_weight = {wmax}  |  dist = {wd}")


# ===========================================================================
# 2.1(a)  H₂/STO-3G second-quantized Hamiltonian
# ===========================================================================
print("=" * 64)
print("2.1(a)  Build H₂ (R=0.74 Å, STO-3G) Hamiltonian")
print("=" * 64)

# ── Build Hamiltonian via openfermionpyscf ──
mol_ham_h2 = generate_molecular_hamiltonian(
    [("H", (0, 0, 0)), ("H", (0, 0, 0.74))], "sto-3g", 1)
ham_fermi = get_fermion_operator(mol_ham_h2)
nspin = mol_ham_h2.n_qubits
norb = nspin // 2

# ── PySCF reference info (for display only) ──
mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, verbose=0)
mf = scf.RHF(mol).run()

print(f"  Spatial orbs = {norb},  Spin orbs = {nspin}")
print(f"  E_RHF  = {mf.e_tot:.10f} Ha")
print(f"  ε₀(σ)  = {mf.mo_energy[0]:.8f},  ε₁(σ*) = {mf.mo_energy[1]:.8f}")
print(f"  FermionOperator (via generate_molecular_hamiltonian): {len(ham_fermi.terms)} terms  ✓")

# ===========================================================================
# 2.1(b)  JW mapping — example verification
# ===========================================================================
print(f"\n{'=' * 64}")
print("2.1(b)  JW mapping verification")
print("=" * 64)
print("  Full manual derivation of h₀₁ a₀†a₁ + h.c. → (h₀₁/4)(X₀X₁+Y₀Y₁)")
print("  is in Problem2/2.1b-JW-derivation.tex.")

# Code check with a non-zero term (regular MO basis has h₀₀ ≠ 0)
h00 = mol_ham_h2.one_body_tensor[0, 0]
term = FermionOperator('0^ 0', h00)
res = jordan_wigner(term)
expected = QubitOperator((), h00 / 2) + QubitOperator('Z0', -h00 / 2)
print(f"  Code:  h₀₀ a₀†a₀ → {res}")
print(f"  Check: matches (h₀₀/2)(I − Z₀) = {res == expected}  ✓")

# ===========================================================================
# 2.1(c)  Pauli term statistics: JW / Parity / BK on H₂
# ===========================================================================
print(f"\n{'=' * 64}")
print("2.1(c)  Pauli term count — H₂/STO-3G (4 spin orbs)")
print("=" * 64)

ham_jw = jordan_wigner(ham_fermi)
ham_parity = binary_code_transform(ham_fermi, parity_code(nspin))
ham_bk = bravyi_kitaev(ham_fermi)

for label, op in [("JW", ham_jw), ("Parity", ham_parity), ("BK", ham_bk)]:
    _print_stats(label, op)

# ── Store H₂ stats for later comparison in (e) ──
h2_stats = {
    "JW": _stats(ham_jw), "Parity": _stats(ham_parity), "BK": _stats(ham_bk),
}

# ── Show JW terms explicitly for problem (c) requirement ──
print("\n  JW Pauli terms:")
for i, (term, coeff) in enumerate(sorted(ham_jw.terms.items(),
                                          key=lambda x: len(x[0]))):
    pstr = " ".join(f"{op}{q}" for q, op in sorted(term)) if term else "I"
    print(f"  [{i:2d}] {coeff:+.10f}  {pstr}")

# ===========================================================================
# 2.1(d)  Z₂ symmetry tapering — 4→2 qubits
# ===========================================================================
print(f"\n{'=' * 64}")
print("2.1(d)  Z₂ symmetry tapering (4 → 2 qubits)")
print("=" * 64)

# Stabilizer subgroup: Nα=Nβ=1 → S̄α = −Z₀Z₂,  S̄β = −Z₁Z₃
stab_alpha = QubitOperator('Z0 Z2', -1.0)
stab_beta = QubitOperator('Z1 Z3', -1.0)
assert len(commutator(stab_alpha, ham_jw).terms) == 0
assert len(commutator(stab_beta, ham_jw).terms) == 0
print("  [−Z₀Z₂, H_JW] = [−Z₁Z₃, H_JW] = 0  ✓  (Z₂×Z₂ symmetry)")

ham_tapered, removed = taper_off_qubits(
    ham_jw, [stab_alpha, stab_beta], output_tapered_positions=True)

print(f"  Tapered: {count_qubits(ham_tapered)} qubits, {len(ham_tapered.terms)} terms")
print(f"  Quenched qubits: {removed}")

H_tap_mat = get_sparse_operator(ham_tapered).toarray()
E_tapered = float(np.linalg.eigvalsh(H_tap_mat)[0])
cisolver = fci.FCI(mf)
E_FCI, _ = cisolver.kernel()

print(f"  H_red:")
for term, coeff in sorted(ham_tapered.terms.items(),
                          key=lambda x: len(x[0])):
    pstr = " ".join(f"{op}{q}" for q, op in sorted(term)) if term else "II"
    print(f"    {coeff.real:+.10f}  {pstr}")
print(f"  E_tapered = {E_tapered:.10f} Ha")
print(f"  E_FCI     = {E_FCI:.10f} Ha")
print(f"  Δ = {abs(E_tapered - E_FCI):.2e}  ✓  (spectrum preserved in stabilizer subspace)")

# ===========================================================================
# 2.1(e)  H₂ vs N₂ — JW / Parity / BK comparison
# ===========================================================================
print(f"\n{'=' * 64}")
print("2.1(e)  H₂ vs N₂ — JW / Parity / BK comparison")
print("=" * 64)

# ── N₂: use generate_molecular_hamiltonian ──
mol_ham_n2 = generate_molecular_hamiltonian(
    [("N", (0, 0, 0)), ("N", (0, 0, 1.10))], "sto-3g", 1)
ferm_n2 = get_fermion_operator(mol_ham_n2)
nq_n2 = mol_ham_n2.n_qubits

n2_stats = {}
for label, op in [
    ("JW", jordan_wigner(ferm_n2)),
    ("Parity", binary_code_transform(ferm_n2, parity_code(nq_n2))),
    ("BK", bravyi_kitaev(ferm_n2)),
]:
    n2_stats[label] = _stats(op)

# ── H₂ stats from (c) — no recalculation needed ──
for name, n_spin, fermi_terms, stats in [
    ("H₂", nspin, len(ham_fermi.terms), h2_stats),
    ("N₂", nq_n2, len(ferm_n2.terms), n2_stats),
]:
    print(f"\n  ── {name} ──")
    print(f"  Spin orbitals: {n_spin}   Fermi terms: {fermi_terms}")
    print(f"  {'Mapping':>8}  {'Pauli terms':>12}  {'Max weight':>10}")
    print(f"  {'─' * 36}")
    for label in ["JW", "Parity", "BK"]:
        n, w, _ = stats[label]
        print(f"  {label:>8}  {n:>12}  {w:>10}")

    # Weight distribution
    all_w = sorted(set().union(*[stats[k][2].keys() for k in ["JW", "Parity", "BK"]]))
    print(f"\n  {'w':>4}  {'JW':>8}  {'Parity':>8}  {'BK':>8}")
    print(f"  {'─' * 32}")
    for w in all_w:
        jw_c = stats["JW"][2].get(w, 0)
        pa_c = stats["Parity"][2].get(w, 0)
        bk_c = stats["BK"][2].get(w, 0)
        print(f"  {w:>4}  {jw_c:>8}  {pa_c:>8}  {bk_c:>8}")

# ===========================================================================
# Advanced Challenge
# ===========================================================================
print(f"\n{'=' * 64}")
print("Advanced Challenge: BK max weight for N₂ & circuit depth reduction")
print("=" * 64)

jw_n, jw_w, _ = n2_stats["JW"]
bk_n, bk_w, _ = n2_stats["BK"]

print(f"\n  N₂/STO-3G: {nq_n2} spin orbitals ({nq_n2 // 2} spatial)")
print(f"  JW max Pauli weight = {jw_w}   O(n) = n = {nq_n2}")
print(f"  BK max Pauli weight = {bk_w}   O(log n) scaling")
print(f"    (single Majorana: ~2·log₂({nq_n2})+1 ≈ {2*int(np.ceil(np.log2(nq_n2)))+1};")
print(f"     multi-body product terms in H push it to {bk_w})")

depth_ratio = jw_w / bk_w
print(f"\n  Circuit depth reduction ratio:")
print(f"    max(JW weight) / max(BK weight) = {jw_w} / {bk_w} = {depth_ratio:.2f}×")
print(f"  Trotter circuit depth is dominated by the highest-weight Pauli term,")
print(f"  which requires O(weight) entangling gates. BK achieves a")
print(f"  {depth_ratio:.1f}× depth reduction vs JW for N₂/STO-3G.")
print(f"  This advantage grows with system size: O(log n) vs O(n).")
