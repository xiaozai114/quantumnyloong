"""
Q01: Build Hamiltonian and Mapping Comparison
Part (a): Use PySCF to compute RHF for H2 and extract integrals
"""

import numpy as np
from pyscf import gto, scf

# ---- H2 molecule, STO-3G, bond length 0.74 Å ----
mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
mf = scf.RHF(mol)
mf.kernel()

print(f"RHF energy: {mf.e_tot:.6f} Ha")
print(f"Nuclear repulsion: {mol.energy_nuc():.6f} Ha")
print(f"Spatial orbitals: {mol.nao}")
print(f"Electrons: {mol.nelec}")

# ---- One-body integrals ----
h1e_ao = mf.get_hcore()          # AO basis
C = mf.mo_coeff                   # MO coefficients
h1e_mo = C.T @ h1e_ao @ C        # MO basis

# ---- Two-body integrals ----
eri_ao = mol.intor("int2e")       # AO basis, chemist's notation (pq|rs)
eri_mo = np.einsum("pi,pqrs->iqrs", C, eri_ao)
eri_mo = np.einsum("qj,iqrs->ijrs", C, eri_mo)
eri_mo = np.einsum("rk,ijrs->ijks", C, eri_mo)
eri_mo = np.einsum("sl,ijks->ijkl", C, eri_mo)

# ---- Nuclear repulsion ----
E_core = mol.energy_nuc()

# ---- Spatial orbital -> Spin orbital ----
n_spatial = mol.nao  # = 2 for H2 STO-3G
n_spin = 2 * n_spatial  # = 4 spin orbitals (qubits)

# Convention: (0a, 1a, 0b, 1b) -> qubits 0,1,2,3
# h_{ps,qt} = delta_st * h_{pq}  (spin orthogonality)
h_spin = np.zeros((n_spin, n_spin))
for p in range(n_spatial):
    for q in range(n_spatial):
        h_spin[p, q] = h1e_mo[p, q]          # alpha block
        h_spin[p + n_spatial, q + n_spatial] = h1e_mo[p, q]  # beta block

# Two-body in spin-orbital basis
# (ps,qt|ru,sv) = delta_su * delta_tv * (pq|rs)
eri_spin = np.zeros((n_spin, n_spin, n_spin, n_spin))
for p in range(n_spatial):
    for q in range(n_spatial):
        for r in range(n_spatial):
            for s in range(n_spatial):
                # aa|aa
                eri_spin[p, q, r, s] = eri_mo[p, q, r, s]
                # bb|bb
                eri_spin[p+n_spatial, q+n_spatial, r+n_spatial, s+n_spatial] = eri_mo[p, q, r, s]
                # aa|bb
                eri_spin[p, q, r+n_spatial, s+n_spatial] = eri_mo[p, q, r, s]
                # bb|aa
                eri_spin[p+n_spatial, q+n_spatial, r, s] = eri_mo[p, q, r, s]

# ---- Antisymmetrized integrals (pq||rs) = (pq|rs) - (pq|sr) ----
def antisym(eri, p, q, r, s):
    return eri[p, q, r, s] - eri[p, q, s, r]

# ---- Print summary ----
print(f"\n--- Spin-orbital integrals ---")
print(f"h_spin shape: {h_spin.shape}")
print(f"eri_spin shape: {eri_spin.shape}")
print(f"\nOne-body integrals (spatial, MO basis):")
print(h1e_mo)
print(f"\nNon-zero two-body integrals (spatial, MO basis):")
for p in range(n_spatial):
    for q in range(n_spatial):
        for r in range(n_spatial):
            for s in range(n_spatial):
                v = eri_mo[p, q, r, s]
                if abs(v) > 1e-10:
                    print(f"  ({p}{q}|{r}{s}) = {v:.6f}")

# ---- FCI reference ----
from pyscf import fci
cisolver = fci.FCI(mf)
e_fci = cisolver.kernel()[0]
print(f"\nFCI energy: {e_fci:.6f} Ha")
print(f"HF energy: {mf.e_tot:.6f} Ha")
print(f"Correlation energy: {e_fci - mf.e_tot:.6f} Ha")
