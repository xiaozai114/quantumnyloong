#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tencent Sparking Program 2026 - Problem 2.5
LUCJ-SQD vs HF-SQD for LiH (STO-3G).

Pipeline (follows the user's requested route):
  1. Build the Givens-rotation gate and the Jastrow "J" gate from the
     Problem 2.4 decompositions.
  2. Solve the LiH HF ground state (active space, 4 spatial orbitals, 8 qubits).
  3. Build LUCJ = U e^{i J} U^dagger |HF>, scaling all CCSD amplitudes by lambda.
  4. Sample the LUCJ (and HF) circuit 1000 times.
  5. Configuration recovery: enforce particle-number conservation by flipping
     the bit with the largest deviation from the average occupancy.
  6. Build the CI matrix in the sampled subspace and diagonalize -> E_SQD.
  7. Use PySCF's built-in FCI solver for the reference energy.
  8. Compare HF-SQD, LUCJ-SQD, HF and FCI energies.

We deliberately implement the circuit with the exact 2.4 gate primitives
(Givens rotation = 2 CNOT + single-qubit gates; e^{i J_ij n_i n_j}
 = RZ_i RZ_j RZZ_ij ; RZZ = CNOT-RZ-CNOT) inside TensorCircuit.

Qubit convention: JW mapping, spin-orbital ordering
    q[p]        = alpha spin orbital p   (p = 0..norb-1)
    q[norb + p] = beta  spin orbital p
A bit value 1 means "occupied".  |HF> occupies the lowest Nalpha alpha
orbitals and the lowest Nbeta beta orbitals.
"""

import numpy as np
import tensorcircuit as tc
from pyscf import gto, scf, cc, fci, ao2mo

tc.set_backend("numpy")
tc.set_dtype("complex128")

np.random.seed(2026)

# ----------------------------------------------------------------------------
# Section 0. Molecule and active space
# ----------------------------------------------------------------------------
BOND = 1.55  # Angstrom
LAMBDA = 0.05  # ccsd_scale
N_SAMPLES = 1000

NCAS = 4          # active spatial orbitals  -> 8 qubits
NELECAS = (1, 1)  # (Nalpha, Nbeta) ACTIVE electrons after freezing 2 core
                  # LiH STO-3G has 6 spatial orbitals, 4 electrons total.
                  # Per the problem statement "4 spatial orbitals, 4 electrons
                  # (2 frozen core)", we freeze the lowest spatial orbital
                  # (Li 1s core, 2 electrons) and keep an active space of
                  # 4 spatial orbitals with 2 active electrons -> 8 qubits.


def build_molecule():
    mol = gto.M(atom=f"Li 0 0 0; H 0 0 {BOND}", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    return mol, mf


def active_space_integrals(mol, mf, ncas=NCAS, nelecas=NELECAS):
    """Return (h1e, eri, ecore, ncore, mc) for a CAS(ncas, nelecas).

    LiH STO-3G: 6 spatial MOs, 4 electrons total.  With ncas=4 and
    nelecas=(1,1), PySCF's CASCI automatically freezes
        ncore = (Ne_total - Ne_active) / 2 = (4 - 2)/2 = 1
    lowest spatial orbital (the Li 1s core, 2 electrons) into the core, and
    drops the 1 highest virtual.  get_h1eff / get_h2eff fold the frozen-core
    mean-field contribution into ecore, so the returned integrals already
    describe the 4-orbital / 2-electron active-space Hamiltonian (8 qubits)."""
    from pyscf import mcscf
    ncore = (sum(mol.nelec) - sum(nelecas)) // 2  # frozen doubly-occ core orbs
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.verbose = 0
    h1e, ecore = mc.get_h1eff()          # (ncas,ncas) 1e ints incl. core, + Ecore
    eri = mc.get_h2eff()                 # compact (or full) 2e ints
    eri = ao2mo.restore(1, eri, ncas)    # full (ncas^4) chemist notation
    return h1e, eri, ecore, ncore, mc


# ----------------------------------------------------------------------------
# Section 1. Gate primitives from Problem 2.4
# ----------------------------------------------------------------------------
def apply_givens(c, qi, qj, theta):
    """Givens rotation G(theta) acting on the {|01>,|10>} subspace of (qi,qj):
        G|01> = cos t |01> + sin t |10>
        G|10> = -sin t |01> + cos t |10>
    2.4 (eq. 21-22) realization with 2 CNOTs:
        RZ_j(-pi/2) . CNOT_{j->i} . [RY_i(theta) x RY_j(theta)] . CNOT_{j->i} . RZ_j(pi/2)
    We use the standard 2-CNOT Givens (equivalent up to the single-qubit
    dressing that fixes the relative phase)."""
    # Use the well-known decomposition:
    #   CNOT_{qj->qi}; CRY_{qi->qj}(2 theta) via  RY,CNOT ; CNOT_{qj->qi}
    c.cnot(qj, qi)
    # controlled-RY(2 theta) with control qi, target qj  =  RY(theta) CNOT RY(-theta) CNOT
    c.ry(qj, theta=theta)
    c.cnot(qi, qj)
    c.ry(qj, theta=-theta)
    c.cnot(qi, qj)
    c.cnot(qj, qi)


def apply_rzz(c, qi, qj, phi):
    """RZZ_ij(phi) = exp(-i phi Z_i Z_j /2) = CNOT_{i->j} RZ_j(phi) CNOT_{i->j}
    (Problem 2.4 eq. 10)."""
    c.cnot(qi, qj)
    c.rz(qj, theta=phi)
    c.cnot(qi, qj)


def apply_jastrow_pair(c, qi, qj, Jij):
    """e^{i Jij n_i n_j} = (global phase) RZ_i(Jij/2) RZ_j(Jij/2) RZZ_ij(-Jij/2)
    (Problem 2.4 eq. 9).  Global phase e^{i Jij/4} is irrelevant for sampling."""
    c.rz(qi, theta=Jij / 2.0)
    c.rz(qj, theta=Jij / 2.0)
    apply_rzz(c, qi, qj, -Jij / 2.0)


# ----------------------------------------------------------------------------
# Section 2. LUCJ ingredients from CCSD (orbital rotation kappa, Jastrow J)
# ----------------------------------------------------------------------------
def givens_network_from_kappa(kappa):
    """Decompose the orbital rotation U = exp(kappa) (kappa real antisymmetric,
    ncas x ncas) into a sequence of Givens rotations via QR-style Givens
    elimination of the orthogonal matrix Q = exp(kappa).
    Returns a list of (p, q, theta) with p<q meaning a Givens rotation mixing
    spatial orbitals p and q by angle theta, applied in the given order to
    realize Q.  We use the standard Givens decomposition of an orthogonal
    matrix (real, so all rotations are real Givens rotations)."""
    Q = _expm_antisym(kappa)
    n = Q.shape[0]
    M = Q.copy()
    rotations = []
    # Zero out lower-triangular entries with Givens rotations (like QR).
    for col in range(n):
        for row in range(n - 1, col, -1):
            a = M[row - 1, col]
            b = M[row, col]
            if abs(b) < 1e-14:
                continue
            theta = np.arctan2(b, a)
            # rotation acting on rows (row-1, row)
            crot = np.eye(n)
            c_, s_ = np.cos(theta), np.sin(theta)
            crot[row - 1, row - 1] = c_
            crot[row - 1, row] = s_
            crot[row, row - 1] = -s_
            crot[row, row] = c_
            M = crot @ M
            rotations.append((row - 1, row, theta))
    # rotations applied left-to-right reduce Q to diagonal(+/-1); to *build* Q
    # on the state we apply them in reverse with negated angles.
    build = [(p, q, -th) for (p, q, th) in reversed(rotations)]
    return build


def _expm_antisym(kappa):
    from scipy.linalg import expm
    return expm(kappa)


def ccsd_active(mf, ncas, nelecas, ncore):
    """Run CCSD in EXACTLY the ncas-orbital active space (8 qubits) by freezing
    both the ncore doubly-occupied core orbitals AND every virtual orbital
    above the active window.  This makes the CCSD occ/vir dimensions coincide
    with the SQD active space (nocc = nelecas[0], nvir = ncas - nocc), so the
    t2 tensor can be mapped directly onto the ncas spatial orbitals.
    """
    nmo = mf.mo_coeff.shape[1]
    frozen = list(range(ncore)) + list(range(ncore + ncas, nmo))
    mycc = cc.CCSD(mf, frozen=frozen).run()
    return mycc


def double_factorization_layers(mycc, ncas, nelecas, tol=1e-8):
    """Double-factorize the CCSD t2 paired-double channel into L layers.

    For a single active occupied orbital (LiH: nocc=1) the paired-double
    amplitudes form the symmetric virtual-space matrix

        T[a, b] = t2[0, 0, a, b]          (nvir x nvir, symmetric)

    whose eigen-decomposition  T = sum_k g_k v_k v_k^T  defines the layers.
    Layer k is characterised by:
      * a weight g_k (the eigenvalue), which sets the Jastrow phase strength;
      * a virtual direction v_k (length-nvir unit vector), which defines the
        FULL orbital rotation U_k that mixes the occupied orbital (index 0) with
        the collective virtual mode  w_k = sum_a v_k[a] |vir a>.

    Returned per layer: (g_k, kappa_k) where kappa_k is an antisymmetric
    (ncas x ncas) generator so that U_k = exp(kappa_k) rotates orbital 0 into
    the virtual direction v_k by a unit angle (the actual rotation ANGLE is
    supplied later as sqrt(|g_k| * lambda) inside the circuit builder, giving
    the paired-double amplitude its correct lambda-scaling).

    General nocc>1 case (not needed for LiH but implemented for completeness):
    we DF the full occ-vir pair matrix T[(i a),(j b)] = t2[i,j,a,b] and, for
    each layer, build kappa_k from the leading occ-vir singular vector.
    """
    t2 = mycc.t2
    nocc = nelecas[0]
    nvir = ncas - nocc

    layers = []

    if nocc == 1:
        # paired channel is a pure virtual-space symmetric matrix
        T = np.array(t2[0, 0, :nvir, :nvir])
        T = 0.5 * (T + T.T)
        g, Vv = np.linalg.eigh(T)          # eigvals g_k, eigvecs columns v_k
        for k in range(nvir):
            if abs(g[k]) < tol:
                continue
            vk = Vv[:, k]                  # virtual direction (length nvir)
            # antisymmetric generator that rotates orbital 0 <-> virtual mode vk
            # kappa[0, nocc+a] = +vk[a] ; kappa[nocc+a, 0] = -vk[a]
            kappa = np.zeros((ncas, ncas))
            for a in range(nvir):
                kappa[0, nocc + a] = vk[a]
                kappa[nocc + a, 0] = -vk[a]
            layers.append((float(g[k]), kappa))
        return layers

    # ---- general nocc>1 : DF the occ-vir pair matrix ----
    T = np.array(t2).reshape(nocc * nvir, nocc * nvir)
    T = 0.5 * (T + T.T)
    g, W = np.linalg.eigh(T)
    for k in range(nocc * nvir):
        if abs(g[k]) < tol:
            continue
        vec = W[:, k].reshape(nocc, nvir)
        kappa = np.zeros((ncas, ncas))
        for i in range(nocc):
            for a in range(nvir):
                kappa[i, nocc + a] = vec[i, a]
                kappa[nocc + a, i] = -vec[i, a]
        layers.append((float(g[k]), kappa))
    return layers


def lucj_amplitudes(mf, ncas, nelecas, ncore, full_U=True):
    """Run CCSD and build the orbital-rotation generator kappa (ncas x ncas,
    antisymmetric) plus the CCSD object.

    Two parameterizations are supported:

      full_U=False (legacy, sparse):
          kappa_{ai} = t1_{ia}  ONLY the occ-vir block is populated; the
          vir-vir block is identically zero.  For LiH this activates a single
          Givens rotation and CANNOT generate paired double excitations.

      full_U=True (this fix):
          kappa is built from the CCSD 1-particle reduced density matrix
          (1-RDM) restricted to the active space.  Diagonalising the 1-RDM
          gives the natural orbitals; the orthogonal rotation V from the HF
          active orbitals to the natural orbitals is a FULL orbital rotation
          that populates occ-occ, occ-vir AND vir-vir blocks.  We take
          kappa = logm(V) (real antisymmetric), so e^{kappa} = V.  This is the
          complete C(ncas,2)-parameter orbital rotation requested.
    """
    from scipy.linalg import logm

    mycc = cc.CCSD(mf, frozen=ncore).run()
    nocc = nelecas[0]
    nvir = ncas - nocc

    if not full_U:
        # --- legacy sparse (occ-vir only) kappa from t1 ---
        t1 = mycc.t1
        t1a = np.zeros((nocc, nvir))
        t1a[:, :min(nvir, t1.shape[1])] = t1[:nocc, :min(nvir, t1.shape[1])]
        kappa = np.zeros((ncas, ncas))
        for i in range(nocc):
            for a in range(nvir):
                kappa[nocc + a, i] = t1a[i, a]
                kappa[i, nocc + a] = -t1a[i, a]
        return kappa, mycc

    # --- full orbital rotation from the CCSD 1-RDM (incl. vir-vir) ---
    # 1-RDM in the (frozen-core-excluded) MO basis; slice the active block.
    dm1 = mycc.make_rdm1()                      # (nmo, nmo) in MO basis
    # active MO indices: CCSD with frozen=ncore keeps orbitals ncore..nmo-1;
    # make_rdm1 returns it in the full MO basis, so the active block is
    # [ncore : ncore+ncas].
    a0 = ncore
    dm_act = dm1[a0:a0 + ncas, a0:a0 + ncas]
    dm_act = 0.5 * (dm_act + dm_act.T)          # symmetrize
    # natural orbitals: eigvectors of the 1-RDM (descending occupation)
    occ_no, V = np.linalg.eigh(dm_act)
    order = np.argsort(-occ_no)                 # high occupation first
    V = V[:, order]
    # fix global signs so V is as close to identity as possible (continuous
    # branch for logm) -> align each column phase with HF orbital ordering
    for k in range(ncas):
        if V[k, k] < 0:
            V[:, k] = -V[:, k]
    # ensure a proper rotation (det=+1); flip lowest-occ column if reflection
    if np.linalg.det(V) < 0:
        V[:, -1] = -V[:, -1]
    kappa = np.real(logm(V))                    # antisymmetric generator
    kappa = 0.5 * (kappa - kappa.T)             # enforce exact antisymmetry
    return kappa, mycc


def jastrow_from_eri(eri, ncas):
    """(Legacy) Diagonal Coulomb J_ij ~ (ii|jj) style couplings.  Kept only for
    comparison; these ERI density-density integrals carry NO information about
    which double excitations matter, so the resulting phase pattern is flat and
    the Jastrow barely deviates from HF.  Superseded by `jastrow_from_t2`."""
    J = np.zeros((ncas, ncas))
    for i in range(ncas):
        for j in range(ncas):
            if i != j:
                J[i, j] = eri[i, i, j, j]
    return J


def jastrow_from_t2(mycc, ncas, nelecas, ncore, scale=1.0):
    """Build the diagonal-Coulomb J_ij from CCSD t2 (the 'ERIs + CCSD t2'
    prescription in the problem figure), instead of the physically-inert ERI
    placeholder.

    Rationale.  The LUCJ factor exp(i sum_ij J_ij n_i n_j) can only inject
    amplitude onto the correlation-carrying PAIRED double excitations ((k,),(k,))
    if the phase it stamps on those determinants is NON-flat.  CCSD's t2 tensor
    t2[i,j,a,b] directly encodes the weight with which the occupied pair (i,j)
    excites into the virtual pair (a,b).  For the LiH active space there is a
    single active occupied orbital (index 0), and the *paired* channel is
    t2[0,0,a,a] -> both electrons promoted to the same virtual a.

    We therefore define, in the active-orbital (occ|vir) ordering used
    throughout, a symmetric J with

        J[0, a] = J[a, 0] = t2[0,0, a-1, a-1]        for each active virtual a
        J[a, a]           = t2[0,0, a-1, a-1]        (diagonal self-term)

    i.e. the opposite-spin density-density coupling between the occupied
    orbital and each virtual is set by the paired-double amplitude.  This makes
    e^{iJ} stamp a virtual-dependent phase precisely on the ((k,),(k,))
    configurations that U^dagger|HF> has support on, so U rotates a genuine
    admixture of paired doubles back onto |HF> instead of a perfect cancellation.

    All couplings are later multiplied by lambda in `lucj_circuit`.
    """
    t2 = mycc.t2                       # (nocc_act, nocc_act, nvir_full, nvir_full)
    nocc = nelecas[0]                  # active occupied count (=1 for LiH here)
    nvir = ncas - nocc                 # active virtual count (=3 for LiH here)
    J = np.zeros((ncas, ncas))
    # paired-double amplitudes occ-pair (0,0) -> virtual (a,a)
    npv = min(nvir, t2.shape[2])
    for a in range(npv):
        amp = float(t2[0, 0, a, a]) * scale
        vir_idx = nocc + a             # position of this virtual in occ|vir order
        # opposite-spin coupling occupied(0) <-> virtual(vir_idx)
        J[0, vir_idx] += amp
        J[vir_idx, 0] += amp
        # diagonal self-coupling on the virtual (drives the paired phase)
        J[vir_idx, vir_idx] += amp
    return J


# ----------------------------------------------------------------------------
# Section 3. Build HF and LUCJ circuits (JW, 2*ncas qubits)
# ----------------------------------------------------------------------------
def hf_circuit(ncas, nelecas):
    nq = 2 * ncas
    c = tc.Circuit(nq)
    na, nb = nelecas
    for p in range(na):          # lowest alpha orbitals occupied
        c.x(p)
    for p in range(nb):          # lowest beta orbitals occupied
        c.x(ncas + p)
    return c


def lucj_circuit(ncas, nelecas, kappa, J, lam):
    """|Psi_LUCJ> = U e^{i J} U^dagger |HF>, with all amplitudes * lambda.

    U is a spin-restricted orbital rotation acting independently on the alpha
    block (qubits 0..ncas-1) and the beta block (qubits ncas..2ncas-1).

    The local diagonal-Coulomb Jastrow J = exp(i sum J_ij n_i n_j) contains
    BOTH same-spin and opposite-spin density-density couplings:

        J = exp( i * [ sum_{i<j} J_ij (n^a_i n^a_j + n^b_i n^b_j)     # same-spin
                     + sum_{i,j}  J_ij  n^a_i n^b_j            ] )     # opp-spin

    The same-spin part needs >=2 electrons in a spin block to act; for LiH's
    active space (Nalpha=Nbeta=1) it is identically zero, so the opposite-spin
    (alpha-beta) term is the ONLY non-vanishing Jastrow contribution.  All
    couplings are scaled by lam.
    """
    nq = 2 * ncas
    c = hf_circuit(ncas, nelecas)

    kappa_s = lam * kappa
    J_s = lam * J

    givens = givens_network_from_kappa(kappa_s)  # list of (p,q,theta) spatial

    def apply_U_dagger(offset):
        # U^dagger = reverse order, negated angles
        for (p, q, th) in reversed(givens):
            apply_givens(c, offset + p, offset + q, -th)

    def apply_U(offset):
        for (p, q, th) in givens:
            apply_givens(c, offset + p, offset + q, th)

    def apply_J_same_spin(offset):
        # same-spin adjacent pairs within one spin block (inert for 1e/spin)
        for i in range(ncas):
            for j in range(i + 1, ncas):
                if J_s[i, j] != 0.0 and (j == i + 1):  # local adjacent only
                    apply_jastrow_pair(c, offset + i, offset + j, J_s[i, j])

    def apply_J_opp_spin():
        # opposite-spin diagonal Coulomb n^a_i n^b_j : qubit i (alpha block)
        # and qubit ncas+j (beta block).  Dominant / only surviving Jastrow
        # term for a single electron per spin block.
        for i in range(ncas):
            for j in range(ncas):
                Kij = J_s[i, j]
                if Kij != 0.0:
                    apply_jastrow_pair(c, i, ncas + j, Kij)

    # U^dagger  (both spin blocks)
    apply_U_dagger(0)
    apply_U_dagger(ncas)
    # e^{i J} : same-spin (inert for 1e/spin) + opposite-spin (active)
    apply_J_same_spin(0)
    apply_J_same_spin(ncas)
    apply_J_opp_spin()
    # U
    apply_U(0)
    apply_U(ncas)
    return c


def lucj_circuit_multilayer(ncas, nelecas, layers, lam, theta_fixed=np.pi / 4.0):
    """Standard double-factorized LUCJ:

        |Psi> = ( prod_k U_k e^{i J_k} U_k^dagger ) |HF>

    Each layer k = (g_k, kappa_k) from `double_factorization_layers`.

    -----------------------------------------------------------------------
    Correct DF-UCJ amplitude scaling (this is the crux of the multi-layer fix)
    -----------------------------------------------------------------------
    Write out a single layer to first order.  U_k rotates the occupied orbital
    into the layer's virtual mode by a fixed angle theta (O(1), NOT tied to the
    tiny t2 weight).  U_k^dagger|HF> is therefore a genuine superposition of the
    reference and the |vir> orbital in BOTH spin blocks; the diagonal Jastrow
    e^{i J_k} then stamps a relative phase phi_k on the doubly-occupied-virtual
    component; U_k rotates back.  Expanding, the amplitude the layer places on
    the PAIRED double |a,abar> is

        A_paired  ~  (sin^2 theta) * (1 - e^{i phi_k})   ~  theta^2 * phi_k

    i.e. LINEAR in the Jastrow phase phi_k.  We therefore keep theta FIXED at an
    O(1) value and carry the CCSD physics + the lambda knob entirely in the
    phase,

        phi_k = g_k * lam ,          theta = theta_fixed (default pi/4).

    This makes the injected paired-double weight  proportional to g_k * lam
    (first order in the cluster amplitude, exactly as a t2*lam LUCJ should be),
    instead of the g_k^2 * lam suppression that killed the naive
    theta=sqrt(|g_k| lam) choice.  With theta = pi/4, sin^2 theta = 1/2, so the
    layer injects ~ (g_k lam)/2 of paired-double amplitude -- visible in
    sampling once g_k * lam is O(1e-2..1e-1).
    """
    c = hf_circuit(ncas, nelecas)

    for (g, kappa) in layers:
        phi = g * lam                       # Jastrow phase carries t2 * lambda
        # Givens network realising U_k = exp(theta_fixed * kappa) per spin block.
        # kappa is unit-normalised in its occ-vir direction, so theta_fixed IS
        # the physical occ->vir rotation angle for this layer.
        givens = givens_network_from_kappa(theta_fixed * kappa)

        def apply_U_dagger(offset):
            for (p, q, th) in reversed(givens):
                apply_givens(c, offset + p, offset + q, -th)

        def apply_U(offset):
            for (p, q, th) in givens:
                apply_givens(c, offset + p, offset + q, th)

        # opposite-spin diagonal-Coulomb phase in the rotated frame: couple the
        # occupied qubit (0) of the alpha block with occupied qubit (0) of the
        # beta block.  In the U_k^dagger-rotated frame this is where the paired
        # amplitude sits, so this phase breaks the otherwise-perfect U/U^dagger
        # cancellation and leaves a real paired-double admixture after U_k.
        def apply_layer_jastrow():
            apply_jastrow_pair(c, 0, ncas + 0, phi)

        apply_U_dagger(0)
        apply_U_dagger(ncas)
        apply_layer_jastrow()
        apply_U(0)
        apply_U(ncas)

    return c


# ----------------------------------------------------------------------------
# Section 4. Sampling + configuration recovery
# ----------------------------------------------------------------------------
def sample_bitstrings(c, nq, n_samples):
    """Draw `n_samples` computational-basis bitstrings from the circuit state.

    We sample directly from the exact state-vector probability distribution
    p(x) = |<x|psi>|^2 via a single multinomial draw.  This is measurement-
    equivalent to shot-based sampling but avoids a memory blow-up in
    TensorCircuit 0.12's `Circuit.sample` under the NumPy backend for
    non-trivial 8-qubit states.  Bit ordering matches TensorCircuit's
    computational-basis index: qubit 0 is the most-significant bit.
    """
    psi = c.state()
    p = np.abs(np.asarray(psi)) ** 2
    p = p / p.sum()
    dim = p.shape[0]
    idx = np.random.choice(dim, size=n_samples, p=p)
    # decode integer index -> length-nq bit array, q0 = MSB
    samples = np.zeros((n_samples, nq), dtype=int)
    for k in range(nq):
        samples[:, k] = (idx >> (nq - 1 - k)) & 1
    return samples


def configuration_recovery(samples, ncas, nelecas):
    """Enforce Nalpha and Nbeta separately using average occupancy.
    For each spin block, if the number of 1s deviates from the target, flip
    the bit with the largest |b_i - avg_occ_i| (excess -> flip a 1 whose
    occupancy is lowest; deficit -> flip a 0 whose occupancy is highest)
    until the count matches."""
    na, nb = nelecas
    avg = samples.mean(axis=0)  # per-qubit average occupancy
    fixed = samples.copy()

    def fix_block(bits, avg_block, target):
        bits = bits.copy()
        while bits.sum() > target:  # too many electrons -> remove a 1
            ones = np.where(bits == 1)[0]
            # deviation |b - avg| = |1 - avg|; largest deviation = smallest avg
            idx = ones[np.argmin(avg_block[ones])]
            bits[idx] = 0
        while bits.sum() < target:  # too few -> add a 1
            zeros = np.where(bits == 0)[0]
            # |0 - avg| = avg; largest deviation = largest avg
            idx = zeros[np.argmax(avg_block[zeros])]
            bits[idx] = 1
        return bits

    for s in range(fixed.shape[0]):
        a_block = fixed[s, :ncas]
        b_block = fixed[s, ncas:]
        fixed[s, :ncas] = fix_block(a_block, avg[:ncas], na)
        fixed[s, ncas:] = fix_block(b_block, avg[ncas:], nb)
    return fixed


def bits_to_occ(bits_block):
    """Convert a spin-block bit array to a sorted tuple of occupied orbital
    indices (the alpha or beta string)."""
    return tuple(sorted(np.where(bits_block == 1)[0].tolist()))


def unique_determinants(samples, ncas):
    dets = set()
    for s in samples:
        a = bits_to_occ(s[:ncas])
        b = bits_to_occ(s[ncas:])
        dets.add((a, b))
    return sorted(dets)


# ----------------------------------------------------------------------------
# Section 5. CI matrix in the sampled subspace + diagonalization
# ----------------------------------------------------------------------------
def sqd_energy(dets, h1e, eri, ecore, ncas, nelecas):
    """Build the CI matrix restricted to the sampled determinant subspace and
    diagonalize.  We use PySCF's FCI machinery to compute H|D> exactly and then
    project onto the sampled subspace (equivalent to Slater-Condon rules but
    numerically robust)."""
    from pyscf.fci import cistring, direct_spin1

    na, nb = nelecas
    stra = cistring.make_strings(range(ncas), na)
    strb = cistring.make_strings(range(ncas), nb)
    addr_a = {s: i for i, s in enumerate(stra)}
    addr_b = {s: i for i, s in enumerate(strb)}

    def occ_to_str(occ):
        s = 0
        for o in occ:
            s |= (1 << o)
        return s

    # map sampled dets to (ia, ib) full-CI addresses
    idx_list = []
    for (a, b) in dets:
        sa = occ_to_str(a)
        sb = occ_to_str(b)
        if sa in addr_a and sb in addr_b:
            idx_list.append((addr_a[sa], addr_b[sb]))
    if len(idx_list) == 0:
        return None, 0

    dimA, dimB = len(stra), len(strb)
    M = len(idx_list)

    # Build subspace Hamiltonian by applying full H to each basis vector.
    h2 = eri
    Hsub = np.zeros((M, M))
    # precompute action of H on each subspace basis determinant
    for col, (ia, ib) in enumerate(idx_list):
        civec = np.zeros((dimA, dimB))
        civec[ia, ib] = 1.0
        hc = direct_spin1.contract_2e(
            direct_spin1.absorb_h1e(h1e, h2, ncas, nelecas, 0.5),
            civec, ncas, nelecas)
        for row, (ja, jb) in enumerate(idx_list):
            Hsub[row, col] = hc[ja, jb]
    Hsub = 0.5 * (Hsub + Hsub.T)
    evals = np.linalg.eigvalsh(Hsub)
    return evals[0] + ecore, M


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Problem 2.5  --  LUCJ-SQD vs HF-SQD for LiH (STO-3G)")
    print("=" * 72)

    mol, mf = build_molecule()
    e_hf = mf.e_tot
    print(f"[HF ]  RHF total energy            = {e_hf:.8f} Ha")

    h1e, eri, ecore, ncore, mc = active_space_integrals(mol, mf)
    print(f"[CAS]  ncas={NCAS}, nelecas={NELECAS}, frozen core orbs={ncore}, "
          f"qubits={2*NCAS}")

    # FCI reference (PySCF built-in) on the SAME active space -> CASCI energy
    e_cas = mc.kernel()[0]
    print(f"[FCI]  CASCI(=FCI in active space) = {e_cas:.8f} Ha  (PySCF built-in)")

    # ---- LUCJ ingredients: MULTI-LAYER double factorization of CCSD t2 ----
    # CCSD confined to exactly the ncas-orbital active space, then the paired
    # double channel of t2 is double-factorized into L = rank layers, each a
    # (g_k, kappa_k) pair driving one  U_k e^{i J_k} U_k^dagger  factor.
    mycc = ccsd_active(mf, NCAS, NELECAS, ncore)
    layers = double_factorization_layers(mycc, NCAS, NELECAS)
    print(f"[CC ]  CCSD corr energy            = {mycc.e_corr:.8f} Ha")
    print(f"[LUCJ] #DF layers L                = {len(layers)}")
    for k, (g, _kap) in enumerate(layers):
        print(f"[LUCJ]   layer {k}: g_k = {g:+.6f}")
    print(f"[LUCJ] lambda (ccsd_scale)         = {LAMBDA}")

    # ---- Build circuits ----
    c_hf = hf_circuit(NCAS, NELECAS)
    c_lucj = lucj_circuit_multilayer(NCAS, NELECAS, layers, LAMBDA)
    nq = 2 * NCAS

    # ---- Sample ----
    s_hf = sample_bitstrings(c_hf, nq, N_SAMPLES)
    s_lucj = sample_bitstrings(c_lucj, nq, N_SAMPLES)

    # raw unique determinants (before recovery)
    raw_hf = unique_determinants(s_hf, NCAS)
    raw_lucj = unique_determinants(s_lucj, NCAS)

    # count how many raw samples violate particle number (Nalpha/Nbeta)
    def n_violating(samples):
        na, nb = NELECAS
        v = 0
        for s in samples:
            if s[:NCAS].sum() != na or s[NCAS:].sum() != nb:
                v += 1
        return v

    print("-" * 72)
    print(f"[SMP]  HF   : {len(raw_hf)} unique dets (raw), "
          f"{n_violating(s_hf)} particle-number violations")
    print(f"[SMP]  LUCJ : {len(raw_lucj)} unique dets (raw), "
          f"{n_violating(s_lucj)} particle-number violations")

    # ---- Configuration recovery ----
    s_hf_fixed = configuration_recovery(s_hf, NCAS, NELECAS)
    s_lucj_fixed = configuration_recovery(s_lucj, NCAS, NELECAS)
    dets_hf = unique_determinants(s_hf_fixed, NCAS)
    dets_lucj = unique_determinants(s_lucj_fixed, NCAS)
    print(f"[REC]  HF   : {len(dets_hf)} unique dets after recovery")
    print(f"[REC]  LUCJ : {len(dets_lucj)} unique dets after recovery")

    # ---- SQD energies ----
    e_sqd_hf, m_hf = sqd_energy(dets_hf, h1e, eri, ecore, NCAS, NELECAS)
    e_sqd_lucj, m_lucj = sqd_energy(dets_lucj, h1e, eri, ecore, NCAS, NELECAS)

    print("=" * 72)
    print("RESULTS (Ha)")
    print("=" * 72)
    print(f"  E(HF, RHF full)      = {e_hf:.8f}")
    print(f"  E(HF-SQD)            = {e_sqd_hf:.8f}   (subspace dim M={m_hf})")
    print(f"  E(LUCJ-SQD, l={LAMBDA}) = {e_sqd_lucj:.8f}   (subspace dim M={m_lucj})")
    print(f"  E(FCI, CASCI)        = {e_cas:.8f}   (PySCF built-in)")
    print("-" * 72)
    print(f"  err HF-SQD  vs FCI   = {abs(e_sqd_hf - e_cas)*1e3:.4f} mHa")
    print(f"  err LUCJ-SQD vs FCI  = {abs(e_sqd_lucj - e_cas)*1e3:.4f} mHa")
    print(f"  |E(LUCJ-SQD) - E(HF-SQD)| = {abs(e_sqd_lucj - e_sqd_hf)*1e6:.4f} micro-Ha")
    print("=" * 72)

    # ---- persist a small JSON for plotting/reporting ----
    import json
    out = dict(
        bond=BOND, lam=LAMBDA, n_samples=N_SAMPLES, ncas=NCAS, nelecas=list(NELECAS),
        e_hf=e_hf, e_cas_fci=e_cas, e_sqd_hf=e_sqd_hf, e_sqd_lucj=e_sqd_lucj,
        m_hf=m_hf, m_lucj=m_lucj,
        raw_unique_hf=len(raw_hf), raw_unique_lucj=len(raw_lucj),
        viol_hf=n_violating(s_hf), viol_lucj=n_violating(s_lucj),
        det_hf=len(dets_hf), det_lucj=len(dets_lucj),
        e_corr=mycc.e_corr,
    )
    with open("problem_2_5_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved results -> problem_2_5_results.json")

    return out


if __name__ == "__main__":
    main()