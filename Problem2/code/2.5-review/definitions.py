"""
Tencent Sparking Program 2026 - Problem 2.5 (a)
LUCJ-SQD vs HF-SQD: How Entanglement Improves Sampling  --  DEFINITIONS

This module contains ONLY the setup and the building-block definitions:
molecular problem, CCSD->LUCJ generators, the quantum circuits (TensorCircuit),
the bitstring sampler, and the SQD (selected-CI) exact diagonalisation.
The driver / experiment logic lives in `experiment.py`.

Stack:
    - PySCF        : molecular integrals, RHF, CCSD amplitudes, exact FCI.
    - TensorCircuit: state preparation + measurement sampling.
    - Backend      : TensorFlow (set explicitly below).

Physics / conventions
---------------------
* LiH STO-3G has 6 spatial orbitals. Following the problem statement
  ("LiH: 4 spatial orbitals, 4 electrons, 2 frozen core" and Problem 5's
  "LiH, N = 8"), we work in an active space of n = 4 spatial orbitals
  (8 spin-orbitals -> 8 qubits) with 2 frozen-core electrons, i.e. 2 active
  electrons (N_alpha = N_beta = 1). This is the standard SQD setting for LiH.
* Jordan-Wigner ordering used here: qubit index = spin-orbital index with the
  "interleaved" layout q = 2*p + s  (p = spatial orbital, s = 0 for alpha,
  1 for beta). A computational basis bitstring |b_{2n-1} ... b_0> then encodes
  a Slater determinant: b_{2p+s} = 1 means spin-orbital (p, s) is occupied.
* LUCJ ansatz (see Problem 2.4 solution):
        |Psi_LUCJ> = U e^{i J_local} U^\\dagger |HF>,
  with U = e^{kappa} an orbital rotation built from CCSD t1 amplitudes and
  J_local = sum_{i,j} J_ij n_i n_j a diagonal Coulomb built from CCSD t2.
  The scalar lambda multiplies the correlation amplitudes:
        lambda = 0 -> |HF>,   lambda = 1 -> full (unscaled) UCJ.
  Small lambda (0.05) keeps the state close to HF so that measurement still
  concentrates on the physically relevant configurations while spreading a
  little amplitude onto the important double excitations that SQD needs.

Author: WorkBuddy
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # silence TF info logs
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import tensorcircuit as tc

# --- TensorCircuit backend: TensorFlow (required by the task) ---------------
K = tc.set_backend("tensorflow")
tc.set_dtype("complex128")

from pyscf import gto, scf, cc, fci, ao2mo


# ============================================================================
# 1. Molecular problem: LiH (STO-3G) in an active space
# ============================================================================
def build_lih_active_space():
    """RHF on LiH, freeze the 2 core electrons, return an active-space
    (4 spatial orbitals, 2 electrons) one/two-body Hamiltonian plus the
    reference FCI/HF energies for the full molecule."""
    mol = gto.M(
        atom="Li 0 0 0; H 0 0 1.55",   # bond length 1.55 Angstrom
        basis="sto-3g",
        spin=0,
        charge=0,
        verbose=0,
    )
    mf = scf.RHF(mol).run()

    # --- active space selection: freeze 1 core spatial orbital (2 electrons)
    ncore = 1                      # 1 frozen spatial orbital (Li 1s) -> 2 e-
    nact = 4                       # 4 active spatial orbitals
    nelec_act = mol.nelectron - 2 * ncore   # = 4 - 2 = 2 active electrons
    act = slice(ncore, ncore + nact)

    mo = mf.mo_coeff
    mo_core = mo[:, :ncore]
    mo_act = mo[:, act]

    # core density and effective one-body core contribution
    dm_core = 2.0 * mo_core @ mo_core.T
    hcore = mf.get_hcore()
    vhf_core = mf.get_veff(mol, dm_core)
    e_core = (
        mol.energy_nuc()
        + np.einsum("ij,ji->", hcore, dm_core)
        + 0.5 * np.einsum("ij,ji->", vhf_core, dm_core)
    )

    # one-body integrals in the active MO basis (spatial)
    h1_act = mo_act.T @ (hcore + vhf_core) @ mo_act
    # two-body integrals (chemist's notation) in the active MO basis
    eri_act = ao2mo.kernel(mol, mo_act)
    eri_act = ao2mo.restore(1, eri_act, nact)   # full (nact^4) tensor

    # exact FCI energy inside the active space (this is our accuracy target)
    cisolver = fci.direct_spin1.FCI()
    e_fci_act, fcivec = cisolver.kernel(
        h1_act, eri_act, nact, (nelec_act // 2, nelec_act // 2),
        ecore=e_core,
    )

    data = dict(
        mol=mol, mf=mf,
        nact=nact, nelec_act=nelec_act,
        h1=h1_act, eri=eri_act, e_core=e_core,
        e_hf=mf.e_tot, e_fci=e_fci_act,
        cisolver=cisolver, fcivec=fcivec,
        ncore=ncore,
    )
    return data


# ============================================================================
# 2. CCSD amplitudes -> LUCJ generators (orbital rotation kappa + Jastrow J)
# ============================================================================
def ccsd_amplitudes(data):
    """Run CCSD in the same active space to obtain t1, t2 amplitudes used to
    build the LUCJ orbital-rotation and diagonal-Coulomb generators."""
    mol, mf = data["mol"], data["mf"]
    # frozen = number of frozen spatial orbitals (core)
    mycc = cc.CCSD(mf, frozen=data["ncore"]).run()
    t1 = np.asarray(mycc.t1)     # shape (nocc, nvir)
    t2 = np.asarray(mycc.t2)     # shape (nocc, nocc, nvir, nvir)
    return t1, t2


def lucj_generators(data, t1, t2, lam):
    """Build the generators of the LUCJ ansatz, scaled by lambda.

    Physics
    -------
    For a closed-shell reference the singles amplitudes vanish by Brillouin's
    theorem (here max|t1| ~ 0.04), so the orbital rotation U = e^{kappa} built
    from t1 is nearly the identity and, acting together with the *diagonal*
    Coulomb factor e^{iJ}, leaves the HF number-eigenstate essentially
    unchanged (e^{iJ} only adds a phase to a number eigenstate).

    The configuration-generating power of LUCJ therefore comes from the paired
    DOUBLE excitations encoded in t2. In the STO-3G LiH active space the
    dominant double is (i,i)->(a,a) with t2[i,i,a,a]. We realise it as a pair
    of occupied<->virtual Givens rotations applied SIMULTANEOUSLY on the alpha
    and beta spin sectors, so that

        |HF> = |i_alpha i_beta>  --Givens(theta)-->
               cos^2 theta |i_a i_b> + sin^2 theta |a_a a_b> + ...(singles)

    i.e. amplitude leaks onto the physically important double excitation
    |a_alpha a_beta>. The Givens angle is set by the (scaled) CCSD amplitude
        theta_{i->a} = lambda * arcsin-like weight ~ lambda * |t2[i,i,a,a]| .

    Returns
    -------
    theta_pairs : list of (i_spatial, a_spatial, theta) occupied->virtual
                  Givens rotations (applied on both spin sectors).
    kappa       : (nact,nact) antisymmetric single-particle generator (t1).
    Jmat        : (nact,nact) symmetric diagonal-Coulomb matrix (t2 diagonal).
    """
    nact = data["nact"]
    nocc = data["nelec_act"] // 2          # doubly occupied active spatial orbs
    nvir = nact - nocc

    # --- (weak) orbital rotation from t1 -----------------------------------
    kappa = np.zeros((nact, nact))
    for i in range(nocc):
        for a in range(nvir):
            val = lam * t1[i, a]
            kappa[nocc + a, i] += val
            kappa[i, nocc + a] -= val

    # --- diagonal Coulomb J from diagonal doubles (adds correlation phases) -
    Jmat = np.zeros((nact, nact))
    for i in range(nocc):
        for a in range(nvir):
            w = lam * t2[i, i, a, a]
            p, q = i, nocc + a
            Jmat[p, q] += w
            Jmat[q, p] += w

    # --- occupied<->virtual Givens angles from dominant paired doubles ------
    theta_pairs = []
    for i in range(nocc):
        for a in range(nvir):
            # scale so lambda=1 gives an O(0.3 rad) mixing for the leading
            # double; lambda=0.05 -> gentle leakage, keeping the state near HF.
            theta = lam * t2[i, i, a, a] * 3.0
            if abs(theta) > 1e-10:
                theta_pairs.append((i, nocc + a, theta))
    return theta_pairs, kappa, Jmat


# ============================================================================
# 3. Quantum circuits (TensorCircuit)
# ============================================================================
# Spin-orbital -> qubit map (interleaved): qubit = 2*p + s, s in {0=alpha,1=beta}
def qubit_index(p, s):
    return 2 * p + s


def hf_occupied_qubits(data):
    """Qubits occupied in the HF determinant: lowest nocc spatial orbitals,
    both spins."""
    nocc = data["nelec_act"] // 2
    occ = []
    for p in range(nocc):
        occ.append(qubit_index(p, 0))   # alpha
        occ.append(qubit_index(p, 1))   # beta
    return occ


def add_hf_state(c, data):
    """Prepare |HF> by flipping the occupied qubits from |0...0>."""
    for q in hf_occupied_qubits(data):
        c.x(q)
    return c


def add_orbital_rotation(c, kappa, data, dagger=False):
    """Apply U = exp(kappa) (or U^dagger) as a product of two-qubit Givens
    rotations acting on adjacent spatial orbitals, replicated on each spin
    sector. Givens angle theta_{p,p+1} taken from kappa[p+1, p].

    A Givens rotation G(theta) on spin-orbitals (i, j) is implemented with the
    2-CNOT decomposition of Problem 2.4:
        RX_i(+pi/2) . CNOT_{i->j} . (RX_i(theta) x RY_j(theta)) . CNOT_{i->j} . RX_i(-pi/2)
    Here we use the equivalent, numerically exact single 'givens' primitive of
    tensorcircuit-style construction via RXX/RYY, which realises
        exp(-i theta (XX+YY)/2)  on the |01>,|10> subspace.
    """
    nact = data["nact"]
    # nearest-neighbour Givens sweep on each spin sector
    pairs = [(p, p + 1) for p in range(nact - 1)]
    seq = pairs if not dagger else list(reversed(pairs))
    for (p, q) in seq:
        theta = kappa[q, p]
        if dagger:
            theta = -theta
        if abs(theta) < 1e-12:
            continue
        for s in (0, 1):                      # alpha and beta sectors
            i, j = qubit_index(p, s), qubit_index(q, s)
            _givens(c, i, j, theta)
    return c


def _givens(c, i, j, theta):
    """Givens rotation G(theta) = exp(-i theta (X_i X_j + Y_i Y_j)/2).
    Implemented with the explicit 2-CNOT decomposition (Problem 2.4, eq. 21)."""
    half_pi = np.pi / 2.0
    c.rx(i, theta=half_pi)
    c.cnot(i, j)
    c.rx(i, theta=theta)
    c.ry(j, theta=theta)
    c.cnot(i, j)
    c.rx(i, theta=-half_pi)
    return c


def add_diagonal_coulomb(c, Jmat, data, local=False):
    """Apply exp(i J n n) = prod_{p<q} exp(i J_pq n_p n_q).
    Using n = (I - Z)/2 (Problem 2.4, eq. 9):
        exp(i J_pq n_p n_q) = (phase) RZ_p(J/2) RZ_q(J/2) RZZ_pq(-J/2)
    with RZZ(phi) = CNOT . RZ(phi) . CNOT.
    We map spatial-orbital coupling J_pq onto same-spin qubit pairs.

    local : bool
        If True, apply the "local" (nearest-neighbour) truncation of LUCJ:
        keep only R_ZZ between ADJACENT spatial orbitals (|p-q| == 1) and drop
        all non-adjacent couplings. The dropped terms are exactly the source of
        the LUCJ truncation error (which scales ~ lambda^2, see Optional
        Problem 1 / Problem 2.5.b).
    """
    nact = data["nact"]
    for p in range(nact):
        for q in range(p + 1, nact):
            if local and (q - p) != 1:          # <-- local truncation
                continue
            J = Jmat[p, q]
            if abs(J) < 1e-12:
                continue
            for s in (0, 1):
                a, b = qubit_index(p, s), qubit_index(q, s)
                # single-qubit RZ pieces
                c.rz(a, theta=-J / 2.0)   # RZ(theta)=exp(-i theta Z/2); want e^{-iJ Z_a/4}
                c.rz(b, theta=-J / 2.0)
                # RZZ_ab(-J/2): CNOT-RZ-CNOT
                c.cnot(a, b)
                c.rz(b, theta=J / 2.0)    # gives e^{-i(-J/2)Z_aZ_b/2} = e^{+iJ Z_aZ_b/4}
                c.cnot(a, b)
    return c


def add_paired_givens(c, theta_pairs, data, local=False):
    """Apply occupied<->virtual Givens rotations on BOTH spin sectors, which
    is what actually leaks HF amplitude onto the paired double excitations.

    local : bool
        If True, apply the LUCJ "local" truncation: an occupied<->virtual
        double p->q is realised by a chain of nearest-neighbour Givens on the
        1D qubit line. Non-adjacent (|p-q| > 1) doubles would require a
        long-range Givens (equivalently, a chain of R_ZZ / SWAP-networked
        Givens) whose non-adjacent R_ZZ pieces are DROPPED by the local
        truncation. In this minimal LiH active space that means only the
        adjacent double (|p-q| == 1) survives; the non-adjacent dominant
        doubles are lost, and their missing weight is the truncation error
        (~ lambda^2).
    """
    for (p, q, theta) in theta_pairs:
        if local and abs(q - p) != 1:          # <-- local truncation
            continue
        for s in (0, 1):                       # alpha and beta simultaneously
            i, j = qubit_index(p, s), qubit_index(q, s)
            _givens(c, i, j, theta)
    return c


def build_circuit(data, theta_pairs=None, kappa=None, Jmat=None, local=False):
    """Full state-preparation circuit.

    theta_pairs/kappa/Jmat = None  ->  plain HF state.
    Otherwise builds |Psi_LUCJ> = U e^{iJ} U^dagger |HF> where the dominant
    correlation is injected by the occupied<->virtual paired Givens rotations
    (theta_pairs); the weak t1 orbital rotation and diagonal-Coulomb phases are
    applied as refinements, exactly following the LUCJ structure.

    local : bool
        If False -> FULL UCJ: every orbital pair contributes (no truncation),
        so growing lambda monotonically improves the sampled subspace and the
        SQD energy decreases toward FCI.
        If True  -> LUCJ with the "local" (nearest-neighbour) truncation:
        only adjacent-orbital R_ZZ / Givens are kept, non-adjacent ones are
        dropped. The dropped correlation is a truncation error growing ~
        lambda^2, which competes with the sampling-coverage gain and produces
        an OPTIMAL lambda* (the mechanism behind Problem 2.5.b / Optional 1).
    """
    nq = 2 * data["nact"]
    c = tc.Circuit(nq)
    add_hf_state(c, data)
    if theta_pairs is not None:
        add_orbital_rotation(c, kappa, data, dagger=True)          # U^dagger (weak)
        add_diagonal_coulomb(c, Jmat, data, local=local)           # e^{iJ} phases
        add_paired_givens(c, theta_pairs, data, local=local)       # dominant doubles
        add_orbital_rotation(c, kappa, data, dagger=False)         # U (weak)
    return c


def hf_state_index(data):
    """Flat state-vector index of the HF computational basis state.

    Helper used by the lambda scan to compute the exact probability that a
    measurement leaves the HF configuration. Defined here (with the other
    circuit primitives) rather than in the driver so it can be reused.
    """
    nq = 2 * data["nact"]
    occ = hf_occupied_qubits(data)
    idx = 0
    for q in occ:
        idx |= (1 << (nq - 1 - q))          # qubit q is the (nq-1-q)-th bit
    return idx


# ============================================================================
# 4. Sampling (TensorCircuit, TF backend)
# ============================================================================
def sample_bitstrings(c, nq, shots, seed=0):
    """Sample `shots` computational-basis bitstrings.

    Returns an (shots, nq) int array with column q = value of qubit q.

    Convention: in TensorCircuit the flat state-vector index `idx` orders
    qubit 0 as the MOST-significant bit and qubit (nq-1) as the least
    significant, i.e.  idx = sum_q b[q] * 2**(nq-1-q).  We therefore decode
        b[q] = (idx >> (nq-1-q)) & 1
    (verified: preparing X on q0,q1 of an 8-qubit register yields idx=192).
    """
    tf = tc.backend
    np.random.seed(seed)
    # Exact probability vector, then multinomial sampling (robust & fast <=8q)
    psi = c.state()
    amp = np.asarray(tf.numpy(psi))
    probs = np.abs(amp) ** 2
    probs = probs / probs.sum()
    nstate = probs.shape[0]
    idx = np.random.choice(nstate, size=shots, p=probs)
    shifts = (nq - 1 - np.arange(nq))[None, :]          # q0 -> MSB
    bits = ((idx[:, None] >> shifts) & 1).astype(int)   # column q = qubit q
    return bits, idx


# ============================================================================
# 5. SQD: exact diagonalization in the sampled configuration subspace
# ============================================================================
def bits_to_determinant(bitrow, nact):
    """Convert a qubit bitstring (b[q]=value of qubit q, q=2p+s) into
    (alpha_occ_tuple, beta_occ_tuple) spatial-orbital occupation lists."""
    a_occ, b_occ = [], []
    for p in range(nact):
        if bitrow[qubit_index(p, 0)]:
            a_occ.append(p)
        if bitrow[qubit_index(p, 1)]:
            b_occ.append(p)
    return tuple(a_occ), tuple(b_occ)


def particle_number_ok(det, nelec_act):
    na = nelec_act // 2
    return len(det[0]) == na and len(det[1]) == na


def sqd_energy(bits, data):
    """Selected-CI (SQD) energy: collect unique valid determinants sampled,
    build the CI matrix H_kl = <D_k|H|D_l> in that subspace, diagonalize.

    We reuse PySCF's FCI machinery: build the full-space CI Hamiltonian action
    restricted to the sampled address set. Because the active space is small
    (dim 36 total for (4o,2e) singlet-adaptable, here we work in the full
    spin-resolved FCI space of dimension C(4,1)*C(4,1)=16), exact projection is
    cheap and unambiguous.
    """
    nact = data["nact"]
    na = nb = data["nelec_act"] // 2
    h1, eri, e_core = data["h1"], data["eri"], data["e_core"]

    # --- enumerate the sampled, particle-number-conserving determinants ----
    dets = set()
    n_valid = 0
    for row in bits:
        det = bits_to_determinant(row, nact)
        if particle_number_ok(det, data["nelec_act"]):
            dets.add(det)
            n_valid += 1
    dets = sorted(dets)
    if len(dets) == 0:
        return dict(energy=np.nan, ndet=0, nunique=0, nvalid=0)

    # --- map determinant -> full FCI address (alpha_addr, beta_addr) -------
    from pyscf.fci import cistring
    strs_a = cistring.make_strings(range(nact), na)
    strs_b = cistring.make_strings(range(nact), nb)
    addr_a = {int(s): k for k, s in enumerate(strs_a)}
    addr_b = {int(s): k for k, s in enumerate(strs_b)}

    def occ_to_str(occ):
        s = 0
        for o in occ:
            s |= (1 << o)
        return s

    dim_a, dim_b = len(strs_a), len(strs_b)

    # Build a projector onto the sampled subspace and use FCI H|v> to form the
    # subspace Hamiltonian matrix exactly.
    sub_addrs = []
    for (aocc, bocc) in dets:
        ia = addr_a[occ_to_str(aocc)]
        ib = addr_b[occ_to_str(bocc)]
        sub_addrs.append(ia * dim_b + ib)
    sub_addrs = np.array(sorted(set(sub_addrs)))
    m = len(sub_addrs)

    # Absorb h1 into effective 1-body for FCI contraction
    h2e = fci.direct_spin1.absorb_h1e(h1, eri, nact, (na, nb), 0.5)

    Hsub = np.zeros((m, m))
    for col, addr in enumerate(sub_addrs):
        v = np.zeros(dim_a * dim_b)
        v[addr] = 1.0
        v = v.reshape(dim_a, dim_b)
        hv = fci.direct_spin1.contract_2e(h2e, v, nact, (na, nb)).reshape(-1)
        Hsub[:, col] = hv[sub_addrs]
    Hsub = 0.5 * (Hsub + Hsub.T)   # symmetrize (numerical safety)

    evals = np.linalg.eigvalsh(Hsub)
    e_sqd = evals[0] + e_core
    return dict(
        energy=float(e_sqd),
        ndet=m,
        nunique=len(dets),
        nvalid=n_valid,
        subspace_dim=m,
    )


# ============================================================================
# 6. Multi-layer (double-factorized) LUCJ
#    Ported / adapted from the cross-worker's study in code/2.5-review2
#    (problem_2_5_lucj_sqd.py), re-expressed in THIS module's conventions
#    (interleaved qubits q = 2p + s, MSB-first indexing, TF backend).
#
#    Ansatz:   |Psi> = ( prod_{k=1..L} U_k e^{i J_k} U_k^dagger ) |HF>
#
#    Layers come from the eigendecomposition ("double factorization") of the
#    paired-double channel of CCSD t2:
#        T[a,b] = t2[0,0,a,b]  (symmetric, nvir x nvir)
#        T = sum_k g_k v_k v_k^T
#    Layer k: U_k rotates the occupied orbital 0 into the collective virtual
#    mode w_k = sum_a v_k[a] |vir a> by a FIXED O(1) angle theta; the diagonal
#    Jastrow phase phi_k = g_k * lambda is applied between the alpha- and
#    beta-occupied qubits in the rotated frame.  First-order expansion gives a
#    paired-double AMPLITUDE ~ sin^2(theta) * phi_k, i.e. linear in g_k*lambda
#    (probability ~ (g_k lambda)^2 / 4 for theta = pi/4).
#
#    Why this is legitimate for SQD even though theta = pi/4 makes the state a
#    poor *wavefunction*: SQD only needs the right determinants to APPEAR in
#    the samples -- amplitudes are re-optimised by the subspace
#    diagonalisation.  "Support over fidelity."
# ============================================================================
def ccsd_t2_active(data):
    """CCSD confined EXACTLY to the active window: freeze the core orbital(s)
    AND every virtual above the active space, so t2 has shape
    (nocc_act, nocc_act, nvir_act, nvir_act) and maps 1:1 onto the nact
    spatial orbitals (no dangling external virtuals)."""
    mf = data["mf"]
    nmo = mf.mo_coeff.shape[1]
    ncore, nact = data["ncore"], data["nact"]
    frozen = list(range(ncore)) + list(range(ncore + nact, nmo))
    mycc = cc.CCSD(mf, frozen=frozen).run()
    e_corr = float(mycc.e_corr) if mycc.e_corr is not None else 0.0
    return np.asarray(mycc.t1), np.asarray(mycc.t2), e_corr


def df_layers(data, t2, tol=1e-8):
    """Double-factorize the paired-double channel of t2 into layers.

    Returns a list of (g_k, kappa_k), ordered by decreasing |g_k|, where
    kappa_k is the (nact x nact) antisymmetric generator rotating occupied
    orbital 0 into the k-th collective virtual direction v_k (unit norm).
    The physical rotation angle is supplied later (theta_fixed)."""
    nocc = data["nelec_act"] // 2
    nvir = data["nact"] - nocc
    tmat = np.array(t2[0, 0, :nvir, :nvir])
    tmat = 0.5 * (tmat + tmat.T)
    g, V = np.linalg.eigh(tmat)
    order = np.argsort(-np.abs(g))
    layers = []
    for k in order:
        if abs(g[k]) < tol:
            continue
        vk = V[:, k]
        kappa = np.zeros((data["nact"], data["nact"]))
        for a in range(nvir):
            kappa[0, nocc + a] = vk[a]
            kappa[nocc + a, 0] = -vk[a]
        layers.append((float(g[k]), kappa))
    return layers


def givens_network_from_kappa(kappa):
    """Decompose the orthogonal rotation Q = exp(kappa) into a sequence of
    adjacent-row Givens rotations (QR-style elimination).  Returns a list of
    (p, q=p+1, theta): applying these left-to-right on the orbital register
    builds Q.  All rotations act on ADJACENT spatial orbitals, so the network
    is compatible with the 'local' nearest-neighbour connectivity of LUCJ."""
    from scipy.linalg import expm
    qmat = expm(kappa)
    n = qmat.shape[0]
    m = qmat.copy()
    rots = []
    for col in range(n):
        for row in range(n - 1, col, -1):
            a, b = m[row - 1, col], m[row, col]
            if abs(b) < 1e-14:
                continue
            th = np.arctan2(b, a)
            c_, s_ = np.cos(th), np.sin(th)
            rmat = np.eye(n)
            rmat[row - 1, row - 1] = c_
            rmat[row - 1, row] = s_
            rmat[row, row - 1] = -s_
            rmat[row, row] = c_
            m = rmat @ m
            rots.append((row - 1, row, th))
    # Left-to-right the rotations reduce Q to (signed) identity; to BUILD Q
    # apply them reversed with negated angles.
    return [(p, q, -th) for (p, q, th) in reversed(rots)]


def _jastrow_pair(c, a, b, phi):
    """e^{i phi n_a n_b} (up to a global phase), same primitive/sign
    conventions as `add_diagonal_coulomb`."""
    c.rz(a, theta=-phi / 2.0)
    c.rz(b, theta=-phi / 2.0)
    c.cnot(a, b)
    c.rz(b, theta=phi / 2.0)
    c.cnot(a, b)
    return c


def build_multilayer_circuit(data, layers, lam, theta_fixed=np.pi / 4.0,
                             n_layers=None):
    """DF-LUCJ:  |Psi> = prod_k U_k e^{i J_k} U_k^dagger |HF>.

    Amplitude-in-phase scaling (the crux, after the cross-worker's finding):
    theta is FIXED at an O(1) value; the CCSD physics and the lambda knob live
    entirely in the Jastrow phase phi_k = g_k * lambda.  Naively setting
    theta_k = sqrt(|g_k| lam) squares the tiny t2 weight and the paired
    doubles never become samplable.

    n_layers : use only the first n layers (ablation studies); None = all."""
    nq = 2 * data["nact"]
    c = tc.Circuit(nq)
    add_hf_state(c, data)
    use = layers if n_layers is None else layers[:n_layers]
    for (g, kappa) in use:
        phi = g * lam
        net = givens_network_from_kappa(theta_fixed * kappa)
        # U_k^dagger on both spin sectors
        for (p, q, th) in reversed(net):
            for s in (0, 1):
                _givens(c, qubit_index(p, s), qubit_index(q, s), -th)
        # diagonal-Coulomb phase between alpha- and beta-occupied qubits
        # (qubits 0 and 1 in the interleaved layout -- nearest neighbours)
        _jastrow_pair(c, qubit_index(0, 0), qubit_index(0, 1), phi)
        # U_k
        for (p, q, th) in net:
            for s in (0, 1):
                _givens(c, qubit_index(p, s), qubit_index(q, s), th)
    return c


def det_state_index(data, a_occ, b_occ):
    """Flat state-vector index of the determinant with the given alpha/beta
    spatial occupations (interleaved layout, qubit 0 = MSB)."""
    nq = 2 * data["nact"]
    bits = [0] * nq
    for p in a_occ:
        bits[qubit_index(p, 0)] = 1
    for p in b_occ:
        bits[qubit_index(p, 1)] = 1
    idx = 0
    for b in bits:
        idx = (idx << 1) | b
    return idx


def paired_double_probability(data, psi):
    """(p_hf, p_paired): probability on the HF determinant and the TOTAL
    probability on the paired doubles |a_alpha a_beta> (both electrons in the
    same virtual a)."""
    p = np.abs(np.asarray(psi)) ** 2
    nocc = data["nelec_act"] // 2
    p_hf = float(p[det_state_index(data, tuple(range(nocc)),
                                   tuple(range(nocc)))])
    p_paired = 0.0
    for a in range(nocc, data["nact"]):
        p_paired += float(p[det_state_index(data, (a,), (a,))])
    return p_hf, p_paired


def support_bits(psi, nq, thresh=1e-12):
    """Bit rows (shots x nq layout) of every computational-basis state with
    probability above `thresh` -- the 'infinite-sample' determinant pool."""
    p = np.abs(np.asarray(psi)) ** 2
    idxs = np.where(p > thresh)[0]
    shifts = (nq - 1 - np.arange(nq))[None, :]
    return ((idxs[:, None] >> shifts) & 1).astype(int)


def dets_to_bits(data, dets):
    """Convert a list of ((a_occ),(b_occ)) determinants into a bit-row array
    consumable by `sqd_energy` (for hand-built subspaces, e.g. the
    variational floor {HF} + all paired doubles)."""
    nq = 2 * data["nact"]
    rows = np.zeros((len(dets), nq), dtype=int)
    for r, (aocc, bocc) in enumerate(dets):
        for p in aocc:
            rows[r, qubit_index(p, 0)] = 1
        for p in bocc:
            rows[r, qubit_index(p, 1)] = 1
    return rows
