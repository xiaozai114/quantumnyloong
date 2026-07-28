"""
lucj_sqd_framework.py
=====================
A well-encapsulated, easily tunable LUCJ-SQD experiment framework.

Tunable knobs (all via the `SQDConfig` dataclass):
  - molecule            : one of the built-in Problem-2 molecules, or a custom spec
  - bond / geometry     : override bond length (diatomics) or full atom string
  - active space        : ncas / nelecas (defaults follow the problem's qubit count)
  - ccsd_scale (lambda) : LUCJ amplitude scaling (0 -> HF, larger -> more correlated)
  - n_reps              : number of LUCJ repetitions
  - local_truncation    : keep only adjacent-qubit RZZ ("local"), or full UCJ
  - shots (S)           : number of sampled bitstrings
  - noise_model         : "none" | "bitflip" | "depolarizing" | "readout"
  - noise params        : p_flip, p_2q, p_1q, p_read0, p_read1
  - config_recovery     : toggle SQD configuration recovery (particle-number repair)
  - samples_per_batch / num_batches / max_iterations : SQD solver controls
  - n_seeds             : repeat sampling+SQD for statistics

Only pre-installed libraries are used: pyscf, ffsim, qiskit, qiskit-addon-sqd.

Typical use
-----------
    from lucj_sqd_framework import SQDConfig, LUCJSQDExperiment
    cfg = SQDConfig(molecule="LiH", ccsd_scale=0.5, shots=3000,
                    noise_model="bitflip", p_flip=0.02, n_seeds=20)
    exp = LUCJSQDExperiment(cfg)
    res = exp.run()
    print(res.summary())

    # lambda sweep helper:
    sweep = exp.sweep("ccsd_scale", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np
from pyscf import gto, scf, mcscf, cc, fci, ao2mo
import ffsim
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian


# ----------------------------------------------------------------------
# Built-in molecule library (STO-3G).  Active spaces follow the qubit
# count stated in Problem 2 (= 2 * n_active_orbitals).  See NOTE below.
# ----------------------------------------------------------------------
# NOTE: For H2O (14q) and N2 (20q) the problem's "frozen core" remark is
# inconsistent with the stated qubit count; we honor the qubit count and
# use the full space.  LiH (8q) uses the consistent frozen-core reading.
MOLECULE_LIBRARY = {
    "H2":   dict(atom="H 0 0 0; H 0 0 {d}", default_bond=0.74, ncas=2,  nelecas=2),
    "LiH":  dict(atom="Li 0 0 0; H 0 0 {d}", default_bond=1.55, ncas=4, nelecas=2),
    "H2O":  dict(atom=("O 0.0000 0.0000 0.1173; "
                       "H 0.0000 0.7572 -0.4692; "
                       "H 0.0000 -0.7572 -0.4692"), default_bond=None, ncas=7, nelecas=10),
    "N2":   dict(atom="N 0 0 0; N 0 0 {d}", default_bond=1.10, ncas=10, nelecas=14),
    "C2H4": dict(atom=("C 0.0000 0.0000 0.6685; C 0.0000 0.0000 -0.6685; "
                       "H 0.0000 0.9227 1.2321; H 0.0000 -0.9227 1.2321; "
                       "H 0.0000 0.9227 -1.2321; H 0.0000 -0.9227 -1.2321"),
                 default_bond=None, ncas=14, nelecas=16),
}


@dataclass
class SQDConfig:
    # --- system ---
    molecule: str = "LiH"
    bond: Optional[float] = None           # override diatomic bond length (Angstrom)
    atom: Optional[str] = None             # fully custom geometry string (wins over `molecule`)
    basis: str = "sto-3g"
    ncas: Optional[int] = None             # override active orbitals
    nelecas: Optional[int] = None          # override active electrons

    # --- LUCJ ansatz ---
    ccsd_scale: float = 0.5                 # lambda
    n_reps: int = 1
    local_truncation: bool = True           # keep only adjacent RZZ

    # --- sampling ---
    shots: int = 3000
    n_seeds: int = 10
    base_seed: int = 20260727

    # --- noise ---
    # Post-hoc bitstring models (fast, incoherent):
    #   none | bitflip | depolarizing | readout
    # Gate-level Aer models (realistic, coherent trajectory simulation):
    #   aer_depolarizing | aer_thermal
    noise_model: str = "none"
    p_flip: float = 0.0                     # per-bit flip prob (bitflip model)
    p_2q: float = 0.01                      # two-qubit depolarizing rate
    p_1q: float = 0.001                     # single-qubit depolarizing rate
    p_read0: float = 0.0                    # P(measure 1 | true 0)  (readout)
    p_read1: float = 0.0                    # P(measure 0 | true 1)  (readout)
    # Aer gate-level options:
    aer_method: str = "statevector"         # statevector (trajectory) | density_matrix
    aer_basis: tuple = ("rz", "sx", "x", "cx")
    readout_error: float = 0.0              # symmetric readout error added to Aer model
    t1_us: float = 100.0                    # thermal relaxation T1 (us), aer_thermal
    t2_us: float = 80.0                     # thermal relaxation T2 (us), aer_thermal
    gate_time_1q_ns: float = 35.0
    gate_time_2q_ns: float = 300.0

    # --- SQD solver ---
    config_recovery: bool = True            # SQD configuration recovery (iterative)
    samples_per_batch: Optional[int] = None
    num_batches: int = 3
    max_iterations: int = 6
    symmetrize_spin: bool = True

    def resolved_active_space(self):
        if self.atom is not None:
            return self.atom, self.ncas, self.nelecas
        lib = MOLECULE_LIBRARY[self.molecule]
        atom = lib["atom"]
        if "{d}" in atom:
            d = self.bond if self.bond is not None else lib["default_bond"]
            atom = atom.format(d=d)
        ncas = self.ncas if self.ncas is not None else lib["ncas"]
        nelecas = self.nelecas if self.nelecas is not None else lib["nelecas"]
        return atom, ncas, nelecas


@dataclass
class SQDResult:
    config: dict
    norb: int
    nelec: tuple
    e_hf: float
    e_ccsd: float
    e_fci: float
    e_sqd_mean: float
    e_sqd_std: float
    e_sqd_best: float
    err_mean_mha: float
    err_best_mha: float
    n_configs_mean: float
    trunc_infid: float
    n_2q: int
    n_1q: int
    depth: int
    eps_eff: float
    walltime_s: float

    def summary(self) -> str:
        c = self.config
        return (
            f"[{c['molecule']}] norb={self.norb} nelec={self.nelec} "
            f"lambda={c['ccsd_scale']} shots={c['shots']} noise={c['noise_model']} "
            f"recovery={c['config_recovery']}\n"
            f"  E_HF={self.e_hf:.6f}  E_CCSD={self.e_ccsd:.6f}  E_FCI={self.e_fci:.6f}\n"
            f"  E_SQD={self.e_sqd_mean:.6f} +/- {self.e_sqd_std*1e3:.2f} mHa "
            f"(best={self.e_sqd_best:.6f})\n"
            f"  err_mean={self.err_mean_mha:+.3f} mHa  err_best={self.err_best_mha:+.3f} mHa  "
            f"M={self.n_configs_mean:.1f}\n"
            f"  gates: N_2q={self.n_2q} N_1q={self.n_1q} depth={self.depth} "
            f"eps_eff={self.eps_eff:.4f}  trunc_infid={self.trunc_infid:.2e}  "
            f"({self.walltime_s:.1f}s)"
        )


class LUCJSQDExperiment:
    """Builds the molecule once; `run()` executes the pipeline for the current config."""

    def __init__(self, config: SQDConfig):
        self.cfg = config
        self._build_system()

    # ---------- system + amplitudes ----------
    def _build_system(self):
        atom, ncas, nelecas = self.cfg.resolved_active_space()
        mol = gto.M(atom=atom, basis=self.cfg.basis, verbose=0)
        mf = scf.RHF(mol).run()
        mc = mcscf.CASCI(mf, ncas, nelecas)
        h1e, ecore = mc.get_h1eff()
        mo_cas = mf.mo_coeff[:, mc.ncore:mc.ncore + ncas]
        eri = ao2mo.restore(1, np.asarray(mc.get_h2eff(mo_cas)), ncas)
        na = nelecas // 2
        nelec = (na, nelecas - na)

        # fake RHF on active integrals -> CCSD amplitudes for LUCJ
        molf = gto.M(verbose=0); molf.nelectron = nelecas; molf.incore_anyway = True
        fk = scf.RHF(molf)
        fk.get_hcore = lambda *a: h1e
        fk.get_ovlp = lambda *a: np.eye(ncas)
        fk._eri = ao2mo.restore(8, eri, ncas)
        fk.energy_nuc = lambda *a: ecore
        fk.kernel()
        mycc = cc.CCSD(fk).run()
        e_fci, _ = fci.direct_spin1.FCI().kernel(h1e, eri, ncas, nelec, ecore=ecore)

        self.h1e, self.eri, self.ecore = h1e, eri, ecore
        self.norb, self.nelec = ncas, nelec
        self.t1, self.t2 = mycc.t1, mycc.t2
        self.e_hf, self.e_ccsd, self.e_fci = fk.e_tot, mycc.e_tot, e_fci
        self._linop = ffsim.linear_operator(
            ffsim.MolecularHamiltonian(h1e, eri, constant=ecore),
            norb=ncas, nelec=nelec)

    # ---------- ansatz ----------
    def _interaction_pairs(self):
        if not self.cfg.local_truncation:
            return None
        aa = [(p, p + 1) for p in range(self.norb - 1)]
        ab = [(p, p) for p in range(self.norb)]
        return (aa, ab)

    def _lucj_state(self, lam):
        hf = ffsim.hartree_fock_state(self.norb, self.nelec)
        if lam == 0.0:
            return hf
        ucj = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            lam * self.t2, t1=lam * self.t1,
            n_reps=self.cfg.n_reps, interaction_pairs=self._interaction_pairs())
        return ffsim.apply_unitary(hf, ucj, norb=self.norb, nelec=self.nelec)

    def _gate_counts(self, lam):
        """Count 1q/2q gates of the LUCJ JW circuit -> effective depolarizing error."""
        try:
            from ffsim import qiskit as fq
            from qiskit import QuantumCircuit, QuantumRegister, transpile
            if lam == 0.0:
                return 0, self.norb, 1
            ucj = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
                lam * self.t2, t1=lam * self.t1,
                n_reps=self.cfg.n_reps, interaction_pairs=self._interaction_pairs())
            qr = QuantumRegister(2 * self.norb)
            qc = QuantumCircuit(qr)
            qc.append(fq.PrepareHartreeFockJW(self.norb, self.nelec), qr)
            qc.append(fq.UCJOpSpinBalancedJW(ucj), qr)
            qc2 = transpile(qc, basis_gates=["rz", "rx", "ry", "cx", "x", "h"],
                            optimization_level=1)
            ops = qc2.count_ops()
            n2q = ops.get("cx", 0)
            n1q = sum(v for k, v in ops.items()
                      if k in ["rz", "rx", "ry", "x", "h"])
            return n2q, n1q, qc2.depth()
        except Exception:
            return 0, 0, 0

    # ---------- noise on sampled bitstrings ----------
    def _apply_noise(self, samples, rng):
        """samples: list of '0/1' strings length 2*norb. Returns noisy list."""
        cfg = self.cfg
        if cfg.noise_model == "none":
            return samples
        arr = np.array([[int(b) for b in s] for s in samples], dtype=np.int8)

        if cfg.noise_model == "bitflip":
            mask = rng.random(arr.shape) < cfg.p_flip
            arr = np.where(mask, 1 - arr, arr)

        elif cfg.noise_model == "depolarizing":
            # gate-count-derived effective per-qubit flip rate (Optional-2 model):
            # eps_eff ~ (p_2q N_2q + p_1q N_1q) / 2, spread over the qubits.
            n2q, n1q, _ = self._gate_counts(cfg.ccsd_scale)
            eps = (cfg.p_2q * n2q + cfg.p_1q * n1q) / 2.0
            per_qubit = min(0.5, eps / max(1, arr.shape[1]))
            mask = rng.random(arr.shape) < per_qubit
            arr = np.where(mask, 1 - arr, arr)

        elif cfg.noise_model == "readout":
            # asymmetric readout error
            r = rng.random(arr.shape)
            flip0 = (arr == 0) & (r < cfg.p_read0)   # 0 -> 1
            flip1 = (arr == 1) & (r < cfg.p_read1)   # 1 -> 0
            arr = np.where(flip0 | flip1, 1 - arr, arr)
        else:
            raise ValueError(f"unknown noise_model: {cfg.noise_model}")

        return ["".join(map(str, row)) for row in arr]

    def _eps_eff(self):
        n2q, n1q, _ = self._gate_counts(self.cfg.ccsd_scale)
        return (self.cfg.p_2q * n2q + self.cfg.p_1q * n1q) / 2.0

    # ---------- Aer gate-level noisy sampling ----------
    def _build_lucj_circuit(self, lam):
        """Transpiled LUCJ JW circuit (HF prep + UCJ + measure) in Aer basis."""
        from ffsim import qiskit as fq
        from qiskit import QuantumCircuit, QuantumRegister, transpile
        qr = QuantumRegister(2 * self.norb)
        qc = QuantumCircuit(qr)
        qc.append(fq.PrepareHartreeFockJW(self.norb, self.nelec), qr)
        if lam > 0:
            ucj = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
                lam * self.t2, t1=lam * self.t1,
                n_reps=self.cfg.n_reps, interaction_pairs=self._interaction_pairs())
            qc.append(fq.UCJOpSpinBalancedJW(ucj), qr)
        qc.measure_all()
        return transpile(qc, basis_gates=list(self.cfg.aer_basis),
                         optimization_level=1)

    def _build_aer_noise_model(self):
        from qiskit_aer.noise import (
            NoiseModel, depolarizing_error, thermal_relaxation_error,
            ReadoutError)
        cfg = self.cfg
        nm = NoiseModel()
        one_q = [g for g in cfg.aer_basis if g != "cx"]
        if cfg.noise_model == "aer_depolarizing":
            if cfg.p_2q > 0:
                nm.add_all_qubit_quantum_error(depolarizing_error(cfg.p_2q, 2), ["cx"])
            if cfg.p_1q > 0:
                nm.add_all_qubit_quantum_error(depolarizing_error(cfg.p_1q, 1), one_q)
        elif cfg.noise_model == "aer_thermal":
            t1 = cfg.t1_us * 1e3   # ns
            t2 = cfg.t2_us * 1e3
            e1 = thermal_relaxation_error(t1, t2, cfg.gate_time_1q_ns)
            e2 = thermal_relaxation_error(t1, t2, cfg.gate_time_2q_ns)
            e2 = e2.expand(thermal_relaxation_error(t1, t2, cfg.gate_time_2q_ns))
            nm.add_all_qubit_quantum_error(e1, one_q)
            nm.add_all_qubit_quantum_error(e2, ["cx"])
        else:
            raise ValueError(f"unknown aer noise_model: {cfg.noise_model}")
        if cfg.readout_error > 0:
            p = cfg.readout_error
            nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
        return nm

    def _sample_aer(self, lam, shots, seed):
        """Return list of bitstrings from a gate-level noisy Aer simulation."""
        from qiskit_aer import AerSimulator
        qc = self._build_lucj_circuit(lam)
        nm = self._build_aer_noise_model()
        sim = AerSimulator(method=self.cfg.aer_method, noise_model=nm)
        result = sim.run(qc, shots=shots, seed_simulator=int(seed)).result()
        counts = result.get_counts()
        samples = []
        for bitstr, c in counts.items():
            key = bitstr.replace(" ", "")
            samples.extend([key] * c)
        return samples

    # ---------- SQD ----------
    def _run_sqd(self, bit_array, rng):
        cfg = self.cfg
        spb = cfg.samples_per_batch or max(50, cfg.shots // 3)
        if cfg.config_recovery:
            # full iterative SQD with configuration recovery
            res = diagonalize_fermionic_hamiltonian(
                self.h1e, self.eri, bit_array,
                samples_per_batch=spb, norb=self.norb, nelec=self.nelec,
                num_batches=cfg.num_batches, max_iterations=cfg.max_iterations,
                symmetrize_spin=cfg.symmetrize_spin, seed=rng)
            return res.energy + self.ecore
        else:
            # no recovery: post-select correct Hamming weight, single diagonalization
            from qiskit_addon_sqd.fermion import (
                solve_fermion, bitstring_matrix_to_ci_strs)
            from qiskit_addon_sqd.counts import counts_to_arrays
            arr = bit_array.array
            mat = np.unpackbits(arr, axis=1, bitorder="big")[:, -2 * self.norb:]
            mat = mat.astype(bool)
            na, nb = self.nelec
            left = mat[:, :self.norb]
            right = mat[:, self.norb:]
            keep = (left.sum(1) == nb) & (right.sum(1) == na)
            mat = mat[keep]
            if len(mat) == 0:
                return np.nan
            uniq = np.unique(mat, axis=0)
            e, _, _, _ = solve_fermion(uniq, self.h1e, self.eri, spin_sq=0.0)
            return e + self.ecore

    # ---------- main entry ----------
    def run(self) -> SQDResult:
        cfg = self.cfg
        t0 = time.time()
        lam = cfg.ccsd_scale
        psi = self._lucj_state(lam)

        # truncation infidelity (local vs full) for diagnostics
        if cfg.local_truncation and lam > 0:
            psi_full = ffsim.apply_unitary(
                ffsim.hartree_fock_state(self.norb, self.nelec),
                ffsim.UCJOpSpinBalanced.from_t_amplitudes(
                    lam * self.t2, t1=lam * self.t1, n_reps=cfg.n_reps),
                norb=self.norb, nelec=self.nelec)
            trunc_infid = float(1 - abs(np.vdot(psi, psi_full)) ** 2)
        else:
            trunc_infid = 0.0

        use_aer = cfg.noise_model.startswith("aer_")
        energies, ms = [], []
        for s in range(cfg.n_seeds):
            rng = np.random.default_rng(cfg.base_seed + 1000 * s)
            if use_aer:
                # gate-level noisy sampling (coherent trajectory simulation)
                samples = self._sample_aer(lam, cfg.shots, cfg.base_seed + 1000 * s)
            else:
                # ideal statevector sampling + post-hoc incoherent bitstring noise
                samples = ffsim.sample_state_vector(
                    psi, norb=self.norb, nelec=self.nelec, shots=cfg.shots,
                    bitstring_type=ffsim.BitstringType.STRING, seed=rng)
                samples = self._apply_noise(samples, rng)
            ms.append(len(set(samples)))
            ba = BitArray.from_samples(samples, num_bits=2 * self.norb)
            try:
                energies.append(self._run_sqd(ba, rng))
            except Exception:
                energies.append(np.nan)

        e = np.array(energies, dtype=float)
        e_mean = float(np.nanmean(e))
        e_std = float(np.nanstd(e))
        e_best = float(np.nanmin(e))
        n2q, n1q, depth = self._gate_counts(lam)

        return SQDResult(
            config=asdict(cfg), norb=self.norb, nelec=self.nelec,
            e_hf=self.e_hf, e_ccsd=self.e_ccsd, e_fci=self.e_fci,
            e_sqd_mean=e_mean, e_sqd_std=e_std, e_sqd_best=e_best,
            err_mean_mha=(e_mean - self.e_fci) * 1000,
            err_best_mha=(e_best - self.e_fci) * 1000,
            n_configs_mean=float(np.mean(ms)),
            trunc_infid=trunc_infid,
            n_2q=n2q, n_1q=n1q, depth=depth, eps_eff=self._eps_eff(),
            walltime_s=time.time() - t0)

    def sweep(self, param: str, values: Sequence):
        """Vary one config field over `values`; returns list of SQDResult."""
        results = []
        original = getattr(self.cfg, param)
        for v in values:
            setattr(self.cfg, param, v)
            results.append(self.run())
        setattr(self.cfg, param, original)
        return results


# ----------------------------------------------------------------------
# CLI demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo 1: LiH noiseless, single run ===")
    exp = LUCJSQDExperiment(SQDConfig(molecule="LiH", ccsd_scale=0.5,
                                      shots=3000, n_seeds=10, noise_model="none"))
    print(exp.run().summary())

    print("\n=== Demo 2: LiH lambda sweep (noiseless) ===")
    for r in exp.sweep("ccsd_scale", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]):
        print(f"  lambda={r.config['ccsd_scale']:.2f}: "
              f"err_mean={r.err_mean_mha:+7.3f} mHa  M={r.n_configs_mean:.1f}")

    print("\n=== Demo 3: LiH with bit-flip noise, recovery ON vs OFF ===")
    for rec in (True, False):
        cfg = SQDConfig(molecule="LiH", ccsd_scale=0.5, shots=3000, n_seeds=10,
                        noise_model="bitflip", p_flip=0.03, config_recovery=rec)
        r = LUCJSQDExperiment(cfg).run()
        print(f"  recovery={rec}: err_mean={r.err_mean_mha:+7.3f} mHa "
              f"(best={r.err_best_mha:+.3f})  M={r.n_configs_mean:.1f}")

    print("\n=== Demo 4: LiH depolarizing noise (gate-count derived eps_eff) ===")
    cfg = SQDConfig(molecule="LiH", ccsd_scale=0.5, shots=3000, n_seeds=10,
                    noise_model="depolarizing", p_2q=0.01, p_1q=0.001)
    r = LUCJSQDExperiment(cfg).run()
    print(r.summary())

    print("\n=== Demo 5: LiH Aer GATE-LEVEL depolarizing, recovery ON vs OFF ===")
    for rec in (True, False):
        cfg = SQDConfig(molecule="LiH", ccsd_scale=0.5, shots=3000, n_seeds=8,
                        noise_model="aer_depolarizing", p_2q=0.01, p_1q=0.001,
                        config_recovery=rec)
        r = LUCJSQDExperiment(cfg).run()
        print(f"  recovery={rec}: err_mean={r.err_mean_mha:+8.3f} mHa "
              f"(best={r.err_best_mha:+.3f})  M={r.n_configs_mean:.1f}")

    print("\n=== Demo 6: LiH Aer gate-error sweep (p_2q), recovery ON ===")
    exp5 = LUCJSQDExperiment(SQDConfig(molecule="LiH", ccsd_scale=0.5, shots=3000,
                                       n_seeds=8, noise_model="aer_depolarizing",
                                       p_1q=0.0, config_recovery=True))
    for r in exp5.sweep("p_2q", [0.0, 0.005, 0.01, 0.02, 0.05]):
        print(f"  p_2q={r.config['p_2q']:.3f}: err_mean={r.err_mean_mha:+8.3f} mHa "
              f"M={r.n_configs_mean:.1f}")
