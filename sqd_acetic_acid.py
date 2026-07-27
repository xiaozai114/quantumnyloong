
import numpy as np
from functools import partial #绑定签名，方便调用
from pyscf import gto, scf, mcscf, ao2mo, cc
import ffsim
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.primitives import StatevectorSampler
from qiskit_addon_sqd.counts import generate_bit_array_uniform
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch


# 建分子（醋酸 CH₃COOH）
mol = gto.M(atom="""
    C    0.0000   0.0000   0.0000    # C1 甲基碳
    C    0.0000   0.0000   1.5000    # C2 羰基碳
    O    0.0000   1.2000   2.1000    # O1 羰基氧 (C=O)
    O    0.0000  -1.2000   2.1000    # O2 羟基氧 (C-O)
    H    0.0000  -1.0000  -0.5000    # H1 甲基H
    H    0.8660   0.5000  -0.5000    # H2 甲基H
    H   -0.8660   0.5000  -0.5000    # H3 甲基H
    H    0.0000  -2.0000   1.6000    # H4 羟基H
""", basis="sto-3g", verbose=0)

mf = scf.RHF(mol)
mf.max_cycle = 100
mf.init_guess = 'vsap'
mf.kernel()
assert mf.converged, "HF 没收敛！"
print(f"  HF 能量 = {mf.e_tot:.6f} Ha")

print(f"\n  轨道能量（前 24 个）：")
n_total = mol.nao_nr()
# 冻结所有内层
n_frozen = 4   
print(f"  冻结 {n_frozen} 个 1s 内层轨道")

remaining = n_total - n_frozen   
occ_idx = np.where(mf.mo_occ > 0)[0]
vir_idx = np.where(mf.mo_occ == 0)[0]
remaining_occ = [i for i in occ_idx if i >= n_frozen]
remaining_vir = [i for i in vir_idx if i >= n_frozen]

n_occ_active = min(8, len(remaining_occ))   # HOMO 以下 6 个
n_vir_active = min(8, len(remaining_vir))   # LUMO 以上 6 个
active_occ = remaining_occ[-n_occ_active:]   
active_vir = remaining_vir[:n_vir_active]  
active_space = sorted(active_occ + active_vir)

norb = len(active_space)
n_electrons = int(sum(mf.mo_occ[active_space]))
nelec = ((n_electrons + mol.spin)//2, (n_electrons - mol.spin)//2)
print(f"  活性空间: 选 {norb} 轨道（HOMO±{n_occ_active}, LUMO±{n_vir_active}）, {nelec} 电子")
print(f"  活性轨道能量: {[f'{mf.mo_energy[i]:+.3f}' for i in active_space]}")

cas = mcscf.CASCI(mf, norb, nelec)
mo = cas.sort_mo(active_space, base=0)
cas.run()
exact_energy = cas.e_tot
print(f"  CASCI 精确能量（活性空间内FCI）= {exact_energy:.6f} Ha")

hcore, nre = cas.get_h1cas(mo)
eri = ao2mo.restore(1, cas.get_h2cas(mo), norb)
print(f"  hcore 形状 = {hcore.shape}, eri 形状 = {eri.shape}")
print(f"  nre = {nre:.6f} Ha")


frozen_list = [i for i in range(n_total) if i not in active_space]
ccsd = cc.CCSD(mf, frozen=frozen_list)
ccsd.max_cycle = 200
ccsd.run()
print(f"  CCSD 能量 = {ccsd.e_tot:.6f} Ha")

aa = [(p, p+1) for p in range(norb-1)]
ab = [(p, p) for p in range(0, norb, 2)]
ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
    t2=ccsd.t2, t1=ccsd.t1, n_reps=3, interaction_pairs=(aa, ab))

q = QuantumRegister(2*norb, name="q")
circuit = QuantumCircuit(q)
circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), q)
circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), q)
circuit.measure_all()

sampler = StatevectorSampler(seed=np.random.default_rng(42))
bit_array_lucj = sampler.run([circuit], shots=50000).result()[0].data.meas
counts = bit_array_lucj.get_counts()
print(f"  LUCJ 采样: {bit_array_lucj.num_bits} 比特, {len(counts)} 不同组态")

def run_sqd(bit_array, label, samples_per_batch=500, max_iter=10):
    rng = np.random.default_rng(42)
    sci_solver = partial(solve_sci_batch, spin_sq=0.0, max_cycle=200)
    history = []

    def callback(results):
        best = min(results, key=lambda r: r.energy).energy + nre
        history.append(best)
        it = len(history)
        dim = int(np.prod(results[0].sci_state.amplitudes.shape))
        print(f"    [{label}] 迭代 {it}: 能量={best:.6f}, 子空间={dim}, 误差={abs(best-exact_energy):.6f}")

    result = diagonalize_fermionic_hamiltonian(
        hcore, eri, bit_array,
        samples_per_batch=samples_per_batch,
        norb=norb, nelec=nelec,
        num_batches=1, energy_tol=1e-3, occupancies_tol=1e-3,
        max_iterations=max_iter, sci_solver=sci_solver,
        symmetrize_spin=True, carryover_threshold=1e-6,
        callback=callback, seed=rng,
    )
    sqd_energy = result.energy + nre
    return sqd_energy, history

print(f"\n  CASCI 精确 = {exact_energy:.6f} Ha")
print(f"\n--- LUCJ 采样 ---")
e_lucj, hist_lucj = run_sqd(bit_array_lucj, "LUCJ",max_iter=10,samples_per_batch=500)
print(f"  最终: SQD(LUCJ) = {e_lucj:.6f}, 误差 = {abs(e_lucj-exact_energy):.6f}")

print("\n" + "=" * 65)
print("计算 CCSD(T) 全轨道作为近似 FCI 标尺...")
print("=" * 65)

# CCSD(T) 冻同样轨道（和 CASCI 同活性空间，看激发截断）
ccsd_frozen = cc.CCSD(mf, frozen=frozen_list)
ccsd_frozen.max_cycle = 200
ccsd_frozen.kernel()
e_ccsd_frozen = ccsd_frozen.e_tot
try:
    e_t_frozen = ccsd_frozen.ccsd_t()
    e_ccsdt_frozen = e_ccsd_frozen + e_t_frozen
except:
    e_ccsdt_frozen = e_ccsd_frozen

# CCSD(T) 全轨道（不冻结，近似 FCI，看活性空间截断）
ccsd_full = cc.CCSD(mf)
ccsd_full.max_cycle = 200
ccsd_full.kernel()
try:
    e_t_full = ccsd_full.ccsd_t()
    e_ccsdt_full = ccsd_full.e_tot + e_t_full
except:
    e_ccsdt_full = ccsd_full.e_tot

print(f"  CCSD(T) 冻{norb}轨道 = {e_ccsdt_frozen:.6f} Ha")
print(f"  CCSD(T) 全{n_total}轨道 = {e_ccsdt_full:.6f} Ha  ← 近似 FCI 标尺")

# 用 CCSD(T) 全轨道当标尺
ref_energy = e_ccsdt_full

print("\n" + "=" * 65)
print("醋酸 SQD 误差分析（双标尺对比）")
print("=" * 65)
print(f"""
  分子: 醋酸 CH₃COOH（8 原子, 32 电子）
  基组: sto-3g（{n_total} 轨道）
  活性空间: {norb} 轨道, {nelec} 电子

  ┌─── 各方法能量 ───┐
  HF               = {mf.e_tot:.6f} Ha
  CCSD（冻{norb}）      = {ccsd.e_tot:.6f} Ha
  CCSD(T)（冻{norb}）   = {e_ccsdt_frozen:.6f} Ha
  CASCI（{norb}轨道）   = {exact_energy:.6f} Ha  ← 活性空间内精确
  CCSD(T)（全{n_total}） = {e_ccsdt_full:.6f} Ha  ← 近似 FCI（金标准）
  SQD(LUCJ)        = {e_lucj:.6f} Ha

  ┌─── 误差对比（两个标尺）───┐

  标尺1: CASCI（活性空间内精确）
    SQD 误差 = {abs(e_lucj - exact_energy):.6f} Ha
    ← 只含子空间覆盖误差（③）+ 采样（④）+ 配置恢复（⑤）
    ← 不含活性空间截断（②）和基组截断（①）

  标尺2: CCSD(T) 全轨道（近似 FCI）
    CASCI 误差   = {abs(exact_energy - ref_energy):.6f} Ha  ← 活性空间截断（②）
    CCSD 误差    = {abs(ccsd.e_tot - ref_energy):.6f} Ha  ← 激发截断 + 活性空间截断
    SQD 总误差   = {abs(e_lucj - ref_energy):.6f} Ha  ← 子空间 + 活性空间截断

  ┌─── 误差分解（以 CCSD(T) 全轨道为标尺）───┐
    ② 活性空间截断 = {abs(exact_energy - ref_energy):.6f} Ha  (CASCI vs CCSD(T)全)
    ③ 子空间覆盖   = {abs(e_lucj - exact_energy):.6f} Ha  (SQD vs CASCI)
    总误差         = {abs(e_lucj - ref_energy):.6f} Ha  (SQD vs CCSD(T)全)

  结论:
    {'活性空间截断是主因' if abs(exact_energy - ref_energy) > abs(e_lucj - exact_energy) else '子空间覆盖是主因'}
    活性空间截断 {abs(exact_energy - ref_energy)/abs(e_lucj - ref_energy)*100:.0f}% + 子空间覆盖 {abs(e_lucj - exact_energy)/abs(e_lucj - ref_energy)*100:.0f}%
""")
