"""
Problem 2, Part 1(a): H₂ 第二量子化哈密顿量构建
===================================================
构建 H₂/STO-3G 分子的第二量子化电子哈密顿量:
  Ĥ = Σ h_pq a†_p a_q + ½ Σ (pq|rs) a†_p a†_r a_s a_q + E_nuc

流程: 分子 → RHF → AO积分 → MO积分 → 自旋轨道 → FermionOperator

Tencent Sparking Program 2026 — Quantum Computing
"""

import numpy as np
from pyscf import gto, scf

# ================================================================
# 1. 构建 H₂ 分子 (STO-3G, R=0.74 Å)
# ================================================================
mol = gto.M(
    atom="H 0 0 0; H 0 0 0.74",
    basis="sto-3g",
    spin=0,
    verbose=0,
)

print("=" * 64)
print("Problem 2 — 1(a): H₂ 第二量子化哈密顿量构建")
print("=" * 64)
print(f"\n分子:  H₂, R=0.74 Å, STO-3G (最小基组)")
print(f"  电子数  = {mol.nelectron}")
print(f"  AO 数   = {mol.nao}  (每个 H 一个 1s)")
print(f"  MO 数   = {mol.nao}  (σ 成键 + σ* 反键)")
print(f"  自旋轨道 = {2 * mol.nao}  (每个 MO → α, β 两个自旋轨道)")

# ================================================================
# 2. RHF 计算 → 分子轨道系数 C 和轨道能量 ε
# ================================================================
mf = scf.RHF(mol)
mf.kernel()

norb = mol.nao
nspin = 2 * norb

print(f"\n{'─' * 40}")
print(f"RHF 轨道能量")
print(f"{'─' * 40}")
print(f"  ε₀ (σ)  = {mf.mo_energy[0]:.10f} Ha")
print(f"  ε₁ (σ*) = {mf.mo_energy[1]:.10f} Ha")
print(f"\nMO 系数 C (AO → MO 变换矩阵):")
print(f"\n{mf.mo_coeff}")

# ================================================================
# 3. 提取积分 (MO 基组)
# ================================================================

# 3a. 单电子积分 h_pq (AO → MO)
h1e_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
h1e_mo = mf.mo_coeff.T @ h1e_ao @ mf.mo_coeff

print(f"\n{'─' * 40}")
print(f"单电子积分 h_pq (MO 基组)")
print(f"{'─' * 40}")
print(f"  h = \n{h1e_mo}")
print(f"  对角元 (正则轨道能量):  [{h1e_mo[0,0]:.10f},  {h1e_mo[1,1]:.10f}]")
print(f"  非对角元 (≈0, 正则HF):  {h1e_mo[0,1]:.2e}")

# 3b. 双电子积分 (pq|rs) (AO → MO, 化学记号)
eri_ao = mol.intor('int2e_sph')
eri_mo = np.einsum(
    'pqrs,pi,qj,rk,sl->ijkl',
    eri_ao, *([mf.mo_coeff] * 4), optimize=True,
)

print(f"\n{'─' * 40}")
print(f"双电子积分 (pq|rs) (MO 基组, 化学记号)")
print(f"{'─' * 40}")
print(f"  形状: {eri_mo.shape}  (共 {eri_mo.size} 个元素)")
print(f"  非零元素:")
for p in range(norb):
    for q in range(norb):
        for r in range(norb):
            for s in range(norb):
                val = eri_mo[p, q, r, s]
                if abs(val) > 1e-12:
                    label = ''
                    if (p,q,r,s) == (0,0,0,0):
                        label = '  ← J₀₀ (σ 自库仑)'
                    elif (p,q,r,s) == (0,0,1,1):
                        label = '  ← J₀₁ (σ-σ* 库仑)'
                    elif (p,q,r,s) == (0,1,0,1):
                        label = '  ← K₀₁ (σ-σ* 交换)'
                    elif (p,q,r,s) == (1,1,1,1):
                        label = '  ← J₁₁ (σ* 自库仑)'
                    print(f"  ({p}{q}|{r}{s}) = {val:12.8f}{label}")

# 3c. 核排斥能
E_nuc = mol.energy_nuc()
print(f"\n  核排斥能 E_nuc = 1/R = {E_nuc:.10f} Ha")

# ================================================================
# 4. 展开到自旋轨道基组
# ================================================================
# 每个空间轨道 φ_p → φ_{p,↑}(自旋指标 2p) 和 φ_{p,↓}(自旋指标 2p+1)
# 单电子积分:  只有同自旋的块非零
# 双电子积分:  自旋守恒 → αααα, ααββ, ββαα, ββββ 四个块
h1e_spin = np.zeros((nspin, nspin))
for p in range(norb):
    for q in range(norb):
        h1e_spin[2*p, 2*q] = h1e_mo[p, q]
        h1e_spin[2*p+1, 2*q+1] = h1e_mo[p, q]

eri_spin = np.zeros((nspin, nspin, nspin, nspin))
for p in range(norb):
    for q in range(norb):
        for r in range(norb):
            for s in range(norb):
                v = eri_mo[p, q, r, s]
                if abs(v) < 1e-14:
                    continue
                # αααα
                eri_spin[2*p, 2*q, 2*r, 2*s] = v
                # ααββ
                eri_spin[2*p, 2*q, 2*r+1, 2*s+1] = v
                # ββαα
                eri_spin[2*p+1, 2*q+1, 2*r, 2*s] = v
                # ββββ
                eri_spin[2*p+1, 2*q+1, 2*r+1, 2*s+1] = v

print(f"\n{'─' * 40}")
print(f"自旋轨道基组")
print(f"{'─' * 40}")
print(f"  空间轨道: {norb}  →  自旋轨道: {nspin}")
print(f"  指标:  0=α₀, 1=β₀, 2=α₁, 3=β₁")
print(f"\n  单电子积分 h (4×4), 非零元素:")
for i in range(nspin):
    for j in range(nspin):
        if abs(h1e_spin[i,j]) > 1e-14:
            print(f"    h[{i},{j}] = {h1e_spin[i,j]:14.10f}")

print(f"\n  双电子积分 (pq|rs) 非零元素数: {np.count_nonzero(np.abs(eri_spin) > 1e-14)}")

# ================================================================
# 5. 构建第二量子化哈密顿量
# ================================================================
# Ĥ = E_nuc + Σ h_pq a†_p a_q + ½ Σ (pq|rs) a†_p a†_r a_s a_q
#
# 注意: 化学记号 (pq|rs) 对应算符 a†_p a†_r a_s a_q
# 因为 (pq|rs) = ∫ φ*_p(r₁) φ_q(r₁) φ*_r(r₂) φ_s(r₂) / r₁₂ dr₁dr₂
# 对于 Hermitian 算符，两个产生算符应该对应 φ*_p 和 φ*_r

from openfermion import FermionOperator

ham = FermionOperator()

# 单电子项: Σ h_pq a†_p a_q
for p in range(nspin):
    for q in range(nspin):
        if abs(h1e_spin[p, q]) > 1e-14:
            ham += FermionOperator(f"{p}^ {q}", h1e_spin[p, q])

# 双电子项: ½ Σ (pq|rs) a†_p a†_r a_s a_q
for p in range(nspin):
    for q in range(nspin):
        for r in range(nspin):
            for s in range(nspin):
                val = 0.5 * eri_spin[p, q, r, s]
                if abs(val) > 1e-14:
                    ham += FermionOperator(f"{p}^ {r}^ {s} {q}", val)

# 核排斥 (常数项)
ham += FermionOperator((), E_nuc)

print(f"\n{'─' * 40}")
print(f"第二量子化哈密顿量 (FermionOperator)")
print(f"{'─' * 40}")
print(f"  总项数: {len(ham.terms)}")
print(f"\n  所有项:")
for i, (term, coeff) in enumerate(ham.terms.items()):
    op_str = str(term) if term else '(常数'
    print(f"  [{i:2d}]  {coeff:14.10f}  {op_str}")

print(f"""
{'=' * 64}
  最终结果:
  ┌─────────────────────────────────────────────────────────────────┐
  │ Ĥ = {E_nuc:.4f} + Σ h_pq a†_p a_q + ½ Σ (pq|rs) a†_p a†_r a_s a_q │
  │ 共 {len(ham.terms)} 项, 4 个自旋轨道 (α₀, β₀, α₁, β₁), 2 个电子       │
  └─────────────────────────────────────────────────────────────────┘

  验证参考:
  　　 E_RHF = {mf.e_tot:.10f} Ha   (Szabo & Ostlund Table 3.5: −1.1167 Ha)
  　　 E_FCI = −1.1373 Ha            (本基组精确解, 见 extras/ 子目录完整脚本)
""")
