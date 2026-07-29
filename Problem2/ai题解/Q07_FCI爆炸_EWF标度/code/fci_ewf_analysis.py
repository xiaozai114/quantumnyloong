#!/usr/bin/env python3
"""Q07: FCI 空间爆炸与 EWF 标度优势的数值验证."""

from math import comb, log10
from itertools import product

# ── (a) 五分子 FCI 维数 ──
molecules = [
    ("H2",   2,  2),
    ("LiH",  4,  4),
    ("H2O",  7, 10),
    ("N2",  10, 14),
    ("C2H4",14, 16),
]

print("=" * 60)
print("(a) FCI 维数表")
print("=" * 60)
print(f"{'分子':<8} {'N(轨道)':<10} {'Ne(电子)':<10} {'D=2N选Ne':<15} {'log10(D)':<10}")
print("-" * 60)
for name, N, Ne in molecules:
    D = comb(2 * N, Ne)
    print(f"{name:<8} {N:<10} {Ne:<10} {D:<15} {log10(D):<10.2f}")

# ── (c) C2H4 最大碎片 vs 全分子 ──
print("\n" + "=" * 60)
print("(c) C2H4 最大碎片 FCI 维数")
print("=" * 60)
D_full = comb(28, 16)
D_frag = comb(20, 10)
print(f"  全分子 D_full  = C(28,16) = {D_full}")
print(f"  最大碎片 D_frag = C(20,10) = {D_frag}")
print(f"  比值 D_frag/D_full = {D_frag/D_full:.6f} ≈ {D_frag/D_full*100:.2f}%")
print(f"  对角化代价比 D_full^3 / (2*D_frag^3) = {D_full**3 / (2*D_frag**3):.2e}")

# ── (d) 验证不等式 C(2N,Ne) > t^(1/3) * C(2nfrag, Ne*nfrag/N) ──
print("\n" + "=" * 60)
print("(d) 验证 EWF 恒优于 FCI 的不等式")
print("    C(2N,Ne) > t^(1/3) * C(2nfrag, Ne*nfrag/N),  t=N/nfrag")
print("=" * 60)
print(f"{'分子':<8} {'N':>4} {'Ne':>4} {'nfrag':>6} {'t':>6} "
      f"{'C(2N,Ne)':>12} {'t^(1/3)*C(frag)':>15} {'比值':>8} {'成立':>6}")
print("-" * 75)

test_cases = [
    ("H2",   2,  2, 1),
    ("LiH",  4,  4, 2),
    ("H2O",  7, 10, 3),
    ("N2",  10, 14, 5),
    ("C2H4",14, 16, 7),
    ("C2H4",14, 16, 10),
]

for name, N, Ne, nfrag in test_cases:
    if nfrag > N:
        continue
    t = N / nfrag
    # 碎片电子数（均匀分布，取整数）
    Nfrag = round(Ne * nfrag / N)
    if Nfrag < 0 or Nfrag > 2 * nfrag:
        continue
    lhs = comb(2 * N, Ne)
    rhs = t ** (1/3) * comb(2 * nfrag, Nfrag)
    ratio = lhs / rhs
    ok = "✓" if lhs > rhs else "✗"
    print(f"{name:<8} {N:>4} {Ne:>4} {nfrag:>6} {t:>6.1f} "
          f"{lhs:>12} {rhs:>15.1f} {ratio:>8.2f} {ok:>6}")

# ── 进阶: NISQ 约束 nfrag <= 10 ──
print("\n" + "=" * 60)
print("进阶: 20 qubit NISQ 约束 (nfrag <= 10)")
print("=" * 60)
for name, N, Ne in [("N2", 10, 14), ("C2H4", 14, 16)]:
    nfrag_opt = min(N, 10)
    F = -(-N // nfrag_opt)  # ceil
    Nfrag = round(Ne * nfrag_opt / N)
    D_frag = comb(2 * nfrag_opt, Nfrag)
    print(f"  {name}: N={N}, Ne={Ne}, 最优 nfrag={nfrag_opt}, "
          f"F={F}, Nfrag={Nfrag}, D_frag={D_frag}")
