"""
O01：最优 ccsd_scale (λ) 扫描

分子：LiH / sto-3g（4 轨道，8 量子比特）
λ ∈ [0, 0.5]，扫描 LUCJ-SQD 能量曲线

用法：
  cd /Users/zhouzihan/Desktop/sqd
  python /Users/zhouzihan/WorkBuddy/2026-07-25-16-51-21/Problem2_Solutions/O01_最优ccsd_scale/code/o01_solution.py
"""

import sys, os, time
import numpy as np

sys.path.insert(0, '/Users/zhouzihan/Desktop/sqd')

import common.backend  # noqa: F401
import tensorcircuit as tc
from common.chemistry import molecule_report
from common.circuits import prepare_hf, build_lucj, sample_counts
from common.sqd import sqd_from_counts

# ================================================================
# 1. 取 LiH 积分
# ================================================================
rep = molecule_report("LiH", do_of=False)
norb, nocc = rep["norb"], rep["nocc"]
nq = 2 * norb
h1e, eri, ecore = rep["h1e"], rep["eri"], rep["ecore"]
t1, t2 = rep["t1"], rep["t2"]

print(f"LiH: norb={norb}, nocc={nocc}, nq={nq}")
print(f"  E_HF  = {rep['E_HF']:.6f}")
print(f"  E_FCI = {rep['E_FCI']:.6f}")
print(f"  关联能 = {(rep['E_FCI']-rep['E_HF'])*1000:.3f} mHa")

# ================================================================
# 2. λ 扫描
# ================================================================
lam_list = np.arange(0.0, 0.525, 0.025)
n_shots = 8000
max_dets = 8000

results = []
hf_bs = "".join("1" if q < 2 * nocc else "0" for q in range(nq))

print(f"\n{'λ':>6} {'E_SQD':>14} {'误差(mHa)':>10} {'组态数':>8} {'M':>6} {'时间(s)':>8}")
print("-" * 58)

for lam in lam_list:
    t0 = time.time()

    # λ=0 时 LUCJ 退化为 HF（t1=t2=0），电路只有 prepare_hf
    c = tc.Circuit(nq)
    prepare_hf(c, norb, nocc)
    if lam > 1e-10:
        build_lucj(c, norb, nocc, t1, t2, eri=eri,
                   ccsd_scale=lam, local=True, doubles=True)

    # 采样
    counts = sample_counts(c, n_shots, nq, seed=42)

    # SQD
    res = sqd_from_counts(counts, nq, nocc, nocc, h1e, eri, ecore,
                          hf_bs=hf_bs, max_dets=max_dets, method="directed")
    e_sqd = res["E_sqd"]
    err = abs(e_sqd - rep["E_FCI"]) * 1000
    dt = time.time() - t0

    results.append(dict(lam=lam, E_sqd=e_sqd, err=err,
                        n_configs=len(counts), M=res["M"], time=dt))
    print(f"{lam:>6.2f} {e_sqd:>14.6f} {err:>10.3f} {len(counts):>8} {res['M']:>6} {dt:>8.2f}")

# ================================================================
# 3. 找最优 λ
# ================================================================
best = min(results, key=lambda r: r["err"])
print(f"\n最优 λ* = {best['lam']:.2f}")
print(f"  E_SQD = {best['E_sqd']:.6f}")
print(f"  误差  = {best['err']:.3f} mHa")
print(f"  组态  = {best['n_configs']}, M = {best['M']}")

# ================================================================
# 4. 理论推导验证：ε(λ) = A/λ² + Bλ²
# ================================================================
print(f"\n{'='*60}")
print("理论模型：ε(λ) = A/λ² + Bλ²")
print(f"{'='*60}")

# 用实测数据拟合 A, B（排除 λ=0）
valid = [r for r in results if r["lam"] > 0.01 and r["err"] > 0.001]
if len(valid) >= 3:
    # log(ε) = log(A/λ² + Bλ²) 非线性，用简化拟合
    # 小 λ 区间：ε ≈ A/λ² → log(ε) ≈ log(A) - 2log(λ)
    # 大 λ 区间：ε ≈ Bλ² → log(ε) ≈ log(B) + 2log(λ)
    small = [r for r in valid if r["lam"] <= 0.15]
    large = [r for r in valid if r["lam"] >= 0.25]

    if small:
        log_lam_s = np.log([r["lam"] for r in small])
        log_err_s = np.log([r["err"] for r in small])
        A = np.exp(np.mean(log_err_s) + 2 * np.mean(log_lam_s))
        print(f"  小 λ 区：ε ≈ A/λ², A ≈ {A:.4f}")

    if large:
        log_lam_l = np.log([r["lam"] for r in large])
        log_err_l = np.log([r["err"] for r in large])
        B = np.exp(np.mean(log_err_l) - 2 * np.mean(log_lam_l))
        print(f"  大 λ 区：ε ≈ Bλ², B ≈ {B:.4f}")

    if small and large:
        lam_star = (A / B) ** 0.25
        eps_star = A / lam_star**2 + B * lam_star**2
        print(f"\n  理论最优：λ* = (A/B)^(1/4) = {lam_star:.3f}")
        print(f"  理论最小误差：ε* = {eps_star:.3f} mHa")
        print(f"  实测最优：λ* = {best['lam']:.2f}, ε* = {best['err']:.3f} mHa")

print(f"""
竞争机制：
  λ ↑ → 采样覆盖率 ↑（更多激发组态）→ ε_coverage ∝ 1/λ² ↓
  λ ↑ → local 截断误差 ↑（非相邻 RZZ 丢弃 ∝ λ²）→ ε_truncation ∝ λ² ↑

  总误差 ε(λ) = A/λ² + Bλ²
  最小值在 dε/dλ = 0 → λ* = (A/B)^(1/4)

  λ < λ*：覆盖率不足主导（HF 组态太多，激发太少）
  λ > λ*：截断误差主导（LUCJ 波函数本身不准）
  λ = λ*：两者平衡（木桶效应的又一体现）
""")

# ================================================================
# 5. 输出 CSV（供画图）
# ================================================================
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lambda_scan.csv")
with open(csv_path, "w") as f:
    f.write("lambda,E_SQD,error_mHa,n_configs,M\n")
    for r in results:
        f.write(f"{r['lam']:.4f},{r['E_sqd']:.6f},{r['err']:.4f},{r['n_configs']},{r['M']}\n")
print(f"CSV 已保存: {csv_path}")
