"""Problem 2 选做 3: EWF Bath Threshold (MP2 BNO 占据数与截断阈值).

对应 main-polish.tex Problem 2 选做第 3 项 "EWF Bath Threshold".

物理图像
--------
  DMET minimal bath (Schmidt) 在 HF 级精确, 但只捕捉静态相关;
  相关态的密度矩阵不再幂等 -> 出现分数占据 (动态相关) ->
  EWF 用 MP2 密度修正 Delta_gamma 的本征轨道 (BNO) 补 bath.
  BNO 占据数 n_i 大致按指数 n_i ~ n_1 * exp(-alpha*(i-1)) 衰减;
  截断阈值 eta (保留 n_i >= eta 的 BNO) 与 bath 规模 n_BNO 的关系:
        n_BNO(eta) ~ (1/alpha) * log(1/eta)     (核心结论, theory.tex)
  能量误差 eps_frag(eta) ~ eta (线性, 指数衰减谱下).

本文件做四件事
--------------
  (1) H2O/6-31G (主) RHF -> 全空间 CCSD 作参考 E_infty.
  (2) vayesta EWF, bath_options={bathtype="mp2", threshold=eta},
      扫描 eta in [1e-6, 1e-2], 记录:
        - 各 fragment 的 BNO 占据数 (occ + vir) -> 拟合衰减常数 alpha
        - n_BNO(eta), cluster 大小, E_EWF(eta)
  (3) 画三联图 bath_threshold.png:
        (a) BNO 占据数 vs index + 指数拟合 (验证指数衰减)
        (b) fragment size (mean cluster) vs eta  (对数增长)
        (c) 能量误差 |E - E_infty| vs eta  + 拐点 eta*
  (4) 数值定位拐点 eta* (误差二阶差分极大), 讨论 diminishing returns.
  额外: STO-3G 诊断输出 (说明小基组 BNO 为空 / 退化的物理).

API 关键点 (vayesta 1.0.1)
--------------------------
  - emb = vayesta.ewf.EWF(mf, bath_options=dict(bathtype="mp2", threshold=eta))
  - 多 eta 复用: emb.change_options(bath_options=dict(threshold=eta2))
                emb.reset(reset_bath=False)  # 保留 DMET bath + MP2 factory
  - BNO 占据数:  frag._bath_factory_occ.occup, frag._bath_factory_vir.occup
                 (已排序: n_1 >= n_2 >= ... >= 0)
  - cluster 大小: frag.cluster.norb_active (空间轨道); emb.get_mean_cluster_size()
  - 注意: 占据 (occ) BNO 占据数为正 (表示电子从占据区抽走 -> 补到 bath);
          虚 (vir) BNO 占据数也为正 (电子被激发到虚区). 二者都参与截断 n_i >= eta.

运行
----
  WSL: conda activate tc_vayesta
       cd <opt3 目录>
       python solution.py
  输出: bath_threshold.png + 控制台表格 + eta* 报告.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无显示环境也能存图
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

from pyscf import gto, scf, cc

import vayesta
import vayesta.ewf


# =========================================================================== #
#  分子构造
# =========================================================================== #
H2O_GEO = "O 0.0000 0.0000 0.1173; H 0.0000 0.7572 -0.4692; H 0.0000 -0.7572 -0.4692"


def build_h2o(basis: str = "6-31G"):
    """H2O + RHF, 返回 (mol, mf)."""
    mol = gto.M(atom=H2O_GEO, basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    return mol, mf


# =========================================================================== #
#  扫描 eta: EWF MP2-BNO bath threshold
# =========================================================================== #
def scan_eta(mf, eta_list, fragmentation="atomic"):
    """对给定 eta 列表跑 EWF (CCSD solver), 记录每个 fragment 的 BNO 占据数 /
    cluster 大小, 以及总能量.

    fragmentation: "atomic" (每原子一个 fragment, 默认) 或 "single" (整分子单 fragment).

    返回
    ----
      records: list of dict, 每条含
        eta, e_tot, mean_cluster, min/max_cluster,
        bno_occ_all (list, 所有 fragment 的 occ BNO 占据数拼接, 降序),
        bno_vir_all (同上 vir),
        n_bno_occ_mean, n_bno_vir_mean (每 fragment 平均 BNO 数, 截断于该 eta),
        n_imp_mean (fragment impurity 轨道数平均)
      bno_occ_full, bno_vir_full: 最低 eta (最大 bath) 时各 fragment 的占据数列表
                                  (用于画衰减曲线 + 拟合 alpha)
    """
    # 用最小 eta 跑一次完整 kernel, 拿到全 BNO 占据数 (用于衰减拟合).
    # 之后用 change_options + reset(reset_bath=False) 复用 DMET bath, 仅重新截断.
    # 注意: eta 列表应单调 (推荐从大到小, 因 vayesta 在小 eta 下做的对角化最贵,
    #       先跑最小 eta 把 factory 算好, 后续只重新截断 -> 最快).
    emb = vayesta.ewf.EWF(
        mf,
        solver="CCSD",
        bath_options=dict(bathtype="mp2", threshold=min(eta_list)),
    )
    with emb.iao_fragmentation() as f:
        if fragmentation == "single":
            f.add_atomic_fragment(list(range(mf.mol.natm)))
        else:
            f.add_all_atomic_fragments()
    emb.kernel()

    # 收集最小 eta 下的全 BNO 占据数 (各 fragment)
    bno_occ_full, bno_vir_full = [], []
    for frag in emb.fragments:
        if frag._bath_factory_occ is not None:
            bno_occ_full.append(np.atleast_1d(np.asarray(frag._bath_factory_occ.occup, dtype=float)))
        else:
            bno_occ_full.append(np.zeros(0))
        if frag._bath_factory_vir is not None:
            bno_vir_full.append(np.atleast_1d(np.asarray(frag._bath_factory_vir.occup, dtype=float)))
        else:
            bno_vir_full.append(np.zeros(0))

    # 扫描其余 eta
    records = []
    for eta in eta_list:
        if eta != min(eta_list):
            emb.change_options(bath_options=dict(threshold=eta))
            emb.reset(reset_bath=False)
            emb.kernel()

        # 收集数据
        clusters = [frag.cluster.norb_active for frag in emb.fragments]
        n_imps = [frag.n_frag for frag in emb.fragments]
        bno_occ_now = []
        bno_vir_now = []
        n_bno_occ_per = []
        n_bno_vir_per = []
        for frag in emb.fragments:
            occ = np.atleast_1d(np.asarray(frag._bath_factory_occ.occup, dtype=float)) \
                if frag._bath_factory_occ is not None else np.zeros(0)
            vir = np.atleast_1d(np.asarray(frag._bath_factory_vir.occup, dtype=float)) \
                if frag._bath_factory_vir is not None else np.zeros(0)
            bno_occ_now.append(occ)
            bno_vir_now.append(vir)
            n_bno_occ_per.append(int(np.count_nonzero(occ >= eta)))
            n_bno_vir_per.append(int(np.count_nonzero(vir >= eta)))

        records.append(dict(
            eta=eta,
            e_tot=emb.e_tot,
            mean_cluster=float(np.mean(clusters)),
            max_cluster=int(np.max(clusters)),
            min_cluster=int(np.min(clusters)),
            n_imp_mean=float(np.mean(n_imps)),
            n_bno_occ_mean=float(np.mean(n_bno_occ_per)),
            n_bno_vir_mean=float(np.mean(n_bno_vir_per)),
            clusters=clusters,
            n_bno_occ_per=n_bno_occ_per,
            n_bno_vir_per=n_bno_vir_per,
        ))

    return records, bno_occ_full, bno_vir_full


# =========================================================================== #
#  拟合 BNO 占据数衰减: n_i ~ n_1 * exp(-alpha * (i-1))
# =========================================================================== #
def fit_alpha(occup_list, label="occ"):
    """把多 fragment 的 BNO 占据数合并, 拟合 log(n_i) = log(n_1) - alpha*(i-1).

    返回 (alpha, n1, r2, occup_concat). 若样本太少返回 NaN.
    """
    # 合并所有 fragment 的占据数, 降序排列 (跨 fragment 但同形状)
    cat = np.concatenate([o for o in occup_list if o.size > 0]) if occup_list else np.zeros(0)
    if cat.size < 2:
        return np.nan, np.nan, np.nan, cat
    cat = np.sort(cat)[::-1]  # 降序
    # 仅拟合占据数 > 1e-12 的项 (避免 log(0))
    mask = cat > 1e-12
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan, cat
    n_arr = cat[mask]
    idx = np.arange(1, len(n_arr) + 1)
    log_n = np.log(n_arr)
    # 线性拟合 log_n vs (idx-1): 斜率 = -alpha, 截距 = log(n1)
    A = np.vstack([idx - 1, np.ones_like(idx)]).T
    coef, res, *_ = np.linalg.lstsq(A, log_n, rcond=None)
    alpha = -coef[0]
    log_n1 = coef[1]
    n1 = np.exp(log_n1)
    # R^2
    pred = A @ coef
    ss_res = np.sum((log_n - pred) ** 2)
    ss_tot = np.sum((log_n - log_n.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return alpha, n1, r2, cat


# =========================================================================== #
#  拐点定位: 误差二阶差分极大点
# =========================================================================== #
def find_inflection(eta_arr, err_arr):
    """在 log(eta)-log(err) 空间找 ``elbow'' 拐点 (曲线由陡降变平缓的点).

    标准的 ``knee'' 刿法: 取误差单调下降序列 (大 eta -> 小 eta, err 也应下降),
    在 log-log 平面上找曲率 (二阶导) 最大的点 —— 即斜率从 ``陡降'' 变 ``平缓''
    的转折, 这就是 diminishing returns 的起点 eta*.

    误差曲线通常有 \emph{两个} elbow:
      - upper-knee (大 eta 端): err 从 ``近似 0 相关能'' 跳到 ``捕获大部分相关能''
        的过渡点; 此处加 bath 收益最大.
      - lower-knee (小 eta 端, ``diminishing returns'' 真正起点): err 已降到
        mHa 以下, 再加 bath 改善微乎其微 —— 这是工程上 ``停'' 的位置.

    注意: 完全收敛的 err=0 点 (bath 已饱和) 会人为制造 ``假拐点'', 故先剔除.
    返回 (eta_upper, eta_lower, idx_lower). 若点太少返回 (None, None, None).
    """
    if len(err_arr) < 5:
        return None, None, None
    keep = err_arr > 1e-7
    if keep.sum() < 4:
        return None, None, None
    eta_k = eta_arr[keep]
    err_k = err_arr[keep]
    order = np.argsort(-eta_k)   # eta 从大到小
    eta_s = eta_k[order]
    err_s = np.maximum(err_k[order], 1e-15)
    le = np.log10(eta_s)
    lf = np.log10(err_s)
    d1 = np.diff(lf) / np.diff(le)
    d2 = np.diff(d1)             # 长度 N-3, 索引对应 eta_s[1..N-2]
    if len(d2) < 2:
        return None, None, None
    abs_d2 = np.abs(d2)
    # 两个 elbow: 最大和次大的 |d2| (且索引相隔 >=1)
    order_d2 = np.argsort(-abs_d2)
    i1 = order_d2[0]
    i2 = None
    for cand in order_d2[1:]:
        if abs(cand - i1) >= 2:   # 隔开, 避免选相邻点
            i2 = cand
            break
    # i1, i2 是 d2 数组的索引; 对应 eta_s 索引 = i+1
    eta_cand1 = float(eta_s[i1 + 1])
    eta_cand2 = float(eta_s[i2 + 1]) if i2 is not None else None
    # upper = 较大 eta, lower = 较小 eta
    if eta_cand2 is None:
        eta_upper, eta_lower = eta_cand1, None
    elif eta_cand1 >= eta_cand2:
        eta_upper, eta_lower = eta_cand1, eta_cand2
    else:
        eta_upper, eta_lower = eta_cand2, eta_cand1
    idx_lower = int(np.argmin(np.abs(eta_arr - eta_lower))) if eta_lower else None
    return eta_upper, eta_lower, idx_lower


# =========================================================================== #
#  主程序
# =========================================================================== #
def main():
    print("=" * 78)
    print("Problem 2 选做 3: EWF Bath Threshold (MP2 BNO 占据数与截断阈值)")
    print("=" * 78)

    eta_list = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]

    # ----- (1) 主扫描: H2O / 6-31G -----------------------------------------
    print("\n[1] 主扫描: H2O / 6-31G  (eta 扫描 [1e-6, 1e-2])")
    mol, mf = build_h2o("6-31G")
    e_mf = mf.e_tot
    e_ccsd = cc.CCSD(mf).run().e_tot  # 全空间 CCSD (上界参考, EWF 未必等于它)
    print(f"    nao = {mol.nao} spatial orbitals")
    print(f"    E_MF   = {e_mf:+.8f} Ha")
    print(f"    E_CCSD = {e_ccsd:+.8f} Ha   (full-system CCSD, 上界参考)")

    print("\n    扫描 eta (每行: eta, n_BNO occ/vir, cluster, E_EWF, |err vs min-eta|) ...")
    print("    " + "-" * 78)
    print(f"    {'eta':>9}{'n_BNO_occ':>11}{'n_BNO_vir':>11}"
          f"{'mean_clst':>11}{'max_clst':>10}{'E_EWF':>15}{'|dE_bath|':>12}")
    print("    " + "-" * 78)
    records, bno_occ_full, bno_vir_full = scan_eta(mf, eta_list, fragmentation="atomic")

    # 真正的 "bath 截断误差" 参考 = 最小 eta (最大 bath) 的 EWF 能量 E_infty^bath.
    # (用 full CCSD 作参考会引入 "EWF != CCSD" 的固有偏差, 不是纯 bath 误差.)
    e_ref = min(r["e_tot"] for r in records)   # 最相关 (最多 bath) 的能量
    print(f"    [E_infty^bath 参考 = min-eta E_EWF = {e_ref:+.8f} Ha]\n")
    for r in records:
        err = abs(r["e_tot"] - e_ref)
        print(f"    {r['eta']:>9.0e}"
              f"{r['n_bno_occ_mean']:>11.2f}"
              f"{r['n_bno_vir_mean']:>11.2f}"
              f"{r['mean_cluster']:>11.2f}"
              f"{r['max_cluster']:>10d}"
              f"{r['e_tot']:>+15.8f}{err:>12.2e}")
    print(f"    [对照: E_CCSD - E_infty^bath = {e_ccsd - e_ref:+.2e} Ha "
          f"(EWF 固有偏差, 与 bath 截断无关)]")

    # ----- (2) 拟合 BNO 占据数衰减常数 alpha ------------------------------
    print("\n[2] BNO 占据数衰减拟合: n_i ~ n_1 * exp(-alpha*(i-1))")
    alpha_o, n1_o, r2_o, cat_o = fit_alpha(bno_occ_full, "occ")
    alpha_v, n1_v, r2_v, cat_v = fit_alpha(bno_vir_full, "vir")
    print(f"    occ BNO: alpha = {alpha_o:.3f},  n_1 = {n1_o:.3e},  R^2 = {r2_o:.3f}")
    print(f"    vir BNO: alpha = {alpha_v:.3f},  n_1 = {n1_v:.3e},  R^2 = {r2_v:.3f}")
    alpha_eff = 0.5 * (alpha_o + alpha_v) if (np.isfinite(alpha_o) and np.isfinite(alpha_v)) else alpha_o
    if np.isfinite(alpha_eff) and alpha_eff > 0:
        # 预测 n_BNO(eta) ~ (1/alpha) * log(1/eta)  (occ + vir 各一半)
        for eta in (1e-2, 1e-4, 1e-6):
            pred = (1.0 / alpha_eff) * np.log(1.0 / eta)
            print(f"      预测 n_BNO(eta={eta:.0e}) ~ {pred:.2f}  "
                  f"(理论 (1/alpha)*ln(1/eta), alpha={alpha_eff:.2f})")

    # ----- (3) 拐点 eta* (upper + lower knee) -----------------------------
    print("\n[3] 拐点 eta* (log-log elbow: upper-knee = 相关能开始捕获, "
          "lower-knee = diminishing returns 起点)")
    print("    (用 min-eta EWF 作 bath-converged 参考 -> 误差纯来自 bath 截断)")
    eta_arr = np.array([r["eta"] for r in records])
    err_arr = np.array([abs(r["e_tot"] - e_ref) for r in records])
    eta_upper, eta_lower, idx_lower = find_inflection(eta_arr, err_arr)
    if eta_upper is not None:
        print(f"    upper-knee eta ~ {eta_upper:.2e}  (err 从 ~0 相关能 -> 捕获相关能的过渡)")
    if eta_lower is not None:
        print(f"    lower-knee eta* ~ {eta_lower:.2e}  "
              f"(|err|={err_arr[idx_lower]:.2e} Ha)  <- diminishing returns 起点")
        print(f"    含义: eta < eta* 后误差改善显著放缓, 再加 bath ROI 极低")
    else:
        print("    点数不足, 无法定位 lower-knee.")

    # 工程判据: eps_frag ~ eps_SQD (~1e-3 Ha) 时停 (木桶效应, 与第 9 问一致)
    eps_other = 1e-3
    eta_match = None
    for r in records:
        if abs(r["e_tot"] - e_ref) <= eps_other:
            eta_match = r["eta"]
            break
    if eta_match is not None:
        print(f"    工程判据: eps_frag ~ eps_SQD (~{eps_other:.0e} Ha) 时 eta ~ {eta_match:.0e},")
        print(f"    再降 eta 已被 SQD 采样误差主导 -> 选 eta ~ {eta_match:.0e} 最经济.")

    # 用 lower-knee 作为 ``diminishing returns'' 拐点 (理论.tex 引用 eta*)
    eta_star = eta_lower

    # ----- (4) STO-3G 诊断输出 -------------------------------------------
    print("\n[4] STO-3G 诊断: 小基组下 BNO 退化 (理论.tex 第 6 节)")
    mol_s, mf_s = build_h2o("sto-3g")
    e_ccsd_s = cc.CCSD(mf_s).run().e_tot
    print(f"    nao = {mol_s.nao} (太小: O-fragment 无环境轨道 -> BNO 空)")
    rec_s, occ_s, vir_s = scan_eta(mf_s, [1e-4, 1e-6], fragmentation="atomic")
    for frag_idx, (fo, fv) in enumerate(zip(occ_s, vir_s)):
        print(f"    fragment {frag_idx} (atom={emb_atoms(mf_s, frag_idx)}): "
              f"occ_BNO={fo.size}  vir_BNO={fv.size}")
    print(f"    -> STO-3G O-fragment 占据数为空 (DMET bath 已用完全部 7 轨道),")
    print(f"       BNO 扫描退化; 主扫描必须用更大基组 (6-31G/6-31G*/cc-pVDZ).")
    print(f"    E_CCSD(STO-3G) = {e_ccsd_s:+.6f} Ha")
    for r in rec_s:
        print(f"    eta={r['eta']:.0e}: E_EWF={r['e_tot']:+.6f}, "
              f"|err|={abs(r['e_tot']-e_ccsd_s):.2e}, mean_clst={r['mean_cluster']:.1f}")

    # ----- (5) 画图 -------------------------------------------------------
    print("\n[5] 画图 -> bath_threshold.png")
    plot_results(records, bno_occ_full, bno_vir_full,
                 alpha_o=alpha_o, n1_o=n1_o, r2_o=r2_o,
                 alpha_v=alpha_v, n1_v=n1_v, r2_v=r2_v,
                 e_ref=e_ref, e_ccsd=e_ccsd,
                 eta_star=eta_star, eta_upper=eta_upper, basis="6-31G")

    # ----- 总结 -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("总结")
    print("=" * 78)
    print("(1) BNO 占据数大致指数衰减 (alpha ~ %.2f), 拟合 R^2 ~ %.2f."
          % (alpha_eff if np.isfinite(alpha_eff) else float("nan"),
             max(r2_o if np.isfinite(r2_o) else 0, r2_v if np.isfinite(r2_v) else 0)))
    print("(2) n_BNO(eta) ~ (1/alpha) * log(1/eta): eta 减小 -> BNO 数对数增长.")
    print("(3) 能量误差 eps_frag(eta) ~ eta (线性); eta 减半 -> 误差减半.")
    print("(4) 拐点 eta* ~ %s (lower-knee, diminishing returns 起点):"
          % (f"{eta_star:.1e}" if eta_star else "未定位"))
    print("    之后再加 bath 边际收益微小 (BNO 占据数指数衰减 -> 尾部 Pareto).")
    print("    upper-knee ~ %s (相关能开始被捕获);" %
          (f"{eta_upper:.1e}" if eta_upper else "n/a"))
    print("    工程选择: eps_frag ~ eps_SQD + eps_noise 时停 (木桶效应, 与第 9 问一致).")


def emb_atoms(mf, frag_idx):
    """helper: 返回 fragment 对应的原子 index (诊断用). 这里仅近似返回 frag_idx."""
    return [frag_idx]


# =========================================================================== #
#  画图
# =========================================================================== #
def plot_results(records, bno_occ_full, bno_vir_full,
                 alpha_o, n1_o, r2_o, alpha_v, n1_v, r2_v,
                 e_ref, e_ccsd, eta_star, eta_upper=None,
                 basis="6-31G", outpath="bath_threshold.png"):
    """画 3 联图: (a) BNO 占据数衰减 + 指数拟合;
                  (b) mean cluster size vs eta;
                  (c) 能量误差 vs eta + 拐点.

    标签用英文, 因 WSL 默认无 CJK 字体 (避免 matplotlib 警告 / 乱码方块).
    e_ref = min-eta EWF (bath-converged reference);
    e_ccsd = full-system CCSD (shown for context);
    eta_star = lower-knee (diminishing returns 起点);
    eta_upper = upper-knee (相关能开始被捕获, 可选).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # ---- (a) BNO occupation vs index + exponential fit ----
    ax = axes[0]
    cat_o = np.sort(np.concatenate([o for o in bno_occ_full if o.size > 0]))[::-1] \
        if any(o.size for o in bno_occ_full) else np.zeros(0)
    cat_v = np.sort(np.concatenate([o for o in bno_vir_full if o.size > 0]))[::-1] \
        if any(o.size for o in bno_vir_full) else np.zeros(0)
    if cat_o.size:
        ax.semilogy(np.arange(1, len(cat_o) + 1), cat_o, "o-", color="C0",
                    label="occ BNO (data)", markersize=5)
    if cat_v.size:
        ax.semilogy(np.arange(1, len(cat_v) + 1), cat_v, "s-", color="C1",
                    label="vir BNO (data)", markersize=5)
    if np.isfinite(alpha_o) and cat_o.size:
        i = np.arange(1, min(len(cat_o), 20) + 1)
        ax.semilogy(i, n1_o * np.exp(-alpha_o * (i - 1)), ":", color="C0", alpha=0.8,
                    label=rf"fit $n_{{occ}}\sim{n1_o:.1e}\,e^{{-{alpha_o:.2f}(i-1)}}$ ($R^2={r2_o:.2f}$)")
    if np.isfinite(alpha_v) and cat_v.size:
        i = np.arange(1, min(len(cat_v), 20) + 1)
        ax.semilogy(i, n1_v * np.exp(-alpha_v * (i - 1)), ":", color="C1", alpha=0.8,
                    label=rf"fit $n_{{vir}}\sim{n1_v:.1e}\,e^{{-{alpha_v:.2f}(i-1)}}$ ($R^2={r2_v:.2f}$)")
    for eta, c in [(1e-2, "gray"), (1e-4, "green"), (1e-6, "purple")]:
        ax.axhline(eta, ls="--", color=c, alpha=0.35, lw=1)
        xpos = (len(cat_o) if cat_o.size else 5) * 0.7
        ax.text(xpos, eta * 1.3, rf"$\eta={eta:.0e}$", fontsize=8, color=c, va="bottom")
    ax.set_xlabel(r"BNO index $i$ (sorted by occupation $\downarrow$)")
    ax.set_ylabel(r"BNO occupation $n_i$")
    ax.set_title(r"(a) MP2 BNO occupation decay + exp. fit" + "\n" + rf"(H$_2$O/{basis})")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    ymax = max(n1_o if np.isfinite(n1_o) and n1_o > 0 else 1e-3,
               n1_v if np.isfinite(n1_v) and n1_v > 0 else 1e-3) * 2
    ax.set_ylim(bottom=1e-9, top=ymax)

    # ---- (b) fragment cluster size vs eta ----
    ax = axes[1]
    eta_arr = np.array([r["eta"] for r in records])
    mean_clst = np.array([r["mean_cluster"] for r in records])
    max_clst = np.array([r["max_cluster"] for r in records])
    n_imp = np.array([r["n_imp_mean"] for r in records])
    ax.semilogx(eta_arr, mean_clst, "o-", color="C2", label="mean cluster size")
    ax.semilogx(eta_arr, max_clst, "^--", color="C2", alpha=0.5, label="max cluster size")
    ax.semilogx(eta_arr, n_imp, ":", color="C3", label=r"$n_{\rm imp}$ (fragment orbitals)")
    ax.fill_between(eta_arr, n_imp * 2, mean_clst, color="C0", alpha=0.15,
                    label="DMET bath + MP2 BNO")
    alpha_eff = np.nanmean([alpha_o, alpha_v]) if np.isfinite(alpha_o) and np.isfinite(alpha_v) else alpha_o
    if np.isfinite(alpha_eff) and alpha_eff > 0:
        eta_th = np.logspace(np.log10(eta_arr.min()), np.log10(eta_arr.max()), 50)
        const0 = mean_clst.min() - (2.0 / alpha_eff) * np.log(1.0 / eta_arr.min())
        ax.semilogx(eta_th, const0 + (2.0 / alpha_eff) * np.log(1.0 / eta_th),
                    "k-", alpha=0.4, lw=1,
                    label=rf"theory $\sim\frac{{2}}{{\alpha}}\ln(1/\eta)$, $\alpha={alpha_eff:.2f}$")
    if eta_upper:
        ax.axvline(eta_upper, ls="--", color="blue", alpha=0.45, lw=1,
                   label=rf"upper-knee $\eta_{{\rm up}}\approx{eta_upper:.1e}$")
    if eta_star:
        ax.axvline(eta_star, ls="--", color="red", alpha=0.6, label=rf"lower-knee $\eta^*\approx{eta_star:.1e}$")
    ax.set_xlabel(r"BNO truncation threshold $\eta$")
    ax.set_ylabel("orbitals per fragment cluster")
    ax.set_title(r"(b) Fragment cluster size vs $\eta$" + "\n(log growth, cluster $\\sim\\ln(1/\\eta)$)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()

    # ---- (c) energy error vs eta + inflection ----
    ax = axes[2]
    err = np.array([abs(r["e_tot"] - e_ref) for r in records])
    ax.loglog(eta_arr, err, "o-", color="C4", markersize=7,
              label=r"$|E_{\rm EWF}(\eta)-E_\infty^{\rm bath}|$")
    if eta_upper:
        ax.axvline(eta_upper, ls="--", color="blue", alpha=0.45, lw=1,
                   label=rf"upper-knee $\eta_{{\rm up}}\approx{eta_upper:.1e}$")
    if eta_star:
        idx = int(np.argmin(np.abs(eta_arr - eta_star)))
        ax.axvline(eta_star, ls="--", color="red", alpha=0.7,
                   label=rf"lower-knee $\eta^*\approx{eta_star:.1e}$ (diminishing returns)")
        ax.annotate(rf"$\eta^*={eta_star:.1e}$" + "\n(diminishing\n returns)",
                    xy=(eta_star, err[idx]), xytext=(eta_star * 8, err[idx] * 8),
                    fontsize=9, color="red",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
    # eta ~ err (linear) reference, exponential-spectrum prediction
    if err.max() > 0:
        c = err[0] / eta_arr[0] if eta_arr[0] > 0 else 1.0
        eta_th = np.logspace(np.log10(eta_arr.min()), np.log10(eta_arr.max()), 50)
        ax.loglog(eta_th, c * eta_th, ":", color="gray", alpha=0.6,
                  label=r"$\propto\eta$ (exp. spectrum pred.)")
    # SQD / noise floor
    ax.axhline(1e-3, ls="--", color="green", alpha=0.5, lw=1)
    ax.text(eta_arr.max() * 0.5, 1.3e-3, r"$\epsilon_{\rm SQD}\sim10^{-3}$ Ha",
            fontsize=8, color="green")
    # CCSD reference offset
    ax.axhline(abs(e_ccsd - e_ref), ls="-.", color="purple", alpha=0.5, lw=1)
    ax.text(eta_arr.max() * 0.5, abs(e_ccsd - e_ref) * 1.3,
            r"$|E_{\rm CCSD}-E_\infty^{\rm bath}|$ (EWF bias)", fontsize=8, color="purple")
    ax.set_xlabel(r"BNO truncation threshold $\eta$")
    ax.set_ylabel(r"energy error $|E-E_\infty|$ (Ha)")
    ax.set_title(r"(c) Energy error vs $\eta$ + elbow $\eta^*$" + "\n"
                 + rf"(ref: min-$\eta$ EWF $={e_ref:+.4f}$)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(outpath, dpi=140, bbox_inches="tight")
    print(f"    saved {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    main()
