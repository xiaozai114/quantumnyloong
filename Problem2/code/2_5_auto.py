import numpy as np
import tensorcircuit as tc
from pyscf import gto, scf, fci, cc, ao2mo

# 设置 TensorCircuit 计算后端
tc.set_backend("tensorflow")


# ==============================================================================
# 1. 基础门定义
# ==============================================================================

def apply_rzz(c, i, j, phi):
    """
    实现 RZZ(phi) = exp(-i * phi * Z_i * Z_j / 2)
    对应 RZZ_{ij}(phi) = CNOT_{i->j} RZ_j(phi) CNOT_{i->j}
    """
    c.cnot(i, j)
    c.rz(j, theta=phi)  # TC 中的 rz(theta) 恰为 exp(-i * theta * Z / 2)
    c.cnot(i, j)


def apply_givens(c, i, j, theta):
    """
    实现数守恒实数 Givens rotation G_{ij}(theta)
    对应: CNOT_{j->i} -> CRY_{i->j}(2*theta) -> CNOT_{j->i}
    将 {|01>, |10>} 空间旋转 theta 角，严格保持粒子数守恒。
    """
    c.cnot(j, i)
    c.cry(i, j, theta=2.0 * theta)
    c.cnot(j, i)


def apply_jastrow_pair(c, i, j, J_ij):
    """
    实现单个二体 Jastrow 项 exp(i * J_ij * n_i * n_j)
    对应文档式 (9): RZ_i(J_ij/2) * RZ_j(J_ij/2) * RZZ_{ij}(-J_ij/2)
    （忽略整体相位 exp(i * J_ij / 4)）
    """
    c.rz(i, theta=J_ij / 2.0)
    c.rz(j, theta=J_ij / 2.0)
    apply_rzz(c, i, j, phi=-J_ij / 2.0)


# ==============================================================================
# 2. PySCF 分子积分提取与 CCSD 振幅获取 (LiH 分子)
# ==============================================================================
print("=" * 70)
print("1. 运行 PySCF 计算 LiH (STO-3G, R_LiH = 1.55 Å)...")
print("=" * 70)

mol = gto.M(
    atom='Li 0 0 0; H 0 0 1.55',
    basis='sto-3g',
    charge=0,
    spin=0,
    verbose=0
)

# RHF 计算
mf = scf.RHF(mol).run()
e_hf = mf.e_tot

# FCI 计算 (精准参考基准)
fci_solver = fci.FCI(mf)
e_fci = fci_solver.kernel()[0]

# CCSD 计算 (用于获取真实的 t1, t2 振幅，乘以 ccsd_scale λ 后驱动 LUCJ 线路)
mycc = cc.CCSD(mf).run()
t1 = mycc.t1  # 形状: (nocc_spatial, nvir_spatial)
t2 = mycc.t2  # 形状: (nocc_spatial, nocc_spatial, nvir_spatial, nvir_spatial)

# 提取分子积分
norb_spatial = mf.mo_coeff.shape[1]  # LiH/STO-3G: 6 空间轨道
nspin = 2 * norb_spatial             # 12 自旋轨道 / 量子比特
nelec = mol.nelectron                # 4 个电子 (2 alpha, 2 beta)
nocc_spatial = nelec // 2            # 2 个占据空间轨道
nvir_spatial = norb_spatial - nocc_spatial  # 4 个虚空间轨道

h1e_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri_mo = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape((norb_spatial,) * 4)
ecore = mol.energy_nuc()

# 转换为自旋轨道 (Spin-Orbital) 积分，采用 Block 映射: 0..norb_spatial-1 为 alpha, norb_spatial..2*norb_spatial-1 为 beta
h_sp = np.zeros((nspin, nspin))
g_sp = np.zeros((nspin, nspin, nspin, nspin))  # (pq|rs)

for p in range(nspin):
    for q in range(nspin):
        if p // norb_spatial == q // norb_spatial:  # 自旋相同
            h_sp[p, q] = h1e_mo[p % norb_spatial, q % norb_spatial]
            for r in range(nspin):
                for s in range(nspin):
                    if r // norb_spatial == s // norb_spatial:
                        g_sp[p, q, r, s] = eri_mo[p % norb_spatial, q % norb_spatial, r % norb_spatial, s % norb_spatial]

print(f"Hartree-Fock 能量 (E_HF)  : {e_hf:.8f} Ha")
print(f"Full CI 参考能量   (E_FCI) : {e_fci:.8f} Ha")
print(f"轨道数: {norb_spatial} 空间轨道, {nspin} 自旋轨道")


# ==============================================================================
# 3. 构建 LUCJ 量子线路
# ==============================================================================
def build_lucj_circuit(lambda_scale):
    """
    根据时间顺序构建电路：
    |HF>  -->  U^dagger(λ)  -->  exp(i * J_local(λ))  -->  U(λ)

    量子比特映射 (block 约定):
      q[0..norb_spatial-1]           → α 自旋轨道
      q[norb_spatial..2*norb_spatial-1] → β 自旋轨道
    LiH: 6 空间轨道 → 12 qubit, 占据 α 轨道 0,1 / β 轨道 0,1
    """
    c = tc.Circuit(nspin)

    # --- Step 1: 制备参考态 |HF> ---
    # α 电子占据空间轨道 0,1 → qubit 0, 1
    # β 电子占据空间轨道 0,1 → qubit norb_spatial, norb_spatial+1
    n_alpha = nelec // 2
    nso = norb_spatial
    hf_occ_alpha = list(range(n_alpha))                  # [0, 1]
    hf_occ_beta  = [nso + i for i in range(n_alpha)]     # [6, 7]
    hf_occ = hf_occ_alpha + hf_occ_beta
    for q in hf_occ:
        c.x(q)

    if lambda_scale == 0.0:
        return c  # λ = 0 时退化为纯 HF 态线路

    # 参数准备 (将 CCSD 振幅按 λ 缩放)
    # t1[occ, vir]: 占据空间轨道 occ → 虚空间轨道 vir
    # 使用 HOMO (orbital n_alpha-1) → LUMO (virtual 0) 的振幅，这是最主要的激发
    homo_idx = n_alpha - 1                     # = 1, σ 成键轨道
    lumo_idx = 0                                # 最低虚轨道 (σ*)
    occ_alpha = homo_idx                        # α HOMO → qubit 1
    vir_alpha = nocc_spatial + lumo_idx         # α LUMO → qubit 2
    occ_beta  = nso + homo_idx                  # β HOMO → qubit 7
    vir_beta  = nso + nocc_spatial + lumo_idx   # β LUMO → qubit 8

    theta_alpha = lambda_scale * t1[homo_idx, lumo_idx]
    theta_beta  = lambda_scale * t1[homo_idx, lumo_idx]
    # 局域 Jastrow 耦合参量 J 源于 t2 拟合
    # 使用 HOMO→LUMO 双激发振幅 t2[1,1,0,0] 作为基准耦合强度
    J0 = lambda_scale * abs(t2[homo_idx, homo_idx, lumo_idx, lumo_idx])
    J_params = {
        # α 链 (qubits 0..5): 占据轨道 0,1; 虚轨道 2,3,4,5
        (0, 1): J0,           # occupied-occupied, 最强
        (1, 2): J0 * 0.8,     # occupied-virtual 边界
        (2, 3): J0 * 0.4,     # virtual-virtual
        (3, 4): J0 * 0.2,
        (4, 5): J0 * 0.1,
        # β 链 (qubits 6..11)
        (6, 7): J0,
        (7, 8): J0 * 0.8,
        (8, 9): J0 * 0.4,
        (9, 10): J0 * 0.2,
        (10, 11): J0 * 0.1,
        # α-β 界面
        (5, 6): J0 * 0.6,
    }

    # --- Step 2: 作用 U^dagger(λ) ---
    apply_givens(c, occ_beta, vir_beta, -theta_beta)
    apply_givens(c, occ_alpha, vir_alpha, -theta_alpha)

    # --- Step 3: 作用局域 Jastrow 层 exp(i * J_local) ---
    for (i, j), J_val in J_params.items():
        apply_jastrow_pair(c, i, j, J_val)

    # --- Step 4: 作用 U(λ) ---
    apply_givens(c, occ_alpha, vir_alpha, theta_alpha)
    apply_givens(c, occ_beta, vir_beta, theta_beta)

    return c


# ==============================================================================
# 4. Slater-Condon 规则与经典 SQD 后处理
# ==============================================================================
def build_ci_matrix(basis_bitstrings):
    """根据 Slater-Condon 规则求解子空间 CI 矩阵 H_kl = <D_k| H |D_l>"""
    M = len(basis_bitstrings)
    H = np.zeros((M, M))
    
    def anti_g(p, q, r, s):
        return g_sp[p, r, q, s] - g_sp[p, s, q, r]

    occs = [np.where(np.array(bs) == 1)[0] for bs in basis_bitstrings]

    for i in range(M):
        for j in range(i, M):
            occ_i, occ_j = occs[i], occs[j]
            diff_i = np.setdiff1d(occ_i, occ_j)
            diff_j = np.setdiff1d(occ_j, occ_i)
            
            if len(diff_i) == 0:  # 对角项
                val = ecore + np.sum(h_sp[occ_i, occ_i])
                for p in occ_i:
                    for q in occ_i:
                        val += 0.5 * anti_g(p, q, p, q)
                H[i, j] = val
            elif len(diff_i) == 1:  # 单激发
                p, q = diff_i[0], diff_j[0]
                common = np.intersect1d(occ_i, occ_j)
                phase = (-1) ** (np.sum(occ_i < p) + np.sum(occ_j < q))
                val = h_sp[p, q]
                for r in common:
                    val += anti_g(p, r, q, r)
                H[i, j] = phase * val
            elif len(diff_i) == 2:  # 双激发
                p, q = diff_i[0], diff_i[1]
                r, s = diff_j[0], diff_j[1]
                phase = (-1) ** (np.sum(occ_i < p) + np.sum(occ_i < q) + 
                                 np.sum(occ_j < r) + np.sum(occ_j < s))
                H[i, j] = phase * anti_g(p, q, r, s)
            else:
                H[i, j] = 0.0
            H[j, i] = H[i, j]
    return H


def run_sqd_pipeline(lambda_scale, n_samples=1000):
    """运行完整 SQD 流程: 电路构建 -> 测量采样 -> 粒子数约束恢复 -> CI 对角化"""
    circuit = build_lucj_circuit(lambda_scale)

    # 测量采样
    sample_counts = circuit.sample(batch=n_samples, format="count_dict_bin")

    # 粒子数守恒过滤 (Configuration Recovery)
    # TC 的 count_dict_bin 中 qubit 0 是最高位 → 前 norb_spatial 位是 α, 后 norb_spatial 位是 β
    n_alpha_target = nelec // 2
    n_beta_target  = nelec // 2
    nso = norb_spatial

    valid_bitstrings = []
    for bs_str in sample_counts.keys():
        bs_tuple = tuple(int(b) for b in bs_str)
        if sum(bs_tuple[:nso]) == n_alpha_target and sum(bs_tuple[nso:]) == n_beta_target:
            valid_bitstrings.append(bs_tuple)

    # 确保 HF 参考态包含在子空间内
    # α 占据 0..n_alpha-1, β 占据 nso..nso+n_alpha-1
    hf_list = [0] * nspin
    for i in range(n_alpha_target):
        hf_list[i] = 1
        hf_list[nso + i] = 1
    hf_tuple = tuple(hf_list)
    if hf_tuple not in valid_bitstrings:
        valid_bitstrings.append(hf_tuple)

    # 构建 CI 矩阵并求解最低本征值
    H_ci = build_ci_matrix(valid_bitstrings)
    e_sqd = np.linalg.eigvalsh(H_ci)[0]

    return e_sqd, len(valid_bitstrings), len(sample_counts)


# ==============================================================================
# 5. 执行对比测试 (验证电路逻辑 + 显示 SQD 结果)
# ==============================================================================
print("\n" + "=" * 70)
print("2. 运行 TensorCircuit LUCJ-SQD 模拟对比...")
print("=" * 70)

# λ=0 基准 (纯 HF)
e_sqd_hf, M_hf, raw_hf = run_sqd_pipeline(lambda_scale=0.0, n_samples=1000)

# λ=1.0 (完整 CCSD 振幅), 较大样本
e_sqd_lucj, M_lucj, raw_lucj = run_sqd_pipeline(lambda_scale=1.0, n_samples=50000)

print(f"\n{'方法':<20} {'S':<8} {'比特串种类':<10} {'M':<6} {'E (Ha)':<16} {'误差 (mHa)':<12}")
print("-" * 70)
print(f"{'HF-SQD  (λ=0)':<20} {1000:<8} {raw_hf:<10} {M_hf:<6} {e_sqd_hf:<16.8f} {abs(e_sqd_hf-e_fci)*1000:<12.4f}")
print(f"{'LUCJ-SQD (λ=1)':<20} {50000:<8} {raw_lucj:<10} {M_lucj:<6} {e_sqd_lucj:<16.8f} {abs(e_sqd_lucj-e_fci)*1000:<12.4f}")
print(f"{'Full CI 参考':<20} {'-':<8} {'-':<10} {'-':<6} {e_fci:<16.8f} {'0.0000':<12}")

# 理论分析
theta = abs(t1[nocc_spatial-1, 0])
J0_val = abs(t2[nocc_spatial-1, nocc_spatial-1, 0, 0])
p_non_hf = theta**4  # O(θ⁴) 量级
print(f"\n理论分析:")
print(f"  Givens 角 θ = t1[HOMO→LUMO] = {theta:.6f} rad")
print(f"  Jastrow 耦合 J0 = |t2[HOMO→LUMO]| = {J0_val:.6f}")
print(f"  单次测量非HF概率 ~ O(θ⁴) ≈ {p_non_hf:.2e}")
print(f"  50k 采样期望非HF计数 ≈ {p_non_hf * 50000:.1f}")
print(f"\n  结论: LiH 平衡键长下关联极弱，单对 Givens 的 LUCJ 线路")
print(f"        无法在当前样本量下产生可观测的非 HF 态。建议:")
print(f"        - 使用更大键长 (如 R=3.0 Å, 关联显著增强)")
print(f"        - 增加 Givens 旋转对数 (更完整的 UCC 线路)")
print(f"        - 增大采样数到 10⁶ 以上")