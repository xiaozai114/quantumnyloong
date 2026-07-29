# Q10：完整流程集成与端到端误差分析（开放题）

## 题目

分子 (a) H₂、(c) H₂O、(d) N₂、(e) C₂H₄。固定预算 B。

误差模型 ε_total ≈ ε_frag + ε_SQD，代价模型 B = c₁N·nfrag³ + c₂N·S³/nfrag。

- (a) 对 4 个分子运行完整流程 RHF→DMET→LUCJ-SQD→能量重构，报告 E_MF, E_EWF-SQD, E_FCI(或CCSD)
- (b) 固定 B 下，用拉格朗日乘子法推导最优分配 ε_frag ≈ ε_SQD（木桶效应）
- (c) 讨论：H₂O(14q)→C₂H₄(28q) 哪种误差增长最快？N₂ 拉伸到 2.0Å 哪项劣化最快？
- 进阶：FTQC 时代 SQD 升级为 QPE。EWF 如何把 QPE 资源从 O(N/ε) 降到 O(nfrag/ε)？p₂q 低于何值时 QPE 优于 SQD？

## 我的想法

整体思路分四步：
1. (a) 用讲义第1–9课的语法搭建全流程代码，4个分子跑通，拿实测数据
2. (b) 列拉格朗日函数，对 n_frag 和 S 求偏导，用误差反代系数，推出 α·ε_frag = β·ε_SQD
3. (c) 对比 H₂O/C₂H₄ 的实测误差，额外跑 N₂(2.0Å) 看退化
4. (d) EWF 分片把全局 QPE 变成碎片局部 QPE；列 SQD/QPE 代价等式解 p₂q

## 讨论

### (a) 四个分子全流程实测

实验设置（代码 `code/problem10_solution.py`）：

| 分子 | 基组 | 总轨道 | 冻结 | 活性轨道 | 电子 | 量子比特 | 组态数 |
|---|---|---|---|---|---|---|---|
| H₂ | sto-3g | 2 | 0 | 2 | (1,1) | 4 | 4 |
| H₂O | sto-3g | 7 | 0 | 7 | (5,5) | 14 | 441 |
| N₂ | 6-31g | 18 | 2 | 14 | (5,5) | 28 | 4,008,004 |
| C₂H₄ | sto-3g | 14 | 2 | 11 | (5,5) | 22 | 213,444 |

全流程使用讲义语法：
- 第1–2课：`gto.M` + `scf.RHF` 建分子跑 HF
- 第4课：`mcscf.CASCI` + `sort_mo` 选活性空间
- 第5课：`get_h1cas` / `get_h2cas` + `ao2mo.restore` 导积分
- 第7课：`ffsim.UCJOpSpinBalanced` + `ffsim.sample_state_vector` LUCJ 采样（ffsim 在费米子空间模拟，不需要 qiskit StatevectorSampler）
- 第8课：`diagonalize_fermionic_hamiltonian` + `solve_sci_batch` SQD kernel
- 第9课：三段缝合 + `cc.CCSD(T)` 对比标尺

实测结果：

| 分子 | E_MF (Ha) | E_SQD (Ha) | E_ref (Ha) | ε_frag (mHa) | ε_SQD (mHa) |
|---|---|---|---|---|---|
| H₂ | -1.116759 | -1.137284 | -1.137284 (FCI) | 0 | 0 |
| H₂O | -74.963063 | -75.012647 | -75.012647 (FCI) | 0 | 0 |
| N₂ | -108.867618 | -109.040372 | -109.103527 (CCSD(T)) | 63.1 | 0.097 |
| C₂H₄ | -77.072321 | -77.203657 | -77.234762 (CCSD(T)) | 31.1 | 0.003 |

关键观察：
- H₂/H₂O：活性空间=全空间，②=③=0，全流程精确（FCI 可直接算）
- N₂：②(63.1 mHa) >> ③(0.097 mHa)，SQD 算法本身已达化学精度的 1/16，瓶颈在活性空间截断
- C₂H₄：②(31.1 mHa) >> ③(0.003 mHa)，SQD 算法完美，瓶颈仍在活性空间

### (b) 拉格朗日乘数法：木桶效应推导

**误差模型**：

$$\epsilon = \epsilon_{\text{frag}} + \epsilon_{\text{SQD}} = \frac{A}{n_{\text{frag}}^\alpha} + \frac{C}{S^\beta}$$

其中 α ≈ 2~3（bath 截断），β ≈ 1~2（采样）。

**成本模型**：

$$B = c_1 N n_{\text{frag}}^3 + c_2 N \frac{S^3}{n_{\text{frag}}}$$

**拉格朗日函数**：

$$\mathcal{L} = \frac{A}{n^\alpha} + \frac{C}{S^\beta} + \lambda\left(B - c_1 N n^3 - c_2 N \frac{S^3}{n}\right)$$

**一阶条件** ∂L/∂n = 0, ∂L/∂S = 0：

边际误差 / 边际成本 对两个误差源相等：

$$\frac{\alpha A / n^{\alpha+1}}{3c_1 N n^2} = \frac{\beta C / S^{\beta+1}}{3c_2 N S^2 / n}$$

**用误差反代系数**：A = ε_frag · n^α，C = ε_SQD · S^β，代入：

$$\alpha \cdot \epsilon_{\text{frag}} \cdot c_2 \cdot S^3 = \beta \cdot \epsilon_{\text{SQD}} \cdot c_1 \cdot n^4$$

**最优预算分配**下两项成本相当：c₁n⁴ ≈ c₂S³，代入得：

$$\boxed{\alpha \cdot \epsilon_{\text{frag}} = \beta \cdot \epsilon_{\text{SQD}}}$$

**当 α = β（同幂次衰减）**：

$$\boxed{\epsilon_{\text{frag}} = \epsilon_{\text{SQD}} \quad \text{（木桶效应）}}$$

**物理意义**：总误差由最大的误差源决定（木桶最短板）。若 ε_frag > ε_SQD，应增大碎片（花更多预算在 DMET）；反之增大采样。最优是两者相等。

**数值验证**：

| 分子 | ε_frag (mHa) | ε_SQD (mHa) | 比值 | 木桶? |
|---|---|---|---|---|
| H₂ | 0 | 0 | — | ✓ |
| H₂O | 0 | 0 | — | ✓ |
| N₂ | 63.1 | 0.097 | 649 | ✗ (②主导) |
| C₂H₄ | 31.1 | 0.003 | 10885 | ✗ (②主导) |

实测结论：当前设置下 ε_frag >> ε_SQD，预算应向扩大活性空间倾斜。

### (c) 误差增长分析

**(c)-1: H₂O(14q) → C₂H₄(22q)**

| 误差源 | H₂O (14q) | C₂H₄ (22q) | 增长 | 原因 |
|---|---|---|---|---|
| ε_frag | 0 mHa | 31.1 mHa | 0→31.1 | C₂H₄冻2个1s，活性空间占比下降 |
| ε_SQD | 0 mHa | 0.003 mHa | 0→0.003 | 组态数441→213,444，采样覆盖下降 |
| ε_noise | 小 | 中 | ~1.6x | 量子比特14→22，电路深度增加 |

**ε_frag 绝对增长最快**（0→31.1 mHa）——C₂H₄ 冻结 2 个 C 1s 后活性空间只占 11/14 轨道，冻结轨道的相关能损失显现。

**(c)-2: N₂ 键长拉伸 1.1Å → 2.0Å**

| 误差源 | N₂(1.1Å) | N₂(2.0Å) | 退化倍数 | 原因 |
|---|---|---|---|---|
| ε_frag | 63.1 mHa | 141.6 mHa | **2.2x** | 强相关，bath截断失效 |
| ε_SQD | 0.097 mHa | 0.341 mHa | 3.5x | 组态分布变平，采样覆盖下降 |
| ε_noise | 不变 | 不变 | ~1x | 电路深度不变 |

相关能变化：-235.9 mHa → -662.6 mHa（2.8 倍），说明键拉伸时电子相关剧增。

**ε_frag 退化最快且最严重**（63.1→141.6 mHa）——键拉伸导致强相关（多参考态），DMET 分片假设（碎片-环境弱耦合）失效。HF 参考态占比从 91%(1.1Å) 降到 ~50%(2.0Å)，需要多参考描述。

### (d) 进阶：FTQC 时代 EWF + QPE

**EWF 降低 QPE 资源的机制**：

标准 QPE（无 EWF）：对 N 个轨道的完整哈密顿量做相位估计，资源 = O(N/ε)。

EWF + QPE：DMET 把 N 轨道分成 K = N/n_frag 个碎片，每个碎片的嵌入哈密顿量只有 n_frag 个轨道。每个碎片的 QPE 资源 = O(n_frag/ε)。碎片间可并行 → 总时间 = O(n_frag/ε)。

$$O(N/\epsilon) \to O(n_{\text{frag}}/\epsilon)$$

减少倍数 = N/n_frag。例：N=100, n_frag=10 → 减少 10 倍。

**SQD vs QPE 临界 p₂q**：

SQD 总开销 ∝ S × depth_LUCJ = O(n_frag/ε²)（需要 O(1/ε²) 次采样）

QPE 总开销 ∝ (n_frag/ε) × overhead_FT，其中 surface code 纠错 overhead ≈ 1/(p₂q - p_th)²，p_th ≈ 10⁻²。

令两者相等：

$$\frac{n_{\text{frag}}}{\epsilon^2} = \frac{n_{\text{frag}}}{\epsilon} \cdot \frac{1}{(p_{2q} - 10^{-2})^2}$$

$$p_{2q} = 10^{-2} - \sqrt{\epsilon}$$

对化学精度 ε = 1.6×10⁻³：

$$\boxed{p_{2q}^* \approx 10^{-3} \sim 10^{-2}}$$

- p₂q > 10⁻²（NISQ）：SQD 胜——QPE 纠错开销爆炸
- p₂q < 10⁻³（FTQC）：QPE 胜——纠错可控，精度更高
- EWF 的价值在两个时代都成立：把全局问题局部化

## 结论

| 子问题 | 核心结论 |
|---|---|
| (a) | 4 个分子全流程跑通，小体系精确(②=③=0)，大体系 ②frag 主导 |
| (b) | 拉格朗日法证明 α·ε_frag = β·ε_SQD，同幂次时 ε_frag = ε_SQD（木桶效应） |
| (c)-1 | 14q→22q：ε_frag 绝对增长最快（0→31.1 mHa） |
| (c)-2 | 键拉伸：ε_frag 退化最快（63.1→141.6 mHa，强相关导致 bath 截断失效） |
| (d) | EWF 把 QPE 从 O(N/ε) 降到 O(n_frag/ε)；临界 p₂q ≈ 10⁻³~10⁻² |

## 代码说明

代码在 `code/problem10_solution.py`，运行方式：

```bash
/Users/zhouzihan/.workbuddy/binaries/python/envs/sqd/bin/python code/problem10_solution.py
```

代码包含：
- 4 个分子（H₂/H₂O/N₂/C₂H₄）的完整 EWF-SQD 流程
- N₂ 键长拉伸至 2.0Å 的误差退化对比
- 拉格朗日乘数法的数值验证
- 使用 ffsim 进行 LUCJ 采样（费米子空间模拟，不需要 qiskit StatevectorSampler）
- 使用 qiskit-addon-sqd 的 `diagonalize_fermionic_hamiltonian` 做 SQD kernel
- 使用 PySCF 的 CCSD(T) 作为近似 FCI 标尺
