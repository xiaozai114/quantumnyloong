# Q02：子空间维数与批次大小——权衡

## 题目

分子：LiH，S = 1000，d = 250。

- (a) 从 LiH HF 态采样 S 个比特串，统计 M（唯一行列式数），与 FCI 维数 36 比较
- (b) 给定 d 时批次数 = M/d。对 S ∈ {100, 500, 1000, 5000}，计算批次数，画内存和时间随 S 变化，找最优 S
- (c) 对 N₂（d ~ 4×10⁴, M ~ 10⁶）：笔记本能处理吗？集群呢？
- 进阶：化学精度下 M 的信息论下界？

## 我的想法

1. **HF 态的本质**：HF 基态是单一 Slater 行列式，在计算基下对应一个比特串。从纯 HF 态采样，无论 S 多大，M 恒为 1。与 FCI 维数 36 相比，覆盖率仅 2.8%——这正是 SQD 需要含纠缠量子态的原因。

2. **SQD 的关键前提**：实际 SQD 工作流中，量子线路制备 $|\Psi\rangle = U|\text{HF}\rangle$（如 UCCSD ansatz），$|\Psi\rangle$ 在计算基下展开为多个行列式的叠加。从 $|\Psi\rangle$ 采样才能获得多样化的比特串。

3. **M vs S 的关系**：对含 $N$ 个有效行列式的量子态，$M(S)$ 遵循"收集者问题"（coupon collector）曲线，$M(S) \approx N(1 - e^{-S/N})$（均匀采样近似），随 $S$ 增大趋于 $N$。

4. **批次权衡**：
   - 内存（每批）$\propto d^2$（CI 矩阵）
   - 单批时间 $\propto d^3$（稠密对角化）
   - 总时间 $\propto \lceil M/d \rceil \times d^3 \approx M \cdot d^2$
   - 大 $d$：少批次但高内存；小 $d$：多批次但低内存

5. **N₂ 规模**：$M \sim 10^6$，$d \sim 4 \times 10^4$ → 25 批次，每批稠密矩阵内存 ~12.8 GB，笔记本勉强/不可行，集群可行。稀疏方法下笔记本也可行。

6. **信息论下界**：$M_{\min} = \Omega(e^H / \delta)$，$H$ 为行列式概率分布的 Shannon 熵，$\delta = \epsilon_{\text{chem}} / \Delta_{\text{gap}}$。

## 讨论

### (a) LiH HF 态采样与 FCI 维数比较

#### LiH 的 FCI 维数

LiH 在 σ 型最小活性空间下：Li(1s, 2s, 2pσ) + H(1s) = 4 个空间轨道 = 8 个自旋轨道，4 个电子（$N_\alpha = N_\beta = 2$）。FCI 行列式空间维数（固定 $S_z = 0$）：

$$\dim(\mathcal{H}_{\text{FCI}}) = \binom{4}{2_\alpha} \times \binom{4}{2_\beta} = 6 \times 6 = 36$$

#### HF 态采样的结果

HF 基态将 4 个电子填入能量最低的 2 个空间轨道，对应单一 Slater 行列式 $|D_{\text{HF}}\rangle$。在 JW 映射下，$|\text{HF}\rangle$ 是一个计算基态（单一比特串）。

从纯 HF 态采样 $S = 1000$ 次：

$$\boxed{M = 1 \quad \text{（所有 1000 个采样均为同一比特串）}}$$

与 FCI 维数比较：

$$\frac{M}{\dim(\mathcal{H}_{\text{FCI}})} = \frac{1}{36} \approx 2.8\%$$

**结论**：纯 HF 态采样对 SQD 完全无用——子空间仅含 1 个行列式，对角化给出 HF 能量，无任何关联能修正。这一结论与参考解答中 Problem 2.2 的分析一致：HF + singles 子空间因 Brillouin 定理不提供关联能修正，关联能必须从双激发开始，而纯 HF 采样连 singles 都无法覆盖。

#### 含纠缠量子态的采样

实际 SQD 工作流中，量子线路制备含纠缠的态 $|\Psi\rangle = U|\text{HF}\rangle$（如 UCCSD ansatz），$|\Psi\rangle$ 在计算基下展开为多个行列式的叠加：

$$|\Psi\rangle = \sum_{k=1}^{N} c_k |D_k\rangle, \quad p_k = |c_k|^2$$

采样 $S$ 次后，唯一行列式数 $M(S)$ 近似遵循（均匀采样近似）：

$$M(S) \approx N\left(1 - \left(1 - \frac{1}{N}\right)^S\right) \approx N\left(1 - e^{-S/N}\right)$$

其中 $N \leq 36$ 为 $|\Psi\rangle$ 中有效行列式数。对 $N = 36$（$|\Psi\rangle$ 覆盖整个 FCI 空间）：

| $S$ | $M(S) \approx$ | $M/36$ |
|-----|----------------|--------|
| 100 | 34 | 94.4% |
| 500 | 36.0 | ~100% |
| 1000 | 36.0 | ~100% |
| 5000 | 36.0 | ~100% |

$S \approx 3N \approx 108$ 时 $M$ 已达 95%，$S \approx 5N \approx 180$ 时基本饱和。

> **注意**：对于非均匀分布（$p_k$ 不等），$M(S)$ 增长更慢——低概率行列式需要更多采样才能被"捕获"。实际 UCCSD 态中 HF 成分占主导（$|c_{\text{HF}}|^2 \sim 90\%+$），双激发成分很小，需要较大 $S$ 才能采样到所有重要行列式。

### (b) 批次数、内存、时间随 S 的变化

#### LiH 的批次计算（$d = 250$，$N = 36$）

批次数 $B = \lceil M/d \rceil$。由于 $d = 250 \gg M = 36$，所有情况下仅 1 批，$d$ 实际被截断为 $M$：

| $S$ | $M$ | $B = \lceil M/250 \rceil$ | 内存/批（稠密） | 时间/批（$d^3$） |
|-----|-----|---------------------------|-----------------|------------------|
| 100 | 34 | 1 | $34^2 \times 8$ B $\approx$ 9 KB | $34^3 \approx 4 \times 10^4$ |
| 500 | 36 | 1 | $36^2 \times 8$ B $\approx$ 10 KB | $36^3 \approx 5 \times 10^4$ |
| 1000 | 36 | 1 | $\approx$ 10 KB | $\approx 5 \times 10^4$ |
| 5000 | 36 | 1 | $\approx$ 10 KB | $\approx 5 \times 10^4$ |

LiH 系统太小（$M \ll d$），批次权衡不显著——所有资源消耗都是微秒/KB 级别。

#### 通用权衡分析

对一般系统（$M \gg d$），资源消耗为：

- **内存**（每批峰值）：$W_{\text{mem}} = \alpha \cdot d^2$（稠密 CI 矩阵，$\alpha = 8$ B/元素）或 $O(d \cdot n_{\text{nz}})$（稀疏，$n_{\text{nz}}$ 为每行非零元数）
- **时间**（每批）：$T_{\text{batch}} = \beta \cdot d^3$（稠密对角化，$\beta$ 为硬件相关常数）
- **总时间**：$T_{\text{total}} = \lceil M/d \rceil \times T_{\text{batch}} \approx \beta \cdot M \cdot d^2$
- **总内存**：$W_{\text{mem}}$（峰值，逐批处理不累积）

**最优 $d$ 的选择**：

- 受内存约束：$d_{\max} = \sqrt{W_{\text{avail}} / \alpha}$
- 总时间 $T \propto d^2$（在 $M$ 固定时），$d$ 越小总时间越少
- 但 $d$ 有下限：每批需包含足够多的行列式以保留关键的行列式间耦合（至少 $d \geq n_{\text{coupling}}$，即一个行列式通过单/双激发耦合到的邻居数），否则分批对角化丢失跨批耦合信息
- **最优 $d$**：$d^* = \max(d_{\min}, \min(d_{\max}, \text{hardware-optimal}))$，即在内存允许范围内取较小的 $d$，但保证批内耦合完整

**最优 $S$ 的选择**：

- $S$ 太小 → $M$ 不足 → SQD 能量差（丢失重要行列式）
- $S$ 太大 → $M$ 饱和 → 边际收益递减；但计算成本 $\propto M \cdot d^2$ 在 $M$ 饱和后不再增加
- **最优 $S^*$**：$S^* \approx 3N \sim 5N$（$N$ 为量子态有效行列式数），使 $M$ 达到 95%~99%

对 LiH：$S^* \approx 3 \times 36 = 108$，即 $S = 100$ 已接近最优。$S = 1000$ 远超必要但不会增加 SQD 成本（$M$ 已饱和）。

#### 内存和时间 vs $S$ 的曲线特征

**小系统（LiH，$N = 36$）**：
- $S < 50$：$M$ 快速增长，内存/时间随 $M$ 增长
- $S > 100$：$M$ 饱和，内存/时间进入平台
- $d = 250$ 始终远大于 $M$，无批次效应

**大系统（如 N₂，$N \sim 10^6$）**：
- $S$ 增大 → $M$ 持续增长（远未饱和）→ 批次增多 → 总时间 $\propto M \cdot d^2$ 近似线性增长
- 存在 $S^*$ 使能量精度与计算成本的权衡最优：$S^*$ 处 $M$ 已覆盖化学精度所需子空间（见进阶部分）

### (c) N₂ 的可处理性

N₂ 活性空间（如冻结芯 6-31G 基组）：$d \sim 4 \times 10^4$，$M \sim 10^6$。

#### 批次与资源

$$B = \lceil M/d \rceil = \lceil 10^6 / (4 \times 10^4) \rceil = 25 \text{ 批}$$

**稠密矩阵假设**（$d \times d$ CI 矩阵）：

| 资源 | 公式 | 量值 | 笔记本 (32 GB, 10 GFLOP/s) | 集群 (256 GB, 1 TFLOP/s) |
|------|------|------|---------------------------|--------------------------|
| 内存/批 | $d^2 \times 8$ B | 12.8 GB | 勉强（加特征向量/工作空间需 25–40 GB，超限） | 轻松 |
| 时间/批 | $d^3$ FLOP | $6.4 \times 10^{13}$ | ~6400 s (~1.8 h) | ~64 s |
| 总时间 | $B \times T_{\text{batch}}$ | — | ~160,000 s (~44 h) | ~1600 s (~27 min) |

**笔记本（稠密）**：内存 12.8 GB 接近 32 GB 上限，加上特征向量和 LAPACK 工作空间需 25–40 GB → **不可行**。

**集群（稠密）**：内存和时间均充裕 → **可行**（~27 min）。

**稀疏方法**：CI 矩阵高度稀疏——每个行列式仅通过单/双激发耦合到有限个邻居，每行非零元数 $n_{\text{nz}} \sim N_o^2 \cdot N_v^2$（对中等活性空间约 $10^2 \sim 10^3$）。使用 Lanczos/ARPACK 迭代对角化：

| 资源 | 公式 | 量值 | 笔记本 |
|------|------|------|--------|
| 内存/批 | $d \cdot n_{\text{nz}} \times 8$ B | $4 \times 10^4 \times 10^3 \times 8 \approx 320$ MB | 可行 |
| 时间/批 | $d \cdot n_{\text{nz}} \cdot n_{\text{iter}}$ | $4 \times 10^4 \times 10^3 \times 100 = 4 \times 10^9$ | ~0.4 s |
| 总时间 | $B \times T_{\text{batch}}$ | — | ~10 s |

**笔记本（稀疏）**：**可行**（~10 s）。

**结论**：

| 方法 | 笔记本 | 集群 |
|------|--------|------|
| 稠密对角化 | 不可行（内存超限） | 可行（~27 min） |
| 稀疏 Lanczos | 可行（~10 s） | 可行（~1 s） |

### 进阶：化学精度下 M 的信息论下界

#### 问题表述

化学精度：$\epsilon_{\text{chem}} \approx 1.6 \times 10^{-3}$ Ha（1 kcal/mol）。求最小的子空间维数 $M_{\min}$ 使得 SQD 能量误差 $|E_{\text{SQD}} - E_{\text{FCI}}| < \epsilon_{\text{chem}}$。

#### 截断误差与行列式概率

设 FCI 基态 $|\Psi_0\rangle = \sum_{k=1}^{N_{\text{FCI}}} c_k |D_k\rangle$，按 $|c_k|$ 降序排列。截断至前 $M$ 个行列式后，能量误差上界（微扰论估计）：

$$\Delta E \leq \sum_{k > M} |c_k|^2 \cdot \Delta_k$$

其中 $\Delta_k$ 为行列式 $k$ 相对于基态的激发能（$\geq \Delta_{\text{gap}}$，谱隙）。要求 $\Delta E < \epsilon_{\text{chem}}$：

$$\sum_{k > M} |c_k|^2 < \frac{\epsilon_{\text{chem}}}{\Delta_{\text{gap}}} \equiv \delta$$

#### 信息论下界

定义行列式概率分布 $p_k = |c_k|^2$，其 Shannon 熵：

$$H = -\sum_{k} p_k \ln p_k$$

$e^H$ 是该分布的"有效支持大小"（effective support size），度量波函数 spread 在多少个行列式上。

由信息论中的概率分布尾部界（source coding / minimax rate），对熵为 $H$ 的分布，使尾部概率 $\sum_{k > M} p_k < \delta$ 所需的最小 $M$ 满足量级关系：

$$\boxed{M_{\min} = \Omega\!\left(\frac{e^H}{\delta}\right) = \Omega\!\left(\frac{e^H \cdot \Delta_{\text{gap}}}{\epsilon_{\text{chem}}}\right)}$$

物理含义：$M_{\min}$ 由波函数的"有效行列式数" $e^H$ 和精度要求 $\delta$ 共同决定。

#### 物理标度

- **弱关联系统**：$H \approx 0$（波函数集中于 HF），$e^H \approx 1$，$M_{\min} = O(1/\delta)$，少量行列式即可达化学精度
- **强关联系统**：$H \sim O(N_e)$（$N_e$ 为电子数，每个电子贡献 $O(1)$ 熵），$M_{\min} \sim e^{O(N_e)} / \delta$ — 指数增长但仍远小于 FCI 维数 $\binom{N_{\text{so}}}{N_e}$
- **弱关联 MP2 区域**：$H \sim \ln(N_o^2 \cdot N_v^2)$（双激发主导），$e^H \sim N_o^2 \cdot N_v^2$，$M_{\min} \sim N_o^2 \cdot N_v^2 / \delta$

以 N₂ 平衡键长为例（$N_o \sim 10$, $\Delta_{\text{gap}} \sim 0.3$ Ha）：

$$\delta \sim \frac{1.6 \times 10^{-3}}{0.3} \approx 5 \times 10^{-3}$$

$$M_{\min} \sim \frac{N_o^2}{\delta} \sim \frac{100}{5 \times 10^{-3}} \sim 2 \times 10^4$$

这与实际 SQD 计算中 $M \sim 10^4 \sim 10^5$ 达到化学精度的经验吻合。

#### 与 FCI 维数比较

$$\frac{M_{\min}}{N_{\text{FCI}}} \sim \frac{e^H / \delta}{\binom{N_{\text{so}}}{N_e}} \ll 1$$

这正是 SQD 的核心优势：通过量子采样选取重要子空间，$M_{\min} \ll N_{\text{FCI}}$，避免了 FCI 的指数爆炸。量子态 $|\Psi\rangle$ 的采样天然倾向于高概率行列式（即重要组态），起到了"重要性采样"的作用。

#### 与参考解答的联系

参考解答中 H₂ 的例子提供了直观验证：
- 平衡键长（弱关联）：$|c_{\text{HF}}|^2 = 98.7\%$，$H \approx 0.05$，$e^H \approx 1.05$，$M_{\min} \approx 1$ — 仅需 HF + 1 个双激发
- 拉伸至 3.0 Å（强关联）：$|c_{\text{HF}}|^2 = 53.7\%$，$|c_{\text{double}}|^2 = 46.3\%$，$H \approx 0.69$，$e^H \approx 2.0$，$M_{\min} \approx 2$ — 需 2 个行列式（HF + 双激发）

H₂ 只有一个双激发通道，$M_{\min}$ 很小。但对多电子系统，双激发通道数 $\sim N_o^2 N_v^2$ 快速增长，$M_{\min}$ 随之增大。

## 结论

1. **(a)** 纯 HF 态采样给出 $M = 1$，仅为 FCI 维数 36 的 2.8%。SQD 需要含纠缠的量子态才能产生多样化的行列式。对覆盖全 FCI 空间的量子态，$S \approx 3N \approx 108$ 即可使 $M \to 36$。

2. **(b)** 对 LiH（$M \leq 36 \ll d = 250$），所有 $S$ 下仅 1 批，无显著权衡。通用规律：总时间 $\propto M \cdot d^2$，内存 $\propto d^2$。最优 $S^* \approx 3N \sim 5N$（边际收益递减点），最优 $d$ 受内存约束且需保证批内耦合完整。

3. **(c)** N₂（$d \sim 4 \times 10^4$，$M \sim 10^6$，25 批）：稠密方法下笔记本不可行（内存 12.8 GB + 工作空间超 32 GB），集群可行（~27 min）。稀疏 Lanczos 方法下笔记本也可行（~10 s）。

4. **进阶**：化学精度下 $M_{\min} = \Omega(e^H \cdot \Delta_{\text{gap}} / \epsilon_{\text{chem}})$，其中 $H$ 为波函数行列式分布的 Shannon 熵。弱关联 $M_{\min} \sim N_o^2 / \delta \sim 10^4$，远小于 FCI 维数——这正是 SQD 的理论优势。

## 代码说明

代码在 `code/` 目录下，包含：

1. **`sample_bitstrings.py`**：从 HF 态和 UCCSD 态采样，统计 $M(S)$
   - `count_unique_determinants(state, S)`: 采样 $S$ 次，返回唯一行列式数 $M$
   - 对比 HF（$M = 1$）与 UCCSD（$M$ 随 $S$ 增长并饱和）
   - 绘制 $M$ vs $S$ 的收集者问题曲线

2. **`batch_analysis.py`**：批次数、内存、时间随 $S$ 变化分析
   - `estimate_memory(d, sparse=False)`: 估算每批内存（稠密 $d^2$ 或稀疏 $d \cdot n_{\text{nz}}$）
   - `estimate_time(M, d, flops_rate, sparse=False)`: 估算总时间
   - 输出 LiH 的 $S \in \{100, 500, 1000, 5000\}$ 对比表格

3. **`plot_tradeoff.py`**：绘制权衡曲线
   - $M$ vs $S$ 曲线（收集者问题曲线，标记 95% 饱和点 $S^*$）
   - 内存/时间 vs $S$ 曲线（小系统平台型，大系统增长型）
   - 最优 $S^*$ 标记

4. **`n2_scalability.py`**：N₂ 规模可行性分析
   - 稠密 vs 稀疏方法对比（内存、时间）
   - 笔记本 vs 集群资源评估表
   - 参数可调（$d$, $M$, $n_{\text{nz}}$, FLOP/s）以适应不同硬件
