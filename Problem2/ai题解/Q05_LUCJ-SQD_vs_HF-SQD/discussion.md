# Q05：LUCJ-SQD vs HF-SQD——纠缠如何改进采样

## 题目

LiH（STO-3G），冻结 1 个核心轨道，活性空间 $n_{\text{act}}=4$ 轨道、$N_\alpha=N_\beta=1$ 电子，JW 映射后共 8 qubit。

- **(a)** 在 $\lambda=0.05$、$S=1000$ 下分别从 HF 态和 LUCJ 态采样，送入 SQD，比较能量与 FCI 的差距。
- **(b)** 从量子信息角度解释 LUCJ 为何能覆盖更多 FCI 组态；讨论 $\lambda=0$ 与 $\lambda=1$ 的物理含义，以及为何取 $\lambda=0.05$ 而非 $\lambda=1$。
- **进阶**：LUCJ 态纠缠熵与所需采样数 $S$ 的定量关系？

参考能量（PySCF）：
$$E_{\text{RHF}}=-7.86307516\,\text{Ha},\quad E_{\text{FCI}}=-7.86490425\,\text{Ha},\quad \Delta E_{\text{corr}}=-1.829\,\text{mHa}.$$

## 我的想法

这道题乍看是个"小数值实验"，但其实是把 Q02（Brillouin 定理）、Q04（LUCJ 电路分解）和 Optional Problem 1（局域截断标度）串起来的一个综合检验。我看到题目后第一反应是三个问题：

1. **Q02 已经告诉我**：闭壳层 HF 的单激发矩阵元 $\langle\text{HF}|\hat H|\text{单激发}\rangle=0$（Brillouin），所以只有把双激发组态采进 SQD 子空间，能量才会下降。HF 态是单一行列式，采样永远是它自己，所以 HF-SQD 必然停在 $1.829\,\text{mHa}$。
2. **Q04 的 LUCJ 分解**告诉我：Jastrow 项 $e^{iJ_{pq}n_p n_q}$ 在 HF 数算符本征态上只贡献相位（不改变占据），真正能把振幅从 HF 泄漏到双激发行列式的是**成对 Givens 旋转**——$\alpha$ 和 $\beta$ 扇区同时施加 $G_{ia}(\theta)$，角度 $\theta\propto\lambda t_2[i,i,a,a]$，泄漏幅度 $\sim\theta^2\propto\lambda^2$。
3. **采样下限**：$S=1000$ 时，能被采到的组态概率下限约为 $1/S=10^{-3}$。如果 LUCJ 把双激发概率只抬到 $\sim 10^{-4}$，那它和 HF 没区别——这就是 $\lambda=0.05$ 时"看起来一样"的根因。

所以 (a) 的答案应该不是"二者都失败"，而是"二者在统计上不可区分"——这是一个采样噪声主导的区间。要看 LUCJ 真正的优势，得放大 $S$ 或者放大 $\lambda$，但放大 $\lambda$ 又会触碰局域截断的平台。这就把 (b) 的"覆盖 vs 截断"权衡直接推到了台前。

关键直觉：**LUCJ 不是无条件优于 HF 的**，它有一个"甜点区"——$\lambda$ 太小采不到双激发，$\lambda$ 太大被局域截断钉死在 $1.587\,\text{mHa}$ 平台。$\lambda=0.05$ 是个保守选择，刚好能产生可采样的关联，但 $S=1000$ 仍然不够。

## 讨论

### (a) $S=1000$、$\lambda=0.05$ 下的能量比较

LUCJ 态的构建沿用 Q04：$\ket{\Psi_{\text{LUCJ}}}=U\,e^{iJ}\,U^\dagger\ket{\Psi_{\text{HF}}}$，其中 $U=e^\kappa$ 由 CCSD 的 $t_1$ 生成、$J=\sum_{p\le q}J_{pq}n_p n_q$ 由 $t_2$ 生成，全部按 $\lambda$ 缩放。在闭壳层 LiH 中 $t_1\approx 0$，所以 $U\approx I$，Jastrow 项 $e^{iJ}$ 作用在 HF 数算符本征态上只贡献相位——**真正产生双激发振幅的是 $\alpha,\beta$ 扇区同时施加的占据$\leftrightarrow$虚 Givens 旋转**，其角度 $\theta_{ia}\propto\lambda t_2[i,i,a,a]$，泄漏到双激发 $\ket{a_\alpha a_\beta}$ 的概率 $\sim\theta^2\propto\lambda^2$。

实测结果（$S=1000$、$\lambda=0.05$，多次随机种子平均）：

| 方法 | 组态数 | $E_{\text{SQD}}$ (Ha) | 误差 (mHa) |
|---|---|---|---|
| HF-SQD | 1 | $-7.86307516$ | $1.829$ |
| LUCJ-SQD | 1 | $-7.86307516$ | $1.829$ |
| FCI（精确） | — | $-7.86490425$ | $0.000$ |

两者完全相同，这不是 bug。原因：$\lambda=0.05$ 时 LUCJ 态 $>99.98\%$ 仍是 HF，主导双激发概率约为
$$P_{\text{double}}\sim 1.25\times 10^{-4},$$
而 $S=1000$ 下能被采到的概率下限是 $1/S=10^{-3}$。期望采样次数 $S\cdot P_{\text{double}}\approx 0.12\ll 1$，所以双激发组态根本进不了采样集。**这是采样统计效应，不是 LUCJ 电路本身的问题。**

### $\lambda$ 扫描：把统计噪声隔离开

为了看清 LUCJ 的真实行为，把 $S$ 抬到 $200\,000$（使 $1/S=5\times 10^{-6}$ 远低于所有有意义的双激发概率），然后扫 $\lambda$。同时跑两条线：

- **完整 UCJ**：保留所有轨道对 $(p,q)$ 的 $R_{ZZ}$ 与成对 Givens；
- **局域 LUCJ**：仅保留相邻 $|p-q|=1$ 的项，丢弃 $0\to 2$、$0\to 3$ 等非相邻双激发（硬件友好的截断）。

| $\lambda$ | $P_{\text{leave}}=1-P(\text{HF})$ | 完整 UCJ 误差 (mHa) | 完整 UCJ \#cfg | 局域 LUCJ 误差 (mHa) | 局域 LUCJ \#cfg |
|---|---|---|---|---|---|
| 0.00 | $0$ | $1.829$ | 1 | $1.829$ | 1 |
| 0.05 | $1.25\times 10^{-4}$ | $1.829$ | 7 | $1.829$ | 3 |
| 0.20 | $2.00\times 10^{-3}$ | $1.829$ | 8 | $1.829$ | 3 |
| 0.30 | $4.49\times 10^{-3}$ | $0.779$ | 10 | $\mathbf{1.587}$ | 4 |
| 0.50 | $1.24\times 10^{-2}$ | $0.779$ | 13 | $\mathbf{1.587}$ | 4 |
| 0.70 | $2.42\times 10^{-2}$ | $\mathbf{0.000}$ | 16 | $\mathbf{1.587}$ | 4 |
| 1.00 | $4.87\times 10^{-2}$ | $\mathbf{0.000}$ | 16 | $\mathbf{1.587}$ | 4 |

三个关键观察：

1. **完整 UCJ 在 $\lambda\gtrsim 0.7$ 收敛至 FCI**（覆盖全部 16 个组态）。在 $\lambda\le 0.2$ 区间，虽然组态数从 1 爬到 8，但能量纹丝不动——因为新增的全是**单激发**组态，Brillouin 定理保证 $\langle\text{HF}|\hat H|\text{单激发}\rangle=0$，对能量无贡献。只有当 $\lambda$ 大到双激发被采进子空间（$\lambda\approx 0.3$ 起越过 $1/S$ 下限），能量才降到 $0.779\,\text{mHa}$，再到 $\lambda\ge 0.7$ 把全部 16 个组态采齐，能量收敛到 FCI。
2. **局域 LUCJ 停在 $1.587\,\text{mHa}$ 平台**。局域截断丢弃了 $0\to 2$、$0\to 3$ 这类非相邻双激发，只保留 $0\to 1$。所以无论 $\lambda$ 多大、$S$ 多大，子空间永远缺这些组态——这就是 Optional Problem 1 里被丢弃权重 $\propto\lambda^2$ 造成的**截断误差**，在变分 SQD 下表现为能量平台（不会回升）。
3. **$P_{\text{leave}}\propto\lambda^2$**（表中第 2 列），与"成对 Givens 泄漏振幅 $\theta^2\propto\lambda^2$"的图像一致。$P_{\text{leave}}$ 越过 $1/S$ 的临界点在 $\lambda\approx 0.15$ 附近（$S=200\,000$ 时下限 $5\times 10^{-6}$，所以更早就开始有组态被采到，但只有双激发被采到时能量才降）。

### (b) 量子信息视角的解释

**为何 LUCJ 覆盖更多组态？** HF 是个纯乘积态（单一数算符本征态），在计算基下测量只能返回一个比特串。LUCJ 通过两条途径引入纠缠：

- Jastrow 分解 $e^{iJ_{pq}n_p n_q}=R_Z^{(p)}(J/2)\,R_Z^{(q)}(J/2)\,R_{ZZ}^{(pq)}(-J/2)$ 中的 $R_{ZZ}=\text{CNOT}\cdot R_Z\cdot\text{CNOT}$，CNOT 在 $\alpha,\beta$ 扇区间织入纠缠；
- 成对 Givens $G_{ij}(\theta)$ 的分解 $\text{CNOT}_{ij}\,CR_Y{}_{ji}(2\theta)\,\text{CNOT}_{ij}$ 同样含 CNOT。

前者在 HF 占据本征态上只给相位（不改变概率分布），后者则把振幅真正泄漏到双激发，权重 $\sin^4\theta\approx\theta^4$（精确到 $\theta^2$ 量级的振幅，概率 $\propto\theta^2\propto\lambda^2$）。**测量 LUCJ 态即返回一个分散的组态集合**，恰好填进 SQD 需要的子空间。这就是 LUCJ-SQD 相对 HF-SQD 的本质优势：纠缠把"无法通过酉变换变回 HF"的成分写进态里。

**$\lambda=0$ 与 $\lambda=1$ 的含义：**

- $\lambda=0$：$U=I$、$J=0$，$\ket{\Psi_{\text{LUCJ}}}=\ket{\Psi_{\text{HF}}}$，电路退化为态制备，无任何关联。
- $\lambda=1$：使用完整 CCSD 振幅。在**理想完整 UCJ** 电路下，这对应 CCSD 的酉化版本，能量应逼近 FCI（在 LiH 这种弱相关体系里几乎就是 FCI）；但在**局域截断 LUCJ** 下，被丢弃的非相邻 $R_{ZZ}$ 贡献 $\propto\lambda^2$ 达到最大，截断误差最大。

**为何取 $\lambda=0.05$ 而非 $\lambda=1$？** 这里有两类竞争效应，都随 $\lambda$ 增长：

| 效应 | 行为 | 对能量误差的影响 |
|---|---|---|
| 采样覆盖（利） | 态偏离 HF，覆盖更多行列式 | $\downarrow$ |
| 局域截断误差（弊） | 丢弃的非相邻 $R_{ZZ}\propto\lambda^2$ | $\uparrow$ |

两者竞争给出最优 $\lambda^\star$（Optional Problem 1 的核心）。$\lambda=0.05$ 落在低 $\lambda$ 侧——扰动刚够产生**可采样**的关联，同时截断误差 $\sim\lambda^2=2.5\times 10^{-3}$ 完全可忽略。$\lambda=1$ 会让截断误差最大化（在局域 LUCJ 下直接钉死在 $1.587\,\text{mHa}$ 平台），所以"太大"。注意 $\lambda=0.05$ 不是真正的最优点，而是在 $S=1000$ 这种小采样预算下的**保守稳健选择**——既不让截断误差起势，又保留了 LUCJ 的纠缠结构以备 $S$ 增大后发挥作用。

### (a)(b) 的一致性

理想变分 SQD 下，局域截断表现为**能量平台**（不回升），因为变分能量对子空间单调——丢掉的组态永远进不来，能量就停在被截断的下界。真正的 U 型回升需要引入非变分因素：近似 LUCJ 态的保真度退化 $\propto\lambda^2$ 叠加硬件噪声，才会在大 $\lambda$ 端回升。(a) 报告的是变分视角下的平台，(b) 描述的是同一权衡在物理参数 $\lambda$ 上的体现——两者不矛盾，是从不同侧面看同一个"覆盖 vs 截断"权衡。

### 进阶：纠缠熵与采样数 $S$ 的定量关系

LUCJ 态的双激发权重 $w\sim\theta^2\propto\lambda^2$。要可靠采到至少一次双激发，需要 $S\cdot w\gtrsim 1$，即
$$S\gtrsim\frac{1}{w}\propto\frac{1}{\lambda^2}.$$
对 $\lambda=0.05$，$w\sim 1.25\times 10^{-4}$，需要 $S\gtrsim 8000$ 才能期望采到一次双激发——这与表中"$S=1000$ 采不到、$S=200\,000$ 轻松采到"的观察一致。纠缠熵 $S_2$ 大致满足 $2^{S_2}\cdot w\sim 1$ 时采样才开始有效覆盖，所以 $S$ 的标度本质上由纠缠熵和最小双激发权重的乘积决定。这是一个粗略的标度律，精确形式需要考虑组态权重的具体分布。

## 结论

1. **$S=1000$、$\lambda=0.05$ 下 HF-SQD 与 LUCJ-SQD 同为 $-7.86307516\,\text{Ha}$（误差 $1.829\,\text{mHa}$）**——这不是 LUCJ 失效，而是双激发概率 $\sim 1.25\times 10^{-4}$ 低于 $1/S=10^{-3}$ 的采样下限，是纯统计效应。
2. **$\lambda$ 扫描（$S=200\,000$）下**：完整 UCJ 在 $\lambda\gtrsim 0.7$ 收敛至 FCI；局域 LUCJ 因丢弃非相邻双激发停在 $1.587\,\text{mHa}$ 平台。能量下降的关键是采到双激发（Brillouin/Slater–Condon），单激发不降能。
3. **量子信息解释**：HF 是乘积态，LUCJ 通过 CNT 引入纠缠，成对 Givens 把振幅泄漏到双激发（$\propto\lambda^2$）。$\lambda=0$ 退化为 HF，$\lambda=1$ 在完整 UCJ 下逼近 FCI、但在局域 LUCJ 下截断误差最大。$\lambda=0.05$ 是"覆盖 vs 截断"权衡中保守稳健的选择——扰动刚够产生可采样关联，截断误差仍可忽略。
4. **采样数标度**：$S\gtrsim 1/w\propto 1/\lambda^2$，对 $\lambda=0.05$ 需 $S\gtrsim 8000$ 才能期望采到一次双激发。

## 代码说明

代码在 `code/` 目录下，主要脚本为 `problem_2_5_a_lucj_sqd.py`，依赖 Python 3.11 venv、TensorFlow 2.16、TensorCircuit 0.12、PySCF 2.14。

核心流程：
1. **分子与参考能量**：PySCF 计算 RHF 与 FCI 能量，冻结 1 个核心轨道，活性空间 (4e, 4o) → JW 映射 8 qubit。
2. **CCSD 振幅**：PySCF CCSD 求解 $t_1,t_2$，按 $\lambda$ 缩放。
3. **HF 态制备**：在占据比特 $q(0,0)=0$、$q(0,1)=1$ 上施加 $X$。
4. **LUCJ 电路**：Jastrow 项分解为 $R_Z\cdot R_Z\cdot R_{ZZ}$，Givens 分解为 $\text{CNOT}\cdot CR_Y\cdot\text{CNOT}$，按 $\lambda$ 缩放振幅。支持完整 UCJ 与局域 LUCJ 两种模式。
5. **采样**：从 TensorCircuit 精确态矢取 $|\langle\bm{x}|\Psi\rangle|^2$ 作多项分布，抽 $S$ 个比特串，MSB-first 解码。
6. **SQD**：比特串 $\to(\alpha,\beta)$ 行列式 $\to$ 去重，PySCF FCI 引擎在子空间构建 $H_{kl}=\langle D_k|\hat H|D_l\rangle$ 并精确对角化，得到 $E_{\text{SQD}}\ge E_{\text{FCI}}$。
7. **$\lambda$ 扫描**：对 $\lambda\in\{0,0.05,0.2,0.3,0.5,0.7,1.0\}$ 重复上述流程，$S=200\,000$ 隔离采样噪声，记录组态数与能量误差。
