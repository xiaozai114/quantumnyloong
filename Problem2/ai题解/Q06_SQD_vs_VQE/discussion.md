# Q06：SQD vs VQE——量子资源与能量精度（开放题）

## 题目

分子 (b) LiH（$N=8$）。VQE：$p=4$，$N_{\text{iter}}=100$；SQD：$S=1000$，$N_{\text{iter}}=5$；$\epsilon=10^{-3}$ Ha。

- **(a) 量子资源**：计算两种方法各自的线路执行总次数，比较数量级。
- **(b) 能量精度**：证明 $E_{\text{SQD}} = \min_{|\phi\rangle\in\mathcal{S}} \langle\phi|\hat{H}|\phi\rangle \leq \langle\Psi|\hat{H}|\Psi\rangle = E_{\text{VQE}}$。等号何时成立？
- **(c) 全面对比**：从线路深度、映射灵活性、ansatz 选择、噪声鲁棒性、测量开销、变分范围六个维度比较 VQE 与 SQD；并讨论 NISQ 时代与 FTQC 时代各自的最优选择。
- **进阶**：混合方案——VQE 浅线路做态制备 + SQD 子空间对角化做后处理，能量下界如何？

---

## 我的想法

读完题目后，我先把两个方法的"工作流"在脑子里过一遍，看看它们各自把工作量压在了哪一侧（量子侧 vs 经典侧）：

- **VQE** 是"全变分"路线：参数化的浅线路反复跑，每次迭代都要把哈密顿量拆成 $O(N^4)$ 个 Pauli 串逐一测量，还要做参数移位求梯度。量子侧承担了几乎所有"重活"，经典侧只做优化器更新。
- **SQD** 是"少跑量子、多做经典"路线：用一个**固定**的线路（典型是 LUCJ）采样一堆比特串，然后把"哪些组态重要"这件事交给经典对角化去决定。量子侧只负责"提供候选组态池"。

基于这个直觉，我预判：
1. 在 (a) 里，VQE 的总执行次数会因为"$N^4$ Pauli 项 × $1/\epsilon^2$ 采样 × $2p$ 参数移位 × $N_{\text{iter}}$ 迭代"四个因子叠在一起，量级会远大于 SQD 的"$S \times N_{\text{iter}}$"。
2. 在 (b) 里，关键观察是"VQE 制备的态 $|\Psi\rangle$ 本身就在 SQD 的采样子空间 $\mathcal{S}$ 内"——因为 $\mathcal{S}$ 就是按 $|\Psi\rangle$ 的支撑定义的。而 SQD 在 $\mathcal{S}$ 内做的是 FCI 精确对角化，取的是子空间内的最小值；最小值必然 $\leq$ 任何一个特定态（包括 $|\Psi\rangle$）的能量。所以 $E_{\text{SQD}} \leq E_{\text{VQE}}$。
3. 在 (c) 里，NISQ 时代量子资源稀缺、噪声大，SQD 的"少跑量子"占优；FTQC 时代量子资源充足，QPE 这种"深但只跑一次且给精确解"的方法会反超。
4. 进阶的混合方案最妙：用 VQE 的浅线路当"态制备器"，再用 SQD 的子空间对角化当"精度提升器"，理论上应该满足三明治不等式 $E_0 \leq E_{\text{hybrid}} \leq E_{\text{VQE}}$——下界来自子空间是全空间的子集（$\lambda_{\min}(H|_\mathcal{S}) \geq \lambda_{\min}(H|_{\text{full}})$），上界来自 (b) 的论证。

---

## 讨论

### (a) 量子资源对比

#### VQE 的总线路执行次数

VQE 单步迭代的开销来源：

1. **梯度估计（参数移位法则）**：$p$ 个参数，每个参数移位需要 2 次能量求值 → $2p$ 次线路执行；
2. **单次能量求值的 Pauli 串数**：第二量子化哈密顿量 $\hat{H} = \sum h_{pq} a_p^\dagger a_q + \frac{1}{2}\sum h_{pqrs} a_p^\dagger a_q^\dagger a_r a_s$，经 Jordan–Wigner 映射后共有 $O(N^4)$ 个 Pauli 项（双电子积分张量有 4 个下标）；
3. **每个 Pauli 串的采样精度**：要达到精度 $\epsilon$，由 Chernoff–Hoeffding 界，每个 Pauli 项的测量次数为 $O(1/\epsilon^2)$。

因此单次迭代的总测量次数：

$$n_{\text{VQE}}^{(\text{1 iter})} = 2p \times N^4 \times \frac{1}{\epsilon^2}$$

代入题目数值 $p=4$，$N=8$，$\epsilon=10^{-3}$ Ha：

$$n_{\text{VQE}}^{(\text{1 iter})} = 2 \times 4 \times 8^4 \times 10^{6} = 8 \times 4096 \times 10^{6} \approx 3.3 \times 10^{10}$$

乘以总迭代次数 $N_{\text{iter}} = 100$：

$$\boxed{n_{\text{VQE}} \approx 100 \times 3.3 \times 10^{10} \approx 3.3 \times 10^{12}}$$

#### SQD 的总线路执行次数

SQD 的迭代结构完全不同：
- 每次迭代从**固定**的量子线路（LUCJ）采样 $S$ 个比特串，量子侧只做"采样"，**不测 Pauli 期望值**；
- 采到的比特串送入经典侧做投影对角化（不占量子资源）；
- 迭代的意义是"用上一轮的对角化结果反馈调整线路"（组态恢复 / 自洽），但每轮的量子开销仍是 $S$。

因此总线路执行次数：

$$\boxed{n_{\text{SQD}} = N_{\text{iter}} \times S = 5 \times 1000 = 5000}$$

#### 数量级对比

| | VQE | SQD |
|---|---|---|
| 总线路执行次数 | $\sim 10^{12}$ | $\sim 10^{3}$ |
| 量子资源比 | $n_{\text{VQE}} / n_{\text{SQD}} \approx 10^{9}$ |

**结论**：SQD 比 VQE 节省约 9 个数量级的量子资源。这个差距本质来自两点：
1. VQE 把"求精度"这件事压在量子侧（$1/\epsilon^2$ 采样），SQD 把"求精度"压在经典侧（对角化是精确的，无统计涨落）；
2. VQE 必须为每个 Pauli 项单独测量（$N^4$ 因子），SQD 一次性采样就拿到所有组态信息。

#### 复杂度的通用形式

更一般地，去掉具体数值：

$$n_{\text{VQE}} = O\!\left(\frac{N^4}{\epsilon^2} \cdot p \cdot N_{\text{iter}}\right), \qquad n_{\text{SQD}} = O(S \cdot N_{\text{iter}})$$

VQE 对 $\epsilon$ 的依赖是二次的（统计采样），SQD 对 $\epsilon$ 几乎不敏感（经典对角化精度由子空间维数决定，与量子测量噪声无关）。这是 SQD 在 NISQ 时代立足的根本。

---

### (b) 变分不等式 $E_{\text{SQD}} \leq E_{\text{VQE}}$ 的证明

#### 设定

设 VQE 制备的量子态为：

$$|\Psi\rangle = \sum_{x} c_x |x\rangle$$

其中 $|x\rangle$ 是计算基（比特串本征态），$c_x$ 是振幅。

SQD 的核心步骤是"对 $|\Psi\rangle$ 在计算基下采样，收集出现过的组态"。这些被采到的组态张成一个子空间：

$$\mathcal{S} = \mathrm{span}\{|x\rangle : c_x \neq 0\}$$

（严格来说，采样是随机的，可能漏掉某些 $c_x \neq 0$ 的小振幅组态；这里取"理想采样"的极限，即 $\mathcal{S}$ 覆盖 $|\Psi\rangle$ 的全部支撑。采样截断的影响在进阶部分讨论。）

SQD 在 $\mathcal{S}$ 内做精确对角化，取最小本征值作为能量估计：

$$E_{\text{SQD}} = \min_{|\phi\rangle \in \mathcal{S}} \langle\phi|\hat{H}|\phi\rangle$$

VQE 的能量是它制备的特定态 $|\Psi\rangle$ 的期望值：

$$E_{\text{VQE}} = \langle\Psi|\hat{H}|\Psi\rangle$$

#### 证明

**第一步**：$|\Psi\rangle \in \mathcal{S}$。

因为 $|\Psi\rangle = \sum_x c_x |x\rangle$，而所有 $c_x \neq 0$ 的 $|x\rangle$ 都在 $\mathcal{S}$ 中，所以 $|\Psi\rangle$ 是 $\mathcal{S}$ 中向量的线性组合，即 $|\Psi\rangle \in \mathcal{S}$。

**第二步**：最小值 $\leq$ 任意特定值。

$E_{\text{SQD}}$ 是 $\hat{H}$ 在 $\mathcal{S}$ 上的**最小**期望值——对 $\mathcal{S}$ 中**所有**归一化态取最小。$|\Psi\rangle$ 是 $\mathcal{S}$ 中的**一个**特定归一化态。最小值必然 $\leq$ 集合中任意元素的值：

$$\boxed{E_{\text{SQD}} = \min_{|\phi\rangle \in \mathcal{S}} \langle\phi|\hat{H}|\phi\rangle \;\leq\; \langle\Psi|\hat{H}|\Psi\rangle = E_{\text{VQE}}}$$

#### 等号成立条件

等号成立当且仅当 $|\Psi\rangle$ 已经是 $\hat{H}$ 在 $\mathcal{S}$ 内的基态，即：

$$\hat{H}\big|_{\mathcal{S}} \,|\Psi\rangle = E_{\text{SQD}} \,|\Psi\rangle$$

换句话说，VQE 的 ansatz 已经"碰巧"命中了采样子空间内的精确基态。这在实际中几乎不会发生——因为 VQE 的 ansatz 受限于变分形式（如 UCCSD 截断），而 $\mathcal{S}$ 内的 FCI 基态是 unrestricted 的。所以**实际中严格不等号成立**：$E_{\text{SQD}} < E_{\text{VQE}}$。

#### 直觉解释

可以把 SQD 想成"在 VQE 给出的子空间里再做一次 FCI"：VQE 的工作是"挑出一个好的子空间"（决定哪些组态重要），SQD 的工作是"在这个子空间里精确求解"。前者是变分近似，后者是子空间内的精确解，所以 SQD 的能量永远不会比 VQE 高。这也就是为什么 SQD 被称为"post-VQE 的精度提升器"。

---

### (c) 六维全面对比与时代选择

#### 六维度对比表

| 维度 | VQE | SQD | 谁更优 |
|---|---|---|---|
| **线路深度** | 浅，$O(p)$（参数化 ansatz，门数随参数线性增长） | 中等，LUCJ 的 CNOT 深度 $O(n)$（本地纠缠 + 化学-激发双层） | VQE |
| **映射灵活性** | 任意映射（只测 Pauli 串，与映射无关） | 任意映射（只采样比特串，也与映射无关） | 平手 |
| **ansatz 选择** | 高度灵活（UCCSD、HEA、ADAPT-VQE、q-UCC 等） | 固定用 LUCJ（振幅来自经典 CCSD，不容变分） | VQE |
| **噪声鲁棒性** | 差（噪声污染期望值 → 梯度失真 → 优化卡在局部极小或 barren plateau） | 强（采样天然容忍噪声 + 组态恢复可纠错） | SQD |
| **测量开销** | $\sim 10^{12}$（由 (a)） | $\sim 5000$（由 (a)） | SQD |
| **变分范围** | 完全变分（参数可调，能适配不同分子/键长） | 非变分（线路固定，靠经典对角化调精度） | VQE |

**总结**：VQE 在"灵活性"维度赢（浅线路、自由 ansatz、完全变分），SQD 在"实用性"维度赢（噪声鲁棒、测量开销低 9 个数量级）。

#### NISQ 时代（$p_{2q} \sim 0.01$）：SQD 更实用

1. **测量开销**：VQE 的 $10^{12}$ 次测量在 NISQ 上根本跑不完——即使每次测量只需 1 μs，也要 $10^6$ 秒 ≈ 11.6 天。SQD 的 5000 次采样在毫秒级完成。
2. **噪声**：VQE 的梯度估计依赖**精确**的期望值测量，二阶门噪声让期望值偏差直接传到梯度上，优化器要么卡在局部极小，要么陷入 barren plateau。SQD 采样后做经典对角化（经典侧无噪声），组态恢复还能从带噪比特串里"猜回"正确的电子组态。
3. **迭代次数**：VQE 需要 100 次迭代，每次都要重跑线路；SQD 只需 5 次，且每次的量子侧开销只有 VQE 的 $1/10^9$。

**结论**：NISQ 时代量子资源极度稀缺、噪声极大，SQD 的"少跑量子、多做经典"策略完胜。VQE 虽然理论上更优雅，但在当前硬件上几乎跑不出有意义的结果。

#### FTQC 时代（$p_{2q} < 10^{-4}$）：QPE 碾压两者

进入 FTQC 时代后，纠错让量子线路可以"深而精确"，此时评价标准从"单次执行成本"转为"总 T-gate 数 + 是否给出精确解"。

| 方法 | 线路深度 | 重复次数 | T-gate 总数 | 解的精度 |
|---|---|---|---|---|
| VQE | $O(p)$（浅） | $10^{12}$ | $O(p \times 10^{12})$（巨大） | 变分上界 |
| SQD | $O(n)$（中） | $5000$ | $O(n \times 5000)$（中等） | 子空间精确 |
| QPE | $O(1/\epsilon)$（深） | $1$ | $O(N/\epsilon)$（最小） | 精确本征值 |

关键观察：
- VQE 虽然"浅线路"，但要重复 $10^{12}$ 次，**总资源反而最大**——这是 NISQ 思维在 FTQC 时代的彻底失效。
- QPE 虽然线路深，但**只跑一次**，总资源最小，且给出的是**精确**能量（非变分上界，无 ansatz 偏差）。
- SQD 居中——比 VQE 好（重复少 9 个数量级），比 QPE 差（仍是子空间精确，不是全空间精确）。

**结论**：FTQC 时代排名 $\boxed{\text{QPE} > \text{SQD} > \text{VQE}}$。EWF（量子嵌入）+ QPE 是 FTQC 时代的终极方案：嵌入把大分子切成小碎片，QPE 在每个碎片上给精确解，整体能量加和得到化学精度。

---

### 进阶：混合方案——VQE 浅线路做态制备 + SQD 做后处理

#### 思路

把两个方法的优点拼起来：
1. **VQE 阶段**：用浅线路（如 UCCSD 截断）制备一个"还不错"的态 $|\Psi_{\text{VQE}}\rangle$，作为态制备器。这一步量子开销小，且 ansatz 灵活。
2. **SQD 阶段**：对 $|\Psi_{\text{VQE}}\rangle$ 采样，收集组态张成子空间 $\mathcal{S}$，在 $\mathcal{S}$ 内做精确对角化，得到 $E_{\text{hybrid}} = \lambda_{\min}(\hat{H}|_\mathcal{S})$。

#### 三明治不等式

这个混合方案的能量满足"三明治不等式"：

$$\boxed{E_0 \leq E_{\text{hybrid}} \leq E_{\text{VQE}}}$$

**上界** $E_{\text{hybrid}} \leq E_{\text{VQE}}$：直接套用 (b) 的证明——$\mathcal{S}$ 是按 $|\Psi_{\text{VQE}}\rangle$ 的支撑定义的，$|\Psi_{\text{VQE}}\rangle \in \mathcal{S}$，子空间最小值 $\leq$ 任意特定态的能量。

**下界** $E_0 \leq E_{\text{hybrid}}$：$\mathcal{S}$ 是全 Hilbert 空间的子空间，$\hat{H}|_\mathcal{S}$ 的最小本征值 $\geq$ $\hat{H}$ 在全空间的最小本征值（在更小的集合上取最小，结果不会更小）。即：

$$E_{\text{hybrid}} = \lambda_{\min}(\hat{H}|_\mathcal{S}) \geq \lambda_{\min}(\hat{H}|_{\text{full}}) = E_0$$

#### 物理意义

- VQE 给出的是**上界**（变分原理），但受限于 ansatz，可能离 $E_0$ 较远；
- SQD 后处理把上界"往下压"——在 VQE 选出的子空间内做 FCI，能量只能降不能升；
- 随着 $S \to \infty$，$\mathcal{S}$ 覆盖 $|\Psi_{\text{VQE}}\rangle$ 的全部支撑，$E_{\text{hybrid}}$ 收敛到 $\lambda_{\min}(\hat{H}|_{\mathrm{supp}(\Psi_{\text{VQE}})})$，仍 $\geq E_0$（除非 ansatz 恰好覆盖真实基态）。

要进一步逼近 $E_0$，需要让 $\mathcal{S}$ 覆盖真实基态 $|\Psi_0\rangle$ 的支撑——这超出了"固定 VQE 态 + 采样"的范围，需要 SQD 的自洽迭代（组态恢复）来扩展子空间。

#### 严格上界的细节

注意一个细节：由于实际采样是随机的，会丢弃 $|\Psi_{\text{VQE}}\rangle$ 的小振幅组态，导致 $|\Psi_{\text{VQE}}\rangle \notin \mathcal{S}$（严格意义）。此时直接比较 $E_{\text{hybrid}}$ 与 $E_{\text{VQE}} = \langle\Psi_{\text{VQE}}|\hat{H}|\Psi_{\text{VQE}}\rangle$ 会有采样截断误差（量级 $\sim 10^{-5}$ Ha）。

严格可比的上界应该取"VQE 态投影到 $\mathcal{S}$ 后的能量"：

$$E_{\text{VQE}}|_\mathcal{S} = \frac{\langle\Psi_{\mathcal{S}}|\hat{H}|\Psi_{\mathcal{S}}\rangle}{\langle\Psi_{\mathcal{S}}|\Psi_{\mathcal{S}}\rangle}, \quad |\Psi_{\mathcal{S}}\rangle = P_\mathcal{S} |\Psi_{\text{VQE}}\rangle$$

此时 $|\Psi_{\mathcal{S}}\rangle \in \mathcal{S}$，严格满足 $E_{\text{hybrid}} \leq E_{\text{VQE}}|_\mathcal{S}$。代码中实现了这个严格上界（见 `sqd_diag.py:vqe_energy_in_subspace`）。

---

## 结论

1. **(a) 量子资源**：$n_{\text{VQE}} \approx 3.3 \times 10^{12}$，$n_{\text{SQD}} = 5000$，SQD 比 VQE 节省约 9 个数量级。本质原因是 VQE 把"求精度"压在量子侧（$1/\epsilon^2$ 统计采样 + $N^4$ Pauli 串），SQD 把"求精度"压在经典侧（精确对角化）。

2. **(b) 变分不等式**：$E_{\text{SQD}} \leq E_{\text{VQE}}$。证明关键是 $|\Psi\rangle \in \mathcal{S}$（VQE 态在 SQD 采样子空间内），而 SQD 取的是子空间内的最小期望值，最小值 $\leq$ 任意特定态。等号成立当且仅当 VQE 态恰为 $\mathcal{S}$ 内的基态（实际中几乎不发生）。

3. **(c) 六维对比**：VQE 在线路深度、ansatz 灵活性、变分范围上占优；SQD 在噪声鲁棒性、测量开销上占优（映射灵活性平手）。
   - **NISQ 时代**：选 SQD——量子资源稀缺 + 噪声大，"少跑量子、多做经典"是唯一可行路线。
   - **FTQC 时代**：排名 QPE > SQD > VQE——QPE 给精确解且总 T-gate 最少，VQE 的"浅线路"优势被 $10^{12}$ 次重复抵消。

4. **进阶混合方案**：VQE 浅线路做态制备 + SQD 子空间对角化做后处理，满足三明治不等式 $E_0 \leq E_{\text{hybrid}} \leq E_{\text{VQE}}$。上界来自 (b)，下界来自子空间是全空间的子集。随采样数增加，$E_{\text{hybrid}}$ 单调逼近 $\lambda_{\min}(\hat{H}|_{\mathrm{supp}(\Psi_{\text{VQE}})})$。

---

## 代码说明

代码在 `code/` 目录下，共 4 个文件，实现了进阶部分的混合方案并验证三明治不等式。

### 文件结构

```
code/
├── hamiltonian.py     # 构建 LiH 哈密顿量
├── vqe.py             # UCCSD 截断 VQE
├── sqd_diag.py        # SQD 采样 + 子空间对角化
└── main.py            # 主流水线 + 收敛曲线绘制
```

### 各文件职责

**`hamiltonian.py`** — 分子哈密顿量构建
- 用 OpenFermion + PySCF 构建 LiH（STO-3G，键长 1.595 Å）的分子哈密顿量。
- 关键步骤：冻结 Li 1s 芯轨道，取 4 个 active 空间轨道 → 8 qubit（对应题目 $N=8$）。
- 输出：JW 映射后的稀疏矩阵 `H`、active 电子数（=2）、active-space FCI 能量 `E0`（作为三明治不等式下界）。

**`vqe.py`** — VQE 浅线路态制备
- 用 TensorCircuit 实现 UCCSD 截断 ansatz：从 HF 参考态 $|11000000\rangle$ 出发，作用保粒子数的 Givens 旋转（单激发 + 双激发）。
- 参数数：`count_params` 返回 $n_{\text{single}} + n_{\text{double}}$（LiH active space 下为 14 个参数）。
- 优化器：scipy L-BFGS-B，纯 numpy 后端（无需 GPU）。
- 输出：优化后的能量 `E_VQE`、态振幅向量 `psi`、参数 `params`。

**`sqd_diag.py`** — SQD 后处理
- `sample_configs(psi, n_shots, ...)`：对 VQE 态在计算基下采样 `n_shots` 次，去重并过滤到正确粒子数扇区，返回组态索引列表。
- `subspace_diagonalize(H, config_indices)`：把 `H` 投影到这些组态张成的子空间，取最小本征值 `E_hybrid`。
- `vqe_energy_in_subspace(psi, H, config_indices)`：计算 VQE 态投影到子空间后的能量 `E_VQE|S`，作为严格可比的上界（解决采样截断导致 $|\Psi\rangle \notin \mathcal{S}$ 的细节问题）。

**`main.py`** — 主流水线
1. 构建哈密顿量，打印 $E_0$（下界）；
2. 跑 VQE，得到 $E_{\text{VQE}}$（上界）；
3. 固定 $S=2000$ 采样 + 子空间对角化，得到 $E_{\text{hybrid}}$；
4. 验证三明治不等式 $E_0 \leq E_{\text{hybrid}} \leq E_{\text{VQE}}|_\mathcal{S}$；
5. 扫描采样数 $S \in \{5, 10, ..., 2560\}$（每个 $S$ 取 8 个 seed 平均），绘制 $E_{\text{hybrid}}$ 随子空间维数收敛到 $E_0$ 的曲线，保存为 `sandwich_convergence.png`。

### 运行方式

```bash
/path/to/qchem/bin/python main.py
```

### 预期输出

- 终端打印 $E_0$、$E_{\text{VQE}}$、$E_{\text{hybrid}}$、$E_{\text{VQE}}|_\mathcal{S}$，并验证三明治不等式成立。
- `sandwich_convergence.png`：横轴为子空间维数 $\dim(\mathcal{S})$，纵轴为能量；红色曲线 $E_{\text{hybrid}}$ 从上方单调逼近绿色虚线 $E_0$，蓝色虚线 $E_{\text{VQE}}$ 在最上方。
