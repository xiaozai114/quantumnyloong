# Q01：构建哈密顿量与映射比较

> Build Hamiltonian and Mapping Comparison
> 分子：(a) H₂ STO-3G（4 qubit）、(d) N₂ STO-3G（20 qubit）

---

## 题目

将费米子哈密顿量映射到量子比特 Pauli 算符。JW 和 Parity 是两种互斥的映射方式。闭壳层分子有 $N_\alpha$、$N_\beta$ 守恒（Z2 对称性），各消除 1 个 qubit。

- **(a)** 用 PySCF 计算 H₂ 的 RHF，提取 $h_{pq}$ 和 $(pq|rs)$，写出二次量子化 H
- **(b)** 以 $h_{01}a_0^\dagger a_1$ 为例，推导 JW 下的 Pauli 展开
- **(c)** 统计 H₂ 在 JW 和 Parity 下的 Pauli 项数和最大权重
- **(d)** 用两个 Z2 对称性（$N_\alpha=1, N_\beta=1$）把 H₂ 从 4 减到 2 qubit
- **(e)** 对 N₂（20 qubit）重复 (c)
- **进阶**：BK 映射 Z 串长 O(log n)，对 N₂（n=10）max weight 是多少？线路深度降多少倍？

---

## (a) 构建二次量子化哈密顿量

### 用户的想法

> "对原子轨道直接量子化"

### 讨论

大方向正确，但精确表述是：**用 PySCF 经典计算得到积分数值，再代入二次量子化的代数形式**。不是"直接量子化原子轨道"——轨道本身已经是量子化的，这里要的是用 RHF 算出的积分系数来**组装** H。

### 具体步骤

1. **PySCF 定义分子**：`gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")`
2. **跑 RHF**：`scf.RHF(mol).kernel()` → 得到分子轨道、轨道能量、$E_{\text{HF}}$
3. **提取积分**：
   - 单电子：$h_{pq} = \langle\phi_p|\hat{h}|\phi_q\rangle$（动能 + 核引力），用 `mf.get_hcore()` + 基变换
   - 双电子：$(pq|rs)$（化学家记号），用 `mol.intor("int2e")` + 基变换
4. **空间轨道 → 自旋轨道**：每个空间轨道 p 变成 $p\alpha$ 和 $p\beta$ 两个自旋轨道。H₂ 有 2 个空间轨道 → 4 个自旋轨道 → 4 qubit。$h_{p\alpha,q\beta} = \delta_{\alpha\beta} h_{pq}$（自旋正交）
5. **写出 H**：

$$H = \sum_{pq} h_{pq}\, a_p^\dagger a_q + \frac{1}{2}\sum_{pqrs} (pq\|rs)\, a_p^\dagger a_q^\dagger a_s a_r + E_{\text{core}}$$

其中 $(pq\|rs) = (pq|rs) - (pq|sr)$ 是反对称化积分，$E_{\text{core}}$ 是核排斥能。

### 关键注意点

- $p,q,r,s$ 是**自旋轨道**指标（不是空间轨道），所以求和范围是 0–3
- 自旋正交性使得很多积分天然为零（如 $h_{0\alpha,1\beta} = 0$）
- $E_{\text{core}}$ 是常数（核-核排斥），不涉及电子算符

---

## (b) $h_{01}a_0^\dagger a_1$ 的 JW 展开

### 用户的想法

> "直接塞JW变换，化简即可"

### 讨论

完全正确。关键是**利用 Pauli 代数在同一 qubit 上消掉 Z 串**——相邻轨道的 hopping 项没有 Z 串残留。

### 推导

JW 映射：
$$a_j^\dagger = \frac{X_j - iY_j}{2}\prod_{k<j}Z_k, \qquad a_j = \frac{X_j + iY_j}{2}\prod_{k<j}Z_k$$

对 $j=0$ 和 $j=1$：
$$a_0^\dagger = \frac{X_0 - iY_0}{2}, \qquad a_1 = \frac{X_1 + iY_1}{2}\, Z_0$$

乘起来：
$$a_0^\dagger a_1 = \frac{1}{4}(X_0 - iY_0)\, Z_0\, (X_1 + iY_1)$$

**关键简化**：$(X_0 - iY_0)$ 和 $Z_0$ 作用在**同一个 qubit** 上。用 Pauli 乘法 $XZ = -iY$，$YZ = iX$：

$$(X_0 - iY_0)Z_0 = X_0Z_0 - iY_0Z_0 = (-iY_0) - i(iX_0) = X_0 - iY_0$$

所以 Z 串被吸收：

$$a_0^\dagger a_1 = \frac{1}{4}(X_0 - iY_0)(X_1 + iY_1)$$

展开：

$$= \frac{1}{4}\big(X_0X_1 + iX_0Y_1 - iY_0X_1 + Y_0Y_1\big)$$

**加上厄米共轭** $a_1^\dagger a_0$（H 是厄米的，hopping 项总是成对出现）：

$$a_0^\dagger a_1 + a_1^\dagger a_0 = \frac{1}{2}\big(X_0X_1 + Y_0Y_1\big)$$

### 结论

$$h_{01}(a_0^\dagger a_1 + a_1^\dagger a_0) = \frac{h_{01}}{2}\big(X_0X_1 + Y_0Y_1\big)$$

**相邻 qubit 的 hopping 只有 X X 和 Y Y，没有 Z 串**——weight = 2。如果轨道不相邻（如 $a_0^\dagger a_3$），Z 串不会被消掉，weight 更大。

---

## (c) H₂ 的 Pauli 项数与最大权重

### 用户的想法

> "没get到"

### 讨论

这题要的是两个数字：**① Pauli 项总数**（H 展开成 Pauli string 后有几项）；**② 最大权重**（最长的 Pauli string 中非恒等算符的数量）。权重越大，VQE 测量越难。

### H₂（4 qubit）的 Pauli 项

按类型分类：

| 类型 | 例子 | 数量 | 权重 |
|------|------|------|------|
| 常数 | $E_{\text{core}} + \sum h_{pp}/2$ | 1 | 0 |
| 单 Z | $Z_0, Z_1, Z_2, Z_3$（来自 $\hat{n}_p = (I-Z_p)/2$） | 4 | 1 |
| 双 Z | $Z_iZ_j$（来自 Coulomb 积分） | 6 | 2 |
|  hopping | $X_iX_j, Y_iY_j$（可能带 Z 串） | ~8 | 2–4 |
| 对产生/湮灭 | $X_iY_jZ\cdots$ 等 | ~4 | 3–4 |

**JW 映射**：
- 总 Pauli 项数 ≈ **15**（取决于非零积分）
- 最大权重 = **4**（$a_0^\dagger a_3$ 的 Z 串 $Z_1Z_2$ + $X_0X_3$ 或 $Y_0Y_3$ → weight 4）

**Parity 映射**：
- 总 Pauli 项数 ≈ **15**（相同——两种映射只是换了 Pauli 基）
- 最大权重 = **4**（hopping 的 X 串代替了 JW 的 Z 串，但长度一样）

**结论**：对 H₂ 这种小分子，JW 和 Parity 的项数和最大权重相同——区别要到更大分子才显现。

---

## (d) Z2 对称性约化：4 → 2 qubit

### 用户的想法

> "全部叠加Z？不太确定"

### 讨论

方向对了一半——对称算符确实是 Z 型，但不是"全部叠加"，而是**两个特定的双 Z 算符** $Z_0Z_1$ 和 $Z_2Z_3$。

### 稳定子与对称性验证

**粒子数守恒算符**（在 JW 下 $\hat{n}_j = (I - Z_j)/2$）：

$$\hat{N}_\alpha = \hat{n}_0 + \hat{n}_1 = I - \frac{Z_0 + Z_1}{2}, \qquad
  \hat{N}_\beta  = \hat{n}_2 + \hat{n}_3 = I - \frac{Z_2 + Z_3}{2}$$

**Z₂ 对称算符**（$(-1)^{\hat{N}}$）：

$$S_\alpha = (-1)^{\hat{N}_\alpha} = Z_0 Z_1, \qquad S_\beta = (-1)^{\hat{N}_\beta} = Z_2 Z_3$$

在 $N_\alpha = N_\beta = 1$ 的物理扇区内：
$$S_\alpha |\psi\rangle = -|\psi\rangle, \qquad S_\beta |\psi\rangle = -|\psi\rangle$$

定义 stabilizer $\bar{S}_\alpha = -Z_0 Z_1$、$\bar{S}_\beta = -Z_2 Z_3$，使物理扇区内 $\bar{S}_\alpha = \bar{S}_\beta = +I$。$S_\alpha, S_\beta$ 满足 $S^2 = I$、$[S_\alpha, S_\beta] = 0$，构成阿贝尔 $\mathbb{Z}_2 \times \mathbb{Z}_2$ 对称群。

**对称性验证**：$[H_{\text{JW}}, Z_0 Z_1] = [H_{\text{JW}}, Z_2 Z_3] = 0$——哈密顿量不改变任一自旋方向的电子数。

### Tapering 约化

4 qubit 全空间 $\mathcal{H} = (\mathbb{C}^2)^{\otimes 4}$（16 维）按 $S_\alpha, S_\beta$ 的 $\pm 1$ 本征值分解为 $2^2 = 4$ 个子空间，每个 4 维。物理扇区 $\mathcal{H}_{+1,+1}$（$\bar{S}_\alpha = \bar{S}_\beta = +I$）为 4 维——恰好对应 2 电子扇区。

用 Clifford 变换 $U$ 将 stabilizer 旋转到单 qubit 的 $Z$：$U\bar{S}_\alpha U^\dagger = Z_{t_1}$，$U\bar{S}_\beta U^\dagger = Z_{t_2}$。在旋转基下，靶 qubit $t_1, t_2$ 固定为 $|0\rangle$（$Z = +1$），含 $X_{t_i}$ 或 $Y_{t_i}$ 的项投影为零被消去。OpenFermion 的 `taper_off_qubits` 实现了此过程。

### 约化后的 2-qubit H（保留 qubit 0 和 2）

$$H_{\text{red}} = c_0\, I + c_1\, Z_0 + c_2\, Z_2 + c_3\, Z_0 Z_2 + c_4\, Y_0 Y_2$$

**数值系数**（由原始积分经 tapering 求得）：

| 系数 | 数值 (Ha) | 对应项 |
|------|-----------|--------|
| $c_0$ | $-0.3383$ | $I$（常数项） |
| $c_1$ | $-0.3948$ | $Z_0$ |
| $c_2$ | $-0.3948$ | $Z_2$（$c_1 = c_2$，因 α/β 对称） |
| $c_3$ | $+0.0112$ | $Z_0 Z_2$ |
| $c_4$ | $-0.1812$ | $Y_0 Y_2$（hopping 项） |

**基态能量**（将 $H_{\text{red}}$ 对角化）：

$$E_0(H_{\text{red}}) = -1.1373\;\text{Ha}$$

与 FCI 精确值一致（$\Delta \sim 10^{-16}$ Ha），确认稳定子空间分解正确——消去的恰好是不含物理信息的规范自由度。

> **注**：若用 `symmetry_conserving_bravyi_kitaev`（SCBK），得到的是 $X_0 X_2$ 形式（而非 $Y_0 Y_2$），且 $c_1 = c_2 = +0.3948$、$c_4 = +0.1812$（符号翻转，相当于 $H \otimes H$ 旋转）。本质是同一物理的两种代数实现。

### Parity 下的约化

Parity 映射中，qubit j 存储前 j+1 个轨道占据数的**奇偶性**（而不是占据数本身）。最后一个 qubit 直接存储总奇偶性——在固定电子数的 sector 内是常数，可以直接删掉。中间的 qubit 存储 α/β 奇偶性——同样可以删。

**JW vs Parity 的区别**：JW 的对称性需要"发现"（$Z_0Z_1$ 是本征值已知的算符），Parity 的对称性是"内置"的（某个 qubit 直接就是奇偶性）。所以 Parity 约化更自然。

---

## (e) N₂（20 qubit）重复 (c)

### 用户的想法

> "同理重复，答案是3？"

### 讨论

用户猜的 3 可能是把 BK 的 $\log_2(10)$ 和本题搞混了。题目问的是 JW/Parity 的 max weight。

### 结果

**N₂/STO-3G（20 qubit）**：费米子哈密顿量共 **8269 项**，经映射后压缩为 **2951 个 Pauli 项**。

| 映射 | Pauli 项数 | 最大权重 |
|------|-----------|---------|
| JW     | 2951 | **20** |
| Parity | 2951 | 20 |
| BK     | 2951 | **13** |

**Pauli 权重分布**（按权重 $w$ 统计项数）：

| $w$ | JW | Parity | BK |
|-----|-----|--------|-----|
| 1 | 20 | 2 | 15 |
| 2 | 190 | 52 | 54 |
| 3 | 0 | 36 | 153 |
| 4 | 276 | 408 | 138 |
| 5 | 8 | 16 | 187 |
| 6 | 344 | 289 | 179 |
| 7 | 0 | 42 | **418** |
| 8 | 400 | 441 | 386 |
| 9 | 4 | 24 | **437** |
| 10 | 588 | 469 | **465** |
| 11 | 0 | 37 | 254 |
| 12 | 460 | 442 | 180 |
| 13 | 8 | 63 | **84** |
| 14 | 328 | 266 | 0 |
| 15 | 0 | 50 | 0 |
| 16 | 200 | 171 | 0 |
| 17 | 4 | 21 | 0 |
| 18 | 108 | 98 | 0 |
| 19 | 0 | 15 | 0 |
| 20 | 12 | 8 | 0 |
| **合计** | **2951** | **2951** | **2951** |

**关键观察**：
- JW 和 Parity 的权重分布覆盖整个 $[1, 20]$ 区间，最大权重 = 20（最远 hopping $a_0^\dagger a_{19}$ 的 Z 串长 18 + $X_0 X_{19}$）
- BK 的权重被限制在 $[1, 13]$，**完全没有权重 $> 13$ 的项**
- BK 下大部分项集中在权重 7–11（共 1865 项，占 63%），这是 Fenwick 树结构将 parity-check 编码在 $O(\log n)$ 个 qubit 的直接结果

**结论**：JW 和 Parity 的 max weight 都是 $O(n)$——这是它们在 NISQ 时代的致命弱点（VQE 测量需要同时操作 20 个 qubit）。BK 的 $O(\log n)$ 优势在 N₂ 上已经显现（13 vs 20）。

---

## 进阶：BK 映射

### 用户的想法

> "答案是3？"

### 讨论

用户猜的 3 可能是把 BK 的理论下限 $\log_2(n)$ 和实际最大权重搞混了。BK（Bravyi-Kitaev）映射用 **Fenwick 树结构**编码占据数和奇偶性，Z 串长度从 $O(n)$ 降为 $O(\log n)$。

### 结果

对 N₂（$n=10$ 空间轨道 → 20 自旋轨道），从 (e) 的完整计算得：

| 映射 | 最大 Pauli 权重 | 渐进复杂度 |
|------|----------------|-----------|
| JW   | 20 | $O(n) = n = 20$ |
| BK   | **13** | $O(\log n)$ |

$$\boxed{\text{电路深度比值} = \frac{\max(\text{JW weight})}{\max(\text{BK weight})} = \frac{20}{13} \approx 1.54\times}$$

**物理解释**：在 Trotter 化的量子电路中，每个权重为 $w$ 的 Pauli 项需要 $O(w)$ 个两比特纠缠门（阶梯型 CNOT 网络需 $w-1$ 个 CNOT）。**最大权重项决定电路深度的主导项**——低权重项可并行调度，高权重项涉及大量 qubit 难以并行。因此 BK 在 N₂ 上实现了约 **1.5× 的电路深度缩减**。

### 为什么是 13 而非理论下限？

BK 单体 Majorana 算符的理论最大权重约为 $2\log_2(n_{\text{spin}}) + 1 \approx 2 \times 4.32 + 1 \approx 9.6$，即单个 $\gamma_{2j}$ 或 $\gamma_{2j+1}$ 的权重上界约 9–11。

然而，双电子积分项 $a_p^\dagger a_r^\dagger a_s a_q$ 展开后是 **4 个 Majorana 算符的乘积**：

$$a_p^\dagger a_r^\dagger a_s a_q \propto \gamma_{2p}\gamma_{2r}\gamma_{2s}\gamma_{2q} + \cdots$$

当 4 个 Majorana 串的翻转集合（update set）重叠最大化时，权重可达 $2 \cdot \max(\text{update}) + 2 \cdot \max(\text{parity}) \approx 13$。

**BK 的实际最大权重 13 来自多体项的组合效应，而非单体极限。**

### 随体系增长的趋势

| 体系大小 $n$ | JW: $O(n)$ | BK: $O(\log n)$ | 深度比 |
|-------------|-----------|-----------------|--------|
| H₂ (4)   | 4  | 4  | 1.0×（无差异） |
| N₂ (20)  | 20 | 13 | 1.5× |
| C₆H₆ (42) | ~42 | ~16 | 2.6× |
| 蛋白质活性位点 (100+) | ~100+ | ~18 | 5×+ |

BK 的优势随体系增大愈发显著——在 NISQ 时代电路深度极度受限的情境下，$O(\log n)$ vs $O(n)$ 的差距对实际可计算体系的大小有决定性影响。

---

## 总结

| 映射 | H₂ max weight | N₂ max weight | Z 串长度 | N₂ 深度比（vs JW） |
|------|--------------|--------------|---------|-------------------|
| JW     | 4 | 20 | $O(n)$ | 1.0×（基准） |
| Parity | 4 | 20 | $O(n)$ | 1.0× |
| BK     | 4 | **13** | $O(\log n)$ | **1.54×** |

**核心结论**：
- H₂（4 qubit）太小，三种映射的 max weight 都是 4，看不出差异
- N₂（20 qubit）上 BK 的优势开始显现：max weight 13 vs 20，电路深度降低约 1.5×
- 体系越大，BK 的 $O(\log n)$ 优势越显著（C₆H₆ 约 2.6×，100+ qubit 约 5×+）
- 对 NISQ 时代的 VQE，BK 的浅线路优势巨大；但对 SQD（不需要测量 Hamiltonian 期望值），max weight 不那么重要——SQD 只关心采样效率

---

## 代码说明

代码在 `code/` 目录下（待补充）。
