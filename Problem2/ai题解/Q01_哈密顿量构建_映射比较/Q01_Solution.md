# Q01 解答：构建哈密顿量与映射比较

分子：(a) H₂ STO-3G（4 qubit）、(d) N₂ STO-3G（20 qubit）

---

## (a) 二次量子化哈密顿量

用 PySCF 跑 RHF，提取积分后写出：

$$H = \sum_{pq} h_{pq}\, a_p^\dagger a_q + \frac{1}{2}\sum_{pqrs} (pq\|rs)\, a_p^\dagger a_q^\dagger a_s a_r + E_{\text{core}}$$

- $p,q,r,s$ 是**自旋轨道**指标（0–3），排序 $(0\alpha, 1\alpha, 0\beta, 1\beta)$
- $(pq\|rs) = (pq|rs) - (pq|sr)$ 是反对称化积分
- $E_{\text{core}}$ 是核排斥能（常数）
- 自旋正交：$h_{p\alpha, q\beta} = \delta_{\alpha\beta} h_{pq}$，很多积分为零

参考值：$E_{\text{HF}} = -1.1167$ Ha，$E_{\text{FCI}} = -1.1373$ Ha。

---

## (b) $h_{01}a_0^\dagger a_1$ 的 JW 展开

JW 映射：$a_j^\dagger = \frac{X_j - iY_j}{2}\prod_{k<j}Z_k$

对 $j=0, 1$：

$$a_0^\dagger = \frac{X_0 - iY_0}{2}, \qquad a_1 = \frac{X_1 + iY_1}{2} Z_0$$

乘积：

$$a_0^\dagger a_1 = \frac{1}{4}(X_0 - iY_0)\, Z_0\, (X_1 + iY_1)$$

**关键**：$(X_0 - iY_0)$ 和 $Z_0$ 在同一 qubit 上。用 $XZ = -iY$，$YZ = iX$：

$$(X_0 - iY_0)Z_0 = X_0Z_0 - iY_0Z_0 = -iY_0 + X_0 = X_0 - iY_0$$

Z 串被吸收：

$$a_0^\dagger a_1 = \frac{1}{4}(X_0 - iY_0)(X_1 + iY_1) = \frac{1}{4}\big(X_0X_1 + iX_0Y_1 - iY_0X_1 + Y_0Y_1\big)$$

加上厄米共轭 $a_1^\dagger a_0$（虚部消去）：

$$h_{01}\big(a_0^\dagger a_1 + a_1^\dagger a_0\big) = \frac{h_{01}}{2}\big(X_0X_1 + Y_0Y_1\big)$$

**结论**：相邻 qubit 的 hopping 只有 XX 和 YY，无 Z 串，weight = 2。

---

## (c) H₂ 的 Pauli 项数与最大权重

按类型分类（4 qubit）：

| 类型 | 来源 | 数量 | 权重 |
|------|------|------|------|
| 常数 | $E_{\text{core}}$ + 对角项 | 1 | 0 |
| 单 Z | $\hat{n}_p = (I-Z_p)/2$ | 4 | 1 |
| 双 Z | Coulomb 积分 | 6 | 2 |
| hopping | XX + YY（同自旋） | 4 | 2 |
| hopping | 带 Z 串（跨自旋/远端） | 2–4 | 3–4 |

**JW**：≈15 项，max weight = **4**（$a_0^\dagger a_3$ 带 $Z_1Z_2$）

**Parity**：≈15 项，max weight = **4**（X 串代替 Z 串，长度一样）

**结论**：H₂ 太小，两种映射无区别。

---

## (d) Z2 对称性约化：4 → 2 qubit

**对称算符**：$\hat{S}_\alpha = Z_0Z_1$，$\hat{S}_\beta = Z_2Z_3$

**验证**：$N_\alpha = 1$ → $n_0 + n_1 = 1$ → $Z_0Z_1 = -1$（两种占据方式都给 −1）

**约化**：$Z_1 = -Z_0$，$Z_3 = -Z_2$ → qubit 1 和 3 冗余，消去。

**数算符约化**：

$$\hat{n}_1 = \frac{I - Z_1}{2} = \frac{I + Z_0}{2} = I - \hat{n}_0, \qquad \hat{n}_3 = I - \hat{n}_2$$

**hopping 约化**：$X_0X_1 + Y_0Y_1 \to 2X_0$（对称 sector 内）

**约化后 2-qubit H**：

$$H_{\text{red}} = c_0 I + c_1 Z_0 + c_2 Z_2 + c_3 Z_0Z_2 + c_4 X_0 + c_5 X_2$$

约化前 15 项 → 约化后 6 项；max weight 4 → 2。

**JW vs Parity 的区别**：JW 需"发现"对称算符 $Z_0Z_1$ 并做 tapering；Parity 的最后一个 qubit 直接存储总奇偶性，是常数可直接删掉——**Parity 的 Z2 对称性是内置的，约化更自然**。

---

## (e) N₂（20 qubit）重复 (c)

- 空间轨道 10 → 自旋轨道 20（qubit）
- 电子 14（$N_\alpha = N_\beta = 7$）
- FCI 维数 $\binom{20}{14} = 38{,}760$

**JW**：Pauli 项数 $O(N^4) \sim 10^4$，max weight = **20**（最远 hopping $a_0^\dagger a_{19}$：Z 串长 18 + XX → weight 20）

**Parity**：项数相同，max weight = **20**

---

## 进阶：BK 映射

BK 用二叉树编码，Z 串长 $O(\log n)$。

对 N₂（20 qubit）：max weight ≈ $\lceil\log_2 20\rceil + 2 = $ **7**

| 映射 | max weight（N₂） | Z 串长 | 适用 |
|------|-----------------|--------|------|
| JW | 20 | $O(n)$ | 教学、SQD |
| Parity | 20 | $O(n)$ | Z2 约化 |
| BK | ~7 | $O(\log n)$ | VQE（浅线路） |

线路深度降低约 3 倍（20 → 7）。

**注**：SQD 不需要测量 H 期望值，max weight 不重要；VQE 才需要。
