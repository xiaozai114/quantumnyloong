# Q07：FCI 空间爆炸与 EWF 的标度优势

## 题目

FCI 维数 $\binom{2N}{N_e}$ 随体系增大指数爆炸。EWF（嵌入波函数）方法将体系切成 $F$ 个碎片，每个碎片含 $n_{\text{frag}}$ 个空间轨道、$N_{\text{frag}}$ 个电子，独立求解后拼接。

给定 5 个测试分子（H₂、LiH、H₂O、N₂、C₂H₄）。C₂H₄ 的 EWF 最大碎片为 20 qubit（$n_{\text{frag}}=10$，即 10 个空间轨道）。

- **(a)** 计算 5 个分子的 FCI 维数 $\binom{2N}{N_e}$，列表观察趋势。
- **(b)** 证明：固定 $n_{\text{frag}}$ 时，碎片数 $F = N/n_{\text{frag}}$，EWF 总代价为 $O(N)$。
- **(c)** 计算 C₂H₄ 最大碎片（20 qubit）的 FCI 维数，与全分子 $\sim 3.04\times10^7$ 比较。
- **(d)** 推导交叉点方程：直接 FCI 代价 $O\!\bigl(\binom{2N}{N_e}^3\bigr)$ vs EWF 代价 $F\times O\!\bigl(\binom{2n_{\text{frag}}}{N_{\text{frag}}}^3\bigr)$，证明 $n_{\text{frag}}<N$ 时 EWF 总优于直接 FCI。
- **进阶挑战**：20 qubit NISQ 约束下（$n_{\text{frag}}\le 10$），N₂ 和 C₂H₄ 的最优 $n_{\text{frag}}$？

## 我的想法

这道题的核心矛盾是 **指数爆炸 vs 线性标度**。

- 直接 FCI 的维数是 $\binom{2N}{N_e}$，对角化代价是维数的三次方 $D^3$，随着体系增大这是灾难性的。
- EWF 的思路是"分而治之"：把 $N$ 个轨道切成 $F$ 个碎片，每个碎片只算自己的 FCI，代价从"一个超大的 $D^3$"变成"$F$ 个小 $D_{\text{frag}}^3$"。
- 关键洞察：如果固定碎片大小 $n_{\text{frag}}$，那么 $D_{\text{frag}}$ 是常数（不随 $N$ 增长），只有碎片数 $F$ 线性增长，所以总代价 $O(N)$。
- 第 (d) 问的本质是要证明：在 $n_{\text{frag}}<N$ 的所有情况下，$F$ 个碎片的总代价严格小于直接 FCI 的代价。这需要证一个组合数的不等式。

我的证明思路是：设 $t = N/n_{\text{frag}}$，把 $\binom{2N}{N_e} = \binom{ta}{tb}$ 展开，拆出 $\binom{a}{b}$ 因子，证明剩余部分的下界大于 $t^{1/3}$。

## 讨论

### (a) 五分子 FCI 维数

FCI 维数公式：$D = \binom{2N}{N_e}$，其中 $N$ 为空间轨道数，$N_e$ 为电子数，$2N$ 为自旋轨道数。

| 分子 | 基组 | 空间轨道 $N$ | 电子数 $N_e$ | FCI 维数 $\binom{2N}{N_e}$ |
|:---:|:---:|:---:|:---:|---:|
| H₂ | STO-3G | 2 | 2 | $\binom{4}{2} = 6$ |
| LiH | STO-3G | 4 | 4 | $\binom{8}{4} = 70$ |
| H₂O | STO-3G | 7 | 10 | $\binom{14}{10} = 1001$ |
| N₂ | STO-3G | 10 | 14 | $\binom{20}{14} = 38760$ |
| C₂H₄ | STO-3G | 14 | 16 | $\binom{28}{16} = 30421755$ |

**趋势观察**：

- 从 H₂ 到 C₂H₄，轨道数仅增加 7 倍（2→14），但 FCI 维数增加了约 **500 万倍**（6→3.04×10⁷）。
- 每多 2 个空间轨道，维数约翻一个数量级。这正是"指数爆炸"的含义。
- C₂H₄ 的 FCI 维数 $\sim 3\times 10^7$，对角化代价 $\sim D^3 \sim 2.8\times 10^{22}$，这在经典计算机上已经不可行。
- 对角化代价按 $D^3$ 估计：H₂ 约 $10^2$，C₂H₄ 约 $10^{22}$——相差 20 个数量级。

### (b) 固定 $n_{\text{frag}}$ 时 EWF 的 $O(N)$ 标度

**设定**：将 $N$ 个空间轨道均匀切成 $F$ 个碎片，每片 $n_{\text{frag}}$ 个空间轨道。

**推理**：

1. 碎片数 $F = N / n_{\text{frag}}$（线性依赖 $N$）。
2. 每个碎片的 FCI 维数 $D_{\text{frag}} = \binom{2n_{\text{frag}}}{N_{\text{frag}}}$。当 $n_{\text{frag}}$ 固定时，$D_{\text{frag}}$ 是常数，不随 $N$ 增长。
3. 每个碎片的对角化代价 $O(D_{\text{frag}}^3) = O(1)$（因为 $D_{\text{frag}}$ 是常数）。
4. EWF 总代价 = 碎片数 × 每碎片代价 = $F \times O(D_{\text{frag}}^3) = \frac{N}{n_{\text{frag}}} \times O(1) = O(N)$。

**结论**：固定 $n_{\text{frag}}$，EWF 的计算代价随体系大小 $N$ **线性**增长，而非指数增长。这是 EWF 最核心的标度优势。

直觉上：体系每增大一点，只需多算几个相同大小的碎片，而非重新面对一个更大的指数空间。

### (c) C₂H₄ 最大碎片 FCI 维数

C₂H₄ 最大碎片为 20 qubit，即 $n_{\text{frag}} = 10$ 个空间轨道。

碎片内电子数按均匀分布估计：$N_{\text{frag}} = N_e \cdot n_{\text{frag}} / N = 16 \times 10 / 14 \approx 11.4$。取半填充估计（自旋轨道半满），碎片 FCI 维数最大取在 $N_{\text{frag}} = n_{\text{frag}} = 10$ 时：

$$D_{\text{frag}}^{\max} = \binom{2 \times 10}{10} = \binom{20}{10} = 184756$$

**与全分子比较**：

$$\frac{D_{\text{frag}}^{\max}}{D_{\text{full}}} = \frac{184756}{30421755} \approx 0.00607 \approx 0.6\%$$

碎片维数仅为全分子维数的 **0.6%**，降低了约 **99.4%**。

从对角化代价看：

- 直接 FCI：$D_{\text{full}}^3 \approx (3.04 \times 10^7)^3 \approx 2.8 \times 10^{22}$
- EWF（2 个碎片）：$2 \times D_{\text{frag}}^3 \approx 2 \times (1.85 \times 10^5)^3 \approx 1.26 \times 10^{16}$

代价降低约 **6 个数量级**，从不可行变为可行。

### (d) 交叉点方程与 EWF 恒优于 FCI 的证明

**代价比较**：

- 直接 FCI 代价：$C_{\text{FCI}} = O\!\bigl(\binom{2N}{N_e}^3\bigr)$
- EWF 代价：$C_{\text{EWF}} = F \times O\!\bigl(\binom{2n_{\text{frag}}}{N_{\text{frag}}}^3\bigr) = \frac{N}{n_{\text{frag}}} \times O\!\bigl(\binom{2n_{\text{frag}}}{N_{\text{frag}}}^3\bigr)$

假设电子均匀分布 $N_{\text{frag}} = N_e \cdot n_{\text{frag}} / N$，令两者相等得交叉点方程：

$$\binom{2n_{\text{frag}}}{N_e \cdot n_{\text{frag}} / N} = \binom{2N}{N_e} \cdot \left(\frac{n_{\text{frag}}}{N}\right)^{1/3}$$

**定理**：对于 $n_{\text{frag}} < N$（即 $t = N/n_{\text{frag}} > 1$），左边严格小于右边，故 EWF 总代价严格小于直接 FCI。

**证明**：

设 $t = N / n_{\text{frag}} > 1$，记 $a = 2n_{\text{frag}}$，$b = N_e \cdot n_{\text{frag}} / N$，则：

$$\binom{2N}{N_e} = \binom{ta}{tb}, \qquad \binom{2n_{\text{frag}}}{N_{\text{frag}}} = \binom{a}{b}$$

将 $\binom{ta}{tb}$ 展开为连乘积：

$$\binom{ta}{tb} = \prod_{i=1}^{tb} \frac{ta - i + 1}{tb - i + 1}$$

按 $i \bmod t$ 分组：当 $i - 1$ 是 $t$ 的倍数时（即 $i = jt + 1$，$j = 0, 1, \ldots, b{-}1$），共 $b$ 项。代入 $i = jt+1$：

$$\frac{ta - i + 1}{tb - i + 1} = \frac{ta - jt}{tb - jt} = \frac{t(a-j)}{t(b-j)} = \frac{a-j}{b-j}$$

这 $b$ 项恰好给出 $\binom{a}{b}$：

$$\prod_{j=0}^{b-1} \frac{a-j}{b-j} = \frac{a!/(a-b)!}{b!} = \binom{a}{b}$$

剩余 $b(t-1)$ 项（$t \nmid (i-1)$ 的项）的每一项：

$$\frac{ta - i + 1}{tb - i + 1} > 1 \quad (\text{因 } ta > tb \text{，即 } a > b)$$

所以：

$$\binom{ta}{tb} = \binom{a}{b} \times \prod_{\substack{i=1 \\ t \nmid (i-1)}}^{tb} \frac{ta - i + 1}{tb - i + 1}$$

剩余乘积中取最小因子（$i = tb$ 时）：

$$\frac{ta - tb + 1}{tb - tb + 1} = t(a - b) + 1$$

其余因子均 $\ge 1$，故：

$$\binom{ta}{tb} \ge \bigl[t(a-b) + 1\bigr] \cdot \binom{a}{b}$$

由于分子未占满（$N_e < 2N$），有 $a > b$，对于整数情形 $a - b \ge 1$，因此：

$$t(a - b) + 1 \ge t + 1 > t > t^{1/3} \quad (t > 1)$$

综合得：

$$\boxed{\binom{2N}{N_e} = \binom{ta}{tb} > t^{1/3} \cdot \binom{a}{b} = \left(\frac{N}{n_{\text{frag}}}\right)^{1/3} \cdot \binom{2n_{\text{frag}}}{N_{\text{frag}}}}$$

即：

$$\binom{2n_{\text{frag}}}{N_{\text{frag}}} < \binom{2N}{N_e} \cdot \left(\frac{n_{\text{frag}}}{N}\right)^{1/3}$$

两边立方并乘以 $F = N / n_{\text{frag}}$：

$$F \cdot \binom{2n_{\text{frag}}}{N_{\text{frag}}}^3 < \frac{N}{n_{\text{frag}}} \cdot \binom{2N}{N_e}^3 \cdot \frac{n_{\text{frag}}}{N} = \binom{2N}{N_e}^3$$

$$\Longrightarrow \quad C_{\text{EWF}} < C_{\text{FCI}}$$

当且仅当 $n_{\text{frag}} = N$（$t = 1$，不切分）时取等。$\blacksquare$

**物理意义**：只要碎片比全分子小（$n_{\text{frag}} < N$），EWF 的总代价就严格小于直接 FCI。碎片越小，优势越大；但碎片太小会丢失电子相关性（精度下降）。EWF 的实际选择是在 **计算代价** 与 **化学精度** 之间取折中。

### 进阶挑战：20 qubit NISQ 约束下的最优 $n_{\text{frag}}$

约束：$n_{\text{frag}} \le 10$（20 qubit）。

**N₂**（$N = 10$，$N_e = 14$，20 qubit）：

- 全分子刚好 20 qubit，$n_{\text{frag}} = 10$ 时 $F = 1$，退化为直接 FCI。
- EWF 无收益——N₂ 本身就卡在 NISQ 上限内。
- **最优 $n_{\text{frag}} = 10$**：直接 FCI，无需碎片化。

**C₂H₄**（$N = 14$，$N_e = 16$，28 qubit）：

- 全分子 28 qubit 超过 NISQ 上限，必须碎片化。
- $n_{\text{frag}} = 10$ 时 $F = \lceil 14/10 \rceil = 2$（碎片大小 10+4 或按化学键重分）。
- 碎片越小则 $F$ 越大、$D_{\text{frag}}$ 越小，代价更低但精度更差。
- 在 NISQ 上限内，**$n_{\text{frag}} = 10$** 是最优：用满 NISQ 预算，$F$ 最小（仅 2 片），每片 FCI 维数 $\binom{20}{10} = 184756$ 可控，精度最优。

**小结**：NISQ 约束下最优策略是"用满量子比特预算"——$n_{\text{frag}} = 10$，碎片尽可能大以保精度，碎片数尽可能少以降代价。

## 结论

1. **FCI 维数指数爆炸**：H₂→C₂H₄ 仅 7 倍轨道增长，维数增长 500 万倍。C₂H₄ 的 $D \approx 3\times10^7$，$D^3 \approx 10^{22}$，经典不可行。

2. **EWF 线性标度**：固定 $n_{\text{frag}}$ 时每碎片代价 $O(1)$，碎片数 $F = O(N)$，总代价 $O(N)$。这是从指数到线性的根本跃变。

3. **C₂H₄ 碎片降维**：最大碎片 $\binom{20}{10} = 184756$，仅为全分子的 0.6%，对角化代价降 6 个数量级。

4. **EWF 恒优**：通过组合数不等式 $\binom{ta}{tb} > t^{1/3}\binom{a}{b}$（$t > 1$）证明，$n_{\text{frag}} < N$ 时 EWF 代价严格小于 FCI，仅 $n_{\text{frag}} = N$ 时相等。

5. **NISQ 最优**：$n_{\text{frag}} \le 10$ 约束下，N₂ 退化为直接 FCI（$F{=}1$），C₂H₄ 取 $n_{\text{frag}} = 10$（$F{=}2$）。

## 代码说明

代码在 `code/fci_ewf_analysis.py`，用 Python `math.comb` 完成以下计算：

1. **(a) 维数表**：对 5 个分子计算 $\binom{2N}{N_e}$，输出表格。
2. **(c) 比值**：计算 $\binom{20}{10} / \binom{28}{16}$，验证约 0.6%。
3. **(d) 不等式验证**：对若干 $(N, n_{\text{frag}})$ 组合，验证 $\binom{2N}{N_e} > t^{1/3} \binom{2n_{\text{frag}}}{N_e \cdot n_{\text{frag}}/N}$ 严格成立。
4. **进阶**：打印 N₂ 和 C₂H₄ 在 $n_{\text{frag}} \le 10$ 下的碎片数与每片维数。
