# Q04：LUCJ 量子线路的门分解

## 题目

考虑一层 Local Unitary Cluster Jastrow（LUCJ）形式的波函数拟设

$$
\ket{\Psi_{\mathrm{LUCJ}}}=U\,e^{i\hat J_{\mathrm{local}}}\,U^\dagger\ket{\mathrm{HF}},
$$

其中 $U=e^{\hat\kappa}$ 是由 Givens 旋转构成的轨道旋转，$\hat J_{\mathrm{local}}$ 是局域截断的 Jastrow 算符。由于算符从右向左作用于态，实际线路执行顺序为

$$
\ket{\mathrm{HF}}\xrightarrow{\;U^\dagger\;} \xrightarrow{\;e^{i\hat J_{\mathrm{local}}}\;}\xrightarrow{\;U\;}\ket{\Psi_{\mathrm{LUCJ}}}.
$$

要求回答：

- (a) 在 Jordan–Wigner 映射下，把二体 Jastrow 算符 $e^{iJ_{ij}\hat n_i\hat n_j}$ 推导为 $R_Z$ 与 $R_{ZZ}$ 的组合；
- (b) 为 H$_2$（4 个自旋轨道、4 个量子比特）搭建完整单层 LUCJ 线路，统计 CNOT 数量与线路深度；
- (c) 对一维链 $n=20$，论证完整 UCJ 的 $R_{ZZ}$ 层深度为 $O(n^2)$、LUCJ 为 $O(1)$，给出具体数值；
- 进阶：二维 heavy-hex 拓扑上 LUCJ 的 $R_{ZZ}$ 层深度。

约定：$R_Z(\phi)=e^{-i\phi Z/2}$、$R_{ZZ}(\phi)=e^{-i\phi Z_iZ_j/2}$，位串顺序 $\ket{b_3b_2b_1b_0}$，H$_2$ 的 HF 态为 $\ket{0011}$。

## 我的想法

第一反应是：Jastrow 相 $e^{iJ\hat n_i\hat n_j}$ 看起来是两体算符，但 JW 映射下 $\hat n_i$ 只含局域 $Z$，所以它本质上是几个对易 Pauli 串的乘积，应该可以直接拆成单比特 $R_Z$ 加一个 $R_{ZZ}$。

Givens 旋转这部分则要想清楚"为什么不能直接用 $R_Y$"：纯 $R_Y$ 会把 $\ket{00}$ 与 $\ket{11}$ 也混合掉，破坏电子数守恒；只有把 $R_Y$ 用 CNOT 夹起来、变成"受控"形式（CRY 或更紧凑的 2-CNOT 实现），才能把混合限制在 $\ket{01}\leftrightarrow\ket{10}$ 子空间。这是 LUCJ 区别于一般化学启发拟设的关键。

资源估计的核心是区分两件事：(1) **项数**（串行执行时等于深度）；(2) **并行后深度**（按边着色调度后能压到多少）。完整 UCJ 在 20 比特上有 $\binom{20}{2}=190$ 项，但即使是 $K_{20}$ 的边着色也只有 19 种颜色——所以"原生 $R_{ZZ}$ 并行深度"是 19，仍是 $O(n)$。题目的 $O(n^2)$ 指的是**项数**；LUCJ 把项数从 $O(n^2)$ 砍到 $O(n)$，再借助一维链的 2-边着色把并行深度压到 $O(1)=2$。

## 讨论

### 1. Jastrow 项到 $R_Z$/$R_{ZZ}$ 的映射

JW 映射下占据数算符为

$$
\hat n_i=\hat a_i^\dagger\hat a_i=\frac{I-Z_i}{2}.
$$

对 $i\neq j$，将两个 $Z$ 展开：

$$
\hat n_i\hat n_j=\frac{1}{4}(I-Z_i)(I-Z_j)=\frac{1}{4}(I-Z_i-Z_j+Z_iZ_j).
$$

右侧四项两两对易（都是 $Z$ 型 Pauli 串），所以指数可直接拆开：

$$
\begin{aligned}
e^{iJ_{ij}\hat n_i\hat n_j}
&=\exp\!\left[\frac{iJ_{ij}}{4}(I-Z_i-Z_j+Z_iZ_j)\right] \\
&=e^{iJ_{ij}/4}\cdot e^{-iJ_{ij}Z_i/4}\cdot e^{-iJ_{ij}Z_j/4}\cdot e^{+iJ_{ij}Z_iZ_j/4} \\
&=e^{iJ_{ij}/4}\cdot R_Z^{(i)}\!\left(\tfrac{J_{ij}}{2}\right)\cdot R_Z^{(j)}\!\left(\tfrac{J_{ij}}{2}\right)\cdot R_{ZZ}^{(ij)}\!\left(-\tfrac{J_{ij}}{2}\right).
\end{aligned}
$$

整体相位 $e^{iJ_{ij}/4}$ 不影响任何测量概率或能量期望，丢弃即可。物理结论是：

> 一个二体 Jastrow 相位 $\Leftrightarrow$ 两个 $R_Z$ + 一个 $R_{ZZ}$；其中 $R_{ZZ}$ 由两个 CNOT 实现：
> $$R_{ZZ}^{(ij)}(\phi)=\mathrm{CNOT}_{i\to j}\,R_Z^{(j)}(\phi)\,\mathrm{CNOT}_{i\to j}.$$

### 2. H₂ 的局域 Jastrow 层

H$_2$ 四个量子比特排成一维链 $0-1-2-3$。LUCJ 只保留相邻轨道对，所以

$$
\hat J_{\mathrm{local}}=J_{01}\hat n_0\hat n_1+J_{12}\hat n_1\hat n_2+J_{23}\hat n_2\hat n_3.
$$

三个二体项都只是 $\hat n$ 的多项式，所以两两对易，指数可拆：

$$
e^{i\hat J_{\mathrm{local}}}=e^{iJ_{01}\hat n_0\hat n_1}\cdot e^{iJ_{12}\hat n_1\hat n_2}\cdot e^{iJ_{23}\hat n_2\hat n_3}.
$$

代入上面的映射，再把**同一量子比特上的多个 $R_Z$ 合并**（因为它们也两两对易），就得到完整的 Jastrow 门序列（忽略整体相位，记 $\doteq$）：

$$
\boxed{
\begin{aligned}
e^{i\hat J_{\mathrm{local}}}\doteq\;&
R_Z^{(0)}\!\left(\tfrac{J_{01}}{2}\right)
R_Z^{(1)}\!\left(\tfrac{J_{01}+J_{12}}{2}\right)
R_Z^{(2)}\!\left(\tfrac{J_{12}+J_{23}}{2}\right)
R_Z^{(3)}\!\left(\tfrac{J_{23}}{2}\right) \\
&\times R_{ZZ}^{(01)}\!\left(-\tfrac{J_{01}}{2}\right)
R_{ZZ}^{(12)}\!\left(-\tfrac{J_{12}}{2}\right)
R_{ZZ}^{(23)}\!\left(-\tfrac{J_{23}}{2}\right).
\end{aligned}
}
$$

### 3. 数守恒 Givens 旋转

Givens 旋转 $G_{ij}(\theta)$ 的作用是只在 $\ket{01}$、$\ket{10}$ 这两个**单占据**态之间旋转，而严格保持 $\ket{00}$、$\ket{11}$：

$$
\begin{aligned}
G(\theta)\ket{01}&=\cos\theta\,\ket{01}+\sin\theta\,\ket{10},\\
G(\theta)\ket{10}&=-\sin\theta\,\ket{01}+\cos\theta\,\ket{10},\\
G(\theta)\ket{00}&=\ket{00},\quad G(\theta)\ket{11}=\ket{11}.
\end{aligned}
$$

对应矩阵（基序 $\{\ket{00},\ket{01},\ket{10},\ket{11}\}$）

$$
G(\theta)=
\begin{pmatrix}
1&0&0&0\\
0&\cos\theta&-\sin\theta&0\\
0&\sin\theta&\cos\theta&0\\
0&0&0&1
\end{pmatrix}.
$$

**CRY 直观实现**：把一个 $\mathrm{CRY}_{0\to1}(2\theta)$ 夹在两个同向 $\mathrm{CNOT}_{1\to0}$ 之间，就能精确实现 $G(\theta)$：

$$
\boxed{
\mathrm{CNOT}_{1\to0}\;\longrightarrow\;\mathrm{CRY}_{0\to1}(2\theta)\;\longrightarrow\;\mathrm{CNOT}_{1\to0}.
}
$$

这个结构非常适合教学说明：CRY 等价于"控制为 $\ket{1}$ 时才执行 $R_Y$"，再加上前后两个 CNOT 做"控制位反转"，最后整个门的非平凡作用就被严格限制在 $\ket{01}\leftrightarrow\ket{10}$ 子空间里。

> **陷阱提示**：如果把中间换成无控制的 $R_{Y,1}(2\theta)$，门会额外混合 $\ket{00}\leftrightarrow\ket{11}$，破坏粒子数守恒，只保留费米子宇称——这种"广义 Givens"不能用于 LUCJ 的轨道旋转。

**2-CNOT 紧凑实现**（题目资源口径）：CRY 本身展开到 $\{R_Y,\mathrm{CNOT}\}$ 通常要 4 个 CNOT，不满足"2 CNOT + 单比特门"的要求。一个标准的 2-CNOT 实现基于单激发算符

$$
U_{\mathrm{SE}}(\theta)=\exp\!\left[-\frac{i\theta}{2}(X_1X_0+Y_1Y_0)\right],
$$

它天然只混合 $\ket{01},\ket{10}$（非对角元带 $-i$ 相位），按时间顺序写成

$$
R_X^{(1)}\!\left(\tfrac{\pi}{2}\right)
\to \mathrm{CNOT}_{1\to0}
\to \big[R_X^{(1)}(\theta)\otimes R_Y^{(0)}(\theta)\big]
\to \mathrm{CNOT}_{1\to0}
\to R_X^{(1)}\!\left(-\tfrac{\pi}{2}\right),
$$

再在 $q_1$ 前后各加一个局域相位校正 $R_Z^{(1)}(-\pi/2)$、$R_Z^{(1)}(+\pi/2)$，就能把 $-i$ 相位修掉、得到实形式 $G(\theta)$。

总结：**一个 Givens = 2 CNOT + 6 单比特门**；若单比特门可并行，门深度为 7。

### 4. H₂ 单层 LUCJ 的 CNOT 数与公式 $N=4g+6$

设轨道旋转 $U$ 编译为 $g$ 个 Givens 旋转。线路从左到右执行：$U^\dagger$ → Jastrow → $U$。

- $U^\dagger$ 的 $g$ 个 Givens 共需 $2g$ 个 CNOT；
- Jastrow 层 3 个 $R_{ZZ}$，每个 2 个 CNOT，共 $6$；
- $U$ 同样需要 $2g$ 个 CNOT。

所以单层 LUCJ 的 CNOT 总数为

$$
\boxed{N_{\mathrm{CNOT}}=2g+6+2g=4g+6.}
$$

若把 $U$ 视为 4 个自旋轨道上的一般实正交旋转，则可用 $g=\binom{4}{2}=6$ 个 Givens 完成，代入：

$$
\boxed{N_{\mathrm{CNOT}}=4\times 6+6=30.}
$$

如果额外限制保持 $N_\alpha,N_\beta$ 守恒（自旋分块），可用的 $g$ 会减少，但公式 $4g+6$ 仍然成立，只需替换对应的 $g$。

### 5. H₂ 的线路深度

**Jastrow 部分**：4 个合并后的 $R_Z$ 可同时执行（不同比特），贡献 1 层。3 个 $R_{ZZ}$ 的边集合为 $\{(0,1),(1,2),(2,3)\}$，是一维路径，边色数为 2：$\{(0,1),(2,3)\}$ 一组、$\{(1,2)\}$ 一组。每个 $R_{ZZ}$ 按 $\mathrm{CNOT}-R_Z-\mathrm{CNOT}$ 占 3 层，所以

$$
D_J=1+3+3=7.
$$

**Givens 部分**：单个 Givens 的基础门深度为 7。

- **保守串行**：6 个 Givens 直接串行，$D_U=6\times7=42$；总深度

$$
D_{\mathrm{LUCJ}}^{\mathrm{conservative}}=42+7+42=91.
$$

- **匹配式三层分解**：把 $\binom{4}{2}=6$ 个 Givens 安排到三层不相交匹配
  $$\{(0,1),(2,3)\},\quad \{(0,2),(1,3)\},\quad \{(0,3),(1,2)\},$$
  每层两个 Givens 并行，$D_U=3\times7=21$，总深度降为

$$
\boxed{D_{\mathrm{LUCJ}}^{\mathrm{optimized}}=21+7+21=49.}
$$

若计入 HF 初态制备，$\ket{0000}\to\ket{0011}$ 的两个 $X$ 门可并行执行，深度再加 1。

### 6. $n=20$ 一维链：UCJ vs LUCJ

**项数（串行深度上界）**：

| 方案 | 二体项数 | 量级 |
|---|---|---|
| 完整 UCJ | $\binom{20}{2}=190$ | $O(n^2)$ |
| LUCJ | $20-1=19$ | $O(n)$ |

**原生 $R_{ZZ}$ 并行深度**（忽略 SWAP 路由、允许任意两比特耦合）：

- 完整 UCJ 的相互作用图是 $K_{20}$，边色数 $=n-1=19$，故**原生 $R_{ZZ}$ 层深度**为 19，记为 $O(n)$；题目所说的 $O(n^2)$ 指的是**项数**而非调度后的并行深度。
- LUCJ 的相互作用图是路径 $P_{20}$，边色数 $=2$：
  $$\mathcal{E}_{\mathrm{even}}=\{(0,1),(2,3),\ldots,(18,19)\},\quad \mathcal{E}_{\mathrm{odd}}=\{(1,2),(3,4),\ldots,(17,18)\},$$
  故

$$
\boxed{D_{R_{ZZ},\mathrm{LUCJ}}^{\mathrm{native}}=2=O(1).}
$$

若把每个 $R_{ZZ}$ 都展开为 $\mathrm{CNOT}-R_Z-\mathrm{CNOT}$，则 LUCJ 的 CNOT 层深度为 $2\times2=4$。

**真实硬件上的补充**：完整 UCJ 的非邻接项在一维硬件上必须借助 SWAP 路由，实际深度会进一步膨胀；而 LUCJ 的所有作用都已经是邻接的，不需要 SWAP，所以"理论 $O(1)$"在硬件上也是 $O(1)$。

### 7. 进阶：heavy-hex 拓扑上的 LUCJ

IBM 的 heavy-hex 拓扑是一张最大度 $\Delta=3$ 的图。它的边色数为 $\chi'(G)=\Delta=3$（由 Vizing 定理，heavy-hex 是简单图且不属于奇数阶完全图那一类需要 $\Delta+1$ 色的特例），所以 LUCJ 在 heavy-hex 上原生 $R_{ZZ}$ 层深度为

$$
D_{R_{ZZ},\mathrm{LUCJ}}^{\mathrm{heavy\text{-}hex}}=3=O(1).
$$

与一维链的 2 相比，heavy-hex 因为多了一层耦合而增加到 3，但仍是常数，这是 LUCJ 在 IBM 量子硬件上具有实际吸引力的重要原因。

## 结论

1. **Jastrow 映射**：$e^{iJ_{ij}\hat n_i\hat n_j}$ 在 JW 映射下精确等价于 2 个 $R_Z$ + 1 个 $R_{ZZ}$（外加一个全局相位，可丢弃）。
2. **Givens 数守恒**：用 CRY 夹在两个同向 CNOT 之间可精确实现实 Givens；要满足题目"2 CNOT + 单比特门"口径，须改用基于 $U_{\mathrm{SE}}=\exp[-i\theta(X_1X_0+Y_1Y_0)/2]$ 的 2-CNOT 实现，加上前后 $R_Z$ 相位校正。
3. **H₂ 单层 LUCJ**：Jastrow 层 3 个 $R_{ZZ}$ 深度 7；若 $U$ 含 $g$ 个 Givens，则 CNOT 总数 $N=4g+6$；一般 4 模实旋转 $g=6$ 时 $N=30$。
4. **深度**：保守串行执行下 $D=91$；用三层匹配式 Givens 编译则 $D=49$。
5. **$n=20$ 资源对比**：项数从 UCJ 的 190（$O(n^2)$）降到 LUCJ 的 19（$O(n)$）；原生 $R_{ZZ}$ 并行深度由 UCJ 的 19（$O(n)$，需 $K_{20}$ 边着色）降到 LUCJ 的 2（$O(1)$，一维链 2-边着色）。
6. **进阶**：heavy-hex 上 LUCJ 的原生 $R_{ZZ}$ 深度为 3（仍 $O(1)$），这是 LUCJ 在 IBM 硬件上工程可行的核心原因。

## 代码说明

代码在 `code/` 目录下（若已实现）。预期实现包括：

- `jastrow_decomp.py`：根据式 (1) 把任意 $J_{ij}$ 编译为 $R_Z$/$R_{ZZ}$ 门序列；
- `givens_gate.py`：实现 2-CNOT 版本的数守恒 Givens 旋转，并验证它对 $\ket{00},\ket{01},\ket{10},\ket{11}$ 的作用矩阵；
- `h2_lucj.py`：组装 H$_2$ 的完整单层 LUCJ 线路，统计 CNOT 数与深度；
- `chain20_resources.py`：对 $n=20$ 一维链生成 LUCJ 的相邻对集合，按奇偶分两组并行，验证深度为 2。

校验要点：
1. 纯态层析得到的矩阵应与解析的 $G(\theta)$ 完全一致（容差 $10^{-10}$）；
2. 计数器统计 CNOT 应得 $N=4g+6$（$g=6$ 时为 30）；
3. 线路深度按 Qiskit `Operator` 拆分后应为 7（Jastrow）+ 7（每个 Givens）的整数倍。
