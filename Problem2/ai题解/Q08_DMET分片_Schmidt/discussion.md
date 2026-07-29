# Q08：DMET 分片——Schmidt 分解与浴的精确性

## 题目

分子 (c) H₂O，DMET 按原子分片（O、H、H 三个碎片）。

- **(a)** 对任一双粒子态做 Schmidt 分解，证明秩 $r \leq \min(\dim\mathcal{H}_{\text{imp}}, \dim\mathcal{H}_{\text{env}})$；结合 HF 密度矩阵的幂等性 $\gamma^2=\gamma$，推导浴大小上界 $n_{\text{bath}} \leq n_{\text{imp}}$。
- **(b)** 构造 DMET 最小浴（碎片自然轨道 + 浴轨道配对），证明它能精确复现全分子 HF 密度矩阵的碎片块 $\gamma_{\text{II}}^{\text{embed}} = \gamma_{\text{II}}^{\text{full}}$。关键工具：投影算符性质 $\gamma^2=\gamma$。
- **(c)** 当求解器升级为 CCSD/SQD 时，$\gamma^{\text{corr}} \neq (\gamma^{\text{corr}})^2$。为什么最小浴不再精确？MP2 浴补充了什么关联？
- **进阶**：DMET 自洽化学势 $\mu$ 如何确定？one-shot vs 自洽 DMET 的权衡？

## 我的想法

这道题的核心张力是 **"幂等性是 DMET 最小浴的数学基石"**，而关联态恰好打破幂等性。

- **(a)** 有两条独立的上界链：
  - Schmidt 链：任何双粒子态展开系数矩阵 $C$ 的 SVD 给出 $r = \text{rank}(C) \leq \min(d_{\text{imp}}, d_{\text{env}})$，这是"几何上"的上界。
  - 幂等链：HF 的 $\gamma$ 是投影算符，把 $\gamma^2=\gamma$ 按碎片-环境分块，左下块给出 $(I-\gamma_{\text{EE}})\gamma_{\text{EI}} = \gamma_{\text{EI}}\gamma_{\text{II}}$，两边取秩得到 $\text{rank}(\gamma_{\text{EI}}) \leq \text{rank}(\gamma_{\text{II}}) \leq n_{\text{imp}}$。这是"代数上"的上界，比 Schmidt 更紧。
  - 浴轨道数 $n_{\text{bath}} = \text{rank}(\gamma_{\text{EI}})$，所以 $n_{\text{bath}} \leq n_{\text{imp}}$。
- **(b)** 把幂等性看作"碎片与环境的纠缠被压缩到 $n_{\text{imp}}$ 个独立 2×2 块"：
  - 对角化 $\gamma_{\text{II}}$ 得碎片自然轨道 $v_k$ 与占据 $\lambda_k$。
  - 浴轨道 $w_k = (I-P_{\text{imp}})\gamma\tilde{v}_k$ 的几何含义是"环境对碎片自然轨道 $v_k$ 的响应"。
  - 用 $\gamma^2=\gamma$ 把 $|w_k|^2$ 算出来：恰好是 $\lambda_k(1-\lambda_k)$，于是 $c_k=\sqrt{\lambda_k(1-\lambda_k)}$、$\mu_k=1-\lambda_k$。
  - 嵌入 $\gamma$ 退化成 $n_{\text{imp}}$ 个独立的 $2\times2$ 块 $\bigoplus_k \begin{pmatrix}\lambda_k & c_k \\ c_k & 1-\lambda_k\end{pmatrix}$，每块幂等（直接验证 $\lambda^2+c^2=\lambda$），整体幂等 → Slater 行列式。
  - 嵌入空间 $\gamma$-不变 + $[F,\gamma]=0$ 保持 → 嵌入 HF 方程自洽 → 碎片块完全复现。
- **(c)** 关联态的特征是 1-RDM 的特征值（自然轨道占据数）取 $0<n_p<1$ 的连续值，而幂等要求 $n_p\in\{0,1\}$，所以 $\gamma^{\text{corr}}$ 必然非幂等。**(b) 的所有证明都穿过 $\gamma^2=\gamma$ 这根针眼**，幂等失效则：秩不等式失效（$n_{\text{bath}}$ 可能 $>n_{\text{imp}}$）、2×2 块结构失效（$\mu_k\neq 1-\lambda_k$、$c_k\neq\sqrt{\lambda_k(1-\lambda_k)}$）、配对结构失效（一个碎片轨道可耦合多个环境方向）。HF 浴只编码了静态纠缠，缺失的动力学关联由 MP2 浴从 $\delta\gamma^{\text{MP2}}$ 中提取补上。
- **进阶** $\mu$ 是约束碎片电子数的拉格朗日乘子：$N_{\text{imp}}(\mu) = \text{Tr}_{\text{imp}}[\gamma^{\text{corr}}(\mu)]$。$N_{\text{imp}}$ 随 $\mu$ 单调递减，用牛顿法搜索。one-shot 快但电子数不匹配、精度受限；自洽多轮迭代 $\mu$（必要时也迭代浴），精度高但慢 3–5 倍。

## 讨论

### (a) Schmidt 分解上界 + 幂等性给 $n_{\text{bath}} \leq n_{\text{imp}}$

#### Schmidt 分解的秩上界

任意双粒子纯态 $|\Psi\rangle \in \mathcal{H}_{\text{imp}}\otimes\mathcal{H}_{\text{env}}$，在碎片基 $\{|a_i\rangle\}_{i=1}^{d_{\text{imp}}}$ 与环境基 $\{|b_j\rangle\}_{j=1}^{d_{\text{env}}}$ 下展开：

$$|\Psi\rangle = \sum_{i=1}^{d_{\text{imp}}}\sum_{j=1}^{d_{\text{env}}} C_{ij}\,|a_i\rangle\otimes|b_j\rangle$$

系数矩阵 $C \in \mathbb{C}^{d_{\text{imp}}\times d_{\text{env}}}$。对其做 SVD：$C = U\Sigma V^\dagger$，把 $U$、$V$ 的列分别吸收进新的碎片/环境基 $|u_k\rangle = \sum_i U_{ik}|a_i\rangle$、$|v_k\rangle = \sum_j V_{jk}|b_j\rangle$，得到 Schmidt 形式：

$$|\Psi\rangle = \sum_{k=1}^{r} \lambda_k\,|u_k\rangle\otimes|v_k\rangle, \qquad r = \text{rank}(C),\;\lambda_k\geq 0$$

由于矩阵秩不超过任一维数：

$$\boxed{r = \text{rank}(C) \leq \min(d_{\text{imp}},\,d_{\text{env}}) = \min(\dim\mathcal{H}_{\text{imp}},\,\dim\mathcal{H}_{\text{env}})}$$

这是纯线性代数结论，对任何纯态都成立。

#### HF 密度矩阵的幂等性给更紧的上界

HF 基态是单个 Slater 行列式，其 1-RDM $\gamma$ 是占据空间上的正交投影算符，特征值为 0 或 1，满足：

$$\gamma^2 = \gamma$$

把 $\gamma$ 按碎片 (I) 与环境 (E) 分块：

$$\gamma = \begin{pmatrix}\gamma_{\text{II}} & \gamma_{\text{IE}} \\ \gamma_{\text{EI}} & \gamma_{\text{EE}}\end{pmatrix}$$

幂等条件 $\gamma^2=\gamma$ 的**左下块**给出：

$$\gamma_{\text{EI}}\gamma_{\text{II}} + \gamma_{\text{EE}}\gamma_{\text{EI}} = \gamma_{\text{EI}}$$

把右端移到左边：

$$(I - \gamma_{\text{EE}})\,\gamma_{\text{EI}} = \gamma_{\text{EI}}\,\gamma_{\text{II}} \tag{1}$$

两边取秩。右端用秩的次可乘性：

$$\text{rank}(\gamma_{\text{EI}}\gamma_{\text{II}}) \leq \min(\text{rank}\,\gamma_{\text{EI}},\,\text{rank}\,\gamma_{\text{II}}) \leq \text{rank}\,\gamma_{\text{II}} \leq n_{\text{imp}}$$

左端：矩阵左乘任何矩阵秩只减不增，$\text{rank}((I-\gamma_{\text{EE}})\gamma_{\text{EI}}) \leq \text{rank}(\gamma_{\text{EI}})$。结合 (1) 两端秩相等：

$$\text{rank}(\gamma_{\text{EI}}) \geq \text{rank}(\text{左端}) = \text{rank}(\text{右端}) \leq \text{rank}(\gamma_{\text{II}}) \leq n_{\text{imp}}$$

注意这里**没有断言** $I-\gamma_{\text{EE}}$ 可逆——即使不可逆，左乘只减秩这一条已经足够。结合等式 (1) 两端秩相等，得到：

$$\text{rank}(\gamma_{\text{EI}}) = \text{rank}(\gamma_{\text{EI}}\gamma_{\text{II}}) \leq \text{rank}(\gamma_{\text{II}}) \leq n_{\text{imp}}$$

#### 浴轨道数 = 耦合矩阵秩

碎片-环境的独立耦合方向数 = $\gamma_{\text{EI}}$ 的秩；每个独立耦合方向配一个浴轨道，因此：

$$\boxed{\,n_{\text{bath}} = \text{rank}(\gamma_{\text{EI}}) \leq n_{\text{imp}}\,}$$

对闭壳层 HF，几乎总是取等号 $n_{\text{bath}} = n_{\text{imp}}$（每个碎片轨道对应一个浴轨道，构成 $n_{\text{imp}}$ 对）。

**与 Schmidt 上界的关系**：Schmidt 给出 $r\leq\min(d_{\text{imp}},d_{\text{env}})$，对 1-RDM 而言 $d_{\text{imp}}$ 就是 $n_{\text{imp}}$，所以 Schmidt 上界是 $n_{\text{bath}}\leq n_{\text{imp}}$。幂等性给的也是 $n_{\text{bath}}\leq n_{\text{imp}}$，看起来数值一样，但来源不同：Schmidt 是几何的、对任意态成立；幂等是代数的、只对 HF 这种 Slater 行列式成立。下一问 (b) 会看到，正是幂等性额外保证了"浴属性有解析公式 + 2×2 块幂等"——这些 Schmidt 给不了。

### (b) 最小浴构造与 HF 精确复现

#### 第一步：对角化碎片块得到自然轨道

$\gamma_{\text{II}}$ 是 Hermitian 半正定，可对角化：

$$\gamma_{\text{II}}\,v_k = \lambda_k\,v_k,\qquad k=1,\ldots,n_{\text{imp}},\qquad 0\leq\lambda_k\leq 1$$

$\{v_k\}$ 是碎片自然轨道，$\lambda_k$ 是其占据数。

#### 第二步：构造浴轨道

把 $v_k$ 升维到全空间（碎片部分填 $v_k$，环境部分填零），记为 $\tilde{v}_k$。构造"环境对 $v_k$ 的响应向量"：

$$w_k = (I - P_{\text{imp}})\,\gamma\,\tilde{v}_k$$

其中 $P_{\text{imp}}$ 是碎片子空间的投影。$w_k$ 完全落在环境子空间里，几何上就是"全分子 $\gamma$ 作用在 $v_k$ 上之后，扣除碎片部分剩下来的环境部分"。

归一化得浴轨道 $b_k = w_k / |w_k|$。关键是 $|w_k|$ 有闭式表达——这正是幂等性的妙用：

$$|w_k|^2 = \tilde{v}_k^\dagger\,\gamma\,(I - P_{\text{imp}})\,\gamma\,\tilde{v}_k = \tilde{v}_k^\dagger\,\gamma^2\,\tilde{v}_k - \tilde{v}_k^\dagger\,\gamma\,P_{\text{imp}}\,\gamma\,\tilde{v}_k$$

用 $\gamma^2=\gamma$ 把第一项换成 $\tilde{v}_k^\dagger\gamma\tilde{v}_k$；用 $\gamma_{\text{II}}v_k=\lambda_k v_k$ 把第一项算成 $\lambda_k$、第二项算成 $|\gamma_{\text{II}}v_k|^2 = \lambda_k^2$：

$$|w_k|^2 = \lambda_k - \lambda_k^2 = \lambda_k(1-\lambda_k)$$

于是：

- **耦合强度**：$c_k = |w_k| = \sqrt{\lambda_k(1-\lambda_k)}$
- **浴占据数**：$\mu_k = 1 - \lambda_k$（由 2×2 块幂等条件推出，下面验证）

#### 第三步：嵌入 $\gamma$ 退化成 $n_{\text{imp}}$ 个独立 2×2 块

在基 $\{v_1,\ldots,v_{n_{\text{imp}}}, b_1,\ldots,b_{n_{\text{imp}}}\}$ 下，全分子 $\gamma$ 限制到嵌入空间的形式是：

$$\gamma\big|_{\text{embed}} = \bigoplus_{k=1}^{n_{\text{imp}}} \begin{pmatrix}\lambda_k & c_k \\ c_k & \mu_k\end{pmatrix} = \bigoplus_{k=1}^{n_{\text{imp}}} \begin{pmatrix}\lambda_k & \sqrt{\lambda_k(1-\lambda_k)} \\ \sqrt{\lambda_k(1-\lambda_k)} & 1-\lambda_k\end{pmatrix}$$

"独立"的根据：
- 碎片自然轨道之间无耦合——$\gamma_{\text{II}}$ 已对角化。
- 浴轨道之间无耦合——浴由 $\gamma_{\text{EI}}$ 的列空间构造，不同 $k$ 对应不同列方向，在 $\gamma^2=\gamma$ 下彼此正交。

#### 第四步：2×2 块幂等 → 整体幂等 → Slater 行列式

直接验证每个块的平方：

$$\begin{pmatrix}\lambda & c \\ c & 1-\lambda\end{pmatrix}^2 = \begin{pmatrix}\lambda^2 + c^2 & c(\lambda + 1 - \lambda) \\ c(\lambda + 1 - \lambda) & c^2 + (1-\lambda)^2\end{pmatrix}$$

代入 $c^2 = \lambda(1-\lambda)$：

- 左上：$\lambda^2 + \lambda(1-\lambda) = \lambda^2 + \lambda - \lambda^2 = \lambda$ ✓
- 对角：$c^2 + (1-\lambda)^2 = \lambda(1-\lambda) + (1-\lambda)^2 = (1-\lambda)(\lambda + 1 - \lambda) = 1-\lambda$ ✓
- 非对角：$c(\lambda + 1 - \lambda) = c$ ✓

每个块幂等，整体 $\gamma\big|_{\text{embed}}^2 = \gamma\big|_{\text{embed}}$，故对应一个 Slater 行列式——嵌入空间的 HF 态。

#### 第五步：嵌入空间 $\gamma$-不变 + Fock 对易保持

**$\gamma$-不变**：$\gamma$ 作用在 $v_k$ 上得到 $\lambda_k v_k + c_k b_k$（碎片 ∪ 浴，不漏出嵌入空间）；作用在 $b_k$ 上得到 $c_k v_k + \mu_k b_k$（用 (1) 式 $(I-\gamma_{\text{EE}})\gamma_{\text{EI}}=\gamma_{\text{EI}}\gamma_{\text{II}}$ 推出，同样不漏出）。所以 $\gamma$ 把嵌入空间映射到自身。

**Fock 对易保持**：全分子 HF 满足 $[F,\gamma]=0$。因为嵌入空间 $\gamma$-不变，$F$ 与 $\gamma$ 都能干净地限制到嵌入空间，对易关系保持：

$$[F\big|_{\text{embed}},\,\gamma\big|_{\text{embed}}] = 0$$

这就是嵌入空间的 HF 方程。所以 $\gamma\big|_{\text{embed}}$ 是 $H_{\text{embed}}$ 的 HF 密度矩阵。

#### 第六步：碎片块完全复现

每个 2×2 块的**左上角**就是 $\lambda_k$，所以：

$$\gamma_{\text{II}}^{\text{embed}} = \text{diag}(\lambda_1,\ldots,\lambda_{n_{\text{imp}}})$$

而 $\gamma_{\text{II}}^{\text{full}}$ 在自然轨道基下也是 $\text{diag}(\lambda_1,\ldots,\lambda_{n_{\text{imp}}})$（第一步对角化的定义），所以：

$$\boxed{\,\gamma_{\text{II}}^{\text{embed}} = \gamma_{\text{II}}^{\text{full}}\,}$$

**这是 DMET 的"无损性"证明**：用 $n_{\text{imp}}$ 个浴轨道，HF 信息在嵌入层一分不差地保留下来。注意整个证明的支柱是 $\gamma^2=\gamma$——这恰恰是下一问要被打翻的假设。

### (c) 关联态打破幂等性，最小浴失效

#### 为什么 $\gamma^{\text{corr}} \neq (\gamma^{\text{corr}})^2$

HF 态是单个 Slater 行列式，1-RDM 是占据空间上的正交投影，特征值（自然轨道占据数）取 $n_p\in\{0,1\}$，必然满足 $\gamma^2=\gamma$（因为 $0^2=0$、$1^2=1$）。

CCSD/SQD 给出的是**关联态**，不再是单个 Slater 行列式。关联态的特征是 1-RDM 的特征值取 $0<n_p<1$ 的连续值，例如 $n_p = 0.83, 0.12, 0.97$ 等。

验证：若 $\gamma$ 的特征值为 $n_p$，则 $\gamma^2$ 的特征值为 $n_p^2$。$\gamma^2=\gamma$ 要求 $n_p^2 = n_p$，即 $n_p\in\{0,1\}$。关联态 $n_p\notin\{0,1\}$，故：

$$\boxed{\,(\gamma^{\text{corr}})^2 \neq \gamma^{\text{corr}}\,}$$

#### 为什么最小浴不再精确——三处失效

(b) 的整套证明**全部穿过 $\gamma^2=\gamma$ 这根针眼**。一旦幂等失效，三处会同时崩塌：

1. **浴数上界失效**。$n_{\text{bath}}\leq n_{\text{imp}}$ 来自 (1) 式的秩不等式 $(I-\gamma_{\text{EE}})\gamma_{\text{EI}} = \gamma_{\text{EI}}\gamma_{\text{II}}$，这个等式本身是 $\gamma^2=\gamma$ 的左下块。关联态下等式不成立，秩约束解开，Schmidt 秩可能超过 $n_{\text{imp}}$——纠缠结构比 HF 复杂，$n_{\text{imp}}$ 个浴容不下。

2. **浴属性公式失效**。$c_k=\sqrt{\lambda_k(1-\lambda_k)}$、$\mu_k = 1-\lambda_k$ 都是用 $\gamma^2=\gamma$ 在 2×2 块内推出来的。关联态下 2×2 块不再幂等，这些闭式不再成立。

3. **配对结构失效**。"每个碎片自然轨道恰好配一个浴轨道"依赖 $\gamma^2=\gamma$ 退化出的块对角结构。关联态下一个碎片自然轨道可能与多个环境方向耦合，纠缠谱更复杂。

**结论**：用 HF 构造的 $n_{\text{imp}}$ 个浴轨道，对关联态来说**不够**——丢失了部分碎片-环境纠缠信息，碎片电子数和密度矩阵都会有系统偏差。

#### MP2 浴补充了什么关联

HF 浴编码的是**平均场级别**的碎片-环境纠缠，即**静态关联**（占主导的、可由单 Slater 行列式捕获的部分）。缺失的是**动力学关联**——电子之间的瞬时涨落耦合。

MP2（二阶微扰理论）给出关联态 1-RDM 的一阶修正：

$$\gamma^{\text{corr}} \approx \gamma^{\text{HF}} + \delta\gamma^{\text{MP2}}$$

MP2 浴自然轨道（BNO）从 $\delta\gamma^{\text{MP2}}$ 的环境部分提取。BNO 占据数大致呈指数衰减——前几个 BNO 携带大部分动力学关联信息，截断阈值 $\eta$ 控制数量。

**MP2 浴补充的内容**：

- HF 浴只捕获了 $\gamma^{\text{HF}}$ 的碎片-环境耦合（静态部分）。
- MP2 浴额外捕获 $\delta\gamma^{\text{MP2}}$ 的碎片-环境耦合（动力学部分）。
- 截断阈值 $\eta$ 越小，浴越大，精度越高但代价越大。

实际 DMET/QMIT 工作流通常是：HF 浴（强制 $n_{\text{imp}}$ 个，保静态）+ MP2 浴（按 $\eta$ 截断，补动力学），合计 $n_{\text{imp}} + n_{\text{BNO}}$ 个浴轨道。

### 进阶：化学势 $\mu$ 的牛顿搜索 + one-shot vs 自洽

#### $\mu$ 的角色：约束碎片电子数的拉格朗日乘子

DMET 把全分子切成 $F$ 个碎片独立求解，每个碎片的电子数不一定等于化学上的"应当属于该碎片的电子数"。引入化学势 $\mu$ 作拉格朗日乘子，约束：

$$N_{\text{imp}}(\mu) = \text{Tr}_{\text{imp}}\!\big[\gamma^{\text{corr}}(\mu)\big] = N_{\text{imp}}^{\text{target}}$$

构造嵌入哈密顿量：

$$H_{\text{imp}}(\mu) = H_{\text{embed}} + \mu\,\hat{N}_{\text{imp}}$$

$\hat{N}_{\text{imp}}$ 在计算基下是对角矩阵，$\mu$ 只改变 CI 矩阵的对角元，不影响非对角耦合，代价极低。

#### 单调性 + 牛顿搜索

物理上，$\mu$ 越大 → 占据碎片轨道越不划算 → $N_{\text{imp}}$ 越小，故 $N_{\text{imp}}(\mu)$ 随 $\mu$ **单调递减**。

牛顿迭代：

$$\mu^{(t+1)} = \mu^{(t)} - \frac{N_{\text{imp}}(\mu^{(t)}) - N_{\text{imp}}^{\text{target}}}{\partial N_{\text{imp}}/\partial\mu\big|_{\mu^{(t)}}}$$

分母 $\partial N_{\text{imp}}/\partial\mu = -\text{Var}_{\text{imp}}(\hat{N})$ 是碎片占据数的方差（负的），可由求解器一次性给出。单调性保证牛顿法全局收敛、通常 3–5 轮即达 $10^{-6}$ 精度。

#### One-shot DMET vs 自洽 DMET

| 维度 | One-shot DMET | 自洽 DMET |
|:---|:---|:---|
| 浴构造 | 用 $\gamma^{\text{HF}}$ 一次性构造 | 可选：每轮用 $\gamma^{\text{corr}}$ 重做浴 |
| $\mu$ 调整 | $\mu=0$ 不迭代 | 牛顿法迭代 3–5 轮 |
| 求解器调用 | $F$ 次（每碎片 1 次） | $\sim 4F$ 次（每碎片 3–5 次） |
| 碎片电子数 | 不保证匹配 | 严格匹配 $N_{\text{imp}}^{\text{target}}$ |
| 精度 | 较低（HF 浴对关联态有偏） | 较高（$\mu$ 自洽抵消偏置） |
| 总开销 | 1× | 3–5× |
| 典型场景 | 大体系扫描、初筛 | 高精度能量/性质 |

**权衡要点**：

- one-shot 把所有"误差"压到浴构造上，但跳过 $\mu$ 迭代省了 3–5 倍时间。适合大体系扫参、或者后续要做高阶修正只需要一个合理初值。
- 自洽 DMET 让 $\mu$ 收敛到电子数匹配，是对 one-shot 的"修补"——精度上限是"给定浴下的最好结果"。如果还想进一步提升，要做**浴自洽**（用 $\gamma^{\text{corr}}$ 重做浴），代价再翻数倍。
- 实践中常采用折中：one-shot 构浴 + $\mu$ 自洽（不重做浴），即"半自洽"，3–5 倍开销换大部分精度提升。

## 结论

1. **Schmidt 上界 + 幂等上界**：纯几何的 Schmidt 分解给 $r\leq\min(d_{\text{imp}},d_{\text{env}})$；HF 的 $\gamma^2=\gamma$ 给更具体的代数上界 $n_{\text{bath}} = \text{rank}(\gamma_{\text{EI}})\leq\text{rank}(\gamma_{\text{II}})\leq n_{\text{imp}}$，闭壳层通常取等号。

2. **HF 浴精确复现**：用碎片自然轨道 + 浴轨道 $w_k=(I-P_{\text{imp}})\gamma\tilde{v}_k$ 配对，借助 $\gamma^2=\gamma$ 算出 $c_k=\sqrt{\lambda_k(1-\lambda_k)}$、$\mu_k=1-\lambda_k$；嵌入 $\gamma$ 退化为 $n_{\text{imp}}$ 个独立 2×2 块、整体幂等、$\gamma$-不变、Fock 对易保持，故 $\gamma_{\text{II}}^{\text{embed}}=\gamma_{\text{II}}^{\text{full}}$。这是 DMET 的"无损性"证明。

3. **关联态打破幂等 → 最小浴失效**：关联态 1-RDM 特征值取 $0<n_p<1$，必然 $({\gamma^{\text{corr}}})^2\neq\gamma^{\text{corr}}$。三处同时崩塌——秩上界、浴属性公式、配对结构。HF 浴只编码静态纠缠，丢失动力学关联。

4. **MP2 浴补动力学关联**：从 $\delta\gamma^{\text{MP2}}$ 提取 BNO，前几个 BNO 携带大部分动力学信息，按 $\eta$ 截断。HF 浴（静态）+ MP2 浴（动力学）是实际工作流的标准组合。

5. **$\mu$ 牛顿搜索 + one-shot vs 自洽**：$\mu$ 是约束 $N_{\text{imp}}=N_{\text{imp}}^{\text{target}}$ 的拉格朗日乘子，只改 CI 对角元；$N_{\text{imp}}(\mu)$ 单调递减，牛顿法 3–5 轮收敛。one-shot 快但电子数不匹配、精度受限；自洽多花 3–5 倍时间换严格匹配和更高精度。

## 代码说明

代码在 `code/` 目录下。本题为理论证明为主，代码用于**数值验证**关键结论：

1. **(a) 秩上界验证**：构造随机 HF 密度矩阵 $\gamma = XX^\dagger$（$X$ 列正交），分块后计算 $\text{rank}(\gamma_{\text{EI}})$，验证 $\leq n_{\text{imp}}$。
2. **(b) 浴属性公式验证**：对 H₂O/STO-3G 跑 RHF，取 O 原子为碎片，对角化 $\gamma_{\text{II}}$ 得 $\lambda_k$；构造 $w_k=(I-P_{\text{imp}})\gamma\tilde{v}_k$，数值计算 $|w_k|$，与 $\sqrt{\lambda_k(1-\lambda_k)}$ 对比；构造 2×2 块验证幂等 $\gamma_{\text{embed}}^2 = \gamma_{\text{embed}}$；验证 $\gamma_{\text{II}}^{\text{embed}} = \gamma_{\text{II}}^{\text{full}}$。
3. **(c) 关联态失效演示**：在相同体系跑 CCSD，计算 $\|\gamma_{\text{corr}}^2 - \gamma_{\text{corr}}\|_F$ 应非零；比较 HF 浴下 CCSD 的 $\gamma_{\text{II}}^{\text{embed}}$ 与全分子 CCSD 的 $\gamma_{\text{II}}^{\text{full}}$，观察偏差；加入 MP2 浴（BNO 按 $\eta=10^{-3}$ 截断），观察偏差收敛。
4. **进阶：$\mu$ 牛顿迭代**：实现 $N_{\text{imp}}(\mu)$ 单调下降曲线绘制 + 牛顿法收敛轨迹（3–5 轮到 $10^{-6}$），对比 one-shot（$\mu=0$）和自洽的碎片电子数匹配情况。
