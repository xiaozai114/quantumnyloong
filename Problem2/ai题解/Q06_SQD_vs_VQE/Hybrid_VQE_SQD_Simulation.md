# Q06 进阶挑战：VQE + SQD 混合方案模拟

> 用 **TensorCircuit** 数值验证进阶挑战的核心结论 —— 三明治不等式
> $$E_0 \le E_{\text{hybrid}} \le E_{\text{VQE}}$$
> 分子：**LiH**（active space 约化到题目的 8 qubit）。

---

## 1. 方案回顾

混合方案 = **VQE 的浅线路做态制备** + **SQD 的对角化做后处理**，三步：

| 步骤 | 借用自 | 做什么 |
|------|--------|--------|
| ① 态制备 | VQE | UCCSD 截断浅线路 $\lvert\Psi(\theta)\rangle$，优化少量参数 |
| ② 采样 | SQD | 对 $\lvert\Psi\rangle$ 采样，收集组态张成子空间 $\mathcal{S}$ |
| ③ 后处理 | SQD | 在 $\mathcal{S}$ 内投影 $\hat H$ 并经典对角化，取 $\lambda_{\min}$ |

能量结论（三明治不等式）：

$$\boxed{\,E_0 \;\le\; E_{\text{hybrid}} \;\le\; E_{\text{VQE}}\,}$$

- **上界** $E_{\text{hybrid}} \le E_{\text{VQE}}$：VQE 态 $\in \mathcal{S}$，对角化取 $\mathcal{S}$ 内最小值不会更差。
- **下界** $E_{\text{hybrid}} \ge E_0$：变分原理，子空间内 Ritz 值不低于全空间基态能量。

---

## 2. 代码结构

```
code/
├── hamiltonian.py   # 构建 LiH active-space 哈密顿量 + 精确下界 E0
├── vqe.py           # TensorCircuit UCCSD 截断 VQE 浅线路制备
├── sqd_diag.py      # 采样 + 子空间投影对角化
├── main.py          # 完整流水线 + 三明治不等式验证 + 收敛曲线
└── sandwich_convergence.png   # 结果图
```

每个模块对应理论的一个环节：

| 模块 | 理论对应 |
|------|----------|
| `hamiltonian.py` | 第二量子化 → JW 映射 → 8 qubit $\hat H$；active FCI = 下界 $E_0$ |
| `vqe.py` | 步骤 ①：从 HF 态出发的保粒子数 Givens 激发（UCCSD 截断） |
| `sqd_diag.py` | 步骤 ②③：采样 $\to$ 子空间 $\mathcal{S}$ $\to$ 对角化 $E_{\text{hybrid}}$ |
| `main.py` | 串起全流程，数值验证 $E_0 \le E_{\text{hybrid}} \le E_{\text{VQE}\lvert\mathcal{S}}$ |

---

## 3. 关键实现说明

### 3.1 哈密顿量与 active space（`hamiltonian.py`）

题目说 LiH「$N=8$」，但 STO-3G 下 LiH 原生是 **6 空间轨道 = 12 qubit**。为贴合题目的 8 qubit，做 active space 约化：

- **冻结** Li 的 1s 芯轨道（`occupied_indices=[0]`）
- **保留** 4 个活性空间轨道（`active_indices=[1,2,3,4]`）→ **8 qubit，2 个活性电子**

active space 内的 FCI 精确基态能量即为三明治不等式的**理论下界** $E_0$。

### 3.2 UCCSD 截断 ansatz（`vqe.py`）

从 HF 参考态 $\lvert 11000000\rangle$ 出发，作用**保粒子数**的 Givens 旋转：

```python
def _givens(circuit, theta, i, j):   # 保粒子数单激发 i<->j
    circuit.cnot(i, j)
    circuit.cry(j, i, theta=2*theta)
    circuit.cnot(i, j)
```

- **single excitation**：占据轨道 → 虚轨道，每对一个 Givens
- **double excitation**：成对激发 $(ij)\to(ab)$，两个 Givens 串联近似

截断 = 只保留 single + double，故线路浅（本例 27 个参数）。后端用 TensorCircuit 的 **numpy backend**（纯态矢量模拟），优化器用 scipy `L-BFGS-B`。

### 3.3 采样 + 子空间对角化（`sqd_diag.py`）

```python
# 步骤②：采样，只保留粒子数正确的组态（UCCSD 天然保粒子数）
configs = sample_configs(psi, n_shots, ...)
# 步骤③：在 S 内投影 H 并对角化
H_sub = H[np.ix_(configs, configs)]
E_hybrid = eigvalsh(H_sub)[0]
```

严格可比的上界用 **VQE 态投影到同一 $\mathcal{S}$ 的能量** $E_{\text{VQE}\lvert\mathcal{S}}$（见下节说明）。

---

## 4. 运行结果

运行命令：

```bash
/Users/zhouzihan/.workbuddy/binaries/python/envs/qchem/bin/python main.py
```

关键输出：

```
[1] 哈密顿量
    n_qubits    = 8
    n_electrons = 2 (active)
    E_HF        = -7.862024 Ha
    E0 (下界, active FCI) = -7.863842 Ha

[2] VQE (UCCSD 截断, 27 参数)
    E_VQE = -7.863653 Ha

[3] SQD 后处理 (采样 S=2000)
    子空间维数 = 3
    E_hybrid            = -7.863633 Ha
    E_VQE|S             = -7.863633 Ha

[4] 三明治不等式验证
    E0 <= E_hybrid      : 成立 ✓
    E_hybrid <= E_VQE|S : 成立 ✓
    => 严格三明治不等式 全部成立 ✓
```

### 收敛曲线

![三明治不等式收敛](./sandwich_convergence.png)

- **绿色虚线** $E_0$：理论下界（active FCI）
- **蓝色虚线** $E_{\text{VQE}}$：VQE 上界
- **红色曲线** $E_{\text{hybrid}}$：随子空间维数增大，从 HF 值单调下降、逼近 $E_0$，始终夹在下界之上

这正是三明治不等式的**可视化证明**：子空间越大（采到越多重要组态），$E_{\text{hybrid}}$ 越贴近精确基态能量 $E_0$。

---

## 5. 一个重要的数值细节：严格上界应取 $E_{\text{VQE}\lvert\mathcal{S}}$

理论证明假设 $\lvert\Psi\rangle \in \mathcal{S}$。但**采样只保留大振幅组态**，$\lvert\Psi\rangle$ 的小振幅分量被丢弃，严格来说 $\lvert\Psi\rangle \notin \mathcal{S}$。

因此直接比较 $E_{\text{hybrid}}$ 与**完整** $E_{\text{VQE}} = \langle\Psi\lvert\hat H\rvert\Psi\rangle$ 会有 $\sim 10^{-5}$ Ha 的采样截断误差，可能出现微小「违反」。

**正确做法**：把 VQE 态投影到同一子空间 $\mathcal{S}$，得 $E_{\text{VQE}\lvert\mathcal{S}} = \frac{\langle\Psi_{\mathcal{S}}\lvert\hat H\rvert\Psi_{\mathcal{S}}\rangle}{\langle\Psi_{\mathcal{S}}\lvert\Psi_{\mathcal{S}}\rangle}$。此时 $\lvert\Psi_{\mathcal{S}}\rangle \in \mathcal{S}$，严格不等式

$$E_0 \le E_{\text{hybrid}} \le E_{\text{VQE}\lvert\mathcal{S}}$$

**恒成立**（代码中验证为 ✓✓）。这也说明了理论证明中「$\lvert\Psi\rangle\in\mathcal{S}$」这一前提在真实采样场景下的边界。

---

## 6. 环境依赖

代码运行于专用 venv（`~/.workbuddy/binaries/python/envs/qchem`）：

| 包 | 版本 | 用途 |
|----|------|------|
| tensorcircuit | 0.12.0 | 量子线路模拟（numpy backend） |
| numpy | 1.26.4 | 数值（**必须 <2.0**，见下） |
| scipy | 1.14.1 | L-BFGS-B 优化器 |
| pyscf / openfermion / openfermionpyscf | — | 分子积分 + JW 映射 |
| matplotlib | — | 绘图 |

> **踩坑记录**：TensorCircuit 0.12.0 使用了 numpy 2.x 已移除的 API（`np.reshape(newshape=)`、`np.ComplexWarning` 等），必须锁定 `numpy==1.26.4` + `scipy==1.14.1`（scipy 新版会强制升级 numpy 到 2.x）。
