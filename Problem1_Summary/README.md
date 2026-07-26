# Problem 1 汇总（Tencent Sparking Program 2026 — Quantum Computing）

本文件夹汇总了题目 **Problem 1: Preliminaries and Warm-up** 的**全部解答内容**（第 1–8 小问），
以便一处查阅。所有内容均从原仓库文件**读取并复制/整理**而来，**未修改或删除任何原有文件**。

> 汇总文档以原 `Problem1/Problem1-Solution.tex` 的题面与排版格式为骨架，
> 补齐了原文件中缺失的第 4、5、7、8 题解答（来源见下表）。

---

## 目录结构

```
Problem1_Summary/
├── Problem1_Complete_Solution.tex   # 一份完整 tex（第 1–8 题全部解答）
├── Problem1_Complete_Solution.pdf   # 需本地编译生成（见下方“编译说明”）
├── README.md                        # 本文件
├── figs/                            # tex 引用的图片（测量线路图 + 第 7 题电路图）
│   ├── q6_a1_zz_measure.png
│   ├── q6_a2_xx_measure.png
│   ├── q6_a3_bell_measure.png
│   ├── q6_b_pauli_measure.png
│   ├── q6_c_ghz_to_single.png
│   ├── q6_d_fanout_f4.png
│   ├── q6_e_swap.png
│   ├── q6_e_swap_decomposed.png
│   ├── q6_circuits.txt              # 上述线路的文本版
│   └── problem7_circuit.pdf         # 第 7 题可逆电路图
└── code/                            # 若干 py 文件（数值/线路验证）
    ├── problem1_circuits.py         # 第 6 题：测量线路绘制（qiskit）
    ├── problem1_measurement.py      # 第 6 题：测量数值验证（numpy）
    ├── problem7_reversible.py       # 第 7 题：S3 可逆电路（TensorCircuit）
    ├── 8_grover.py                  # 第 8 题：Grover 搜索（TensorCircuit）
    └── quantum_circuits.py          # 第 7、8 题共用库（build_s3）
```

---

## 小问 → 解答来源对照

Problem 1 各小问及其在原仓库中的解答/代码来源如下（本汇总据此整理）：

| 小问 | 内容 | 解答/代码来源（原仓库） |
|------|------|------------------------|
| 1 | 证明 $R_n(\theta)$ 的展开形式 | `Problem1/Problem1-Solution.tex` |
| 2 | 任意单比特酉 $U=e^{i\alpha}R_n(\theta)$ | `Problem1/Problem1-Solution.tex` |
| 3 | 用 $R_Y,R_Z$ 实现 $R_n(\theta)$ | `Problem1/Problem1-Solution.tex` |
| 4 (a–d) | 态制备 State Preparation | `Problem1/problem1(1-4done).tex` |
| 4(d) 详解 | CSS 编码最优线路 + 门数下界证明 | `problem1_4d_answer.tex` / `problem1_4d_answer.pdf` |
| 5 (a–f) | 门构造 Gate Construction | `Problem1/Problem1(1-5done).tex` |
| 6 (a–e) | 测量 Measurement | `Problem1/Problem1-Solution.tex`；绘图 `problem1_circuits.py`；数值 `problem1_measurement.py`；图 `Problem1/figs/q6_*.png` |
| 7 | 可逆电路 $S_3$（选做） | `Problem1/code/problem7_reversible.py`、`lib/quantum_circuits.py`、`Problem1/code/problem7_circuit.pdf` |
| 8 | Grover 搜索（选做） | `Problem1/code/8_grover.py`、`lib/quantum_circuits.py` |

说明：
- 第 6 题的两个脚本文件头均标注“问题 1.6：测量”，因此 `problem1_circuits.py`（尽管文件名为 circuits）实际对应**测量**小问的线路绘制。
- 第 4(c) 题目在原 `Problem1-Solution.tex` 中的系数存在笔误（非归一化）。本汇总采用与解答一致的归一化写法
  $\tfrac{3}{10}\ket{0001}+\tfrac{2}{5}\ket{0010}+\tfrac{\sqrt6}{4}\ket{0100}+\tfrac{\sqrt6}{4}\ket{1000}$（振幅平方和为 1）。
- 第 4(d) 给出两种思路：`problem1(1-4done).tex` 的构造式解法（11 CNOT + 6 单比特门、无辅助比特），
  以及 `problem1_4d_answer.tex` 的 CSS 编码最优线路（27 门）与门数紧下界证明，二者均收录于汇总 tex。

---

## 编译说明（生成 PDF）

汇总文档为中文 `ctexart`，含 TikZ 线路图与外部图片，需 **XeLaTeX** 编译（本机当前未安装 TeX 引擎，故未附带已编译的 PDF）。

在装有 TeX Live/MacTeX 的环境中，于本文件夹内执行：

```bash
xelatex Problem1_Complete_Solution.tex
xelatex Problem1_Complete_Solution.tex   # 第二遍处理交叉引用
```

或直接将本文件夹上传到 **Overleaf**（编译器选择 XeLaTeX）即可得到 `Problem1_Complete_Solution.pdf`。
编译需保证 `figs/` 目录与 tex 位于同一相对路径下（tex 中使用 `\includegraphics{figs/...}`）。

---

## 代码运行说明

`code/` 下的脚本为从原仓库**原样复制**，其中 `problem7_reversible.py` 与 `8_grover.py` 依赖
`from lib.quantum_circuits import QuantumCircuits`，并按“脚本位于仓库 `Problem1/code/` 下、
向上三级即仓库根、根下有 `lib/`”的布局设置 `sys.path`。因此建议**在原仓库根目录环境下运行**这些脚本；
本文件夹内附带的 `quantum_circuits.py` 仅作内容归档。

依赖：`numpy`、`tensorcircuit`（第 7、8 题）、`qiskit` 与 `matplotlib`（第 6 题绘图）。
