# common/ — SQD 子项目公共库

子项目 `sqd_practice_tc` 的公共库，为 15 道练习题提供统一的化学积分、量子比特映射、
SQD 流程、DMET/EWF 碎片化、量子电路、真机交互等能力。

## 模块总览

| 模块 | 职责 | 关键依赖 |
|------|------|---------|
| `backend.py` | 量子计算后端统一配置（GPU→jax，否则 numpy）；numpy 2.x / openfermion 兼容 shim | numpy, jax(可选), tensorcircuit |
| `chemistry.py` | PySCF 取 RHF 积分（h1e/eri/ecore）、FCI/CCSD/MP2 参考解、CCSD 振幅 t1/t2 | pyscf, openfermionpyscf |
| `mapping.py` | 费米子→量子比特映射（JW/Parity/BK）、Pauli 统计、Z₂ 约化、稀疏基态 | openfermion, scipy |
| `ci.py` | Slater-Condon CI 矩阵构建与对角化、JW bitstring 解析、全 FCI 基 | numpy, scipy |
| `sqd.py` | SQD 流程：配置恢复（max_dev/directed）、α×β 笛卡尔积子空间、对角化 | numpy, common.ci, common.circuits |
| `circuits.py` | 量子电路：HF 制备、LUCJ ansatz、fswap/Givens/doubles、快速采样、态矢量 | tensorcircuit, numpy |
| `dmet.py` | DMET 碎片化：Löwdin 正交、Schmidt bath、团簇积分、cluster_scf/RDM | numpy, scipy, pyscf |
| `ewf_ref.py` | EWF 参考：碎片提取、RDM 杂质归属（DM 式）/ 团簇 H 投影（WF 式）能量重构 | pyscf, common.dmet |
| `cluster_solver.py` | 团簇 SQD 求解：小团簇全 FCI、大团簇 LUCJ 采样+恢复+乘积子空间 | common.sqd, common.circuits, common.dmet |
| `parallel.py` | 线程级并行映射 + threadpoolctl BLAS 线程限制 | concurrent.futures, threadpoolctl |
| `hardware.py` | 真机/模拟器统一采样（tensorcircuit.cloud）、资源统计、比特序转换 | tensorcircuit.cloud, common.circuits |
| `cost.py` | 代价指标聚合与 cost_summary.md 渲染 | — |
| `exercise.py` | 每题 results.json 存档与一致性断言（check_close） | — |

## 依赖关系

```
chemistry ─┬─→ mapping ──→ ci ──→ sqd ──→ cluster_solver
           │                              ↑
           └─→ dmet ────→ ewf_ref ────────┤
                                         │
circuits ────────────────────────────────┘

backend（被所有 import tensorcircuit 的模块隐式依赖）
parallel（被 cluster_solver / 各 run.py 显式调用）
hardware（封装 circuits.sample_counts + tensorcircuit.cloud）
```

## 数据流（端到端 SQD-EWF 管线，Q10）

```
molecule_report(name)                     # chemistry: RHF 积分 + FCI/CCSD 参考
  → ewf_reference(name, frag_atoms)       # ewf_ref + dmet: 碎片化 + 团簇参考
    → solve_cluster_sqd(frag)             # cluster_solver: 每团簇
        → prepare_hf + build_lucj         #   circuits: 量子电路
        → sample_counts                   #   circuits: 采样
        → config_recovery_counts          #   sqd: 配置恢复
        → bitstrings_to_ci_strs           #   sqd: α/β 串分解
        → run_sqd_product                 #   sqd: 笛卡尔积子空间对角化
    → E = E_MF + Σ_f e_corr_proj × r_f   # ewf_ref: 能量重构
```

## 设计原则

- **模块单一职责**：化学积分、映射、CI、SQD、电路、碎片化各自独立，可单独 import。
- **后端集中管理**：`import common.backend` 即完成 tensorcircuit 后端初始化，无需散落
  `tc.set_backend`。
- **线程安全**：`sample_counts` 支持 `seed` 参数（独立 RNG）；`parallel_map` 用
  `threadpoolctl` 限制每线程 BLAS 线程避免超订。
- **安全默认**：真机采样 `dry_run=True`（不提交）；凭据 env-only（`QPU_TOKEN`）。
- **可测试**：`tests/` 含 12 项测试，含与 qiskit-addon-sqd 的严格一致性验证。

## 运行环境

```bash
PY=/home/Public/miniconda3/envs/vayesta/bin/python
$PY -m pytest tests/ -v          # 12 项测试
$PY q01/run.py                   # 运行单题
```

详见 [API.md](API.md) 获取各函数签名与用法。
