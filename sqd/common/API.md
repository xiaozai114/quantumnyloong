# common/ API 参考

各模块公开函数的签名、参数、返回值与用法。函数级 docstring 亦有说明。

---

## chemistry.py — 化学积分

### `molecule_report(name, do_fci=True, do_of=True)`

取 PySCF RHF 积分与参考解。

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | str | 分子名：`"H2"`/`"LiH"`/`"H2O"`/`"N2"`/`"C2H4"` |
| `do_fci` | bool | 是否算 FCI（大分子可关） |
| `do_of` | bool | 是否构造 openfermion FermionOperator |

返回 dict：`mol, mf, norb, nocc, h1e, eri, ecore, E_HF, E_FCI, E_CCSD, t1, t2, fermion_hamiltonian`。

```python
from common.chemistry import molecule_report
rep = molecule_report("H2")
print(rep["E_FCI"], rep["norb"], rep["h1e"].shape)
```

---

## mapping.py — 费米子→量子比特映射

### `build_fermion_hamiltonian(inter=None, h1e=None, eri=None, ecore=None, norb=None, fermion_hamiltonian=None)`
构造 openfermion `FermionOperator`。优先用 `inter`（InteractionOperator）。

### `jw(ham)` / `parity(ham)` / `bk(ham)`
费米子→量子比特映射，返回 `QubitOperator`。

### `pauli_stats(qop)` → `(nterms, max_weight)`
统计 Pauli 项数与最大 Pauli 权重。

### `reduce_2q(qop, nq)`
闭壳层 Z₂ 粒子数约化（4q→2q）。

### `qubit_matrix(qop, nq)` / `qubit_matrix_sparse(qop, nq)`
稠密 / 稀疏 Pauli 矩阵。

### `ground_energy(qop, nq)` → `float`
稀疏迭代求基态能量（大体系用，比 `diagonalize(...)[0]` 快）。

### `diagonalize(qop, nq)` → `np.ndarray`
稠密全谱对角化（小体系用）。

```python
from common.mapping import jw, pauli_stats, ground_energy
qop = jw(rep["fermion_hamiltonian"])
print(pauli_stats(qop))          # (15, 4)
print(ground_energy(qop, nq))    # = E_FCI
```

---

## ci.py — Slater-Condon CI

### `full_fci_basis(norb, n_alpha, n_beta)` → `List[str]`
枚举 (n_α, n_β) 扇区的全 FCI JW bitstring。

### `int_to_bitstring(key, nq)` → `str`
整数 key（LSB=qubit0）→ bitstring（`bs[q]` = qubit q）。

### `parse_jw_bitstring(bs, nq)` → `(alpha, beta, n_alpha, n_beta)`
解析 JW bitstring 的 α/β 占据。

### `solve_subspace(basis, h1e, eri, nq, ecore)` → `(E, evals, evecs)`
在给定行列式子空间用 Slater-Condon 构建CI 矩阵并对角化。

---

## sqd.py — SQD 流程

### `config_recovery(counts, nq, n_alpha, n_beta, method="max_dev")` → `List[str]`
配置恢复：修正违反粒子数的 bitstring。

| 参数 | 说明 |
|------|------|
| `method` | `"max_dev"`（默认，贪心最大似然）或 `"directed"`（定向，必收敛） |

### `config_recovery_counts(counts, nq, n_alpha, n_beta, method, return_stats)` → `Dict[int,int]`
同上，保留恢复后各构型计数（供串权重排序）。

### `bitstrings_to_ci_strs(counts, nq, open_shell=False, n_alpha=None, n_beta=None)` → `(List[int], List[int])`
counts → (α串集, β串集)，按权重降序。`n_alpha`/`n_beta` 给定时过滤扇区外构型。

### `run_sqd(h1e, eri, nq, ecore, basis)` → `dict(E_sqd, evals, evecs, basis, M)`
给定行列式列表的 Slater-Condon 对角化。

### `run_sqd_product(h1e, eri, nq, ecore, a_strs, b_strs, max_dets=8000, include=None)` → `dict`
α×β 笛卡尔积子空间对角化。`include` 为必须包含的 JW bitstring（如 HF）。

### `sqd_from_counts(counts, nq, n_alpha, n_beta, h1e, eri, ecore, hf_bs=None, max_dets=8000, method="max_dev")` → `dict`
端到端：恢复 → α/β 串 → 乘积子空间 SQD。返回 `E_sqd, M, n_recovered, n_ci_strs_a/b`。

```python
from common.sqd import sqd_from_counts
res = sqd_from_counts(counts, nq, nocc, nocc, h1e, eri, ecore, hf_bs=hf_bs)
print(res["E_sqd"], res["M"])
```

---

## circuits.py — 量子电路

### `prepare_hf(c, norb, nocc)`
在电路 `c` 上制备 HF 态（前 nocc 空间轨道占据）。

### `build_lucj(c, norb, nocc, t1, t2, eri, ccsd_scale=1.0, local=True, doubles=False, doubles_thresh=1e-4)` → `dict`
构造 LUCJ ansatz：轨道旋转 e^κ + 对角 Coulomb e^{iJ} + 可选双激发通道。

### `rzz(c, i, j, theta)` / `givens(c, i, j, theta)`
RZZ 门（2 CNOT + RZ）与 Givens 门（2 CNOT + RY）。

### `statevector(c)` → `np.ndarray`
返回扁平态矢量（openfermion 序：qubit0=LSB）。

### `expectation(psi, H)` → `float`
态矢量对 Hamiltonian 矩阵的期望值。

### `sample_counts(c, n_shots, nq, seed=None)` → `Dict[int,int]`
采样返回 `{openfermion_int_key: count}`。`seed` 给定时用独立 RNG（线程安全）。

```python
from common.circuits import prepare_hf, build_lucj, sample_counts
c = tc.Circuit(nq); prepare_hf(c, norb, nocc)
counts = sample_counts(c, 8000, nq, seed=42)
```

---

## dmet.py — DMET 碎片化

### `build_clusters(mf, frag_atom_lists, tol=1e-8)` → `List[dict]`
DMET 碎片化：每个碎片返回 `h1e, eri, norb, nocc, n_imp, nbath, C_cl, schmidt, atoms`。

### `cluster_scf(cl)` → `dict(E_elec, dm1, converged)`
团簇 HF 自洽场。

### `cluster_mp2_t2(cl, scf_res)` → `np.ndarray`
团簇 MP2 双激发振幅 t2。

### `hf_rdms(cl, scf_res)` → `(dm1, dm2)`
团簇 HF 的 1-/2-RDM。

### `rdms_from_sqd(basis, vec, norb, n_alpha, n_beta)` → `(dm1, dm2)`
从 SQD 波函数振幅提取 1-/2-RDM。

---

## ewf_ref.py — EWF 参考

### `ewf_reference(name, bath_type="dmet", threshold=1e-6, solver="FCI", frag_atoms=None)` → `dict`
EWF 碎片提取与能量重构。

| 参数 | 说明 |
|------|------|
| `frag_atoms` | 碎片原子组列表，如 `[[0],[1,2]]`；None 用原子碎片 |

返回 `e_mf, e_tot_ewf`(DM式), `e_tot_ewf_wf`(WF式), `e_corr_ewf`, `fragments`(含 `heff/eris/norb/nocc/nelec/e_corr_proj/e_corr_wf/label`)。

### `cluster_fci(heff, eris, norb, nelec)` → `float`
团簇 FCI 电子能量（pyscf direct_spin1）。

```python
from common.ewf_ref import ewf_reference
r = ewf_reference("H2O", frag_atoms=[[0],[1,2]])
print(r["e_tot_ewf"], r["e_tot_ewf_wf"])
```

---

## cluster_solver.py — 团簇 SQD 求解

### `solve_cluster_sqd(frag, n_shots=20000, lam=5.0, seed=None)` → `(E_sqd, E_hf, M, stats)`
单团簇 SQD：小团簇全 FCI；大团簇 LUCJ 采样+恢复+乘积子空间+扩展。

### `solve_clusters(frags, n_shots, lam, parallel=True, max_workers=None)` → `List`
并行求解多团簇（碎片级并行）。

### `expand_configs(basis, vec, norb, nocc, weight_cut=0.02, max_M=1500)` → `List`
S-CORE 式子空间扩展（重要构型的保 Sz 单+双激发）。

---

## parallel.py — 线程并行

### `parallel_map(fn, items, max_workers=None)` → `List`
保序并行映射（ThreadPoolExecutor + threadpoolctl BLAS 限制）。

```python
from common.parallel import parallel_map
results = parallel_map(lambda x: x**2, [1,2,3,4])
```

---

## hardware.py — 真机/模拟器采样

### `sample(c, n_shots, nq, backend="sim", device="", token="", dry_run=True, reverse=True, task_label="")` → `Dict[int,int]`
统一采样接口。

| 参数 | 说明 |
|------|------|
| `backend` | `"sim"`（本地模拟器）或 `"qpu"`（tensorcircuit.cloud） |
| `dry_run` | True 时 qpu 分支只做资源统计不提交 |
| `token` | 留空则从 `QPU_TOKEN` 环境变量读取 |

### `circuit_resource_summary(c)` → `dict(nq, n_1q, n_2q)`
电路资源统计（基于 QASM 文本）。

### `circuit_to_qasm(c)` → `str`
导出 OpenQASM 文本。

### `counts_bitstring_to_int(counts, nq, reverse=True)` → `Dict[int,int]`
真机 counts 比特序转 openfermion int key。

```python
from common.hardware import sample
counts = sample(c, 4000, nq, backend="qpu", device="tianji-s2", dry_run=False)
```

---

## exercise.py — 结果存档

### `save_results(qid, exact, quantum, meta=None, tol=1e-3, out_dir=None)`
写 `results.json` 并打印对比。

### `check_close(qid, exact, got, tol, label)`
断言 `|exact - got| < tol`，打印 PASS/FAIL。

---

## cost.py — 代价汇总

### `cost_row(qid, title, metrics)` → `dict`
构造代价表行。

### `render_summary(rows, out_path)`
渲染 `cost_summary.md`。
