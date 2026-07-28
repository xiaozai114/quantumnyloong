"""SQD 真机/模拟器参考案例（H₂ 关联能）。

完整演示 SQD 端到端流程，可在「本地模拟器」与「量子云真机」之间切换：

    电路构造（HF 制备 + LUCJ ansatz）
      -> sample(backend="sim"|"qpu", dry_run=True|False)  # common.hardware
      -> 配置恢复（method="directed"|"max_dev"，common.sqd）
      -> α×β 笛卡尔积子空间对角化（qiskit-addon-sqd 对齐）
      -> 能量 + 与 FCI 对比

真机提交：设置环境变量 QPU_TOKEN 并传 dry_run=False、device=<芯片名>。
首次联调请用一个已知电路核对 common.hardware.counts_bitstring_to_int 的
``reverse`` 取值，确认 HF 构型落在期望的 int key 上。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common.backend  # noqa: F401
import tensorcircuit as tc

from common.chemistry import molecule_report
from common.circuits import prepare_hf, build_lucj
from common.hardware import sample, circuit_resource_summary
from common.sqd import sqd_from_counts


def run_sqd_example(backend: str = "sim", device: str = "",
                    dry_run: bool = True, n_shots: int = 4000,
                    recover_method: str = "directed", lam: float = 0.5):
    """H₂ LUCJ-SQD：采样 -> 配置恢复 -> 子空间对角化 -> 能量。"""
    rep = molecule_report("H2")
    norb, nocc = rep["norb"], rep["nocc"]
    nq = 2 * norb
    h1e, eri, ecore = rep["h1e"], rep["eri"], rep["ecore"]
    hf_bs = "".join("1" if q < 2 * nocc else "0" for q in range(nq))

    # 1) 电路构造：HF 制备 + LUCJ（CCSD 振幅缩放 λ）
    c = tc.Circuit(nq)
    prepare_hf(c, norb, nocc)
    build_lucj(c, norb, nocc, rep["t1"], rep["t2"], eri=rep["eri"],
               ccsd_scale=lam, local=True, doubles=True, doubles_thresh=1e-5)

    # 2) 资源统计（dry-run 与真提交都会打印）
    res = circuit_resource_summary(c)
    print(f"[SQD-example] circuit: nq={res['nq']} 1q={res['n_1q']} 2q={res['n_2q']} "
          f"| backend={backend} device={device or '-'} dry_run={dry_run}")

    # 3) 采样（sim 本地 / qpu 经 tensorcircuit.cloud）
    counts = sample(c, n_shots, nq, backend=backend, device=device,
                    dry_run=dry_run, task_label="sqd_h2_example")

    # 4) 后处理：配置恢复（method 可选）+ α×β 笛卡尔积子空间 SQD
    #    sqd_from_counts 内部：config_recovery_counts(method=...) ->
    #    bitstrings_to_ci_strs -> run_sqd_product
    # 注意：sqd_from_counts 默认用 directed；如需 max_dev 对比，可自行调用
    # config_recovery_counts(counts, nq, nocc, nocc, method="max_dev") 等。
    from common.sqd import (config_recovery_counts, bitstrings_to_ci_strs,
                            run_sqd_product)
    rc = config_recovery_counts(counts, nq, nocc, nocc, method=recover_method)
    a_strs, b_strs = bitstrings_to_ci_strs(rc, nq)
    sqd_res = run_sqd_product(h1e, eri, nq, ecore, a_strs, b_strs,
                              include=[hf_bs])

    # 5) 结果
    e_sqd = sqd_res["E_sqd"]
    print(f"[SQD-example] recovered configs={len(rc)}  "
          f"ci_strs=({sqd_res['n_ci_strs_a']},{sqd_res['n_ci_strs_b']})  "
          f"subspace M={sqd_res['M']}")
    print(f"[SQD-example] E_SQD   = {e_sqd:.6f} Ha")
    print(f"[SQD-example] E_HF    = {rep['E_HF']:.6f} Ha")
    print(f"[SQD-example] E_FCI   = {rep['E_FCI']:.6f} Ha")
    print(f"[SQD-example] 关联能捕获 = {(e_sqd - rep['E_HF']) / (rep['E_FCI'] - rep['E_HF']):.4f}")
    return e_sqd


if __name__ == "__main__":
    # 默认本地模拟器（无需凭据）
    run_sqd_example(backend="sim")
    print("\n--- 真机提交示例（需 QPU_TOKEN，此处仅 dry-run）---")
    run_sqd_example(backend="qpu", device="tianji-s2", dry_run=True)
    # 真提交：
    #   export QPU_TOKEN=<your_token>
    #   run_sqd_example(backend="qpu", device="tianji-s2", dry_run=False)
