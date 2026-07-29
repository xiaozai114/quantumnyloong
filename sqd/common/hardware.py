"""硬件/真机采样抽象层（基于 TensorCircuit Cloud 交互逻辑）。

统一 `sample()` 接口，让各题的采样代码在「本地模拟器」与「量子云真机」之间
无缝切换：

- backend="sim"（默认）：本地态矢量快速采样（common.circuits.sample_counts），
  等价于理想模拟器，无需任何凭据，用于开发/回归/CI。
- backend="qpu"：经 ``tensorcircuit.cloud`` 提交到云真机：
  * ``Device.from_name(device)`` 构造设备，``device.set_token(token)`` 注入凭据，
    ``device.submit_task(circuit=..., shots=...)`` 提交任务，
    ``task.results()`` 阻塞取回 ``{bitstring: count}``。
  * 默认 dry_run=True：只做电路资源统计，**不实际提交**，避免误消耗真机
    额度；返回本地模拟器的 counts 以便流程贯通。
  * dry_run=False 且 token 可用时才真正提交。

设计原则（安全默认）：
  1. 凭据 env-only：Token 从环境变量 QPU_TOKEN 读取，绝不硬编码。
  2. 计费的真机提交默认关闭，需显式 dry_run=False 才触发。
  3. 真机 counts 的比特序与本地约定统一（openfermion: qubit0=LSB）。
"""
from __future__ import annotations

import os
from typing import Dict

from common.circuits import sample_counts as _sim_sample_counts
from common.circuits import bitrev


# ── 电路资源统计（dry-run 用）──

def circuit_resource_summary(c) -> Dict[str, int]:
    """统计电路规模（比特数、单比特门数、双比特及以上门数），供 dry-run 报告。

    直接遍历 TensorCircuit 原生门列表 ``c.to_qir()``：每个门的 ``index``
    元组给出作用比特，按其长度分类（1=单比特，2=双比特，>2=多比特并入
    n_2q）。避免依赖 ``to_openqasm``/``to_qiskit`` —— 后者在 qiskit>=1.0
    下因 tensorcircuit 0.12.0 使用已移除的 ``qiskit.extensions`` 而报错。
    """
    nq = getattr(c, "_nqubits", None)
    if nq is None:
        nq = c.circuit_param.get("nqubits", -1)
    n1q, n2q = 0, 0
    for g in c.to_qir():
        idx = g.get("index", ()) or ()
        k = len(idx)
        if k <= 1:
            n1q += 1
        else:
            # 双比特（cnot 等）与多比特块（LUCJ 'any'）统一计入 n_2q
            n2q += 1
    return dict(nq=int(nq), n_1q=n1q, n_2q=n2q)


def circuit_to_qasm(c) -> str:
    """将 TensorCircuit 电路导出为 OpenQASM 文本（TC 自带接口）。

    注意：tensorcircuit 0.12.0 的 ``to_openqasm`` 依赖 ``to_qiskit``，后者
    在 qiskit>=1.0 环境下会因 ``qiskit.extensions`` 已被移除而失败。资源
    统计已改走 ``circuit_resource_summary`` 的原生实现，不再依赖此函数；
    仅在确有 QASM 文本需求且 qiskit<1.0 时可用。
    """
    if hasattr(c, "to_openqasm"):
        return c.to_openqasm()
    raise RuntimeError("当前 TensorCircuit 版本不支持 to_openqasm")


# ── counts 比特序转换 ──

def counts_bitstring_to_int(counts: Dict[str, int], nq: int,
                            reverse: bool = True) -> Dict[int, int]:
    """将真机返回的 {bitstring: count} 转为 {openfermion_int_key: count}。

    云返回的 bitstring 按经典寄存器顺序；本项目本地约定为 openfermion 序
    （qubit0=LSB）。reverse=True 时对 TC 的 qubit0=MSB 约定做比特反序，
    与 common.circuits.statevector 保持一致。

    真机接入首次联调时，请用一个已知电路核对 reverse 取值（True/False），
    确认 HF 构型落在期望的 int key 上再批量运行。
    """
    out: Dict[int, int] = {}
    for bs, ct in counts.items():
        bs = "".join(ch for ch in bs.strip() if ch in "01")
        if len(bs) != nq:
            bs = bs.zfill(nq)
        x = int(bs, 2)
        key = int(bitrev(x, nq)) if reverse else x
        out[key] = out.get(key, 0) + int(ct)
    return out


# ── tc.cloud 真机提交 ──

def _submit_via_tc_cloud(c, n_shots: int, device: str, token: str,
                         remarks: str = "sqd_practice_tc"):
    """经 tensorcircuit.cloud 提交电路到云真机，返回 {bitstring: count}。

    使用 tc.cloud 的标准交互逻辑：Device.from_name -> set_token ->
    submit_task -> task.results()（阻塞等待）。
    """
    from tensorcircuit.cloud import abstraction

    dev = abstraction.Device.from_name(device)
    tok = token or os.environ.get("QPU_TOKEN", "")
    if not tok:
        raise RuntimeError(
            "真机提交需要 QPU_TOKEN 环境变量（env-only，勿在代码中硬编码）。")
    dev.set_token(tok)
    tasks = dev.submit_task(circuit=c, shots=n_shots, remarks=remarks)
    task = tasks[0]
    print(f"[HW] task submitted, state={task.state()}")
    counts_bs = task.results()  # 阻塞等待 -> {bitstring: count}
    return counts_bs


# ── 统一采样接口 ──

def sample(c, n_shots: int, nq: int, *, backend: str = "sim",
           device: str = "", token: str = "", dry_run: bool = True,
           reverse: bool = True, task_label: str = "") -> Dict[int, int]:
    """统一采样：返回 {openfermion_int_key: count}，可直接喂给 config_recovery。

    Args:
        c: TensorCircuit 电路。
        n_shots: 采样次数。
        nq: 量子比特数。
        backend: "sim"（本地模拟器）或 "qpu"（云真机，tc.cloud）。
        device: 真机芯片名（backend="qpu" 且真提交时必填）。
        token: QPU Token；留空则从环境变量 QPU_TOKEN 读取（env-only）。
        dry_run: True（默认）时 qpu 分支只做资源统计，不真正提交。
        reverse: 真机 counts 比特序反转开关（见 counts_bitstring_to_int）。

    Returns:
        counts: {int_key: count}
    """
    if backend == "sim":
        return _sim_sample_counts(c, n_shots, nq)

    if backend != "qpu":
        raise ValueError(f"unknown backend: {backend!r} (expect 'sim'|'qpu')")

    # ---- QPU 分支（tc.cloud）----
    res = circuit_resource_summary(c)
    print(f"[HW] qpu backend | device={device or '(unset)'} "
          f"| shots={n_shots} | nq={res['nq']} "
          f"1q={res['n_1q']} 2q={res['n_2q']} | dry_run={dry_run}")

    if dry_run:
        # 安全默认：不提交，返回本地模拟器采样以贯通流程
        print("[HW] dry_run=True -> 不提交真机，返回本地模拟器 counts。"
              " 需真实提交请设 dry_run=False 并配置 QPU_TOKEN/device。")
        return _sim_sample_counts(c, n_shots, nq)

    if not device:
        raise ValueError("真机提交必须指定 device（芯片名）。")
    counts_bs = _submit_via_tc_cloud(
        c, n_shots, device, token, remarks=task_label or "sqd_practice_tc")
    print(f"[HW] outcomes={len(counts_bs)}")
    return counts_bitstring_to_int(counts_bs, nq, reverse=reverse)


if __name__ == "__main__":
    import common.backend  # noqa: F401
    import tensorcircuit as tc
    from common.chemistry import molecule_report
    from common.circuits import prepare_hf

    rep = molecule_report("H2", do_fci=False, do_of=False)
    norb, nocc = rep["norb"], rep["nocc"]
    nq = 2 * norb
    c = tc.Circuit(nq)
    prepare_hf(c, norb, nocc)

    print("[selftest] sim      :", sample(c, 200, nq, backend="sim"))
    print("[selftest] qpu(dry) :", sample(c, 200, nq, backend="qpu",
                                          device="tianji-s2", dry_run=True))
    # bitstring 转换核对
    demo = {"1010": 100, "0000": 50}
    print("[selftest] bs->int  :", counts_bitstring_to_int(demo, 4))
    # 资源统计核对
    print("[selftest] resource :", circuit_resource_summary(c))
