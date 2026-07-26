"""
Problem 7: Reversible Circuit
使用 Toffoli 和 NOT 门构造可逆经典电路，计算布尔函数 S₃(x)。

S₃(x) = (¬x₁ ∨ ¬x₂) ∧ (x₁ ∨ ¬x₂) ∧ (¬x₁ ∨ x₂ ∨ ¬x₃)
        ∧ (¬x₁ ∨ x₂ ∨ x₃) ∧ (x₁ ∨ x₂ ∨ x₃)

通过真值表分析，S₃(x) 等价于单一极小项: ¬x₁ ∧ ¬x₂ ∧ x₃。

电路图使用 Qiskit 的 .draw() 方法生成（替代 matplotlib 手绘），
利用 TensorCircuit 的 to_qiskit() 接口完成转换。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import tensorcircuit as tc
from lib.quantum_circuits import QuantumCircuits

K = tc.set_backend("numpy")


def s3_truth_table():
    """返回 S₃(x) 的真值表: {(x₁, x₂, x₃): S₃(x)}"""
    table = {}
    for x1 in range(2):
        for x2 in range(2):
            for x3 in range(2):
                c1 = (not x1) or (not x2)          # ¬x₁ ∨ ¬x₂
                c2 = x1 or (not x2)                # x₁ ∨ ¬x₂
                c3 = (not x1) or x2 or (not x3)    # ¬x₁ ∨ x₂ ∨ ¬x₃
                c4 = (not x1) or x2 or x3          # ¬x₁ ∨ x₂ ∨ x₃
                c5 = x1 or x2 or x3                # x₁ ∨ x₂ ∨ x₃
                result = c1 and c2 and c3 and c4 and c5
                table[(x1, x2, x3)] = int(result)
    return table


def verify_circuit():
    """对全部 8 种输入验证可逆电路的正确性。"""
    print("=" * 50)
    print("S₃(x) 可逆电路验证")
    print("=" * 50)
    print(f"{'x₁':>4} {'x₂':>4} {'x₃':>4}  |  {'期望':>4}  {'输出':>4}  {'通过':>4}")
    print("-" * 40)

    truth = s3_truth_table()
    all_pass = True

    for x1 in range(2):
        for x2 in range(2):
            for x3 in range(2):
                c = tc.Circuit(5)

                if x1: c.x(0)
                if x2: c.x(1)
                if x3: c.x(2)

                QuantumCircuits.build_s3(c)

                result = c.measure(4, with_prob=False)
                output = int(result[0][0])
                expected = truth[(x1, x2, x3)]
                ok = "✓" if output == expected else "✗"
                if output != expected:
                    all_pass = False

                print(f"{x1:>4} {x2:>4} {x3:>4}  |  {expected:>4}  {output:>4}  {ok:>4}")

    print("-" * 40)
    if all_pass:
        print("全部 8 种输入验证通过！S₃(x) = ¬x₁ ∧ ¬x₂ ∧ x₃")
    else:
        print("存在验证失败的输入！")
    print()


def draw_circuit_diagram(save_path="problem7_circuit.pdf"):
    """使用 Qiskit 绘制 S₃ 可逆电路图。

    流程: TensorCircuit 构建电路 → to_qiskit() 转换 → Qiskit .draw() 出图。
    """
    # 1. 用 TensorCircuit 构建 S₃ 电路（复用 build_s3）
    c = tc.Circuit(5)
    QuantumCircuits.build_s3(c)

    # 2. 通过 TensorCircuit 的 Qiskit 接口转换
    qc_raw = c.to_qiskit()

    # 3. 创建带标签的 Qiskit 电路（to_qiskit 生成默认 q_0..q_4，我们重新标注）
    from qiskit.circuit import QuantumRegister
    from qiskit import QuantumCircuit

    qr = QuantumRegister(5, "q")
    qc = QuantumCircuit(qr)
    # 复制门操作
    for instruction, qargs, cargs in qc_raw.data:
        qc.append(instruction, qargs, cargs)

    # 4. 用 Qiskit 绘制 (output="mpl" 生成 matplotlib 图，可直接存 PDF)
    style = {
        "figwidth": 11,
        "dpi": 150,
        "fontsize": 12,
        "showindex": True,
        "displaytext": {
            "x": "X",
            "ccx": "",
        },
    }
    fig = qc.draw(output="mpl", style=style, scale=1.0)

    # 5. 添加标题和信息框（Qiskit 图上的 matplotlib 操作）
    ax = fig.axes[0]
    ax.set_title("Problem 7: S$_3$(x) Reversible Circuit (Toffoli + NOT)\n"
                 "via TensorCircuit → Qiskit → .draw()",
                 fontsize=13, pad=15)

    info_text = (
        "S$_3$(x) = $\\neg x_1 \\wedge \\neg x_2 \\wedge x_3$\n"
        "Gates: 4 NOT + 3 Toffoli = 7\n"
        "Qubits: 3 input + 1 ancilla + 1 result = 5    Product: 7 $\\times$ 5 = 35\n"
        "Pipeline: TensorCircuit build_s3 → .to_qiskit() → qiskit.draw(output='mpl')"
    )
    fig.text(0.5, -0.08, info_text, fontsize=9, ha="center", va="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
             transform=ax.transAxes)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"电路图已保存至: {save_path}  (Qiskit 绘制)")
    # 同时输出文本版
    print("\n文本版电路图 (qiskit .draw(output='text')):")
    print(qc.draw(output="text"))


if __name__ == '__main__':
    print("\nS₃(x) 真值表:")
    print("-" * 25)
    truth = s3_truth_table()
    for (x1, x2, x3), val in truth.items():
        print(f"  ({x1}, {x2}, {x3}): {val}")
    print()

    verify_circuit()
    draw_circuit_diagram()
