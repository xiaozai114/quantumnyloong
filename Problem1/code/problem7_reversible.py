"""
Problem 7: Reversible Circuit
使用 Toffoli 和 NOT 门构造可逆经典电路，计算布尔函数 S₃(x)。

S₃(x) = (¬x₁ ∨ ¬x₂) ∧ (x₁ ∨ ¬x₂) ∧ (¬x₁ ∨ x₂ ∨ ¬x₃)
        ∧ (¬x₁ ∨ x₂ ∨ x₃) ∧ (x₁ ∨ x₂ ∨ x₃)

通过真值表分析，S₃(x) 等价于单一极小项: ¬x₁ ∧ ¬x₂ ∧ x₃。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
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
    """使用 matplotlib 绘制 S₃ 可逆电路图。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.5)

    y = {'x1': 4, 'x2': 3, 'x3': 1.5, 'anc': 2.25, 'res': 0.5}
    labels = ['$x_1$', '$x_2$', 'anc', '$x_3$', 'res']
    y_positions = [y['x1'], y['x2'], y['anc'], y['x3'], y['res']]

    for yi in y_positions:
        ax.plot([0, 13.5], [yi, yi], 'k-', linewidth=0.8)
    for yi, label in zip(y_positions, labels):
        ax.text(-0.5, yi, label, fontsize=12, ha='right', va='center')

    positions = {
        'X1': 2, 'X2': 2,
        'T1': 4.5, 'T2': 7.5, 'T3': 10.5,
        'X3': 12, 'X4': 12,
    }

    for px in [1, 3, 6, 9, 11, 13]:
        ax.axvline(x=px, color='gray', linestyle=':', linewidth=0.5)

    def draw_not(ax, x, y_q):
        r = 0.2
        ax.add_patch(plt.Circle((x, y_q), r, color='white', ec='black', linewidth=1.5, zorder=5))
        ax.text(x, y_q, 'X', fontsize=8, ha='center', va='center', zorder=6, fontweight='bold')

    def draw_control(ax, x, y_q):
        ax.plot(x, y_q, 'ko', markersize=6, zorder=5)

    def draw_target(ax, x, y_q):
        r = 0.25
        ax.add_patch(plt.Circle((x, y_q), r, color='white', ec='black', linewidth=1.5, zorder=5))
        ax.plot([x - 0.1, x + 0.1], [y_q, y_q], 'k-', linewidth=1.5, zorder=6)
        ax.plot([x, x], [y_q - 0.1, y_q + 0.1], 'k-', linewidth=1.5, zorder=6)

    def draw_vertical(ax, x, y1, y2):
        ax.plot([x, x], [y1, y2], 'k-', linewidth=1.5, zorder=4)

    # NOT gates (Steps 1 & 5)
    draw_not(ax, positions['X1'], y['x1'])
    draw_not(ax, positions['X2'], y['x2'])

    # CCNOT 1: x1, x2 → anc
    draw_control(ax, positions['T1'], y['x1'])
    draw_control(ax, positions['T1'], y['x2'])
    draw_target(ax, positions['T1'], y['anc'])
    draw_vertical(ax, positions['T1'], y['x1'], y['anc'])

    # CCNOT 2: anc, x3 → res
    draw_control(ax, positions['T2'], y['anc'])
    draw_control(ax, positions['T2'], y['x3'])
    draw_target(ax, positions['T2'], y['res'])
    draw_vertical(ax, positions['T2'], y['anc'], y['res'])

    # CCNOT 3: x1, x2 → anc (uncompute)
    draw_control(ax, positions['T3'], y['x1'])
    draw_control(ax, positions['T3'], y['x2'])
    draw_target(ax, positions['T3'], y['anc'])
    draw_vertical(ax, positions['T3'], y['x1'], y['anc'])

    # NOT gates (uncompute)
    draw_not(ax, positions['X3'], y['x1'])
    draw_not(ax, positions['X4'], y['x2'])

    stage_y = 4.8
    for i, x_pos in enumerate([1.5, 4.5, 7.5, 10.5, 12.5], 1):
        ax.text(x_pos, stage_y, f'({i})', fontsize=9, ha='center')

    ax.set_title('Problem 7: S$_3$(x) Reversible Circuit (Toffoli + NOT)', fontsize=13, pad=15)
    ax.axis('off')

    info_text = (
        "S$_3$(x) = $\\neg x_1 \\wedge \\neg x_2 \\wedge x_3$\n"
        "Gates: 4 NOT + 3 Toffoli = 7\n"
        "Qubits: 3 (input) + 1 (ancilla) + 1 (result) = 5\n"
        "Product: 7 x 5 = 35\n"
        "Steps: (1) flip $x_1$, $x_2$  (2) anc = $\\neg x_1 \\wedge \\neg x_2$\n"
        "       (3) res = anc $\\wedge$ $x_3$  (4) uncompute anc  (5) uncompute flip"
    )
    ax.text(7, -1.2, info_text, fontsize=9, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"电路图已保存至: {save_path}")
    plt.close()


if __name__ == '__main__':
    print("\nS₃(x) 真值表:")
    print("-" * 25)
    truth = s3_truth_table()
    for (x1, x2, x3), val in truth.items():
        print(f"  ({x1}, {x2}, {x3}): {val}")
    print()

    verify_circuit()
    draw_circuit_diagram()
