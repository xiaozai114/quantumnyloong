"""
问题 1.6：测量 —— 量子线路绘制
用 qiskit 的 .draw(output='mpl') 把第 6 问中的所有量子线路保存为 PNG。
（图内文字用英文/数学符号，避免 matplotlib 缺中文字体；注释与终端输出为中文。）
"""
import os
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Gate

OUT = "/mnt/d/sqd/Problem1/figs"      # 脚本在 WSL 中运行，对应 D:\sqd\Problem1\figs
os.makedirs(OUT, exist_ok=True)
DPI = 150
TEXTFILE = os.path.join(OUT, "q6_circuits.txt")
_txt = open(TEXTFILE, "w", encoding="utf-8")


def save(qc, name, title):
    """绘制线路并以 PNG 保存；title 为图的英文/数学标题。
    同时把 ASCII/文本版线路写入 q6_circuits.txt，便于核对结构。"""
    fig = qc.draw(output="mpl", initial_state=True)
    fig.suptitle(title, fontsize=12)
    path = os.path.join(OUT, name + ".png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  已保存：{path}")
    _txt.write("=" * 60 + "\n")
    _txt.write(title + "\n")
    _txt.write("-" * 60 + "\n")
    _txt.write(str(qc.draw(output="text", initial_state=True)))
    _txt.write("\n\n")


print("=" * 64)
print("问题 1.6：量子线路绘制（输出到 Problem1/figs/）")
print("=" * 64)

# ============================================================
# (a1) Z1Z2 的非破坏性测量（奇偶校验写入辅助比特）
# ============================================================
q = QuantumRegister(2, "q")
a = QuantumRegister(1, "a")
qc = QuantumCircuit(q, a, ClassicalRegister(1, "c"))
qc.cx(q[0], a[0])
qc.cx(q[1], a[0])
qc.measure(a[0], 0)
save(qc, "q6_a1_zz_measure", "(a) Non-destructive measurement of $Z_1Z_2$")

# ============================================================
# (a2) X1X2 的非破坏性测量（H 旋到 Z 基 -> 奇偶校验 -> 复原）
# ============================================================
q = QuantumRegister(2, "q")
a = QuantumRegister(1, "a")
qc = QuantumCircuit(q, a, ClassicalRegister(1, "c"))
qc.h(q[0]); qc.h(q[1])
qc.cx(q[0], a[0]); qc.cx(q[1], a[0])
qc.h(q[0]); qc.h(q[1])
qc.measure(a[0], qc.clbits[0])
save(qc, "q6_a2_xx_measure", "(a) Non-destructive measurement of $X_1X_2$")

# ============================================================
# (a3) Bell 基测量：CNOT(q1->q2)；H(q1)；测量两个比特
# ============================================================
q = QuantumRegister(2, "q")
qc = QuantumCircuit(q, ClassicalRegister(2, "c"))
qc.cx(q[0], q[1])
qc.h(q[0])
qc.measure(q[0], 0); qc.measure(q[1], 1)
save(qc, "q6_a3_bell_measure",
     "(a) Bell-basis measurement  ($\\Phi^+\\to 00$, project onto $+1$ eigenspace)")

# ============================================================
# (b) 任意 Pauli 算符 P=P1⊗P2 的通用投影测量（模板）
#     基旋转 U_i (R P_i R†=Z) -> CNOT 奇偶校验 -> 复原 U_i†
# ============================================================
q = QuantumRegister(2, "q")
a = QuantumRegister(1, "a")
qc = QuantumCircuit(q, a, ClassicalRegister(1, "c"))
U1 = Gate("U_1", 1, []); U2 = Gate("U_2", 1, [])
U1d = Gate("U_1†", 1, []); U2d = Gate("U_2†", 1, [])
qc.append(U1, [q[0]]); qc.append(U2, [q[1]])
qc.cx(q[0], a[0]); qc.cx(q[1], a[0])
qc.append(U1d, [q[0]]); qc.append(U2d, [q[1]])
qc.measure(a[0], qc.clbits[0])
save(qc, "q6_b_pauli_measure",
     "(b) General Pauli measurement of $P=P_1\\otimes P_2$  ($U_iP_iU_i^\\dagger=Z$)")

# ============================================================
# (c) GHZ(n=3) -> 单量子比特：测量 q1,q2 于 X 基 + 对 q0 经典前馈 Z^s
# ============================================================
q = QuantumRegister(3, "q")
qc = QuantumCircuit(q, ClassicalRegister(2, "c"))
# GHZ 态制备（作为上下文）
qc.h(q[0]); qc.cx(q[0], q[1]); qc.cx(q[1], q[2])
# 在 X 基下测量 q1,q2
qc.h(q[1]); qc.h(q[2])
qc.measure(q[1], 0); qc.measure(q[2], 1)
# 对 q0 作经典前馈 Z^s（s 为测量结果的奇偶）
qc.z(q[0])   # 受经典比特控制（见标题说明）
save(qc, "q6_c_ghz_to_single",
     "(c) GHZ$\\to$ single qubit: measure $q_1,q_2$ in $X$ basis; feed-forward $Z^s$ on $q_0$")

# ============================================================
# (d) 扇出门 F4：GHZ 猫态辅助比特 + 测量（常数深度）
# ============================================================
q = QuantumRegister(5, "q")
a = QuantumRegister(5, "a")
qc = QuantumCircuit(q, a, ClassicalRegister(5, "c"))
# 在 a0..a4 上制备 GHZ
qc.h(a[0]); qc.cx(a[0], a[1]); qc.cx(a[1], a[2]); qc.cx(a[2], a[3]); qc.cx(a[3], a[4])
# 把控制 q0 注入猫态
qc.cx(q[0], a[0])
# 把 ai 耦合到目标 qi (i=1..4)
for i in range(1, 5):
    qc.cx(a[i], q[i])
# 在 X 基下测量所有辅助比特
for i in range(5):
    qc.h(a[i])
qc.measure(a, [0, 1, 2, 3, 4])
# 对目标 qi 作相位修正 Z^{m_i⊕m0}（受经典控制）
for i in range(1, 5):
    qc.z(q[i])
save(qc, "q6_d_fanout_f4",
     "(d) Fanout $F_4$ via GHZ ancillas + measurement (constant depth)")

# ============================================================
# (e1) SWAP 检验（高层：受控交换门）
# ============================================================
c = QuantumRegister(1, "c")
psi = QuantumRegister(1, "psi")
phi = QuantumRegister(1, "phi")
qc = QuantumCircuit(c, psi, phi, ClassicalRegister(1, "m"))
qc.h(c[0]); qc.cswap(c[0], psi[0], phi[0]); qc.h(c[0]); qc.measure(c[0], 0)
save(qc, "q6_e_swap",
     "(e) Swap test  $P(c=0)=(1+|\\langle\\psi|\\phi\\rangle|^2)/2$")

# ============================================================
# (e2) SWAP 检验的线路分解：cswap = 2·CNOT + Toffoli
# ============================================================
c = QuantumRegister(1, "c")
psi = QuantumRegister(1, "psi")
phi = QuantumRegister(1, "phi")
qc = QuantumCircuit(c, psi, phi, ClassicalRegister(1, "m"))
qc.h(c[0])
qc.cx(psi[0], phi[0])
qc.ccx(c[0], phi[0], psi[0])          # Toffoli
qc.cx(psi[0], phi[0])
qc.h(c[0]); qc.measure(c[0], 0)
save(qc, "q6_e_swap_decomposed",
     "(e) Swap test decomposed: cswap = $2\\cdot$CNOT $+$ Toffoli")

print("\n全部线路已绘制完成。")
print(f"文本版线路已写入：{TEXTFILE}")
_txt.close()
