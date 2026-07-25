"""
问题 1.6：测量 —— 数值验证

用 NumPy 态矢量对问题 1.6 中的线路进行数值验证：
  (a) 对 Z1Z2 与 X1X2 作非破坏性测量，并投影到二者的共同 +1 本征态。
  (b) 任意 Pauli 算符 P 的通用投影测量。
  (c) 通过测量，从 n 量子比特 GHZ 型态制备 alpha|0> + sqrt(1-alpha^2)|1>。
  (e) 计算 |<psi|phi>| 的 SWAP 检验。

全局约定：长度为 2^n 的 n 量子比特向量，量子比特 i  <->  比特 i = (idx>>i)&1。
所有算子均按“作用在量子比特 i 上的门 = 作用在比特 i 上”的方式构造。
"""
import numpy as np

# ============================================================
# 基本门
# ============================================================
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
Sdg = S.conj().T
P0 = np.array([[1, 0], [0, 0]], dtype=complex)   # |0><0|
P1 = np.array([[0, 0], [0, 1]], dtype=complex)   # |1><1|

PAULI = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}
# 基旋转 R，满足 R P R^dag = Z（在奇偶校验 CNOT 之前施加）
ROT = {'I': I2, 'Z': I2, 'X': H, 'Y': H @ Sdg}


def kron_all(mats):
    r = mats[0]
    for m in mats[1:]:
        r = np.kron(r, m)
    return r


def _embed(mats_by_qubit, n):
    """mats_by_qubit[i] = 作用在量子比特 i（比特 i）上的单量子比特算符。
    按约定“量子比特 i = 比特 i”构造 2^n 维算符。"""
    ordered = [mats_by_qubit[n - 1 - k] for k in range(n)]   # kron 把 [0] 放在最高位
    return kron_all(ordered)


def op1(gate, target, n):
    return _embed([gate if i == target else I2 for i in range(n)], n)


def CNOT(c, t, n):
    mats0 = [I2] * n
    mats0[c] = P0
    mats1 = [I2] * n
    mats1[c] = P1
    mats1[t] = X
    return _embed(mats0, n) + _embed(mats1, n)


def pauli_op(pstr):
    """pstr[i] = 作用在量子比特 i 上的 Pauli（长度 = 量子比特数）。"""
    return _embed([PAULI[p] for p in pstr], len(pstr))


def rand_state(d):
    v = np.random.randn(d) + 1j * np.random.randn(d)
    return v / np.linalg.norm(v)


def measure_qubit(state, a, n):
    """在计算基下对量子比特 a 作投影测量。
    返回 (p0, |psi0>, |psi1>)，分别对应测量结果 0 / 1（已归一化的分支）。"""
    p0 = 0.0
    v0 = np.zeros_like(state)
    v1 = np.zeros_like(state)
    for idx in range(2 ** n):
        if (idx >> a) & 1:
            v1[idx] = state[idx]
        else:
            v0[idx] = state[idx]
            p0 += abs(state[idx]) ** 2
    p1 = 1.0 - p0
    return (p0,
            v0 / np.sqrt(p0) if p0 > 1e-15 else v0,
            v1 / np.sqrt(p1) if p1 > 1e-15 else v1)


def reduced_density(state, keep, n):
    """对 `keep` 中的量子比特求约化密度矩阵（比特 i = 量子比特 i）。
    `keep` 中第 p 个量子比特对应输出行下标的第 p 位。"""
    keep = list(keep)
    N = 2 ** n
    dk = len(keep)
    dim_k = 2 ** dk
    env = [q for q in range(n) if q not in keep]
    rho = np.zeros((dim_k, dim_k), dtype=complex)
    for idx in range(N):
        r = 0
        for p, q in enumerate(keep):
            if (idx >> q) & 1:
                r |= (1 << p)
        for idx2 in range(N):
            if all(((idx >> q) & 1) == ((idx2 >> q) & 1) for q in env):
                r2 = 0
                for p, q in enumerate(keep):
                    if (idx2 >> q) & 1:
                        r2 |= (1 << p)
                rho[r, r2] += state[idx] * state[idx2].conj()
    return rho


def tensor_ancilla(state, n_data):
    """在最高位（比特 = n_data）上追加一个 |0> 辅助比特。"""
    full = np.zeros(2 ** (n_data + 1), dtype=complex)
    full[:2 ** n_data] = state
    return full


print("=" * 64)
print("问题 1.6：测量 —— 数值验证")
print("=" * 64)

# ============================================================
# (a) 对 Z1Z2 与 X1X2 作稳定子测量 -> 投影到 |Phi+>
# ============================================================
print("\n(a) Z1Z2 与 X1X2 的共同 +1 本征态（= |Φ+>）")
Phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)   # q0,q1 上的 (|00>+|11>)/√2

ok_a = True
for trial in range(5):
    psi = rand_state(4)                                   # 数据比特 q0,q1
    full = np.zeros(16, dtype=complex)
    full[:4] = psi                                        # 辅助比特 a_Z=q2, a_X=q3 处于 |00>

    # 通过辅助比特 q2 测量 Z0Z1
    s1 = CNOT(0, 2, 4) @ CNOT(1, 2, 4) @ full
    pZ, s1p, _ = measure_qubit(s1, 2, 4)                  # +1 分支（结果为 0）

    # 通过辅助比特 q3 测量 X0X1（对两个数据比特施 H、奇偶校验、再复原）
    M = op1(H, 0, 4) @ op1(H, 1, 4) @ CNOT(0, 3, 4) @ CNOT(1, 3, 4) @ op1(H, 0, 4) @ op1(H, 1, 4)
    s2 = M @ s1p
    pX, s2p, _ = measure_qubit(s2, 3, 4)                  # +1 分支

    p_succ = pZ * pX
    p_exact = abs(Phi_plus.conj() @ psi) ** 2
    rho = reduced_density(s2p, [0, 1], 4)
    fidelity = np.real(np.trace(np.outer(Phi_plus, Phi_plus.conj()) @ rho))
    ok_a &= abs(p_succ - p_exact) < 1e-9 and fidelity > 1 - 1e-9
    print(f"  第 {trial} 次：成功概率(模拟)={p_succ:.6f}  成功概率(精确)={p_exact:.6f}  "
          f"F(|Φ+>)={fidelity:.10f}")

# 紧凑的 Bell 测量映射  Phi+->00, Phi-->10, Psi+->01, Psi-->11
bell = {
    'Phi+': np.array([1, 0, 0, 1], complex) / np.sqrt(2),
    'Phi-': np.array([1, 0, 0, -1], complex) / np.sqrt(2),
    'Psi+': np.array([0, 1, 1, 0], complex) / np.sqrt(2),
    'Psi-': np.array([0, 1, -1, 0], complex) / np.sqrt(2),
}
U_bell = op1(H, 0, 2) @ CNOT(0, 1, 2)
print("  Bell 测量（CNOT q0→q1；H q0）映射：")
for name, st in bell.items():
    out = U_bell @ st
    idx = int(np.argmax(np.abs(out)))
    print(f"    {name} -> |{format(idx, '02b')}>")
print("  (a) 通过" if ok_a else "  (a) 失败")

# ============================================================
# (b) 任意 Pauli 算符 P 的通用投影测量
# ============================================================
print("\n(b) 任意 Pauli 算符 P 的投影测量")


def measure_pauli(state, pstr):
    """模拟用于测量 Pauli 串 `pstr`（作用在量子比特 0..len(pstr)-1 上）的辅助比特线路。
    返回 (p_plus, |psi+>, |psi->)，其中 p_plus = P(本征值 +1)，
    两个分支为测量后的 (n+1) 量子比特态。"""
    nd = len(pstr)
    n = nd + 1
    anc = nd
    before = np.eye(2 ** n, dtype=complex)
    after = np.eye(2 ** n, dtype=complex)
    cnots = np.eye(2 ** n, dtype=complex)
    for i, p in enumerate(pstr):
        if p == 'I':
            continue
        before = op1(ROT[p], i, n) @ before
        after = after @ op1(ROT[p].conj().T, i, n)
        cnots = CNOT(i, anc, n) @ cnots
    U = after @ cnots @ before
    s = U @ tensor_ancilla(state, nd)
    p0, sp, sm = measure_qubit(s, anc, n)               # m=0 -> 本征值 +1
    return p0, sp, sm


ok_b = True
tests = ['X', 'Y', 'Z', 'XX', 'XZ', 'YY', 'ZZ', 'XZY', 'IZX', 'XYZI']
for pstr in tests:
    nd = len(pstr)
    P = pauli_op(pstr)
    for _ in range(3):
        psi = rand_state(2 ** nd)
        expP = np.real(psi.conj() @ P @ psi)
        p_plus = (1 + expP) / 2                          # 精确的 P(本征值 +1)
        p_sim, sp, sm = measure_pauli(psi, pstr)
        ok_b &= abs(p_sim - p_plus) < 1e-9
        # 测量后态必须落在 P 的对应本征子空间内
        for branch, val in ((sp, +1), (sm, -1)):
            rho = reduced_density(branch, list(range(nd)), nd + 1)
            w, vecs = np.linalg.eigh(rho)
            dom = vecs[:, -1]
            if w[-1] < 1 - 1e-9:                         # 辅助比特应当是干净的 |0>
                continue
            ok_b &= np.linalg.norm(P @ dom - val * dom) < 1e-7
print(f"  已测试 Pauli 串：{tests}")
print("  本征值读出与本征子空间一致性："
      + ("通过" if ok_b else "失败"))

# ============================================================
# (c) GHZ 型态 -> 单量子比特  alpha|0>+sqrt(1-a^2)|1>  （X 基测量）
# ============================================================
print("\n(c) GHZ → 单量子比特（基于测量，确定性）")


def ghz_state(n, alpha):
    v = np.zeros(2 ** n, dtype=complex)
    v[0] = alpha
    v[(1 << n) - 1] = np.sqrt(max(0.0, 1 - alpha ** 2))
    return v


def target1(a):
    return np.array([a, np.sqrt(max(0.0, 1 - a * a))], dtype=complex)


ok_c = True
for n in [2, 3, 4]:
    for alpha in [0.1, 0.3, 0.5, 0.8, 0.95]:
        psi = ghz_state(n, alpha)
        s = psi.copy()
        for i in range(1, n):
            s = op1(H, i, n) @ s                       # 在 X 基下测量量子比特 1..n-1
        rest = list(range(1, n))
        probs_ok, fids = [], []
        for outcome in range(2 ** (n - 1)):
            mask = 0
            for j, q in enumerate(rest):
                if (outcome >> j) & 1:
                    mask |= (1 << q)
            vec = np.zeros(2, dtype=complex)
            for idx in range(2 ** n):
                if (idx & ~1) == mask:                 # 量子比特 0 自由，其余须匹配 mask
                    vec[idx & 1] += s[idx]
            p_out = np.linalg.norm(vec) ** 2
            vec = vec / np.linalg.norm(vec)
            s_parity = bin(outcome).count('1') % 2
            if s_parity:
                vec = Z @ vec                          # 经典前馈 Z^s
            fids.append(abs(target1(alpha).conj() @ vec))
            probs_ok.append(p_out)
        total_p = sum(probs_ok)
        min_fid = min(fids)
        ok_c &= abs(total_p - 1.0) < 1e-9 and min_fid > 1 - 1e-9
    print(f"  n={n}：各结果概率之和={total_p:.6f}，"
          f"相对目标态的最小保真度={min_fid:.10f}")
print("  (c) 通过" if ok_c else "  (c) 失败")

# ============================================================
# (e) 计算 |<psi|phi>| 的 SWAP 检验
# ============================================================
print("\n(e) SWAP 检验： P(c=0) = (1 + |<ψ|φ>|²) / 2")


def swap_test(psi, phi):
    c, a, b = 0, 1, 2
    sw = CNOT(a, b, 3) @ CNOT(b, a, 3) @ CNOT(a, b, 3)
    cswap = op1(P0, c, 3) + op1(P1, c, 3) @ sw
    U = op1(H, c, 3) @ cswap @ op1(H, c, 3)
    full = np.zeros(8, dtype=complex)                   # |0>_c |psi>_a |phi>_b
    for i in (0, 1):
        for j in (0, 1):
            full[2 * i + 4 * j] = psi[i] * phi[j]
    s = U @ full
    p0, _, _ = measure_qubit(s, c, 3)
    return p0


ok_e = True
for _ in range(5):
    psi = rand_state(2)
    phi = rand_state(2)
    p0_sim = swap_test(psi, phi)
    p0_exact = (1 + abs(phi.conj() @ psi) ** 2) / 2
    ok_e &= abs(p0_sim - p0_exact) < 1e-9
p0_orth = swap_test(np.array([1, 0], complex), np.array([0, 1], complex))
p0_same = swap_test(np.array([1, 0], complex), np.array([1, 0], complex))
print(f"  随机测试均吻合；正交态 → P(c=0)={p0_orth:.6f}（=0.5），"
      f"相同态 → P(c=0)={p0_same:.6f}（=1.0）")
print("  (e) 通过" if ok_e and abs(p0_orth - 0.5) < 1e-9 and abs(p0_same - 1) < 1e-9
      else "  (e) 失败")

print("\n" + "=" * 64)
print("总计：", "全部通过" if (ok_a and ok_b and ok_c and ok_e) else "存在失败")
print("=" * 64)
