"""
Problem 3: QAOA for MaxCut Problem
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Graph Definitions
# ============================================================

# C4: 4-cycle
C4_edges = [(0,1), (1,2), (2,3), (0,3)]
C4_n = 4

# G6
G6_edges = [(0,1), (3,4), (2,5), (0,3), (4,5), (1,2), (1,4)]
G6_n = 6

# G9
G9_edges = [(0,1), (3,4), (7,8), (2,5), (0,3), (4,5),
            (6,7), (1,2), (4,7), (5,8), (3,6), (1,4)]
G9_n = 9

# ============================================================
# 1. MaxCut Objective Function
# ============================================================
print("=" * 60)
print("Problem 3: QAOA for MaxCut")
print("=" * 60)

def maxcut_value(bitstring, edges):
    """Compute cut value for a given bitstring assignment."""
    x = [int(b) for b in bitstring]
    return sum(1 for i, j in edges if x[i] != x[j])

def maxcut_hamiltonian_terms(edges, n_qubits):
    """Return list of (i,j) pairs that define Z_i Z_j terms.
    H = sum_{(i,j) in E} 0.5 * (I - Z_i Z_j)
    """
    return edges

def brute_force_maxcut(edges, n_qubits):
    """Find exact MaxCut by brute force."""
    max_val = 0
    best_configs = []
    for val in range(2**n_qubits):
        bitstr = format(val, f'0{n_qubits}b')
        cut = maxcut_value(bitstr, edges)
        if cut > max_val:
            max_val = cut
            best_configs = [bitstr]
        elif cut == max_val:
            best_configs.append(bitstr)
    return max_val, best_configs

# Compute exact solutions
for name, edges, n in [("C4", C4_edges, C4_n),
                         ("G6", G6_edges, G6_n),
                         ("G9", G9_edges, G9_n)]:
    mc, best = brute_force_maxcut(edges, n)
    print(f"{name}: MaxCut = {mc}, {len(best)} solution(s), e.g. {best[0]}")


# ============================================================
# 2. QAOA Circuit Simulation (Statevector, p=1)
# ============================================================
print("\n" + "=" * 60)
print("2. QAOA-MaxCut Warm-up (p=1)")
print("=" * 60)

# Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

def multi_kron(matrices):
    """Kronecker product of a list of matrices."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result

def single_qubit_gate(gate, target, n_qubits):
    """Apply single-qubit gate to target qubit."""
    ops = [I2] * n_qubits
    ops[target] = gate
    return multi_kron(ops)

def two_qubit_gate(gate, q1, q2, n_qubits):
    """Apply two-qubit diagonal-like gate ZZ interaction."""
    # RZZ gate representation
    ops = [I2] * n_qubits
    # For RZZ, we apply Z⊗Z on qubits q1,q2
    result = np.eye(2**n_qubits, dtype=complex)
    Z_full = multi_kron([Z if i in (q1,q2) else I2 for i in range(n_qubits)])
    return result

class QAOASimulator:
    """Statevector simulator for QAOA."""
    
    def __init__(self, n_qubits, edges):
        self.n = n_qubits
        self.edges = edges
    
    def initial_state(self):
        """|s> = |+>^⊗n"""
        plus = np.array([1, 1]) / np.sqrt(2)
        state = plus
        for _ in range(self.n - 1):
            state = np.kron(state, plus)
        return state
    
    def apply_rz(self, state, qubit, theta):
        """Apply RZ(theta) = e^{-i theta Z/2} to qubit."""
        H_z = single_qubit_gate(Z, qubit, self.n)
        return np.diag(np.exp(-1j * theta/2 * np.diag(H_z))) @ state
    
    def apply_rx(self, state, qubit, theta):
        """Apply RX(theta) = e^{-i theta X/2} to qubit."""
        H_x = single_qubit_gate(X, qubit, self.n)
        # e^{-i theta X/2} = cos(theta/2)I - i sin(theta/2)X
        I_full = np.eye(2**self.n)
        return (np.cos(theta/2) * I_full - 1j * np.sin(theta/2) * H_x) @ state
    
    def apply_rzz(self, state, q1, q2, theta):
        """Apply RZZ(theta) = e^{-i theta Z⊗Z/2}."""
        # Construct Z⊗Z on qubits q1,q2
        ops = [I2] * self.n
        ops[q1] = Z
        ops[q2] = Z
        H_zz = multi_kron(ops)
        return (np.cos(theta/2) * np.eye(2**self.n) -
                1j * np.sin(theta/2) * H_zz) @ state
    
    def apply_problem_layer(self, state, gamma):
        """U(C,gamma) = ∏_{(i,j)∈E} e^{-iγ (I-Z_i Z_j)/2} 
        = e^{-iγ|E|/2} * ∏_{(i,j)∈E} RZZ(γ)_{ij}
        """
        for i, j in self.edges:
            state = self.apply_rzz(state, i, j, gamma)
        return state
    
    def apply_mixer_layer(self, state, beta):
        """U(B,beta) = ∏_j e^{-iβ X_j} = ∏_j RX(2β)_j"""
        for j in range(self.n):
            state = self.apply_rx(state, j, 2*beta)
        return state
    
    def qaoa_circuit(self, gamma, beta, p=1):
        """Depth-p QAOA circuit."""
        state = self.initial_state()
        for layer in range(p):
            state = self.apply_problem_layer(state, gamma[layer])
            state = self.apply_mixer_layer(state, beta[layer])
        return state
    
    def expectation_value(self, gamma, beta, p=1):
        """Compute <γ,β|C|γ,β>."""
        if p == 1:
            g = [gamma] if np.isscalar(gamma) else gamma
            b = [beta] if np.isscalar(beta) else beta
        else:
            g, b = gamma, beta
        
        state = self.qaoa_circuit(g, b, p)
        
        # Compute expectation of each Z_i Z_j term
        exp_val = 0.0
        for i, j in self.edges:
            ops = [I2] * self.n
            ops[i] = Z
            ops[j] = Z
            H_zz = multi_kron(ops)
            exp_val += 0.5 * (1 - np.real(state.conj() @ H_zz @ state))
        
        return exp_val
    
    def sample_bitstrings(self, gamma, beta, n_shots=100, p=1):
        """Sample bitstrings from QAOA state."""
        if p == 1:
            g = [gamma] if np.isscalar(gamma) else gamma
            b = [beta] if np.isscalar(beta) else beta
        else:
            g, b = gamma, beta
        
        state = self.qaoa_circuit(g, b, p)
        probs = np.abs(state)**2
        
        # Sample
        indices = np.random.choice(2**self.n, size=n_shots, p=probs)
        bitstrings = [format(idx, f'0{self.n}b') for idx in indices]
        return bitstrings


# ============================================================
# Run QAOA on all three graphs (p=1)
# ============================================================

def optimize_qaoa(sim, p=1, method='COBYLA', n_restarts=5):
    """Optimize QAOA parameters."""
    best_result = None
    best_energy = -np.inf
    
    for restart in range(n_restarts):
        # Random initialization in [-π, π]
        x0 = np.random.uniform(-np.pi, np.pi, 2*p)
        
        def objective(params):
            g = params[:p]
            b = params[p:]
            return -sim.expectation_value(g, b, p)  # Minimize negative
        
        result = minimize(objective, x0, method=method,
                         options={'maxiter': 200, 'disp': False})
        
        energy = -result.fun
        if energy > best_energy:
            best_energy = energy
            best_result = result
    
    g_opt = best_result.x[:p]
    b_opt = best_result.x[p:]
    return best_energy, g_opt, b_opt

# Optimize each graph
for name, edges, n in [("C4", C4_edges, C4_n),
                         ("G6", G6_edges, G6_n),
                         ("G9", G9_edges, G9_n)]:
    sim = QAOASimulator(n, edges)
    exact_maxcut, _ = brute_force_maxcut(edges, n)
    
    energy, g_opt, b_opt = optimize_qaoa(sim, p=1, n_restarts=10)
    
    # Sample and get best cut
    bitstrings = sim.sample_bitstrings(g_opt[0], b_opt[0], n_shots=1000)
    best_cut = max(maxcut_value(bs, edges) for bs in bitstrings)
    
    print(f"\n{name}:")
    print(f"  Exact MaxCut: {exact_maxcut}")
    print(f"  QAOA <C>: {energy:.4f}")
    print(f"  QAOA Best Cut: {best_cut}")
    print(f"  Approx Ratio: {best_cut/exact_maxcut:.3f}")
    print(f"  Optimal params: γ={g_opt[0]:.4f}, β={b_opt[0]:.4f}")


# ============================================================
# 3. Effect of QAOA Depth p
# ============================================================
print("\n" + "=" * 60)
print("3. QAOA Depth p Analysis")
print("=" * 60)

for name, edges, n in [("C4", C4_edges, C4_n),
                         ("G6", G6_edges, G6_n),
                         ("G9", G9_edges, G9_n)]:
    sim = QAOASimulator(n, edges)
    exact_mc, _ = brute_force_maxcut(edges, n)
    
    print(f"\n{name} (Exact MaxCut: {exact_mc}):")
    for p in [1, 2, 3]:
        energy, g_opt, b_opt = optimize_qaoa(sim, p=p, n_restarts=5)
        bitstrings = sim.sample_bitstrings(g_opt, b_opt, n_shots=500, p=p)
        best_cut = max(maxcut_value(bs, edges) for bs in bitstrings)
        print(f"  p={p}: <C>={energy:.4f}, Best Cut={best_cut}, "
              f"Ratio={best_cut/exact_mc:.3f}")


# ============================================================
# 4. Initialization Strategies Comparison
# ============================================================
print("\n" + "=" * 60)
print("4. Initialization Strategy Comparison")
print("=" * 60)

def optimize_with_init(sim, p, init_strategy, method='COBYLA'):
    """Optimize with specific initialization strategy."""
    n_params = 2 * p
    
    if init_strategy == 'zero':
        x0 = np.zeros(n_params)
    elif init_strategy == 'random':
        x0 = np.random.uniform(-np.pi, np.pi, n_params)
    elif init_strategy == 'linear_ramp':
        # Annealing-inspired: gamma increases, beta decreases
        dt = 1.0 / p
        gamma_init = np.array([(l + 0.5) * dt * np.pi for l in range(p)])
        beta_init = np.array([(1 - (l + 0.5) * dt) * np.pi for l in range(p)])
        x0 = np.concatenate([gamma_init, beta_init])
    
    def objective(params):
        g = params[:p]
        b = params[p:]
        return -sim.expectation_value(g, b, p)
    
    result = minimize(objective, x0, method=method,
                     options={'maxiter': 200, 'disp': False})
    return -result.fun, result.nfev

sim_g6 = QAOASimulator(G6_n, G6_edges)
for strategy in ['zero', 'random', 'linear_ramp']:
    energies = []
    nfevs = []
    for _ in range(5):
        e, nf = optimize_with_init(sim_g6, p=2, init_strategy=strategy)
        energies.append(e)
        nfevs.append(nf)
    print(f"  {strategy:>12}: E={np.mean(energies):.4f}±{np.std(energies):.4f}, "
          f"NFEV={np.mean(nfevs):.0f}")


# ============================================================
# 5. Optimizer Comparison
# ============================================================
print("\n" + "=" * 60)
print("5. Optimizer Comparison on G6 (p=2)")
print("=" * 60)

def optimize_qaoa_general(sim, p, method):
    """General QAOA optimizer supporting multiple methods."""
    x0 = np.random.uniform(-np.pi, np.pi, 2*p)
    
    def objective(params):
        g = params[:p]
        b = params[p:]
        return -sim.expectation_value(g, b, p)
    
    try:
        if method in ['L-BFGS-B', 'BFGS']:
            result = minimize(objective, x0, method=method,
                            options={'maxiter': 200})
        else:
            result = minimize(objective, x0, method=method,
                            options={'maxiter': 200})
        return -result.fun
    except:
        return -np.inf

sim_g6 = QAOASimulator(G6_n, G6_edges)
for method in ['COBYLA', 'Nelder-Mead', 'Powell', 'L-BFGS-B']:
    energies = []
    for _ in range(5):
        e = optimize_qaoa_general(sim_g6, p=2, method=method)
        energies.append(e)
    print(f"  {method:>12}: E={np.mean(energies):.4f}±{np.std(energies):.4f}")
