"""
Problem 1, Question 8: Grover's Search Algorithm
=================================================
Using the reversible circuit from Problem 7 (QuantumCircuits.build_s3),
find x = (x1, x2, x3) satisfying S3(x) = True with Grover's algorithm.

S3(x) = (not x1 or not x2) and (x1 or not x2) and (not x1 or x2 or not x3)
      and (not x1 or x2 or x3) and (x1 or x2 or x3)
    = not x1 and not x2 and x3

Satisfying assignment: |001> (x1=0, x2=0, x3=1)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

# Patch numpy 2.5+ compatibility for tensorcircuit
# (np.reshape no longer accepts `newshape` keyword)
_orig_reshape = np.reshape
def _patched_reshape(a, newshape=None, shape=None, **kwargs):
    if newshape is not None and shape is None:
        shape = newshape
    if shape is not None:
        return _orig_reshape(a, shape, **kwargs)
    return _orig_reshape(a, **kwargs)
np.reshape = _patched_reshape

# np.ComplexWarning was removed in numpy 2.0+
if not hasattr(np, 'ComplexWarning'):
    np.ComplexWarning = np.exceptions.ComplexWarning

import tensorcircuit as tc
from lib.quantum_circuits import QuantumCircuits

K = tc.set_backend("numpy")


def build_grover_oracle(c):
    """
    Oracle for S3 using the reversible circuit from Problem 7.

    Qubit layout (5 qubits):
        q0, q1, q2 : data qubits (x1, x2, x3)
        q3         : ancilla (from Problem 7, clean |0> after oracle)
        q4         : result qubit (from Problem 7, clean |0> after oracle)

    Strategy:
        1. Compute S3 into q4 using QuantumCircuits.build_s3
        2. Apply Z on q4 (phase flip when S3=1)
        3. Uncompute S3 (apply build_s3 again, self-inverse)

    After the oracle, all ancillas return to |0>.
    """
    QuantumCircuits.build_s3(c)
    c.z(4)
    QuantumCircuits.build_s3(c)
    return c


def build_grover_diffusion(c):
    """
    Grover diffusion operator on 3 data qubits:
        D = H^3 . (2|000><000| - I) . H^3

    Implementation:
        H^3 . X^3 . MCZ(q0,q1,q2) . X^3 . H^3

    MCZ uses q3 as ancilla (clean |0> after oracle):
        Toffoli(q0,q1,q3); H(q2); CNOT(q3,q2); H(q2); Toffoli(q0,q1,q3)
    """
    for i in range(3):
        c.h(i)
    for i in range(3):
        c.x(i)

    # Multi-controlled-Z on (q0, q1, q2) using q3 as ancilla
    c.toffoli(0, 1, 3)
    c.h(2)
    c.cnot(3, 2)
    c.h(2)
    c.toffoli(0, 1, 3)

    for i in range(3):
        c.x(i)
    for i in range(3):
        c.h(i)

    return c


def build_grover_circuit(n_iterations=1):
    """
    Build the complete Grover search circuit.

    Total qubits: 5 (same as Problem 7 reversible circuit)
    """
    c = tc.Circuit(5)

    # Step 1: Initialize uniform superposition on data qubits
    for i in range(3):
        c.h(i)

    # Step 2: Apply Grover iterations (oracle + diffusion)
    for _ in range(n_iterations):
        c = build_grover_oracle(c)
        c = build_grover_diffusion(c)

    return c


def run_grover(n_iterations=1):
    """Run Grover's algorithm and return measurement results."""
    c = build_grover_circuit(n_iterations)

    # Get state vector and extract probabilities
    state = c.state()
    probs = np.abs(np.asarray(state)) ** 2  # shape (2,2,2,2,2)

    # Marginalize over ancilla qubits (q3, q4) to get data qubit probabilities
    # Reshape state tensor: indices are (q0, q1, q2, q3, q4)
    probs_reshaped = probs.reshape(2, 2, 2, 2, 2)
    data_probs = probs_reshaped.sum(axis=(3, 4))  # sum over q3, q4

    return data_probs


def verify():
    """Verify Grover's algorithm finds the correct solution."""
    n_qubits = 3
    N = 2 ** n_qubits  # 8
    M = 1              # number of solutions

    theta = np.arcsin(np.sqrt(M / N))
    k_opt = int(np.floor(np.arccos(np.sqrt(M / N)) / (2 * theta)))

    print("=" * 55)
    print("Grover Search for S3(x) = True")
    print("=" * 55)
    print(f"  N = {N}, M = {M}")
    print(f"  theta = arcsin(sqrt(M/N)) = {theta:.6f} rad")
    print(f"  Optimal iterations k = {k_opt}")
    print(f"  Qubits: 5 (3 data + 2 ancilla from Problem 7)")
    print()

    for k in range(k_opt + 2):
        data_probs = run_grover(n_iterations=k)

        # Print probability for each data state
        print(f"--- After {k} Grover iteration(s) ---")
        best_idx = None
        best_prob = 0.0
        for i in range(2):
            for j in range(2):
                for l in range(2):
                    idx = (i, j, l)
                    p = data_probs[idx]
                    if p > best_prob:
                        best_prob = p
                        best_idx = idx
                    tag = "  <-- SOLUTION" if (i, j, l) == (0, 0, 1) else ""
                    print(f"  |{i}{j}{l}> : {p:.4f} ({p*100:.1f}%){tag}")
        print(f"  Best: |{best_idx[0]}{best_idx[1]}{best_idx[2]}> with P = {best_prob:.4f}")
        print()

    # Final verification
    data_probs = run_grover(n_iterations=k_opt)
    final_idx = np.unravel_index(np.argmax(data_probs), data_probs.shape)
    final_prob = data_probs[final_idx]

    print("=" * 55)
    print("Verification")
    print("=" * 55)
    print(f"  Grover (k={k_opt}) most probable state: |{final_idx[0]}{final_idx[1]}{final_idx[2]}>")
    print(f"  Probability: {final_prob:.4f} ({final_prob*100:.1f}%)")

    x1, x2, x3 = final_idx
    c1 = (not x1) or (not x2)
    c2 = x1 or (not x2)
    c3 = (not x1) or x2 or (not x3)
    c4 = (not x1) or x2 or x3
    c5 = x1 or x2 or x3
    result = c1 and c2 and c3 and c4 and c5
    print(f"  S3({x1}, {x2}, {x3}) = {result}")
    assert result, "Grover failed to find the satisfying assignment!"
    print(f"  Grover search SUCCESS!")


if __name__ == "__main__":
    verify()
