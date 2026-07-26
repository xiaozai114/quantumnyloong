"""
Q03: Configuration Recovery
(a) Generate noisy bitstrings, implement recovery, verify correctness
(c) Plot recovery success rate vs noise rate, find failure threshold
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- H2 parameters ----
hf_bitstring = np.array([0, 0, 1, 1])  # |0011>
K = 4        # number of qubits
Ne = 2       # number of electrons
S = 1000     # sample count

np.random.seed(42)

def generate_noisy_bitstrings(hf_bs, p, S):
    """Generate S noisy bitstrings by independently flipping each bit with probability p."""
    bs = np.tile(hf_bs, (S, 1))
    flips = np.random.random((S, K)) < p
    bs[flips] = 1 - bs[flips]
    return bs

def compute_avg_occupancy(bitstrings):
    """Compute average occupancy n_bar_i."""
    return np.mean(bitstrings, axis=0)

def configuration_recovery(bitstrings, n_bar, Ne):
    """
    Greedy configuration recovery:
    For each violating bitstring, flip the bit with largest |b_i - n_bar_i|
    until particle number constraint is satisfied.
    """
    recovered = bitstrings.copy()
    for s in range(len(recovered)):
        d = recovered[s].copy()
        while np.sum(d) != Ne:
            if np.sum(d) > Ne:
                # Need to remove electrons: flip 1->0
                candidates = np.where(d == 1)[0]
                scores = np.abs(d[candidates] - n_bar[candidates])
                flip_idx = candidates[np.argmax(scores)]
                d[flip_idx] = 0
            else:
                # Need to add electrons: flip 0->1
                candidates = np.where(d == 0)[0]
                scores = np.abs(d[candidates] - n_bar[candidates])
                flip_idx = candidates[np.argmax(scores)]
                d[flip_idx] = 1
        recovered[s] = d
    return recovered

def success_rate(recovered, hf_bs):
    """Fraction of recovered bitstrings matching the true HF state."""
    return np.mean(np.all(recovered == hf_bs, axis=1))

# ---- (a) Single run with p=0.3 ----
p = 0.3
noisy = generate_noisy_bitstrings(hf_bitstring, p, S)
n_bar = compute_avg_occupancy(noisy)

print(f"=== (a) p={p}, S={S} ===")
print(f"True HF bitstring: {hf_bitstring}")
print(f"Expected n_bar:    {(1-p)*hf_bitstring + p*(1-hf_bitstring)}")
print(f"Actual n_bar:      {n_bar}")

# Count violating bitstrings
violating = np.sum(np.sum(noisy, axis=1) != Ne)
print(f"Violating bitstrings: {violating}/{S} ({violating/S*100:.1f}%)")

# Recovery
recovered = configuration_recovery(noisy, n_bar, Ne)
rate_before = np.mean(np.all(noisy == hf_bitstring, axis=1))
rate_after = success_rate(recovered, hf_bitstring)
print(f"Success rate before recovery: {rate_before:.3f}")
print(f"Success rate after recovery:  {rate_after:.3f}")

# ---- (c) Sweep noise rate ----
noise_rates = np.arange(0.10, 0.51, 0.05)
success_rates = []

print(f"\n=== (c) Noise sweep ===")
for p in noise_rates:
    noisy = generate_noisy_bitstrings(hf_bitstring, p, S)
    n_bar = compute_avg_occupancy(noisy)
    recovered = configuration_recovery(noisy, n_bar, Ne)
    rate = success_rate(recovered, hf_bitstring)
    success_rates.append(rate)
    print(f"  p={p:.2f}: success_rate={rate:.3f}")

success_rates = np.array(success_rates)

# Find failure threshold (50% success rate)
from numpy import interp
p_crit = interp(0.5, success_rates[::-1], noise_rates[::-1])
print(f"\nFailure threshold p_crit (50% success): {p_crit:.3f}")

# ---- Plot ----
plt.figure(figsize=(8, 5))
plt.plot(noise_rates, success_rates, 'ro-', linewidth=2, markersize=8)
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='50% threshold')
plt.axvline(x=p_crit, color='blue', linestyle='--', alpha=0.7, label=f'$p_{{crit}}$ = {p_crit:.2f}')
plt.xlabel('Noise flip probability $p$', fontsize=13)
plt.ylabel('Recovery success rate', fontsize=13)
plt.title('Q03: Configuration Recovery Success Rate vs Noise (H$_2$, $S$=1000)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig('recovery_success_rate.png', dpi=150)
print(f"\nPlot saved to recovery_success_rate.png")
