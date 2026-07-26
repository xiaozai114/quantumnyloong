"""
Q03 (a): Generate noisy bitstrings, implement configuration recovery, verify correctness.
H2 HF bitstring |0011>, noise flip probability 30%, sample count S=1000.
"""

import numpy as np

# ---- H2 parameters ----
hf_bitstring = np.array([0, 0, 1, 1])  # |0011>
K = 4        # number of qubits
Ne = 2       # number of electrons
S = 1000     # sample count
p = 0.30     # noise flip probability

np.random.seed(42)

# ---- Step 1: Generate S noisy bitstrings ----
bs = np.tile(hf_bitstring, (S, 1))
flips = np.random.random((S, K)) < p
bs[flips] = 1 - bs[flips]

print(f"True HF bitstring: {hf_bitstring}")
print(f"Expected n_bar:    {(1-p)*hf_bitstring + p*(1-hf_bitstring)}")

# ---- Step 2: Compute average occupancy ----
n_bar = np.mean(bs, axis=0)
print(f"Actual n_bar:      {n_bar}")

# ---- Step 3: Count violating bitstrings ----
violating = np.sum(np.sum(bs, axis=1) != Ne)
print(f"Violating bitstrings: {violating}/{S} ({violating/S*100:.1f}%)")

# ---- Step 4: Configuration recovery ----
# For each violating bitstring, flip the bit with largest |b_i - n_bar_i|
# until particle number constraint sum(d_i) = Ne is satisfied.
recovered = bs.copy()
for s in range(S):
    d = recovered[s].copy()
    while np.sum(d) != Ne:
        if np.sum(d) > Ne:
            # Too many electrons: flip 1->0, pick largest |b_i - n_bar_i| among d_i=1
            candidates = np.where(d == 1)[0]
            scores = np.abs(d[candidates] - n_bar[candidates])
            flip_idx = candidates[np.argmax(scores)]
            d[flip_idx] = 0
        else:
            # Too few electrons: flip 0->1, pick largest |b_i - n_bar_i| among d_i=0
            candidates = np.where(d == 0)[0]
            scores = np.abs(d[candidates] - n_bar[candidates])
            flip_idx = candidates[np.argmax(scores)]
            d[flip_idx] = 1
    recovered[s] = d

# ---- Step 5: Verify ----
all_satisfied = np.all(np.sum(recovered, axis=1) == Ne)
rate_before = np.mean(np.all(bs == hf_bitstring, axis=1))
rate_after = np.mean(np.all(recovered == hf_bitstring, axis=1))

print(f"\nAll bitstrings satisfy Ne=2 after recovery: {all_satisfied}")
print(f"Success rate before recovery: {rate_before:.3f}")
print(f"Success rate after recovery:  {rate_after:.3f}")
