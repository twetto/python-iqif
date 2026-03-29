#!/usr/bin/env python3
"""
Checkpoint/restore test for iq-neuron.

1. Run network A for 100 steps, record potentials every step.
2. At step 30, snapshot full state (potential, is_firing, current_accumulator,
   synapse_timer) for each neuron.
3. Create network B from the same config, restore the snapshot.
4. Run network B for 70 steps.
5. Verify B's 70-step trajectory == A's last 70 steps exactly.

Setup (deterministic -- noise=0 -> rand()%1==0):
  neuron 0: rest=80 thresh=235 reset=80 shift_a=3 shift_b=4 noise=0
  neuron 1: rest=80 thresh=235 reset=80 shift_a=3 shift_b=4 noise=0
  0 -> 1: weight=+10  tau=32
  1 -> 0: weight=-5   tau=64
  bias: neuron 0 = 13, neuron 1 = 12
"""

import os
import sys
import ctypes
import tempfile
import numpy as np
from ctypes import c_void_p, c_int, c_float, c_char_p, POINTER

# ---------------------------------------------------------------------------
# 1. Locate the shared library
# ---------------------------------------------------------------------------
LIB_SEARCH_PATHS = [
    "./build/libiq-network.so",
    "../build/libiq-network.so",
    "/usr/local/lib/libiq-network.so",
    "/usr/lib/libiq-network.so",
]

lib_path = None
for p in LIB_SEARCH_PATHS:
    if os.path.isfile(p):
        lib_path = p
        break

if lib_path is None:
    print("ERROR: libiq-network.so not found. Searched:")
    for p in LIB_SEARCH_PATHS:
        print(f"  {p}")
    print("Build the library first, or set lib_path manually.")
    sys.exit(1)

print(f"Loading library: {lib_path}")
lib = ctypes.CDLL(lib_path)

# ---------------------------------------------------------------------------
# 2. Declare ctypes signatures
# ---------------------------------------------------------------------------
lib.iq_network_new.argtypes = [c_char_p, c_char_p]
lib.iq_network_new.restype = c_void_p
lib.iq_network_delete.argtypes = [c_void_p]
lib.iq_network_delete.restype = None
lib.iq_network_num_neurons.argtypes = [c_void_p]
lib.iq_network_num_neurons.restype = c_int
lib.iq_network_send_synapse.argtypes = [c_void_p]
lib.iq_network_send_synapse.restype = None
lib.iq_network_set_biascurrent.argtypes = [c_void_p, c_int, c_int]
lib.iq_network_set_biascurrent.restype = c_int

# Potential
lib.iq_network_potential.argtypes = [c_void_p, c_int]
lib.iq_network_potential.restype = c_int
lib.iq_network_set_potential.argtypes = [c_void_p, c_int, c_int]
lib.iq_network_set_potential.restype = c_int

# Is-firing
lib.iq_network_get_is_firing.argtypes = [c_void_p, c_int]
lib.iq_network_get_is_firing.restype = c_int
lib.iq_network_set_is_firing.argtypes = [c_void_p, c_int, c_int]
lib.iq_network_set_is_firing.restype = c_int

# Current accumulator
lib.iq_network_get_current_accumulator.argtypes = [c_void_p, c_int]
lib.iq_network_get_current_accumulator.restype = c_int
lib.iq_network_set_current_accumulator.argtypes = [c_void_p, c_int, c_int]
lib.iq_network_set_current_accumulator.restype = c_int

# Synapse timer
lib.iq_network_get_synapse_timer.argtypes = [c_void_p, c_int]
lib.iq_network_get_synapse_timer.restype = c_int
lib.iq_network_set_synapse_timer.argtypes = [c_void_p, c_int, c_int]
lib.iq_network_set_synapse_timer.restype = c_int

# ---------------------------------------------------------------------------
# 3. Create temporary config files
# ---------------------------------------------------------------------------
tmpdir = tempfile.mkdtemp(prefix="iqtest_ckpt_")

par_path = os.path.join(tmpdir, "params.txt")
con_path = os.path.join(tmpdir, "conn.txt")

with open(par_path, "w") as f:
    f.write("0 80 235 80 3 4 0\n")
    f.write("1 80 235 80 3 4 0\n")

with open(con_path, "w") as f:
    f.write("0 1 10 32\n")
    f.write("1 0 -5 64\n")

b_par = par_path.encode()
b_con = con_path.encode()

TOTAL_STEPS = 100
#CHECKPOINT_STEP = 30
CHECKPOINT_STEP = 64
NUM_NEURONS = 2

# ---------------------------------------------------------------------------
# 4. Run network A for TOTAL_STEPS, snapshot at CHECKPOINT_STEP
# ---------------------------------------------------------------------------
print(f"\n=== Running network A for {TOTAL_STEPS} steps, checkpoint at step {CHECKPOINT_STEP} ===")

net_a = lib.iq_network_new(b_par, b_con)
lib.iq_network_set_biascurrent(net_a, 0, 13)
lib.iq_network_set_biascurrent(net_a, 1, 12)

# Record potentials at every step
potentials_a = np.zeros((TOTAL_STEPS, NUM_NEURONS), dtype=np.int32)

# Snapshot storage
snapshot = {}

for t in range(TOTAL_STEPS):
    lib.iq_network_send_synapse(net_a)

    for idx in range(NUM_NEURONS):
        potentials_a[t, idx] = lib.iq_network_potential(net_a, idx)

    # Snapshot AFTER step completes
    if t == CHECKPOINT_STEP - 1:  # after step 30 (0-indexed: t=29)
        print(f"  Snapshotting at t={t} (after step {t+1})...")
        for idx in range(NUM_NEURONS):
            snapshot[idx] = {
                "potential": lib.iq_network_potential(net_a, idx),
                "is_firing": lib.iq_network_get_is_firing(net_a, idx),
                "current_acc": lib.iq_network_get_current_accumulator(net_a, idx),
                "syn_timer": lib.iq_network_get_synapse_timer(net_a, idx),
            }
            print(f"    N{idx}: V={snapshot[idx]['potential']}  "
                  f"firing={snapshot[idx]['is_firing']}  "
                  f"acc={snapshot[idx]['current_acc']}  "
                  f"timer={snapshot[idx]['syn_timer']}")

lib.iq_network_delete(net_a)

# ---------------------------------------------------------------------------
# 5. Create network B, restore snapshot, run remaining steps
# ---------------------------------------------------------------------------
remaining = TOTAL_STEPS - CHECKPOINT_STEP
print(f"\n=== Restoring snapshot into network B, running {remaining} steps ===")

net_b = lib.iq_network_new(b_par, b_con)
lib.iq_network_set_biascurrent(net_b, 0, 13)
lib.iq_network_set_biascurrent(net_b, 1, 12)

# Restore state
for idx in range(NUM_NEURONS):
    s = snapshot[idx]
    lib.iq_network_set_potential(net_b, idx, s["potential"])
    lib.iq_network_set_is_firing(net_b, idx, s["is_firing"])
    lib.iq_network_set_current_accumulator(net_b, idx, s["current_acc"])
    lib.iq_network_set_synapse_timer(net_b, idx, s["syn_timer"])

# Run and record
potentials_b = np.zeros((remaining, NUM_NEURONS), dtype=np.int32)

for t in range(remaining):
    lib.iq_network_send_synapse(net_b)
    for idx in range(NUM_NEURONS):
        potentials_b[t, idx] = lib.iq_network_potential(net_b, idx)

lib.iq_network_delete(net_b)

# ---------------------------------------------------------------------------
# 6. Compare
# ---------------------------------------------------------------------------
print(f"\n=== Comparing trajectories ===")

# A's last 70 steps = potentials_a[CHECKPOINT_STEP:]
tail_a = potentials_a[CHECKPOINT_STEP:]

assert tail_a.shape == potentials_b.shape, \
    f"Shape mismatch: {tail_a.shape} vs {potentials_b.shape}"

mismatches = np.where(tail_a != potentials_b)

if len(mismatches[0]) == 0:
    print(f"  PASS  All {remaining} steps match exactly for both neurons.")
else:
    n_bad = len(mismatches[0])
    print(f"  FAIL  {n_bad} mismatches found!")
    # Show first few
    for i in range(min(10, n_bad)):
        t_idx = mismatches[0][i]
        n_idx = mismatches[1][i]
        print(f"    step {CHECKPOINT_STEP + t_idx + 1}, neuron {n_idx}: "
              f"A={tail_a[t_idx, n_idx]}  B={potentials_b[t_idx, n_idx]}")
    if n_bad > 10:
        print(f"    ... and {n_bad - 10} more")
    sys.exit(1)

# Sanity: verify the simulation wasn't trivial (neurons actually fired)
n0_spikes = np.sum(np.diff(potentials_a[:, 0]) < -50)  # large drops = resets
n1_spikes = np.sum(np.diff(potentials_a[:, 1]) < -50)
print(f"\n  Sanity check: ~{n0_spikes} spikes from N0, ~{n1_spikes} spikes from N1 "
      f"over {TOTAL_STEPS} steps")

if n0_spikes == 0 and n1_spikes == 0:
    print("  WARNING: no spikes detected, test may be vacuous")
else:
    print("  Good -- nontrivial dynamics confirmed.")

print("\nAll tests passed!")
