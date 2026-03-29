#!/usr/bin/env python3
"""
Test script for current_accumulator get/set via ctypes.

Setup:
  neuron 0: rest=80 thresh=235 reset=80 shift_a=3 shift_b=4 noise=0
  neuron 1: rest=80 thresh=235 reset=80 shift_a=3 shift_b=4 noise=0
  connection 0->1: weight=+10  tau=32  (excitatory)
  connection 1->0: weight=-5   tau=64  (inhibitory)
  bias current: neuron 0 = 13, neuron 1 = 12
  surrogate tau: 8 (default)

Expected behavior:
  - No equilibrium exists for either neuron, so both ramp up and fire.
  - Neuron 0 fires first (~20 steps), depositing +10 into neuron 1's
    accumulator. Neuron 1 fires slightly later, depositing -5 into
    neuron 0's accumulator.
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
# Construction / destruction
lib.iq_network_new.argtypes = [c_char_p, c_char_p]
lib.iq_network_new.restype = c_void_p
lib.iq_network_delete.argtypes = [c_void_p]
lib.iq_network_delete.restype = None

# Basic accessors
lib.iq_network_num_neurons.argtypes = [c_void_p]
lib.iq_network_num_neurons.restype = c_int
lib.iq_network_send_synapse.argtypes = [c_void_p]
lib.iq_network_send_synapse.restype = None
lib.iq_network_set_biascurrent.argtypes = [c_void_p, c_int, c_int]
lib.iq_network_set_biascurrent.restype = c_int
lib.iq_network_potential.argtypes = [c_void_p, c_int]
lib.iq_network_potential.restype = c_int
lib.iq_network_spike_count.argtypes = [c_void_p, c_int]
lib.iq_network_spike_count.restype = c_int

# --- Functions under test ---
# Per-neuron get (existing)
lib.iq_network_get_current_accumulator.argtypes = [c_void_p, c_int]
lib.iq_network_get_current_accumulator.restype = c_int
# Per-neuron set (new)
lib.iq_network_set_current_accumulator.argtypes = [c_void_p, c_int, c_int]
lib.iq_network_set_current_accumulator.restype = c_int
# Bulk get (new)
lib.iq_network_get_all_current_accumulators.argtypes = [c_void_p, POINTER(c_int)]
lib.iq_network_get_all_current_accumulators.restype = None
# Bulk set (new)
lib.iq_network_set_all_current_accumulators.argtypes = [c_void_p, POINTER(c_int)]
lib.iq_network_set_all_current_accumulators.restype = None


# ---------------------------------------------------------------------------
# 3. Create temporary config files
# ---------------------------------------------------------------------------
tmpdir = tempfile.mkdtemp(prefix="iqtest_")

par_path = os.path.join(tmpdir, "params.txt")
con_path = os.path.join(tmpdir, "conn.txt")

with open(par_path, "w") as f:
    # index  rest  threshold  reset  shift_a  shift_b  noise
    f.write("0 80 235 80 3 4 0\n")
    f.write("1 80 235 80 3 4 0\n")

with open(con_path, "w") as f:
    # pre  post  weight  tau
    f.write("0 1 10 32\n")
    f.write("1 0 -5 64\n")

print(f"Config files in: {tmpdir}")


# ---------------------------------------------------------------------------
# 4. Helper
# ---------------------------------------------------------------------------
def bulk_get(net, n):
    """Return current_accumulators as a numpy array."""
    buf = np.zeros(n, dtype=np.int32)
    lib.iq_network_get_all_current_accumulators(net, buf.ctypes.data_as(POINTER(c_int)))
    return buf

def bulk_set(net, arr):
    """Write a numpy array into current_accumulators."""
    a = np.asarray(arr, dtype=np.int32)
    lib.iq_network_set_all_current_accumulators(net, a.ctypes.data_as(POINTER(c_int)))


# ---------------------------------------------------------------------------
# 5. Tests
# ---------------------------------------------------------------------------
passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}")
        failed += 1


# --- Test A: per-neuron set/get round-trip ---
print("\n=== Test A: per-neuron set/get round-trip ===")
net = lib.iq_network_new(par_path.encode(), con_path.encode())
n = lib.iq_network_num_neurons(net)
check("num_neurons == 2", n == 2)

lib.iq_network_set_current_accumulator(net, 0, 42)
lib.iq_network_set_current_accumulator(net, 1, -17)
check("get[0] == 42", lib.iq_network_get_current_accumulator(net, 0) == 42)
check("get[1] == -17", lib.iq_network_get_current_accumulator(net, 1) == -17)

# Out-of-range returns 0 (failure)
ret = lib.iq_network_set_current_accumulator(net, 99, 1)
check("set out-of-range returns 0", ret == 0)
lib.iq_network_delete(net)


# --- Test B: bulk set/get round-trip ---
print("\n=== Test B: bulk set/get round-trip ===")
net = lib.iq_network_new(par_path.encode(), con_path.encode())

bulk_set(net, [100, -200])
vals = bulk_get(net, 2)
check("bulk get[0] == 100", vals[0] == 100)
check("bulk get[1] == -200", vals[1] == -200)

lib.iq_network_delete(net)


# --- Test C: bulk and per-neuron agree ---
print("\n=== Test C: bulk and per-neuron consistency ===")
net = lib.iq_network_new(par_path.encode(), con_path.encode())

lib.iq_network_set_current_accumulator(net, 0, 7)
lib.iq_network_set_current_accumulator(net, 1, -3)
vals = bulk_get(net, 2)
check("per-neuron set -> bulk get[0]", vals[0] == 7)
check("per-neuron set -> bulk get[1]", vals[1] == -3)

bulk_set(net, [55, 66])
check("bulk set -> per-neuron get[0]", lib.iq_network_get_current_accumulator(net, 0) == 55)
check("bulk set -> per-neuron get[1]", lib.iq_network_get_current_accumulator(net, 1) == 66)

lib.iq_network_delete(net)


# --- Test D: simulation run, observe accumulator after spike ---
print("\n=== Test D: simulation with spike propagation ===")
net = lib.iq_network_new(par_path.encode(), con_path.encode())

lib.iq_network_set_biascurrent(net, 0, 13)
lib.iq_network_set_biascurrent(net, 1, 12)

spike_times = {0: [], 1: []}
acc_log = {0: [], 1: []}
potential_log = {0: [], 1: []}
steps = 100

for t in range(steps):
    lib.iq_network_send_synapse(net)

    for idx in range(2):
        potential_log[idx].append(lib.iq_network_potential(net, idx))
        acc_log[idx].append(lib.iq_network_get_current_accumulator(net, idx))

    # spike_count resets on read, so nonzero means spike this step
    for idx in range(2):
        sc = lib.iq_network_spike_count(net, idx)
        if sc > 0:
            spike_times[idx].append(t)

print(f"  Neuron 0 spike times: {spike_times[0]}")
print(f"  Neuron 1 spike times: {spike_times[1]}")

check("neuron 0 fired at least once", len(spike_times[0]) > 0)
check("neuron 1 fired at least once", len(spike_times[1]) > 0)

# After neuron 0 fires, neuron 1's accumulator should have received +10
if spike_times[0]:
    t0 = spike_times[0][0]
    if t0 + 1 < steps:
        acc1_after = acc_log[1][t0 + 1]
        print(f"  Neuron 1 accumulator at t={t0+1} (after N0 first spike): {acc1_after}")
        check("neuron 1 accumulator nonzero after N0 spike", acc1_after != 0)

lib.iq_network_delete(net)


# --- Test E: set accumulator mid-simulation ---
print("\n=== Test E: inject current via set_current_accumulator ===")
net = lib.iq_network_new(par_path.encode(), con_path.encode())

# No bias current -- neuron sits at rest
# Inject a large value into the accumulator and see if it affects potential
v_before = lib.iq_network_potential(net, 0)
lib.iq_network_set_current_accumulator(net, 0, 50)
lib.iq_network_send_synapse(net)
v_after = lib.iq_network_potential(net, 0)

print(f"  V before inject: {v_before}, V after one step with acc=50: {v_after}")
check("potential changed after accumulator injection", v_after > v_before)

lib.iq_network_delete(net)


# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("All tests passed!")
else:
    print("Some tests FAILED.")
    sys.exit(1)
