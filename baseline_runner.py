# baseline_runner.py

from pytket.passes import FullPeepholeOptimise, SynthesiseTket, SequencePass
from pytket.circuit import Circuit
import pandas as pd

class CircuitStats:
    """
    Statistics for a Circuit.
    """
    def __init__(self, circ: Circuit):
        self.n_gates = circ.n_gates
        self.depth = circ.depth()
        self.two_qubit_gates = circ.n_2qb_gates()

# Import your circuits
from benchmarks.GHZ import GHZ_4, GHZ_8
from benchmarks.QAOA import QAOA_ring_4_p1, QAOA_ring_8_p2
from benchmarks.HEA import HEA_4_d2, HEA_6_d4
from benchmarks.RAND import RAND_4_d10, RAND_6_d20

# -----------------------------
# Define baseline TKET pipeline
# -----------------------------
baseline = SequencePass([
    FullPeepholeOptimise(),
    SynthesiseTket(),
])

# -----------------------------
# Circuit list
# -----------------------------
circuits = [
    ("GHZ_4", GHZ_4()),
    ("GHZ_8", GHZ_8()),
    ("QAOA_ring_4_p1", QAOA_ring_4_p1()),
    ("QAOA_ring_8_p2", QAOA_ring_8_p2()),
    ("HEA_4_d2", HEA_4_d2()),
    ("HEA_6_d4", HEA_6_d4()),
    ("RAND_4_d10", RAND_4_d10()),
    ("RAND_6_d20", RAND_6_d20()),
]

# -----------------------------
# Run baseline evaluation
# -----------------------------
results = []

for name, circ in circuits:
    print(f"Running baseline TKET on {name}...")

    # Record before-stats
    stats_before = CircuitStats(circ)

    # Compile using baseline TKET
    baseline.apply(circ)

    # Record after-stats
    stats_after = CircuitStats(circ)

    results.append({
        "name": name,
        "n_qubits": circ.n_qubits,
        "depth_before": stats_before.depth,
        "depth_after": stats_after.depth,
        "gate_count_before": stats_before.n_gates,
        "gate_count_after": stats_after.n_gates,
        "two_qubit_before": stats_before.two_qubit_gates,
        "two_qubit_after": stats_after.two_qubit_gates,
    })

# -----------------------------
# Save results to CSV
# -----------------------------
df = pd.DataFrame(results)
df.to_csv("baseline_results.csv", index=False)

print("\n=== BASELINE RESULTS ===")
print(df)
print("\nSaved to baseline_results.csv")
