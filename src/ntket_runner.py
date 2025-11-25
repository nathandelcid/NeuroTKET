# neuro_runner.py

from pytket.passes import FullPeepholeOptimise, SynthesiseTket, SequencePass
from pytket.utils import CircuitStats
import pandas as pd

from GHZ import GHZ_4, GHZ_8
from QAOA import QAOA_ring_4_p1, QAOA_ring_8_p2
from HEA import HEA_4_d2, HEA_6_d4
from RAND import RAND_4_d10, RAND_6_d20

from neurotket_pass import NeuroTKETPass

neuro_pipeline = SequencePass([
    FullPeepholeOptimise(),
    NeuroTKETPass,      # <-- your ML slot
    FullPeepholeOptimise(),
    SynthesiseTket(),
])

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

results = []

for name, circ in circuits:
    print(f"Running NeuroTKET pipeline on {name}...")

    stats_before = CircuitStats(circ)
    neuro_pipeline.apply(circ)
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

df = pd.DataFrame(results)
df.to_csv("neuro_results.csv", index=False)

print("\n=== NEUROTKET PIPELINE RESULTS (CURRENTLY NO-OP) ===")
print(df)
print("\nSaved to neuro_results.csv")
