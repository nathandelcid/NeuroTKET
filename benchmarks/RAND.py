import math
import random
from pytket.circuit import Circuit
from pytket.utils import CircuitStats

# RAND_4_d10
# RAND_6_d20

def random_circuit(n_qubits: int, depth: int,
                   seed: int | None = None) -> Circuit:
    """
    Random circuit with a simple universal-ish gate set: {H, X, Rz, CX}.
    Applies `depth` layers where each layer applies a single random gate.
    """
    if seed is not None:
        random.seed(seed)

    circ = Circuit(n_qubits)
    gate_types = ["H", "X", "Rz", "CX"]

    for _ in range(depth):
        gt = random.choice(gate_types)
        if gt in ["H", "X", "Rz"]:
            q = random.randrange(n_qubits)
            if gt == "H":
                circ.H(q)
            elif gt == "X":
                circ.X(q)
            elif gt == "Rz":
                angle = random.uniform(0, 2 * math.pi)
                circ.Rz(angle, q)
        else:  # "CX"
            c = random.randrange(n_qubits)
            t = random.randrange(n_qubits)
            if c != t:
                circ.CX(c, t)
            # if c == t, skip this step to avoid invalid CX

    return circ


def RAND_4_d10() -> Circuit:
    """Random circuit: 4 qubits, depth = 10."""
    return random_circuit(n_qubits=4, depth=10, seed=42)


def RAND_6_d20() -> Circuit:
    """Random circuit: 6 qubits, depth = 20."""
    return random_circuit(n_qubits=6, depth=20, seed=99)


if __name__ == "__main__":
    for name, ctor in [("RAND_4_d10", RAND_4_d10),
                       ("RAND_6_d20", RAND_6_d20)]:
        print(f"=== {name} ===")
        c = ctor()
        CircuitStats(c)
        print()
