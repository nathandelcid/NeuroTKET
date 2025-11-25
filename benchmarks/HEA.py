import math
import random
from pytket.circuit import Circuit
from pytket.utils import CircuitStats

# Hardware-efficient Ansatz

# HEA_4_d2 (4 qubits, depth=2)
# HEA_6_d4 (6 qubits, depth=4)

def hardware_efficient_ansatz(n_qubits: int, depth: int,
                              seed: int | None = None) -> Circuit:
    """
    Simple hardware-efficient ansatz:

    For each of `depth` layers:
      - Single-qubit Euler-angle-like rotations on each qubit:
          Rz(theta1) -> Rx(theta2) -> Rz(theta3)
        with pseudo-random angles (optionally seeded for reproducibility).
      - Entangling layer: CZs in a ring (0-1, 1-2, ..., n-1-0).
    """
    if seed is not None:
        random.seed(seed)

    circ = Circuit(n_qubits)

    for layer in range(depth):
        # Single-qubit rotations
        for q in range(n_qubits):
            # Simple deterministic-ish angles using random for variety
            theta1 = random.uniform(0, 2 * math.pi)
            theta2 = random.uniform(0, 2 * math.pi)
            theta3 = random.uniform(0, 2 * math.pi)
            circ.Rz(theta1, q)
            circ.Rx(theta2, q)
            circ.Rz(theta3, q)

        # Entangling ring of CZs
        for q in range(n_qubits):
            q_next = (q + 1) % n_qubits
            circ.CZ(q, q_next)

    return circ


def HEA_4_d2() -> Circuit:
    """Hardware-efficient ansatz with 4 qubits and depth = 2."""
    return hardware_efficient_ansatz(n_qubits=4, depth=2, seed=123)


def HEA_6_d4() -> Circuit:
    """Hardware-efficient ansatz with 6 qubits and depth = 4."""
    return hardware_efficient_ansatz(n_qubits=6, depth=4, seed=456)


if __name__ == "__main__":
    for name, ctor in [("HEA_4_d2", HEA_4_d2),
                       ("HEA_6_d4", HEA_6_d4)]:
        print(f"=== {name} ===")
        c = ctor()
        CircuitStats(c)
        print()
