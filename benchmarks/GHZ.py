from pytket.circuit import Circuit
from pytket.utils import CircuitStats

# GHZ_4 (4 qubits)
# GHZ_8 (8 qubits)

def ghz_circuit(n_qubits: int) -> Circuit:
    """
    Prepare an n-qubit GHZ state:
        (|0...0> + |1...1>) / sqrt(2)
    Circuit pattern:
        H on qubit 0, then a CX chain: 0→1→2→...→n-1
    """
    circ = Circuit(n_qubits)
    circ.H(0)
    for i in range(n_qubits - 1):
        circ.CX(i, i + 1)
    return circ


def GHZ_4() -> Circuit:
    """GHZ state on 4 qubits."""
    return ghz_circuit(4)


def GHZ_8() -> Circuit:
    """GHZ state on 8 qubits."""
    return ghz_circuit(8)


if __name__ == "__main__":
    for name, ctor in [("GHZ_4", GHZ_4), ("GHZ_8", GHZ_8)]:
        print(f"=== {name} ===")
        c = ctor()
        CircuitStats(c)
        print()
