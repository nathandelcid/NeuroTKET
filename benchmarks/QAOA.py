import math
from pytket.circuit import Circuit
from pytket.utils import CircuitStats

# QAOA_ring_4_p1 (4 qubits, p=1)
# QAOA_ring_8_p2 (8 qubits, p=2)

def qaoa_maxcut_ring(n_qubits: int, p_layers: int,
                     gamma: float = 0.7, beta: float = 0.3) -> Circuit:
    """
    QAOA for MaxCut on a ring graph with n_qubits nodes and p_layers QAOA layers.
    Uses fixed angles (gamma, beta) for all layers for simplicity.

    Cost Hamiltonian: sum over edges (i, j) of Z_i Z_j
    Mixer: sum over i of X_i

    Implementation:
      - Start with H on all qubits
      - For each layer:
          * For each edge (i, j): e^{-i gamma Z_i Z_j} via CX - Rz(-2*gamma) - CX
          * For each qubit i: e^{-i beta X_i} via Rx(2*beta)
    """
    circ = Circuit(n_qubits)

    # Initial layer of H on all qubits
    for q in range(n_qubits):
        circ.H(q)

    # Edges of a ring graph
    edges = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]

    for _ in range(p_layers):
        # Problem unitary (ZZ terms)
        for (i, j) in edges:
            circ.CX(i, j)
            circ.Rz(-2 * gamma, j)
            circ.CX(i, j)

        # Mixer unitary (X terms)
        for q in range(n_qubits):
            circ.Rx(2 * beta, q)

    return circ


def QAOA_ring_4_p1() -> Circuit:
    """QAOA MaxCut on a 4-node ring, p = 1."""
    return qaoa_maxcut_ring(n_qubits=4, p_layers=1)


def QAOA_ring_8_p2() -> Circuit:
    """QAOA MaxCut on an 8-node ring, p = 2."""
    return qaoa_maxcut_ring(n_qubits=8, p_layers=2)


if __name__ == "__main__":
    for name, ctor in [("QAOA_ring_4_p1", QAOA_ring_4_p1),
                       ("QAOA_ring_8_p2", QAOA_ring_8_p2)]:
        print(f"=== {name} ===")
        c = ctor()
        CircuitStats(c)
        print()
