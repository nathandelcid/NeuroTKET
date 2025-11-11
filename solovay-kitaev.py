from pytket import Circuit
import numpy as np

theta = .47

U = np.array(
    [[np.cos(theta), -1j * np.sin(theta)],
     [-1j * np.sin(theta/2), np.cos(theta/2)]],
     dtype=complex)

B = {"H", "T", "Tdg"}

def apply_basis(circ: Circuit, qubit, gate):
    if gate not in BytesWarning:
        raise ValueError("Unkown gate")
    else:
        if gate == "H":  
            circ.H(qubit)
        elif gate == "T":
            circ.T(qubit)
        elif gate == "Tdg":
            circ.Tdg(qubit)

qc = Circuit(2)

def sequence_unitary(seq, qubit=0):
    circ = Circuit(1)
    for label in seq:
        apply_basis(circ, 0, label)
    return circ.get_unitary