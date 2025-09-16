"""
PennyLane Example: Quantum Circuit as a Differentiable Function
--------------------------------------------------------------
This script demonstrates how to use PennyLane to define a quantum circuit
as a differentiable function and evaluate its expectation value.
"""
import pennylane as qml
import numpy as np

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def bell_circuit(theta):
    # Create a Bell state and apply a rotation
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(theta, wires=0)
    return qml.expval(qml.PauliZ(0))

# Set rotation angle
theta = np.pi / 4
# Evaluate the circuit
result = bell_circuit(theta)
print("Expectation value:", result)
