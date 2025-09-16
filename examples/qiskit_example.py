"""
Qiskit Example: Create and Simulate a Bell State
-------------------------------------------------
This script demonstrates how to use Qiskit to create a Bell state,
simulate it, and print the measurement results.
"""
from qiskit import QuantumCircuit, Aer, execute

# Create a 2-qubit quantum circuit
circuit = QuantumCircuit(2, 2)

# Apply Hadamard gate to qubit 0 (creates superposition)
circuit.h(0)
# Apply CNOT gate (entangles qubits 0 and 1)
circuit.cx(0, 1)
# Measure both qubits
circuit.measure([0, 1], [0, 1])

# Use the QasmSimulator backend
simulator = Aer.get_backend('qasm_simulator')
# Execute the circuit 1024 times (shots)
result = execute(circuit, simulator, shots=1024).result()
# Get the counts of measurement outcomes
counts = result.get_counts()
print("Bell state counts:", counts)
