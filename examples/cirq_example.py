"""
Cirq Example: Simulate a Quantum Circuit
----------------------------------------
This script demonstrates how to use Cirq to create a Bell state circuit,
simulate it, and print the measurement results.
"""
import cirq

# Create two qubits
q0, q1 = cirq.LineQubit.range(2)
# Build the circuit
circuit = cirq.Circuit(
    cirq.H(q0),           # Hadamard on qubit 0
    cirq.CNOT(q0, q1),    # CNOT from qubit 0 to 1
    cirq.measure(q0, q1, key='result')
)

# Simulate the circuit
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1000)
# Print the histogram of results
print("Measurement results:", result.histogram(key='result'))
