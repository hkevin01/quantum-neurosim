"""
Amazon Braket Example: Run a Circuit on a Local Simulator
--------------------------------------------------------
This script demonstrates how to use the Amazon Braket SDK to create a Bell state circuit,
run it on the local simulator, and print the measurement counts.
"""
from braket.circuits import Circuit
from braket.devices import LocalSimulator

# Create a Bell state circuit
circuit = Circuit().h(0).cnot(0, 1).measure(0, 1)
# Use the local simulator
device = LocalSimulator()
# Run the circuit with 1000 shots
result = device.run(circuit, shots=1000).result()
# Print the measurement counts
print("Measurement counts:", result.measurement_counts)
