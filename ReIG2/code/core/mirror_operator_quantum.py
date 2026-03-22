"""
ReIG2/twinRIG - Mirror Operator Quantum Implementation
=======================================================

Quantum implementation of the Empathy Operator M̂ using Qiskit.
Based on Section 13.3 / Chapter 8.3 of the ReIG2 framework.

M̂ |ψ_self⟩ ⊗ |φ_other⟩ = |φ_self⟩ ⊗ |ψ_other⟩

Requirements:
    pip install qiskit qiskit-aer numpy matplotlib

Reference: ReIG2_twinRIG_integrated.pdf, Section 13
           chapter8_coherence_ethics.pdf
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

# Qiskit imports (with fallback for demo without Qiskit)
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.quantum_info import Operator, Statevector, DensityMatrix
    from qiskit.circuit.library import SwapGate
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not installed. Running in NumPy-only mode.")


@dataclass
class EmpathyBarrier:
    """
    共感障壁パラメータ (Section 13.3.2)
    
    δ_total = (δ_cultural + δ_religious + δ_economic + δ_linguistic) / 4
    """
    delta_cultural: float = 0.0    # 文化的距離
    delta_religious: float = 0.0   # 宗教的距離
    delta_economic: float = 0.0    # 経済的距離
    delta_linguistic: float = 0.0  # 言語的距離
    
    @property
    def total(self) -> float:
        """総合共感障壁 δ"""
        return (self.delta_cultural + self.delta_religious + 
                self.delta_economic + self.delta_linguistic) / 4


class MirrorOperatorQuantum:
    """
    量子版共感演算子 M̂ (Qiskit実装)
    
    Properties:
    - Unitary: M̂†M̂ = I
    - Self-inverse: M̂² = I
    - Hermitian: M̂† = M̂
    
    Matrix representation for 2-level system (Eq. 72):
    M̂ = [[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]]
    """
    
    def __init__(self, n_qubits_per_party: int = 1):
        """
        Initialize Mirror Operator.
        
        Args:
            n_qubits_per_party: Number of qubits per self/other (default: 1)
        """
        self.n_qubits = n_qubits_per_party
        self.total_qubits = 2 * n_qubits_per_party
        
        # Build operator matrix
        self.matrix = self._build_swap_matrix()
        
        if QISKIT_AVAILABLE:
            self.operator = Operator(self.matrix)
    
    def _build_swap_matrix(self) -> np.ndarray:
        """
        Build SWAP matrix for perspective exchange.
        
        For single qubit: 4x4 matrix
        For n qubits: 2^(2n) x 2^(2n) matrix
        """
        dim = 2 ** self.n_qubits
        total_dim = dim * dim  # self ⊗ other
        
        M = np.zeros((total_dim, total_dim), dtype=complex)
        
        for i in range(dim):
            for j in range(dim):
                # |i⟩|j⟩ → |j⟩|i⟩
                idx_in = i * dim + j
                idx_out = j * dim + i
                M[idx_out, idx_in] = 1.0
        
        return M
    
    def verify_properties(self) -> Dict[str, bool]:
        """Verify M̂ satisfies required properties."""
        M = self.matrix
        I = np.eye(len(M))
        
        return {
            "unitary": np.allclose(M.conj().T @ M, I),
            "self_inverse": np.allclose(M @ M, I),
            "hermitian": np.allclose(M, M.conj().T)
        }
    
    def apply_numpy(
        self,
        state_self: np.ndarray,
        state_other: np.ndarray
    ) -> np.ndarray:
        """
        Apply M̂ using NumPy (no Qiskit required).
        
        Args:
            state_self: Self state vector |ψ⟩
            state_other: Other state vector |φ⟩
            
        Returns:
            Transformed state |φ⟩|ψ⟩
        """
        # Tensor product
        combined = np.kron(state_self, state_other)
        
        # Apply M̂
        result = self.matrix @ combined
        
        return result
    
    def create_circuit(
        self,
        include_measurement: bool = False
    ) -> 'QuantumCircuit':
        """
        Create Qiskit circuit implementing M̂.
        
        Uses SWAP gate decomposition:
        SWAP = (CNOT_{01})(CNOT_{10})(CNOT_{01})
        """
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit not available")
        
        qr_self = QuantumRegister(self.n_qubits, 'self')
        qr_other = QuantumRegister(self.n_qubits, 'other')
        
        if include_measurement:
            cr = ClassicalRegister(self.total_qubits, 'measure')
            qc = QuantumCircuit(qr_self, qr_other, cr)
        else:
            qc = QuantumCircuit(qr_self, qr_other)
        
        # Apply SWAP between corresponding qubits
        for i in range(self.n_qubits):
            qc.swap(qr_self[i], qr_other[i])
        
        if include_measurement:
            qc.measure(qr_self, cr[:self.n_qubits])
            qc.measure(qr_other, cr[self.n_qubits:])
        
        return qc
    
    def create_circuit_with_init(
        self,
        state_self: np.ndarray,
        state_other: np.ndarray,
        include_measurement: bool = False
    ) -> 'QuantumCircuit':
        """
        Create circuit with initial state preparation.
        """
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit not available")
        
        qr_self = QuantumRegister(self.n_qubits, 'self')
        qr_other = QuantumRegister(self.n_qubits, 'other')
        
        if include_measurement:
            cr = ClassicalRegister(self.total_qubits, 'measure')
            qc = QuantumCircuit(qr_self, qr_other, cr)
        else:
            qc = QuantumCircuit(qr_self, qr_other)
        
        # Normalize states
        state_self = state_self / np.linalg.norm(state_self)
        state_other = state_other / np.linalg.norm(state_other)
        
        # Initialize states
        qc.initialize(state_self, qr_self)
        qc.initialize(state_other, qr_other)
        
        # Apply SWAP (Mirror)
        for i in range(self.n_qubits):
            qc.swap(qr_self[i], qr_other[i])
        
        if include_measurement:
            qc.measure(qr_self, cr[:self.n_qubits])
            qc.measure(qr_other, cr[self.n_qubits:])
        
        return qc
    
    def simulate(
        self,
        circuit: 'QuantumCircuit',
        shots: int = 1024
    ) -> Dict:
        """
        Simulate circuit execution.
        """
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit not available")
        
        simulator = AerSimulator()
        
        # Get statevector if no measurements
        if circuit.num_clbits == 0:
            sv_circuit = circuit.copy()
            sv_circuit.save_statevector()
            result = simulator.run(sv_circuit).result()
            statevector = result.get_statevector()
            return {
                "statevector": np.array(statevector),
                "probabilities": np.abs(np.array(statevector)) ** 2
            }
        else:
            result = simulator.run(circuit, shots=shots).result()
            counts = result.get_counts()
            return {"counts": counts, "shots": shots}


class ImperfectMirrorOperator(MirrorOperatorQuantum):
    """
    不完全共感演算子 (共感障壁を考慮)
    
    M̂' = (1-δ) M̂ + δ (I/d)
    
    Where δ is the empathy barrier.
    """
    
    def __init__(
        self,
        n_qubits_per_party: int = 1,
        barrier: Optional[EmpathyBarrier] = None
    ):
        super().__init__(n_qubits_per_party)
        self.barrier = barrier or EmpathyBarrier()
        
        # Build imperfect operator
        self.imperfect_matrix = self._build_imperfect_matrix()
    
    def _build_imperfect_matrix(self) -> np.ndarray:
        """
        Build imperfect SWAP with depolarizing noise.
        
        M̂' = (1-δ)M̂ + δ(I/d)
        """
        delta = self.barrier.total
        d = len(self.matrix)
        
        # Perfect SWAP + noise
        imperfect = (1 - delta) * self.matrix + delta * np.eye(d) / d
        
        return imperfect
    
    def create_noisy_circuit(
        self,
        state_self: np.ndarray,
        state_other: np.ndarray
    ) -> Tuple['QuantumCircuit', 'NoiseModel']:
        """
        Create circuit with noise model representing empathy barrier.
        """
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit not available")
        
        # Create base circuit
        qc = self.create_circuit_with_init(state_self, state_other)
        
        # Create noise model
        delta = self.barrier.total
        noise_model = NoiseModel()
        
        # Add depolarizing error to SWAP gates
        if delta > 0:
            error = depolarizing_error(delta, 2)  # 2-qubit gate
            noise_model.add_all_qubit_quantum_error(error, ['swap'])
        
        return qc, noise_model
    
    def simulate_with_noise(
        self,
        state_self: np.ndarray,
        state_other: np.ndarray,
        shots: int = 1024
    ) -> Dict:
        """
        Simulate with noise representing empathy barrier.
        """
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit not available")
        
        qc, noise_model = self.create_noisy_circuit(state_self, state_other)
        
        # Add measurement
        qc.measure_all()
        
        simulator = AerSimulator(noise_model=noise_model)
        result = simulator.run(qc, shots=shots).result()
        
        return {
            "counts": result.get_counts(),
            "noise_level": self.barrier.total,
            "shots": shots
        }


class CooperativeTransitionCircuit:
    """
    協力相転移を実現する量子回路
    
    H_total = H_self ⊗ I + I ⊗ H_other - λ M̂
    """
    
    def __init__(
        self,
        lambda_empathy: float = 0.5,
        n_trotter_steps: int = 10
    ):
        self.lambda_empathy = lambda_empathy
        self.n_steps = n_trotter_steps
        self.mirror_op = MirrorOperatorQuantum(n_qubits_per_party=1)
    
    def create_evolution_circuit(
        self,
        dt: float = 0.1,
        state_self: Optional[np.ndarray] = None,
        state_other: Optional[np.ndarray] = None
    ) -> 'QuantumCircuit':
        """
        Create time evolution circuit with empathy coupling.
        
        U(t) ≈ [exp(-iH_self dt) ⊗ exp(-iH_other dt) · exp(iλM̂dt)]^n
        """
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit not available")
        
        qr_self = QuantumRegister(1, 'self')
        qr_other = QuantumRegister(1, 'other')
        qc = QuantumCircuit(qr_self, qr_other)
        
        # Initialize if provided
        if state_self is not None:
            state_self = state_self / np.linalg.norm(state_self)
            qc.initialize(state_self, qr_self)
        
        if state_other is not None:
            state_other = state_other / np.linalg.norm(state_other)
            qc.initialize(state_other, qr_other)
        
        # Trotter steps
        for _ in range(self.n_steps):
            # Local evolution: exp(-iH_self dt)
            qc.rz(dt, qr_self[0])
            qc.rz(dt, qr_other[0])
            
            # Empathy coupling: partial SWAP
            # Approximate exp(iλM̂dt) ≈ I + iλM̂dt for small dt
            # Use controlled operations to implement partial SWAP
            theta = self.lambda_empathy * dt
            
            # Partial SWAP via Rxx, Ryy, Rzz decomposition
            qc.rxx(theta, qr_self[0], qr_other[0])
            qc.ryy(theta, qr_self[0], qr_other[0])
            qc.rzz(theta, qr_self[0], qr_other[0])
        
        return qc
    
    def compute_mutual_coherence(
        self,
        statevector: np.ndarray
    ) -> float:
        """
        Compute mutual coherence C_mutual = ⟨M̂⟩.
        """
        M = self.mirror_op.matrix
        rho = np.outer(statevector, statevector.conj())
        return float(np.real(np.trace(rho @ M)))


def demo_quantum_mirror():
    """Demonstrate quantum Mirror Operator."""
    print("=" * 60)
    print("Quantum Mirror Operator (M̂) Demo")
    print("ReIG2/twinRIG Chapter 13 Implementation")
    print("=" * 60)
    
    # 1. Basic properties
    print("\n1. Mirror Operator Properties")
    print("-" * 40)
    
    M_op = MirrorOperatorQuantum(n_qubits_per_party=1)
    props = M_op.verify_properties()
    
    print(f"   Unitary (M̂†M̂ = I): {props['unitary']}")
    print(f"   Self-inverse (M̂² = I): {props['self_inverse']}")
    print(f"   Hermitian (M̂† = M̂): {props['hermitian']}")
    
    print("\n   Matrix representation:")
    print(M_op.matrix.astype(int))
    
    # 2. NumPy-based computation
    print("\n2. Perspective Exchange (NumPy)")
    print("-" * 40)
    
    # |0⟩ and |1⟩
    state_0 = np.array([1, 0], dtype=complex)
    state_1 = np.array([0, 1], dtype=complex)
    
    # |0⟩|1⟩ → |1⟩|0⟩
    result_01 = M_op.apply_numpy(state_0, state_1)
    print(f"   M̂|0⟩|1⟩ = {np.round(result_01, 3)}")
    print(f"   (Should be |1⟩|0⟩ = [0, 0, 1, 0])")
    
    # |1⟩|0⟩ → |0⟩|1⟩
    result_10 = M_op.apply_numpy(state_1, state_0)
    print(f"   M̂|1⟩|0⟩ = {np.round(result_10, 3)}")
    print(f"   (Should be |0⟩|1⟩ = [0, 1, 0, 0])")
    
    # Superposition: |+⟩ = (|0⟩+|1⟩)/√2
    state_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    result_plus = M_op.apply_numpy(state_plus, state_0)
    print(f"\n   M̂|+⟩|0⟩ = {np.round(result_plus, 3)}")
    
    # 3. Qiskit circuit (if available)
    if QISKIT_AVAILABLE:
        print("\n3. Qiskit Circuit Implementation")
        print("-" * 40)
        
        qc = M_op.create_circuit()
        print("   Circuit structure:")
        print(qc.draw(output='text'))
        
        # Simulate
        print("\n   Simulation: |0⟩|1⟩ → ?")
        qc_init = M_op.create_circuit_with_init(state_0, state_1)
        result = M_op.simulate(qc_init)
        print(f"   Final statevector: {np.round(result['statevector'], 3)}")
        print(f"   Probabilities: {np.round(result['probabilities'], 3)}")
    else:
        print("\n3. Qiskit not available - skipping circuit demo")
    
    # 4. Imperfect Mirror (with barrier)
    print("\n4. Imperfect Mirror with Empathy Barrier")
    print("-" * 40)
    
    barrier = EmpathyBarrier(
        delta_cultural=0.1,
        delta_religious=0.15,
        delta_economic=0.2,
        delta_linguistic=0.05
    )
    
    print(f"   Cultural barrier: {barrier.delta_cultural}")
    print(f"   Religious barrier: {barrier.delta_religious}")
    print(f"   Economic barrier: {barrier.delta_economic}")
    print(f"   Linguistic barrier: {barrier.delta_linguistic}")
    print(f"   Total barrier δ: {barrier.total:.4f}")
    
    M_imperfect = ImperfectMirrorOperator(barrier=barrier)
    
    # Compare perfect vs imperfect
    result_perfect = M_op.apply_numpy(state_0, state_1)
    result_imperfect = M_imperfect.imperfect_matrix @ np.kron(state_0, state_1)
    
    print(f"\n   Perfect M̂|0⟩|1⟩: {np.round(result_perfect, 3)}")
    print(f"   Imperfect M̂'|0⟩|1⟩: {np.round(result_imperfect, 3)}")
    
    fidelity = np.abs(np.vdot(result_perfect, result_imperfect)) ** 2
    print(f"   Fidelity: {fidelity:.4f}")
    
    # 5. Cooperative evolution (if Qiskit available)
    if QISKIT_AVAILABLE:
        print("\n5. Cooperative Phase Transition Circuit")
        print("-" * 40)
        
        coop = CooperativeTransitionCircuit(lambda_empathy=0.5, n_trotter_steps=5)
        
        # Start from |0⟩|1⟩ (asymmetric)
        qc_coop = coop.create_evolution_circuit(
            dt=0.1,
            state_self=state_0,
            state_other=state_1
        )
        
        print(f"   λ (empathy strength): {coop.lambda_empathy}")
        print(f"   Trotter steps: {coop.n_steps}")
        print(f"   Circuit depth: {qc_coop.depth()}")
        
        # Simulate
        qc_coop.save_statevector()
        simulator = AerSimulator()
        result = simulator.run(qc_coop).result()
        final_sv = np.array(result.get_statevector())
        
        # Compute mutual coherence
        C_mutual = coop.compute_mutual_coherence(final_sv)
        print(f"   Final mutual coherence: {C_mutual:.4f}")
        print(f"   (C_mutual > 0 indicates cooperation)")
    
    # 6. Multi-qubit extension
    print("\n6. Multi-qubit Extension")
    print("-" * 40)
    
    M_2qubit = MirrorOperatorQuantum(n_qubits_per_party=2)
    print(f"   2-qubit per party: {M_2qubit.total_qubits} total qubits")
    print(f"   Matrix dimension: {M_2qubit.matrix.shape}")
    props_2q = M_2qubit.verify_properties()
    print(f"   Properties verified: {all(props_2q.values())}")
    
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    demo_quantum_mirror()
