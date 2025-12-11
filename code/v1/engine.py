"""
twinRIG V1 - Core Quantum State Engine
=======================================

This module implements the fundamental quantum state representation and evolution
mechanics for the World Generation Tensor System (T_World).

Mathematical Foundation (ReIG2/twinRIG Framework):
- State vector |Ψ⟩ lives in H = H_M ⊗ H_C ⊗ H_E ⊗ H_F ⊗ H_S
- Evolution: |Ψ(t+dt)⟩ = U_res(dt)|Ψ(t)⟩
- Extended Hamiltonian: Ĥ = H₀ + τH_future + εH_entropy + PFH·H_ethics

Reference: ReIG2_twinRIG_2025_December.pdf, Sections 2-5
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from enum import Enum
import scipy.linalg as la


class SubspaceType(Enum):
    """Enumeration of the 5 fundamental subspaces (Eq. 48)."""
    MEANING = "meaning"      # H_M: 意味空間
    CONTEXT = "context"      # H_C: 文脈空間
    ETHICS = "ethics"        # H_E: 倫理空間
    FUTURE = "future"        # H_F: 未来空間
    STABILITY = "stability"  # H_S: 安定性空間


@dataclass
class SubspaceConfig:
    """Configuration for each subspace dimension."""
    subspace_type: SubspaceType
    dimension: int
    basis_labels: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.dimension < 2:
            raise ValueError(f"Subspace dimension must be ≥2, got {self.dimension}")


class QuantumState:
    """
    Quantum state |Ψ⟩ in composite Hilbert space.
    
    The state lives in H_full = H_M ⊗ H_C ⊗ H_E ⊗ H_F ⊗ H_S (Eq. 48).
    
    Attributes:
        amplitudes: Complex amplitude vector
        subspace_dims: Tuple of dimensions for each subspace
        total_dim: Total dimension of composite space
    """
    
    def __init__(
        self,
        amplitudes: Optional[NDArray[np.complex128]] = None,
        subspace_dims: Tuple[int, ...] = (3, 3, 3, 3, 3),
        normalize: bool = True
    ):
        """
        Initialize quantum state.
        
        Args:
            amplitudes: Initial state vector. If None, creates uniform superposition.
            subspace_dims: Dimensions (d_M, d_C, d_E, d_F, d_S)
            normalize: Whether to normalize the state
        """
        self.subspace_dims = subspace_dims
        self.total_dim = int(np.prod(subspace_dims))
        
        if amplitudes is None:
            # Uniform superposition: |Ψ⟩ = 1/√D Σᵢ|i⟩
            self.amplitudes = np.ones(self.total_dim, dtype=np.complex128)
        else:
            if len(amplitudes) != self.total_dim:
                raise ValueError(
                    f"Amplitude dimension {len(amplitudes)} does not match "
                    f"total dimension {self.total_dim}"
                )
            self.amplitudes = np.array(amplitudes, dtype=np.complex128)
        
        if normalize:
            self.normalize()
    
    def normalize(self) -> QuantumState:
        """Normalize state to unit norm: ||Ψ|| = 1."""
        norm = np.linalg.norm(self.amplitudes)
        if norm > 1e-10:
            self.amplitudes /= norm
        return self
    
    def norm(self) -> float:
        """Calculate L2 norm: ||Ψ|| = √(⟨Ψ|Ψ⟩)."""
        return float(np.linalg.norm(self.amplitudes))
    
    def inner_product(self, other: QuantumState) -> complex:
        """
        Calculate inner product: ⟨Φ|Ψ⟩.
        
        Args:
            other: Another quantum state
            
        Returns:
            Complex inner product
        """
        if self.total_dim != other.total_dim:
            raise ValueError("States must have same dimension")
        return complex(np.vdot(other.amplitudes, self.amplitudes))
    
    def expectation_value(self, operator: NDArray[np.complex128]) -> float:
        """
        Calculate expectation value: ⟨Ψ|Ô|Ψ⟩.
        
        Args:
            operator: Hermitian operator matrix
            
        Returns:
            Real expectation value
        """
        result = np.vdot(self.amplitudes, operator @ self.amplitudes)
        return float(np.real(result))
    
    def measure_subspace(self, subspace_index: int) -> NDArray[np.float64]:
        """
        Measure probability distribution in a single subspace.
        
        Args:
            subspace_index: Index of subspace (0=M, 1=C, 2=E, 3=F, 4=S)
            
        Returns:
            Probability distribution over basis states
        """
        if subspace_index < 0 or subspace_index >= len(self.subspace_dims):
            raise ValueError(f"Invalid subspace index: {subspace_index}")
        
        # Reshape to tensor form
        tensor = self.amplitudes.reshape(self.subspace_dims)
        
        # Sum over all other axes
        probs = np.abs(tensor) ** 2
        axes_to_sum = tuple(i for i in range(len(self.subspace_dims)) if i != subspace_index)
        marginal_probs = np.sum(probs, axis=axes_to_sum)
        
        return marginal_probs
    
    def fidelity(self, other: QuantumState) -> float:
        """
        Calculate fidelity: F(Ψ,Φ) = |⟨Ψ|Φ⟩|².
        
        Args:
            other: Another quantum state
            
        Returns:
            Fidelity between 0 and 1
        """
        return float(np.abs(self.inner_product(other)) ** 2)
    
    def copy(self) -> QuantumState:
        """Create a deep copy of the state."""
        return QuantumState(
            amplitudes=self.amplitudes.copy(),
            subspace_dims=self.subspace_dims,
            normalize=False
        )
    
    def __repr__(self) -> str:
        return (
            f"QuantumState(dim={self.total_dim}, "
            f"subspaces={self.subspace_dims}, "
            f"norm={self.norm():.6f})"
        )


class TimeEvolutionEngine:
    """
    Time evolution engine for quantum states.
    
    Implements the extended time evolution operator (Eq. 3):
    U_res(t; τ, ε, PFH) = exp(-i Ĥ(t, τ, ε, PFH) / ℏ)
    
    Uses Trotter-Suzuki decomposition for non-commuting terms (Eq. 17).
    """
    
    def __init__(
        self,
        hamiltonian_terms: List[NDArray[np.complex128]],
        dt: float = 0.01,
        hbar: float = 1.0,
        trotter_order: int = 2
    ):
        """
        Initialize evolution engine.
        
        Args:
            hamiltonian_terms: List of Hamiltonian component matrices
            dt: Time step for evolution
            hbar: Reduced Planck constant (default 1.0 in natural units)
            trotter_order: Order of Trotter decomposition (1 or 2)
        """
        self.hamiltonian_terms = hamiltonian_terms
        self.dt = dt
        self.hbar = hbar
        self.trotter_order = trotter_order
        self.dim = hamiltonian_terms[0].shape[0]
        
        # Verify all Hamiltonians have same dimension
        for H in hamiltonian_terms:
            if H.shape != (self.dim, self.dim):
                raise ValueError("All Hamiltonians must have same dimension")
    
    def evolve_step(
        self,
        state: QuantumState,
        coefficients: Optional[List[float]] = None
    ) -> QuantumState:
        """
        Evolve state by one time step.
        
        Args:
            state: Current quantum state
            coefficients: Coefficients [1, τ, ε, PFH] for Hamiltonian terms
            
        Returns:
            New evolved state
        """
        if coefficients is None:
            coefficients = [1.0] * len(self.hamiltonian_terms)
        
        if len(coefficients) != len(self.hamiltonian_terms):
            raise ValueError("Number of coefficients must match Hamiltonian terms")
        
        # Build total Hamiltonian: Ĥ = Σₖ cₖ Hₖ
        H_total = sum(c * H for c, H in zip(coefficients, self.hamiltonian_terms))
        
        if self.trotter_order == 1:
            # First-order Trotter: U ≈ Πₖ exp(-i cₖ Hₖ dt)
            U = np.eye(self.dim, dtype=np.complex128)
            for c, H in zip(coefficients, self.hamiltonian_terms):
                U = la.expm(-1j * c * H * self.dt / self.hbar) @ U
        else:
            # Second-order symmetric Trotter (Eq. 20)
            # U ≈ Πₖ exp(-i cₖ Hₖ dt/2) · Πₖ' exp(-i cₖ' Hₖ' dt/2)
            U = np.eye(self.dim, dtype=np.complex128)
            
            # Forward sweep
            for c, H in zip(coefficients, self.hamiltonian_terms):
                U = la.expm(-1j * c * H * self.dt / (2 * self.hbar)) @ U
            
            # Backward sweep
            for c, H in zip(reversed(coefficients), reversed(self.hamiltonian_terms)):
                U = la.expm(-1j * c * H * self.dt / (2 * self.hbar)) @ U
        
        # Apply evolution
        new_amplitudes = U @ state.amplitudes
        
        return QuantumState(
            amplitudes=new_amplitudes,
            subspace_dims=state.subspace_dims,
            normalize=True
        )
    
    def evolve(
        self,
        state: QuantumState,
        n_steps: int,
        coefficients: Optional[List[float]] = None,
        callback: Optional[callable] = None
    ) -> Tuple[QuantumState, List[QuantumState]]:
        """
        Evolve state for multiple time steps.
        
        Args:
            state: Initial state
            n_steps: Number of evolution steps
            coefficients: Hamiltonian coefficients
            callback: Optional function called after each step
            
        Returns:
            Tuple of (final_state, trajectory)
        """
        current_state = state.copy()
        trajectory = [current_state.copy()]
        
        for step in range(n_steps):
            current_state = self.evolve_step(current_state, coefficients)
            trajectory.append(current_state.copy())
            
            if callback is not None:
                callback(step, current_state)
        
        return current_state, trajectory


class WorldOperator:
    """
    World construction operator T̂_World (Eq. 58).
    
    Implements the full transformation chain:
    T̂_World = T_I ∘ T_R ∘ T_C ∘ U_multi ∘ U_res
    """
    
    def __init__(
        self,
        dim: int,
        subspace_dims: Tuple[int, ...],
        contraction_factor: float = 0.95
    ):
        """
        Initialize world operator.
        
        Args:
            dim: Total Hilbert space dimension
            subspace_dims: Dimensions of each subspace
            contraction_factor: κ < 1 for convergence (Theorem 5.1)
        """
        self.dim = dim
        self.subspace_dims = subspace_dims
        self.contraction_factor = contraction_factor
        
        # Build component operators
        self._build_operators()
    
    def _build_operators(self):
        """Build the component transformation operators."""
        # T_C: Cognitive transformation (Eq. 54)
        self.T_C = self._make_random_unitary() * self.contraction_factor
        
        # T_R: Recognition transformation (Eq. 56)
        self.T_R = self._make_random_unitary() * self.contraction_factor
        
        # T_I: Integration transformation (Eq. 57)
        self.T_I = self._make_random_unitary() * self.contraction_factor
    
    def _make_random_unitary(self) -> NDArray[np.complex128]:
        """Generate a random unitary matrix."""
        # QR decomposition of random complex matrix
        A = np.random.randn(self.dim, self.dim) + 1j * np.random.randn(self.dim, self.dim)
        Q, R = np.linalg.qr(A)
        # Ensure determinant is 1
        D = np.diag(np.diag(R) / np.abs(np.diag(R)))
        return Q @ D
    
    def apply(self, state: QuantumState) -> QuantumState:
        """
        Apply world operator to state.
        
        Args:
            state: Input quantum state
            
        Returns:
            Transformed state
        """
        # T̂_World = T_I ∘ T_R ∘ T_C
        new_amplitudes = self.T_I @ (self.T_R @ (self.T_C @ state.amplitudes))
        
        return QuantumState(
            amplitudes=new_amplitudes,
            subspace_dims=state.subspace_dims,
            normalize=True
        )
    
    def find_fixed_point(
        self,
        initial_state: QuantumState,
        max_iterations: int = 1000,
        tolerance: float = 1e-8
    ) -> Tuple[QuantumState, int, List[float]]:
        """
        Find the fixed point |I⟩ such that T̂_World|I⟩ = |I⟩ (Theorem 5.1).
        
        Uses Picard iteration with Banach contraction mapping.
        
        Args:
            initial_state: Starting state |Ψ₀⟩
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Tuple of (fixed_point, iterations, convergence_history)
        """
        current = initial_state.copy()
        history = []
        
        for iteration in range(max_iterations):
            next_state = self.apply(current)
            
            # Calculate distance
            distance = np.linalg.norm(next_state.amplitudes - current.amplitudes)
            history.append(distance)
            
            if distance < tolerance:
                return next_state, iteration + 1, history
            
            current = next_state
        
        return current, max_iterations, history


if __name__ == "__main__":
    # Quick test
    print("twinRIG V1 Engine Test")
    print("=" * 40)
    
    # Create state
    state = QuantumState(subspace_dims=(3, 3, 3, 3, 3))
    print(f"Initial state: {state}")
    
    # Measure meaning subspace
    probs = state.measure_subspace(0)
    print(f"Meaning subspace probabilities: {probs}")
    
    # Test world operator
    world_op = WorldOperator(dim=state.total_dim, subspace_dims=state.subspace_dims)
    fixed_point, iterations, history = world_op.find_fixed_point(state)
    print(f"Fixed point found in {iterations} iterations")
    print(f"Final convergence: {history[-1]:.2e}")
