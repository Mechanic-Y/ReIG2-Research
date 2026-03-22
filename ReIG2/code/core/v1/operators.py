"""
twinRIG V1 - Quantum Operators Module
=====================================

This module implements the Hamiltonian operators and phase transition operators
for the ReIG2/twinRIG quantum framework.

Mathematical Foundation:
- Extended Hamiltonian (Eq. 5): Ĥ = H₀ + τH_future + εH_entropy + PFH·H_ethics
- Phase Transition Operator (Eq. 25): G = P ∘ E ∘ R

Reference: ReIG2_twinRIG_2025_December.pdf, Sections 2, 4
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
import scipy.linalg as la
from enum import Enum, auto


class OperatorType(Enum):
    """Types of operators in the framework."""
    BASE = auto()        # H₀: Base Hamiltonian
    FUTURE = auto()      # H_future: Future contribution
    ENTROPY = auto()     # H_entropy: Fluctuation/entropy
    ETHICS = auto()      # H_ethics: Ethical alignment
    ROTATION = auto()    # R: Torsion operator
    EXPANSION = auto()   # E: Expansion operator
    JUMP = auto()        # P: Phase jump operator


@dataclass
class OperatorConfig:
    """Configuration for operator construction."""
    dim: int
    coupling_strength: float = 0.5
    seed: Optional[int] = None
    sparse: bool = False


class HamiltonianFactory:
    """
    Factory class for constructing Hamiltonian operators.
    
    Implements the extended Hamiltonian (Eq. 5):
    Ĥ(t, τ, ε, PFH) = H₀(t) + τH_future + εH_entropy + PFH·H_ethics
    """
    
    def __init__(
        self,
        dim: int,
        subspace_dims: Tuple[int, ...],
        coupling_strength: float = 0.3,
        seed: Optional[int] = None
    ):
        """
        Initialize Hamiltonian factory.
        
        Args:
            dim: Total Hilbert space dimension
            subspace_dims: Dimensions of each subspace (M, C, E, F, S)
            coupling_strength: Inter-subspace coupling strength
            seed: Random seed for reproducibility
        """
        self.dim = dim
        self.subspace_dims = subspace_dims
        self.coupling_strength = coupling_strength
        
        if seed is not None:
            np.random.seed(seed)
        
        # Build frequency parameters
        self.omega = {
            'M': 1.0,   # Meaning frequency
            'C': 0.7,   # Context frequency
            'E': 0.5,   # Ethics frequency
            'F': 0.8,   # Future frequency
            'S': 0.3    # Stability frequency
        }
    
    def create_base_hamiltonian(self) -> NDArray[np.complex128]:
        """
        Create base Hamiltonian H₀ (Eq. 6).
        
        H₀ = Σₖ ωₖ Nₖ  (number operators for each subspace)
        
        Returns:
            Hermitian matrix representing H₀
        """
        H0 = np.zeros((self.dim, self.dim), dtype=np.complex128)
        
        # Build diagonal terms (number operators)
        for i in range(self.dim):
            # Convert linear index to multi-index
            idx = self._linear_to_multi_index(i)
            energy = sum(
                self.omega[key] * idx[j] 
                for j, key in enumerate(['M', 'C', 'E', 'F', 'S'])
            )
            H0[i, i] = energy
        
        return H0
    
    def create_future_hamiltonian(self) -> NDArray[np.complex128]:
        """
        Create future contribution Hamiltonian H_future (Eq. 7).
        
        H_future = Σₖ ωₖ aₖ†aₖ · fₖ(future state)
        
        Returns:
            Hermitian matrix representing H_future
        """
        H_future = np.zeros((self.dim, self.dim), dtype=np.complex128)
        
        # Create ladder operators for future subspace
        d_F = self.subspace_dims[3]  # Future dimension
        
        for i in range(self.dim):
            for j in range(self.dim):
                idx_i = self._linear_to_multi_index(i)
                idx_j = self._linear_to_multi_index(j)
                
                # Only couple states differing in future index
                diff = [idx_i[k] - idx_j[k] for k in range(5)]
                if sum(abs(d) for d in diff) == 1 and diff[3] != 0:
                    H_future[i, j] = self.coupling_strength * self.omega['F']
        
        # Make Hermitian
        H_future = (H_future + H_future.conj().T) / 2
        
        return H_future
    
    def create_entropy_hamiltonian(self) -> NDArray[np.complex128]:
        """
        Create entropy/fluctuation Hamiltonian H_entropy (Eq. 8).
        
        H_entropy = -kT Σᵢ pᵢ log pᵢ · Πᵢ
        
        Returns:
            Hermitian matrix representing H_entropy
        """
        H_entropy = np.zeros((self.dim, self.dim), dtype=np.complex128)
        
        # Create random off-diagonal couplings
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                if np.random.random() < 0.1:  # Sparse coupling
                    coupling = self.coupling_strength * np.random.randn()
                    H_entropy[i, j] = coupling
                    H_entropy[j, i] = coupling
        
        # Add entropy-like diagonal terms
        for i in range(self.dim):
            idx = self._linear_to_multi_index(i)
            # Entropy increases with "distance" from ground state
            entropy_value = sum(idx[k] for k in range(5)) / sum(d-1 for d in self.subspace_dims)
            H_entropy[i, i] = entropy_value * self.omega['E']
        
        return H_entropy
    
    def create_ethics_hamiltonian(self) -> NDArray[np.complex128]:
        """
        Create ethics Hamiltonian H_ethics (Eq. 9).
        
        H_ethics = Σᵢⱼ V^eth_ij σᵢ ⊗ σⱼ
        
        Returns:
            Hermitian matrix representing H_ethics
        """
        H_ethics = np.zeros((self.dim, self.dim), dtype=np.complex128)
        
        # Couple ethics subspace with stability subspace
        d_E = self.subspace_dims[2]  # Ethics dimension
        d_S = self.subspace_dims[4]  # Stability dimension
        
        for i in range(self.dim):
            for j in range(self.dim):
                idx_i = self._linear_to_multi_index(i)
                idx_j = self._linear_to_multi_index(j)
                
                # Only couple states differing in ethics or stability
                diff = [idx_i[k] - idx_j[k] for k in range(5)]
                if sum(abs(d) for d in diff) <= 2:
                    if diff[2] != 0 or diff[4] != 0:
                        V_eth = self._ethics_potential(idx_i[2], idx_j[2], idx_i[4], idx_j[4])
                        H_ethics[i, j] += V_eth
        
        # Make Hermitian
        H_ethics = (H_ethics + H_ethics.conj().T) / 2
        
        return H_ethics
    
    def _ethics_potential(self, e1: int, e2: int, s1: int, s2: int) -> float:
        """Calculate ethics potential V^eth_ij."""
        # Higher ethics index = more "ethical", couple to stability
        alignment = 1.0 if (e1 >= e2 and s1 >= s2) else -0.5
        return self.coupling_strength * alignment * self.omega['E']
    
    def _linear_to_multi_index(self, linear_idx: int) -> Tuple[int, ...]:
        """Convert linear index to multi-index (M, C, E, F, S)."""
        idx = []
        remaining = linear_idx
        for d in reversed(self.subspace_dims):
            idx.append(remaining % d)
            remaining //= d
        return tuple(reversed(idx))
    
    def _multi_to_linear_index(self, multi_idx: Tuple[int, ...]) -> int:
        """Convert multi-index to linear index."""
        linear = 0
        multiplier = 1
        for i, d in enumerate(reversed(self.subspace_dims)):
            linear += multi_idx[-(i+1)] * multiplier
            multiplier *= d
        return linear
    
    def create_all_hamiltonians(self) -> Dict[str, NDArray[np.complex128]]:
        """
        Create all Hamiltonian components.
        
        Returns:
            Dictionary of Hamiltonians: {'H0', 'H_future', 'H_entropy', 'H_ethics'}
        """
        return {
            'H0': self.create_base_hamiltonian(),
            'H_future': self.create_future_hamiltonian(),
            'H_entropy': self.create_entropy_hamiltonian(),
            'H_ethics': self.create_ethics_hamiltonian()
        }
    
    def create_extended_hamiltonian(
        self,
        tau: float = 0.5,
        epsilon: float = 0.3,
        pfh: float = 0.2
    ) -> NDArray[np.complex128]:
        """
        Create full extended Hamiltonian (Eq. 5).
        
        Ĥ = H₀ + τH_future + εH_entropy + PFH·H_ethics
        
        Args:
            tau: Time resonance parameter τ
            epsilon: Entropy resonance parameter ε
            pfh: Philosophical resonance parameter PFH
            
        Returns:
            Full extended Hamiltonian
        """
        hamiltonians = self.create_all_hamiltonians()
        
        H = (hamiltonians['H0'] + 
             tau * hamiltonians['H_future'] + 
             epsilon * hamiltonians['H_entropy'] + 
             pfh * hamiltonians['H_ethics'])
        
        return H


class PhaseOperator:
    """
    Phase transition generation operator G (Eq. 25).
    
    G = P ∘ E ∘ R
    
    Where:
    - R: Torsion/rotation operator (Eq. 26-27)
    - E: Expansion operator (Eq. 28-30)
    - P: Phase jump operator (Eq. 31)
    """
    
    def __init__(
        self,
        dim: int,
        rotation_angle: float = np.pi / 4,
        expansion_rate: float = 0.1,
        jump_probability: float = 0.1
    ):
        """
        Initialize phase operator.
        
        Args:
            dim: Hilbert space dimension
            rotation_angle: Base rotation angle θ
            expansion_rate: Expansion rate r (conceptual, not dimensional)
            jump_probability: Probability of phase jump
        """
        self.dim = dim
        self.rotation_angle = rotation_angle
        self.expansion_rate = expansion_rate
        self.jump_probability = jump_probability
        
        # Build component operators
        self._build_operators()
    
    def _build_operators(self):
        """Build R, E, P operator matrices."""
        # Rotation operator R (Eq. 26-27)
        # R(S) = R(θ(S)) · S where R(θ) = exp(iθĴ)
        J = self._create_angular_momentum()
        self.R = la.expm(1j * self.rotation_angle * J)
        
        # Expansion operator E (Eq. 28-30)
        # E(S) = (1 + r(S)) · S
        self.E = (1 + self.expansion_rate) * np.eye(self.dim, dtype=np.complex128)
        
        # Phase jump operator P (Eq. 31)
        # P(S) = S + Δφ(S)
        phase_shifts = np.exp(2j * np.pi * np.random.random(self.dim))
        self.P = np.diag(phase_shifts)
    
    def _create_angular_momentum(self) -> NDArray[np.complex128]:
        """Create angular momentum-like operator J."""
        # Create a random Hermitian generator
        A = np.random.randn(self.dim, self.dim) + 1j * np.random.randn(self.dim, self.dim)
        J = (A + A.conj().T) / 2
        # Normalize
        J = J / np.linalg.norm(J, 'fro') * np.sqrt(self.dim)
        return J
    
    def rotation(self, state: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Apply rotation/torsion operation R (Eq. 26-27).
        
        Args:
            state: State vector
            
        Returns:
            Rotated state
        """
        return self.R @ state
    
    def expansion(self, state: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Apply expansion operation E (Eq. 28-30).
        
        Note: This is a conceptual expansion (amplitude scaling),
        not dimensional expansion.
        
        Args:
            state: State vector
            
        Returns:
            Expanded state
        """
        return self.E @ state
    
    def phase_jump(
        self,
        state: NDArray[np.complex128],
        apply_stochastic: bool = True
    ) -> NDArray[np.complex128]:
        """
        Apply phase jump operation P (Eq. 31).
        
        Args:
            state: State vector
            apply_stochastic: If True, apply jump with probability
            
        Returns:
            State after phase jump
        """
        if apply_stochastic and np.random.random() > self.jump_probability:
            return state.copy()
        return self.P @ state
    
    def apply(
        self,
        state: NDArray[np.complex128],
        apply_jump_stochastic: bool = True
    ) -> NDArray[np.complex128]:
        """
        Apply full phase transition operator G = P ∘ E ∘ R.
        
        Args:
            state: Input state vector
            apply_jump_stochastic: If True, phase jump is stochastic
            
        Returns:
            Transformed state
        """
        # Apply in order: R → E → P
        state_r = self.rotation(state)
        state_e = self.expansion(state_r)
        state_p = self.phase_jump(state_e, apply_jump_stochastic)
        
        # Normalize
        norm = np.linalg.norm(state_p)
        if norm > 1e-10:
            state_p = state_p / norm
        
        return state_p


class KrausChannel:
    """
    Kraus operator representation of quantum channels (Eq. 80).
    
    E(ρ) = Σₖ Kₖ ρ Kₖ†
    
    Implements:
    - Projection measurement
    - Dephasing (T2 process)
    - Amplitude damping (T1 process)
    """
    
    def __init__(self, dim: int):
        """
        Initialize Kraus channel.
        
        Args:
            dim: Hilbert space dimension
        """
        self.dim = dim
    
    @staticmethod
    def projection_channel(
        dim: int,
        measurement_basis: NDArray[np.complex128]
    ) -> List[NDArray[np.complex128]]:
        """
        Create projection measurement Kraus operators (Eq. 81).
        
        K₀ = P_obs, K₁ = I - P_obs
        
        Args:
            dim: Dimension
            measurement_basis: Measurement basis vector
            
        Returns:
            List of Kraus operators
        """
        # Projection onto measurement basis
        P = np.outer(measurement_basis, measurement_basis.conj())
        I = np.eye(dim, dtype=np.complex128)
        
        # K₀ = sqrt(P), K₁ = sqrt(I-P)
        K0 = la.sqrtm(P)
        K1 = la.sqrtm(I - P)
        
        return [K0, K1]
    
    @staticmethod
    def dephasing_channel(
        dim: int,
        gamma: float = 0.1
    ) -> List[NDArray[np.complex128]]:
        """
        Create dephasing (T2) Kraus operators (Eq. 82-83).
        
        K₀ = √(1-γ) I, K₁ = √γ σz
        
        Args:
            dim: Dimension
            gamma: Dephasing rate
            
        Returns:
            List of Kraus operators
        """
        I = np.eye(dim, dtype=np.complex128)
        
        # Create generalized σz
        sigma_z = np.diag([(-1)**i for i in range(dim)]).astype(np.complex128)
        
        K0 = np.sqrt(1 - gamma) * I
        K1 = np.sqrt(gamma) * sigma_z
        
        return [K0, K1]
    
    @staticmethod
    def amplitude_damping_channel(
        dim: int,
        gamma: float = 0.1
    ) -> List[NDArray[np.complex128]]:
        """
        Create amplitude damping (T1) Kraus operators (Eq. 84).
        
        Models |1⟩ → |0⟩ transitions (forgetting process).
        
        Args:
            dim: Dimension
            gamma: Damping rate
            
        Returns:
            List of Kraus operators
        """
        K0 = np.eye(dim, dtype=np.complex128)
        K0[1, 1] = np.sqrt(1 - gamma)
        
        K1 = np.zeros((dim, dim), dtype=np.complex128)
        K1[0, 1] = np.sqrt(gamma)
        
        return [K0, K1]
    
    @staticmethod
    def apply_channel(
        rho: NDArray[np.complex128],
        kraus_ops: List[NDArray[np.complex128]]
    ) -> NDArray[np.complex128]:
        """
        Apply quantum channel to density matrix.
        
        E(ρ) = Σₖ Kₖ ρ Kₖ†
        
        Args:
            rho: Input density matrix
            kraus_ops: List of Kraus operators
            
        Returns:
            Output density matrix
        """
        rho_out = np.zeros_like(rho)
        for K in kraus_ops:
            rho_out += K @ rho @ K.conj().T
        return rho_out


if __name__ == "__main__":
    print("twinRIG V1 Operators Test")
    print("=" * 40)
    
    # Test Hamiltonian factory
    dim = 243  # 3^5
    subspace_dims = (3, 3, 3, 3, 3)
    
    factory = HamiltonianFactory(dim, subspace_dims, seed=42)
    hamiltonians = factory.create_all_hamiltonians()
    
    for name, H in hamiltonians.items():
        print(f"{name}: shape={H.shape}, Hermitian={np.allclose(H, H.conj().T)}")
    
    # Test phase operator
    print("\nPhase Operator Test:")
    phase_op = PhaseOperator(dim)
    state = np.random.randn(dim) + 1j * np.random.randn(dim)
    state = state / np.linalg.norm(state)
    
    new_state = phase_op.apply(state)
    print(f"Input norm: {np.linalg.norm(state):.6f}")
    print(f"Output norm: {np.linalg.norm(new_state):.6f}")
    
    # Test Kraus channels
    print("\nKraus Channel Test:")
    rho = np.outer(state, state.conj())
    dephasing_ops = KrausChannel.dephasing_channel(dim, gamma=0.1)
    rho_out = KrausChannel.apply_channel(rho, dephasing_ops)
    print(f"Trace preservation: {np.trace(rho_out):.6f}")
