"""
twinRIG V2 - Scalable Quantum State Engine (Sparse Matrix Architecture)
========================================================================

CRITICAL UPGRADE from V1:
- Full sparse matrix support (scipy.sparse CSR/CSC format)
- Scales to 32,768+ dimensions (15+ qubits)
- Memory: O(nnz) instead of O(n²) where nnz << n²
- Evolution: expm_multiply instead of dense expm
- Computational complexity reduced from O(n³) to O(nnz) per operation

Innovation 1: Sparse Matrix Implementation
- Compressed Sparse Row (CSR) representation
- Krylov subspace methods for evolution (expm_multiply)
- Memory: O(nnz) where nnz << n² for quantum resonance operators
- Evolution: O(nnz·m) where m ≈ 30-50 << n

Innovation 2: Multi-Axis Non-Commutative Time
- Four distinct Hamiltonian generators G^(k) for k ∈ {0,1,2,3}
- Each generator couples different subspace pairs
- Non-zero commutators: ||[G^(i), G^(j)]|| > 0 for i ≠ j

Reference: ReIG2_twinRIG_2025_December.pdf, Section 3 (Multi-dimensional Time)
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Union, Dict, Callable
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, LinearOperator
from enum import Enum


class SubspaceType(Enum):
    """Enumeration of the 5 fundamental subspaces."""
    MEANING = "meaning"
    CONTEXT = "context"
    ETHICS = "ethics"
    FUTURE = "future"
    STABILITY = "stability"


class TimeAxis(Enum):
    """Four temporal axes for multi-dimensional time evolution (Section 3.2)."""
    PHYSICAL = 0    # Standard physical time
    CULTURAL = 1    # Cultural/semantic time
    SOCIAL = 2      # Social interaction time
    PERSONAL = 3    # Subjective personal time


@dataclass
class SubspaceConfig:
    """Configuration for each subspace dimension."""
    subspace_type: SubspaceType
    dimension: int
    basis_labels: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.dimension < 2:
            raise ValueError(f"Subspace dimension must be ≥2, got {self.dimension}")


@dataclass
class V2Config:
    """Configuration for V2 engine."""
    subspace_dims: Tuple[int, ...] = (5, 5, 5, 5, 5)  # Default: 3125 dimensions
    sparse_threshold: float = 0.01  # Sparsity threshold for switching to sparse
    use_sparse: bool = True
    krylov_dimension: int = 30  # Krylov subspace dimension for expm_multiply
    seed: Optional[int] = None


class QuantumState:
    """
    Quantum state |Ψ⟩ in composite Hilbert space.
    
    V2 Note: State vectors remain dense (they're typically not sparse),
    but operators are sparse for efficient evolution.
    """
    
    def __init__(
        self,
        amplitudes: Optional[NDArray[np.complex128]] = None,
        subspace_dims: Tuple[int, ...] = (5, 5, 5, 5, 5),
        normalize: bool = True
    ):
        self.subspace_dims = subspace_dims
        self.total_dim = int(np.prod(subspace_dims))
        
        if amplitudes is None:
            self.amplitudes = np.ones(self.total_dim, dtype=np.complex128)
        else:
            if len(amplitudes) != self.total_dim:
                raise ValueError(
                    f"Amplitude dimension {len(amplitudes)} != total dimension {self.total_dim}"
                )
            self.amplitudes = np.array(amplitudes, dtype=np.complex128)
        
        if normalize:
            self.normalize()
    
    def normalize(self) -> QuantumState:
        norm = np.linalg.norm(self.amplitudes)
        if norm > 1e-10:
            self.amplitudes /= norm
        return self
    
    def norm(self) -> float:
        return float(np.linalg.norm(self.amplitudes))
    
    def inner_product(self, other: QuantumState) -> complex:
        if self.total_dim != other.total_dim:
            raise ValueError("States must have same dimension")
        return complex(np.vdot(other.amplitudes, self.amplitudes))
    
    def expectation_value(self, operator: Union[sp.spmatrix, NDArray]) -> float:
        """Calculate ⟨Ψ|Ô|Ψ⟩ with sparse operator support."""
        if sp.issparse(operator):
            result = np.vdot(self.amplitudes, operator @ self.amplitudes)
        else:
            result = np.vdot(self.amplitudes, operator @ self.amplitudes)
        return float(np.real(result))
    
    def measure_subspace(self, subspace_index: int) -> NDArray[np.float64]:
        if subspace_index < 0 or subspace_index >= len(self.subspace_dims):
            raise ValueError(f"Invalid subspace index: {subspace_index}")
        
        tensor = self.amplitudes.reshape(self.subspace_dims)
        probs = np.abs(tensor) ** 2
        axes_to_sum = tuple(i for i in range(len(self.subspace_dims)) if i != subspace_index)
        return np.sum(probs, axis=axes_to_sum)
    
    def fidelity(self, other: QuantumState) -> float:
        return float(np.abs(self.inner_product(other)) ** 2)
    
    def copy(self) -> QuantumState:
        return QuantumState(
            amplitudes=self.amplitudes.copy(),
            subspace_dims=self.subspace_dims,
            normalize=False
        )
    
    def __repr__(self) -> str:
        return f"QuantumState(dim={self.total_dim}, subspaces={self.subspace_dims})"


class SparseTimeEvolutionEngine:
    """
    V2 Sparse Time Evolution Engine.
    
    Key improvements over V1:
    1. Sparse Hamiltonian storage (CSR format)
    2. Krylov subspace methods for evolution (no matrix exponential computation)
    3. Multi-axis time evolution support
    4. Scales to 30,000+ dimensions
    
    Evolution computed via:
    |Ψ(t+dt)⟩ = expm_multiply(-i H dt, |Ψ(t)⟩)
    
    This uses Krylov subspace approximation with O(nnz·m) complexity
    where nnz = number of non-zeros, m = Krylov dimension (~30-50).
    """
    
    def __init__(
        self,
        hamiltonian_terms: List[sp.spmatrix],
        dt: float = 0.01,
        hbar: float = 1.0,
        method: str = 'krylov'
    ):
        """
        Initialize sparse evolution engine.
        
        Args:
            hamiltonian_terms: List of sparse Hamiltonian matrices
            dt: Time step
            hbar: Planck constant (natural units: 1.0)
            method: Evolution method ('krylov' or 'trotter')
        """
        self.hamiltonian_terms = hamiltonian_terms
        self.dt = dt
        self.hbar = hbar
        self.method = method
        self.dim = hamiltonian_terms[0].shape[0]
        
        # Verify dimensions
        for H in hamiltonian_terms:
            if H.shape != (self.dim, self.dim):
                raise ValueError("All Hamiltonians must have same dimension")
        
        # Convert to CSR for efficient operations
        self.hamiltonian_terms = [sp.csr_matrix(H) for H in hamiltonian_terms]
    
    def evolve_step(
        self,
        state: QuantumState,
        coefficients: Optional[List[float]] = None
    ) -> QuantumState:
        """
        Evolve state by one time step using sparse methods.
        
        Uses scipy.sparse.linalg.expm_multiply for efficient computation
        without forming the full matrix exponential.
        """
        if coefficients is None:
            coefficients = [1.0] * len(self.hamiltonian_terms)
        
        if len(coefficients) != len(self.hamiltonian_terms):
            raise ValueError("Coefficient count must match Hamiltonian count")
        
        # Build total sparse Hamiltonian
        H_total = sum(c * H for c, H in zip(coefficients, self.hamiltonian_terms))
        
        if self.method == 'krylov':
            # Krylov subspace evolution: |Ψ(t+dt)⟩ = exp(-iHdt)|Ψ(t)⟩
            # expm_multiply computes this WITHOUT forming the full exponential
            new_amplitudes = expm_multiply(
                -1j * H_total * self.dt / self.hbar,
                state.amplitudes
            )
        else:
            # Trotter decomposition (still sparse)
            new_amplitudes = state.amplitudes.copy()
            for c, H in zip(coefficients, self.hamiltonian_terms):
                new_amplitudes = expm_multiply(
                    -1j * c * H * self.dt / self.hbar,
                    new_amplitudes
                )
        
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
        callback: Optional[Callable] = None
    ) -> Tuple[QuantumState, List[QuantumState]]:
        """Evolve state for multiple steps."""
        current_state = state.copy()
        trajectory = [current_state.copy()]
        
        for step in range(n_steps):
            current_state = self.evolve_step(current_state, coefficients)
            trajectory.append(current_state.copy())
            
            if callback is not None:
                callback(step, current_state)
        
        return current_state, trajectory
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Calculate memory usage of sparse Hamiltonians."""
        total_nnz = sum(H.nnz for H in self.hamiltonian_terms)
        total_bytes = sum(H.data.nbytes + H.indices.nbytes + H.indptr.nbytes 
                        for H in self.hamiltonian_terms)
        dense_bytes = len(self.hamiltonian_terms) * self.dim * self.dim * 16  # complex128
        
        return {
            'sparse_bytes': total_bytes,
            'dense_bytes': dense_bytes,
            'compression_ratio': dense_bytes / max(total_bytes, 1),
            'total_nnz': total_nnz,
            'sparsity': 1 - total_nnz / (len(self.hamiltonian_terms) * self.dim * self.dim)
        }


class MultiAxisEvolutionEngine:
    """
    Multi-Axis Non-Commutative Time Evolution Engine (Section 3).
    
    Implements multi-dimensional time evolution (Eq. 13):
    Û_multi(τ, ε, PFH) = exp(-i Σₖ Ĥₖ fₖ(τ, ε, PFH) / ℏ)
    
    Four temporal axes with non-commuting generators:
    - Physical time (k=0)
    - Cultural time (k=1)
    - Social time (k=2)
    - Personal time (k=3)
    
    Non-commutativity: [G^(i), G^(j)] ≠ 0 for i ≠ j
    """
    
    def __init__(
        self,
        generators: Dict[TimeAxis, sp.spmatrix],
        weight_functions: Optional[Dict[TimeAxis, Callable]] = None,
        dt: float = 0.01
    ):
        """
        Initialize multi-axis engine.
        
        Args:
            generators: Dict mapping TimeAxis → sparse Hamiltonian generator
            weight_functions: Dict mapping TimeAxis → f_k(τ, ε, PFH)
            dt: Time step
        """
        self.generators = generators
        self.dt = dt
        self.dim = list(generators.values())[0].shape[0]
        
        # Default weight functions (Eq. 15)
        if weight_functions is None:
            self.weight_functions = {
                TimeAxis.PHYSICAL: lambda tau, eps, pfh: 1 + tau,
                TimeAxis.CULTURAL: lambda tau, eps, pfh: tau * np.exp(-eps**2 / 2),
                TimeAxis.SOCIAL: lambda tau, eps, pfh: pfh * np.sqrt(tau + 0.01),
                TimeAxis.PERSONAL: lambda tau, eps, pfh: eps * np.sin(np.pi * tau / 2)
            }
        else:
            self.weight_functions = weight_functions
    
    def compute_commutator_norm(
        self,
        axis1: TimeAxis,
        axis2: TimeAxis
    ) -> float:
        """
        Compute ||[G^(axis1), G^(axis2)]|| (Section 3.3).
        
        Non-zero commutator indicates non-commutative time axes.
        """
        G1 = self.generators[axis1]
        G2 = self.generators[axis2]
        
        # [A, B] = AB - BA
        commutator = G1 @ G2 - G2 @ G1
        
        # Frobenius norm of sparse matrix
        return float(sp.linalg.norm(commutator, 'fro'))
    
    def analyze_commutators(self) -> Dict[Tuple[TimeAxis, TimeAxis], float]:
        """Analyze all commutator norms."""
        results = {}
        axes = list(self.generators.keys())
        for i, ax1 in enumerate(axes):
            for ax2 in axes[i+1:]:
                results[(ax1, ax2)] = self.compute_commutator_norm(ax1, ax2)
        return results
    
    def evolve_step(
        self,
        state: QuantumState,
        tau: float = 0.5,
        epsilon: float = 0.3,
        pfh: float = 0.2,
        use_trotter: bool = True,
        trotter_steps: int = 10
    ) -> QuantumState:
        """
        Evolve state along all time axes simultaneously.
        
        If use_trotter=True, applies Trotter-Suzuki decomposition (Eq. 17):
        U ≈ [Πₖ exp(-i Gₖ fₖ dt/M)]^M
        
        Args:
            state: Input state
            tau: Time resonance τ
            epsilon: Entropy resonance ε
            pfh: Philosophical resonance PFH
            use_trotter: Use Trotter decomposition for non-commuting case
            trotter_steps: Number of Trotter steps M
        """
        # Compute weight functions
        weights = {
            axis: self.weight_functions[axis](tau, epsilon, pfh)
            for axis in self.generators.keys()
        }
        
        if use_trotter:
            # Symmetric Trotter-Suzuki (Eq. 20)
            # U_sym = Πₖ exp(-i Gₖ dt/2) · Π'ₖ exp(-i Gₖ dt/2)
            new_amplitudes = state.amplitudes.copy()
            
            for _ in range(trotter_steps):
                dt_step = self.dt / trotter_steps
                
                # Forward sweep
                for axis, G in self.generators.items():
                    w = weights[axis]
                    new_amplitudes = expm_multiply(
                        -1j * w * G * dt_step / 2,
                        new_amplitudes
                    )
                
                # Backward sweep
                for axis in reversed(list(self.generators.keys())):
                    G = self.generators[axis]
                    w = weights[axis]
                    new_amplitudes = expm_multiply(
                        -1j * w * G * dt_step / 2,
                        new_amplitudes
                    )
        else:
            # Sum all generators (approximation for weakly non-commuting case)
            H_total = sum(weights[ax] * G for ax, G in self.generators.items())
            new_amplitudes = expm_multiply(
                -1j * H_total * self.dt,
                state.amplitudes
            )
        
        return QuantumState(
            amplitudes=new_amplitudes,
            subspace_dims=state.subspace_dims,
            normalize=True
        )
    
    def evolve(
        self,
        state: QuantumState,
        n_steps: int,
        tau: float = 0.5,
        epsilon: float = 0.3,
        pfh: float = 0.2,
        callback: Optional[Callable] = None
    ) -> Tuple[QuantumState, List[QuantumState]]:
        """Evolve state for multiple steps along all time axes."""
        current = state.copy()
        trajectory = [current.copy()]
        
        for step in range(n_steps):
            current = self.evolve_step(current, tau, epsilon, pfh)
            trajectory.append(current.copy())
            
            if callback is not None:
                callback(step, current)
        
        return current, trajectory


class SparseWorldOperator:
    """
    V2 Sparse World Construction Operator.
    
    Implements T̂_World = T_I ∘ T_R ∘ T_C with sparse matrices
    for large-scale systems.
    """
    
    def __init__(
        self,
        dim: int,
        subspace_dims: Tuple[int, ...],
        contraction_factor: float = 0.95,
        sparsity: float = 0.05
    ):
        """
        Initialize sparse world operator.
        
        Args:
            dim: Total dimension
            subspace_dims: Subspace dimensions
            contraction_factor: κ < 1 for Banach contraction
            sparsity: Fraction of non-zero elements
        """
        self.dim = dim
        self.subspace_dims = subspace_dims
        self.contraction_factor = contraction_factor
        self.sparsity = sparsity
        
        self._build_sparse_operators()
    
    def _build_sparse_operators(self):
        """Build sparse transformation operators."""
        # Create sparse random matrices with controlled sparsity
        self.T_C = self._make_sparse_contraction()
        self.T_R = self._make_sparse_contraction()
        self.T_I = self._make_sparse_contraction()
    
    def _make_sparse_contraction(self) -> sp.csr_matrix:
        """Create sparse contraction operator."""
        # Random sparse matrix
        nnz = int(self.dim * self.dim * self.sparsity)
        rows = np.random.randint(0, self.dim, nnz)
        cols = np.random.randint(0, self.dim, nnz)
        data = (np.random.randn(nnz) + 1j * np.random.randn(nnz)) / np.sqrt(nnz)
        
        T = sp.csr_matrix((data, (rows, cols)), shape=(self.dim, self.dim))
        
        # Scale to have spectral radius < 1
        # Approximate spectral norm
        v = np.random.randn(self.dim)
        for _ in range(10):
            v = T @ v
            v = v / np.linalg.norm(v)
        approx_norm = np.linalg.norm(T @ v)
        
        T = T * (self.contraction_factor / max(approx_norm, 0.1))
        
        return sp.csr_matrix(T)
    
    def apply(self, state: QuantumState) -> QuantumState:
        """Apply world operator using sparse matrices."""
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
        """Find fixed point using sparse iterations."""
        current = initial_state.copy()
        history = []
        
        for iteration in range(max_iterations):
            next_state = self.apply(current)
            distance = np.linalg.norm(next_state.amplitudes - current.amplitudes)
            history.append(distance)
            
            if distance < tolerance:
                return next_state, iteration + 1, history
            
            current = next_state
        
        return current, max_iterations, history
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        total_nnz = self.T_C.nnz + self.T_R.nnz + self.T_I.nnz
        sparse_bytes = sum(
            T.data.nbytes + T.indices.nbytes + T.indptr.nbytes
            for T in [self.T_C, self.T_R, self.T_I]
        )
        dense_bytes = 3 * self.dim * self.dim * 16
        
        return {
            'total_nnz': total_nnz,
            'sparse_bytes': sparse_bytes,
            'dense_bytes_equivalent': dense_bytes,
            'compression_ratio': dense_bytes / max(sparse_bytes, 1),
            'sparsity_actual': 1 - total_nnz / (3 * self.dim * self.dim)
        }


if __name__ == "__main__":
    print("twinRIG V2 Sparse Engine Test")
    print("=" * 50)
    
    # Test with larger dimension
    subspace_dims = (5, 5, 5, 5, 5)  # 3125 dimensions
    dim = int(np.prod(subspace_dims))
    
    print(f"\nConfiguration:")
    print(f"  Subspace dims: {subspace_dims}")
    print(f"  Total dimension: {dim}")
    
    # Create state
    state = QuantumState(subspace_dims=subspace_dims)
    print(f"  State created: {state}")
    
    # Test sparse world operator
    world_op = SparseWorldOperator(dim, subspace_dims, sparsity=0.01)
    stats = world_op.get_memory_stats()
    print(f"\nSparse World Operator:")
    print(f"  Compression ratio: {stats['compression_ratio']:.1f}x")
    print(f"  Actual sparsity: {stats['sparsity_actual']*100:.1f}%")
    
    # Find fixed point
    print("\nFinding fixed point...")
    fixed, iters, history = world_op.find_fixed_point(state, max_iterations=200)
    print(f"  Converged in {iters} iterations")
    print(f"  Final distance: {history[-1]:.2e}")
