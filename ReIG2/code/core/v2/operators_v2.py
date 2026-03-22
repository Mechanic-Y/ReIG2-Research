"""
twinRIG V2 - Sparse Quantum Operators Module
=============================================

Sparse matrix implementation of quantum operators for scalability.

Key Features:
- CSR/CSC sparse matrix format
- O(nnz) memory instead of O(n²)
- Efficient Hamiltonian construction with controlled sparsity
- Multi-axis generator construction

Reference: ReIG2_twinRIG_2025_December.pdf
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator
from enum import Enum, auto

from engine_v2 import TimeAxis


class OperatorType(Enum):
    """Types of operators in the framework."""
    BASE = auto()
    FUTURE = auto()
    ENTROPY = auto()
    ETHICS = auto()
    MULTI_AXIS = auto()


@dataclass
class SparseOperatorConfig:
    """Configuration for sparse operator construction."""
    dim: int
    coupling_strength: float = 0.5
    target_sparsity: float = 0.99  # Fraction of zeros
    seed: Optional[int] = None


class SparseHamiltonianFactory:
    """
    V2 Sparse Hamiltonian Factory.
    
    Creates sparse Hamiltonian operators with controlled sparsity.
    Achieves 99%+ sparsity for typical quantum resonance operators.
    """
    
    def __init__(
        self,
        dim: int,
        subspace_dims: Tuple[int, ...],
        coupling_strength: float = 0.3,
        seed: Optional[int] = None,
        target_sparsity: float = 0.99
    ):
        """
        Initialize sparse Hamiltonian factory.
        
        Args:
            dim: Total Hilbert space dimension
            subspace_dims: Dimensions of each subspace
            coupling_strength: Coupling parameter
            seed: Random seed
            target_sparsity: Target fraction of zero elements (0.99 = 99% zeros)
        """
        self.dim = dim
        self.subspace_dims = subspace_dims
        self.coupling_strength = coupling_strength
        self.target_sparsity = target_sparsity
        
        if seed is not None:
            np.random.seed(seed)
        
        self.omega = {
            'M': 1.0, 'C': 0.7, 'E': 0.5, 'F': 0.8, 'S': 0.3
        }
    
    def create_base_hamiltonian(self) -> sp.csr_matrix:
        """
        Create sparse base Hamiltonian H₀.
        
        Diagonal operator: H₀ = Σₖ ωₖ Nₖ
        Sparsity: (n-n)/n² = 1 - 1/n ≈ 99.97% for n=3125
        """
        # Build diagonal terms efficiently
        diag_values = np.zeros(self.dim, dtype=np.complex128)
        
        for i in range(self.dim):
            idx = self._linear_to_multi_index(i)
            energy = sum(
                self.omega[key] * idx[j]
                for j, key in enumerate(['M', 'C', 'E', 'F', 'S'])
            )
            diag_values[i] = energy
        
        return sp.diags(diag_values, format='csr')
    
    def create_future_hamiltonian(self) -> sp.csr_matrix:
        """
        Create sparse future contribution Hamiltonian.
        
        Uses ladder operators that couple adjacent states in future subspace.
        """
        rows, cols, data = [], [], []
        
        d_F = self.subspace_dims[3]  # Future dimension
        
        for i in range(self.dim):
            idx_i = self._linear_to_multi_index(i)
            
            # Raise future index
            if idx_i[3] < d_F - 1:
                idx_j = list(idx_i)
                idx_j[3] += 1
                j = self._multi_to_linear_index(tuple(idx_j))
                
                coupling = self.coupling_strength * self.omega['F'] * np.sqrt(idx_i[3] + 1)
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([coupling, coupling])
        
        H = sp.csr_matrix((data, (rows, cols)), shape=(self.dim, self.dim))
        return H
    
    def create_entropy_hamiltonian(self) -> sp.csr_matrix:
        """
        Create sparse entropy/fluctuation Hamiltonian.
        
        Random sparse couplings with controlled density.
        """
        # Calculate number of non-zero off-diagonal elements
        nnz_target = int(self.dim * self.dim * (1 - self.target_sparsity) / 2)
        nnz_target = max(nnz_target, self.dim)  # At least n elements
        
        rows, cols, data = [], [], []
        
        # Random sparse off-diagonal couplings
        for _ in range(nnz_target):
            i = np.random.randint(0, self.dim)
            j = np.random.randint(0, self.dim)
            if i < j:
                coupling = self.coupling_strength * np.random.randn() * 0.1
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([coupling, coupling])
        
        # Add diagonal entropy terms
        for i in range(self.dim):
            idx = self._linear_to_multi_index(i)
            entropy_val = sum(idx) / sum(d-1 for d in self.subspace_dims) * self.omega['E']
            rows.append(i)
            cols.append(i)
            data.append(entropy_val)
        
        H = sp.csr_matrix((data, (rows, cols)), shape=(self.dim, self.dim))
        return H
    
    def create_ethics_hamiltonian(self) -> sp.csr_matrix:
        """
        Create sparse ethics Hamiltonian.
        
        Couples ethics and stability subspaces.
        """
        rows, cols, data = [], [], []
        
        d_E = self.subspace_dims[2]
        d_S = self.subspace_dims[4]
        
        for i in range(self.dim):
            idx_i = self._linear_to_multi_index(i)
            
            # Couple to adjacent ethics states
            if idx_i[2] < d_E - 1:
                idx_j = list(idx_i)
                idx_j[2] += 1
                j = self._multi_to_linear_index(tuple(idx_j))
                
                coupling = self.coupling_strength * self.omega['E'] * 0.5
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([coupling, coupling])
            
            # Couple to adjacent stability states
            if idx_i[4] < d_S - 1:
                idx_j = list(idx_i)
                idx_j[4] += 1
                j = self._multi_to_linear_index(tuple(idx_j))
                
                coupling = self.coupling_strength * self.omega['S'] * 0.3
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([coupling, coupling])
        
        H = sp.csr_matrix((data, (rows, cols)), shape=(self.dim, self.dim))
        return H
    
    def create_all_hamiltonians(self) -> Dict[str, sp.csr_matrix]:
        """Create all Hamiltonian components as sparse matrices."""
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
    ) -> sp.csr_matrix:
        """
        Create full sparse extended Hamiltonian.
        
        Ĥ = H₀ + τH_future + εH_entropy + PFH·H_ethics
        """
        hamiltonians = self.create_all_hamiltonians()
        
        H = (hamiltonians['H0'] +
             tau * hamiltonians['H_future'] +
             epsilon * hamiltonians['H_entropy'] +
             pfh * hamiltonians['H_ethics'])
        
        return H.tocsr()
    
    def create_multi_axis_generators(self) -> Dict[TimeAxis, sp.csr_matrix]:
        """
        Create generators for multi-axis time evolution (Section 3).
        
        Each generator couples different subspace pairs to create
        non-commuting time axes.
        """
        generators = {}
        
        # Physical time: couples Meaning-Context
        generators[TimeAxis.PHYSICAL] = self._create_subspace_coupling(0, 1, 1.0)
        
        # Cultural time: couples Context-Ethics
        generators[TimeAxis.CULTURAL] = self._create_subspace_coupling(1, 2, 0.8)
        
        # Social time: couples Ethics-Future
        generators[TimeAxis.SOCIAL] = self._create_subspace_coupling(2, 3, 0.6)
        
        # Personal time: couples Future-Stability
        generators[TimeAxis.PERSONAL] = self._create_subspace_coupling(3, 4, 0.4)
        
        return generators
    
    def _create_subspace_coupling(
        self,
        idx1: int,
        idx2: int,
        strength: float
    ) -> sp.csr_matrix:
        """Create sparse coupling between two subspaces."""
        rows, cols, data = [], [], []
        
        d1 = self.subspace_dims[idx1]
        d2 = self.subspace_dims[idx2]
        
        for i in range(self.dim):
            idx_i = self._linear_to_multi_index(i)
            
            # Raise first index, lower second (or vice versa)
            if idx_i[idx1] < d1 - 1 and idx_i[idx2] > 0:
                idx_j = list(idx_i)
                idx_j[idx1] += 1
                idx_j[idx2] -= 1
                j = self._multi_to_linear_index(tuple(idx_j))
                
                coupling = strength * self.coupling_strength
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([coupling, coupling])
        
        # Add diagonal terms for Hermiticity
        for i in range(self.dim):
            idx = self._linear_to_multi_index(i)
            diag_val = strength * (idx[idx1] - idx[idx2]) * 0.1
            rows.append(i)
            cols.append(i)
            data.append(diag_val)
        
        G = sp.csr_matrix((data, (rows, cols)), shape=(self.dim, self.dim))
        return G
    
    def _linear_to_multi_index(self, linear_idx: int) -> Tuple[int, ...]:
        """Convert linear index to multi-index."""
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
    
    def get_sparsity_stats(self) -> Dict[str, float]:
        """Calculate sparsity statistics for all Hamiltonians."""
        hamiltonians = self.create_all_hamiltonians()
        
        stats = {}
        total_nnz = 0
        total_elements = 0
        
        for name, H in hamiltonians.items():
            nnz = H.nnz
            n_elements = H.shape[0] * H.shape[1]
            sparsity = 1 - nnz / n_elements
            
            stats[name] = {
                'nnz': nnz,
                'sparsity': sparsity,
                'bytes': H.data.nbytes + H.indices.nbytes + H.indptr.nbytes
            }
            
            total_nnz += nnz
            total_elements += n_elements
        
        stats['total'] = {
            'nnz': total_nnz,
            'sparsity': 1 - total_nnz / total_elements,
            'compression_vs_dense': total_elements * 16 / max(sum(s['bytes'] for s in stats.values() if 'bytes' in s), 1)
        }
        
        return stats


class SparsePhaseOperator:
    """
    V2 Sparse Phase Transition Operator.
    
    G = P ∘ E ∘ R with sparse implementation.
    """
    
    def __init__(
        self,
        dim: int,
        rotation_angle: float = np.pi / 4,
        expansion_rate: float = 0.1,
        jump_probability: float = 0.1,
        sparsity: float = 0.95
    ):
        self.dim = dim
        self.rotation_angle = rotation_angle
        self.expansion_rate = expansion_rate
        self.jump_probability = jump_probability
        self.sparsity = sparsity
        
        self._build_sparse_operators()
    
    def _build_sparse_operators(self):
        """Build sparse R, E, P operators."""
        # Rotation: sparse approximation via local couplings
        self.R = self._build_sparse_rotation()
        
        # Expansion: diagonal scaling
        expansion_diag = (1 + self.expansion_rate) * np.ones(self.dim)
        self.E = sp.diags(expansion_diag, format='csr')
        
        # Phase jump: diagonal phase shifts
        phases = np.exp(2j * np.pi * np.random.random(self.dim))
        self.P = sp.diags(phases, format='csr')
    
    def _build_sparse_rotation(self) -> sp.csr_matrix:
        """Build sparse rotation operator."""
        # Create sparse Hermitian generator
        nnz_target = int(self.dim * (1 - self.sparsity))
        nnz_target = max(nnz_target, self.dim)
        
        rows, cols, data = [], [], []
        
        # Diagonal
        for i in range(self.dim):
            rows.append(i)
            cols.append(i)
            data.append(np.cos(self.rotation_angle))
        
        # Off-diagonal (sparse)
        for _ in range(nnz_target // 2):
            i = np.random.randint(0, self.dim)
            j = np.random.randint(0, self.dim)
            if i < j:
                val = 1j * np.sin(self.rotation_angle) / np.sqrt(nnz_target)
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([val, -val])  # Anti-symmetric imaginary part
        
        R = sp.csr_matrix((data, (rows, cols)), shape=(self.dim, self.dim))
        return R
    
    def apply(
        self,
        state: NDArray[np.complex128],
        apply_jump_stochastic: bool = True
    ) -> NDArray[np.complex128]:
        """Apply G = P ∘ E ∘ R."""
        # R → E → P
        state_r = self.R @ state
        state_e = self.E @ state_r
        
        if apply_jump_stochastic and np.random.random() > self.jump_probability:
            state_p = state_e
        else:
            state_p = self.P @ state_e
        
        # Normalize
        norm = np.linalg.norm(state_p)
        if norm > 1e-10:
            state_p = state_p / norm
        
        return state_p


class SparseKrausChannel:
    """Sparse Kraus channel implementation."""
    
    @staticmethod
    def dephasing_channel(dim: int, gamma: float = 0.1) -> List[sp.csr_matrix]:
        """Create sparse dephasing Kraus operators."""
        I = sp.eye(dim, format='csr', dtype=np.complex128)
        
        # Generalized σz as diagonal
        sigma_z_diag = np.array([(-1)**i for i in range(dim)], dtype=np.complex128)
        sigma_z = sp.diags(sigma_z_diag, format='csr')
        
        K0 = np.sqrt(1 - gamma) * I
        K1 = np.sqrt(gamma) * sigma_z
        
        return [K0, K1]
    
    @staticmethod
    def apply_channel(
        rho: sp.spmatrix,
        kraus_ops: List[sp.spmatrix]
    ) -> sp.spmatrix:
        """Apply quantum channel to density matrix."""
        rho_out = sp.csr_matrix(rho.shape, dtype=np.complex128)
        for K in kraus_ops:
            rho_out += K @ rho @ K.conj().T
        return rho_out


def benchmark_sparse_vs_dense(dims: List[int]) -> Dict[int, Dict]:
    """Benchmark sparse vs dense for different dimensions."""
    import time
    
    results = {}
    
    for dim in dims:
        # Find subspace dims that give approximately this total dim
        d = int(np.round(dim ** 0.2))
        subspace_dims = (d, d, d, d, d)
        actual_dim = d ** 5
        
        print(f"\nDimension: {actual_dim} (subspace: {d})")
        
        # Sparse
        t0 = time.time()
        factory_sparse = SparseHamiltonianFactory(actual_dim, subspace_dims)
        H_sparse = factory_sparse.create_extended_hamiltonian()
        t_sparse = time.time() - t0
        
        sparse_bytes = H_sparse.data.nbytes + H_sparse.indices.nbytes + H_sparse.indptr.nbytes
        
        results[actual_dim] = {
            'sparse_time': t_sparse,
            'sparse_bytes': sparse_bytes,
            'sparse_nnz': H_sparse.nnz,
            'sparsity': 1 - H_sparse.nnz / (actual_dim ** 2),
            'dense_bytes_equivalent': actual_dim ** 2 * 16
        }
        
        print(f"  Sparse: {t_sparse:.3f}s, {sparse_bytes/1024:.1f}KB, {results[actual_dim]['sparsity']*100:.2f}% sparse")
    
    return results


if __name__ == "__main__":
    print("twinRIG V2 Sparse Operators Test")
    print("=" * 50)
    
    # Test configuration
    subspace_dims = (5, 5, 5, 5, 5)  # 3125 dimensions
    dim = int(np.prod(subspace_dims))
    
    print(f"\nDimension: {dim}")
    
    # Create sparse factory
    factory = SparseHamiltonianFactory(dim, subspace_dims, seed=42)
    
    # Get sparsity stats
    stats = factory.get_sparsity_stats()
    
    print("\nSparsity Statistics:")
    for name, s in stats.items():
        if isinstance(s, dict):
            print(f"  {name}: nnz={s.get('nnz', 'N/A')}, sparsity={s.get('sparsity', 0)*100:.2f}%")
    
    # Create multi-axis generators
    print("\nMulti-Axis Generators:")
    generators = factory.create_multi_axis_generators()
    for axis, G in generators.items():
        print(f"  {axis.name}: nnz={G.nnz}, shape={G.shape}")
    
    # Test commutators
    print("\nCommutator Norms ||[G^(i), G^(j)]||:")
    axes = list(generators.keys())
    for i, ax1 in enumerate(axes):
        for ax2 in axes[i+1:]:
            G1, G2 = generators[ax1], generators[ax2]
            comm = G1 @ G2 - G2 @ G1
            comm_norm = sp.linalg.norm(comm, 'fro')
            print(f"  [{ax1.name}, {ax2.name}]: {comm_norm:.4f}")
