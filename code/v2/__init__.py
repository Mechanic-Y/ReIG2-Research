"""
twinRIG V2 - Sparse Matrix Implementation
==========================================

Scalable quantum resonance framework with sparse matrix operations.
Supports dimensions up to 30,000+ with 100-900x memory compression.

Key Improvements over V1:
    - Sparse matrix storage (CSR format)
    - Krylov subspace methods for evolution
    - Multi-axis non-commutative time
    - O(nnz) memory instead of O(n²)

Modules:
    engine_v2: Sparse quantum state and evolution
    operators_v2: Sparse Hamiltonians and operators

Example:
    >>> from engine_v2 import QuantumState, SparseTimeEvolutionEngine
    >>> from operators_v2 import SparseHamiltonianFactory
    >>> 
    >>> # Create state (larger dimension possible)
    >>> state = QuantumState(subspace_dims=(5, 5, 5, 5, 5))  # 3125 dim
    >>> 
    >>> # Create sparse Hamiltonians
    >>> factory = SparseHamiltonianFactory(state.total_dim, state.subspace_dims)
    >>> H_terms = list(factory.create_all_hamiltonians().values())
    >>> 
    >>> # Evolve with Krylov methods
    >>> engine = SparseTimeEvolutionEngine(H_terms, dt=0.01)
    >>> final_state, trajectory = engine.evolve(state, n_steps=100)
"""

from .engine_v2 import (
    QuantumState,
    SparseTimeEvolutionEngine,
    MultiAxisEvolutionEngine,
    SparseWorldOperator,
    TimeAxis
)
from .operators_v2 import (
    SparseHamiltonianFactory,
    SparsePhaseOperator,
    SparseKrausChannel
)

__version__ = "2.0.0"
__all__ = [
    # Engine
    "QuantumState",
    "SparseTimeEvolutionEngine",
    "MultiAxisEvolutionEngine",
    "SparseWorldOperator",
    "TimeAxis",
    # Operators
    "SparseHamiltonianFactory",
    "SparsePhaseOperator",
    "SparseKrausChannel"
]
