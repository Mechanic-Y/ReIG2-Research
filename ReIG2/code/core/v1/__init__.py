"""
twinRIG V1 - Dense Matrix Implementation
=========================================

Core quantum resonance framework with dense matrix operations.
Suitable for dimensions up to ~1,000.

Modules:
    engine: Quantum state and time evolution
    operators: Hamiltonians and phase operators

Example:
    >>> from engine import QuantumState, TimeEvolutionEngine
    >>> from operators import HamiltonianFactory
    >>> 
    >>> # Create state
    >>> state = QuantumState(subspace_dims=(3, 3, 3, 3, 3))
    >>> 
    >>> # Create Hamiltonians
    >>> factory = HamiltonianFactory(state.total_dim, state.subspace_dims)
    >>> H_terms = list(factory.create_all_hamiltonians().values())
    >>> 
    >>> # Evolve
    >>> engine = TimeEvolutionEngine(H_terms, dt=0.01)
    >>> final_state, trajectory = engine.evolve(state, n_steps=100)
"""

from .engine import QuantumState, TimeEvolutionEngine, WorldOperator
from .operators import HamiltonianFactory, PhaseOperator, KrausChannel

__version__ = "1.0.0"
__all__ = [
    "QuantumState",
    "TimeEvolutionEngine", 
    "WorldOperator",
    "HamiltonianFactory",
    "PhaseOperator",
    "KrausChannel"
]
