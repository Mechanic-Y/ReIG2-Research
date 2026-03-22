# ReIG2/twinRIG Implementation Notes

Technical documentation for the code implementation.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [V1 Dense Implementation](#v1-dense-implementation)
3. [V2 Sparse Implementation](#v2-sparse-implementation)
4. [Key Algorithms](#key-algorithms)
5. [Performance Optimization](#performance-optimization)
6. [Testing and Validation](#testing-and-validation)

---

## Architecture Overview

### Module Structure

```
code/
├── v1/                          # Dense matrix implementation
│   ├── engine.py               # Core quantum state & evolution
│   ├── operators.py            # Hamiltonian & phase operators
│   └── demo.py                 # Demonstration script
├── v2/                          # Sparse matrix implementation  
│   ├── engine_v2.py            # Scalable sparse engine
│   ├── operators_v2.py         # Sparse operators
│   └── demo_v2.py              # V2 demonstration
├── non_unitary_quantum.py       # Kraus & Lindblad
├── reig2_full_simulation.py     # 3-qubit system
├── quantum_circuit_implementation.py  # Qiskit circuits
└── figure_generation.py         # Paper visualizations
```

### Class Hierarchy

```
QuantumState
├── V1: Dense numpy arrays
└── V2: Dense state (sparse operators)

TimeEvolutionEngine
├── V1: Dense matrix expm
└── V2 SparseTimeEvolutionEngine: Krylov methods

HamiltonianFactory
├── V1: Dense matrices
└── V2 SparseHamiltonianFactory: CSR sparse

WorldOperator
├── V1: Dense transformations
└── V2 SparseWorldOperator: Sparse contractions
```

---

## V1 Dense Implementation

### QuantumState Class

```python
class QuantumState:
    """
    State vector |Ψ⟩ in composite Hilbert space.
    
    Attributes:
        amplitudes: NDArray[complex128] - state vector
        subspace_dims: Tuple[int, ...] - (d_M, d_C, d_E, d_F, d_S)
        total_dim: int - product of subspace dims
    """
```

**Key Methods**:
- `normalize()`: Ensure ||Ψ|| = 1
- `inner_product(other)`: Compute ⟨Φ|Ψ⟩
- `expectation_value(O)`: Compute ⟨Ψ|O|Ψ⟩
- `measure_subspace(i)`: Marginal probability distribution
- `fidelity(other)`: |⟨Ψ|Φ⟩|²

### HamiltonianFactory Class

Constructs the extended Hamiltonian (Eq. 5):

$$\hat{H} = H_0 + \tau H_{future} + \epsilon H_{entropy} + PFH \cdot H_{ethics}$$

**Component construction**:

1. **H₀ (Base)**: Diagonal number operators
   ```python
   H0[i,i] = Σ_k ω_k * n_k(i)  # Energy eigenvalues
   ```

2. **H_future**: Ladder operators coupling future subspace
   ```python
   H_future[i,j] ≠ 0  if  diff(i,j) in future index only
   ```

3. **H_entropy**: Random sparse off-diagonal + entropy diagonal
   ```python
   H_entropy[i,j] = random coupling (sparse)
   H_entropy[i,i] = entropy(state_i)
   ```

4. **H_ethics**: Ethics-stability coupling
   ```python
   H_ethics[i,j] ≠ 0  if  diff in ethics or stability indices
   ```

### TimeEvolutionEngine

Implements Trotter-Suzuki decomposition (Eq. 17, 20):

```python
def evolve_step(state, coefficients):
    H_total = Σ c_k * H_k
    
    if trotter_order == 1:
        U = Π_k exp(-i c_k H_k dt)
    else:  # Symmetric Trotter
        U = Π_k exp(-i c_k H_k dt/2) × Π'_k exp(-i c_k H_k dt/2)
    
    return U @ state
```

---

## V2 Sparse Implementation

### Key Innovations

1. **CSR Sparse Storage**
   - Memory: O(nnz) instead of O(n²)
   - Operations: O(nnz) matrix-vector multiply

2. **Krylov Subspace Evolution**
   - Uses `scipy.sparse.linalg.expm_multiply`
   - Computes exp(-iHt)|Ψ⟩ WITHOUT forming full exponential
   - Complexity: O(nnz × m) where m ≈ 30-50

3. **Multi-Axis Time**
   - Four generators G^(k) for k ∈ {0,1,2,3}
   - Non-commuting: ||[G^(i), G^(j)]|| > 0

### SparseHamiltonianFactory

```python
def create_base_hamiltonian() -> sp.csr_matrix:
    # Purely diagonal - sparsity = 1 - 1/n
    diag_values = [energy(i) for i in range(dim)]
    return sp.diags(diag_values, format='csr')

def create_future_hamiltonian() -> sp.csr_matrix:
    # Only couples adjacent future states
    # nnz = O(n / d_F)
    rows, cols, data = [], [], []
    for i in range(dim):
        if can_raise_future(i):
            j = raise_future(i)
            coupling = compute_coupling(i, j)
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([coupling, coupling])
    return sp.csr_matrix((data, (rows, cols)))
```

### MultiAxisEvolutionEngine

```python
def evolve_step(state, tau, epsilon, pfh):
    # Compute weight functions f_k(τ, ε, PFH)
    weights = {
        PHYSICAL: 1 + tau,
        CULTURAL: tau * exp(-epsilon²/2),
        SOCIAL: pfh * sqrt(tau),
        PERSONAL: epsilon * sin(π*tau/2)
    }
    
    # Symmetric Trotter for non-commuting generators
    for _ in range(trotter_steps):
        dt_step = dt / trotter_steps
        # Forward sweep
        for axis, G in generators.items():
            state = expm_multiply(-1j * weights[axis] * G * dt_step/2, state)
        # Backward sweep
        for axis in reversed(generators):
            state = expm_multiply(-1j * weights[axis] * G * dt_step/2, state)
    
    return normalize(state)
```

---

## Key Algorithms

### Algorithm 1: Fixed Point Search

```python
def find_fixed_point(initial_state, T_World, tolerance=1e-8):
    """
    Picard iteration for Banach fixed point.
    
    Convergence: ||Ψ_{n+1} - Ψ_n|| ≤ κ^n ||Ψ_1 - Ψ_0||
    """
    current = initial_state
    history = []
    
    for iteration in range(max_iterations):
        next_state = T_World @ current
        next_state = normalize(next_state)
        
        distance = norm(next_state - current)
        history.append(distance)
        
        if distance < tolerance:
            return next_state, iteration, history
        
        current = next_state
    
    return current, max_iterations, history
```

### Algorithm 2: Commutator Analysis

```python
def compute_commutator_norm(G1, G2):
    """
    Compute ||[G1, G2]|| = ||G1 G2 - G2 G1||_F
    """
    commutator = G1 @ G2 - G2 @ G1
    return sparse_frobenius_norm(commutator)

def analyze_all_commutators(generators):
    """Check non-commutativity of time axes."""
    results = {}
    for (ax1, G1), (ax2, G2) in combinations(generators.items(), 2):
        results[(ax1, ax2)] = compute_commutator_norm(G1, G2)
    return results
```

### Algorithm 3: Phase Transition (G = P ∘ E ∘ R)

```python
def apply_phase_transition(state, R, E, P, jump_prob):
    """
    Apply G = P ∘ E ∘ R with stochastic phase jump.
    """
    # R: Torsion (rotation)
    state = R @ state
    
    # E: Expansion (amplitude scaling)
    state = E @ state  # (1 + r) * state
    
    # P: Phase jump (stochastic)
    if random() < jump_prob:
        state = P @ state  # Apply phase shift
    
    return normalize(state)
```

---

## Performance Optimization

### Memory Optimization

| Implementation | Memory per Hamiltonian |
|---------------|------------------------|
| V1 Dense      | 16 × n² bytes         |
| V2 Sparse     | ~24 × nnz bytes       |

For 99% sparsity: **100x memory reduction**

### Computational Optimization

| Operation | V1 Dense | V2 Sparse |
|-----------|----------|-----------|
| H @ |Ψ⟩   | O(n²)    | O(nnz)    |
| exp(-iHt) | O(n³)    | N/A       |
| expm_multiply | N/A  | O(nnz×m)  |

### Parallelization Opportunities

1. **Trotter steps**: Independent for symmetric decomposition
2. **Multi-axis evolution**: Each axis can be parallelized
3. **Batch state evolution**: Multiple initial states

---

## Testing and Validation

### Unit Tests

```python
def test_unitarity():
    """Verify U†U = I for evolution operator."""
    U = engine.get_evolution_operator(dt)
    assert np.allclose(U.conj().T @ U, np.eye(dim))

def test_normalization():
    """Verify state remains normalized."""
    state = QuantumState(...)
    evolved = engine.evolve_step(state)
    assert np.isclose(evolved.norm(), 1.0)

def test_hermiticity():
    """Verify Hamiltonians are Hermitian."""
    for H in hamiltonians:
        assert np.allclose(H, H.conj().T)

def test_kraus_completeness():
    """Verify Σ K†K = I."""
    kraus_ops = KrausOperators.dephasing_channel(dim, gamma)
    sum_KdK = sum(K.conj().T @ K for K in kraus_ops)
    assert np.allclose(sum_KdK, np.eye(dim))
```

### Convergence Tests

```python
def test_fixed_point_convergence():
    """Verify exponential convergence to fixed point."""
    fixed, iterations, history = world_op.find_fixed_point(state)
    
    # Check fixed point property
    applied = world_op.apply(fixed)
    assert np.allclose(applied.amplitudes, fixed.amplitudes)
    
    # Check exponential decay
    for i in range(1, len(history)):
        ratio = history[i] / history[i-1]
        assert ratio < 1.0  # Contracting
```

### Numerical Verification (Theorem 9.1)

```python
def verify_theorem_9_1():
    """
    Verify numerical predictions:
    - O_M(N=0) = 0.500
    - O_M(N=50) ≈ 0.823
    - O_M(N=100) ≈ 0.951
    """
    results = system.run_full_simulation()
    
    assert abs(results['meaning_obs'][0] - 0.5) < 0.01
    assert abs(results['meaning_obs'][50] - 0.823) < 0.05
    assert results['meaning_obs'][-1] > 0.9
```

---

## Common Issues and Solutions

### Issue 1: Numerical Instability

**Symptom**: State norm drifts from 1.0

**Solution**: Renormalize after each evolution step
```python
state = evolve_step(state)
state = state / np.linalg.norm(state)
```

### Issue 2: Slow Convergence

**Symptom**: Fixed point search takes many iterations

**Solution**: Reduce contraction factor or increase dt
```python
world_op = WorldOperator(contraction_factor=0.8)  # Faster convergence
```

### Issue 3: Memory Error in V1

**Symptom**: MemoryError for large dimensions

**Solution**: Switch to V2 sparse implementation
```python
from v2.engine_v2 import SparseTimeEvolutionEngine
```

### Issue 4: Trotter Error Accumulation

**Symptom**: Results diverge from exact evolution

**Solution**: Increase Trotter steps
```python
engine = MultiAxisEvolutionEngine(..., trotter_steps=20)
```
