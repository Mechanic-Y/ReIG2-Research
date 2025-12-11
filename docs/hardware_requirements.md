# ReIG2/twinRIG Quantum Hardware Requirements

Specifications for implementing ReIG2 on quantum computing hardware.

Reference: ReIG2_twinRIG_2025_December.pdf, Section 10

---

## Table of Contents

1. [Overview](#overview)
2. [Hardware Specifications](#hardware-specifications)
3. [Gate Requirements](#gate-requirements)
4. [Circuit Characteristics](#circuit-characteristics)
5. [Error Budget](#error-budget)
6. [Platform Comparison](#platform-comparison)
7. [Recommended Configurations](#recommended-configurations)

---

## Overview

The ReIG2/twinRIG framework can be implemented on various quantum computing platforms. This document specifies the requirements for faithful simulation of the quantum resonance dynamics.

### Minimum System Requirements

| Parameter | Requirement | Optimal |
|-----------|-------------|---------|
| Qubits | 3 (minimum) | 5-10 |
| 1Q Gate Fidelity | > 99.9% | > 99.99% |
| 2Q Gate Fidelity | > 99% | > 99.9% |
| T1 (relaxation) | > 100 μs | > 500 μs |
| T2 (coherence) | > 50 μs | > 200 μs |
| Circuit Depth | < 1000 | < 500 |

---

## Hardware Specifications

### Section 10.2 Requirements (Paper Reference)

```
┌─────────────────────────────────────────────────────────────┐
│  Parameter               │  Minimum     │  Recommended      │
├─────────────────────────────────────────────────────────────┤
│  1-qubit gate fidelity   │  F > 99.9%   │  F > 99.99%       │
│  2-qubit gate fidelity   │  F > 99%     │  F > 99.9%        │
│  T1 (relaxation time)    │  > 100 μs    │  > 500 μs         │
│  T2 (coherence time)     │  > 50 μs     │  > 200 μs         │
│  Circuit depth           │  < 1000      │  < 500 gates      │
│  Total execution time    │  < T2/2      │  < T2/10          │
└─────────────────────────────────────────────────────────────┘
```

### Qubit Quality Metrics

**T1 (Relaxation Time)**:
- Measures amplitude damping (energy relaxation)
- Affects: State population fidelity
- ReIG2 requirement: T1 > 100 μs for 10+ iterations

**T2 (Coherence Time)**:
- Measures phase coherence (dephasing)
- Affects: Quantum interference quality
- ReIG2 requirement: T2 > 50 μs for resonance evolution

**Gate Fidelity**:
- Single-qubit: Rz gates for resonance evolution
- Two-qubit: CNOT for ZZ coupling implementation
- ReIG2 requirement: > 99% for meaningful results

---

## Gate Requirements

### Required Gates

1. **Hadamard (H)**
   - Purpose: Initial superposition
   - Count per iteration: 3 (one per qubit)

2. **Rz(θ) - Z-rotation**
   - Purpose: Resonance evolution (Eq. 103)
   - Formula: Rz(θ) = exp(-iθσz/2)
   - Count per iteration: 3 + coupling terms

3. **CNOT (CX)**
   - Purpose: ZZ coupling decomposition
   - Implementation: ZZ(θ) = CX · Rz(2θ) · CX
   - Count per iteration: 4 (2 per coupling term)

4. **Measurement**
   - Purpose: Observation qubit readout
   - Count: 1 per iteration

### Gate Decomposition

**ZZ Coupling** (Section 10.1):
```
exp(-iθ Z⊗Z) = CNOT · Rz(2θ) · CNOT
```

Circuit:
```
q0: ──●────────●──
      │        │
q1: ──X──Rz(2θ)──X──
```

**Full Resonance Step**:
```
q_M: ──H──Rz(ω_M·dt)───●────────●─────────────────
                        │        │
q_C: ──H──Rz(ω_C·dt)───X──Rz(ε)──X──●────────●────
                                     │        │
q_O: ──H──Rz(ω_O·dt)────────────────X──Rz(ε)──X──M
```

---

## Circuit Characteristics

### Single Iteration Metrics

| Metric | Value |
|--------|-------|
| Total gates | ~15-20 |
| 1Q gates | ~9-12 |
| 2Q gates | 4-6 |
| Depth | ~10-15 |
| Execution time | ~2-5 μs |

### Full Simulation (100 iterations)

| Metric | Value |
|--------|-------|
| Total gates | ~1500-2000 |
| Depth | ~1000-1500 |
| Execution time | ~200-500 μs |

### Trotter Decomposition Impact

With M Trotter steps per iteration:
- Gate count multiplied by M
- Depth multiplied by M
- Error reduced by O(1/M²)

Recommended: M = 5-10 for balance of accuracy and depth.

---

## Error Budget

### Error Sources

1. **Gate errors**: ~0.1-1% per gate
2. **Decoherence**: exp(-t/T2) per step
3. **Measurement errors**: ~1-5%
4. **Crosstalk**: varies by platform

### Error Allocation

For 99% target fidelity over 100 iterations:

```
Per-iteration error budget: < 0.01%

Breakdown:
├── 1Q gate errors: 0.003% × 10 gates = 0.03%
├── 2Q gate errors: 0.01% × 4 gates = 0.04%
├── Decoherence: dt/T2 ≈ 0.02%
└── Total per iteration: ~0.09%

After 100 iterations: ~9% accumulated error
```

### Error Mitigation Strategies

1. **Zero-noise extrapolation (ZNE)**
   - Run at multiple noise levels
   - Extrapolate to zero noise

2. **Probabilistic error cancellation (PEC)**
   - Characterize error channels
   - Apply inverse maps probabilistically

3. **Dynamical decoupling**
   - Insert identity sequences
   - Suppress low-frequency noise

4. **Post-selection**
   - Discard shots with detected errors
   - Use ancilla qubits for error detection

---

## Platform Comparison

### IBM Quantum

**Advantages**:
- Free cloud access
- Good documentation
- Qiskit ecosystem

**Current specifications** (IBM Eagle/Heron):
- 1Q fidelity: 99.95%
- 2Q fidelity: 99.5%
- T1: 200-500 μs
- T2: 100-300 μs

**Suitability**: ★★★★☆ (Good)

### Google Quantum AI

**Advantages**:
- High coherence
- 2D grid connectivity
- Cirq integration

**Current specifications** (Sycamore):
- 1Q fidelity: 99.94%
- 2Q fidelity: 99.62%
- T1: 15-20 μs
- T2: 10-15 μs

**Suitability**: ★★★☆☆ (Moderate - shorter coherence)

### IonQ

**Advantages**:
- Full connectivity
- High fidelity
- Long coherence

**Current specifications**:
- 1Q fidelity: 99.99%
- 2Q fidelity: 99.3%
- T1: seconds
- T2: hundreds of ms

**Suitability**: ★★★★★ (Excellent)

### Rigetti

**Advantages**:
- Qiskit compatible
- Good cloud access

**Current specifications** (Aspen-M):
- 1Q fidelity: 99.7%
- 2Q fidelity: 96%
- T1: 10-30 μs
- T2: 5-15 μs

**Suitability**: ★★☆☆☆ (Limited by 2Q fidelity)

---

## Recommended Configurations

### Minimum Viable (Proof of Concept)

```
Platform: IBM Quantum (free tier)
Qubits: 3
Iterations: 10-20
Trotter steps: 1
Expected fidelity: ~90%
```

### Standard (Research Quality)

```
Platform: IBM Quantum Premium / IonQ
Qubits: 5
Iterations: 50-100
Trotter steps: 5
Expected fidelity: ~95%
Error mitigation: ZNE
```

### Optimal (Publication Quality)

```
Platform: IonQ / IBM Heron
Qubits: 10
Iterations: 100+
Trotter steps: 10
Expected fidelity: ~99%
Error mitigation: Full suite (ZNE + PEC + DD)
```

---

## Circuit Optimization Tips

### 1. Minimize CNOT Count

```python
# Before optimization: 4 CNOTs per ZZ term
# After optimization: Can share CNOTs between adjacent terms
```

### 2. Compile to Native Gates

```python
# IBM native gates: [Rz, SX, X, ECR]
# Convert Hadamard: H = Rz(π/2) · SX · Rz(π/2)
```

### 3. Use Symmetry

```python
# Symmetric initial state reduces circuit depth
# Uniform superposition: only log(n) gates needed
```

### 4. Adaptive Trotter

```python
# Increase Trotter steps where commutators are large
# ||[H_i, H_j]|| → more steps for (i,j) coupling
```

---

## Future Hardware Considerations

### Near-term (2024-2026)

- 1000+ qubit systems
- Error correction codes becoming practical
- Focus: Hybrid classical-quantum algorithms

### Mid-term (2026-2030)

- Logical qubits with error correction
- ReIG2 on logical qubits feasible
- Focus: Full quantum simulation

### Long-term (2030+)

- Fault-tolerant quantum computing
- Arbitrary precision ReIG2 simulation
- Focus: Consciousness modeling applications

---

## References

1. IBM Quantum Systems: https://quantum-computing.ibm.com/
2. Google Quantum AI: https://quantumai.google/
3. IonQ: https://ionq.com/
4. Preskill, J. "Quantum Computing in the NISQ era and beyond" (2018)
5. Endo et al. "Practical Quantum Error Mitigation" (2018)
