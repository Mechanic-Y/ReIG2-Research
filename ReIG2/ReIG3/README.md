# ReIG3 — Resonant Intelligence Grid

## Status: Pre-construction (March 2026)

ReIG3 extends ReIG2 from pairwise subject resonance to **N-body resonance networks** with dynamic topology.

## Position in ReIG Programme

```
Photon Reactor (Phase 0)
    → twinRIG (Phase 1)
        → ReIG2: Gate/Gateway (Phase 2)
            → ReIG3: Grid (Phase 3)   ← HERE
                → RIF: Field (Phase 4)
```

## Core Concepts

- **Resonance Graph**: G = (S, E), E = {(S_i, S_j) : R(S_i, S_j) > θ}
- **Dynamic Resonance Graph**: Time-dependent links R(S_i, S_j, t)
- **Tensor Network**: T_ij = R_ij Ψ_i ⊗ Ψ_j
- **State-Dependent Coupling**: K_ij(t) = K_ij^(0) + K_ij^(field) · σ((Ē−E*)/w)

## Key Results (from v4 analysis)

The three-variable model (θ, A, E) yields the **triple phase-transition structure**:

| Transition | Condition | Physical meaning |
|-----------|-----------|-----------------|
| Phase sync | K > K_c^(θ) = 2/(πg(0)) | Content alignment |
| Amplitude | K > K_c^(A) = 2κ/(σr) | Emotional propagation |
| Energy | (A*)²[η/(...) + λKr] ≥ μE_c | Resonance onset |

The ReIG3→RIF transition is characterized by the **self-consistent field equation**: F(Ē) = Ē

## Planned Contents

- `papers/` — N-body extensions, numerical phase diagram
- `code/` — N-body simulation, phase-transition sweeps, dynamic coupling visualization
- `docs/` — Derivations, Kuramoto comparison, SRRFT connection

## Relationship to Other Components

| Component | Relation |
|-----------|----------|
| ReIG2 | Pairwise limit (N=2) of ReIG3 |
| RIF | Continuum limit (N→∞) of ReIG3 |
| SRRFT | Shared fixed-point structure |

## Author
Yasuyuki Wakita (Mechanic-Y), Independent Researcher
