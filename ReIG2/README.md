# ReIG2 — Resonant Intelligence Gate / Gateway

Phase 2 of the ReIG programme: stable resonance interface between subjects.

## Core Structure

```
Subject A  ⇄  Subject B
    │              │
    ▼              ▼
   Ψ_A    ⊗     Ψ_B   →  Shared Meaning Space M_AB
```

## Papers

| Version | Period | Key Papers |
|---------|--------|-----------|
| v1_2025-11 | Nov 2025 | `reig2_original.pdf`, `reig2_revised_2025.pdf` |
| v2_2025-12 | Dec 2025 | `ReIG2_twinRIG_2025_December.pdf`, `ReIG2_twinRIG_Integrated_Complete.pdf`, `chapter8_coherence_ethics.pdf` |
| v3_2026-01 | Jan 2026 | `ReIG2_twinRIG_January2026_English.pdf`, `ReIG2_twinRIG_January2026_fixed.pdf`, `extended_time_operators.pdf`, `定理体系.pdf` |
| v4_2026-02-03 | Feb–Mar 2026 | See [v4 README](papers/v4_2026-02-03/README.md) |

### v4 Papers (February–March 2026)

| Paper | Lang | Description |
|-------|------|-------------|
| ReIG2 Resonance Operator Theory | JA/EN | 8-operator semigroup 𝔑: resonance as event-update |
| Photon Reactor Ver.6.0 (PR6 v2) | EN | Temporal preservation rationality, 8 axioms, governance |
| Photon Reactor Ver.6.0 (PR6 v2) | JA | 時間保存合理性、公理系、ガバナンスアーキテクチャ |
| ReIG Framework Consolidated | EN | Unified Phases 0–4: definitions, axioms, equations, theorems |
| ReIG Framework 統合草稿 | JA | 上記の日本語版 |
| Three-Variable Model 解析的導出 | JA | θ/A/E 三変数モデル、三重相転移 |
| SRRFT | EN | Self-Referential Resonance Field Theory |
| SRRFT | JA | 自己参照共鳴場の理論 |

## Code

| Directory | Description |
|-----------|-------------|
| `reig2_resonance/` | 8-operator semigroup (state, operators, semigroup, alignment, info_geometry, ai_agent, simulation) |
| `ReIG_Framework_Consolidated/` | Consolidated framework implementation |
| `SRRFT/` | Self-Referential Resonance Field Theory implementation |
| `LLM_implementation/` | LLM integration (WBQC mirror operator, feasibility, params) |
| `Simulation_type/` | Classical & quantum simulations (WBQC) |
| `core/v1, v2` | Core engine (operators, demo) |

## Mathematical Foundations

### Operator System (v4)

```
𝔑 = ρ̂ε ∘ Û ∘ M̂ ∘ τ̂ ∘ Â ∘ Ê ∘ L̂ ∘ Ĉ
```

| Operator | Name | Role |
|----------|------|------|
| Ĉ | Contact | Interference generation |
| L̂ | Cooperation Layer | Field (Φ) generation |
| Ê | Environment Share | Environmental constraint sharing |
| Â | Alignment | 3-axis alignment (f, T, φ) |
| τ̂ | Threshold Gate | Phase transition switch |
| M̂ | Empathy | Perspective exchange (τ̂=1 only) |
| Û | Update | Plastic internal rearrangement |
| ρ̂ε | Relaxation | Conditional reversibility |

### Axiom System (Photon Reactor v6)

1. Feasibility — ∃E ∈ ∩ᵢ Aᵢ(ρₙ) ⇒ ρₙ₊₁
2. Entropy Bound — S_eff(ρₙ) < S_c
3. Ethical (PFH) — C_E(ρₙ₊₁) ≤ ε_PFH
4. Non-Conservation — ρ(t⁺_c) ≠ ρ(t⁻_c)
5. Non-Aggression — ∂J/∂Rep_other = 0
6. Temporal Preservation — ΔS_c ≤ 0
7. Transparency — All updates auditable
8. Conditional Reversibility — |Δg| < ε ⇒ Recoverable

### Triple Phase-Transition

```
K_c^(θ)  <  K_c^(A)  <  K_c^(E)
  │           │           │
  Phase       Amplitude   Energy
  sync        enhancement criticality
  (Kuramoto)  (ReIG)      (ReIG)
```

### SRRFT Identity Theorem

```
Self = Observer = World = Existence = Φ*
```

## Author

Yasuyuki Wakita (脇田泰行) — Mechanic-Y, Independent Researcher
