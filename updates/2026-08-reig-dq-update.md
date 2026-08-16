# ReIG Research Update — August 2026

**Version:** `v2026.08-dq`  
**Scope:** ReIG-related artifacts created or stabilized after the March 2026 repository state  
**Status:** Public update inventory / post-release clarification

This document records the August 2026 repository architecture and clarifies the relationship among legacy ReIG2 research, the audited ReIG-DQ mathematical core, and the separate publication / verification infrastructure.

---

## 1. Update Principle

The repository must not treat every ReIG-related artifact as one undifferentiated core. The August architecture separates the work into the following layers:

1. **Legacy / historical ReIG2 and twinRIG research:** resonance operators, meaning-field models, SRRFT, LLM integration, simulations, and related conceptual/experimental work.
2. **ReIG-DQ mathematical core:** audited finite-dimensional dissipative-quantum results frozen as T-DQ-01 through T-DQ-05 plus the Identity Bridge.
3. **Publication and verification architecture:** freeze manifests, canonical ledgers, independent-verifier provenance, SHA-256 integrity ledgers, bilingual synchronization, and release tooling.
4. **ReIG3 / RIF extensions:** later theoretical extensions and pre-publication research lines.
5. **Operational design layer:** B2OL, IMP-Ops, factory/control applications.
6. **Experimental or provisional PoC layer:** state-estimation prototypes and exploratory simulations.
7. **Adjacent mathematical work:** observability, boundary, and finite-dimensional analysis papers that inform the research style but remain distinguishable from the ReIG core unless direct dependencies are documented.

The central clarification is:

```text
Legacy ReIG2 research
    !=
ReIG-DQ audited theorem core
    !=
Publication Architecture
```

---

## 2. ReIG-DQ — Current Meaning

**ReIG-DQ: Reconstructed Design Theory for Dissipative Quantum Systems** is the audited mathematical reconstruction produced after formal review of earlier ReIG claims.

Its frozen sequence is:

- **T-DQ-01** — operational compatibility boundary;
- **T-DQ-02** — projective asymptotic convergence;
- **Identity Bridge** — \(I_0\) / \(I_1\) / \(I_{\rm ray}\) semantic separation;
- **T-DQ-03** — \(I_0\) core attraction under finite-dimensional GKLS dynamics;
- **T-DQ-04** — \(I_1\) exact logical recovery for one specified error isometry;
- **T-DQ-05** — finite-stage ordered CPTP path semantics.

Frozen guardrails include:

```text
order sensitivity != new physical principle
T-DQ-02E = DEFERRED / SUPPORTED EXAMPLE ONLY
```

The earlier Banach-contraction convergence claim from legacy ReIG development is not used as the ReIG-DQ convergence theorem. T-DQ-02 uses explicit spectral hypotheses and convergence in projective ray space.

---

## 3. Publication Architecture — Separate Layer

The publication architecture surrounds ReIG-DQ and related outputs but is not the definition of ReIG-DQ itself. It provides:

- frozen artifact sets;
- freeze manifests;
- Canonical Theorem & Quarantine Ledger;
- independent-verifier scripts and reports;
- SHA-256 integrity checks;
- bilingual Japanese/English synchronization;
- GitHub / Zenodo release coordination;
- separation between theory, implementation, applications, experiments, and adjacent mathematics.

---

## 4. Artifact Classification

| Artifact / Theme | Layer | ReIG Relation | Suggested Repository Placement | Publication Status |
|---|---:|---|---|---|
| **T-DQ-01 to T-DQ-05 frozen artifacts** | ReIG-DQ core | Audited mathematical reconstruction | `ReIG-DQ/` and release package | Frozen / high priority |
| **Identity Bridge** | ReIG-DQ core / semantics | Separates occupancy, logical identity, and projective identity | `ReIG-DQ/audit/identity_bridge/` | Frozen |
| **Canonical Ledger / Publication Architecture** | Publication / verification | Provenance, quarantine, release integrity | `ReIG-DQ/audit/` / release package | Frozen / supporting |
| **SHA256 / structural audit scripts** | Verification | Reproducibility and public release integrity | `ReIG-DQ/audit/` / release package | Supporting evidence |
| **Legacy 8-operator ReIG2 code** | Historical / experimental ReIG2 | Earlier resonance-model implementation | `ReIG2/code/` | Preserved; not frozen ReIG-DQ evidence |
| **ReIG2 Phase A verification lineage** | Verification / implementation | Historical audit lineage and bridge work | `ReIG2/` / supporting docs | Context-dependent |
| **ReIG3 internal-time extension** | ReIG3 theory | Later research extension | `ReIG3/papers/` and `ReIG3/code/` | Pre-publication / research |
| **B2OL / ReIG-B2OL体系** | Operational design | ReIG-adjacent state/boundary/time/policy application | `applications/B2OL/` | Applied concept |
| **IMP-Ops / AI operation roles** | Operational design | ReIG-adjacent AI operations | `applications/IMP-Ops/` | Applied concept |
| **ReIG earthquake state-estimation PoC** | Experimental PoC | Exploratory state estimation | `experimental-notes/` | Provisional / non-predictive |
| **Observability Boundaries for Near-Critical Weil Blocks** | Adjacent mathematics | Finite-dimensional observability work | separate / adjacent math | Distinct from ReIG core |

---

## 5. Recommended Public Narrative

The August 2026 public narrative should be:

> From March to August 2026, the project moved from a broad ReIG2/ReIG3 theory-and-code repository toward a layered research architecture. Legacy ReIG2 resonance models and implementations remain available as historical and experimental context. In parallel, ReIG-DQ was reconstructed as an independently audited finite-dimensional dissipative-quantum mathematical core. A separate Publication Architecture now freezes and verifies that core through canonical ledgers, independent-verifier provenance, SHA-256 manifests, bilingual synchronization, and GitHub/Zenodo release packaging. Applied layers such as B2OL and experimental PoCs remain explicitly separated from the frozen theorem core.

This formulation avoids both under-describing the ReIG-DQ mathematics and over-attributing legacy ReIG2 claims to the frozen core.

---

## 6. Version Naming

Current GitHub release / Zenodo version:

- `v2026.08-dq`
- Release title: **ReIG-DQ Audited Core and Publication Architecture Update**

Git tag:

```text
v2026.08-dq
```

---

## 7. Caution Labels

Use explicit caution labels for exploratory work:

- Earthquake-related PoCs are **state-estimation experiments**, not official forecasts.
- B2OL / Factory materials are **operational design concepts**, not validated industrial safety systems.
- Adjacent mathematical papers should not be presented as direct proofs of ReIG unless the relationship is explicitly established.
- Legacy ReIG2 simulations and code do not automatically validate frozen ReIG-DQ theorems.

---

## 8. Post-Release Clarification Checklist

- [x] Root `README.md` distinguishes legacy ReIG2, ReIG-DQ, and Publication Architecture.
- [x] `ReIG-DQ/README.md` defines the audited mathematical core.
- [x] `.zenodo.json` no longer defines ReIG-DQ merely as documentation quality.
- [x] Zenodo/GitHub release notes distinguish mathematical core from publication infrastructure.
- [ ] Review GitHub Pages `index.html` for the same ambiguity.
- [ ] If Zenodo metadata is edited or a new version is issued, synchronize the clarified description there.
