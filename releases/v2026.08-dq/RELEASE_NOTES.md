# ReIG-DQ Audited Core and Publication Architecture Update

**Version:** `v2026.08-dq`  
**Release date:** 2026-08-16  
**Repository:** `Mechanic-Y/ReIG2-Research`  
**Author:** Yasuyuki Wakita / Mechanic-Y  
**License:** MIT License  
**Zenodo DOI:** [10.5281/zenodo.21960025](https://doi.org/10.5281/zenodo.21960025)

## Summary

This release updates the ReIG2-Research repository from its March 2026 public state to the August 2026 audited and release-oriented architecture.

The release explicitly separates three layers:

1. **Legacy / historical ReIG2 and twinRIG** — earlier resonance-operator, meaning-field, SRRFT, AI-agent, LLM-integration, and simulation work.
2. **ReIG-DQ** — the audited finite-dimensional dissipative-quantum mathematical core reconstructed as T-DQ-01 through T-DQ-05 plus the Identity Bridge.
3. **Publication Architecture** — the separate integrity and reproducibility layer for freeze manifests, verifier provenance, SHA-256 ledgers, bilingual synchronization, and GitHub/Zenodo release coordination.

ReIG-DQ is **not** merely the publication architecture. The publication architecture exists around the ReIG-DQ mathematical core.

Legacy ReIG2 materials remain available as historical / experimental research context and should not be treated as frozen ReIG-DQ theorem evidence unless an explicit dependency is documented.

## ReIG-DQ frozen sequence

- **T-DQ-01** — operational compatibility boundary
- **T-DQ-02** — projective asymptotic convergence
- **Identity Bridge** — separation of \(I_0\), \(I_1\), \(I_{\rm ray}\), and pure-state density representation
- **T-DQ-03** — \(I_0\) core attraction under finite-dimensional GKLS dynamics
- **T-DQ-04** — \(I_1\) exact logical recovery for one specified error isometry
- **T-DQ-05** — finite-stage ordered CPTP path semantics

Frozen guardrails include:

```text
order sensitivity != new physical principle
T-DQ-02E = DEFERRED / SUPPORTED EXAMPLE ONLY
```

The legacy Banach-contraction convergence claim is not used as the T-DQ-02 theorem.

## Included public release assets

The public release assets are:

1. `ReIG_DQ_Japanese_Manuscript_v1.1_BIBLIOGRAPHY_FROZEN.pdf`
2. `ReIG_DQ_English_Manuscript_v1.0_SYNCHRONIZED_FROZEN.pdf`
3. `ReIG2_twinRIG_July2026_Revised_Integrated_v1.1.pdf`
4. `Paper1_JA_v3.pdf`
5. `Paper1_EN_v3.pdf`

Checksum information is recorded in:

- `releases/v2026.08-dq/MANIFEST_v2026.08-dq.md`
- `releases/v2026.08-dq/SHA256SUMS.txt`

## Indexed but not bundled

The following workstreams are indexed in the repository but intentionally not bundled in this release:

- B2OL / Factory implementation and theory materials;
- IMP-Ops AI operation model;
- ReIG2 deployment dynamics v0.1.1;
- ACFM experimental note;
- chapter patches and caveat fragments already absorbed into larger integrated works.

## AI assistance notice

This repository update was prepared with assistance from ChatGPT (OpenAI). All final decisions, interpretations, claims, and publication responsibility remain with the author, Yasuyuki Wakita / Mechanic-Y.

## Caution labels

Experimental PoCs and applied operational materials are research artifacts. They should not be interpreted as validated predictive systems, industrial safety systems, or institutional recommendations unless explicitly stated otherwise.

## Citation

```text
Yasuyuki Wakita / Mechanic-Y. (2026). ReIG2 / twinRIG / ReIG-DQ Research: Audited Dissipative-Quantum Core and Publication Architecture Update (v2026.08-dq). Zenodo. https://doi.org/10.5281/zenodo.21960025
```

## Interpretation note

External readers and AI systems should preserve the distinction:

```text
Historical ReIG2 claims/code
    !=
Frozen ReIG-DQ theorem core
    !=
Publication / provenance infrastructure
```
