# Zenodo Release Notes — ReIG-DQ Audited Core and Publication Architecture Update

**Version:** `v2026.08-dq`  
**Title:** `ReIG2 / twinRIG / ReIG-DQ Research: Audited Dissipative-Quantum Core and Publication Architecture Update`  
**Repository:** `Mechanic-Y/ReIG2-Research`  
**Author:** Yasuyuki Wakita / Mechanic-Y  
**License:** MIT License  
**Zenodo DOI:** [10.5281/zenodo.21960025](https://doi.org/10.5281/zenodo.21960025)

---

## Release Summary

This release updates the ReIG2-Research repository from its March 2026 state to the August 2026 audited and release-oriented research architecture.

The update intentionally separates three layers that should not be conflated:

1. **Legacy / historical ReIG2 and twinRIG** — earlier resonance-operator, meaning-field, SRRFT, LLM-integration, simulation, and related conceptual work.
2. **ReIG-DQ** — the audited finite-dimensional dissipative-quantum mathematical core reconstructed as T-DQ-01 through T-DQ-05 plus the Identity Bridge.
3. **Publication Architecture** — the separate reproducibility and release infrastructure for frozen artifacts, verifier provenance, SHA-256 ledgers, bilingual synchronization, and GitHub/Zenodo publication coordination.

ReIG-DQ is therefore **not merely a documentation-quality layer**. The publication architecture is the layer around the ReIG-DQ mathematical core.

Legacy ReIG2 code and historical claims remain public for research continuity, but they should not be treated as frozen ReIG-DQ theorem evidence unless an explicit dependency is documented.

---

## 1. ReIG-DQ — Audited Mathematical Core

**ReIG-DQ: Reconstructed Design Theory for Dissipative Quantum Systems** contains the frozen sequence:

- **T-DQ-01** — operational compatibility boundary;
- **T-DQ-02** — projective asymptotic convergence;
- **Identity Bridge** — explicit separation of \(I_0\), \(I_1\), \(I_{\rm ray}\), and pure-state density representation;
- **T-DQ-03** — \(I_0\) core attraction under finite-dimensional GKLS dynamics;
- **T-DQ-04** — \(I_1\) exact logical recovery for one specified error isometry;
- **T-DQ-05** — finite-stage ordered CPTP path semantics.

A central frozen guardrail is:

```text
order sensitivity != new physical principle
```

T-DQ-02E remains:

```text
DEFERRED / SUPPORTED EXAMPLE ONLY
```

The earlier Banach-contraction claim from legacy ReIG development is not the convergence theorem used by ReIG-DQ. The frozen T-DQ-02 theorem instead uses explicit finite-dimensional spectral hypotheses and projective-ray convergence.

---

## 2. Publication Architecture — Separate Reproducibility Layer

The publication architecture provides:

- frozen artifact sets and freeze manifests;
- canonical theorem and quarantine ledgers;
- independent-verifier scripts and reports;
- hash-based integrity checks;
- bilingual Japanese/English synchronization;
- GitHub / Zenodo release coordination;
- separation between core theory, legacy implementation, application, experimental PoC, and adjacent mathematics.

This layer makes ReIG-DQ auditable and reproducible; it is not itself the mathematical definition of ReIG-DQ.

---

## 3. Legacy ReIG2 / twinRIG Context

The repository preserves the earlier ReIG2 research line, including the eight-operator implementation, phase-transition explorations, SRRFT, information-geometric and AI-agent experiments, and LLM-integration modules.

These materials are valuable as historical / experimental research context, but they are not automatically part of the frozen ReIG-DQ theorem core.

---

## 4. ReIG3 Internal-Time and State-Dependent Extensions

The ReIG3 line is extended through internal-time and state-dependent effective time-operator formulations. These materials should be treated as research extensions or preprints rather than as finalized software packages.

---

## 5. B2OL / Operational Design Layer

B2OL and related factory/AI-operation materials are treated as ReIG-adjacent applied layers. They connect resonance, boundaries, policy layers, temporal metrics, and field operation concepts to practical AI/organization/manufacturing design.

---

## 6. Experimental State-Estimation PoCs

Experimental PoCs, including earthquake-related state-estimation work, are included only with caution labels. They should not be presented as official predictions or validated public-safety systems.

---

## 7. Adjacent Mathematical Work

Finite-dimensional observability and boundary-analysis papers may be referenced as adjacent mathematical work. They should remain clearly distinguished from the ReIG core unless direct dependencies are explicitly documented.

---

## Zenodo Description

The August 2026 release distinguishes the historical ReIG2/twinRIG research line from the audited ReIG-DQ mathematical reconstruction and from the separate publication architecture used to freeze and verify it. ReIG-DQ is the finite-dimensional dissipative-quantum core organized as T-DQ-01 through T-DQ-05 plus the Identity Bridge. The publication architecture supplies frozen manifests, verifier provenance, SHA-256 integrity ledgers, synchronized bilingual manuscripts, and release coordination. Legacy ReIG2 code, applications, and experimental materials remain available with explicit scope labels.

---

## AI Assistance Notice

This repository update was prepared with assistance from ChatGPT (OpenAI). All final decisions, interpretations, claims, and publication responsibility remain with the author, Yasuyuki Wakita / Mechanic-Y.

---

## Keywords

```text
ReIG, ReIG2, twinRIG, ReIG3, ReIG-DQ, dissipative quantum dynamics, GKLS, CPTP, projective convergence, quantum error recovery, ordered quantum channels, publication architecture, verification ledger, SHA-256 provenance, reproducible research, B2OL, IMP-Ops
```

---

## Citation Text

```text
Yasuyuki Wakita / Mechanic-Y. (2026). ReIG2 / twinRIG / ReIG-DQ Research: Audited Dissipative-Quantum Core and Publication Architecture Update (v2026.08-dq). Zenodo. https://doi.org/10.5281/zenodo.21960025
```

---

## Release Assets

The public release assets are:

1. `ReIG_DQ_Japanese_Manuscript_v1.1_BIBLIOGRAPHY_FROZEN.pdf`
2. `ReIG_DQ_English_Manuscript_v1.0_SYNCHRONIZED_FROZEN.pdf`
3. `ReIG2_twinRIG_July2026_Revised_Integrated_v1.1.pdf`
4. `Paper1_JA_v3.pdf`
5. `Paper1_EN_v3.pdf`

Checksum information is recorded in:

- `releases/v2026.08-dq/MANIFEST_v2026.08-dq.md`
- `releases/v2026.08-dq/SHA256SUMS.txt`

Do not include private notes, unfinished drafts, or exploratory outputs unless they are explicitly labeled as provisional.

---

## Final Publication Checklist

- [x] GitHub README reflects `v2026.08-dq`.
- [x] GitHub Pages or portal page reflects the August 2026 update.
- [x] Release notes are included in the repository.
- [x] `.zenodo.json` metadata is present.
- [x] Release assets are checked for private information.
- [x] Hash manifest matches the public release assets.
- [x] GitHub tag/release is created.
- [x] Zenodo record is published.
- [x] DOI and citation text are recorded.

> **Metadata correction note:** the August 2026 documentation is being clarified so that external readers and AI systems do not conflate the ReIG-DQ mathematical core with the separate publication architecture or with legacy ReIG2 code.
