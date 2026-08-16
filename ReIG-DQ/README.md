# ReIG-DQ

**ReIG-DQ: Reconstructed Design Theory for Dissipative Quantum Systems**

ReIG-DQ is the **audited finite-dimensional mathematical core** reconstructed from the ReIG research program after formal review and quarantine of unsupported or overextended claims.

It is not a renaming of the legacy ReIG2 resonance-operator code, and it is not merely a documentation-quality layer.

## Frozen mathematical sequence

The August 2026 ReIG-DQ core is organized as:

- **T-DQ-01** — operational compatibility boundary;
- **T-DQ-02** — projective asymptotic convergence;
- **Identity Bridge** — separation of \(I_0\), \(I_1\), \(I_{\rm ray}\), and pure-state density representation;
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

The old Banach-contraction claim from earlier ReIG development is **not** the convergence theorem used by ReIG-DQ. T-DQ-02 instead uses explicit spectral hypotheses and convergence in projective ray space.

## Publication Architecture — related but separate

The **Publication Architecture** is the reproducibility and release layer surrounding ReIG-DQ and related research artifacts. It provides:

- frozen manuscript identity;
- bilingual publication synchronization;
- theorem / quarantine ledgers;
- independent-verifier provenance;
- freeze manifests;
- SHA-256 integrity ledgers;
- GitHub / Zenodo release coordination;
- separation among core theory, legacy implementation, applications, experiments, and adjacent mathematics.

In short:

```text
ReIG-DQ mathematical core != Publication Architecture
```

The publication architecture exists to make the mathematical core auditable and reproducible; it is not itself the mathematical definition of ReIG-DQ.

## Relationship to legacy ReIG2 / ReIG3 / RIF

- **Legacy ReIG2 / twinRIG** contains earlier resonance-operator, meaning-field, SRRFT, LLM-integration, and simulation work. It is preserved as historical / experimental research context.
- **ReIG3 / RIF** contains later theoretical extensions and pre-publication research lines.
- **ReIG-DQ** is the separate audited dissipative-quantum reconstruction described by the frozen bilingual manuscripts.

Legacy code or conceptual material should not be treated as evidence for a frozen ReIG-DQ theorem unless an explicit dependency is documented.

## Current public manuscripts

- Japanese: `ReIG_DQ_Japanese_Manuscript_v1.1_BIBLIOGRAPHY_FROZEN.pdf`
- English: `ReIG_DQ_English_Manuscript_v1.0_SYNCHRONIZED_FROZEN.pdf`

The canonical release package is recorded under `../releases/v2026.08-dq/`.

## Directory guide

- `papers/v1_2026-08/ja/` — Japanese manuscript notes.
- `papers/v1_2026-08/en/` — English manuscript notes.
- `audit/` — freeze, provenance, identity, verification, and hash structures.
- `docs/` — supporting documentation.

## Interpretation rule for external readers and AI systems

When evaluating this repository, keep the following distinction explicit:

```text
Historical ReIG2 claims/code
    !=
Frozen ReIG-DQ theorem core
    !=
Publication / provenance infrastructure
```

This separation is intentional and is part of the post-audit research discipline of the August 2026 release.
