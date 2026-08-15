# ReIG Research Update — August 2026

**Version candidate:** `v2026.08-dq`  
**Scope:** ReIG-related artifacts created or stabilized after the March 2026 repository state  
**Status:** Public update inventory / Zenodo preparation note

This document summarizes the ReIG-related work that should be reflected in the GitHub repository and in the next Zenodo record update. The repository was last structurally updated in March 2026; this August 2026 update adds the bridge from ReIG2/ReIG3 theory to publication architecture, operational layers, and verification-led public packaging.

---

## 1. Update Principle

This update should not treat every later artifact as a single undifferentiated "ReIG core" item. Instead, it separates the work into layers:

1. **Core theory:** ReIG2 / ReIG3 / RIF mathematical and conceptual framework.
2. **Publication and verification architecture:** ReIG-DQ, frozen artifacts, hash ledgers, structural audit scripts.
3. **Operational design layer:** B2OL, IMP-Ops, factory/control applications.
4. **Experimental or provisional PoC layer:** state-estimation prototypes and exploratory simulations.
5. **Adjacent mathematical work:** observability, boundary, and finite-dimensional analysis papers that inform the ReIG research style but should remain distinguishable from the ReIG core.

---

## 2. Candidate Artifacts to Add or Reference

| Artifact / Theme | Layer | ReIG Relation | Suggested Repository Placement | Publication Status |
|---|---:|---|---|---|
| **ReIG-DQ Publication Architecture v0.1** | Publication architecture | Provides a public-quality verification and identity bridge around ReIG outputs | `ReIG2/docs/`, `shared/release/`, or `updates/` | High priority |
| **T-DQ-01 to T-DQ-05 frozen artifacts** | Publication architecture | Freeze set for ReIG-DQ publication packaging | `shared/release/reig-dq/` | High priority if files are available |
| **Identity Bridge ledger** | Publication architecture | Connects author identity, artifact lineage, and release architecture | `shared/release/reig-dq/` | High priority |
| **SHA256 / structural audit scripts** | Verification | Supports reproducibility and public release integrity | `shared/tools/` or `ReIG2/code/audit/` | High priority |
| **ReIG2 Phase A v0.1.2 audit ledger** | Verification / implementation | Records V-03c standalone and external execution lineage | `ReIG2/code/phase-a/` | High priority |
| **ReIG3 internal-time extension** | Core theory | Extends ReIG through state-dependent effective time operators | `ReIG3/papers/` and `ReIG3/code/` | Medium-high priority |
| **B2OL / ReIG-B2OL体系** | Operational design | Links ReIG state, boundary, temporal and policy layers to field/AI operations | `ReIG3/papers/`, `shared/applications/`, or new `B2OL/` | Medium-high priority |
| **IMP-Ops / AI operation roles** | Operational design | Applies ReIG-style role and resonance management to AI operations | `shared/applications/imp-ops/` | Medium priority |
| **StockPilot / ReIG Factory Core concept** | Factory application | Applies resonance scoring and boundary detection to manufacturing control | `shared/applications/factory/` | Medium priority |
| **ReIG earthquake state-estimation PoC** | Experimental PoC | Uses ReIG-like state estimation; must be clearly labeled non-predictive | `shared/experiments/earthquake-poc/` | Low-medium priority, with caution |
| **Observability Boundaries for Near-Critical Weil Blocks** | Adjacent mathematics | Adjacent finite-dimensional observability work; not ReIG core | separate repository or `shared/adjacent-math/` | Optional / separate preferred |

---

## 3. Recommended Public Narrative

The August 2026 update should be framed as:

> From March to August 2026, the project moved from the initial ReIG2/ReIG3 theoretical repository toward a more public, auditable, and publishable research architecture. The main addition is ReIG-DQ: a documentation and verification layer for freezing, auditing, and connecting ReIG-related artifacts. Several operational and applied layers, including B2OL and factory-oriented control concepts, are also now treated as ReIG-adjacent applications rather than as replacements for the core theory.

This keeps the claim modest while making the update meaningful.

---

## 4. Suggested Version Naming

Recommended GitHub release / Zenodo version:

- `v2026.08-dq`
- Alternative: `v2026.08.0`
- Release title: **ReIG-DQ and Publication Architecture Update**

Suggested Git tag:

```text
v2026.08-dq
```

---

## 5. Files Still Needed Before Final Zenodo Deposit

Before making the Zenodo version final, confirm the actual files to upload or link:

- ReIG-DQ core document, preferably PDF + Markdown/source.
- Frozen artifact set T-DQ-01 to T-DQ-05, if intended for public release.
- Identity Bridge document or ledger.
- SHA256SUMS or equivalent hash manifest.
- Audit scripts, if public release is intended.
- Updated README and repository index page.
- Any source ZIP or LaTeX/Markdown package that should be archived.

---

## 6. Caution Labels

Use explicit caution labels for exploratory work:

- Earthquake-related PoCs are **state-estimation experiments**, not official forecasts.
- B2OL / Factory Core materials are **operational design concepts**, not validated industrial safety systems.
- Adjacent mathematical papers should not be presented as direct proofs of ReIG unless the relationship is explicitly established.

---

## 7. Minimal Release Checklist

- [ ] Update root `README.md` to show `v2026.08-dq` as the current repository update.
- [ ] Add or update release notes for Zenodo.
- [ ] Add `.zenodo.json` metadata.
- [ ] Add file manifest and SHA256SUMS if public artifacts are attached.
- [ ] Review GitHub Pages `index.html` after README update.
- [ ] Create GitHub release/tag.
- [ ] Trigger or manually update Zenodo record.
- [ ] Verify DOI metadata and citation text after Zenodo publication.
