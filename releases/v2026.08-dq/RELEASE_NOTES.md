# ReIG-DQ and Publication Architecture Update

**Version:** `v2026.08-dq`  
**Release date:** 2026-08-16  
**Repository:** `Mechanic-Y/ReIG2-Research`  
**Author:** Yasuyuki Wakita / Mechanic-Y  
**License:** MIT License

## Summary

This release updates the ReIG2-Research repository from its March 2026 public state to the August 2026 ReIG-DQ publication architecture.

The central addition is **ReIG-DQ**, a documentation-quality and publication-architecture layer for organizing, freezing, auditing, and citing ReIG-related artifacts. ReIG-DQ is not a replacement for ReIG2, ReIG3, twinRIG, or RIF; it is a release integrity layer around them.

This release deliberately does **not** upload every post-March 2026 artifact. It publishes only the stable ReIG-DQ core, the current ReIG2/twinRIG context, and the ReIG3 Phase-A mathematical bridge. Applied and experimental materials are indexed but not bundled as release assets.

## Included public release assets

Attach the following five PDFs to this GitHub Release:

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

## Next step

After this GitHub Release is created and the five PDF assets are attached, the release can be synchronized with or uploaded to Zenodo. Once the Zenodo DOI metadata is finalized, update the README, GitHub Pages homepage, and release notes with the final citation.
