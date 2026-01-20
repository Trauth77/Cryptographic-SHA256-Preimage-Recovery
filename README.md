# Cryptographic-SHA256-Preimage-Recovery
Empirical validation of SHA-256 preimage recovery using GCIS neural architecture. 100% bit-sign pattern matching across 5 independent password reconstructions (20-32 chars). Includes raw layer data, analysis scripts, and full methodology. Open challenge for cryptanalysts: blind search & sequencing.

# Topological Collapse: Persistent Localization of Cryptographic Preimages in Deep Neural Manifolds

**Author:** Stefan Trauth  
**Affiliation:** Independent Researcher, Neural Systems & Emergent Intelligence Laboratory  
**DOI:** [10.5281/zenodo.18305804](https://doi.org/10.5281/zenodo.18305804)  
**License:** CC BY-NC-ND 4.0

## Abstract

We demonstrate deterministic localization of cryptographic hash preimages within specific layers of deep neural networks trained on information-geometric principles. Using a modified Spin-Glass architecture, MD5 and SHA-256 password preimages are consistently identified in layers ES15-ES20 with up to 100% accuracy.

Key findings:
- 100% byte-level identification for MD5 & SHA-256 passwords (11-32 characters)
- Over 40 successfully detected test passwords
- 41.8% information persistence across 11 independent network runs
- Novel charge-based filtering methodology: 30% noise reduction with zero password loss

These findings suggest the cryptographic "one-way property" represents a geometric barrier rather than mathematical irreversibility.

## Repository Contents

```
├── paper/
│   └── TOPOLOGICAL_COLLAPSE_2026.pdf
├── data/
│   ├── case1/ ... case5/
│   │   ├── report.pdf
│   │   ├── report.txt
│   │   └── layer_data.csv #raw layer outputs
└── README.md

```
## Data Availability

This repository contains raw data for five fully documented use-cases including:
- Complete layer bitstrings
- Sign sequences
- Character position mappings
- Charge polarity analysis

**Additional raw data** (layer activations, extended test cases, analysis scripts) is available upon request for verified researchers and collaborators.

## Seeking Collaborators

We are actively seeking collaboration with:
- Cryptanalysts
- Information theorists
- Security researchers

Open challenges include:
- Blind byte identification without reference string
- Sequence reconstruction from position data
- Extension to additional hash algorithms (SHA-512, SHA-3, BLAKE3)
- Investigation of applicability to asymmetric cryptographic primitives

## Citation

```bibtex
@article{trauth2026topological,
  title={Topological Collapse: Persistent Localization of Cryptographic Preimages in Deep Neural Manifolds},
  author={Trauth, Stefan},
  year={2026},
  doi={10.5281/zenodo.18305804}
}
```

## License

This work is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
