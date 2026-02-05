# GCIS Bitstring Search Tool vXIV

**Preimage Localization in Neural Network Layer Activations**

A GUI-based tool for searching binary substrings (preimage fragments) within the bitstring representations of neural network layer activations. Designed for the GCIS (Generalized Cryptographic Information Space) research pipeline.

## What It Does

The tool loads exported layer activation bitstrings from a GCIS neural network checkpoint and searches for binary representations of passwords, hashes (SHA-256, SHA-512, MD5), or custom binary strings within those layers.

### Core Capabilities

1. **Multi-Format Binary Search** — Searches 1–32 bit binary strings across all loaded layers. Auto-detects bit length from input. Supports password bytes (8-bit ASCII), hash fragments, and arbitrary binary patterns.

2. **Match Rate Analysis** — Categorizes layers by match completeness: 100% (all search bytes found), N-1 (one byte missing), and partial matches. Reports distribution statistics across all layers.

3. **Missing String Analysis** — For N-1 layers, identifies exactly which byte is missing, its position index, and its ASCII/hex representation. Aggregates missing byte frequency across layers.

4. **Position Correlation Analysis** — Compares absolute and normalized bit positions across selected layers. Uses tolerance bands (±16 bits for layers ≥1024 bits, ±8 bits for smaller layers) to determine if preimage bytes consistently localize at the same structural positions.

5. **Publication-Ready Visualizations** — Heatmap (normalized position localization), bar chart (match rate vs. layer size), bitstring map (bit-level localization with found positions highlighted in red), and position match score chart with summary statistics. All exportable as PNG (300 DPI), SVG, and PDF.

6. **Preprint Export System** — Selective layer export with checkboxes (≥94% threshold). Generates simultaneously: summary CSV, LaTeX table, raw bitstring CSV, and publication plots (PNG + SVG per plot type and per layer).

7. **Full Results Export** — CSV + TXT + PDF export with configurable match rate filters (100%, ≥90%, ≥60%, ≥40%, All).

## How It Fits Into the Research Pipeline

```
Crypto_Hash_ECC_Generator_v3.py     →  Password + SHA-256/512 hash
        ↓
GCIS Neural Network (checkpoint)     →  Layer activation CSVs
        ↓
Layer_Activation_Bitstring_Export    →  TXT with layer bitstrings
        ↓
GCIS Bitstring Search Tool vXIV     ←  YOU ARE HERE
        ↓
Preimage localization evidence:
where password/hash bytes appear
within the network's internal
binary representations
```

## Usage

```bash
python Analyse_binärer_code_krypto_XIII.py
```

### GUI Workflow

1. **Load TXT File** — Select the bitstring export file (format: `layer_name: N Neurons, Bitstring: 010011...`)
2. **Set Data Type** — Password, Hash (MD5/SHA-256/SHA-512), or Custom
3. **Enter Identifier** — Used for export filenames
4. **Paste Binary Search String** — Space-separated binary strings (e.g., `01010011 01110100 01100101`)
5. **SEARCH** — Results appear in three panels: Statistics, Layer Details, Missing String Analysis

### Export Options

- **PREPRINT EXPORT** — Select layers → exports plots + CSV + LaTeX + raw bitstrings
- **Heatmap / Bar Chart / Bitstring Map** — Interactive plot windows with save options
- **Position Correlation** — Requires prior layer selection via Preprint Export (min. 2 layers)
- **Export Results** — Full CSV + TXT + PDF simultaneous export

### Input Format

The tool expects TXT files with this pattern per layer:
```
layer_name: 256 Neurons, Bitstring: 0110100101010...
```

## Requirements

- Python 3.x
- tkinter (GUI)
- numpy (position correlation, bitstring visualization)
- matplotlib (all plots)
- reportlab (optional, PDF export)

## Author

Stefan Trauth, 2025–2026
