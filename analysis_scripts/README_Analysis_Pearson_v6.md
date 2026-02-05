# Analysis_Pearson_and_5_Bases_v6.py — GCIS Layer Analyzer v6.0

## Purpose

This is the primary analysis tool for examining the neural network's layer activations. It loads raw layer output data (CSV), computes inter-layer correlations, identifies structural clusters, and extracts bitstrings using 8 independent methods. The bitstrings are the binary representations that are compared against the original password to verify preimage recovery.

## What It Does

### 1. Layer Loading & Harmonization
- Loads CSV files containing raw neuron activation values per layer (rows = iterations, columns = neurons)
- Automatically harmonizes iteration counts across layers (truncates to minimum shared length)

### 2. Bitstring Extraction — 8 Independent Methods

Each method converts neuron activations into a binary string (one bit per neuron). Using multiple independent methods strengthens the evidence: if different mathematical approaches produce the same password-matching pattern, the signal is real.

| Method | Name | Logic |
|--------|------|-------|
| **v0** | Sum → Sign | Sum over all iterations; positive = 1, negative = 0 |
| **v1** | Turning Point Dominance | More upward than downward turning points = 1 |
| **v2** | Skewness | Right-skewed distribution (>0) = 1 |
| **v3** | Median → Sign | Median > 0 = 1 |
| **v4** | FFT Dominance | Dominant frequency above median of all neurons = 1 |
| **v5** | Entropy Threshold | Shannon entropy above median = 1 (high complexity) |
| **v6** | Compressibility | Poorly compressible (high complexity via zlib) = 1 |
| **v7** | Neighbor Correlation | Positively correlated with adjacent neuron = 1 |

### 3. Pearson Correlation Matrix

Computes pairwise Pearson correlation between layers using three aggregation strategies:
- **v0 (Sum)**: Mean activation per iteration (original method)
- **v2 (Skewness)**: Skewness across neurons per iteration
- **v3 (Median)**: Median across neurons per iteration

This reveals which layers behave similarly and which carry independent information.

### 4. Cluster Identification

Groups layers by their correlation sign patterns (positive/negative relationship to every other layer). Layers with identical sign patterns form a cluster, labeled with Greek letters (α, β, γ, ...). The resulting "cluster sequence" (or "genome") describes the network's information-geometric structure.

### 5. Hamming Distance Analysis

Computes Hamming distances between cluster patterns to quantify how structurally different the clusters are from each other.

### 6. Combined Bitstring & Statistics

Concatenates layer bitstrings in analysis order to produce a single binary sequence per method. Reports: length, balance (ones/zeros ratio), Shannon entropy, compression ratio, and average run length.

## How It Fits Into the Research Pipeline

```
Crypto_Hash_ECC_Generator_v3.py  →  Password + SHA-256 hash (ground truth)
        ↓
GCIS Neural Network (checkpoint)  →  Layer activation CSVs
        ↓
Analysis_Pearson_and_5_Bases_v6.py  ←  YOU ARE HERE
        ↓
Bitstrings compared to password binary → Preimage recovery verification
```

## Modes

### GUI Mode (default)
```bash
python Analysis_Pearson_and_5_Bases_v6.py
```
Three-tab interface:
1. **Layer Selection** — Load CSVs, select and order layers via checkboxes
2. **Analysis** — Pearson correlation, clusters, Hamming distances, summary export
3. **Bitstring Methods** — All 8 methods with individual export (TXT + JSON)

Includes visualization: correlation heatmaps, cluster sequence diagrams, Hamming distance matrices.

### CLI Mode
```bash
python Analysis_Pearson_and_5_Bases_v6.py --files layer1.csv layer2.csv --output results/
python Analysis_Pearson_and_5_Bases_v6.py --folder data/ --aggregation v2 --output results/
```

## Output Formats

- **TXT**: Human-readable summaries with v4-compatible export format (`LayerName: N Neurons, Bitstring: 010110...`)
- **JSON**: Machine-readable with full metadata (timestamps, layer details, bitstring statistics)
- **PNG/PDF/SVG**: Visualization exports (correlation matrix, cluster diagram, Hamming matrix)

## Requirements

- Python 3.x
- numpy, pandas, scipy
- tkinter (for GUI, included in standard Python)
- matplotlib (optional, for visualization)

## Author

Stefan Trauth, 2025–2026
