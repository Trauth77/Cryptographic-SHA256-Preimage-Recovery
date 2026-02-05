# Crypto_Hash_ECC_Generator_v3.py — Cryptographic Hash & ECC Generator v3.0

## Purpose

This tool generates the **input data** used in the SHA-256 and ECC preimage/key recovery experiments. It takes a plaintext password and computes cryptographic outputs across four methods, all convertible to binary/hex at configurable bit-chunk sizes (8–32 bit).

The output serves as ground truth for validating the neural network's preimage localization results.

## Supported Cryptographic Methods

| Method | Output | Description |
|--------|--------|-------------|
| **MD5** | 128-bit hash | MD5 digest of password |
| **SHA-256** | 256-bit hash | SHA-256 digest of password |
| **ECC-128** (secp128r1) | Private key, public key, ECDSA signature | Deterministic key derivation from password via truncated SHA-256 (128 bit) |
| **ECC-256** (secp256k1) | Private key, public key, ECDSA signature | Deterministic key derivation from password via SHA-256 (256 bit) |

All ECC keys are derived deterministically — same password always produces the same key pair and signature (RFC 6979).

## What It Does

1. **Password → Crypto Computation**: Select method, enter password, get full cryptographic output.
2. **Bit-Length Compatibility Analysis**: Automatically determines which chunk sizes (8–32 bit) divide both password and output evenly. Critical for aligning binary representations in the neural network layer analysis.
3. **Binary & Hex Chunk Output**: Displays all outputs as chunked binary and hex strings. These are the exact format used as input data for the GCIS neural architecture.
4. **History & Export**: Stores last 25 entries with hashes (JSON). Exports full reports as `.txt`.

## How It Fits Into the Research Pipeline

```
Crypto_Hash_ECC_Generator_v3.py  →  Plaintext + hash/key (ground truth)
        ↓
GCIS Neural Network (checkpoint)  →  Layer activations from hash/key input
        ↓
Analysis Scripts                  →  Bit-sign matching, Pearson correlation,
                                     position mapping in ES/ZFA layers
```

## Requirements

- Python 3.x
- tkinter (included in standard Python)
- `ecdsa` library for ECC methods: `pip install ecdsa`

## Usage

```bash
python Crypto_Hash_ECC_Generator_v1.py
```

Select method → enter password → select bit-length → click COMPUTE & SAVE.

## Author

Claude Opus
