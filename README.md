# Chord Memory Experiments

Small, reproducible experiments with circular phase codes, geometric chords,
associative storage, semantic retrieval, and fault injection.

This repository does **not** claim a new production-ready database, compression
algorithm, or error-correcting code. The experiments produced mixed and mostly
negative results: chord representations can recognize noisy synthetic patterns,
but conventional storage, vector quantization, and error-correcting methods are
usually more practical.

## Main findings

- An early prototype leaked the expected payload into the query. It was excluded
  from the honest benchmarks.
- Full endpoint geometry recognized noisy synthetic chord records well:
  approximately 96% Top-1 accuracy for 1,024 records in the tested hard setting.
- On 1,061 machine-generated text records, a 384-dimensional multilingual E5
  embedding reached 94% Top-1, a 1,000-byte circular phase code reached 91%,
  and SQLite FTS5 reached 80%.
- On the 20 difficult paraphrased queries from that experiment, the same methods
  reached 70%, 55%, and 15% Top-1 respectively.
- On a separate 40-concept Russian word benchmark, full embeddings reached 75%
  Top-1 for both synonyms and definitions.
- Four-bit scalar quantization used an estimated 196 bytes per word and retained
  72.5% Top-1 for synonyms and 70% for definitions.
- A 512-byte circular phase code reached 65% and 75%; a 128-byte code fell to
  22.5% and 32.5%.
- Isolated chord storage preserved independent updates but expanded 8,728 bytes
  of real text to 27.8 MB in one test (about 3,182x).
- The large isolated representation survived severe simulated corruption, but
  this came from extreme redundancy. Hamming codes, checksums, replication, and
  ordinary compression were much more storage-efficient baselines.

See [RESULTS.md](RESULTS.md) for the tested conditions and limitations.

## Repository contents

- `lab.py` — shared-field associative storage benchmark.
- `derived_lab.py` — derived codebooks and isolated records.
- `geometry_chord_lab.py` — true endpoint-geometry recognition.
- `real_fault_lab.py` — controlled disk and network fault injection.
- `test_*.py` — deterministic unit tests.
- `semantic_word_bench.mjs` — synonym and definition retrieval benchmark.
- `results/` — selected aggregate results without source memory contents.

The machine-specific corpus collector and raw local memory records are
intentionally not included.

## Python benchmark

Requirements:

- Python 3.11+
- NumPy

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s . -p "test_*.py" -v
.\.venv\Scripts\python.exe geometry_chord_lab.py --output geometry_results.json
```

The current test suite contains 19 tests.

## Semantic word benchmark

Requirements:

- Node.js 20+
- pnpm or npm

```powershell
npm install
node semantic_word_bench.mjs
```

The first run downloads `Xenova/multilingual-e5-small`. The pipeline is
explicitly disposed after evaluation.

## Interpretation

The most defensible use of the current chord representation is an experimental
noise-tolerant fingerprint or reranking signal. The experiments do not show an
advantage over:

- compressed text for storage;
- scalar or product quantization for semantic vector indexes;
- SQLite FTS for exact lexical retrieval;
- standard ECC and checksums for data integrity.

## Reproducibility and privacy

Randomized experiments use fixed seeds. Faults are injected into serialized
bytes and packet streams; no physical disk or network interface is damaged.

The aggregate real-record fault result reports only byte counts and metrics.
The source records, credentials, machine inventory, user paths, model cache,
and local databases are excluded.

## Status

lecence MIT.
