# Results and limitations

## 1. Honest-query correction

The original prototype included payload-derived information in its query path,
so a zero-memory control could still report perfect recovery. The rewritten
benchmarks require queries to be independent of stored payload values. The
zero-memory control then falls near chance.

## 2. Shared associative field

With 12,288 payload dimensions, exact joint recovery decreased as more records
were superposed:

| Records | Exact joint recovery |
|---:|---:|
| 64 | 100% |
| 128 | 99.74% |
| 256 | 94.34% |
| 512 | 75.26% |
| 1,024 | 41.02% |
| 2,048 | 18.42% |

Scaling the representation by roughly 96 payload dimensions per record restored
about 99.7% in the tested 512–2,048 record range, but storage increased linearly.

## 3. Derived and isolated records

A deterministic codebook derived from record ID and seed avoided storing a
materialized codebook. At 8,192 records it used about 6 MiB of persistent state
instead of about 81 MiB while retaining approximately 99.8% exact recovery.

Isolated records allowed update and deletion without changing unrelated records.
At 1,000 and 1,500 dimensions they reached 100% in the small tested ranges, but
cost roughly 8–12 KB for a one-byte record and no longer benefited from
superposition.

## 4. Geometry

For random geometric records with 128 points and 16 chords per record, full
endpoint geometry outperformed midpoint-only matching. Under 60% chord removal
and angular noise 0.08:

| Records | Full endpoints | Midpoint only |
|---:|---:|---:|
| 64 | 98.96% | 79.17% |
| 256 | 98.44% | 61.20% |
| 1,024 | 96.35% | 45.31% |

This measures recognition of corrupted copies, not language understanding.

Mapping semantic embeddings to 128 points and 16 chords reached only about
29–31% Top-1 on the real-text experiment. Therefore the successful synthetic
geometry result did not transfer into an efficient semantic representation.

## 5. Semantic retrieval

The exploratory real-text benchmark contained 1,061 machine-generated records
and 100 queries:

| Method | Top-1 | Top-3 |
|---|---:|---:|
| Multilingual E5 embedding | 94% | 96% |
| Circular phase code, 1,000 bytes | 91% | 94% |
| SQLite FTS5 | 80% | 86% |

On 20 difficult paraphrases:

| Method | Top-1 | Top-3 |
|---|---:|---:|
| Multilingual E5 embedding | 70% | 80% |
| Circular phase code | 55% | 70% |
| SQLite FTS5 | 15% | 35% |

The corpus was machine-specific, so raw records and the collector are omitted.
These figures should be treated as exploratory rather than a standard benchmark.

## 6. Word-level semantic test

Forty Russian concepts were indexed as single canonical words. Queries used
either a different synonym or a short definition; the expected canonical word
was absent from every query.

| Representation | Bytes/word | Synonym Top-1 | Definition Top-1 |
|---|---:|---:|---:|
| Character n-grams | variable | 10% | 2.5% |
| Float32 embedding | 1,536 | 75% | 75% |
| Int8 scalar quantization | 388 | 75% | 75% |
| Int4 scalar quantization | 196 | 72.5% | 70% |
| 1-bit signs | 48 | 12.5% | 25% |
| Circular phase code, 128 dimensions | 128 | 22.5% | 32.5% |
| Circular phase code, 256 dimensions | 256 | 45% | 52.5% |
| Circular phase code, 512 dimensions | 512 | 65% | 75% |

This is compression of a semantic search index, not compression capable of
reconstructing the original text. For isolated short words, storing the text is
still smaller than storing any of these vectors.

## 7. Storage and fault injection

For 41 real memory records containing 8,728 UTF-8 bytes:

| Representation | Size |
|---|---:|
| Gzip archive | 4,378 bytes |
| Hamming(7,4) equivalent | 18,771 bytes |
| SQLite | 24,576 bytes |
| Isolated chord field | 27,772,416 bytes |

The isolated field retained exact recovery under the tested random bit flips and
30% simulated packet loss. It required 6,781 packets versus 21 packets for the
raw archive. The robustness therefore reflects very high redundancy rather than
free error correction.

## Limitations

- Small custom datasets and one embedding model.
- The real-text dataset is not published.
- Most experiments use fixed seeds rather than confidence intervals over many
  independent runs.
- No standard MTEB, BEIR, product-quantization, OPQ, or RaBitQ evaluation yet.
- Storage estimates do not include every runtime or model overhead.
- Fault injection is controlled simulation, not a physical damaged-drive test.

The current evidence supports further experimentation, but not a claim that
chord memory replaces databases, compression algorithms, vector quantization,
or conventional ECC.
