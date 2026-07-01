from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from lab import MAX_RESULT_BYTES, corrupt_complex


MASK64 = (1 << 64) - 1
GOLDEN64 = 0x9E3779B97F4A7C15
MARKER_DOMAIN = 0x4D41524B45525F31
PAYLOAD_DOMAIN = 0x5041594C4F414431


@dataclass(frozen=True)
class DerivedConfig:
    records: int
    marker_dims: int
    payload_dims: int
    payload_bits: int
    marker_k: int
    bit_k: int
    redundancy: int = 3
    seed: int = 42
    isolated: bool = False

    def validate(self) -> None:
        if self.records < 1:
            raise ValueError("records must be positive")
        if self.payload_bits < 1:
            raise ValueError("payload_bits must be positive")
        if self.redundancy < 1 or self.redundancy % 2 == 0:
            raise ValueError("redundancy must be positive and odd")
        if not 1 <= self.marker_k <= self.marker_dims:
            raise ValueError("marker_k outside marker_dims")
        slots = self.payload_bits * self.redundancy * self.bit_k
        if not 1 <= slots <= self.payload_dims:
            raise ValueError("payload layout exceeds payload_dims")


@dataclass
class DerivedMemory:
    config: DerivedConfig
    payload_field: np.ndarray

    @property
    def persistent_bytes(self) -> int:
        # Field plus a conservative allowance for config and seed scalars.
        return self.payload_field.nbytes + 64


@dataclass
class MaterializedCodebook:
    marker_idx: np.ndarray
    marker_carrier: np.ndarray
    payload_idx: np.ndarray
    payload_carrier: np.ndarray

    @property
    def nbytes(self) -> int:
        return sum(
            value.nbytes
            for value in (
                self.marker_idx,
                self.marker_carrier,
                self.payload_idx,
                self.payload_carrier,
            )
        )


def mix64_scalar(value: int) -> int:
    value = (value + GOLDEN64) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def mix64_array(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint64, copy=True)
    values += np.uint64(GOLDEN64)
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return values


def layout(
    config: DerivedConfig,
    record: int,
    *,
    domain: int,
    dims: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= record < config.records:
        raise IndexError("record outside memory")
    if count > dims:
        raise ValueError("layout count exceeds dimensions")

    base = mix64_scalar(
        config.seed ^ domain ^ (((record + 1) * GOLDEN64) & MASK64)
    )
    offset = mix64_scalar(base ^ 0xA5A5A5A5A5A5A5A5) % dims
    step = mix64_scalar(base ^ 0x5A5A5A5A5A5A5A5A) % dims
    step = max(1, step)
    while math.gcd(step, dims) != 1:
        step += 1
        if step == dims:
            step = 1

    positions = np.arange(count, dtype=np.uint64)
    indices = (
        (np.uint64(offset) + np.uint64(step) * positions)
        % np.uint64(dims)
    ).astype(np.int32)
    phase_input = positions + np.uint64(
        mix64_scalar(base ^ 0xC3C3C3C3C3C3C3C3)
    )
    hashes = mix64_array(phase_input)
    phase_units = (hashes >> np.uint64(40)).astype(np.float32)
    phases = phase_units * np.float32(2.0 * np.pi / (1 << 24))
    carriers = (np.cos(phases) + 1j * np.sin(phases)).astype(np.complex64)
    return indices, carriers


def marker_layout(
    config: DerivedConfig, record: int
) -> tuple[np.ndarray, np.ndarray]:
    return layout(
        config,
        record,
        domain=MARKER_DOMAIN,
        dims=config.marker_dims,
        count=config.marker_k,
    )


def payload_layout(
    config: DerivedConfig, record: int
) -> tuple[np.ndarray, np.ndarray]:
    shape = (
        config.payload_bits,
        config.redundancy,
        config.bit_k,
    )
    count = math.prod(shape)
    indices, carriers = layout(
        config,
        record,
        domain=PAYLOAD_DOMAIN,
        dims=config.payload_dims,
        count=count,
    )
    return indices.reshape(shape), carriers.reshape(shape)


def generate_payloads(config: DerivedConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(
        0,
        2,
        size=(config.records, config.payload_bits),
        dtype=np.int8,
    )


def apply_record(
    memory: DerivedMemory,
    record: int,
    payload_bits: np.ndarray,
    weight: float,
) -> None:
    config = memory.config
    bits = np.asarray(payload_bits, dtype=np.int8)
    if bits.shape != (config.payload_bits,):
        raise ValueError("payload shape does not match config")
    indices, carriers = payload_layout(config, record)
    signs = (1 - 2 * bits).astype(np.float32)
    contributions = (
        carriers * signs[:, None, None] * np.float32(weight)
    ).astype(np.complex64)
    target_field = (
        memory.payload_field[record]
        if config.isolated
        else memory.payload_field
    )
    np.add.at(
        target_field,
        indices.ravel(),
        contributions.ravel(),
    )


def empty_payload_field(config: DerivedConfig) -> np.ndarray:
    shape = (
        (config.records, config.payload_dims)
        if config.isolated
        else (config.payload_dims,)
    )
    return np.zeros(shape, dtype=np.complex64)


def build_memory(
    config: DerivedConfig, payloads: np.ndarray
) -> tuple[DerivedMemory, float]:
    config.validate()
    if payloads.shape != (config.records, config.payload_bits):
        raise ValueError("payload matrix does not match config")
    memory = DerivedMemory(
        config=config,
        payload_field=empty_payload_field(config),
    )
    started = time.perf_counter()
    for record in range(config.records):
        apply_record(memory, record, payloads[record], 1.0)
    return memory, time.perf_counter() - started


def materialize_codebook(config: DerivedConfig) -> MaterializedCodebook:
    marker_idx = np.empty(
        (config.records, config.marker_k), dtype=np.int32
    )
    marker_carrier = np.empty(
        (config.records, config.marker_k), dtype=np.complex64
    )
    payload_shape = (
        config.records,
        config.payload_bits,
        config.redundancy,
        config.bit_k,
    )
    payload_idx = np.empty(payload_shape, dtype=np.int32)
    payload_carrier = np.empty(payload_shape, dtype=np.complex64)
    for record in range(config.records):
        marker_idx[record], marker_carrier[record] = marker_layout(
            config, record
        )
        payload_idx[record], payload_carrier[record] = payload_layout(
            config, record
        )
    return MaterializedCodebook(
        marker_idx,
        marker_carrier,
        payload_idx,
        payload_carrier,
    )


def materialize_markers(
    config: DerivedConfig,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.empty((config.records, config.marker_k), dtype=np.int32)
    carriers = np.empty(
        (config.records, config.marker_k), dtype=np.complex64
    )
    for record in range(config.records):
        indices[record], carriers[record] = marker_layout(config, record)
    return indices, carriers


def retrieve_marker(
    config: DerivedConfig,
    marker_idx: np.ndarray,
    marker_carrier: np.ndarray,
    target: int,
    noise_std: float,
    drop_prob: float,
    rng: np.random.Generator,
) -> int:
    query = np.zeros(config.marker_dims, dtype=np.complex64)
    query[marker_idx[target]] = marker_carrier[target]
    query, _ = corrupt_complex(query, noise_std, drop_prob, rng)
    gathered = query[marker_idx]
    scores = np.real(np.sum(np.conj(marker_carrier) * gathered, axis=1))
    winners = np.flatnonzero(scores == scores.max())
    return int(winners[int(rng.integers(0, len(winners)))])


def decode_payload(
    memory: DerivedMemory,
    record: int,
    noise_std: float,
    drop_prob: float,
    rng: np.random.Generator,
    codebook: MaterializedCodebook | None = None,
) -> np.ndarray:
    config = memory.config
    if codebook is None:
        indices, carriers = payload_layout(config, record)
    else:
        indices = codebook.payload_idx[record]
        carriers = codebook.payload_carrier[record]
    source_field = (
        memory.payload_field[record]
        if config.isolated
        else memory.payload_field
    )
    observed, keep = corrupt_complex(
        source_field[indices], noise_std, drop_prob, rng
    )
    with np.errstate(over="ignore", invalid="ignore"):
        correlations = np.real(
            np.sum(np.conj(carriers) * observed, axis=-1)
        )
    votes = correlations < 0.0
    ties = np.sum(keep, axis=-1) == 0
    if np.any(ties):
        votes[ties] = rng.integers(
            0, 2, size=int(np.sum(ties))
        ).astype(bool)
    return (np.sum(votes, axis=-1) > (config.redundancy // 2)).astype(
        np.int8
    )


def evaluate(
    memory: DerivedMemory,
    payloads: np.ndarray,
    *,
    trials: int,
    noise_std: float,
    drop_prob: float,
    seed: int,
    codebook: MaterializedCodebook | None = None,
) -> dict:
    config = memory.config
    target_rng = np.random.default_rng(seed + 100_000)
    marker_rng = np.random.default_rng(seed + 200_000)
    payload_rng = np.random.default_rng(seed + 300_000)
    targets = target_rng.integers(0, config.records, size=trials)

    marker_started = time.perf_counter()
    if codebook is None:
        marker_idx, marker_carrier = materialize_markers(config)
    else:
        marker_idx = codebook.marker_idx
        marker_carrier = codebook.marker_carrier
    marker_build_seconds = time.perf_counter() - marker_started
    marker_working_bytes = marker_idx.nbytes + marker_carrier.nbytes

    retrieval_ok = 0
    payload_ok = 0
    bit_ok = 0
    decoded_bits = 0
    query_started = time.perf_counter()
    for target_value in targets:
        target = int(target_value)
        predicted = retrieve_marker(
            config,
            marker_idx,
            marker_carrier,
            target,
            noise_std,
            drop_prob,
            marker_rng,
        )
        if predicted != target:
            continue
        retrieval_ok += 1
        predicted_bits = decode_payload(
            memory,
            predicted,
            noise_std,
            drop_prob,
            payload_rng,
            codebook,
        )
        payload_ok += bool(np.array_equal(predicted_bits, payloads[target]))
        bit_ok += int(np.sum(predicted_bits == payloads[target]))
        decoded_bits += config.payload_bits
    query_seconds = time.perf_counter() - query_started

    return {
        "retrieval_acc": retrieval_ok / trials,
        "payload_exact_acc": payload_ok / trials,
        "payload_exact_if_retrieved": (
            payload_ok / retrieval_ok if retrieval_ok else 0.0
        ),
        "bit_acc_if_retrieved": (
            bit_ok / decoded_bits if decoded_bits else 0.0
        ),
        "marker_build_seconds": marker_build_seconds,
        "query_seconds": query_seconds,
        "queries_per_second": trials / query_seconds,
        "marker_working_bytes": marker_working_bytes,
    }


def field_hash(field: np.ndarray) -> str:
    return hashlib.sha256(field.tobytes()).hexdigest()


def update_delete_restart_test() -> dict:
    config = DerivedConfig(
        records=256,
        marker_dims=2048,
        payload_dims=49152,
        payload_bits=8,
        marker_k=32,
        bit_k=16,
        seed=700,
    )
    payloads = generate_payloads(config, seed=701)
    memory, _ = build_memory(config, payloads)
    rng = np.random.default_rng(702)

    operations = 10_000
    for _ in range(operations):
        record = int(rng.integers(0, config.records))
        old = payloads[record].copy()
        apply_record(memory, record, old, -1.0)
        new = rng.integers(0, 2, size=config.payload_bits, dtype=np.int8)
        payloads[record] = new
        apply_record(memory, record, new, 1.0)

    reference, _ = build_memory(config, payloads)
    difference = memory.payload_field - reference.payload_field
    denominator = max(float(np.linalg.norm(reference.payload_field)), 1e-12)
    relative_l2 = float(np.linalg.norm(difference) / denominator)
    max_abs = float(np.max(np.abs(difference)))

    before_delete = memory.payload_field.copy()
    deleted_records = rng.choice(config.records, size=32, replace=False)
    for record_value in deleted_records:
        record = int(record_value)
        apply_record(memory, record, payloads[record], -1.0)
    deleted_field_changed = not np.array_equal(
        before_delete, memory.payload_field
    )
    for record_value in deleted_records:
        record = int(record_value)
        apply_record(memory, record, payloads[record], 1.0)
    reinsert_denominator = max(
        float(np.linalg.norm(before_delete)), 1e-12
    )
    delete_reinsert_relative_l2 = float(
        np.linalg.norm(memory.payload_field - before_delete)
        / reinsert_denominator
    )

    with tempfile.TemporaryDirectory(prefix="chord-derived-") as directory:
        directory_path = Path(directory)
        field_path = directory_path / "field.npy"
        config_path = directory_path / "config.json"
        np.save(field_path, memory.payload_field, allow_pickle=False)
        config_path.write_text(
            json.dumps(asdict(config)), encoding="utf-8"
        )
        loaded_config = DerivedConfig(
            **json.loads(config_path.read_text(encoding="utf-8"))
        )
        loaded_memory = DerivedMemory(
            loaded_config,
            np.load(field_path, allow_pickle=False),
        )
        restart_hash_equal = field_hash(
            loaded_memory.payload_field
        ) == field_hash(
            memory.payload_field
        )
        restart_config_equal = loaded_config == config
        restart_result = evaluate(
            loaded_memory,
            payloads,
            trials=128,
            noise_std=0.5,
            drop_prob=0.3,
            seed=704,
        )

    result = evaluate(
        memory,
        payloads,
        trials=512,
        noise_std=0.5,
        drop_prob=0.3,
        seed=703,
    )
    return {
        "operations": operations,
        "relative_l2_drift": relative_l2,
        "max_abs_drift": max_abs,
        "deleted_records": len(deleted_records),
        "deleted_field_changed": deleted_field_changed,
        "delete_reinsert_relative_l2": delete_reinsert_relative_l2,
        "restart_hash_equal": restart_hash_equal,
        "restart_config_equal": restart_config_equal,
        "post_restart_payload_exact_acc": restart_result[
            "payload_exact_acc"
        ],
        "post_update_payload_exact_acc": result["payload_exact_acc"],
    }


def run_suite() -> dict:
    started = time.perf_counter()
    common = {
        "marker_k": 32,
        "redundancy": 3,
        "seed": 42,
    }
    configs = [
        DerivedConfig(
            records=2048,
            marker_dims=4096,
            payload_dims=196608,
            payload_bits=8,
            bit_k=32,
            **common,
        ),
        DerivedConfig(
            records=4096,
            marker_dims=8192,
            payload_dims=393216,
            payload_bits=8,
            bit_k=32,
            **common,
        ),
        DerivedConfig(
            records=8192,
            marker_dims=16384,
            payload_dims=786432,
            payload_bits=8,
            bit_k=32,
            **common,
        ),
        DerivedConfig(
            records=512,
            marker_dims=4096,
            payload_dims=393216,
            payload_bits=64,
            bit_k=16,
            **common,
        ),
        DerivedConfig(
            records=128,
            marker_dims=4096,
            payload_dims=393216,
            payload_bits=256,
            bit_k=16,
            **common,
        ),
    ]

    results = []
    for config in configs:
        payloads = generate_payloads(config, seed=1000 + config.payload_bits)
        memory, build_seconds = build_memory(config, payloads)
        metrics = []
        for run_seed in range(42, 52):
            metrics.append(
                evaluate(
                    memory,
                    payloads,
                    trials=512,
                    noise_std=0.5,
                    drop_prob=0.3,
                    seed=run_seed,
                )
            )

        aggregate = {}
        for key in (
            "retrieval_acc",
            "payload_exact_acc",
            "payload_exact_if_retrieved",
            "bit_acc_if_retrieved",
            "query_seconds",
            "queries_per_second",
            "marker_build_seconds",
        ):
            values = np.asarray([item[key] for item in metrics])
            aggregate[f"{key}_mean"] = float(np.mean(values))
            aggregate[f"{key}_min"] = float(np.min(values))
            aggregate[f"{key}_max"] = float(np.max(values))

        stored_codebook_estimate = (
            config.records
            * (
                config.marker_k
                + config.payload_bits * config.redundancy * config.bit_k
            )
            * (np.dtype(np.int32).itemsize + np.dtype(np.complex64).itemsize)
        )
        zero_memory = DerivedMemory(
            config,
            empty_payload_field(config),
        )
        zero_result = evaluate(
            zero_memory,
            payloads,
            trials=512,
            noise_std=0.5,
            drop_prob=0.3,
            seed=99,
        )
        raw_payload_bytes = (
            config.records * config.payload_bits + 7
        ) // 8
        results.append(
            {
                **asdict(config),
                "build_seconds": build_seconds,
                "derived_persistent_bytes": memory.persistent_bytes,
                "raw_payload_bytes": raw_payload_bytes,
                "storage_overhead_ratio": (
                    memory.persistent_bytes / raw_payload_bytes
                ),
                "materialized_codebook_bytes_estimate": stored_codebook_estimate,
                "persistent_bytes_saved": stored_codebook_estimate,
                "marker_working_bytes": metrics[0]["marker_working_bytes"],
                "zero_memory_payload_exact_acc": zero_result[
                    "payload_exact_acc"
                ],
                "zero_memory_bit_acc": zero_result[
                    "bit_acc_if_retrieved"
                ],
                **aggregate,
            }
        )

    ab_config = configs[0]
    ab_payloads = generate_payloads(ab_config, seed=1008)
    ab_memory, _ = build_memory(ab_config, ab_payloads)
    materialize_started = time.perf_counter()
    codebook = materialize_codebook(ab_config)
    materialize_seconds = time.perf_counter() - materialize_started
    derived_result = evaluate(
        ab_memory,
        ab_payloads,
        trials=512,
        noise_std=0.5,
        drop_prob=0.3,
        seed=88,
    )
    stored_result = evaluate(
        ab_memory,
        ab_payloads,
        trials=512,
        noise_std=0.5,
        drop_prob=0.3,
        seed=88,
        codebook=codebook,
    )

    return {
        "schema_version": 1,
        "suite": "derived-codebook",
        "controls": {
            "payload_in_query": False,
            "addresses_derived_from_id_and_seed": True,
            "persistent_codebook_stored": False,
            "separate_rng_streams": True,
            "seeds_per_point": 10,
            "trials_per_seed": 512,
        },
        "ab_equivalence": {
            "config": asdict(ab_config),
            "materialize_seconds": materialize_seconds,
            "materialized_codebook_bytes": codebook.nbytes,
            "derived": derived_result,
            "materialized": stored_result,
            "quality_metrics_equal": all(
                derived_result[key] == stored_result[key]
                for key in (
                    "retrieval_acc",
                    "payload_exact_acc",
                    "payload_exact_if_retrieved",
                    "bit_acc_if_retrieved",
                )
            ),
            "derived_query_slowdown": (
                derived_result["query_seconds"]
                / stored_result["query_seconds"]
            ),
            "derived_total_query_slowdown_including_marker_build": (
                (
                    derived_result["query_seconds"]
                    + derived_result["marker_build_seconds"]
                )
                / (
                    stored_result["query_seconds"]
                    + stored_result["marker_build_seconds"]
                )
            ),
        },
        "update_delete_restart": update_delete_restart_test(),
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_isolated_suite() -> dict:
    started = time.perf_counter()
    rows = []
    for records in (16, 32, 64, 128):
        for payload_dims in (1000, 1500):
            common = {
                "records": records,
                "marker_dims": max(512, records * 2),
                "payload_dims": payload_dims,
                "payload_bits": 8,
                "marker_k": 32,
                "bit_k": 32,
                "redundancy": 3,
                "seed": 42,
            }
            shared_config = DerivedConfig(**common, isolated=False)
            isolated_config = DerivedConfig(**common, isolated=True)
            payloads = generate_payloads(shared_config, seed=1500 + records)
            shared_memory, shared_build = build_memory(
                shared_config, payloads
            )
            isolated_memory, isolated_build = build_memory(
                isolated_config, payloads
            )

            shared_metrics = []
            isolated_metrics = []
            for run_seed in range(42, 52):
                shared_metrics.append(
                    evaluate(
                        shared_memory,
                        payloads,
                        trials=256,
                        noise_std=0.5,
                        drop_prob=0.3,
                        seed=run_seed,
                    )
                )
                isolated_metrics.append(
                    evaluate(
                        isolated_memory,
                        payloads,
                        trials=256,
                        noise_std=0.5,
                        drop_prob=0.3,
                        seed=run_seed,
                    )
                )

            before = isolated_memory.payload_field.copy()
            target = records // 2
            old_payload = payloads[target].copy()
            new_payload = 1 - old_payload
            apply_record(isolated_memory, target, old_payload, -1.0)
            apply_record(isolated_memory, target, new_payload, 1.0)
            other_rows = np.arange(records) != target
            other_records_unchanged = bool(
                np.array_equal(
                    before[other_rows],
                    isolated_memory.payload_field[other_rows],
                )
            )
            target_record_changed = not np.array_equal(
                before[target], isolated_memory.payload_field[target]
            )

            zero_memory = DerivedMemory(
                isolated_config,
                empty_payload_field(isolated_config),
            )
            zero_result = evaluate(
                zero_memory,
                payloads,
                trials=512,
                noise_std=0.5,
                drop_prob=0.3,
                seed=99,
            )

            rows.append(
                {
                    "records": records,
                    "payload_dims_per_record": payload_dims,
                    "payload_bits": 8,
                    "bit_k": 32,
                    "trials": 2560,
                    "shared_exact_acc": float(
                        np.mean(
                            [
                                item["payload_exact_acc"]
                                for item in shared_metrics
                            ]
                        )
                    ),
                    "isolated_exact_acc": float(
                        np.mean(
                            [
                                item["payload_exact_acc"]
                                for item in isolated_metrics
                            ]
                        )
                    ),
                    "isolated_bit_acc": float(
                        np.mean(
                            [
                                item["bit_acc_if_retrieved"]
                                for item in isolated_metrics
                            ]
                        )
                    ),
                    "isolated_retrieval_acc": float(
                        np.mean(
                            [
                                item["retrieval_acc"]
                                for item in isolated_metrics
                            ]
                        )
                    ),
                    "shared_persistent_bytes": shared_memory.persistent_bytes,
                    "isolated_persistent_bytes": (
                        isolated_memory.persistent_bytes
                    ),
                    "isolated_bytes_per_record": (
                        isolated_memory.persistent_bytes / records
                    ),
                    "shared_build_seconds": shared_build,
                    "isolated_build_seconds": isolated_build,
                    "other_records_unchanged_after_update": (
                        other_records_unchanged
                    ),
                    "target_record_changed_after_update": (
                        target_record_changed
                    ),
                    "zero_memory_exact_acc": zero_result[
                        "payload_exact_acc"
                    ],
                }
            )

    return {
        "schema_version": 1,
        "suite": "isolated-records",
        "controls": {
            "separate_dense_payload_field_per_record": True,
            "payload_in_query": False,
            "seeds_per_point": 10,
            "trials_per_seed": 256,
            "noise_std": 0.5,
            "drop_prob": 0.3,
        },
        "results": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(encoded.encode("utf-8")) > MAX_RESULT_BYTES:
        raise ValueError("Result exceeds lab limit")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("derived", "isolated"),
        default="derived",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/latest-derived.json"),
    )
    args = parser.parse_args()
    result = (
        run_isolated_suite()
        if args.suite == "isolated"
        else run_suite()
    )
    atomic_write(args.output, result)
    summary = {
        "suite": result["suite"],
        "elapsed_seconds": result["elapsed_seconds"],
        "results": result["results"],
        "output": str(args.output.resolve()),
    }
    if args.suite == "derived":
        summary["ab_equivalence"] = result["ab_equivalence"]
        summary["update_delete_restart"] = result[
            "update_delete_restart"
        ]
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
