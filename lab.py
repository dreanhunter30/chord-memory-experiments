from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


BITS = 8
REDUNDANCY = 3
MAX_MARKER_K = 96
MAX_BIT_K = 48
MAX_RESULT_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class BaseConfig:
    marker_dims: int
    payload_dims: int
    records: int


@dataclass(frozen=True)
class TestConfig:
    marker_k: int
    bit_k: int
    noise_std: float
    drop_prob: float
    trials: int


@dataclass
class Blueprint:
    values: np.ndarray
    marker_idx: np.ndarray
    marker_carrier: np.ndarray
    payload_idx: np.ndarray
    payload_carrier: np.ndarray
    marker_dims: int
    payload_dims: int
    bits: int
    redundancy: int

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for array in (
            self.values,
            self.marker_idx,
            self.marker_carrier,
            self.payload_idx,
            self.payload_carrier,
        ):
            digest.update(array.tobytes())
        return digest.hexdigest()


@dataclass
class CompiledMemory:
    marker_idx: np.ndarray
    marker_carrier: np.ndarray
    payload_idx: np.ndarray
    payload_carrier: np.ndarray
    payload_field: np.ndarray
    marker_dims: int
    bits: int
    redundancy: int

    @property
    def metadata_bytes(self) -> int:
        return sum(
            x.nbytes
            for x in (
                self.marker_idx,
                self.marker_carrier,
                self.payload_idx,
                self.payload_carrier,
            )
        )

    @property
    def field_bytes(self) -> int:
        return self.payload_field.nbytes


def bits_of(values: np.ndarray, bits: int) -> np.ndarray:
    shifts = np.arange(bits, dtype=np.uint16)
    return ((values[:, None].astype(np.uint16) >> shifts) & 1).astype(np.int8)


def values_of(bits: np.ndarray) -> np.ndarray:
    shifts = np.arange(bits.shape[-1], dtype=np.uint16)
    return np.sum(bits.astype(np.uint16) << shifts, axis=-1, dtype=np.uint16)


def random_carrier(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    phase = rng.uniform(0.0, 2.0 * np.pi, size=shape)
    return np.exp(1j * phase).astype(np.complex64)


def build_blueprint(
    base: BaseConfig,
    seed: int,
    *,
    bits: int = BITS,
    redundancy: int = REDUNDANCY,
    max_marker_k: int = MAX_MARKER_K,
    max_bit_k: int = MAX_BIT_K,
) -> Blueprint:
    if bits < 1 or bits > 16:
        raise ValueError("bits must be in [1, 16]")
    if redundancy < 1 or redundancy % 2 == 0:
        raise ValueError("redundancy must be a positive odd number")
    if max_marker_k > base.marker_dims:
        raise ValueError("max_marker_k exceeds marker_dims")
    slots = bits * redundancy * max_bit_k
    if slots > base.payload_dims:
        raise ValueError("payload slots per record exceed payload_dims")

    value_rng = np.random.default_rng(seed)
    marker_rng = np.random.default_rng(seed + 10_000)
    payload_rng = np.random.default_rng(seed + 20_000)
    carrier_rng = np.random.default_rng(seed + 30_000)

    values = value_rng.integers(0, 1 << bits, size=base.records, dtype=np.uint16)
    marker_idx = np.empty((base.records, max_marker_k), dtype=np.int32)
    payload_idx = np.empty(
        (base.records, bits, redundancy, max_bit_k), dtype=np.int32
    )

    for record in range(base.records):
        marker_idx[record] = marker_rng.choice(
            base.marker_dims, size=max_marker_k, replace=False
        )
        payload_idx[record] = payload_rng.choice(
            base.payload_dims, size=slots, replace=False
        ).reshape(bits, redundancy, max_bit_k)

    return Blueprint(
        values=values,
        marker_idx=marker_idx,
        marker_carrier=random_carrier(carrier_rng, marker_idx.shape),
        payload_idx=payload_idx,
        payload_carrier=random_carrier(carrier_rng, payload_idx.shape),
        marker_dims=base.marker_dims,
        payload_dims=base.payload_dims,
        bits=bits,
        redundancy=redundancy,
    )


def compile_memory(
    blueprint: Blueprint,
    marker_k: int,
    bit_k: int,
    *,
    zero_payload: bool = False,
) -> CompiledMemory:
    if not 1 <= marker_k <= blueprint.marker_idx.shape[-1]:
        raise ValueError("marker_k outside blueprint range")
    if not 1 <= bit_k <= blueprint.payload_idx.shape[-1]:
        raise ValueError("bit_k outside blueprint range")

    marker_idx = np.ascontiguousarray(blueprint.marker_idx[:, :marker_k])
    marker_carrier = np.ascontiguousarray(
        blueprint.marker_carrier[:, :marker_k]
    )
    payload_idx = np.ascontiguousarray(blueprint.payload_idx[..., :bit_k])
    payload_carrier = np.ascontiguousarray(
        blueprint.payload_carrier[..., :bit_k]
    )
    payload_field = np.zeros(blueprint.payload_dims, dtype=np.complex64)

    if not zero_payload:
        signs = (1 - 2 * bits_of(blueprint.values, blueprint.bits)).astype(
            np.float32
        )
        contributions = (
            payload_carrier * signs[:, :, None, None]
        ).astype(np.complex64)
        np.add.at(payload_field, payload_idx.ravel(), contributions.ravel())

    return CompiledMemory(
        marker_idx=marker_idx,
        marker_carrier=marker_carrier,
        payload_idx=payload_idx,
        payload_carrier=payload_carrier,
        payload_field=payload_field,
        marker_dims=blueprint.marker_dims,
        bits=blueprint.bits,
        redundancy=blueprint.redundancy,
    )


def corrupt_complex(
    values: np.ndarray,
    noise_std: float,
    drop_prob: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if not 0.0 <= drop_prob <= 1.0:
        raise ValueError("drop_prob must be in [0, 1]")

    observed = values.astype(np.complex64, copy=True)
    if noise_std:
        scale = noise_std / math.sqrt(2.0)
        noise = rng.normal(0.0, scale, size=values.shape) + 1j * rng.normal(
            0.0, scale, size=values.shape
        )
        observed += noise.astype(np.complex64)
    if drop_prob:
        keep = rng.random(values.shape) >= drop_prob
        observed *= keep
    else:
        keep = np.ones(values.shape, dtype=bool)
    return observed, keep


def retrieve_marker(
    memory: CompiledMemory,
    target: int,
    noise_std: float,
    drop_prob: float,
    rng: np.random.Generator,
) -> int:
    query = np.zeros(memory.marker_dims, dtype=np.complex64)
    idx = memory.marker_idx[target]
    query[idx] = memory.marker_carrier[target]
    query, keep = corrupt_complex(query, noise_std, drop_prob, rng)
    if not np.any(keep):
        return int(rng.integers(0, memory.marker_idx.shape[0]))

    gathered = query[memory.marker_idx]
    scores = np.real(
        np.sum(np.conj(memory.marker_carrier) * gathered, axis=1)
    )
    winners = np.flatnonzero(scores == scores.max())
    return int(winners[int(rng.integers(0, len(winners)))])


def decode_payload(
    memory: CompiledMemory,
    record: int,
    noise_std: float,
    drop_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    idx = memory.payload_idx[record]
    stored = memory.payload_field[idx]
    observed, keep = corrupt_complex(stored, noise_std, drop_prob, rng)
    correlations = np.real(
        np.sum(np.conj(memory.payload_carrier[record]) * observed, axis=-1)
    )
    kept = np.sum(keep, axis=-1)

    votes = correlations < 0.0
    ties = kept == 0
    if np.any(ties):
        votes[ties] = rng.integers(0, 2, size=int(np.sum(ties))).astype(bool)
    return (np.sum(votes, axis=-1) > (memory.redundancy // 2)).astype(
        np.int8
    )


def evaluate(
    memory: CompiledMemory,
    expected_values: np.ndarray,
    config: TestConfig,
    seed: int,
) -> dict:
    if len(expected_values) != memory.marker_idx.shape[0]:
        raise ValueError("expected_values length does not match memory")

    target_rng = np.random.default_rng(seed + 100_000)
    marker_rng = np.random.default_rng(seed + 200_000)
    payload_rng = np.random.default_rng(seed + 300_000)
    targets = target_rng.integers(
        0, len(expected_values), size=config.trials
    )

    retrieval_ok = 0
    value_ok = 0
    bit_ok = 0
    decoded = 0
    expected_bits = bits_of(expected_values, memory.bits)

    started = time.perf_counter()
    for target in targets:
        target = int(target)
        predicted = retrieve_marker(
            memory,
            target,
            config.noise_std,
            config.drop_prob,
            marker_rng,
        )
        if predicted != target:
            continue

        retrieval_ok += 1
        predicted_bits = decode_payload(
            memory,
            predicted,
            config.noise_std,
            config.drop_prob,
            payload_rng,
        )
        predicted_value = int(values_of(predicted_bits[None, :])[0])
        value_ok += predicted_value == int(expected_values[target])
        bit_ok += int(np.sum(predicted_bits == expected_bits[target]))
        decoded += memory.bits

    elapsed = time.perf_counter() - started
    return {
        "retrieval_acc": retrieval_ok / config.trials,
        "value_acc_if_retrieved": value_ok / retrieval_ok
        if retrieval_ok
        else 0.0,
        "bit_acc_if_retrieved": bit_ok / decoded if decoded else 0.0,
        "joint_acc": value_ok / config.trials,
        "retrieved_trials": retrieval_ok,
        "seconds": elapsed,
    }


def detect_gpu() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
        name, total, free = [part.strip() for part in output.splitlines()[0].split(",")]
        return {
            "available": True,
            "name": name,
            "memory_total_mib": int(total),
            "memory_free_mib": int(free),
        }
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return {"available": False}


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    keys = (
        "base_id",
        "records",
        "marker_dims",
        "payload_dims",
        "marker_k",
        "bit_k",
        "noise_std",
        "drop_prob",
        "baseline",
    )
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)

    aggregated = []
    metrics = (
        "retrieval_acc",
        "value_acc_if_retrieved",
        "bit_acc_if_retrieved",
        "joint_acc",
        "seconds",
    )
    for group_key, group_rows in groups.items():
        item = dict(zip(keys, group_key))
        item["runs"] = len(group_rows)
        for metric in metrics:
            values = np.asarray([row[metric] for row in group_rows])
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_min"] = float(np.min(values))
            item[f"{metric}_max"] = float(np.max(values))
        item["metadata_bytes"] = group_rows[0]["metadata_bytes"]
        item["field_bytes"] = group_rows[0]["field_bytes"]
        aggregated.append(item)
    return aggregated


def profile_settings(profile: str) -> dict:
    if profile == "smoke":
        return {
            "seeds": [42],
            "bases": [
                BaseConfig(2048, 6144, 64),
                BaseConfig(4096, 12288, 128),
            ],
            "marker_k": [32, 64],
            "bit_k": [16, 32],
            "conditions": [(0.5, 0.3)],
            "trials": 256,
        }
    if profile == "full":
        return {
            "seeds": [42, 43, 44],
            "bases": [
                BaseConfig(2048, 6144, 64),
                BaseConfig(4096, 12288, 128),
                BaseConfig(8192, 24576, 256),
            ],
            "marker_k": [16, 32, 64, 96],
            "bit_k": [8, 16, 32, 48],
            "conditions": [(0.5, 0.3), (0.7, 0.45)],
            "trials": 512,
        }
    if profile == "stress":
        return {
            "seeds": [42, 43, 44],
            "bases": [
                BaseConfig(4096, 12288, records)
                for records in (64, 128, 256, 512, 1024, 2048)
            ],
            "marker_k": [32],
            "bit_k": [8, 16, 32],
            "conditions": [(0.5, 0.3), (0.7, 0.45)],
            "trials": 512,
        }
    if profile == "scale":
        return {
            "seeds": [42, 43, 44],
            "bases": [
                BaseConfig(4096, payload_dims, records)
                for records in (512, 1024, 2048)
                for payload_dims in (12288, 24576, 49152, 98304)
            ]
            + [BaseConfig(4096, 196608, 2048)],
            "marker_k": [32],
            "bit_k": [8, 16, 32, 48],
            "conditions": [(0.5, 0.3)],
            "trials": 512,
        }
    raise ValueError(f"Unknown profile: {profile}")


def run_suite(profile: str) -> dict:
    settings = profile_settings(profile)
    rows = []
    fingerprints: dict[str, set[str]] = {}
    started = time.perf_counter()

    for base_id, base in enumerate(settings["bases"]):
        for seed in settings["seeds"]:
            blueprint = build_blueprint(base, seed)
            fingerprint_key = f"base={base_id},seed={seed}"
            fingerprints.setdefault(fingerprint_key, set()).add(
                blueprint.fingerprint()
            )

            for marker_k in settings["marker_k"]:
                for bit_k in settings["bit_k"]:
                    memory = compile_memory(blueprint, marker_k, bit_k)
                    for noise_std, drop_prob in settings["conditions"]:
                        config = TestConfig(
                            marker_k=marker_k,
                            bit_k=bit_k,
                            noise_std=noise_std,
                            drop_prob=drop_prob,
                            trials=settings["trials"],
                        )
                        result = evaluate(
                            memory, blueprint.values, config, seed
                        )
                        rows.append(
                            {
                                "base_id": base_id,
                                **asdict(base),
                                **asdict(config),
                                "seed": seed,
                                "baseline": "memory",
                                "metadata_bytes": memory.metadata_bytes,
                                "field_bytes": memory.field_bytes,
                                **result,
                            }
                        )

            # Baselines use a representative middle configuration.
            marker_k = settings["marker_k"][len(settings["marker_k"]) // 2]
            bit_k = settings["bit_k"][len(settings["bit_k"]) // 2]
            noise_std, drop_prob = settings["conditions"][0]
            config = TestConfig(
                marker_k, bit_k, noise_std, drop_prob, settings["trials"]
            )
            zero_memory = compile_memory(
                blueprint, marker_k, bit_k, zero_payload=True
            )
            zero_result = evaluate(
                zero_memory, blueprint.values, config, seed
            )
            rows.append(
                {
                    "base_id": base_id,
                    **asdict(base),
                    **asdict(config),
                    "seed": seed,
                    "baseline": "zero_memory",
                    "metadata_bytes": zero_memory.metadata_bytes,
                    "field_bytes": zero_memory.field_bytes,
                    **zero_result,
                }
            )

            memory = compile_memory(blueprint, marker_k, bit_k)
            shuffle_rng = np.random.default_rng(seed + 900_000)
            shuffled = blueprint.values[shuffle_rng.permutation(base.records)]
            shuffled_result = evaluate(memory, shuffled, config, seed)
            rows.append(
                {
                    "base_id": base_id,
                    **asdict(base),
                    **asdict(config),
                    "seed": seed,
                    "baseline": "shuffled_labels",
                    "metadata_bytes": memory.metadata_bytes,
                    "field_bytes": memory.field_bytes,
                    **shuffled_result,
                }
            )

    fingerprint_reuse_ok = all(
        len(values) == 1 for values in fingerprints.values()
    )
    aggregated = aggregate(rows)
    best = []
    for base_id in range(len(settings["bases"])):
        candidates = [
            row
            for row in aggregated
            if row["base_id"] == base_id and row["baseline"] == "memory"
        ]
        best.append(
            max(
                candidates,
                key=lambda row: (
                    row["joint_acc_mean"],
                    -row["metadata_bytes"],
                ),
            )
        )

    return {
        "schema_version": 1,
        "profile": profile,
        "backend": "numpy-cpu",
        "backend_reason": (
            "Vectorized workloads are small; no CUDA Python package is "
            "installed, so GPU setup would cost more disk than this lab."
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "gpu": detect_gpu(),
        },
        "controls": {
            "payload_in_query": False,
            "separate_rng_streams": True,
            "same_blueprint_across_grid": fingerprint_reuse_ok,
            "baselines": ["zero_memory", "shuffled_labels"],
            "result_history_retained": False,
        },
        "settings": {
            "seeds": settings["seeds"],
            "bases": [asdict(base) for base in settings["bases"]],
            "marker_k": settings["marker_k"],
            "bit_k": settings["bit_k"],
            "conditions": settings["conditions"],
            "trials": settings["trials"],
        },
        "elapsed_seconds": time.perf_counter() - started,
        "best_by_base": best,
        "results": aggregated,
    }


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(encoded.encode("utf-8")) > MAX_RESULT_BYTES:
        raise ValueError("Result exceeds the 2 MiB lab limit")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("smoke", "full", "stress", "scale"),
        default="smoke",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/latest.json")
    )
    args = parser.parse_args()

    result = run_suite(args.profile)
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "profile": result["profile"],
                "backend": result["backend"],
                "elapsed_seconds": result["elapsed_seconds"],
                "controls": result["controls"],
                "best_by_base": result["best_by_base"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
