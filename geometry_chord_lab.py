from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from lab import MAX_RESULT_BYTES


@dataclass(frozen=True)
class GeometryConfig:
    points: int
    records: int
    chords_per_record: int
    seed: int


@dataclass
class GeometryBank:
    config: GeometryConfig
    pairs: np.ndarray
    endpoint_angles: np.ndarray
    midpoint_angles: np.ndarray
    arc_lengths: np.ndarray
    pair_codes: np.ndarray


def canonical_chord_geometry(
    first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the minor-arc midpoint angle and central angle in [0, pi]."""
    signed = (second - first + np.pi) % (2.0 * np.pi) - np.pi
    midpoint = (first + signed / 2.0) % (2.0 * np.pi)
    arc = np.abs(signed)
    diameter = np.isclose(arc, np.pi, atol=1e-6)
    midpoint = np.where(diameter, midpoint % np.pi, midpoint)
    return midpoint, arc


def build_bank(config: GeometryConfig) -> GeometryBank:
    if config.points < 4:
        raise ValueError("At least four circle points are required")
    all_pairs = np.asarray(
        [
            (first, second)
            for first in range(config.points)
            for second in range(first + 1, config.points)
        ],
        dtype=np.int16,
    )
    if config.chords_per_record > len(all_pairs):
        raise ValueError("Too many chords requested")

    rng = np.random.default_rng(config.seed)
    pair_rows = np.empty(
        (config.records, config.chords_per_record, 2), dtype=np.int16
    )
    for record in range(config.records):
        selected = rng.choice(
            len(all_pairs), size=config.chords_per_record, replace=False
        )
        pair_rows[record] = all_pairs[selected]

    point_angles = np.linspace(
        0.0,
        2.0 * np.pi,
        config.points,
        endpoint=False,
        dtype=np.float32,
    )
    endpoint_angles = point_angles[pair_rows]
    midpoint, arc = canonical_chord_geometry(
        endpoint_angles[..., 0], endpoint_angles[..., 1]
    )
    pair_codes = (
        pair_rows[..., 0].astype(np.int32) * config.points
        + pair_rows[..., 1].astype(np.int32)
    )
    return GeometryBank(
        config=config,
        pairs=pair_rows,
        endpoint_angles=endpoint_angles.astype(np.float32),
        midpoint_angles=midpoint.astype(np.float32),
        arc_lengths=arc.astype(np.float32),
        pair_codes=pair_codes,
    )


def circular_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs((left - right + np.pi) % (2.0 * np.pi) - np.pi)


def retrieve(
    bank: GeometryBank,
    query_endpoints: np.ndarray,
    *,
    midpoint_sigma: float,
    arc_sigma: float,
) -> dict[str, tuple[int, float]]:
    if len(query_endpoints) == 0:
        return {
            "full_geometry": (0, 0.0),
            "midpoint_only": (0, 0.0),
            "snapped_pair": (0, 0.0),
        }

    query_midpoint, query_arc = canonical_chord_geometry(
        query_endpoints[:, 0], query_endpoints[:, 1]
    )
    midpoint_distance = circular_difference(
        bank.midpoint_angles[:, :, None],
        query_midpoint[None, None, :],
    )
    midpoint_kernel = np.exp(
        -0.5 * (midpoint_distance / midpoint_sigma) ** 2
    )
    template = bank.endpoint_angles[:, :, None, :]
    query = query_endpoints[None, None, :, :]
    direct_distance = (
        circular_difference(template[..., 0], query[..., 0]) ** 2
        + circular_difference(template[..., 1], query[..., 1]) ** 2
    )
    swapped_distance = (
        circular_difference(template[..., 0], query[..., 1]) ** 2
        + circular_difference(template[..., 1], query[..., 0]) ** 2
    )
    endpoint_distance_squared = np.minimum(
        direct_distance, swapped_distance
    ) / 2.0
    endpoint_sigma = max(midpoint_sigma, arc_sigma)
    full_kernel = np.exp(
        -0.5 * endpoint_distance_squared / (endpoint_sigma**2)
    )
    midpoint_scores = np.mean(np.max(midpoint_kernel, axis=1), axis=1)
    full_scores = np.mean(np.max(full_kernel, axis=1), axis=1)

    point_step = 2.0 * np.pi / bank.config.points
    snapped = (
        np.rint((query_endpoints % (2.0 * np.pi)) / point_step).astype(
            np.int32
        )
        % bank.config.points
    )
    snapped.sort(axis=1)
    valid = snapped[:, 0] != snapped[:, 1]
    snapped_codes = (
        snapped[valid, 0] * bank.config.points + snapped[valid, 1]
    )
    if len(snapped_codes):
        pair_scores = np.mean(
            np.any(
                bank.pair_codes[:, :, None]
                == snapped_codes[None, None, :],
                axis=1,
            ),
            axis=1,
        )
    else:
        pair_scores = np.zeros(bank.config.records, dtype=np.float32)

    def winner(scores: np.ndarray) -> tuple[int, float]:
        index = int(np.argmax(scores))
        if len(scores) == 1:
            margin = float(scores[index])
        else:
            copy = scores.copy()
            copy[index] = -np.inf
            margin = float(scores[index] - np.max(copy))
        return index, margin

    return {
        "full_geometry": winner(full_scores),
        "midpoint_only": winner(midpoint_scores),
        "snapped_pair": winner(pair_scores),
    }


def query_from_record(
    bank: GeometryBank,
    record: int,
    dropout: float,
    endpoint_noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    endpoints = bank.endpoint_angles[record]
    keep = rng.random(len(endpoints)) >= dropout
    if not np.any(keep):
        return np.empty((0, 2), dtype=np.float32)
    query = endpoints[keep].copy()
    if endpoint_noise:
        query += rng.normal(0.0, endpoint_noise, size=query.shape)
    return query % (2.0 * np.pi)


def evaluate_condition(
    bank: GeometryBank,
    *,
    dropout: float,
    endpoint_noise: float,
    trials: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    point_step = 2.0 * np.pi / bank.config.points
    midpoint_sigma = 2.0 * point_step
    arc_sigma = 2.0 * point_step
    correct = {
        "full_geometry": 0,
        "midpoint_only": 0,
        "snapped_pair": 0,
    }
    margins = {key: [] for key in correct}
    empty_queries = 0
    started = time.perf_counter()

    for _ in range(trials):
        target = int(rng.integers(0, bank.config.records))
        query = query_from_record(
            bank, target, dropout, endpoint_noise, rng
        )
        if len(query) == 0:
            empty_queries += 1
            continue
        results = retrieve(
            bank,
            query,
            midpoint_sigma=midpoint_sigma,
            arc_sigma=arc_sigma,
        )
        for method, (prediction, margin) in results.items():
            correct[method] += prediction == target
            margins[method].append(margin)

    elapsed = time.perf_counter() - started
    return {
        "dropout": dropout,
        "endpoint_noise_radians": endpoint_noise,
        "trials": trials,
        "empty_queries": empty_queries,
        "full_geometry_acc": correct["full_geometry"] / trials,
        "midpoint_only_acc": correct["midpoint_only"] / trials,
        "snapped_pair_acc": correct["snapped_pair"] / trials,
        "full_geometry_mean_margin": float(
            np.mean(margins["full_geometry"])
        )
        if margins["full_geometry"]
        else 0.0,
        "midpoint_only_mean_margin": float(
            np.mean(margins["midpoint_only"])
        )
        if margins["midpoint_only"]
        else 0.0,
        "snapped_pair_mean_margin": float(
            np.mean(margins["snapped_pair"])
        )
        if margins["snapped_pair"]
        else 0.0,
        "queries_per_second": trials / elapsed,
    }


def aggregate_runs(runs: list[dict]) -> dict:
    output = {
        key: runs[0][key]
        for key in (
            "dropout",
            "endpoint_noise_radians",
            "trials",
        )
    }
    output["runs"] = len(runs)
    output["total_trials"] = sum(run["trials"] for run in runs)
    output["empty_queries"] = sum(run["empty_queries"] for run in runs)
    for key in (
        "full_geometry_acc",
        "midpoint_only_acc",
        "snapped_pair_acc",
        "full_geometry_mean_margin",
        "midpoint_only_mean_margin",
        "snapped_pair_mean_margin",
        "queries_per_second",
    ):
        values = np.asarray([run[key] for run in runs])
        output[f"{key}_mean"] = float(np.mean(values))
        output[f"{key}_min"] = float(np.min(values))
        output[f"{key}_max"] = float(np.max(values))
    return output


def run_suite() -> dict:
    started = time.perf_counter()
    conditions = [
        ("clean", 0.0, 0.0),
        ("moderate", 0.3, 0.03),
        ("hard", 0.6, 0.08),
        ("extreme", 0.8, 0.15),
    ]
    results = []
    for records in (64, 256, 1024):
        config = GeometryConfig(
            points=128,
            records=records,
            chords_per_record=16,
            seed=1000 + records,
        )
        bank = build_bank(config)
        for condition, dropout, noise in conditions:
            runs = [
                evaluate_condition(
                    bank,
                    dropout=dropout,
                    endpoint_noise=noise,
                    trials=128,
                    seed=seed,
                )
                for seed in (42, 43, 44)
            ]
            results.append(
                {
                    "condition": condition,
                    "records": records,
                    **aggregate_runs(runs),
                }
            )

    return {
        "schema_version": 1,
        "suite": "true-circle-chord-geometry",
        "representation": {
            "circle_points": 128,
            "chords_per_record": 16,
            "full_geometry": (
                "order-invariant circular distance between both endpoint "
                "pairs; equivalent to full chord position and length"
            ),
            "midpoint_only": "phase-only ablation",
            "snapped_pair": "nearest discrete endpoint-pair baseline",
            "kernel_sigma_in_point_steps": 2.0,
        },
        "storage_per_record_bytes": {
            "endpoint_pair_uint16": 16 * 2 * 2,
            "full_geometry_float32": 16 * 2 * 4,
            "midpoint_only_float32": 16 * 4,
        },
        "results": results,
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_suite()
    atomic_write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
