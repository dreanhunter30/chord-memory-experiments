from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import shutil
import sqlite3
import tempfile
import time
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

from derived_lab import (
    DerivedConfig,
    DerivedMemory,
    build_memory,
    decode_payload,
    empty_payload_field,
)
from lab import MAX_RESULT_BYTES


def load_real_records(workspace: Path) -> list[tuple[str, str]]:
    sources = [
        ("project", workspace / ".codex-memory" / "memory.sqlite"),
        (
            "personal",
            Path.home() / ".codex" / "personal-memory" / "personal.sqlite",
        ),
    ]
    records: list[tuple[str, str]] = []
    for scope, path in sources:
        connection = sqlite3.connect(
            f"file:{path.resolve()}?mode=ro", uri=True
        )
        try:
            for record_id, text in connection.execute(
                "SELECT id, text FROM memories WHERE active=1 ORDER BY id"
            ):
                records.append((f"{scope}:{record_id}", text))
        finally:
            connection.close()
    return records


def archive_bytes(records: list[tuple[str, str]]) -> bytes:
    return "\n".join(
        json.dumps(
            {"id": record_id, "text": text},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for record_id, text in records
    ).encode("utf-8")


def encode_fixed_payloads(
    records: list[tuple[str, str]],
) -> tuple[np.ndarray, int, int]:
    encoded = [text.encode("utf-8") for _, text in records]
    max_text_bytes = max(map(len, encoded))
    record_bytes = max_text_bytes + 2
    matrix = np.zeros((len(records), record_bytes), dtype=np.uint8)
    for index, value in enumerate(encoded):
        length = len(value)
        matrix[index, 0] = length & 0xFF
        matrix[index, 1] = (length >> 8) & 0xFF
        matrix[index, 2 : 2 + length] = np.frombuffer(value, dtype=np.uint8)
    return (
        np.unpackbits(matrix, axis=1, bitorder="little").astype(np.int8),
        max_text_bytes,
        record_bytes,
    )


def create_sqlite(
    path: Path, records: list[tuple[str, str]]
) -> dict:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            "CREATE TABLE records("
            "id TEXT PRIMARY KEY, text TEXT NOT NULL, crc32 INTEGER NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO records(id, text, crc32) VALUES (?, ?, ?)",
            [
                (
                    record_id,
                    text,
                    zlib.crc32(text.encode("utf-8")),
                )
                for record_id, text in records
            ],
        )
        connection.commit()
        connection.execute("VACUUM")
        connection.commit()

        rng = random.Random(42)
        ids = [record_id for record_id, _ in records]
        started = time.perf_counter()
        for _ in range(20_000):
            record_id = ids[rng.randrange(len(ids))]
            connection.execute(
                "SELECT text FROM records WHERE id=?", (record_id,)
            ).fetchone()
        lookup_seconds = time.perf_counter() - started
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        connection.close()
    return {
        "bytes": path.stat().st_size,
        "lookup_qps": 20_000 / lookup_seconds,
        "integrity": integrity,
    }


def verify_sqlite(
    path: Path, expected: dict[str, str]
) -> tuple[bool, bool, bool, bool]:
    opened = integrity_ok = crc_ok = exact = False
    try:
        connection = sqlite3.connect(
            f"file:{path.resolve()}?mode=ro", uri=True
        )
        try:
            opened = True
            integrity_rows = connection.execute(
                "PRAGMA integrity_check"
            ).fetchall()
            integrity_ok = all(row[0] == "ok" for row in integrity_rows)
            rows = connection.execute(
                "SELECT id, text, crc32 FROM records"
            ).fetchall()
            crc_ok = all(
                zlib.crc32(text.encode("utf-8")) == crc
                for _, text, crc in rows
            )
            exact = (
                len(rows) == len(expected)
                and all(expected.get(record_id) == text for record_id, text, _ in rows)
            )
        finally:
            connection.close()
    except (sqlite3.Error, UnicodeError, OSError):
        pass
    return opened, integrity_ok, crc_ok, exact


def flip_random_bits(
    data: bytearray, count: int, rng: np.random.Generator
) -> None:
    positions = rng.integers(0, len(data) * 8, size=count)
    for position_value in positions:
        position = int(position_value)
        data[position // 8] ^= 1 << (position % 8)


def sqlite_fault_benchmark(
    clean_path: Path,
    records: list[tuple[str, str]],
    directory: Path,
) -> dict:
    clean = clean_path.read_bytes()
    expected = dict(records)
    rng = np.random.default_rng(500)
    modes = {
        "single_bit": ("bits", 1),
        "128_random_bits": ("bits", 128),
        "16_byte_burst": ("burst", 16),
        "truncate_4k": ("truncate", 4096),
    }
    output = {}
    for name, (kind, amount) in modes.items():
        opened = integrity_ok = crc_ok = exact = detected = silent = 0
        trials = 40
        for trial in range(trials):
            damaged = bytearray(clean)
            if kind == "bits":
                flip_random_bits(damaged, amount, rng)
            elif kind == "burst":
                start = int(rng.integers(100, max(101, len(damaged) - amount)))
                for offset in range(amount):
                    damaged[start + offset] ^= 0xFF
            else:
                del damaged[-min(amount, len(damaged) - 100) :]
            damaged_path = directory / f"damaged-{name}-{trial}.sqlite"
            damaged_path.write_bytes(damaged)
            result = verify_sqlite(damaged_path, expected)
            damaged_path.unlink(missing_ok=True)
            opened += result[0]
            integrity_ok += result[1]
            crc_ok += result[2]
            exact += result[3]
            fault_detected = not result[3] and (
                not result[0] or not result[1] or not result[2]
            )
            detected += fault_detected
            silent += bool(not result[3] and not fault_detected)
        output[name] = {
            "trials": trials,
            "open_rate": opened / trials,
            "integrity_ok_rate": integrity_ok / trials,
            "application_crc_ok_rate": crc_ok / trials,
            "exact_recovery_rate": exact / trials,
            "detected_failure_rate": detected / trials,
            "silent_data_corruption_rate": silent / trials,
        }

    # Three independent copies recover from one arbitrarily damaged copy.
    damaged = bytearray(clean)
    flip_random_bits(damaged, 128, rng)
    hashes = [
        hashlib.sha256(clean).hexdigest(),
        hashlib.sha256(clean).hexdigest(),
        hashlib.sha256(damaged).hexdigest(),
    ]
    majority = max(set(hashes), key=hashes.count)
    output["triple_copy_one_copy_damaged"] = {
        "storage_multiplier": 3,
        "majority_recovery": majority
        == hashlib.sha256(clean).hexdigest(),
    }
    return output


def hamming74_encode(data: bytes) -> tuple[np.ndarray, int]:
    source = np.frombuffer(data, dtype=np.uint8)
    nibbles = np.empty(source.size * 2, dtype=np.uint8)
    nibbles[0::2] = source & 0x0F
    nibbles[1::2] = source >> 4
    d = ((nibbles[:, None] >> np.arange(4)) & 1).astype(np.uint8)
    code = np.zeros((len(nibbles), 7), dtype=np.uint8)
    code[:, 2] = d[:, 0]
    code[:, 4] = d[:, 1]
    code[:, 5] = d[:, 2]
    code[:, 6] = d[:, 3]
    code[:, 0] = code[:, 2] ^ code[:, 4] ^ code[:, 6]
    code[:, 1] = code[:, 2] ^ code[:, 5] ^ code[:, 6]
    code[:, 3] = code[:, 4] ^ code[:, 5] ^ code[:, 6]
    bits = code.ravel()
    return bits, len(data)


def hamming74_decode(bits: np.ndarray, byte_length: int) -> bytes:
    code = bits[: byte_length * 14].reshape(-1, 7).copy()
    s1 = code[:, 0] ^ code[:, 2] ^ code[:, 4] ^ code[:, 6]
    s2 = code[:, 1] ^ code[:, 2] ^ code[:, 5] ^ code[:, 6]
    s4 = code[:, 3] ^ code[:, 4] ^ code[:, 5] ^ code[:, 6]
    syndrome = s1 + 2 * s2 + 4 * s4
    rows = np.flatnonzero(syndrome)
    code[rows, syndrome[rows] - 1] ^= 1
    nibbles = (
        code[:, 2]
        | (code[:, 4] << 1)
        | (code[:, 5] << 2)
        | (code[:, 6] << 3)
    )
    values = nibbles[0::2] | (nibbles[1::2] << 4)
    return values.astype(np.uint8).tobytes()[:byte_length]


def hamming_fault_benchmark(data: bytes) -> dict:
    encoded, byte_length = hamming74_encode(data)
    expected_crc = zlib.crc32(data)
    rng = np.random.default_rng(600)
    output = {
        "raw_bytes": len(data),
        "encoded_bytes_equivalent": (len(encoded) + 7) // 8,
        "storage_multiplier": len(encoded) / (len(data) * 8),
    }
    for probability in (1e-5, 1e-4, 1e-3):
        exact = detected = silent = 0
        trials = 100
        for _ in range(trials):
            damaged = encoded.copy()
            damaged ^= (rng.random(len(damaged)) < probability).astype(
                np.uint8
            )
            decoded = hamming74_decode(damaged, byte_length)
            is_exact = decoded == data
            crc_ok = zlib.crc32(decoded) == expected_crc
            exact += is_exact
            detected += bool(not is_exact and not crc_ok)
            silent += bool(not is_exact and crc_ok)
        output[f"independent_bit_flip_p={probability:g}"] = {
            "trials": trials,
            "exact_recovery_rate": exact / trials,
            "detected_failure_rate": detected / trials,
            "silent_data_corruption_rate": silent / trials,
        }
    for burst in (8, 64, 512):
        exact = detected = 0
        trials = 100
        for _ in range(trials):
            damaged = encoded.copy()
            start = int(rng.integers(0, len(damaged) - burst))
            damaged[start : start + burst] ^= 1
            decoded = hamming74_decode(damaged, byte_length)
            is_exact = decoded == data
            exact += is_exact
            detected += bool(
                not is_exact and zlib.crc32(decoded) != expected_crc
            )
        output[f"contiguous_burst_bits={burst}"] = {
            "trials": trials,
            "exact_recovery_rate": exact / trials,
            "detected_failure_rate": detected / trials,
        }
    return output


def evaluate_chord_records(
    memory: DerivedMemory,
    payloads: np.ndarray,
    seed: int,
    noise_std: float = 0.0,
    drop_prob: float = 0.0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    exact = bit_ok = 0
    for record in range(memory.config.records):
        decoded = decode_payload(
            memory, record, noise_std, drop_prob, rng
        )
        exact += bool(np.array_equal(decoded, payloads[record]))
        bit_ok += int(np.sum(decoded == payloads[record]))
    return (
        exact / memory.config.records,
        bit_ok / payloads.size,
    )


def chord_disk_fault_benchmark(
    memory: DerivedMemory, payloads: np.ndarray
) -> dict:
    clean_bytes = memory.payload_field.tobytes()
    clean_hash = hashlib.sha256(clean_bytes).hexdigest()
    rng = np.random.default_rng(700)
    output = {}

    for flips in (1, 128, 8192):
        exact_rates = []
        bit_rates = []
        trials = 20
        for trial in range(trials):
            damaged = bytearray(clean_bytes)
            flip_random_bits(damaged, flips, rng)
            field = np.frombuffer(damaged, dtype=np.complex64).copy()
            damaged_memory = DerivedMemory(memory.config, field.reshape(memory.payload_field.shape))
            exact, bit = evaluate_chord_records(
                damaged_memory, payloads, 710 + trial
            )
            exact_rates.append(exact)
            bit_rates.append(bit)
        output[f"serialized_random_bit_flips={flips}"] = {
            "trials": trials,
            "sha256_detected": True,
            "mean_record_exact_rate": float(np.mean(exact_rates)),
            "mean_bit_accuracy": float(np.mean(bit_rates)),
        }

    for fraction in (0.01, 0.1, 0.3):
        keep = int(len(clean_bytes) * (1.0 - fraction))
        damaged = clean_bytes[:keep] + bytes(len(clean_bytes) - keep)
        field = np.frombuffer(damaged, dtype=np.complex64).copy()
        damaged_memory = DerivedMemory(
            memory.config, field.reshape(memory.payload_field.shape)
        )
        exact, bit = evaluate_chord_records(
            damaged_memory, payloads, 730
        )
        output[f"zero_filled_tail_loss={fraction}"] = {
            "sha256_detected": hashlib.sha256(damaged).hexdigest()
            != clean_hash,
            "record_exact_rate": exact,
            "bit_accuracy": bit,
        }
    return output


def network_packet_benchmark(
    archive: bytes,
    memory: DerivedMemory,
    payloads: np.ndarray,
) -> dict:
    rng = np.random.default_rng(800)
    packet_size = 512
    packets = [
        archive[offset : offset + packet_size]
        for offset in range(0, len(archive), packet_size)
    ]
    output = {}
    corrupt_probability = 0.01
    for loss_probability in (0.01, 0.1, 0.3):
        raw_exact = triple_exact = 0
        trials = 1000
        for _ in range(trials):
            raw_ok = True
            triple_ok = True
            for _packet in packets:
                available = [
                    rng.random() >= loss_probability
                    and rng.random() >= corrupt_probability
                    for _copy in range(3)
                ]
                raw_ok &= available[0]
                triple_ok &= any(available)
            raw_exact += raw_ok
            triple_exact += triple_ok
        output[f"raw_and_triple_loss={loss_probability}"] = {
            "trials": trials,
            "packet_bytes": packet_size,
            "packets": len(packets),
            "packet_corruption_probability": corrupt_probability,
            "raw_crc_exact_rate": raw_exact / trials,
            "triple_crc_exact_rate": triple_exact / trials,
            "triple_storage_multiplier": 3,
        }

    field_bytes = memory.payload_field.tobytes()
    chord_packet_size = 4096
    packet_count = (
        len(field_bytes) + chord_packet_size - 1
    ) // chord_packet_size
    for loss_probability in (0.01, 0.1, 0.3):
        exact_rates = []
        bit_rates = []
        trials = 20
        for trial in range(trials):
            damaged = bytearray(field_bytes)
            lost = rng.random(packet_count) < loss_probability
            corrupt = rng.random(packet_count) < corrupt_probability
            unavailable = lost | corrupt
            for packet in np.flatnonzero(unavailable):
                start = int(packet) * chord_packet_size
                end = min(start + chord_packet_size, len(damaged))
                damaged[start:end] = bytes(end - start)
            field = np.frombuffer(damaged, dtype=np.complex64).copy()
            damaged_memory = DerivedMemory(
                memory.config, field.reshape(memory.payload_field.shape)
            )
            exact, bit = evaluate_chord_records(
                damaged_memory, payloads, 850 + trial
            )
            exact_rates.append(exact)
            bit_rates.append(bit)
        output[f"chord_packet_loss={loss_probability}"] = {
            "trials": trials,
            "packet_bytes": chord_packet_size,
            "packets": packet_count,
            "packet_corruption_probability": corrupt_probability,
            "mean_record_exact_rate": float(np.mean(exact_rates)),
            "mean_bit_accuracy": float(np.mean(bit_rates)),
        }
    return output


def run(workspace: Path) -> dict:
    started = time.perf_counter()
    records = load_real_records(workspace)
    archive = archive_bytes(records)
    payloads, max_text_bytes, record_bytes = encode_fixed_payloads(records)

    config = DerivedConfig(
        records=len(records),
        marker_dims=max(512, len(records) * 2),
        payload_dims=payloads.shape[1] * 3 * 8,
        payload_bits=payloads.shape[1],
        marker_k=32,
        bit_k=8,
        redundancy=3,
        seed=42,
        isolated=True,
    )
    chord_memory, chord_build_seconds = build_memory(config, payloads)
    clean_exact, clean_bits = evaluate_chord_records(
        chord_memory, payloads, 900
    )
    noisy_results = [
        evaluate_chord_records(
            chord_memory,
            payloads,
            910 + seed,
            noise_std=0.5,
            drop_prob=0.3,
        )
        for seed in range(5)
    ]

    query_rng = np.random.default_rng(920)
    query_started = time.perf_counter()
    for _ in range(500):
        record = int(query_rng.integers(0, len(records)))
        decode_payload(
            chord_memory, record, 0.0, 0.0, query_rng
        )
    chord_query_seconds = time.perf_counter() - query_started

    with tempfile.TemporaryDirectory(prefix="chord-real-fault-") as directory:
        directory_path = Path(directory)
        sqlite_path = directory_path / "records.sqlite"
        sqlite_metrics = create_sqlite(sqlite_path, records)
        sqlite_faults = sqlite_fault_benchmark(
            sqlite_path, records, directory_path
        )

    result = {
        "schema_version": 1,
        "suite": "real-text-storage-and-faults",
        "fault_model_note": (
            "Faults are injected into real serialized SQLite/CHORD bytes and "
            "packet streams; this is controlled simulation, not a failing "
            "physical SSD or network interface."
        ),
        "dataset": {
            "active_real_memory_records": len(records),
            "utf8_text_bytes": sum(
                len(text.encode("utf-8")) for _, text in records
            ),
            "archive_bytes": len(archive),
            "gzip_archive_bytes": len(gzip.compress(archive)),
            "max_text_bytes": max_text_bytes,
            "fixed_record_bytes": record_bytes,
            "padded_payload_bytes": payloads.size // 8,
            "content_in_result": False,
        },
        "storage": {
            "sqlite": sqlite_metrics,
            "hamming74": {
                "bytes_equivalent": (
                    len(hamming74_encode(archive)[0]) + 7
                )
                // 8,
                "multiplier_vs_archive": 1.75,
            },
            "isolated_chord": {
                "config": asdict(config),
                "field_bytes": chord_memory.payload_field.nbytes,
                "multiplier_vs_utf8_text": (
                    chord_memory.payload_field.nbytes
                    / sum(len(text.encode("utf-8")) for _, text in records)
                ),
                "build_seconds": chord_build_seconds,
                "clean_record_exact_rate": clean_exact,
                "clean_bit_accuracy": clean_bits,
                "noise_0.5_drop_0.3_record_exact_mean": float(
                    np.mean([item[0] for item in noisy_results])
                ),
                "noise_0.5_drop_0.3_bit_accuracy_mean": float(
                    np.mean([item[1] for item in noisy_results])
                ),
                "clean_decode_qps": 500 / chord_query_seconds,
            },
        },
        "disk_faults": {
            "sqlite_with_crc32": sqlite_faults,
            "hamming74_with_crc32": hamming_fault_benchmark(archive),
            "isolated_chord_with_sha256": chord_disk_fault_benchmark(
                chord_memory, payloads
            ),
        },
        "network_faults": network_packet_benchmark(
            archive, chord_memory, payloads
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return result


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
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.workspace.resolve())
    atomic_write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
