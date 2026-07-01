import unittest

import numpy as np

from derived_lab import (
    DerivedConfig,
    apply_record,
    build_memory,
    decode_payload,
    evaluate,
    generate_payloads,
    marker_layout,
    materialize_codebook,
    payload_layout,
)


class DerivedCodebookTests(unittest.TestCase):
    def config(self):
        return DerivedConfig(
            records=32,
            marker_dims=512,
            payload_dims=4096,
            payload_bits=8,
            marker_k=32,
            bit_k=16,
            seed=17,
        )

    def test_layout_is_deterministic_and_unique(self):
        config = self.config()
        first_idx, first_carrier = payload_layout(config, 7)
        second_idx, second_carrier = payload_layout(config, 7)
        self.assertTrue(np.array_equal(first_idx, second_idx))
        self.assertTrue(np.array_equal(first_carrier, second_carrier))
        self.assertEqual(len(np.unique(first_idx)), first_idx.size)
        marker_idx, _ = marker_layout(config, 7)
        self.assertEqual(len(np.unique(marker_idx)), marker_idx.size)

    def test_materialized_and_derived_decode_are_identical(self):
        config = self.config()
        payloads = generate_payloads(config, seed=18)
        memory, _ = build_memory(config, payloads)
        codebook = materialize_codebook(config)
        for record in range(config.records):
            first = decode_payload(
                memory,
                record,
                0.5,
                0.3,
                np.random.default_rng(record),
            )
            second = decode_payload(
                memory,
                record,
                0.5,
                0.3,
                np.random.default_rng(record),
                codebook,
            )
            self.assertTrue(np.array_equal(first, second))

    def test_ab_quality_metrics_are_identical(self):
        config = self.config()
        payloads = generate_payloads(config, seed=19)
        memory, _ = build_memory(config, payloads)
        codebook = materialize_codebook(config)
        derived = evaluate(
            memory,
            payloads,
            trials=128,
            noise_std=0.5,
            drop_prob=0.3,
            seed=20,
        )
        stored = evaluate(
            memory,
            payloads,
            trials=128,
            noise_std=0.5,
            drop_prob=0.3,
            seed=20,
            codebook=codebook,
        )
        for key in (
            "retrieval_acc",
            "payload_exact_acc",
            "payload_exact_if_retrieved",
            "bit_acc_if_retrieved",
        ):
            self.assertEqual(derived[key], stored[key])

    def test_update_and_delete_match_rebuild(self):
        config = self.config()
        payloads = generate_payloads(config, seed=21)
        memory, _ = build_memory(config, payloads)
        record = 9
        old = payloads[record].copy()
        apply_record(memory, record, old, -1.0)
        new = 1 - old
        payloads[record] = new
        apply_record(memory, record, new, 1.0)
        rebuilt, _ = build_memory(config, payloads)
        self.assertTrue(
            np.allclose(memory.payload_field, rebuilt.payload_field, atol=1e-6)
        )

    def test_derived_persistent_state_excludes_codebook(self):
        config = self.config()
        payloads = generate_payloads(config, seed=22)
        memory, _ = build_memory(config, payloads)
        codebook = materialize_codebook(config)
        self.assertLess(memory.persistent_bytes, codebook.nbytes)

    def test_isolated_update_does_not_change_other_records(self):
        base = self.config()
        config = DerivedConfig(
            **{
                **base.__dict__,
                "isolated": True,
            }
        )
        payloads = generate_payloads(config, seed=23)
        memory, _ = build_memory(config, payloads)
        before = memory.payload_field.copy()
        record = 11
        apply_record(memory, record, payloads[record], -1.0)
        apply_record(memory, record, 1 - payloads[record], 1.0)
        mask = np.arange(config.records) != record
        self.assertTrue(
            np.array_equal(before[mask], memory.payload_field[mask])
        )
        self.assertFalse(
            np.array_equal(before[record], memory.payload_field[record])
        )


if __name__ == "__main__":
    unittest.main()
