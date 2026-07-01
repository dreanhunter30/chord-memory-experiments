import unittest

import numpy as np

from lab import (
    BaseConfig,
    TestConfig,
    bits_of,
    build_blueprint,
    compile_memory,
    decode_payload,
    evaluate,
    retrieve_marker,
    values_of,
)


class ChordLabTests(unittest.TestCase):
    def test_bit_roundtrip_all_uint8(self):
        values = np.arange(256, dtype=np.uint16)
        self.assertTrue(
            np.array_equal(values, values_of(bits_of(values, 8)))
        )

    def test_single_record_exact_retrieval_and_payload(self):
        blueprint = build_blueprint(
            BaseConfig(256, 512, 1),
            seed=1,
            max_marker_k=32,
            max_bit_k=16,
        )
        memory = compile_memory(blueprint, marker_k=32, bit_k=16)
        for seed in range(20):
            rng = np.random.default_rng(seed)
            self.assertEqual(
                retrieve_marker(memory, 0, 0.0, 0.0, rng), 0
            )
            bits = decode_payload(memory, 0, 0.0, 0.0, rng)
            self.assertEqual(int(values_of(bits[None, :])[0]), blueprint.values[0])

    def test_zero_memory_is_near_chance_for_values(self):
        blueprint = build_blueprint(BaseConfig(1024, 4096, 64), seed=2)
        memory = compile_memory(
            blueprint, marker_k=64, bit_k=32, zero_payload=True
        )
        result = evaluate(
            memory,
            blueprint.values,
            TestConfig(64, 32, 0.0, 0.0, 4096),
            seed=2,
        )
        self.assertLess(result["value_acc_if_retrieved"], 0.03)

    def test_query_path_does_not_depend_on_payload_values(self):
        blueprint = build_blueprint(BaseConfig(1024, 4096, 64), seed=3)
        memory = compile_memory(blueprint, marker_k=64, bit_k=32)
        original = []
        changed = []
        for seed in range(50):
            original.append(
                retrieve_marker(
                    memory, seed % 64, 0.5, 0.3, np.random.default_rng(seed)
                )
            )
        payload_rng = np.random.default_rng(999)
        memory.payload_field[:] = (
            payload_rng.normal(size=memory.payload_field.shape)
            + 1j * payload_rng.normal(size=memory.payload_field.shape)
        )
        for seed in range(50):
            changed.append(
                retrieve_marker(
                    memory, seed % 64, 0.5, 0.3, np.random.default_rng(seed)
                )
            )
        self.assertEqual(original, changed)

    def test_blueprint_is_reused_across_hyperparameters(self):
        blueprint = build_blueprint(BaseConfig(1024, 4096, 64), seed=4)
        fingerprint = blueprint.fingerprint()
        for marker_k in (16, 32, 64, 96):
            for bit_k in (8, 16, 32, 48):
                compile_memory(blueprint, marker_k, bit_k)
                self.assertEqual(blueprint.fingerprint(), fingerprint)

    def test_reproducible_evaluation(self):
        blueprint = build_blueprint(BaseConfig(1024, 4096, 64), seed=5)
        memory = compile_memory(blueprint, marker_k=64, bit_k=32)
        config = TestConfig(64, 32, 0.5, 0.3, 256)
        first = evaluate(memory, blueprint.values, config, seed=9)
        second = evaluate(memory, blueprint.values, config, seed=9)
        for metric in (
            "retrieval_acc",
            "value_acc_if_retrieved",
            "bit_acc_if_retrieved",
            "joint_acc",
            "retrieved_trials",
        ):
            self.assertEqual(first[metric], second[metric])

    def test_invalid_even_redundancy_rejected(self):
        with self.assertRaises(ValueError):
            build_blueprint(
                BaseConfig(256, 512, 1),
                seed=1,
                redundancy=2,
                max_marker_k=32,
                max_bit_k=16,
            )


if __name__ == "__main__":
    unittest.main()
