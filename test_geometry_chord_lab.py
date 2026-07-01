import unittest

import numpy as np

from geometry_chord_lab import (
    GeometryConfig,
    build_bank,
    canonical_chord_geometry,
    circular_difference,
    retrieve,
)


class GeometryChordTests(unittest.TestCase):
    def test_geometry_is_invariant_to_endpoint_order(self):
        first = np.asarray([0.2, 5.7])
        second = np.asarray([2.3, 1.1])
        forward = canonical_chord_geometry(first, second)
        reverse = canonical_chord_geometry(second, first)
        self.assertTrue(np.allclose(forward[0], reverse[0]))
        self.assertTrue(np.allclose(forward[1], reverse[1]))

    def test_clean_query_retrieves_record(self):
        bank = build_bank(
            GeometryConfig(
                points=64,
                records=32,
                chords_per_record=12,
                seed=10,
            )
        )
        for record in range(32):
            result = retrieve(
                bank,
                bank.endpoint_angles[record],
                midpoint_sigma=0.2,
                arc_sigma=0.2,
            )
            self.assertEqual(result["full_geometry"][0], record)

    def test_full_geometry_distinguishes_same_midpoint_lengths(self):
        first = np.asarray([0.25, -0.75], dtype=np.float32)
        second = np.asarray([-0.25, 0.75], dtype=np.float32)
        midpoint, arc = canonical_chord_geometry(first, second)
        self.assertLess(
            float(circular_difference(midpoint[0], midpoint[1])),
            1e-6,
        )
        self.assertNotAlmostEqual(float(arc[0]), float(arc[1]))


if __name__ == "__main__":
    unittest.main()
