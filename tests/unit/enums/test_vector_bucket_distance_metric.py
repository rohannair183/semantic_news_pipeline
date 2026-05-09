"""This class tests VectorBucketDistanceMetric."""

import unittest

from src.enums.vector_bucket_distance_metric import VectorBucketDistanceMetric


class TestVectorBucketDistanceMetric(unittest.TestCase):
    """This class tests VectorBucketDistanceMetric helpers."""

    def test_from_value_accepts_l2(self) -> None:
        """from_value: parses l2."""
        self.assertEqual(
            VectorBucketDistanceMetric.from_value("l2"),
            VectorBucketDistanceMetric.L2,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
