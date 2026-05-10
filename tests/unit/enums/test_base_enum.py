"""Unit tests for reusable enum helpers."""

import unittest

from src.enums.base import BaseEnum
from src.enums.yaml_config_type import YAMLConfigType


class SampleEnum(BaseEnum):
    """Local enum used to validate BaseEnum behavior."""

    ALPHA = "alpha"
    BETA = "beta"


class TestBaseEnum(unittest.TestCase):
    """This class tests BaseEnum utility methods."""

    def test_values_str_and_has_value(self):
        """BaseEnum: list and membership helpers return expected values."""
        self.assertEqual(SampleEnum.values(), ["alpha", "beta"])
        self.assertEqual(str(SampleEnum.ALPHA), "alpha")
        self.assertTrue(SampleEnum.has_value("beta"))
        self.assertFalse(SampleEnum.has_value("gamma"))

    def test_from_value_success_and_error(self):
        """BaseEnum: from_value parses known values and rejects unknown ones."""
        self.assertEqual(SampleEnum.from_value("alpha"), SampleEnum.ALPHA)
        with self.assertRaises(ValueError) as raised:
            SampleEnum.from_value("gamma")
        self.assertIn("SampleEnum", str(raised.exception))
        self.assertIn("alpha, beta", str(raised.exception))


class TestYAMLConfigType(unittest.TestCase):
    """This class tests YAMLConfigType enum."""

    def test_ingestion_value(self):
        """YAMLConfigType: exposes ingestion config group value."""
        self.assertEqual(YAMLConfigType.INGESTION.value, "ingestion")
        self.assertEqual(YAMLConfigType.from_value("ingestion"), YAMLConfigType.INGESTION)

    def test_orchestration_value(self):
        """YAMLConfigType: exposes orchestrator YAML directory name."""
        self.assertEqual(YAMLConfigType.ORCHESTRATION.value, "orchestration")
        self.assertEqual(YAMLConfigType.from_value("orchestration"), YAMLConfigType.ORCHESTRATION)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
