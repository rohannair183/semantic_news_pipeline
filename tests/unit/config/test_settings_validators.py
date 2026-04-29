# pyright: reportPrivateUsage=false
"""Unit tests for Settings validator helper methods."""

import unittest

from src.config.settings import Settings


class TestSettingsLoadRequiredSection(unittest.TestCase):
    """This class tests _load_required_section."""

    def test_load_required_section_raises_for_missing_mapping(self):
        """_load_required_section: raises when the section is missing or not a mapping."""
        with self.assertRaises(ValueError):
            Settings._load_required_section(  # pylint: disable=protected-access
                {},
                "article_normalizer",
            )


class TestSettingsLoadOptionalSection(unittest.TestCase):
    """This class tests _load_optional_section."""

    def test_load_optional_section_returns_empty_dict_for_missing_value(self):
        """_load_optional_section: returns empty dict when the section is absent."""
        self.assertEqual(
            Settings._load_optional_section(  # pylint: disable=protected-access
                {},
                "article_ingestor",
            ),
            {},
        )

    def test_load_optional_section_raises_for_non_mapping_value(self):
        """_load_optional_section: raises when the section value is not a mapping."""
        with self.assertRaises(ValueError):
            Settings._load_optional_section(  # pylint: disable=protected-access
                {"article_ingestor": []},
                "article_ingestor",
            )


class TestSettingsLoadSelectedProfiles(unittest.TestCase):
    """This class tests _load_selected_profiles."""

    def test_load_selected_profiles_raises_for_empty_list(self):
        """_load_selected_profiles: raises when profiles_to_run is an empty list."""
        with self.assertRaises(ValueError):
            Settings._load_selected_profiles(  # pylint: disable=protected-access
                [],
                ["technology_daily"],
            )

    def test_load_selected_profiles_raises_for_unknown_profiles(self):
        """_load_selected_profiles: raises when profiles_to_run includes unknown values."""
        with self.assertRaises(ValueError):
            Settings._load_selected_profiles(  # pylint: disable=protected-access
                ["science_daily"],
                ["technology_daily"],
            )


class TestSettingsLoadOptionalPositiveInt(unittest.TestCase):
    """This class tests _load_optional_positive_int."""

    def test_load_optional_positive_int_returns_none_for_missing_value(self):
        """_load_optional_positive_int: returns None when the value is missing."""
        self.assertIsNone(
            Settings._load_optional_positive_int(  # pylint: disable=protected-access
                None,
                "article_ingestor.limit_per_profile",
            )
        )

    def test_load_optional_positive_int_raises_for_non_integer(self):
        """_load_optional_positive_int: raises when the value is not an integer."""
        with self.assertRaises(ValueError):
            Settings._load_optional_positive_int(  # pylint: disable=protected-access
                "not-an-int",
                "article_ingestor.limit_per_profile",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
