"""Shared base enum helpers."""

from enum import Enum
from typing import List, Type, TypeVar

BaseEnumT = TypeVar("BaseEnumT", bound="BaseEnum")


class BaseEnum(str, Enum):
    """Base class for string enums with common utility methods."""

    def __str__(self) -> str:
        return self.value

    @classmethod
    def values(cls) -> List[str]:
        """Return all enum values in declaration order."""
        return [member.value for member in cls]

    @classmethod
    def has_value(cls, value: str) -> bool:
        """Return whether the provided value exists in the enum."""
        return value in cls.values()

    @classmethod
    def from_value(cls: Type[BaseEnumT], value: str) -> BaseEnumT:
        """Parse a value into an enum member with a helpful error message."""
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(
                f"'{value}' is not a valid {cls.__name__}. \
Expected one of: {', '.join(cls.values())}"
            ) from exc
