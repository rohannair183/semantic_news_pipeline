"""PostgreSQL identifier validation for safe DDL composition."""

from __future__ import annotations

import re

_PG_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_pg_identifier(name: str, *, field_name: str) -> str:
    """Return ``name`` if it matches a safe PostgreSQL identifier pattern.

    Raises:
        ValueError: When empty or invalid.
    """
    candidate = name.strip()
    if not candidate:
        raise ValueError(f"{field_name} must be non-empty")
    if not _PG_IDENTIFIER_RE.fullmatch(candidate):
        raise ValueError(
            f"{field_name} must match pattern [A-Za-z_][A-Za-z0-9_]* (got {name!r})",
        )
    return candidate
