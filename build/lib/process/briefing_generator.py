"""Gemini-backed briefings using vector search context from the service layer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from src.config.settings import BriefingGeneratorConfig, Settings
from src.service_layer.vector_search_service import (
    VectorSearchResponse,
    VectorSearchService,
)
from src.utils.timer import Timer


class BriefingGenerator:
    """Load briefing YAML, retrieve chunks via ``VectorSearchService``, call Gemini."""

    def __init__(
        self,
        configuration_root: Optional[Path] = None,
        *,
        vector_search: Optional[VectorSearchService] = None,
        timer: Optional[Timer] = None,
    ) -> None:
        """Load config; resolve API key from ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``.

        Repository ``.env`` is merged via ``Settings.load_repository_dotenv()`` before
        reading credentials (same behavior as other optional env bootstrap).

        Parameters:
            configuration_root: Optional YAML root (defaults to repo ``configuration/``).
            vector_search: Injectable search client (defaults to ``VectorSearchService``).
            timer: Shared timer for embed/query sections when using the default client.
        """
        Settings.load_repository_dotenv()
        self._config = Settings.load_briefing_generator_config(
            configuration_root=configuration_root,
        )
        self._api_key = self._resolve_gemini_api_key()
        self._timer = timer or Timer()
        self._vector_search = vector_search or VectorSearchService(
            configuration_root=configuration_root,
            timer=self._timer,
        )

    @property
    def config(self) -> BriefingGeneratorConfig:
        """Parsed briefing generator configuration."""
        return self._config

    @staticmethod
    def _resolve_gemini_api_key() -> str:
        raw = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not raw or not str(raw).strip():
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY must be set and non-empty",
            )
        return str(raw).strip()

    def generate(self) -> str:
        """Retrieve context per topic and return a Gemini-generated briefing string."""
        blocks: List[Tuple[str, VectorSearchResponse]] = []
        for topic in self._config.topics:
            response = self._vector_search.search_by_text(
                topic.vector_query,
                top_k=self._config.vector_top_k,
            )
            blocks.append((topic.name, response))
        prompt = self._build_prompt(blocks)
        return self._generate_with_gemini(prompt)

    def _format_hits(self, response: VectorSearchResponse) -> str:
        lines: List[str] = []
        for hit in response.hits:
            meta = json.dumps(hit.metadata, sort_keys=True) if hit.metadata else "{}"
            dist_part = "" if hit.distance is None else f", distance={hit.distance}"
            lines.append(f"  - key={hit.key}{dist_part}, metadata={meta}")
        return "\n".join(lines) if lines else "  (no hits)"

    def _build_prompt(self, blocks: List[Tuple[str, VectorSearchResponse]]) -> str:
        header = (
            "You are a news analyst. Using ONLY the retrieved chunk references below "
            "(keys and metadata), write a concise multi-topic briefing. "
            "Ground statements in the provided metadata when it contains text; "
            "do not invent facts not supported by the references.\n"
        )
        parts: List[str] = [header, "Retrieved context:", ""]
        for name, resp in blocks:
            parts.append(f"## Topic: {name}\n")
            parts.append(self._format_hits(resp))
            parts.append("")
        parts.append(
            "Produce a structured briefing with one section per topic.",
        )
        return "\n".join(parts)

    def _generate_with_gemini(self, prompt: str) -> str:
        import google.generativeai as genai  # pylint: disable=import-outside-toplevel

        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(self._config.model)
        response = model.generate_content(prompt)
        try:
            text = response.text
        except ValueError as exc:
            raise RuntimeError("Gemini returned no usable text") from exc
        if text is None or not str(text).strip():
            raise RuntimeError("Gemini returned empty text")
        return str(text).strip()
