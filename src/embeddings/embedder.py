"""Embed chunked parquet into vector embeddings and write to checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config.settings import Settings
from src.embeddings.providers import resolve_provider


class Embedder:
    """Run YAML-configured embedding on chunked parquet files."""

    def __init__(self, configuration_root: Path | None = None):
        """Load embedding settings from YAML configuration."""
        self._config = Settings.load_embedding_config(
            configuration_root=configuration_root,
        )

    @property
    def provider_name(self) -> str:
        """Return the configured embedding provider value."""
        return self._config.provider.value

    @property
    def model_name(self) -> str:
        """Return the configured embedding model name."""
        return self._config.model_name

    def _input_path(self, profile: str) -> Path:
        """Return the expected input parquet path for ``profile``."""
        safe_profile = profile.replace("/", "_")
        return self._config.input_dir / f"{safe_profile}.parquet"

    def _output_path(self, profile: str) -> Path:
        """Return the destination parquet path for ``profile``."""
        safe_profile = profile.replace("/", "_")
        return self._config.output_dir / f"{safe_profile}.parquet"

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed all ``texts`` in one call, delegating batching to the provider."""
        handler = resolve_provider(self._config.provider, self._config.model_name)
        return handler.embed(texts, batch_size=self._config.batch_size)

    def embed_to_parquet(self, profile: str) -> Dict[str, str]:
        """Embed chunks from input parquet and write results with an ``embedding`` column.

        Parameters:
            profile: Name of the chunking profile whose parquet to embed.

        Returns:
            ``{profile: output_path}`` when at least one row was embedded,
            otherwise an empty dict.

        Raises:
            FileNotFoundError: If the input parquet does not exist.
        """
        input_path = self._input_path(profile)
        if not input_path.is_file():
            raise FileNotFoundError(
                f"Chunked parquet not found for profile '{profile}': {input_path}"
            )
        input_df = pd.read_parquet(input_path)
        if input_df.empty:
            return {}
        texts = input_df[self._config.text_column].astype(str).tolist()
        embeddings = self._embed_texts(texts)
        input_df = input_df.copy()
        input_df["embedding"] = embeddings
        output_path = self._output_path(profile)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        input_df.to_parquet(output_path, index=False)
        return {profile: str(output_path)}
