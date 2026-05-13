"""Embed chunked parquet into vector embeddings and write to checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config.settings import Settings
from src.embeddings.providers import resolve_provider
from src.utils.timer import Timer

_CHUNK_ID_COLUMNS = ("source_api_id", "chunk_index", "source_row_index")


class Embedder:
    """Run YAML-configured embedding on chunked parquet files."""

    def __init__(
        self,
        configuration_root: Path | None = None,
        timer: Optional[Timer] = None,
    ):
        """Load embedding settings from YAML configuration."""
        self._timer = timer or Timer()
        self._config = Settings.load_embedding_config(
            configuration_root=configuration_root,
        )

    @property
    def timer(self) -> Timer:
        """Return the timer instance used by this embedder."""
        return self._timer

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
        handler = resolve_provider(
            self._config.provider,
            self._config.model_name,
            timer=self._timer,
        )
        return handler.embed(texts, batch_size=self._config.batch_size)

    @staticmethod
    def _split_new_and_cached(
        input_df: pd.DataFrame,
        output_path: Path,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Separate *input_df* into rows needing embedding and already-cached rows.

        Returns ``(new_df, cached_df)`` where *cached_df* already carries the
        ``embedding`` column from a previous run.
        """
        id_cols = [c for c in _CHUNK_ID_COLUMNS if c in input_df.columns]
        if not id_cols or not output_path.is_file():
            return input_df, pd.DataFrame()

        existing_df = pd.read_parquet(output_path)
        if existing_df.empty or "embedding" not in existing_df.columns:
            return input_df, pd.DataFrame()

        merged = input_df.merge(
            existing_df[id_cols + ["embedding"]],
            on=id_cols,
            how="left",
            indicator=True,
        )
        cached_mask = merged["_merge"] == "both"
        cached_df = merged.loc[cached_mask].drop(columns=["_merge"])
        new_df = merged.loc[~cached_mask].drop(columns=["_merge", "embedding"])
        return new_df, cached_df

    def embed_to_parquet(self, profile: str) -> Dict[str, str]:
        """Embed only new chunks and merge with previously cached embeddings.

        Parameters:
            profile: Name of the chunking profile whose parquet to embed.

        Returns:
            ``{profile: output_path}`` when the output contains at least one
            row, otherwise an empty dict.

        Raises:
            FileNotFoundError: If the input parquet does not exist.
        """
        input_path = self._input_path(profile)
        if not input_path.is_file():
            raise FileNotFoundError(
                f"Chunked parquet not found for profile '{profile}': {input_path}"
            )
        with self._timer.section("embedder.read_parquet"):
            input_df = pd.read_parquet(input_path)
        if input_df.empty:
            return {}

        output_path = self._output_path(profile)
        new_df, cached_df = self._split_new_and_cached(input_df, output_path)

        if new_df.empty:
            result_df = cached_df
        else:
            texts = new_df[self._config.text_column].astype(str).tolist()
            with self._timer.section("embedder.embed_texts"):
                embeddings = self._embed_texts(texts)
            new_df = new_df.copy()
            new_df["embedding"] = embeddings
            result_df = pd.concat([cached_df, new_df], ignore_index=True)

        with self._timer.section("embedder.write_parquet"):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_parquet(output_path, index=False)
        return {profile: str(output_path)}
