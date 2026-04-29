# Ingestion Configuration Layout

Ingestion settings are split across focused YAML files under this folder.

## Files

- `base.yaml`
  - Global Guardian client defaults.
  - Contains keys like `base_url`, `default_page_size`, `max_page_size`, and
    `timeout_seconds`.

- `profiles.yaml`
  - Named profile definitions used by ingestion runs.
  - Contains the top-level `profiles` mapping (for example `main_daily`,
    `main_weekly`) and each profile's query/timeframe fields.

- `article_ingestor.yaml`
  - Orchestration settings for `ArticleIngestor`.
  - Contains `article_ingestor` settings such as `profiles_to_run`,
    `limit_per_profile`, checkpoint/logging options, and directories.

- `article_normalizer.yaml`
  - Row mapping rules for `ArticleNormalizer`.
  - Contains `article_normalizer.row_mappings`.

- `pre_chunk_preprocessor.yaml`
  - Transformation rules for `PreChunkPreprocessor`.
  - Contains `pre_chunk_preprocessor.output_dir` and
    `pre_chunk_preprocessor.operations`.

- `ingestion_config.yaml`
  - Optional local override file.
  - Keep this file small; define only keys you want to override.

## Merge Order

Settings are loaded and deep-merged in this order:

1. `base.yaml`
2. `profiles.yaml`
3. `article_ingestor.yaml`
4. `article_normalizer.yaml`
5. `pre_chunk_preprocessor.yaml`
6. `ingestion_config.yaml` (last, highest precedence)

Later files override earlier files for overlapping keys.

## Editing Guidance

- Add new profile entries in `profiles.yaml`.
- Add ingestion-run behavior in `article_ingestor.yaml`.
- Add normalization field mapping changes in `article_normalizer.yaml`.
- Add pre-chunk operations in `pre_chunk_preprocessor.yaml`.
- Use `ingestion_config.yaml` only for temporary or environment-local overrides.
