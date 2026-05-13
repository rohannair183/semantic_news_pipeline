# Ingestion Configuration

This folder holds the YAML that drives Guardian ingestion, normalization, and pre-chunk preprocessing.

## How these files are loaded

`Settings.load_ingestion_config()` deep-merges these files in order:

1. `base.yaml`
2. `profiles.yaml`
3. `article_ingestor.yaml`
4. `article_normalizer.yaml`
5. `pre_chunk_preprocessor.yaml`
6. `ingestion_config.yaml` if you add it locally as an override

Later files override earlier files on key collisions.

## Files in this folder

- [`base.yaml`](base.yaml): shared Guardian client defaults
- [`profiles.yaml`](profiles.yaml): named query profiles
- [`article_ingestor.yaml`](article_ingestor.yaml): run controls for `ArticleIngestor`
- [`article_normalizer.yaml`](article_normalizer.yaml): row extraction and transform rules
- [`pre_chunk_preprocessor.yaml`](pre_chunk_preprocessor.yaml): declarative parquet cleanup operations

## `base.yaml`

Top-level keys:

- `base_url`: Guardian API base URL
- `default_page_size`: default search page size
- `max_page_size`: upper bound for any profile-level `page_size`
- `timeout_seconds`: request timeout for Guardian API calls

## `profiles.yaml`

Top-level structure:

```yaml
profiles:
  profile_name:
    ...
```

Each profile supports:

- `topic`: string, optional
- `run_date`: single ISO day, optional
- `from_date`: ISO day, optional
- `to_date`: ISO day, optional
- `timeframe`: mapping, optional
- `page_size`: integer between `1` and `max_page_size`
- `query`: non-empty string, optional
- `section`: string, optional, may be empty
- `order_by`: `newest`, `oldest`, or `relevance`
- `use_next_fallback`: boolean, defaults to `true`
- `content_show_fields`: Guardian `show-fields` string, defaults to `all`

Date window rules:

- Use exactly one of `timeframe`, `run_date`, or `from_date` + `to_date`.
- `run_date` means one-day ingestion.
- `from_date` and `to_date` must be provided together.

`timeframe` supports:

- `mode`: `relative` or `explicit`
- `relative`: `past_day`, `past_week`, or `past_month`
- `from_date`: required when `mode: explicit`
- `to_date`: required when `mode: explicit`

## `article_ingestor.yaml`

Top-level structure:

```yaml
article_ingestor:
  ...
```

Supported keys:

- `profiles_to_run`: ordered list of profile names from `profiles.yaml`; defaults to all profiles
- `limit_per_profile`: optional positive integer cap on newly fetched articles
- `save_local_checkpoint`: boolean; when `true`, writes per-profile JSON checkpoints
- `checkpoint_dir`: output directory for ingested JSON checkpoints; defaults to `checkpoints/article_ingestor`
- `enable_usage_logging`: boolean; when `true`, writes API usage logs
- `logs_dir`: output directory for usage logs; defaults to `logs`
- `parquet_dir`: optional parquet output directory consumed by the normalizer; defaults to `checkpoints/parquet`

## `article_normalizer.yaml`

Top-level structure:

```yaml
article_normalizer:
  row_mappings:
    output_field:
      sources:
        - ...
      transform: ...
```

Each `row_mappings.<output_field>` entry supports:

- `sources`: required non-empty list of lookup selectors
- `transform`: optional transform name

Supported source selectors:

- `profile`: injects the active ingestion profile name
- `payload.some.path`: reads from the full API payload mapping
- `fields.some.path`: reads from `payload["fields"]`
- `item.some.path`: reads from nested `item` paths
- `id` or any other bare key: reads a direct top-level key from the record

Supported transforms:

- `parse_iso`: parses ISO date/time strings into normalized datetime values

Related path behavior:

- input checkpoints come from `article_ingestor.checkpoint_dir`
- normalized parquet output goes to `article_ingestor.parquet_dir`

## `pre_chunk_preprocessor.yaml`

Top-level structure:

```yaml
pre_chunk_preprocessor:
  output_dir: ...
  operations:
    - name: ...
      args: ...
```

Supported keys:

- `output_dir`: destination directory for pre-chunk parquet files; defaults to `checkpoints/pre_chunk`
- `operations`: required ordered list of preprocessing steps

Supported operations and args:

- `drop_columns`
  - `args.columns`: non-empty list of column names to remove
- `rename_columns`
  - `args.mapping`: non-empty old-name to new-name mapping
- `trim_whitespace_columns`
  - `args.columns`: non-empty list of text columns to strip
- `drop_empty_rows`
  - `args.required_columns`: non-empty list; rows missing these values are dropped
- `filter_min_numeric`
  - `args.column`: numeric column name
  - `args.min_value`: numeric lower bound
- `coalesce_columns`
  - `args.target`: destination column
  - `args.sources`: non-empty ordered list of fallback columns
- `normalize_text_columns`
  - `args.columns`: non-empty list of text columns to normalize

## Local override file

You can add `ingestion_config.yaml` in this folder for local overrides. Keep it small and only define keys you want to replace after the standard merge order above.
