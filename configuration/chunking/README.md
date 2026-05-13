# Chunking Configuration

This folder defines how normalized article text is converted into chunk-level parquet rows for embedding.

## Files in this folder

- [`chunking.yaml`](chunking.yaml): input/output locations, source column selectors, and named chunking profiles

## `chunking.yaml`

Top-level structure:

```yaml
chunking:
  input_dir: ...
  output_dir: ...
  text_columns: [...]
  id_columns: [...]
  profile_columns: [...]
  passthrough_columns: [...]
  profiles:
    profile_name:
      strategy: ...
      params:
        ...
```

Supported top-level keys:

- `input_dir`: directory of pre-chunk parquet files; defaults to `checkpoints/pre_chunk`
- `output_dir`: destination for combined chunk parquet files; defaults to `checkpoints/chunked_parquet`
- `text_columns`: required non-empty ordered list of candidate source text columns
- `id_columns`: optional ordered list of candidate ID columns
- `profile_columns`: optional ordered list of candidate source-profile columns
- `passthrough_columns`: optional list of columns copied into each chunk row
- `profiles`: required non-empty mapping of named chunking profiles

How column fallback works:

- the chunker picks the first non-empty value from `text_columns`
- it does the same for `id_columns` and `profile_columns`
- any listed `passthrough_columns` are copied through when present

## Chunking profiles

Each `chunking.profiles.<name>` entry supports:

- `strategy`: chunking strategy name
- `params`: required mapping passed to the selected strategy

Supported `strategy` values:

- `semantic_sentence`

## `semantic_sentence` params

Required or supported params:

- `min_chars`: positive integer minimum chunk size before splitting on semantic change
- `max_chars`: positive integer maximum chunk size
- `overlap_chars`: integer `>= 0` and `< max_chars`; overlap added to later chunks
- `similarity_threshold`: float in `[0, 1]`; lower values allow looser sentence grouping
- `sentence_splitter`: sentence splitting mode

Supported `sentence_splitter` values:

- `simple_regex`

Validation rules:

- `max_chars` must be `>= min_chars`
- `overlap_chars` must be smaller than `max_chars`

## Output shape

Each run writes one combined parquet file per profile:

- `{output_dir}/{profile}.parquet`

The chunker rebuilds that file from the available input day files rather than appending incrementally.
