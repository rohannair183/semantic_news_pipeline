# Embeddings Configuration

This folder controls how chunk parquet files are converted into vector embeddings.

## Files in this folder

- [`embeddings.yaml`](embeddings.yaml): embedding pipeline settings

## `embeddings.yaml`

Top-level structure:

```yaml
embeddings:
  input_dir: ...
  output_dir: ...
  text_column: ...
  provider: ...
  model_name: ...
  batch_size: ...
```

Supported keys:

- `input_dir`: source directory for chunk parquet files; defaults to `checkpoints/chunked_parquet`
- `output_dir`: destination directory for embedded parquet files; defaults to `checkpoints/embeddings`
- `text_column`: required source column to embed, usually `chunk_text`
- `provider`: required embedding backend selector
- `model_name`: required model identifier for the selected provider
- `batch_size`: required positive integer batch size

## Supported providers

Values accepted by config validation:

- `sentence_transformers`
- `openai`

Current runtime implementation:

- `sentence_transformers` is implemented in [`src/embeddings/providers.py`](../../src/embeddings/providers.py)
- `openai` is reserved in the enum but does not currently have a provider handler in the registry

## Expected data flow

- input files: `{input_dir}/{profile}.parquet`
- output files: `{output_dir}/{profile}.parquet`

The embedder reuses cached rows already present in the output parquet when the chunk identity columns match, so re-runs only embed new rows.
