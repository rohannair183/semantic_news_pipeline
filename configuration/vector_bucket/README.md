# Vector Bucket Configuration

This folder controls how embedded parquet rows are uploaded to Supabase vector storage.

## Files in this folder

- [`sync.yaml`](sync.yaml): vector upload and index provisioning settings

## `sync.yaml`

Top-level structure:

```yaml
vector_sync:
  input_dir: ...
  bucket_name: ...
  index_name: ...
  dimension: ...
  distance_metric: ...
  embedding_column: ...
  key_columns: [...]
  metadata_columns: [...]
  batch_size: ...
  create_bucket_if_missing: true
  create_index_if_missing: true
```

Supported keys:

- `input_dir`: source directory for embedding parquet files; defaults to `checkpoints/embeddings`
- `bucket_name`: required Supabase vector bucket name
- `index_name`: required vector index name inside the bucket
- `dimension`: required positive integer embedding dimension
- `distance_metric`: required vector distance metric
- `embedding_column`: embedding column name; defaults to `embedding`
- `key_columns`: non-empty list when provided; defaults to `source_api_id`, `chunk_index`, `source_row_index`
- `metadata_columns`: optional list of metadata fields copied into vector objects
- `batch_size`: positive integer up to `500`
- `create_bucket_if_missing`: boolean, defaults to `true`
- `create_index_if_missing`: boolean, defaults to `true`

Supported `distance_metric` values:

- `cosine`
- `euclidean`
- `l2`

## Operational notes

- `dimension` must match the output size of the embedding model
- `batch_size` cannot exceed `500` because of the upload API limit
- the sync stage expects `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`
- bucket/index creation is idempotent; duplicate-resource responses are tolerated on re-runs
