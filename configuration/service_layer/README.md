# Service Layer Configuration

This folder configures runtime semantic retrieval behavior.

## Files in this folder

- [`vector_search.yaml`](vector_search.yaml): semantic search bucket/index target

## `vector_search.yaml`

Top-level structure:

```yaml
vector_search:
  bucket_name: ...
  index_name: ...
  date_metadata_key: ...
```

Supported keys:

- `bucket_name`: required vector bucket name to query
- `index_name`: required vector index name to query
- `date_metadata_key`: optional metadata field used for inclusive date filtering; defaults to `source_day`

## Notes

- semantic query embedding still uses [`../embeddings/embeddings.yaml`](../embeddings/embeddings.yaml) for provider, model, and batch settings
- the service layer expects `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`
- when `date_from` and `date_to` are used, the search layer sends a finite `$in` list of ISO day strings against `date_metadata_key`
