# Process Configuration

This folder contains downstream generation and persistence settings for briefings.

## Files in this folder

- [`briefing_generator.yaml`](briefing_generator.yaml): topic retrieval and Gemini generation settings
- [`briefing_persistence.yaml`](briefing_persistence.yaml): Postgres persistence settings for generated briefings

## `briefing_generator.yaml`

Top-level structure:

```yaml
briefing_generator:
  model: ...
  vector_top_k: ...
  topics:
    - name: ...
      vector_query: ...
      date_filter: ...
```

Supported keys:

- `model`: required Gemini model identifier
- `vector_top_k`: positive integer; defaults to `10`
- `topics`: required non-empty list of topic specs

Each topic supports:

- `name`: required display name
- `vector_query`: required semantic retrieval query string
- `date_filter`: optional retrieval window selector; defaults to `daily`

Supported `date_filter` values:

- `daily`
- `weekly`
- `monthly`

Environment requirements:

- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- Supabase credentials indirectly through the vector search service if retrieval is enabled

## `briefing_persistence.yaml`

Top-level structure:

```yaml
briefing_persistence:
  table_name: ...
  schema_name: public
  ensure_table: true
```

Supported keys:

- `table_name`: required Postgres table name
- `schema_name`: schema name; defaults to `public`
- `ensure_table`: boolean; defaults to `true`

Notes:

- `table_name` and `schema_name` must be valid Postgres identifiers
- when `ensure_table: true`, the runner creates the table if needed and asks PostgREST to reload its schema cache
- direct DDL may use `SUPABASE_POSTGRES_URL` or `DATABASE_URL` from `.env` when the derived `db.<project>.supabase.co` host is unavailable
