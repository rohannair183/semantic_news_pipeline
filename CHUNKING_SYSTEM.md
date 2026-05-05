# Chunking System

## Pipeline position

```
ArticleNormalizer -> PreChunkPreprocessor -> Chunker -> Embedder
```

The chunker sits between pre-chunk preprocessing and embedding.
`PreChunkPreprocessor` writes cleaned day parquet files to
`checkpoints/pre_chunk/{YYYY-MM-DD}.parquet`. `Chunker` reads every
available day file, splits article text into chunks according to a named
profile, and writes one combined parquet per profile at
`checkpoints/chunked_parquet/{profile}.parquet`. The `Embedder` reads
that output downstream.

## Quick start

```python
from src.chunking.chunker import Chunker

chunker = Chunker()
chunker.chunk_to_parquet(profile="default")
```

Or via the smoke script:

```bash
python smoke_chunking.py
```

## Architecture

```
Chunker (chunker.py)
  |
  |-- reads YAML config via Settings.load_chunking_config
  |-- iterates YYYY-MM-DD.parquet day files
  |-- resolves the profile's strategy
  |-- dispatches to a strategy handler via resolve_handler(strategy)
  |-- collects (text, start, end) spans back
  |-- builds chunk rows with lineage metadata
  |-- writes combined output parquet
  |
  +-- strategies.py (registry)
        |
        +-- SemanticChunker (semantic_chunker.py)
        |     parses params dict -> SemanticChunkingParams
        |     delegates to semantic_sentence_chunks (semantic_split.py)
        |
        +-- (future strategies registered here)
```

The separation of concerns:

- **Chunker** owns orchestration: config loading, parquet I/O, row
  iteration, profile resolution, and output writing. It is
  strategy-agnostic.
- **Strategy handlers** own chunking logic. Each handler receives
  `(full_text, params)` where `params` is a raw `dict[str, Any]` from
  YAML. The handler parses, validates, and applies its own algorithm.
- **strategies.py** maps `ChunkingStrategy` enum values to handler
  instances. `Chunker` calls `resolve_handler(strategy)` and never
  imports a specific handler directly.

## Configuration

Source: `configuration/chunking/chunking.yaml`, loaded via
`Settings.load_chunking_config(...)`.

```yaml
chunking:
  input_dir: checkpoints/pre_chunk
  output_dir: checkpoints/chunked_parquet
  text_columns:
    - body_text
  id_columns:
    - api_id
  profile_columns:
    - profile
  passthrough_columns:
    - headline
  profiles:
    default:
      strategy: semantic_sentence
      params:
        min_chars: 200
        max_chars: 2000
        overlap_chars: 100
        similarity_threshold: 0.35
        sentence_splitter: simple_regex
```

### Field reference

| Field | Purpose |
|---|---|
| `input_dir` | Directory of day parquet files to chunk. |
| `output_dir` | Destination for combined per-profile output parquets. |
| `text_columns` | Ordered fallback list; first non-empty column becomes the chunk source text. |
| `id_columns` | Ordered fallback list for source article id. |
| `profile_columns` | Ordered fallback list for source ingestion profile name. |
| `passthrough_columns` | Extra columns copied verbatim onto each chunk row. |
| `profiles.<name>.strategy` | Selects which registered handler to use (e.g. `semantic_sentence`). |
| `profiles.<name>.params` | Strategy-specific parameters passed as a raw dict to the handler. |

## Output schema

Each row in the output parquet represents one chunk:

| Column | Description |
|---|---|
| `source_day` | ISO date of the input day file. |
| `source_row_index` | Row position in the input day parquet. |
| `source_api_id` | Article id from the id_columns fallback. |
| `source_profile` | Ingestion profile from the profile_columns fallback. |
| `source_text_column` | Which text column was used. |
| `chunk_index` | Zero-based index of this chunk within the article. |
| `chunk_text` | The chunk content. |
| `chunk_start_char` | Start offset in the original text (inclusive). |
| `chunk_end_char` | End offset in the original text (exclusive). |
| `chunk_char_len` | Character length of the chunk. |
| `chunking_strategy` | Strategy value (e.g. `semantic_sentence`). |
| `chunking_version` | Schema version, bumped on breaking changes. |
| `chunking_params_hash` | SHA-256 fingerprint of strategy + params for lineage. |
| *(passthrough)* | Any columns listed in `passthrough_columns`. |

## Idempotency

The chunker is stateless across runs. Every call to `chunk_to_parquet`
rebuilds the output from scratch. Idempotency for downstream writes
(embeddings, database) is handled at the database layer:

- Use `(source_api_id, chunk_index, chunking_strategy,
  chunking_params_hash)` as a deterministic primary key.
- `UPSERT` on that key rather than `INSERT`.
- When params change, `chunking_params_hash` changes, producing new
  keys automatically.

## Semantic sentence strategy

The `semantic_sentence` strategy (`SemanticChunker` in
`semantic_chunker.py`) works as follows:

1. Split text into sentences using a regex splitter.
2. Word-wrap any sentence that exceeds `max_chars` at word boundaries.
3. Merge adjacent sentences into chunks using Jaccard word similarity
   and the `min_chars`/`max_chars` bounds.
4. Apply backward overlap of `overlap_chars` characters between
   consecutive chunks for retrieval context.

Params:

| Param | Type | Description |
|---|---|---|
| `min_chars` | int (>= 1) | Minimum chunk size before considering a boundary. |
| `max_chars` | int (>= min_chars) | Maximum chunk size; triggers a flush. |
| `overlap_chars` | int (>= 0, < max_chars) | Characters of backward overlap between chunks. |
| `similarity_threshold` | float [0, 1] | Jaccard threshold; below this, a new chunk starts. |
| `sentence_splitter` | string | Splitter mode. Currently only `simple_regex`. |

## Adding a new chunking strategy

1. Add an enum value in `src/enums/chunking_strategy.py`.
2. Create a handler class in a new module under `src/chunking/`. It
   must implement `chunk(full_text: str, params: dict) -> list[tuple[str, int, int]]`.
3. Register it in `STRATEGY_HANDLERS` in `strategies.py`.
4. Add a YAML profile with `strategy: <new_value>` and `params: {...}`.
5. Add unit tests for the handler and integration coverage for the
   profile.

No changes to `Chunker` are needed. It dispatches via the registry.

## Testing

Tests live in:

- `tests/unit/chunking/` -- unit tests for chunker, strategies,
  handlers, split logic, and chunk records.
- `tests/unit/config/test_settings_chunking_config.py` -- YAML config
  parsing.
- `tests/integration/ingestion/test_ingestion_pipeline.py` -- end-to-end
  pipeline including chunking.

Smoke script: `python smoke_chunking.py`.
