# Chunking System Overview

This document explains how the chunking system works, what it outputs, and how to extend it for additional chunking strategies and evaluation workflows.

## Current Pipeline Position

The chunking stage runs after pre-chunk preprocessing:

1. `ArticleNormalizer` writes day parquet files.
2. `PreChunkPreprocessor` writes cleaned day parquet files to `checkpoints/pre_chunk`.
3. `Chunker` reads those files and writes one combined chunk parquet
   per chunking profile to `checkpoints/chunked_parquet/<profile>.parquet`.

The day-file convention for the input is `YYYY-MM-DD.parquet`; the chunker
always rebuilds the combined output from every available input day.

## Public API

```python
from src.chunking.chunker import Chunker

chunker = Chunker()
chunker.chunk_to_parquet(profile="default")
```

- `profile` is required and must match a key under `chunking.profiles` in YAML.
- Each call is a full rebuild: it reads every `*.parquet` in `chunking.input_dir`
  and writes a fresh combined parquet to
  `{chunking.output_dir}/{profile}.parquet`.
- Unknown profile or unregistered strategy raises `ValueError`.
- The chunker holds **no cross-run state**; nothing on disk other than the
  output parquet itself records what has been processed before.

## Idempotency Contract

The chunker is intentionally stateless across runs. Idempotency for downstream
embedding and database writes lives at the **database layer**, not here. The
recommended pattern is:

- Treat `(source_api_id, chunk_index, chunking_strategy, chunking_params_hash)`
  (all already on each chunk row) as the deterministic identity of a chunk.
- Derive a stable primary key from those fields and `UPSERT` rather than
  `INSERT` when pushing chunks/embeddings into the database.
- If `chunking_params_hash` changes (because chunking config changed), the new
  rows naturally produce new primary keys; old rows can be cleaned up with a
  delete-by-hash query.

Re-running the chunker is therefore always safe: it simply rewrites the
combined parquet, and the database layer dedupes based on chunk identity.

## Configuration

Chunking configuration is loaded from `configuration/chunking/chunking.yaml`
via `Settings.load_chunking_config(...)`.

### Main YAML fields

- `input_dir`: source parquet directory (typically `checkpoints/pre_chunk`).
- `output_dir`: destination directory for combined per-profile parquets
  (defaults to `checkpoints/chunked_parquet`).
- `text_columns`: ordered fallback list used to pick the source text column.
- `id_columns`: ordered fallback list for source article id.
- `profile_columns`: ordered fallback list for source profile.
- `passthrough_columns`: extra source columns copied to each chunk row.
- `profiles`: mapping of `<profile_name> -> { strategy, params }`. Each
  profile selects a registered strategy and supplies its parameters as a
  generic dict that the strategy handler interprets.

## Architecture

```
Chunker (chunker.py)               -- orchestration: config, file I/O, dispatch
  -> resolve_handler(strategy)      -- strategy registry (strategies.py)
    -> SemanticChunker              -- semantic_sentence handler (semantic_chunker.py)
    -> (future) FixedSizeChunker    -- additional strategies
```

`Chunker` is the main entry point. Each profile's `strategy` field selects a
registered handler from `strategies.py`, and its `params` dict is passed
through opaquely to the handler. The handler owns parameter parsing and
validation.

## Implementation Details

### 1) Orchestration (`src/chunking/chunker.py`)

`Chunker.chunk_to_parquet(profile)`:

- resolves the typed `ChunkingProfileConfig` for `profile`
- iterates `*.parquet` under `input_dir`, skipping files without ISO-day stems
- per row, resolves text/id/profile through ordered fallback columns and
  dispatches to the registered strategy handler
- accumulates chunk records across every input day and writes a single
  combined parquet at `{output_dir}/{profile}.parquet`
- returns `{profile_name: combined_path}` when at least one chunk was
  produced, or `{}` when the inputs yielded no chunks

### 2) Strategy registry (`src/chunking/strategies.py`)

`STRATEGY_HANDLERS: dict[ChunkingStrategy, ChunkingStrategyHandler]` maps each
strategy enum to a handler that implements `chunk(full_text, params)`.
`resolve_handler(strategy)` raises `ValueError` for unregistered strategies.

### 3) Strategy handlers (e.g. `src/chunking/semantic_chunker.py`)

Each handler implements the `ChunkingStrategyHandler` protocol:
`chunk(full_text: str, params: dict[str, Any]) -> list[tuple[str, int, int]]`.

The handler owns parsing its `params` dict into typed internal parameters.
For example, `SemanticChunker` parses the dict into `SemanticChunkingParams`
and delegates to `semantic_sentence_chunks`.

### 4) Output row construction (`src/chunking/chunk_records.py`)

Each chunk row includes lineage + metadata:

- source: `source_day`, `source_row_index`, `source_api_id`, `source_profile`
- chunk identity: `chunk_index`, `source_text_column`
- content + offsets: `chunk_text`, `chunk_start_char`, `chunk_end_char`,
  `chunk_char_len`
- run metadata: `chunking_strategy`, `chunking_version`, `chunking_params_hash`
- optional passthrough fields from the source row

The `chunking_params_hash` is a short SHA-256 fingerprint of strategy + params,
stable across runs with identical config. It is the natural component to bake
into the database primary key for idempotent upserts.

## Extension Paths

### A) Add a new chunking strategy

1. Add an enum value in `src/enums/chunking_strategy.py`.
2. Implement a handler class with a `chunk(full_text, params)` method (in a
   new module under `src/chunking/`).
3. Register the handler in `STRATEGY_HANDLERS` in `strategies.py`.
4. Add a new YAML profile with `strategy: <new_value>` and `params: {...}`.
5. Add or update tests for the new handler and any new params.

No changes are required in `Chunker`; it dispatches via the registry.

Examples:

- fixed-token chunking for model-context control
- heading/section-aware chunking for structured documents
- embedding-distance boundary chunking for stronger semantic boundaries

### B) Add better sentence splitting

1. Add an enum value in `src/enums/sentence_splitter_mode.py`.
2. Implement the splitter helper in `semantic_split.py`.
3. Extend param validation in `SemanticChunker._parse_params(...)`.
4. Add tests for edge punctuation, abbreviations, and multiline text.

### C) Add chunk evaluation modules

Recommended approach is a separate evaluation stage that reads chunk parquet
and writes metrics artifacts.

Potential metrics:

- size distribution: chunk char lengths, outlier rates
- boundary quality: sentence break precision/recall against references
- redundancy: overlap duplication rates
- retrieval quality: top-k relevance lift vs baseline chunking

Suggested outputs:

- `checkpoints/chunk_eval/<profile>.parquet` for row-level metrics
- `checkpoints/chunk_eval/summary_<profile>.json` for aggregate metrics

### D) Add experiment/version support

- bump `chunking_version` on schema/logic changes
- add a new chunking profile with the experimental params; outputs land in a
  separate `<profile>.parquet`, so multiple variants coexist without conflict
- the database layer's primary key (which includes `chunking_params_hash`)
  keeps experimental and baseline rows isolated automatically

### E) Add quality/safety guards

- min non-empty chunk text checks
- maximum chunk count per article guardrails
- fallback behavior for very short or malformed input
- optional strict mode that errors on missing `text_columns`

## Testing and Validation Expectations

For any chunking extension, update:

- unit tests under `tests/unit/chunking/`
- config parser tests under `tests/unit/config/`
- integration flow under `tests/integration/ingestion/`
- smoke run via `smoke_chunking.py`

And run the repository validation commands from AGENTS.md.
