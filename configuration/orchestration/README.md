# Orchestration Configuration

This folder defines declarative pipeline execution order for the application CLI and GitHub Actions workflow.

## Files in this folder

- [`orchestrator.yaml`](orchestrator.yaml): default local/full pipeline preset
- [`orchestrator_ci.yaml`](orchestrator_ci.yaml): CI-safe preset with heavyweight or secret-dependent steps disabled

## Top-level schema

```yaml
fail_fast: true
tasks:
  - id: ...
    kind: ...
    enabled: true
    skip_when:
      missing_env_var: ...
    params:
      ...
```

Supported top-level keys:

- `fail_fast`: boolean, defaults to `true`
- `tasks`: required non-empty ordered list of task specs

## Task fields

Each task supports:

- `id`: optional non-empty string; defaults to the task `kind`
- `kind`: required task kind
- `enabled`: boolean, defaults to `true`
- `skip_when`: optional mapping of skip predicates
- `params`: optional mapping whose allowed keys depend on `kind`

Supported `skip_when` keys:

- `missing_env_var`: skip the task when this environment variable is absent or empty

## Supported task kinds

- `article_ingestor`
- `article_normalizer`
- `pre_chunk_preprocessor`
- `chunking`
- `embeddings`
- `vector_sync`
- `briefing_persistence`

## `params` by task kind

### `article_ingestor`

No params are currently allowed.

### `article_normalizer`

Supported params:

- `day`: optional ISO day or `utc_today`

### `pre_chunk_preprocessor`

No params are currently allowed.

### `chunking`

Supported params:

- `profile`: chunking profile name; defaults to `default`

### `embeddings`

Supported params:

- `profile`: embedding input profile name; defaults to `default`

### `vector_sync`

Supported params:

- `profile`: vector sync profile name; defaults to `default`

### `briefing_persistence`

No params are currently allowed.

## Runtime behavior

- tasks execute in YAML order
- disabled tasks are skipped without error
- when `fail_fast` is true, a failed task prevents later tasks from running
- briefing persistence is skipped when the latest persisted run is already on the current UTC day
- local CLI entry point: [`../../src/application/__main__.py`](../../src/application/__main__.py)
