# Configuration Overview

This project keeps runtime settings in YAML under `configuration/`. Most stages have their own subfolder so each concern stays isolated and easy to reason about.

## How configuration is organized

- [`ingestion/README.md`](ingestion/README.md): Guardian API defaults, named ingestion profiles, normalization mappings, and pre-chunk preprocessing rules
- [`chunking/README.md`](chunking/README.md): chunking inputs, output paths, and named chunking strategies
- [`embeddings/README.md`](embeddings/README.md): embedding model and batch settings
- [`vector_bucket/README.md`](vector_bucket/README.md): vector upload and index provisioning settings for Supabase
- [`service_layer/README.md`](service_layer/README.md): semantic search target bucket/index settings
- [`process/README.md`](process/README.md): briefing generation and briefing persistence configuration
- [`orchestration/README.md`](orchestration/README.md): YAML task pipelines for local and CI execution

## Folder-by-folder summary

### `ingestion/`

Controls how raw Guardian content is pulled and transformed before chunking.

- shared Guardian client defaults live in `base.yaml`
- named date/query profiles live in `profiles.yaml`
- stage-specific behavior for ingestion, normalization, and preprocessing lives in separate YAML files

### `chunking/`

Controls how normalized article text becomes chunk-level parquet rows.

- selects source columns
- defines passthrough metadata
- names one or more chunking profiles with strategy params

### `embeddings/`

Controls how chunk parquet files are turned into vectors.

- chooses input and output directories
- chooses embedding provider, model, text column, and batch size

### `vector_bucket/`

Controls upload of embedding rows into Supabase vector storage.

- defines bucket and index names
- defines vector dimension and distance metric
- controls key columns, metadata columns, and upload batch size

### `service_layer/`

Controls which vector bucket/index semantic search queries hit.

- keeps runtime retrieval config separate from vector upload config
- also defines the metadata field used for date filtering

### `process/`

Controls downstream LLM-backed outputs.

- `briefing_generator.yaml` defines topic prompts, date windows, and model choice
- `briefing_persistence.yaml` defines where generated briefings are stored in Postgres

### `orchestration/`

Controls whole-pipeline execution order.

- `orchestrator.yaml` is the default local/full pipeline preset
- `orchestrator_ci.yaml` is the CI-safe preset with heavyweight steps disabled

## Important loading behavior

- Ingestion-related settings are merged from multiple files, documented in [`ingestion/README.md`](ingestion/README.md).
- Other folders are usually loaded as a single YAML file for that concern.
- Environment secrets still come from `.env` or GitHub Actions secrets; these YAML files only define non-secret runtime behavior.
