# Semantic News Pipeline

Builds a daily semantic news dataset from Guardian articles and turns it into searchable vectors plus AI-generated briefings.

| Language | Ingestion | Data Format | Vector Layer | Storage | LLM | Orchestration | CI/CD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Python | Guardian Open Platform API | pandas + parquet | sentence-transformers provider interface + semantic search | Supabase + PostgreSQL | Gemini via `google-genai` | YAML-driven pipeline orchestration | GitHub Actions |

## Problem

Reading daily news across multiple topics is noisy and time-consuming. Keyword search misses related stories that use different language, and manual summarization does not scale. This project solves that by running a batch pipeline that:

- ingests fresh news articles from the Guardian API
- cleans and chunks long-form text into retrieval-ready records
- generates embeddings for semantic search
- syncs vectors and metadata to Supabase
- retrieves relevant context and generates topic briefings with Gemini

The result is a system for turning raw news into structured, reusable semantic data products.

## System Overview

This repository contains a working Python pipeline with YAML-driven orchestration and typed configuration.

**Pipeline stages**

1. **Ingestion**: pulls Guardian articles by named topic profiles with pagination, incremental ID tracking, checkpoints, and usage logging.
2. **Normalization / pre-chunk prep**: cleans article fields and writes parquet outputs for downstream stages.
3. **Chunking**: converts article text into chunk-level records for better embedding and retrieval quality.
4. **Embeddings**: generates vectors from chunk parquet files with a pluggable provider interface.
5. **Vector sync**: uploads vectors plus metadata into Supabase vector storage.
6. **Semantic retrieval**: embeds user queries and runs filtered similarity search over stored vectors.
7. **Briefing generation**: retrieves topic context and produces multi-topic briefings with Gemini.
8. **Briefing persistence**: stores generated briefings in Supabase/Postgres for later consumption.

## System Components

```text
Guardian API
  -> ingestion profiles
  -> normalized parquet
  -> chunked parquet
  -> embeddings parquet
  -> Supabase vector bucket
  -> semantic retrieval
  -> Gemini briefing generation
  -> persisted briefing rows
```

The pipeline is orchestrated declaratively from YAML rather than hard-coded job wiring:

- CLI entry: [`src/application/__main__.py`](src/application/__main__.py)
- orchestrator: [`src/application/orchestrator.py`](src/application/orchestrator.py)
- task dispatch: [`src/application/task_runners.py`](src/application/task_runners.py)
- orchestration config: [`configuration/orchestration/orchestrator.yaml`](configuration/orchestration/orchestrator.yaml)
- configuration guide: [`configuration/README.md`](configuration/README.md)

The same YAML-driven pipeline can also run in GitHub Actions through [`.github/workflows/pipeline.yml`](.github/workflows/pipeline.yml).

## Configuration Layout

All YAML configuration lives under [`configuration/`](configuration). Start with [`configuration/README.md`](configuration/README.md) for the full map, then drill into the folder-level docs:

- [`configuration/ingestion/README.md`](configuration/ingestion/README.md)
- [`configuration/chunking/README.md`](configuration/chunking/README.md)
- [`configuration/embeddings/README.md`](configuration/embeddings/README.md)
- [`configuration/vector_bucket/README.md`](configuration/vector_bucket/README.md)
- [`configuration/service_layer/README.md`](configuration/service_layer/README.md)
- [`configuration/process/README.md`](configuration/process/README.md)
- [`configuration/orchestration/README.md`](configuration/orchestration/README.md)

## Notable Technical Design Choices

- **Incremental ingestion**: avoids reprocessing existing article IDs and writes run checkpoints.
- **Config-driven design**: ingestion, chunking, embeddings, vector sync, and briefing generation are all controlled through YAML.
- **Typed boundaries**: enums and `Settings` loaders validate constrained config values at the boundary.
- **Batch-friendly data flow**: parquet files are used between stages for reproducible intermediate outputs.
- **Retrieval + generation integration**: semantic search is used as the grounding layer for generated briefings.
- **Operational hooks**: timers, retry logic, smoke scripts, and CI workflows make the project closer to production than a notebook prototype.

## Production-Oriented Features

This repo is meant to look like an engineering project, not just an experiment.

- **Unit and integration tests** live under [`tests/unit`](tests/unit) and [`tests/integration`](tests/integration).
- **100% source coverage is the repository validation target** and coverage is measured against `src` only.
- **Linting is enforced** with `pylint` and `ruff`.
- **Integration coverage is included**, not just unit tests.
- **CI runs automatically for pushes to `main` and pull requests targeting `main`** in [`.github/workflows/main.yml`](.github/workflows/main.yml).
- **The YAML-driven pipeline can run in GitHub Actions** via [`.github/workflows/pipeline.yml`](.github/workflows/pipeline.yml).

## Recreate it locally

Install dependencies:

```bash
pip install -e ".[dev,runtime]"
```

Create a local `.env` file in the repository root and add the secrets required for the stages you want to run:

```env
GUARDIAN_API_KEY=your_guardian_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
GEMINI_API_KEY=your_gemini_key
```

If you do not already have these secrets:

- `GUARDIAN_API_KEY`: create a key from the Guardian Open Platform.
- `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`: copy them from your Supabase project settings.
- `GEMINI_API_KEY`: create a Gemini or Google AI Studio API key.

Run the orchestrator locally:

```bash
python -m src.application --config configuration/orchestration/orchestrator.yaml
```

For a lighter local check, you can also run the CI-oriented preset:

```bash
python -m src.application --mode test
```

## Recreate it in GitHub Actions

To run the pipeline from GitHub Actions, add the same secrets in your GitHub repository:

1. Go to `Settings -> Secrets and variables -> Actions`.
2. Add `GUARDIAN_API_KEY` for the orchestrator workflow.
3. Add any additional secrets your enabled stages require, such as `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, and `GEMINI_API_KEY`.

The repo uses two workflows:

- [`.github/workflows/main.yml`](.github/workflows/main.yml) for linting, unit tests, integration tests, and coverage checks on changes to `main`.
- [`.github/workflows/pipeline.yml`](.github/workflows/pipeline.yml) to run the YAML-driven pipeline in GitHub Actions.

## Quality checks

From the repository root:

```bash
python -m unittest discover -s tests/unit -p "test_*.py"
coverage run --source=src -m unittest discover -s tests/unit -p "test_*.py"
coverage report --fail-under=100
python -m unittest discover -s tests/integration -p "test_*.py"
pylint src tests
ruff check .
```

This repository’s engineering guardrails require 100% coverage for the changed scope, clean linting, passing unit tests, and passing integration tests before a change is considered done.

## Best entry points

If you only have a few minutes, these files give the best picture of the project:

- [`src/application/orchestrator.py`](src/application/orchestrator.py): YAML-driven pipeline execution
- [`src/ingestion/article_ingestor.py`](src/ingestion/article_ingestor.py): incremental Guardian ingestion
- [`src/chunking/chunker.py`](src/chunking/chunker.py): parquet-based chunk generation
- [`src/embeddings/embedder.py`](src/embeddings/embedder.py): embedding pipeline
- [`src/vector_sync/bucket_sync.py`](src/vector_sync/bucket_sync.py): vector upload to Supabase
- [`src/service_layer/vector_search_service.py`](src/service_layer/vector_search_service.py): semantic retrieval
- [`src/process/briefing_generator.py`](src/process/briefing_generator.py): retrieval-grounded Gemini briefings
- [`src/process/briefing_persistence.py`](src/process/briefing_persistence.py): persistence layer

## What’s in scope today

This repository is focused on the backend pipeline and data flow. It does not currently include a polished end-user application UI. The strongest parts of the project today are:

- pipeline orchestration
- configurable ingestion and transformation
- vector search infrastructure
- LLM-backed briefing generation
- testing and CI discipline
