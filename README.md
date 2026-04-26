# Semantic News Pipeline

An end-to-end batch pipeline that ingests Guardian news articles, creates vector embeddings, groups related stories, and serves precomputed daily briefings plus semantic search.

## TL;DR

- **Problem solved:** turn raw daily news into searchable, structured, semantic outputs.
- **Core approach:** API ingestion -> relational storage -> embedding generation -> vector indexing -> clustering -> briefing generation.
- **Architecture choice:** daily batch jobs + precomputed outputs to minimize runtime cost.
- **Primary stack:** Guardian Open Platform API, PostgreSQL, pgvector, embedding model, lightweight web serving layer.
- **Scope control:** fixed topics (technology, business, environment, science, world) to stay on the free-tier resources.

## System overview

The pipeline runs once per day and processes only new content.

1. **Ingestion**
	- Pull recent Guardian articles for predefined topics.
	- Handle pagination.
	- Load incrementally by publication date/article ID.

2. **Raw storage**
	- Persist articles in relational tables with minimal transformation.
	- Keep key metadata: `title`, `section`, `published_at`, `url`, `api_id`.

3. **Content preparation**
	- Light cleanup of title/trail/body text.
	- Chunk long articles for better embedding and retrieval quality.

4. **Embedding generation**
	- Convert each article/chunk into a vector representation.

5. **Vector storage + retrieval**
	- Store embeddings in PostgreSQL with pgvector indexes.
	- Enable fast similarity search.

6. **Story grouping**
	- Cluster semantically similar items to detect shared storylines.

7. **Briefing generation**
	- Precompute concise daily briefings per topic from top clusters.

8. **Serving layer**
	- Expose daily briefings, semantic search, and related-article discovery.

## What gets produced each day

- Freshly ingested article records
- Vector embeddings for searchable text units
- Topic-level story clusters
- Precomputed daily topic briefings

## Evaluation plan

The system is designed to be evaluated on:

- **Retrieval relevance:** quality of top-k semantic matches.
- **Cluster coherence:** whether grouped stories are genuinely related.
- **Briefing usefulness:** coverage of key daily developments per topic.
- **Operational reliability:** successful daily runs with incremental loading.

