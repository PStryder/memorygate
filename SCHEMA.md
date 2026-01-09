# MemoryGate Database Schema

**PostgreSQL + pgvector Architecture**

Version: 0.1.0  
Last Updated: January 2, 2026

---

## Overview

MemoryGate uses PostgreSQL with the pgvector extension for vector similarity search. All tables use SQLAlchemy ORM models defined in `models.py`. SQLite mode stores JSON in place of JSONB and can disable vector search. The schema supports:

- AI instance tracking (multiple AI personalities)
- Session/conversation provenance
- Observations with semantic embeddings
- Knowledge graph (concepts + relationships)
- Pattern synthesis
- Document references
- Cold storage tiers, summaries, and tombstones for retention

**Current Implementation Status:**
- ✅ Fully Implemented: ai_instances, sessions, observations, embeddings, documents, concepts, concept_aliases, concept_relationships, patterns, memory_summaries, memory_tombstones
- 🎉 All core features complete!

**Document Storage:**
- Documents stored as references with summaries (not full content)
- Canonical storage: Google Drive
- Full content fetched on demand via Drive API

---

## Enum Types

- `memory_tier` - hot, cold
- `tombstone_action` - archived, rehydrated, purged, summarized

---

## Core Tables

### ai_instances

Tracks different AI personalities (Kee, Hexy, etc.) that interact with the system.

```sql
CREATE TABLE ai_instances (
    id                  INTEGER PRIMARY KEY,
    name                VARCHAR(100) UNIQUE NOT NULL,
    platform            VARCHAR(100) NOT NULL,
    description         TEXT,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Columns:**
- `id` - Auto-increment primary key
- `name` - AI instance name (e.g., "Kee", "Hexy") - UNIQUE
- `platform` - Platform name (e.g., "Claude", "ChatGPT", "GPT-4")
- `description` - Optional description of AI instance
- `created_at` - Instance creation timestamp

**Relationships:**
- One-to-many with sessions
- One-to-many with observations
- One-to-many with patterns
- One-to-many with concepts

**Usage:**
Created/retrieved automatically by `get_or_create_ai_instance()` helper function.

---

### sessions

Represents individual conversations, linking to AI instances and providing temporal context.

```sql
CREATE TABLE sessions (
    id                  INTEGER PRIMARY KEY,
    conversation_id     VARCHAR(255) UNIQUE,
    title               VARCHAR(500),
    ai_instance_id      INTEGER REFERENCES ai_instances(id),
    source_url          VARCHAR(1000),
    started_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_active         TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    summary             TEXT,
    metadata            JSONB DEFAULT '{}'
);
```

**Columns:**
- `id` - Auto-increment primary key
- `conversation_id` - Opaque stable identifier from conversation URL (often UUID, but format varies by platform) - UNIQUE
- `title` - Conversation title
- `ai_instance_id` - Foreign key to ai_instances
- `source_url` - Full URL to conversation (e.g., https://claude.ai/chat/...)
- `started_at` - Session start timestamp
- `last_active` - Last activity timestamp (auto-updates on modification)
- `summary` - Optional conversation summary
- `metadata` - JSONB field for arbitrary metadata

**Note:** In the SQLAlchemy ORM, some columns may be suffixed with `_` to avoid reserved attribute names (e.g., `metadata_` in Python maps to `metadata` in SQL).

**Relationships:**
- Many-to-one with ai_instances
- One-to-many with observations
- One-to-many with patterns

**Usage:**
Created/updated by `memory_init_session()` MCP tool.

---

### observations

Core data storage - individual facts, preferences, or insights with confidence ratings and evidence.

```sql
CREATE TABLE observations (
    id                  INTEGER PRIMARY KEY,
    timestamp           TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    observation         TEXT NOT NULL,
    confidence          FLOAT DEFAULT 0.8 CHECK (confidence >= 0 AND confidence <= 1),
    domain              VARCHAR(100),
    evidence            JSONB DEFAULT '[]',
    session_id          INTEGER REFERENCES sessions(id),
    ai_instance_id      INTEGER REFERENCES ai_instances(id),
    access_count        BIGINT DEFAULT 0,
    last_accessed_at    TIMESTAMP WITH TIME ZONE,
    tier                memory_tier NOT NULL DEFAULT 'hot',
    archived_at         TIMESTAMP WITH TIME ZONE,
    archived_reason     TEXT,
    archived_by         VARCHAR(100),
    score               FLOAT NOT NULL DEFAULT 0.0,
    floor_score         FLOAT NOT NULL DEFAULT -9999.0,
    purge_eligible      BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX ix_observations_domain ON observations(domain);
CREATE INDEX ix_observations_confidence ON observations(confidence);
CREATE INDEX ix_observations_tier ON observations(tier);
CREATE INDEX ix_observations_score ON observations(score);
```

**Columns:**
- `id` - Auto-increment primary key
- `timestamp` - Observation creation time
- `observation` - The actual observation text (embedded for semantic search)
- `confidence` - Confidence level 0.0-1.0 (constraint enforced)
- `domain` - Category/domain tag (indexed)
- `evidence` - JSONB array of supporting evidence strings
- `session_id` - Foreign key to sessions (optional)
- `ai_instance_id` - Foreign key to ai_instances (tracks which AI made observation)
- `access_count` - Number of times retrieved (incremented on search/recall)
- `last_accessed_at` - Timestamp of last access
- `tier` - Hot or cold tier flag
- `archived_at` - Timestamp when archived to cold tier
- `archived_reason` - Reason for archival
- `archived_by` - Actor responsible for archival
- `score` - Retention score (decays over time)
- `floor_score` - Lower bound for score decay
- `purge_eligible` - Soft-purge marker when score is below threshold

**Confidence Levels:**
- `1.0` - Absolute certainty, direct observation
- `0.95-0.99` - Very high confidence, strong evidence
- `0.85-0.94` - High confidence, solid evidence
- `0.70-0.84` - Good confidence, minor uncertainty
- `0.50-0.69` - Moderate confidence, competing interpretations
- `< 0.50` - Low confidence, speculative

**Common Domains:**
- `technical_milestone` - System achievements
- `interaction_patterns` - User behavior
- `project_context` - Project-specific info
- `decision_making` - Reasoning and choices
- `system_architecture` - Technical design

**Relationships:**
- Many-to-one with sessions (optional)
- Many-to-one with ai_instances
- Polymorphically linked to embeddings via (source_type='observation', source_id) - no FK constraint, integrity maintained in application logic

**MCP Tools:**
- Created by: `memory_store()`
- Retrieved by: `memory_recall()`, `memory_search()`

---

### embeddings

Unified vector storage for all embeddable entities (observations, patterns, concepts, documents).

```sql
CREATE TABLE embeddings (
    source_type         VARCHAR(50) NOT NULL,
    source_id           INTEGER NOT NULL,
    model_version       VARCHAR(100) NOT NULL DEFAULT 'text-embedding-3-small',
    embedding           VECTOR(1536) NOT NULL,
    normalized          BOOLEAN DEFAULT TRUE,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (source_type, source_id, model_version)
);

CREATE INDEX ix_embeddings_source ON embeddings(source_type, source_id);
CREATE INDEX ix_embeddings_vector_hnsw ON embeddings USING hnsw (embedding vector_cosine_ops);
```

**Columns:**
- `source_type` - Entity type: 'observation', 'pattern', 'concept', 'document'
- `source_id` - ID of the source entity
- `model_version` - Embedding model used (default: 'text-embedding-3-small')
- `embedding` - 1536-dimensional vector (OpenAI text-embedding-3-small)
- `normalized` - Whether vector is normalized (default: true)
- `created_at` - Embedding generation timestamp

**Composite Primary Key:** (source_type, source_id, model_version)

**Indexes:**
- Composite index on (source_type, source_id) for lookups
- HNSW index on embedding vector for fast cosine similarity search

**Vector Operations:**
- Cosine similarity: `1 - (embedding <=> query_embedding)`
- Used by `memory_search()` for unified semantic search across all source types

**Notes:**
- Embeddings are generated at write/search time by calling the OpenAI embeddings API (currently synchronous in the MCP tool path)
- Embeddings are polymorphically linked to source entities; there is no FK constraint - integrity is maintained in application logic
- HNSW index creation is best-effort; if unsupported by pgvector version, the system still functions with brute-force vector operations
- All vectors normalized to unit length for consistent similarity metrics
- ORM uses callable defaults (dict/list) for JSONB fields to avoid shared mutable defaults

---

### patterns

Synthesized understanding across multiple observations - higher-level insights that evolve over time.

```sql
CREATE TABLE patterns (
    id                          INTEGER PRIMARY KEY,
    category                    VARCHAR(100) NOT NULL,
    pattern_name                VARCHAR(255) NOT NULL,
    pattern_text                TEXT NOT NULL,
    confidence                  FLOAT DEFAULT 0.8,
    last_updated                TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    evidence_observation_ids    JSONB DEFAULT list,
    session_id                  INTEGER REFERENCES sessions(id),
    ai_instance_id              INTEGER REFERENCES ai_instances(id),
    access_count                BIGINT DEFAULT 0,
    last_accessed_at            TIMESTAMP WITH TIME ZONE,
    tier                        memory_tier NOT NULL DEFAULT 'hot',
    archived_at                 TIMESTAMP WITH TIME ZONE,
    archived_reason             TEXT,
    archived_by                 VARCHAR(100),
    score                       FLOAT NOT NULL DEFAULT 0.0,
    floor_score                 FLOAT NOT NULL DEFAULT -9999.0,
    purge_eligible              BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT uq_pattern_category_name UNIQUE (category, pattern_name)
);

CREATE INDEX ix_patterns_category ON patterns(category);
CREATE INDEX ix_patterns_tier ON patterns(tier);
CREATE INDEX ix_patterns_score ON patterns(score);
```

**Columns:**
- `id` - Auto-increment primary key
- `category` - Pattern category/domain
- `pattern_name` - Pattern identifier (unique within category)
- `pattern_text` - Pattern description (this gets embedded for semantic search)
- `confidence` - Pattern confidence level 0.0-1.0
- `last_updated` - Auto-updated on modification
- `evidence_observation_ids` - JSONB array of supporting observation IDs
- `session_id` - Optional session link
- `ai_instance_id` - AI that synthesized the pattern
- `access_count` / `last_accessed_at` - Usage tracking
- `tier` - Hot or cold tier flag
- `archived_at` - Timestamp when archived to cold tier
- `archived_reason` - Reason for archival
- `archived_by` - Actor responsible for archival
- `score` - Retention score (decays over time)
- `floor_score` - Lower bound for score decay
- `purge_eligible` - Soft-purge marker when score is below threshold

**Unique Constraint:** (category, pattern_name) - prevents duplicates

**Relationships:**
- Many-to-one with sessions (optional)
- Many-to-one with ai_instances
- Logical links to observations via evidence_observation_ids
- Polymorphically linked to embeddings via (source_type='pattern', source_id)

**Pattern Evolution:**
Patterns are designed to evolve as understanding grows. The `memory_update_pattern()` tool performs upsert:
- If pattern exists (by category + pattern_name), updates pattern_text, confidence, evidence
- If pattern doesn't exist, creates new pattern
- Embedding is regenerated on each update

**MCP Tools:**
- Created/Updated by: `memory_update_pattern()` ✅ Implemented (upsert)
- Retrieved by: `memory_get_pattern()` ✅ Implemented
- Listed by: `memory_patterns()` ✅ Implemented (with category/confidence filters)
- Retrieved by: `memory_search()` (unified semantic search) ✅ Implemented

---

### concepts

Knowledge graph nodes - projects, frameworks, components, theories, constructs.

```sql
CREATE TABLE concepts (
    id                  INTEGER PRIMARY KEY,
    name                VARCHAR(255) NOT NULL,
    name_key            VARCHAR(255) NOT NULL,
    type                VARCHAR(50) NOT NULL,
    status              VARCHAR(50),
    domain              VARCHAR(100),
    description         TEXT,
    metadata            JSONB DEFAULT '{}',
    ai_instance_id      INTEGER REFERENCES ai_instances(id),
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count        BIGINT DEFAULT 0,
    last_accessed_at    TIMESTAMP WITH TIME ZONE,
    tier                memory_tier NOT NULL DEFAULT 'hot',
    archived_at         TIMESTAMP WITH TIME ZONE,
    archived_reason     TEXT,
    archived_by         VARCHAR(100),
    score               FLOAT NOT NULL DEFAULT 0.0,
    floor_score         FLOAT NOT NULL DEFAULT -9999.0,
    purge_eligible      BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX ix_concepts_name_key ON concepts(name_key);
CREATE INDEX ix_concepts_type ON concepts(type);
CREATE INDEX ix_concepts_tier ON concepts(tier);
CREATE INDEX ix_concepts_score ON concepts(score);
```

**Columns:**
- `id` - Auto-increment primary key
- `name` - Display name (case preserved)
- `name_key` - Lowercase version for case-insensitive lookups
- `type` - Concept type: project, framework, component, construct, theory
- `status` - Optional status (active, archived, deprecated)
- `domain` - Optional domain categorization
- `description` - Text description (embedded for search)
- `metadata` - JSONB for arbitrary metadata
- `ai_instance_id` - AI that created the concept
- `created_at` - Creation timestamp
- `access_count` / `last_accessed_at` - Usage tracking
- `tier` - Hot or cold tier flag
- `archived_at` - Timestamp when archived to cold tier
- `archived_reason` - Reason for archival
- `archived_by` - Actor responsible for archival
- `score` - Retention score (decays over time)
- `floor_score` - Lower bound for score decay
- `purge_eligible` - Soft-purge marker when score is below threshold

**Concept Types:**
- `project` - Concrete projects
- `framework` - Theoretical frameworks
- `component` - System components
- `construct` - Abstract constructs
- `theory` - Theoretical concepts

**Relationships:**
- Many-to-one with ai_instances
- One-to-many with concept_aliases
- Many-to-many with self via concept_relationships
- Polymorphically linked to embeddings via (source_type='concept', source_id)

**MCP Tools:**
- Created by: `memory_store_concept()` ✅ Implemented
- Retrieved by: `memory_get_concept()` ✅ Implemented (case-insensitive, alias-aware)
- Retrieved by: `memory_search()` (unified semantic search) ✅ Implemented
- Graph query: `memory_related_concepts()` ✅ Implemented

---

### concept_aliases

Alternative names for concepts to prevent fragmentation.

```sql
CREATE TABLE concept_aliases (
    alias               VARCHAR(255) PRIMARY KEY,
    concept_id          INTEGER NOT NULL REFERENCES concepts(id),
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Columns:**
- `alias` - Alternative name (primary key)
- `concept_id` - Foreign key to canonical concept
- `created_at` - Alias creation timestamp

**Usage:**
Allows "Glyph" → "Cathedral-v2" or "SELFHELP" → "selfhelp" mappings.

**MCP Tools:**
- Created by: `memory_add_concept_alias()` ✅ Implemented
- Auto-resolved in: `memory_get_concept()` (aliases work transparently)

---

### concept_relationships

Knowledge graph edges - how concepts relate to each other.

```sql
CREATE TABLE concept_relationships (
    from_concept_id     INTEGER NOT NULL REFERENCES concepts(id),
    to_concept_id       INTEGER NOT NULL REFERENCES concepts(id),
    rel_type            VARCHAR(50) NOT NULL,
    weight              FLOAT DEFAULT 0.5 CHECK (weight >= 0 AND weight <= 1),
    description         TEXT,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (from_concept_id, to_concept_id, rel_type)
);
```

**Columns:**
- `from_concept_id` - Source concept
- `to_concept_id` - Target concept
- `rel_type` - Relationship type
- `weight` - Relationship strength 0.0-1.0
- `description` - Optional description
- `created_at` - Relationship creation timestamp

**Relationship Types:**
- `enables` - One concept enables another
- `version_of` - Version relationship
- `part_of` - Component relationship
- `related_to` - General association
- `implements` - Implementation relationship
- `demonstrates` - Demonstration/example relationship

**Composite Primary Key:** (from_concept_id, to_concept_id, rel_type)

**MCP Tools:**
- Created by: `memory_add_concept_relationship()` ✅ Implemented
- Queried by: `memory_related_concepts()` ✅ Implemented

---

### documents

External document references with summaries (canonical storage: Google Drive).

**Architecture:**
MemoryGate stores document *references* with summaries, not full content. Full document content lives in canonical storage (Google Drive) and is fetched on demand via Drive API.

**What's stored in database:**
- Metadata (title, doc_type, publication date)
- Summary/abstract (embedded for semantic search)
- URL (Google Drive share link or file ID)
- Key concepts (extracted topics)
- Arbitrary metadata (word count, publisher, etc.)

**What's NOT stored:**
- Full document content (articles, papers, books)
- Binary data (PDFs, images)
- Raw HTML/markdown

**Cost savings:** ~100x storage reduction (500 bytes summary vs 50KB full document)

```sql
CREATE TABLE documents (
    id                  INTEGER PRIMARY KEY,
    title               VARCHAR(500) NOT NULL,
    doc_type            VARCHAR(50) NOT NULL,
    content_summary     TEXT,
    url                 VARCHAR(1000),
    publication_date    TIMESTAMP WITH TIME ZONE,
    key_concepts        JSONB DEFAULT list,
    metadata            JSONB DEFAULT dict,
    access_count        BIGINT DEFAULT 0,
    last_accessed_at    TIMESTAMP WITH TIME ZONE,
    tier                memory_tier NOT NULL DEFAULT 'hot',
    archived_at         TIMESTAMP WITH TIME ZONE,
    archived_reason     TEXT,
    archived_by         VARCHAR(100),
    score               FLOAT NOT NULL DEFAULT 0.0,
    floor_score         FLOAT NOT NULL DEFAULT -9999.0,
    purge_eligible      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX ix_documents_tier ON documents(tier);
CREATE INDEX ix_documents_score ON documents(score);
```

**Columns:**
- `id` - Auto-increment primary key
- `title` - Document title
- `doc_type` - Document type (article, paper, book, documentation)
- `content_summary` - Summary text (this gets embedded for semantic search)
- `url` - Google Drive share link or file ID (https://drive.google.com/...)
- `publication_date` - Publication timestamp
- `key_concepts` - JSONB array of associated concept names
- `metadata` - JSONB for arbitrary metadata (word count, publisher, etc.)
- `access_count` / `last_accessed_at` - Usage tracking
- `tier` - Hot or cold tier flag
- `archived_at` - Timestamp when archived to cold tier
- `archived_reason` - Reason for archival
- `archived_by` - Actor responsible for archival
- `score` - Retention score (decays over time)
- `floor_score` - Lower bound for score decay
- `purge_eligible` - Soft-purge marker when score is below threshold
- `created_at` - Record creation timestamp

**Relationships:**
- Polymorphically linked to embeddings via (source_type='document', source_id)
- Logical links to concepts via key_concepts array

**MCP Tools:**
- Created by: `memory_store_document()` ✅ Implemented
- Retrieved by: `memory_search()` (unified search) ✅ Implemented

**Usage Workflow:**
1. Store document with `memory_store_document(title, url, summary, key_concepts)`
2. Summary gets embedded automatically
3. Later: `memory_search("AI consciousness frameworks")` finds document via semantic similarity
4. Fetch full content from Google Drive when needed (not yet implemented - future enhancement)

---

### memory_summaries

Summaries generated during archive and retention workflows.

```sql
CREATE TABLE memory_summaries (
    id                  INTEGER PRIMARY KEY,
    source_type         VARCHAR(50) NOT NULL,
    source_id           INTEGER,
    source_ids          JSONB DEFAULT '[]',
    summary_text        TEXT NOT NULL,
    metadata            JSONB DEFAULT '{}',
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count        BIGINT DEFAULT 0,
    last_accessed_at    TIMESTAMP WITH TIME ZONE,
    tier                memory_tier NOT NULL DEFAULT 'hot',
    archived_at         TIMESTAMP WITH TIME ZONE,
    archived_reason     TEXT,
    archived_by         VARCHAR(100),
    score               FLOAT NOT NULL DEFAULT 0.0,
    floor_score         FLOAT NOT NULL DEFAULT -9999.0,
    purge_eligible      BOOLEAN NOT NULL DEFAULT FALSE
);
```

**Columns:**
- `source_type` / `source_id` - Links back to the original record
- `source_ids` - Optional list of source IDs for cluster summaries
- `summary_text` - Summary content
- `metadata` - JSONB metadata for summarization context
- `access_count` / `last_accessed_at` - Usage tracking
- `tier` / `archived_*` - Cold storage lifecycle fields
- `score` / `floor_score` / `purge_eligible` - Retention scoring fields

**Usage:**
- Created by retention jobs and `archive_memory()` in summarize mode
- Retrieved via `search_cold_memory()` or `rehydrate_memory()`

---

### memory_tombstones

Audit trail for archive, rehydrate, purge, and summarize actions.

```sql
CREATE TABLE memory_tombstones (
    id                  UUID PRIMARY KEY,
    memory_id           VARCHAR(255) NOT NULL,
    action              tombstone_action NOT NULL,
    from_tier           memory_tier,
    to_tier             memory_tier,
    reason              TEXT,
    actor               VARCHAR(100),
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata            JSONB DEFAULT '{}'
);
```

**Columns:**
- `memory_id` - Identifier in the form `type:id` (or `summary:id`)
- `action` - archived | rehydrated | purged | summarized
- `from_tier` / `to_tier` - Tier transition context
- `reason` / `actor` - Why and who performed the change
- `metadata` - JSONB metadata payload

**Usage:**
- Written automatically during archive/rehydrate/purge/summarize actions
- Supports audit and compliance requirements

---

## Database Initialization

### Automatic Setup

The server performs initialization on startup via `init_db()`:

```python
1. Connect to PostgreSQL
2. Enable pgvector extension: CREATE EXTENSION IF NOT EXISTS vector
3. Validate schema revision (Alembic) and optionally auto-upgrade
4. Create indexes defined in migrations (including HNSW if supported)
```

### Manual Reset

To rebuild schema from scratch:

```bash
# Connect to database
fly postgres connect -a memorygate-db

# Drop and recreate schema
DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
\q

# Recreate schema with Alembic
alembic upgrade head
```

If `AUTO_MIGRATE_ON_STARTUP=true`, restarting the server will also apply migrations in dev environments.

---

## Embedding Pipeline

### Generation Flow

```
1. User calls memory_store(observation="...")
2. Create observation record → observation.id
3. Call OpenAI API: embed_text_sync(observation)
4. Store in embeddings table:
   - source_type: 'observation'
   - source_id: observation.id
   - embedding: 1536d vector
5. HNSW index automatically updated
```

### Search Flow

```
1. User calls memory_search(query="...")
2. Generate query embedding via OpenAI API (synchronous)
3. Execute unified pgvector similarity query:
   SELECT * FROM embeddings e
   LEFT JOIN observations/patterns/concepts/documents
   ORDER BY e.embedding <=> query_embedding
4. Return results with similarity scores and source_type discriminator
5. Update access_count, last_accessed_at, and score for hot-tier matches
```

---

## Access Tracking

All main tables include:
- `access_count` - Incremented on every retrieval
- `last_accessed_at` - Updated with current timestamp
- `tier` / `score` - Used by retention to decide hot vs cold

This enables:
- Usage analytics
- Cache eviction strategies
- Access pattern analysis
- Stale data identification

---

## Retention and Cold Storage

Retention uses a hot/cold tier and a decaying score:
- Hot tier is the default for search and recall.
- Scores decay on a periodic retention tick.
- Records at or below -1 are summarized and archived to cold.
- Records at or below -2 are marked purge eligible (soft) or deleted (hard).
- Tombstones preserve an audit trail for archive, rehydrate, purge, and summarize actions.

Cold-tier records are only returned when explicitly requested (e.g., `search_cold_memory()` or `include_cold=true`).

---

## Performance Considerations

### Indexes

**Existing:**
- `ix_observations_domain` - Fast domain filtering
- `ix_observations_confidence` - Fast confidence filtering
- `ix_observations_tier` / `ix_observations_score` - Tiering and decay lookups
- `ix_embeddings_source` - Fast embedding lookups
- `ix_embeddings_vector_hnsw` - Approximate nearest neighbor search
- `ix_concepts_name_key` - Case-insensitive concept lookup
- `ix_concepts_type` - Type-based filtering
- `ix_concepts_tier` / `ix_concepts_score` - Tiering and decay lookups
- `ix_patterns_category` - Category-based pattern lookup
- `ix_patterns_tier` / `ix_patterns_score` - Tiering and decay lookups
- `ix_documents_tier` / `ix_documents_score` - Tiering and decay lookups

### HNSW Index

```sql
CREATE INDEX ix_embeddings_vector_hnsw 
ON embeddings USING hnsw (embedding vector_cosine_ops);
```

**Properties:**
- Approximate nearest neighbor (ANN) algorithm
- O(log n) search complexity
- Optimized for cosine similarity
- Trade-off: Speed vs accuracy
- Excellent for 1000s-millions of vectors

---

## Migration Strategy

When adding new MCP tools or schema changes:

1. Add an Alembic migration in `alembic/versions`
2. Update MCP tool functions in server.py
3. Follow the observations pattern:
   - Store with embedding generation
   - Search with vector similarity
   - Recall with SQL filtering
4. Update this documentation

---

## Data Integrity

### Constraints

- Confidence values: 0.0 ≤ confidence ≤ 1.0
- Relationship weights: 0.0 ≤ weight ≤ 1.0
- Unique conversation_id per session
- Unique (category, pattern_name) per pattern
- Unique alias names

### Cascading

Currently no CASCADE deletes configured. Consider:
- Deleting AI instance behavior (orphan sessions/observations?)
- Deleting session behavior (orphan observations?)
- Deleting concept behavior (orphan relationships/aliases?)

---

## Canonical Invariants

Core principles that guide system behavior and future development:

1. **Embedding Source Text:**
   - `observations.observation` is the canonical text that gets embedded
   - `patterns.pattern_text` for patterns
   - `concepts.description` for concepts
   - `documents.content_summary` for documents

2. **Multi-Model Support:**
   - `embeddings` table may contain multiple vectors per entity (by model_version)
   - Allows gradual model upgrades without data loss

3. **Optional Provenance:**
   - `sessions` are optional provenance; observations can exist without sessions
   - Enables both conversational and non-conversational data storage

4. **Best-Effort Metrics:**
   - Access metrics (`access_count`, `last_accessed_at`) are best-effort and may lag under failures
   - Not used for critical business logic

5. **No Automatic Cascades:**
   - No cascading deletes configured
   - Deletion requires explicit cleanup to prevent accidental data loss
   - Polymorphic relationships maintained in application logic

6. **Document Storage Architecture:**
   - Documents stored as references, not full content
   - Canonical storage: Google Drive
   - Summary embedded for search; full content fetched on demand
   - Reduces database storage by ~100x while maintaining full access

---

## Future Enhancements

### Planned Schema Changes

- [ ] Add `uncertainties` table for tracking open questions
- [ ] Add `session_events` for detailed interaction logging
- [ ] Add `memory_consolidation` for recursive summarization
- [ ] Consider partitioning for observations (by timestamp)
- [ ] Add full-text search indexes for text fields

### Optimization Opportunities

- [ ] Materialized views for common queries
- [ ] Partial indexes for active/high-confidence data
- [ ] Table partitioning for temporal queries
- [ ] Connection pooling configuration
- [ ] Read replicas for search-heavy workloads

---

**Last Updated:** January 2, 2026  
**Schema Version:** 0.1.0  
**Database:** PostgreSQL 15/16/17 (Fly managed) + pgvector
