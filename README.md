Excellent. Below is a **senior-engineer-grade, production-style README** suitable for review by experienced engineers.

You can paste this directly into your `README.md`.

---

# 📘 AI Handbook Generator

Production-Grade RAG + Long-Form Generation System

---

## 1. Executive Summary

This project implements a modular Retrieval-Augmented Generation (RAG) system capable of:

* Uploading and indexing PDF documents
* Performing grounded question answering with citations
* Generating 20,000+ word structured handbooks
* Persisting outputs safely with resume/recovery capability
* Running locally via Docker
* Deploying to AWS EC2 with HTTPS

The system is designed for **correctness, modularity, and reproducibility**, not experimentation.

---

# 2. Architecture Overview

```
User
  │
  ▼
Streamlit UI (Thin Client)
  │
  ▼
RAG Service Layer
  │
  ├── VectorStore (Supabase pgvector)  ← Primary
  ├── FAISS (dev fallback)
  │
  ▼
Grok (xAI API)
  │
  ▼
Structured Long-Form Output
```

---

# 3. Core Design Principles

### 1️⃣ Thin UI

Streamlit handles presentation only.
No business logic lives in UI.

### 2️⃣ Explicit Service Modules

All orchestration exists in:

* `rag_service.py`
* `handbook_service.py`
* `vector_store.py`

### 3️⃣ Swappable Retrieval Backend

`VectorStore` interface allows:

* `SupabaseVectorStore` (production)
* `FaissVectorStore` (dev fallback)

Switch controlled via:

```
VECTOR_BACKEND=supabase | faiss
```

### 4️⃣ Section-Based Long Generation

Handbooks are generated section-by-section to:

* Avoid token overflow
* Prevent model drift
* Reduce hallucination
* Allow resume after crash

---

# 4. Technology Stack

| Layer            | Technology                             |
| ---------------- | -------------------------------------- |
| UI               | Streamlit                              |
| Backend          | Python 3.11                            |
| LLM              | Grok (xAI API)                         |
| Retrieval        | Supabase pgvector                      |
| Dev Retrieval    | FAISS                                  |
| Embeddings       | sentence-transformers/all-MiniLM-L6-v2 |
| Containerization | Docker                                 |
| Deployment       | AWS EC2 + ALB + ACM                    |

---

# 5. Data Flow

## 5.1 PDF Ingestion

1. Upload PDF
2. Extract text page-by-page
3. Chunk with overlap
4. Generate embeddings
5. Insert into Supabase

---

## 5.2 Retrieval Flow

```
User question
   ↓
Embed question
   ↓
Supabase RPC: match_chunks()
   ↓
Top-k similar chunks
   ↓
Prompt builder
   ↓
Grok completion
```

All answers must be grounded in retrieved context.

---

## 5.3 Handbook Generation Flow

```
Topic
   ↓
Generate outline
   ↓
Loop sections:
    - Retrieve relevant chunks
    - Build section prompt
    - Generate section
    - Append to disk
   ↓
Bibliography
```

Output is written incrementally to:

```
storage/handbooks/
```

---

# 6. Supabase Schema (pgvector)

### Extensions

```sql
create extension if not exists vector;
```

---

### documents table

```sql
create table public.documents (
    id uuid primary key default gen_random_uuid(),
    source_path text unique not null,
    title text,
    created_at timestamp with time zone default now()
);
```

---

### chunks table

```sql
create table public.chunks (
    id uuid primary key default gen_random_uuid(),
    document_id uuid references public.documents(id) on delete cascade,
    content text not null,
    metadata jsonb,
    embedding vector(384)
);
```

---

### Vector Index

```sql
create index on public.chunks
using ivfflat (embedding vector_cosine_ops)
with (lists = 100);
```

---

### RPC Function

```sql
create or replace function public.match_chunks(
  query_embedding vector(384),
  match_count int,
  filter jsonb default null
)
returns table (
  id uuid,
  content text,
  metadata jsonb,
  score float
)
language plpgsql
as $$
begin
  return query
  select
    chunks.id,
    chunks.content,
    chunks.metadata,
    1 - (chunks.embedding <=> query_embedding) as score
  from public.chunks
  order by chunks.embedding <=> query_embedding
  limit match_count;
end;
$$;
```

---

# 7. Environment Configuration

Create `.env` (NOT committed):

```env
XAI_API_KEY=your_key
XAI_MODEL=grok-4
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
VECTOR_BACKEND=supabase
```

---

# 8. Local Development

## Option A — Native

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Option B — Docker (Recommended)

```bash
docker compose up -d
```

System available at:

```
http://localhost:8501
```

---

# 9. Docker Architecture

* Secrets injected at runtime
* `.env` not baked into image
* Storage mounted as volume
* Restart policy enabled

---

# 10. AWS Deployment (EC2)

## Infrastructure

* EC2 (Ubuntu 22.04)
* Docker + Compose
* Application Load Balancer
* ACM TLS Certificate
* Route53 DNS

---

## Security Model

* ALB exposes 443 only
* EC2 port 8501 accessible only from ALB
* SSH restricted to personal IP
* Supabase service key stored as env var

---

# 11. Failure Modes & Mitigation

| Risk                 | Mitigation               |
| -------------------- | ------------------------ |
| Token overflow       | Section-based generation |
| Hallucination        | Retrieval required       |
| Embedding drift      | Fixed embedding model    |
| Crash mid-generation | Incremental file writes  |
| Retrieval empty      | Safe refusal response    |
| Supabase downtime    | FAISS fallback           |

---

# 12. Known Trade-offs

* No authentication (demo scope)
* No serverless architecture
* No async queue system
* No horizontal scaling
* Not multi-tenant

---

# 13. Reproducibility Checklist

* [ ] Supabase schema created
* [ ] `.env` configured
* [ ] `VECTOR_BACKEND=supabase`
* [ ] Docker container builds
* [ ] PDF upload works
* [ ] Indexing inserts into Supabase
* [ ] Retrieval returns grounded citations
* [ ] Handbook generation >20k words
* [ ] Restart container → state persists

---

# 14. Repository Structure

```
backend/
  embeddings.py
  ingest.py
  chunking.py
  rag_service.py
  handbook_service.py
  vector_store.py
  vector_store_faiss.py
  vector_store_supabase.py

storage/
  pdfs/
  data/
  handbooks/

app.py
Dockerfile
docker-compose.yml
README.md
```

---

# 15. Performance Notes

* Embedding model: 384-dim vectors
* Supabase ivfflat index
* Batch inserts: 200 rows
* Top-k retrieval default: 6
* Long-form generation tested up to 37k words

---

# 16. What This Is NOT

* Not a chatbot toy
* Not a fine-tuned model
* Not production SaaS
* Not serverless
* Not multi-user secure

This is a clean, controlled AI engineering implementation.

---

# 17. Conclusion

This system demonstrates:

* Proper RAG architecture
* Modular service boundaries
* Swappable retrieval backends
* Deterministic deployment via containers
* Safe long-form generation beyond 20k words
* Cloud-ready infrastructure design

It is designed to be **auditable, reproducible, and extensible**.

---

If you'd like, I can now generate:

* A **short executive summary version** (for non-technical reviewers)
* Or a **diagram-only architecture appendix**
* Or a **security & compliance addendum**
