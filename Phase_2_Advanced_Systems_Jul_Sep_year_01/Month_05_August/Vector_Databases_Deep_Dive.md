# Vector Databases — Deep Dive Comparison Guide
### Phase 2 Supplementary | August 2026 Reference

> A thorough comparison guide for the 5 most important vector databases used in production RAG and AI systems. Critical knowledge for ML System Design interviews.

---

## Part 1 — Why Vector Databases?

Traditional databases (SQL/NoSQL) find data by **exact match** or **range queries**:
```sql
SELECT * FROM products WHERE category = 'electronics' AND price < 100
```

Vector databases find data by **semantic similarity**:
```python
# "Find documents most similar in meaning to this query"
results = vdb.query(
    vector=embed("What laptops have good battery life?"),
    top_k=5
)
# Returns: documents about battery specs, laptop reviews — even if none say 
# "battery life" exactly — because they're semantically close
```

**The core operation:** Approximate Nearest Neighbour (ANN) search in high-dimensional space.

---

## Part 2 — The ANN Algorithms

All vector databases are built on one of these core index types:

### HNSW (Hierarchical Navigable Small World)
**Used by:** Qdrant (default), Weaviate, pgvector, Chroma, Milvus

```
Intuition: A multi-layer graph where:
  - Top layers: long-range connections (skip lists for fast navigation)
  - Bottom layer: short-range connections (actual neighbours)
  - Search: enter at top, greedily navigate to nearest neighbours at each layer

Parameters:
  M: number of edges per node (affects quality and memory)
     Typical: 16–48. Higher M → better recall, more memory
  ef_construction: search beam width during index build
     Typical: 100–200. Higher → better quality index, slower build
  ef_search: search beam width during query
     Typical: 64–200. Dynamic — tune per query for recall/speed tradeoff

Complexity:
  Build: O(n × M × log(n))  
  Query: O(log(n))  ← very fast!
  Memory: O(n × M × d × 4 bytes)  for n vectors, M connections, d dims

When to use: Default choice for most use cases (best recall/speed).
```

### IVF (Inverted File Index)
**Used by:** Faiss (Meta), Milvus, Pinecone (under the hood)

```
Intuition: 
  1. K-Means cluster vectors into N clusters (centroids)
  2. At search time: find top-k nearest centroids, then search only those clusters
  
Parameters:
  nlist: number of clusters. Typical: sqrt(n) to 4×sqrt(n)
  nprobe: clusters to search at query time. Low nprobe → fast but lower recall

Variants:
  IVF_FLAT:   brute-force within selected clusters — exact
  IVF_SQ8:    scalar quantization within clusters — 4× memory reduction
  IVF_PQ:     product quantization — 8–32× memory reduction, more accuracy loss

Complexity:
  Build: O(n × K × iterations)  K-Means is expensive!
  Query: O(nprobe × cluster_size) much less than O(n)

When to use: Very large scale (100M+) where HNSW memory is too expensive.
```

### Product Quantization (PQ)
**Used by:** Faiss, Pinecone, Milvus — as compression layer

```
Intuition:
  Split each d-dimensional vector into m sub-vectors of d/m dimensions each
  For each sub-vector, train a small codebook of 256 centroids
  Replace each sub-vector with the index of its nearest centroid (1 byte)
  
Result: d-dimensional float32 vector (d×4 bytes) → m bytes
Example: 1536-dim vector (6144 bytes) → 96 bytes (64×!) with m=96

Tradeoff: massive compression, moderate quality loss (~5% recall reduction)

When to use: 100M+ vectors where HNSW RAM would be terabytes
```

---

## Part 3 — Key Players Compared

### Pinecone

**Type:** Fully managed SaaS — no self-hosting.

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="...")

# Create index
pc.create_index(
    name="my-rag-index",
    dimension=1536,  # OpenAI text-embedding-3-small
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index("my-rag-index")

# Upsert vectors
index.upsert(vectors=[
    {"id": "doc1", "values": [0.1, 0.2, ...], "metadata": {"text": "...", "source": "pdf"}},
    {"id": "doc2", "values": [0.3, 0.1, ...], "metadata": {"text": "...", "source": "web"}},
])

# Query with metadata filter
results = index.query(
    vector=[0.1, 0.5, ...],
    top_k=5,
    filter={"source": {"$eq": "pdf"}},
    include_metadata=True
)
```

**Pros:** Zero ops overhead, scales automatically, highly optimised, strong SLA.  
**Cons:** Vendor lock-in, expensive at scale, no self-hosting option.  
**Best for:** Production RAG when you want zero infrastructure management.

---

### Weaviate

**Type:** Open-source + managed cloud (Weaviate Cloud).

```python
import weaviate
from weaviate.classes.init import Auth

# Connect to local Weaviate
client = weaviate.connect_to_local()

# Create a schema with vectorizer config
client.collections.create(
    name="Document",
    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(
        model="text-embedding-3-small"
    ),
    properties=[
        wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT),
    ]
)

# Add objects — Weaviate auto-embeds!
collection = client.collections.get("Document")
collection.data.insert({"text": "LLMs use attention mechanisms...", "source": "paper"})

# Hybrid search (vector + BM25)
response = collection.query.hybrid(
    query="how does attention work",
    alpha=0.5,          # 0=BM25 only, 1=vector only, 0.5=balanced
    limit=5
)
```

**Pros:** Native hybrid search, auto-vectorization, GraphQL API, rich filtering.  
**Cons:** More complex setup, higher memory footprint than Qdrant.  
**Best for:** Complex queries needing hybrid search + rich metadata filtering.

---

### Qdrant

**Type:** Open-source (Rust) + managed cloud.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

client = QdrantClient(host="localhost", port=6333)

# Create collection with HNSW config
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    hnsw_config={"m": 16, "ef_construct": 100}
)

# Upsert points
client.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],
            payload={"text": "...", "source": "paper", "year": 2024}
        )
    ]
)

# Search with filter
results = client.search(
    collection_name="documents",
    query_vector=[0.1, 0.5, ...],
    query_filter=Filter(
        must=[FieldCondition(key="year", match=MatchValue(value=2024))]
    ),
    limit=5,
    with_payload=True
)

# Sparse + dense hybrid search (Qdrant 1.7+)
from qdrant_client.models import SparseVector, NamedSparseVector

results = client.search_batch(
    collection_name="documents",
    requests=[
        models.SearchRequest(vector=models.NamedVector(name="dense", vector=dense_vec), limit=10),
        models.SearchRequest(vector=models.NamedSparseVector(name="sparse", vector=SparseVector(indices=[1,5,42], values=[0.3,0.7,0.2])), limit=10),
    ]
)
```

**Pros:** Best performance/memory trade-off, Rust performance, excellent filtration.  
**Cons:** Smaller ecosystem than Weaviate, sparse vector support newer.  
**Best for:** High-performance self-hosted RAG, cost-sensitive deployments.

---

### pgvector (PostgreSQL Extension)

**Type:** PostgreSQL extension — data lives in your existing Postgres infrastructure.

```sql
-- Enable extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    source VARCHAR(100),
    embedding vector(1536)  -- dimension must match embedding model
);

-- Create HNSW index
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
-- OR IVFFlat for larger datasets
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Insert with embedding
INSERT INTO documents (content, source, embedding)
VALUES ('LLMs use attention mechanisms...', 'paper', '[0.1, 0.2, ...]'::vector);

-- Semantic search
SELECT content, source,
       1 - (embedding <=> '[0.1, 0.5, ...]'::vector) AS similarity
FROM documents
ORDER BY embedding <=> '[0.1, 0.5, ...]'::vector
LIMIT 5;

-- Hybrid search: combine SQL filters with vector similarity
SELECT content, source
FROM documents
WHERE source = 'paper' AND year >= 2023
ORDER BY embedding <=> '[0.1, 0.5, ...]'::vector
LIMIT 5;
```

**Python with SQLAlchemy:**
```python
from sqlalchemy import Column, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase
from pgvector.sqlalchemy import Vector

class Document(DeclarativeBase):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    source = Column(String)
    embedding = Column(Vector(1536))  # pgvector type
```

**Pros:** No additional infrastructure — reuse existing Postgres, full ACID, joins work.  
**Cons:** Not as fast as dedicated VDBs for pure ANN at scale, scaling requires Postgres tuning.  
**Best for:** <10M vectors, teams already on Postgres, need ACID + vector in one system.

---

### Chroma (Lightweight, Dev-First)

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_db")

# With built-in embedding
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="...", model_name="text-embedding-3-small"
)

collection = client.get_or_create_collection(
    name="my_docs",
    embedding_function=openai_ef
)

# Add documents — Chroma embeds automatically
collection.add(
    documents=["LLMs use attention...", "RAG retrieves documents..."],
    ids=["doc1", "doc2"],
    metadatas=[{"source": "paper"}, {"source": "blog"}]
)

# Query
results = collection.query(
    query_texts=["how does RAG work?"],
    n_results=3,
    where={"source": "paper"}  # metadata filter
)
```

**Pros:** Simplest API, great for prototyping and notebooks, can run in-memory.  
**Cons:** Not production-ready at scale (no distributed mode, limited performance).  
**Best for:** Rapid prototyping, demos, local development.

---

## Part 4 — Comparison Matrix

| Feature | Pinecone | Weaviate | Qdrant | pgvector | Chroma |
|---|---|---|---|---|---|
| Hosting | SaaS only | Open + Cloud | Open + Cloud | Self-hosted | Open + Cloud |
| Language | — | Go | Rust | C (PG ext) | Python |
| ANN Index | Proprietary | HNSW | HNSW | HNSW / IVF | HNSW |
| Hybrid Search | ✅ (sparse+dense) | ✅ (native BM25) | ✅ (v1.7+) | Manual (SQL+vec) | ❌ |
| Metadata Filter | ✅ | ✅ (GraphQL) | ✅ (rich) | ✅ (SQL WHERE) | ✅ (basic) |
| Multi-tenancy | ✅ namespaces | ✅ tenants | ✅ | ✅ (DB schemas) | Limited |
| Scalability | 🔴 Managed scales | 🟠 Needs config | 🟠 Needs config | 🔴 Postgres limits | 🔴 Single node |
| Cost | 💸 Expensive | 💰 Medium | 💰 Cheapest | 💵 Compute only | Free (self-host) |
| Best at | Zero ops | Rich schema + hybrid | Performance + efficiency | Existing Postgres | Quick prototyping |

---

## Part 5 — Choosing the Right Vector DB

### Decision Framework

```
1. Production scale > 100M vectors?
   → Pinecone (managed) or Qdrant/Milvus (self-hosted at scale)

2. Already on PostgreSQL infrastructure?
   → pgvector first. Avoids new infra. Revisit if performance degrades.

3. Need hybrid search (keyword + semantic) out of the box?
   → Weaviate (best native hybrid) or Qdrant 1.7+

4. Cost-sensitive self-hosted?
   → Qdrant (most efficient per GB, Rust performance)

5. Rapid prototyping / demo?
   → Chroma (simplest API, no setup)

6. Zero infra management, production RAG?
   → Pinecone (best SLA, easiest ops)

7. Complex schema with relationships?
   → Weaviate (GraphQL schema, cross-references between objects)
```

---

## Part 6 — Interview Q&A

**Q: Explain how HNSW achieves O(log n) search complexity.**

> HNSW builds a multi-layer graph. The top layers are sparse with long-range connections; the bottom layer is dense with short-range connections. During search, we enter at the top layer and greedily navigate toward the query point using long-range edges. As we descend layers, the connections get shorter and more precise. This structure is similar to a skip list — the long-range edges allow us to skip large portions of the data space quickly, achieving O(log n) expected search time.

**Q: When would you choose pgvector over a dedicated vector database?**

> pgvector is ideal when: (1) you're already on PostgreSQL and want to avoid new infrastructure, (2) you need to join vector similarity results with relational data (e.g., user tables, metadata), (3) you need ACID guarantees on writes, and (4) your scale is under ~5–10M vectors. Above that scale, or where latency requirements are strict (sub-10ms), a dedicated vector DB like Qdrant or Pinecone will significantly outperform pgvector because they're purpose-built for ANN with in-memory indexes.

**Q: What is the recall-latency trade-off in approximate nearest neighbour search?**

> ANN algorithms trade exact recall for speed. For HNSW, the `ef_search` parameter controls this: a higher value searches more candidates and achieves higher recall (% of true nearest neighbours found) but takes longer. A lower value is faster but may miss some true nearest neighbours. In production, you tune `ef_search` to achieve your recall target (e.g., 95%) while meeting your latency SLA (e.g., P99 < 50ms). Most applications don't need 100% recall — retrieval quality at 90–95% recall is indistinguishable for RAG tasks.

**Q: How does hybrid search work and why is it better than pure vector search?**

> Hybrid search combines dense vector similarity (semantic meaning) with sparse keyword matching (BM25/TF-IDF). Vector search excels at semantic understanding but can miss exact keyword matches (e.g., product codes, names, technical terms). BM25 excels at exact matches but has no semantic understanding. Combining them via Reciprocal Rank Fusion (RRF) takes the rank positions from both and computes: `score = Σ 1/(k + rank_i)` for each system. This hybrid typically outperforms either approach alone, especially for queries mixing semantic intent with specific terminology.

**Q: What is Reciprocal Rank Fusion (RRF)?**

```python
def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """
    rankings: list of ranked result lists from different retrievers
    k: smoothing constant (default 60 from original paper)
    Returns: merged ranked list
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += 1 / (k + rank + 1)
    
    return sorted(scores, key=lambda x: scores[x], reverse=True)

# Example:
vector_results  = ["doc3", "doc1", "doc5", "doc2", "doc7"]   # semantic search
keyword_results = ["doc1", "doc4", "doc3", "doc6", "doc2"]   # BM25 keyword search

merged = reciprocal_rank_fusion([vector_results, keyword_results])
# doc1 and doc3 appear in both lists → high combined rank
```

---

*Vector Databases Deep Dive | Phase 2 Supplementary | Added April 2026*
