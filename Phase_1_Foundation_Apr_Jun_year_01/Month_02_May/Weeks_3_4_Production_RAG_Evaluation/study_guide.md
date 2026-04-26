# Weeks 3–4: Production RAG & Evaluation
### Phase 1 | Month 2 | May 19 – June 1, 2026

> This is the most practically important section for your current MLOps → AI Engineer transition. RAG powers 80%+ of enterprise LLM applications.

---

## 🎯 Learning Objectives

By the end of these two weeks you will be able to:
- Design a production RAG system from scratch
- Implement chunking, embedding, and vector search
- Build re-ranking pipelines
- Evaluate RAG quality with RAGAS and LangSmith
- Answer all RAG system design questions in AI engineer interviews

---

## Part 1 — What Is RAG?

> 💡 **ELI5 (Explain Like I'm 5):**
> Imagine taking an exam. Without RAG, an LLM is taking a *closed-book exam*. It has to answer based purely on what it memorised in "school" (training). If it doesn't know, it might confidently guess (hallucinate). With RAG, it's an *open-book exam*. Before answering, you *retrieve* the exact relevant pages from your filing cabinet, hand them to the LLM, and say "Answer the question using *only* these pages."

> 📖 **Big picture:** LLMs are trained on data up to a cutoff date. After that, they know nothing about what happened. They also can't read your company's private documents, your internal Confluence wiki, or last week's earnings report. And when asked about things they don't know well, they confidently make up answers (hallucination).

### 1.1 The Problem RAG Solves

LLMs have three critical limitations:
1. **Knowledge cutoff** — training data ends at a date; can't answer about recent events
2. **Context window** — can't hold all your company's documents in one prompt
3. **Hallucination** — makes up facts when uncertain, especially for domain-specific knowledge

**RAG (Retrieval-Augmented Generation)** solves this by:
1. **Retrieval:** Find the most relevant documents for the query
2. **Augmented:** Inject those documents into the prompt
3. **Generation:** LLM generates answer grounded in retrieved documents

```
User Query: "What were our Q3 2025 revenue figures?"
     ↓
Retrieval: Search vector DB → find Q3 2025 earnings report chunks
     ↓
Augmentation: Inject document chunks into prompt
     ↓
Generation: LLM generates answer citing retrieved facts
     ↓
Response: "According to the Q3 2025 report, revenue was $2.3B, up 12% YoY"
```

### 1.2 RAG vs Fine-tuning vs Prompt Engineering

| Approach | When to Use | Pros | Cons |
|---|---|---|---|
| Prompt Engineering | Static knowledge, simple tasks | Zero cost, instant | Limited by context window |
| RAG | Large/dynamic knowledge bases | Up-to-date, scalable, explainable | Retrieval can fail, complexity |
| Fine-tuning | Specific style/behaviour, consistent format | Consistent, fast inference | Expensive, data needed, static |
| RAG + Fine-tuning | Best quality on domain tasks | Best of both worlds | Most expensive |

**Rule of thumb:**
- "I want the model to KNOW something" → RAG (inject knowledge)
- "I want the model to BEHAVE differently" → Fine-tuning
- "I want to reduce hallucination on my domain" → RAG first, fine-tune if quality insufficient

---

## Part 2 — The RAG Pipeline: Step by Step

> 📖 **Big picture:** RAG has two distinct phases that run at different times. The **indexing pipeline** runs *offline* (ahead of time) and converts your documents into searchable vectors. The **query pipeline** runs *online* (at request time) and retrieves the right chunks to answer the user's question.
>
> Understanding both pipelines — and where each can fail — is what separates someone who has used LangChain once from a production AI engineer. AI engineer system design interviews often ask you to design a RAG system end-to-end; this section is the blueprint.

### 2.1 Indexing Pipeline (Offline)

```
Documents (PDFs, Word, web pages, databases)
    ↓ Document Loading
    ↓ Preprocessing (clean, normalise)
    ↓ Chunking (split into smaller pieces)
    ↓ Embedding (convert each chunk to a vector)
    ↓ Vector Store (store vectors + metadata)
```

### 2.2 Query Pipeline (Online)

```
User Query
    ↓ Query Embedding
    ↓ Similarity Search (vector DB)
    ↓ Retrieval (top-k chunks)
    ↓ Reranking (optional — improve precision)
    ↓ Context Assembly
    ↓ LLM Generation
    ↓ Response
```

---

## Part 3 — Chunking Strategies

> 💡 **ELI5 (Explain Like I'm 5):**
> Imagine a book has 500 pages. If you scan the whole book into one giant block, a search for "Chapter 3" pulls up the entire book. The bit you care about gets drowned out by 499 pages of irrelevant noise. Instead, we chunk the book into 200-word blocks. Now a search for "Chapter 3" quickly retrieves the *exact* block of text you need without the noise. 

> 📖 **Why chunking is surprisingly important:** Chunking is how you split documents (PDFs, web pages, knowledge base articles) into pieces before embedding them. It sounds trivial but is one of the biggest quality levers in a RAG system.

### 3.1 Why Chunking Matters

Embedding models have a maximum input length (typically 512 tokens). Even if they didn't, embedding the entire document into one vector "averages" the information — the vector won't be specific enough for precise retrieval.

**The chunking dilemma:**
- Too small chunks: miss context, "the answer requires knowing the surrounding para"
- Too large chunks: diluted embeddings, injecting irrelevant content into prompts

> ⚠️ **Before/After: Chunking Quality Impact**
>
> **❌ BEFORE — 2,000-character fixed chunks, no overlap:**
> ```
> Query: "What is the recommended dosage of lisinopril for hypertension?"
>
> Retrieved chunk (2000 chars):
> "...Lisinopril is an ACE inhibitor. It works by relaxing blood vessels.
>  History: approved in 1987. Side effects include dry cough, dizziness,
>  elevated potassium. In 2003, a large trial showed... Also used for
>  heart failure. Generic versions include... Interactions with NSAIDs...
>  [dosage information buried and cut off at chunk boundary]"
>
> LLM answer: "The document discusses lisinopril but I cannot confirm the
> exact dosage." ← HALLUCINATION RISK
> ```
>
> **✅ AFTER — 512-character chunks, 50-char overlap, structure-aware:**
> ```
> Query: "What is the recommended dosage of lisinopril for hypertension?"
>
> Retrieved chunk (in the exact relevant section):
> "Dosage for hypertension: Initial dose 10mg once daily. Increase to
>  20–40mg based on response. Maximum 80mg/day. Renal impairment: start
>  at 5mg. Monitor BP at 2-4 weeks."
>
> LLM answer: "For hypertension, lisinopril is started at 10mg daily,
> titrated to 20–40mg based on blood pressure response." ← PRECISE
> ```
>
> **The fix:** Smaller chunks + overlap + structure-aware splitting so answers aren't cut at boundaries.

### 3.2 Chunking Strategies

#### Strategy 1: Fixed-Size Chunking
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,       # characters
    chunk_overlap=64,     # overlap to avoid cutting mid-sentence context
    separators=["\n\n", "\n", ". ", " ", ""]  # try these splits in order
)

chunks = splitter.split_text(document)
```

**Good for:** General purpose, quick to implement
**Bad for:** Technical documents with structured sections

#### Strategy 2: Semantic Chunking
Split based on embedding similarity between consecutive sentences.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95  # new chunk when cosine distance > 95th percentile
)
chunks = splitter.split_text(document)
```

**Good for:** Documents where topics shift gradually
**Expensive:** Requires embedding every sentence

#### Strategy 3: Document-Structure-Aware Chunking
Respect markdown headers, HTML structure, or document sections.

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers = [("#", "Section"), ("##", "Subsection")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)

# Preserves section hierarchy in metadata
chunks = splitter.split_text(markdown_doc)
# chunk.metadata = {"Section": "Introduction", "Subsection": "Background"}
```

**Good for:** Technical documentation, reports with clear structure
**Best practice:** Include heading info in chunk text AND metadata

#### Strategy 4: Agentic / Sentence-Window Chunking
Embed at sentence level, retrieve surrounding sentences for context.

```python
# Build sentence-level index
sentences = split_into_sentences(document)
sentence_embeddings = embed_model.encode(sentences)

# On retrieval: return sentence + 2 sentences before + 2 after
def get_window(sentences, idx, window=2):
    start = max(0, idx - window)
    end = min(len(sentences), idx + window + 1)
    return " ".join(sentences[start:end])
```

### 3.3 Chunking Best Practices

| Practice | Why |
|---|---|
| Always include overlap (10–15%) | Prevents answers split across chunk boundaries |
| Store original document + chunk position in metadata | Enables citing sources, full-doc retrieval |
| Experiment with chunk sizes | 256, 512, 1024 tokens — measure retrieval quality |
| Include document title/section in chunk text | Helps embedding understand context |
| Strip boilerplate (headers, footers, nav) | Noise reduces retrieval quality |

---

## Part 4 — Embeddings for RAG

> 📖 **Big picture:** The quality of your RAG system is bounded by the quality of your embeddings. If your embedding model doesn't understand that "car" and "automobile" mean the same thing, it won't retrieve the right chunks. Two key decisions:
> 1. **Which embedding model?** Larger models (OpenAI text-embedding-3-large, BAAI/bge-large) produce better embeddings but cost more. For production, benchmark on your actual data before committing.
> 2. **Asymmetric vs symmetric embeddings:** Some models are trained to embed queries and documents differently (asymmetric). This matters because a question and its answer are semantically different text types. Models like `bge-large` have separate query instruction prefixes for this.

### 4.1 Choosing an Embedding Model

```python
# Option 1: OpenAI Embeddings (cloud, strong quality)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# 3072 dimensions, ~$0.13 per million tokens

# Option 2: Open-source (self-hosted, free)
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
# 1024 dimensions, free, strong performance

# Option 3: Cohere (good multilingual)
from langchain_cohere import CohereEmbeddings
embeddings = CohereEmbeddings(model="embed-english-v3.0")
```

### 4.2 MTEB Benchmark — Which Embedding to Choose

The [MTEB (Massive Text Embedding Benchmark)](https://huggingface.co/spaces/mteb/leaderboard) ranks embedding models across 56 tasks.

**Current top performers (2026):**
| Model | MTEB Score | Dims | Speed | Cost |
|---|---|---|---|---|
| text-embedding-3-large | ~64 | 3072 | API | $$$|
| BAAI/bge-m3 | ~65 | 1024 | Self-hosted | Free |
| Cohere embed-v3 | ~64 | 1024 | API | $$ |
| all-MiniLM-L6-v2 | ~57 | 384 | Very fast | Free |
| BAAI/bge-large-en-v1.5 | ~63 | 1024 | Fast | Free |

**Interview question:** "How would you choose an embedding model for production RAG?"
> 1. Start: BAAI/bge-large or text-embedding-3-small (balance quality/cost)
> 2. Benchmark on your own domain data (MTEB scores may not reflect your specific corpus)
> 3. Consider latency: 384-dim MiniLM is 5× faster than 3072-dim large models
> 4. If multilingual: bge-m3 or Cohere multilingual

---

## Part 5 — Vector Databases

> 💡 **ELI5 (Explain Like I'm 5):**
> A regular database is like walking into a library and asking the librarian for a book using its exact ISBN number (exact match). A vector database is like telling the librarian "I want books *similar in plot and feeling* to Harry Potter". The librarian doesn't check every single book; they instantly point you to the fantasy section. It searches by *meaning*.

> 📖 **Big picture:** A vector database is a database optimised for one specific operation: "given a query vector, find the K most similar vectors in my collection."
>
> **Why you can't use a regular database:** In a normal database, you find exact matches (WHERE name = 'Alice'). Similarity search is different: you want the K vectors *closest* in meaning to your query. Checking exact distance between a query and all 10 million stored vectors would take seconds. Vector databases use clever indexes (like HNSW) to find approximate nearest neighbours in milliseconds.

### 5.1 What Is a Vector Database?

Stores high-dimensional vectors with **Approximate Nearest Neighbour (ANN)** search.

Standard databases use exact matching (WHERE id = 5). Vector DBs find the `k` vectors most similar to a query vector using cosine similarity or inner product.

### 5.2 How ANN Works (HNSW)

**HNSW (Hierarchical Navigable Small World):**
```
Layer 2 (sparse):  → → → 
Layer 1:          → → → → → →
Layer 0 (dense):  → → → → → → → → → → 

Search: Start at top layer, greedy navigate to nearest node,
        descend to next layer, repeat until Layer 0.
Result: Approximate nearest neighbours in O(log n) steps.
```

Trade-off: `ef_construction` (build quality) and `M` (connections per node) balance quality/speed.

### 5.3 Vector Database Comparison

| DB | Best For | Hosting | Unique Feature |
|---|---|---|---|
| **Pinecone** | Production SaaS | Cloud | Fully managed, serverless |
| **Weaviate** | Hybrid search | Self-hosted/Cloud | GraphQL API, modules |
| **ChromaDB** | Development/local | Local/Self-hosted | Simple Python API |
| **FAISS** | Research/batch | Self-hosted | Fastest, no persistence |
| **pgvector** | Existing Postgres | Self-hosted | SQL + vectors together |
| **Qdrant** | Production open-source | Cloud/Self | Filter + vector search |
| **Milvus** | Billion-scale | Self-hosted | Horizontal scaling |

### 5.4 Implementation: Build a RAG System

```python
# Full RAG pipeline with LangChain + ChromaDB (local dev)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# 1. Load documents
loader = PyPDFLoader("my_document.pdf")
docs = loader.load()

# 2. Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
chunks = splitter.split_documents(docs)

# 3. Embed and store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# 4. Create retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance: diversity + similarity
    search_kwargs={"k": 5, "fetch_k": 20}
)

# 5. Create RAG chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 6. Query
result = qa_chain.invoke("What are the main findings?")
print(result["result"])
print(result["source_documents"])
```

---

## Part 6 — Retrieval Strategies

> 📖 **Big picture:** Naive RAG does one thing: embed the query, find the K most similar document vectors. This works surprisingly well but fails in predictable ways:
> - A specific keyword like "Error E-4021" has no "semantic meaning" — vector search misses it
> - A vague query like "make it faster" could match irrelevant speed content
> - The top-K retrieved chunks might all be near-duplicates from the same section
>
> **Advanced retrieval strategies** fix these failure modes: hybrid search combines keyword + semantic, reranking re-orders results using a more expensive cross-encoder model, and diversity filters ensure retrieved chunks cover different aspects. Each adds latency and cost, so you need to know which to apply when.

### 6.1 Similarity Metrics

```python
# Cosine Similarity (most common for RAG)
# Measures angle between vectors — scale-invariant
cosine_sim = (A · B) / (||A|| × ||B||)
# Range: [-1, 1], 1 = identical direction, 0 = orthogonal

# Dot Product (used with normalised vectors = cosine)
dot_product = A · B

# Euclidean Distance (less common for text)
euclidean = ||A - B||
```

**Interview Q:** "When would you use inner product vs cosine similarity?"
> When embeddings are L2-normalised, inner product == cosine similarity. If not normalised, inner product is biased toward high-magnitude vectors (common/frequent terms get larger embeddings). For text RAG, always normalise and use cosine/inner product.

### 6.2 Hybrid Search

Pure semantic search can miss keyword matches ("What is the CPT code 99213?"). Pure BM25 keyword search misses synonyms ("heart attack" vs "myocardial infarction").

**Hybrid:** combine both.

```python
# Combine BM25 (keyword) + vector (semantic) with Reciprocal Rank Fusion
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Equal weight hybrid
ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
results = ensemble.invoke("CPT code 99213")
```

### 6.3 Reranking

Initial retrieval (ANN) optimises for speed, not quality. Reranking runs a more powerful cross-encoder over the top-k results.

**Stage 1: Vector search** → retrieves 20 candidates (fast, O(log n))
**Stage 2: Reranker** → scores each of the 20 against query (slower, but only 20 docs)
**Result:** Top-5 re-scored documents, much better precision.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Cohere Rerank — strong cross-encoder model
compressor = CohereRerank(model="rerank-english-v3.0", top_n=5)

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble  # or vector_retriever
)

# Or use a local cross-encoder
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, documents, top_k=5):
    pairs = [(query, doc.page_content) for doc in documents]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, documents), reverse=True)
    return [doc for _, doc in ranked[:top_k]]
```

### 6.4 Query Transformations

The user's raw query is often a poor search query. Transform it first.

```python
# Technique 1: HyDE (Hypothetical Document Embeddings)
# Generate a hypothetical answer and embed THAT instead of the query
def hyde_retrieve(query, retriever, llm):
    hypothetical = llm.invoke(f"Write a short paragraph answering: {query}")
    return retriever.invoke(hypothetical)

# Technique 2: Multi-query generation
# LLM generates 3-5 different phrasings → retrieved docs union
def multi_query_retrieve(query, retriever, llm):
    variations = llm.invoke(f"Generate 3 different phrasings of: '{query}'")
    all_docs = []
    for q in variations:
        all_docs.extend(retriever.invoke(q))
    return deduplicate(all_docs)

# Technique 3: Step-back prompting
# First ask a more general parent question, use for retrieval
def stepback_retrieve(query, retriever, llm):
    parent_question = llm.invoke(f"What is the more general question that encompasses: '{query}'")
    return retriever.invoke(parent_question)
```

---

## Part 7 — RAG Evaluation with RAGAS

> 📖 **Why evaluation is hard:** Unlike a toy chatbot, production RAG needs to be trustworthy. A response might sound confident but cite the wrong source, or answer a different question than what was asked. RAGAS gives you a systematic framework to measure four distinct failure modes:
> 1. **Did we retrieve the right chunks?** (Context recall)
> 2. **Did the retrieved chunks actually contain the answer?** (Context precision)
> 3. **Did the LLM use the retrieved context faithfully?** (Faithfulness — no hallucination)
> 4. **Did the answer actually answer the question?** (Answer relevancy)
>
> Each metric catches a different bug. You need all four. Without measurement, you don’t know which component is failing.

### 7.1 The 4 Core RAGAS Metrics

```
RAGAS Evaluation = LLM-as-a-Judge approach

Input:  question, retrieved contexts, generated answer, ground truth answer
Output: scores for each metric
```

| Metric | Measures | Formula (conceptual) |
|---|---|---|
| **Faithfulness** | Is the answer grounded in retrieved contexts? | (claims in answer supported by context) / (total claims) |
| **Answer Relevancy** | Does the answer actually address the question? | Similarity(generated questions from answer, original question) |
| **Context Precision** | Are retrieved chunks actually relevant? | (relevant chunks in top-k) / (total chunks retrieved) |
| **Context Recall** | Are all necessary facts in retrieved chunks? | (facts in GT answer covered by context) / (total GT facts) |

### 7.2 Implementation

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": ["What is the revenue for Q3 2025?"],
    "answer": ["Revenue for Q3 2025 was $2.3B"],           # RAG output
    "contexts": [["Q3 2025 report: Revenue was $2.3B, up 12% YoY..."]],  # retrieved
    "ground_truth": ["$2.3 billion"],                       # expected answer
}

dataset = Dataset.from_dict(eval_data)

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print(result)
# {'faithfulness': 0.95, 'answer_relevancy': 0.88, 
#  'context_precision': 0.90, 'context_recall': 0.85}
```

### 7.3 Building an Evaluation Dataset

**Option A: Manual golden set**
```python
# 50–200 question-answer pairs manually curated by domain experts
golden_set = [
    {"question": "...", "ground_truth": "...", "context": "..."},
    ...
]
```

**Option B: Synthetic with RAGAS testset generation**
```python
from ragas.testset import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

generator = TestsetGenerator.with_openai()
testset = generator.generate_with_langchain_docs(
    documents=chunks,
    test_size=100,
    distributions={simple: 0.5, reasoning: 0.3, multi_context: 0.2}
)
```

**Option C: Production query logging**
Log real user queries + your RAG responses + human feedback → build eval set from real usage.

### 7.4 Observability with LangSmith

```python
import os
import langsmith

# Set up LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_key"
os.environ["LANGCHAIN_PROJECT"] = "RAG-Production"

# All LangChain calls are now automatically traced
# You can see:
# - Input/output for each step
# - Token usage and costs
# - Latency per step
# - Errors and exceptions

# Add custom metadata
from langchain_core.callbacks import tracing_v2_enabled
with tracing_v2_enabled(project_name="RAG-Production"):
    result = qa_chain.invoke("Your question")
```

---

## Part 8 — Advanced RAG Patterns

> 📖 **Big picture:** Basic RAG (embed → retrieve → generate) works in demos but struggles in production with real documents. Advanced patterns address specific failure modes:
> - **Contextual compression:** Retrieved chunks often include irrelevant sentences — compress to only the relevant part before sending to LLM
> - **Parent-child retrieval:** Small chunks for precise retrieval, but send the larger surrounding context to the LLM (so the answer has full context)
> - **Query rewriting / HyDE:** Rewrite the user's query or generate a hypothetical answer before embedding — often retrieves better results
> - **Self-RAG:** The LLM decides *whether* to retrieve and *critiques* its own outputs
>
> These patterns are frequently asked in AI engineer system design rounds: "Your RAG system isn't accurate enough — what would you improve?"

### 8.1 Contextual Compression

Filter out irrelevant parts of retrieved chunks before sending to LLM.

```python
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compressed_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

### 8.2 Parent-Child Retrieval (Small-to-Big)

Index small chunks for precise retrieval, but return larger parent chunks to the LLM.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# Parent: full document sections
# Child: small sentences for precise retrieval
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=InMemoryStore(),
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=200),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000),
)
```

### 8.3 Self-RAG (Reflection)

LLM decides when retrieval is needed, evaluates retrieved documents, and can retry.

```
Query → Is retrieval needed? 
  NO → Direct generation
  YES → Retrieve → Evaluate relevance → 
    Relevant → Generate → Evaluate faithfulness → 
      Faithful → Return
      Not faithful → Retry retrieval with refined query
```

### 8.4 GraphRAG

Build a knowledge graph from documents, traverse it for retrieval.

```python
# LlamaIndex Knowledge Graph implementation
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore

graph_store = SimpleGraphStore()
index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    graph_store=graph_store,
)

query_engine = index.as_query_engine(include_text=True)
```

**When to use GraphRAG:** Documents with complex entity relationships (legal, financial, medical). Outperforms naive RAG when answers require traversing relationships.

---

## Part 9 — Interview Q&A (25 Questions)

**Q1: Design a production RAG system for a company's internal knowledge base.**
> See system design answer in ML System Design study guide (Month 6). Key components: Document ingestion pipeline (crawler → cleaner → chunker → embedder → vector DB), Query pipeline (query → embed → retrieve → rerank → generate), Evaluation (RAGAS metrics), Observability (LangSmith), Caching (semantic cache for common queries).

**Q2: What is the difference between faithfulness and answer relevancy in RAGAS?**
> - Faithfulness: Does every claim in the answer have support in the retrieved context? Measures hallucination.
> - Answer relevancy: Does the answer actually address the question asked? A faithful answer could be correct but not relevant to the question.
> Both are needed. A RAG system can score high faithfulness (every claim is in context) but low relevancy (generated summarised irrelevant context).

**Q3: Your RAG system has high context recall but low answer quality. What's wrong?**
> High context recall = all necessary documents are retrieved. Low answer quality despite this means the problem is in the generation stage: (1) LLM is ignoring retrieved context, (2) context is too long → "lost in the middle", (3) poor prompt — the LLM doesn't know to use retrieved context, (4) retrieved context is too noisy. Debug: log what's in the prompt sent to LLM.

**Q4: How would you evaluate retrieval quality beyond RAGAS metrics?**
> 1. MRR (Mean Reciprocal Rank): position of first relevant result
> 2. NDCG (Normalized Discounted Cumulative Gain): ranks relevant docs higher
> 3. Hit Rate @ k: fraction of queries where relevant doc is in top-k
> 4. Human evaluation: random sample, human judges rate retrieval relevance
> Always evaluate on your domain — MTEB/RAGAS scores on generic benchmarks may not reflect your specific use case.

**Q5: What's the difference between a vector database and a search engine like Elasticsearch?**
> Elasticsearch uses BM25 (TF-IDF term frequency) for keyword matching — excellent for exact string matches, fails for semantic similarity. Weaviate/Pinecone do vector (semantic) search — finds that "heart attack" = "myocardial infarction". Modern practice: hybrid search combining both (Weaviate's BM25+vector hybrid, Elasticsearch's kNN plugin).

**Q6: What is chunk overlap and why does it matter?**
> When splitting text, overlap means the end of chunk N is included at the start of chunk N+1. Without overlap, context-spanning answers ("...as described above, the metric is...") would be split between chunks. With 10–15% overlap, both chunks would contain the connecting text. Trade-off: increases storage and embedding costs.

**Q7: Explain HyDE and when it helps.**
> Hypothetical Document Embeddings: generate a hypothetical answer to the query, then embed that answer instead of the query. The hypothesis is closer in embedding space to actual relevant documents than the raw query (which may be brief/vague). Helps when queries are very short ("What is QLoRA?") and documents are longer and detailed.

**Q8: What causes RAG systems to fail in production?**
> 1. Poor chunking: answers need cross-chunk context
> 2. Domain vocabulary mismatch: embedding model trained on general text, doesn't understand domain jargon
> 3. Document quality: duplicate, contradictory, or outdated documents
> 4. Query type mismatch: RAG works for factoid questions, not for aggregation ("What is the total?")
> 5. Retrieval too coarse: top-k has many irrelevant chunks
> 6. LLM hallucination despite good retrieval: model doesn't cite/use context properly (fix: instruct to cite, use lower temperature)

**Q9: How would you add conversational memory to a RAG chatbot?**
> Track conversation history. Two strategies:
> (1) Append history to each query context (simple but grows prompt)
> (2) Summarise history then use standalone question rewrite: "Given history: [...], rewrite the final question as a standalone search query" — the rewritten query is then used for retrieval.
> LangChain's `ConversationalRetrievalChain` implements option 2 automatically.

**Q10: What is Maximum Marginal Relevance (MMR) retrieval?**
> MMR balances relevance and diversity. Standard retrieval returns k most similar docs (can be very similar to each other). MMR iteratively selects: the next doc that is most similar to the query AND most different from already-selected docs. Parameter λ controls the balance (λ=1 → pure similarity, λ=0 → pure diversity).

**Q11: Design a RAG evaluation pipeline.**
> 1. Build golden set (50–200 Q&A pairs with ground truth contexts)
> 2. Run RAG pipeline on golden set → collect (retrieved_docs, generated_answer)
> 3. Compute RAGAS metrics: faithfulness, answer_relevancy, context_precision, context_recall
> 4. Track over time in LangSmith/W&B
> 5. A/B test chunking strategies, embedding models, retrieval k, reranking
> 6. Production monitoring: log user feedback (thumbs up/down), flag low-confidence answers

**Q12: How does context window filling strategy affect RAG quality?**
> "Lost in the middle" (Liu et al., 2023): LLMs perform better when relevant info is at the beginning or end of context. Strategies: (1) Put most relevant chunk first, (2) Use reranking to ensure top-1 is most relevant, (3) Limit context length (don't pad with weak matches), (4) Use "assertive" prompts: "The answer is in the provided context. Find and cite it."

**Q13: What is semantic caching and how can it reduce RAG costs by 40%+?**
> Cache LLM responses not by exact query string, but by query embedding. If a new query's embedding is similar (cosine > 0.95) to a cached query, return the cached answer. Implementation: GPTCache, Redis + vector similarity. For "common question" intent (like "What is RAG?"), caching can handle 40–60% of requests. New/unique queries always bypass cache.

**Q14: When would you choose LlamaIndex over LangChain for RAG?**
> LlamaIndex: stronger focus on indexing/retrieval, better multi-document queries, built-in evaluation tools, knowledge graph support. LangChain: broader ecosystem, better for agent orchestration, more integrations, faster iteration. For production RAG with complex retrieval: LlamaIndex. For agent + RAG combination: LangChain or hybrid.

**Q15: How does re-ranking improve RAG? What model do you use?**
> Initial vector search (ANN) returns top-20 by embedding similarity. Reranker (cross-encoder) scores all 20 against query jointly — sees both query and doc at once, unlike bi-encoder. Compute-intensive but only on small candidates set. Top-5 after reranking is usually much better. Use: Cohere Rerank API (managed), `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, fast), BGE reranker (open-source, strong).

---

## Part 10 — BM25 + Semantic Hybrid Search (Deep Dive)

> 📖 **Big picture:** This is the retrieval upgrade that turns a good RAG system into a great one. Pure semantic search (embedding similarity) is excellent at understanding meaning but fails on exact text matches — a product error code like "E-4021" is just random characters semantically. BM25 (the algorithm behind Elasticsearch, Lucene) excels at exact keyword matching but can't understand synonyms.
>
> **Hybrid search runs both in parallel** and combines their scores with a weighting function (Reciprocal Rank Fusion is the standard). The result reliably outperforms either approach alone, especially on knowledge bases with a mix of conceptual questions and specific lookups.
>
> **Why you'll see this in interviews:** "How would you improve RAG retrieval quality?" → Hybrid search is the most well-understood, production-proven answer.

### 10.1 Why Hybrid Search?

Semantic search (embeddings) is great at understanding meaning but misses exact keywords. BM25 (keyword search) catches exact matches but misses synonyms. **Hybrid combines both for the best retrieval quality.**

| Query | BM25 Strength | Semantic Strength |
|---|---|---|
| "Error code E-4021" | ✅ Exact match | ❌ No semantic meaning |
| "How to make the model faster?" | ❌ Misses "optimize", "speed up" | ✅ Understands intent |
| "Python ValueError handling" | ✅ Matches "ValueError" | ✅ Matches error handling concepts |

### 10.2 BM25 Algorithm

```python
# BM25 (Best Matching 25) — TF-IDF variant with document length normalization
import math
from collections import Counter

class BM25:
    def __init__(self, documents: list[str], k1=1.5, b=0.75):
        self.k1 = k1  # Term frequency saturation
        self.b = b     # Document length normalization
        self.docs = [doc.lower().split() for doc in documents]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.N
        self.df = {}  # Document frequency
        for doc in self.docs:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1
    
    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1)
    
    def score(self, query: str, doc_idx: int) -> float:
        doc = self.docs[doc_idx]
        doc_len = len(doc)
        tf = Counter(doc)
        score = 0
        for term in query.lower().split():
            freq = tf.get(term, 0)
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += self.idf(term) * numerator / denominator
        return score
    
    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
```

### 10.3 Hybrid Search Implementation

```python
# Production hybrid search with Reciprocal Rank Fusion (RRF)
from rank_bm25 import BM25Okapi
import numpy as np

class HybridSearcher:
    def __init__(self, documents, embeddings, embedding_model):
        self.documents = documents
        self.embeddings = embeddings  # Pre-computed document embeddings
        self.embedding_model = embedding_model
        
        # BM25 index
        tokenized = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
    
    def semantic_search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        query_embedding = self.embedding_model.encode(query)
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(idx, similarities[idx]) for idx in top_indices]
    
    def keyword_search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(query.split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(idx, scores[idx]) for idx in top_indices]
    
    def hybrid_search_rrf(self, query: str, top_k: int = 5, k: int = 60,
                           semantic_weight: float = 0.7) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion — combines rankings from both methods."""
        semantic_results = self.semantic_search(query, top_k=20)
        keyword_results = self.keyword_search(query, top_k=20)
        
        rrf_scores = {}
        
        # Score semantic results
        for rank, (doc_idx, _) in enumerate(semantic_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + \
                semantic_weight * (1 / (k + rank + 1))
        
        # Score keyword results
        for rank, (doc_idx, _) in enumerate(keyword_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + \
                (1 - semantic_weight) * (1 / (k + rank + 1))
        
        # Sort by combined score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def hybrid_search_weighted(self, query: str, top_k: int = 5,
                                alpha: float = 0.7) -> list[tuple[int, float]]:
        """Weighted combination — normalize and combine scores."""
        semantic_results = dict(self.semantic_search(query, top_k=50))
        keyword_results = dict(self.keyword_search(query, top_k=50))
        
        # Min-max normalize
        def normalize(scores: dict) -> dict:
            if not scores:
                return scores
            min_s, max_s = min(scores.values()), max(scores.values())
            if max_s == min_s:
                return {k: 0.5 for k in scores}
            return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}
        
        norm_semantic = normalize(semantic_results)
        norm_keyword = normalize(keyword_results)
        
        all_docs = set(norm_semantic) | set(norm_keyword)
        combined = {}
        for doc_idx in all_docs:
            combined[doc_idx] = (
                alpha * norm_semantic.get(doc_idx, 0) +
                (1 - alpha) * norm_keyword.get(doc_idx, 0)
            )
        
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
```

### 10.4 Hybrid Search in Vector Databases

```python
# Qdrant native hybrid search
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion

client = QdrantClient("http://localhost:6333")

# Qdrant supports hybrid via prefetch + fusion
results = client.query_points(
    collection_name="documents",
    prefetch=[
        Prefetch(query=query_embedding, using="dense", limit=20),  # Semantic
        Prefetch(query=query_tokens, using="sparse", limit=20),    # BM25/sparse
    ],
    query=FusionQuery(fusion=Fusion.RRF),  # Reciprocal Rank Fusion
    limit=5
)

# Weaviate native hybrid
import weaviate

client = weaviate.Client("http://localhost:8080")
result = client.query.get("Document", ["content", "title"]).with_hybrid(
    query="How to optimize RAG retrieval?",
    alpha=0.75  # 0=keyword only, 1=vector only, 0.75=mostly semantic
).with_limit(5).do()
```

**Interview Key**: "For production RAG, always use hybrid search. RRF (Reciprocal Rank Fusion) is the safest default — it combines rankings without needing score normalization. Set semantic weight higher (0.7) for natural language queries, lower (0.3) for keyword-heavy queries like error codes."

---

## 📚 Further Resources

### Must Complete
| Resource | Link | Time |
|---|---|---|
| DeepLearning.AI: Building and Evaluating Advanced RAG | https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag | 2 hrs |
| DeepLearning.AI: Vector DBs from Embeddings to Apps | https://learn.deeplearning.ai/courses/vector-databases-embeddings-applications | 2 hrs |
| RAGAS Documentation | https://docs.ragas.io/ | 1 hr |
| LangSmith Tutorial | https://docs.smith.langchain.com/ | 1 hr |

### Books / Research Papers
- **"RAG paper"** (Lewis et al., 2020) — https://arxiv.org/abs/2005.11401
- **"Lost in the Middle"** (Liu et al., 2023) — position bias in long contexts
- **Pinecone Learning Center** — best practical guides on vector search

### Your Project This Month
Build a **Production RAG Pipeline**:
```python
# Target architecture:
Documents (PDFs) 
  → PyPDF loader 
  → RecursiveCharacterTextSplitter (512 chunks, 64 overlap)
  → BAAI/bge-large-en-v1.5 embeddings
  → Chroma vector store
  → MMR retriever (k=5)
  → Cohere reranker
  → GPT-4o-mini generator
  → RAGAS evaluation (faithfulness, answer_relevancy, context_precision)
  → LangSmith observability
  → FastAPI wrapper
  → Deploy to GCP Cloud Run
```

> ✅ **End of Month 2 Core Content.** The sections below add day-to-day work depth and production patterns.

---

## Part 9 — Day-to-Day Work: RAG in Production Engineering

### 9.1 RAG at a Retail/CPG Company (Dunnhumby Context)

```
Real use cases you'll build at work:

1. PRODUCT KNOWLEDGE BASE
   Documents: 100K+ product specs, allergen info, nutritional data
   Query: "Which products contain peanuts in the snacks category?"
   RAG retrieves product docs → LLM synthesises answer with citations

2. CUSTOMER INSIGHTS ASSISTANT
   Documents: Market research reports, panel data summaries, trend analyses
   Query: "What were the top emerging food trends in Q3 2025?"
   RAG retrieves relevant reports → LLM summarises with data points

3. INTERNAL DOCUMENTATION / ONBOARDING BOT
   Documents: Confluence pages, runbooks, architecture docs
   Query: "How do I deploy a new model to the ML platform?"
   RAG retrieves deployment guides → LLM produces step-by-step guide

4. SUPPLIER CONTRACT ASSISTANT
   Documents: 500+ supplier contracts (PDF)
   Query: "What are the penalty clauses for late delivery from Supplier X?"
   RAG retrieves contract sections → LLM extracts specific clauses

5. REGULATORY COMPLIANCE CHECKER
   Documents: GDPR regulations, data processing agreements
   Query: "Can we share customer basket data with Partner Y?"
   RAG retrieves relevant regulations → LLM provides guidance
```

### 9.2 Production RAG Architecture (What You'll Actually Build)

```
                            ┌─────────────────────────────────────┐
User Query ──► API Gateway ──► Load Balancer                      │
                            │                                     │
                            │  ┌──────────────────────────────┐   │
                            │  │ RAG Service (FastAPI)         │   │
                            │  │                              │   │
                            │  │ 1. Auth + Rate Limit         │   │
                            │  │ 2. PII Masking               │   │
                            │  │ 3. Query Rewriting           │   │
                            │  │ 4. Hybrid Retrieval          │   │ 
                            │  │    ├─ Vector (Pinecone)      │   │
                            │  │    └─ BM25 (Elasticsearch)   │   │
                            │  │ 5. Reranking (Cohere)        │   │
                            │  │ 6. Context Assembly          │   │
                            │  │ 7. LLM Generation            │   │
                            │  │ 8. Citation Extraction        │   │
                            │  │ 9. Safety Filter             │   │
                            │  │ 10. Response Logging         │   │
                            │  └──────────────────────────────┘   │
                            │                                     │
                            │  Async: Logging → Kafka → Analytics │
                            └─────────────────────────────────────┘
```

### 9.3 Complete Production RAG Service (FastAPI)

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import time
import logging

app = FastAPI(title="RAG Service")
logger = logging.getLogger("rag_service")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    rerank: bool = True
    filters: dict = {}  # e.g. {"category": "snacks", "year": 2025}

class Source(BaseModel):
    document_id: str
    chunk_text: str
    score: float
    metadata: dict

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: float
    model: str
    tokens_used: int

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    start = time.time()
    
    # Step 1: Input validation & PII masking
    clean_query = mask_pii(request.query)
    
    # Step 2: Hybrid retrieval
    vector_results = vector_store.similarity_search(
        clean_query, k=request.top_k * 2,
        filter=request.filters
    )
    bm25_results = bm25_retriever.get_relevant_documents(clean_query)
    
    # Step 3: Deduplicate and merge
    all_results = deduplicate_by_content(vector_results + bm25_results)
    
    # Step 4: Rerank if enabled
    if request.rerank and len(all_results) > request.top_k:
        all_results = reranker.rerank(clean_query, all_results, top_k=request.top_k)
    else:
        all_results = all_results[:request.top_k]
    
    # Step 5: Build context
    context = "\n\n---\n\n".join([
        f"[Source {i+1}: {doc.metadata.get('filename', 'unknown')}]\n{doc.page_content}"
        for i, doc in enumerate(all_results)
    ])
    
    # Step 6: Generate answer with citations
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {clean_query}\n\nAnswer with citations [Source N]:"}
    ]
    
    llm_response = llm_service.complete(messages, model="gpt-4o-mini")
    
    # Step 7: Log for monitoring & evaluation
    latency = (time.time() - start) * 1000
    log_query(request.query, llm_response.content, all_results, latency)
    
    return QueryResponse(
        answer=llm_response.content,
        sources=[Source(
            document_id=doc.metadata.get("id", ""),
            chunk_text=doc.page_content[:200],
            score=getattr(doc, "score", 0.0),
            metadata=doc.metadata
        ) for doc in all_results],
        latency_ms=latency,
        model=llm_response.model,
        tokens_used=llm_response.input_tokens + llm_response.output_tokens
    )
```

### 9.4 Debugging RAG Failures (The Most Common Day-to-Day Task)

```
RAG Failure Mode 1: RETRIEVAL MISS
  Symptom: "I don't have information about that" (but the doc exists!)
  Debug chain:
    1. Is the document indexed? → Check vector store for doc ID
    2. Was it chunked properly? → Check chunk contains the answer text
    3. Is the embedding good? → Compute cosine sim between query and chunk
    4. Is the filter blocking it? → Check metadata filters
    5. Is the chunk too long/short? → Adjust chunk_size

  Fix: Usually a chunking or embedding issue.
    - Try semantic chunking instead of fixed-size
    - Add BM25 hybrid search (catches keyword misses)
    - Add metadata to chunks (section headers, doc title)

RAG Failure Mode 2: HALLUCINATION DESPITE CONTEXT
  Symptom: Answer includes facts NOT in retrieved context
  Debug: 
    1. Was the system prompt strict enough?
    2. Is context too long (model ignoring middle)?
    3. Is the question ambiguous?
  
  Fix:
    - Stronger instruction: "ONLY use information from the context. If not found, say 'Not found.'"
    - Context compression: remove irrelevant chunks
    - "Lost in the middle" fix: put most relevant chunk FIRST and LAST
    
RAG Failure Mode 3: LOW-QUALITY ANSWER
  Symptom: Answer is technically correct but unhelpful/verbose
  Fix:
    - More specific system prompt with examples
    - Add format instructions ("Answer in 2-3 sentences with bullet points")
    - Increase temperature slightly (0 → 0.1) for more natural language
    - Rerank: better input → better output

RAG Failure Mode 4: SLOW RESPONSE
  Symptom: >5s latency
  Profiling:
    ├ Embedding query: ~50ms (fast)
    ├ Vector search: ~100ms (fast with ANN)  
    ├ Reranking: ~200-500ms (cross-encoder is slow)
    ├ LLM generation: ~2-5s (main bottleneck)
    └ Total: ~3-6s
  
  Fix:
    - Cache frequent queries (Redis, 40-60% hit rate for support bots)
    - Stream response (user sees tokens immediately)
    - Use faster model (gpt-4o-mini vs gpt-4o: 2x faster)
    - Reduce context size (fewer/smaller chunks)
```

### 9.5 RAG Monitoring Dashboard (What to Track)

```python
# Metrics to log and monitor in production:

METRICS = {
    # Quality metrics (sample & evaluate periodically)
    "faithfulness": "Are answers grounded in context? (RAGAS)",
    "answer_relevancy": "Does answer address the question?",
    "user_satisfaction": "Thumbs up/down from users",
    
    # Retrieval metrics
    "retrieval_recall": "% of relevant docs in top-k",
    "avg_retrieval_score": "Mean cosine similarity of top-k results",
    "empty_retrieval_rate": "% of queries with no results above threshold",
    
    # Operational metrics (track every request)
    "p50_latency_ms": "Median response time",
    "p99_latency_ms": "Tail latency",
    "error_rate": "% of failed requests",
    "tokens_per_request": "Average token usage (cost proxy)",
    "cache_hit_rate": "% of queries served from cache",
    
    # Cost metrics (aggregate daily)
    "daily_api_cost": "Total OpenAI/Anthropic API spend",
    "cost_per_query": "Average cost per query",
    "total_queries": "Volume trend",
}

# Alert thresholds:
# - p99_latency > 10s → alert
# - error_rate > 5% → alert  
# - empty_retrieval_rate > 30% → investigate (new doc types? embedding drift?)
# - daily_api_cost > budget → throttle or switch to cheaper model
```

### 9.6 Document Ingestion Pipeline (You'll Maintain This)

```python
# Production document ingestion — runs as a scheduled job or triggered by uploads

from pathlib import Path
import hashlib

class DocumentIngestionPipeline:
    def __init__(self, vector_store, embedding_model, splitter):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.splitter = splitter
        self.processed_hashes = set()  # track already-processed docs
    
    def ingest_directory(self, directory: str, file_types=["pdf", "docx", "md"]):
        """Ingest all documents in a directory."""
        results = {"success": 0, "skipped": 0, "failed": 0}
        
        for file_type in file_types:
            for file_path in Path(directory).glob(f"**/*.{file_type}"):
                try:
                    # Skip if already processed (deduplication)
                    file_hash = self._hash_file(file_path)
                    if file_hash in self.processed_hashes:
                        results["skipped"] += 1
                        continue
                    
                    # Load → chunk → embed → store
                    text = self._load_document(file_path)
                    chunks = self.splitter.split_text(text)
                    
                    # Add metadata to each chunk
                    documents = []
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            "text": chunk,
                            "metadata": {
                                "source": str(file_path),
                                "filename": file_path.name,
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                                "file_hash": file_hash,
                                "ingested_at": datetime.utcnow().isoformat()
                            }
                        })
                    
                    self.vector_store.add_documents(documents)
                    self.processed_hashes.add(file_hash)
                    results["success"] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to ingest {file_path}: {e}")
                    results["failed"] += 1
        
        return results
    
    def _hash_file(self, path):
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
```

---

> ✅ **End of Month 2.** You've now built the most important practical skill for AI engineers: production RAG. Month 3 moves to Agents.
