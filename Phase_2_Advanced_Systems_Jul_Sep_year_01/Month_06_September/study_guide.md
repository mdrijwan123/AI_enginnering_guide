# Month 6: Advanced Agent Patterns + ML System Design Framework
### Phase 2 | September 2026

---

## Week 1–2: Advanced Agentic AI Patterns

> 📖 **Big picture:** By now you know how a single LLM agent works (ReAct loop, tool calling, memory). Month 6 Week 1–2 asks: what does a *production-grade* agentic system look like at a company like Google, Meta, or Stripe? The answer involves RLHF (how models learn human preferences), constitutional AI (how models self-correct), and multi-agent orchestration patterns used in real products.
>
> **Why it matters for interviews:** AI engineer roles at top companies increasingly require you to discuss not just "can this agent do the task" but "how do we make this agent *reliably* helpful, safe, and auditable at scale?" This week builds the vocabulary and intuition for those discussions.

### RLHF Concepts (Preview — Deep dive in Month 9)

> 💡 **ELI5 (Explain Like I'm 5):**
> **Pre-training** an LLM is like teaching a dog English by making it read the entire internet — it knows words, but it's wildly unpredictable. **RLHF (Reinforcement Learning from Human Feedback)** is the dog obedience training. You ask the dog to sit. If it sits politely, you give it a treat (high reward). If it bites the furniture, you give it no treat (low reward). Over time, the model learns human preferences.

**Reinforcement Learning from Human Feedback:**
```
1. Pre-train base LLM (next-token prediction)
2. Collect human comparison data (which response is better?)
3. Train Reward Model (RM) to predict human preferences
4. Fine-tune LLM with PPO to maximise RM score
```

**DPO (Direct Preference Optimisation) — 2023:**
```
Simplification: No separate RM, no PPO
Instead: directly optimise on preference pairs (chosen, rejected)
Loss: increase probability of chosen, decrease probability of rejected
Faster, more stable than PPO, comparable quality
Used by: Zephyr, OpenHermes, many open-source instruction models
```

### Constitutional AI (Anthropic's Approach)

> 💡 **ELI5 (Explain Like I'm 5):**
> Imagine a **writer and a strict editor** inside the same brain. The writer generates a draft response. The editor (the "Constitution") checks the draft against a rulebook: "Does this instruction help someone build a bomb?" If yes, the editor rejects it, and the writer creates a new, safe draft *before* the user ever sees anything.

```python
# Constitutional AI process:
# 1. Initial response generation
# 2. Self-critique: "Does my response violate principle X?"
# 3. Revision: "Revise to fix violations"
# 4. Final response is more aligned

constitutional_principles = [
    "Do not assist with illegal activities",
    "Be honest about uncertainty",
    "Respect human autonomy and privacy",
    "Do not produce harmful content"
]

def constitutional_critique_revise(response, principle):
    critique_prompt = f"""
    Critique the following response against this principle: "{principle}"
    
    Response: {response}
    
    Identify any violations:
    """
    critique = llm.invoke(critique_prompt)
    
    revision_prompt = f"""
    Revise the response to fix the violations identified:
    
    Original: {response}
    Critique: {critique}
    
    Revised response:
    """
    return llm.invoke(revision_prompt)
```

### Advanced LangGraph Patterns

#### Parallel Execution
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class ResearchState(TypedDict):
    question: str
    web_results: list
    db_results: list
    final_answer: str

def web_research(state: ResearchState):
    results = web_tool.invoke(state["question"])
    return {"web_results": results}

def db_research(state: ResearchState):
    results = db_tool.invoke(state["question"])
    return {"db_results": results}

def synthesise(state: ResearchState):
    context = state["web_results"] + state["db_results"]
    answer = llm.invoke(f"Context: {context}\n Q: {state['question']}")
    return {"final_answer": answer}

workflow = StateGraph(ResearchState)
workflow.add_node("web", web_research)
workflow.add_node("db", db_research)
workflow.add_node("synthesise", synthesise)

# Parallel: both run simultaneously after start
workflow.set_entry_point("web")
workflow.add_edge("start", "web")
workflow.add_edge("start", "db")    # parallel branch
workflow.add_edge("web", "synthesise")
workflow.add_edge("db", "synthesise")
workflow.add_edge("synthesise", END)
```

#### Reflection Pattern
```python
def generate(state):
    draft = llm.invoke(f"Write a response for: {state['task']}")
    return {"draft": draft, "iterations": state.get("iterations", 0) + 1}

def reflect(state):
    critique = llm.invoke(f"""
    Review this draft for quality:
    {state['draft']}
    
    Rate on: accuracy, completeness, clarity (1-10 each)
    If all >= 8, say DONE. Otherwise list improvements needed.
    """)
    return {"critique": critique}

def should_revise(state):
    if "DONE" in state["critique"] or state["iterations"] >= 3:
        return END
    return "generate"  # revise

workflow.add_conditional_edges("reflect", should_revise)
```

---

## Week 3–4: ML System Design Framework

> 📖 **Big picture:** ML System Design is the interview round where you’re asked to design a complete machine learning system from scratch: "Design YouTube’s recommendation system" or "Design a fraud detection system for a payments company" or "Design a RAG system for enterprise documents."
>
> **Why it’s hard:** You have 45 minutes, a blank whiteboard, and need to demonstrate expertise in: data pipelines, model selection, training, evaluation, serving, scalability, and monitoring. Without a structured framework, you’ll ramble.
>
> **The 7-step framework below is your structure.** In an interview: state the framework out loud at the start, then systematically work through each step. This shows organised thinking and ensures you don’t miss key aspects that interviewers are looking for.

### The 7-Step Framework (Applies to Any ML System Design Question)

```
Step 1: Problem Scoping (2-3 min)
Step 2: Data (2-3 min)  
Step 3: Feature Engineering (2 min)
Step 4: Model Selection (2 min)
Step 5: Training & Evaluation (2 min)
Step 6: Serving & Infrastructure (3-4 min)
Step 7: Monitoring & Maintenance (2 min)
```

### Design 1: Build a Production RAG System

**Step 1: Scoping**
```
- Scale: 10M documents, 100K daily queries, <3s P99 latency
- Quality: >85% faithfulness, >80% answer relevancy
- Constraints: $5K/month budget, GDPR compliance
```

**Step 2: Data**
```
Sources: PDFs, Word, HTML, Confluence, Slack
Ingestion: Batch (nightly) + real-time (webhook on new docs)
Volume: 10M docs × avg 10 pages × 500 tokens/page = 50B tokens
```

**Step 3: Feature/Chunking**
```
Strategy: RecursiveCharacterTextSplitter (512 tokens, 64 overlap)
Embedding: BAAI/bge-large-en-v1.5 (local, 1024-dim)
Special handling: tables → structured format, code → language-tagged
```

**Step 4: Model**
```
Embedding model: BAAI/bge-large-en-v1.5 (free, strong)
Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (local, fast)
Generator: gpt-4o-mini (cost/quality balance for Q&A)
```

**Step 5: Evaluation**
```
Metrics: Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall (RAGAS)
Golden set: 500 Q&A pairs (200 manually curated + 300 synthetic)
A/B testing: new chunking/embedding changes tested on golden set before deployment
```

**Step 6: Serving**
```
Architecture:
  FastAPI → Ingestion Service → Chunker → Embedder → Pinecone
  FastAPI → Query Service → Query Embedder → Pinecone → Reranker → GPT-4o-mini
  
Scalability:
  - Pinecone serverless: auto-scales, no operational overhead
  - Embedding inference: batched, GPU-accelerated (T4 on GCP)
  - Generator: OpenAI API (no serving needed)
  - Semantic cache: Redis + cosine similarity (40% cache hit rate)
  
Cost optimisation:
  - Cache: saves 40% of LLM API calls = ~$2K/month savings
  - gpt-4o-mini vs gpt-4o: 10× cheaper, acceptable quality for Q&A
  - Batch embedding nightly (off-peak pricing)
```

**Step 7: Monitoring**
```
- LangSmith: trace every request
- Prometheus + Grafana: latency, error rate, cache hit rate
- Faithfulness score: sampled 1% of production traffic → LLM judge
- Alerts: P99 latency > 3s, faithfulness < 0.7, error rate > 0.5%
- Data drift: monitor query embedding distribution for topic drift
```

---

### Design 2: LLM-Powered Recommendation System

**User asks:** "Design a product recommendation system using LLMs for an e-commerce platform (Amazon scale)"

**Step 1: Scoping**
```
Scale: 500M users, 100M products, 100K requests/sec
Latency: <200ms P99 (recommendation page load)
Quality: CTR improvement > 5%, revenue attribution
```

**Step 2: Data**
```
User: click history, purchase history, search queries, ratings
Product: title, description, category, reviews, attributes
Interaction: (user, product, action, timestamp) events → Kafka → feature store
```

**Step 3: Architecture Decision — LLM Role**
```
Option A: LLM generates recommendations directly
  - Too slow (200ms target, LLM takes 1-5s)
  - Too expensive ($0.001/request × 100K req/sec = $6M/day)
  - ❌ NOT FEASIBLE at this scale

Option B: Hybrid — Traditional collaborative filtering + LLM for explanation/cold start
  - Stage 1: Matrix factorisation / two-tower model → candidate generation (fast)
  - Stage 2: LLM re-ranks top 50 candidates based on user context + product descriptions
  - LLM only invoked for personalised explanations (cached per product)
  - ✅ FEASIBLE
```

**Step 4: Model**
```
Two-tower model: UserEncoder(user_features) → user_embedding
                 ItemEncoder(item_features) → item_embedding
                 Score = dot(user_emb, item_emb)
                 
LLM role: embed product descriptions (offline, cached)
          generate explanation: "You might like X because..."
```

**Step 5: Serving**
```
Candidate generation: ANN search in Faiss/ScaNN, <50ms
Re-ranking: lightweight MLP model, <10ms
LLM explanation: pre-generated + cached in Redis, <5ms retrieval

Total: 50 + 10 + 5 = 65ms ✅ well under 200ms target
```

---

### Design 3: LLM Serving Platform (Senior AI Engineer Question)

**"Design a serving platform for running LLMs at 10,000 requests/minute"**

```
Key decisions:
1. Model: LLaMA 3 70B INT8 (140 GB → 2× A100 80GB, tensor parallel)
2. Inference: vLLM with continuous batching + PagedAttention
3. Load balancing: NGINX → multiple vLLM replicas
4. Auto-scaling: GKE with GPU node pool, scale on GPU utilisation
5. Caching:
   - Exact match: Redis hash of (model, messages, temperature)
   - Prefix caching: vLLM built-in (KV prefix reuse for common system prompts)
6. Routing:
   - Simple queries → LLaMA 3 8B (cheaper)
   - Complex queries → LLaMA 3 70B
   - Classification model decides
7. TTFT target: <2s (P99)
8. TBT target: <50ms (P99)
9. Cost: (hourly GPU cost) / (requests served per hour)

Infrastructure:
  GKE cluster:
  ├── CPU pool: API gateway, routing, caching
  └── GPU pool:
      ├── A100 node group: LLaMA 70B (2×A100 per replica)
      └── T4 node group: LLaMA 8B (1×T4 per replica)
  
Observability:
  - DCGM Exporter → Prometheus (GPU utilisation, memory)
  - vLLM metrics: batch size, queue depth, GPU utilisation
  - Custom: TTFT, TBT, token throughput
```

---

### ML System Design Interview Tips

**Tip 1: Clarify scale upfront**
> "What is the expected QPS? Number of users? Acceptable latency?" This determines whether you use managed APIs, self-hosted models, or edge devices.

**Tip 2: State trade-offs explicitly**
> Interviewers want to hear you reason through trade-offs, not the "perfect" answer. "I'd use gpt-4o-mini for cost, but would switch to gpt-4o if quality is insufficient" shows good engineering judgment.

**Tip 3: Always mention evaluation**
> Many candidates design the training/serving but forget evaluation. How will you know the system is working? What metrics matter? How do you A/B test changes?

**Tip 4: Address failure modes**
> What happens when: the LLM API goes down (fallback to cached response or smaller local model), the vector DB is slow (cache frequent queries), the embedding model gives poor results (fallback to BM25)?

**Tip 5: Cost awareness is a signal**
> Senior AI engineers are cost-conscious. Mention: "This would cost approximately $X/month at target scale" and propose optimisations. Shows you think like a senior engineer.

---

### Practice Questions (Self-study + Sunday mock)

1. Design a semantic search engine for a legal document repository
2. Design a code review assistant for GitHub PRs using RAG
3. Design a customer support chatbot that escalates to humans
4. Design an LLM-powered resume screening system
5. Design a real-time summarisation system for meeting transcripts

For each: use the 7-step framework, time yourself (30–35 minutes).

---

## 📚 Further Resources

- **"Designing Machine Learning Systems" by Chip Huyen** — Chapter 9 (Continual Learning), 10 (Infrastructure)
- **"Machine Learning System Design Interview" by Ali Amidi** — Sample design questions
- **Chip Huyen's blog** — https://huyenchip.com/blog.html (LLM System Design posts)
- **ML System Design Interview (Educative.io)**
- **DeepLearning.AI: Practical Multi AI Agents (crewAI Advanced)** — https://learn.deeplearning.ai/courses/practical-multi-ai-agents-and-advanced-use-cases-with-crewai

> **End of Phase 2 Core Content.** The sections below add practical ML System Design depth and work applications.

---

## Day-to-Day Work: ML System Design in Practice

### Real ML Systems You'll Design at Work

```
As an AI/LLM engineer, you'll design these systems in your first year:

1. RAG-BASED KNOWLEDGE SYSTEM (Your first big project)
   - Scope: internal docs, 10K documents, 100 users, <5s latency
   - Design: Pinecone + FastAPI + GPT-4o-mini + LangSmith monitoring
   - Key decisions: chunking strategy, embedding model, reranker

2. LLM GATEWAY SERVICE (Shared infrastructure)
   - Central service that all teams use to call LLMs
   - Handles: auth, rate limiting, cost tracking, fallbacks, caching
   - Key decisions: model routing, cache strategy, budget enforcement

3. CUSTOMER INSIGHTS AGENT (Business-facing AI)
   - Multi-agent system: data analyst + chart maker + report writer
   - Connects to BigQuery, generates natural language insights
   - Key decisions: which model per agent, cost per report, approval workflow

4. DATA QUALITY PIPELINE WITH AI (MLOps meets GenAI)
   - LLM validates incoming data feeds for anomalies, PII, format issues
   - Runs as batch job on new data loads
   - Key decisions: batch size, cost per record, false positive rate
```

### The 7-Step Framework Applied to a Real Work Scenario

```
SCENARIO: "Design an AI assistant for the sales team at Dunnhumby"

Step 1: SCOPING (3 min in interview, 1 week at work)
  Users: 200 sales managers
  Queries: "What were Tesco's top 10 selling products in Q3?" 
           "Compare Sainsbury's vs Tesco beverage category growth"
  Scale: 500 queries/day, <10s response time
  Data sensitivity: HIGH (retailer data, contracts)
  
Step 2: DATA
  Sources: 
    - BigQuery: sales data, category data, product data
    - Confluence: client reports, meeting notes, strategy docs
    - SharePoint: presentations, contracts (PDF)
  Data pipeline: nightly sync → chunk → embed → vector store

Step 3: FEATURE ENGINEERING (translate to RAG context)
  Chunking: Markdown-aware for reports, table-aware for data summaries
  Metadata: {client: str, quarter: str, category: str, confidentiality: str}
  Access control: filter by user's client portfolio

Step 4: MODEL SELECTION
  Retrieval: BGE-large embeddings + BM25 hybrid
  Reranking: Cohere Rerank v3
  Generation: GPT-4o-mini (cost-sensitive, 500 queries/day)
  Data queries: text-to-SQL agent for BigQuery (structured data)

Step 5: ARCHITECTURE
  ┌─────────────┐    ┌──────────────┐
  │ Streamlit UI ├───►│ FastAPI      │
  └─────────────┘    │ API Gateway  │
                     └─────┬────────┘
                           │
                     ┌─────▼────────┐    ┌─────────────┐
                     │ Router Agent │───►│ RAG Agent   │─► Vector DB
                     └─────┬────────┘    └─────────────┘
                           │
                     ┌─────▼────────┐    ┌─────────────┐
                     │ SQL Agent    │───►│ BigQuery    │
                     └──────────────┘    └─────────────┘

Step 6: SERVING
  - Cloud Run (GCP) for API service
  - Pinecone for vector store
  - Redis for query caching and rate limiting
  - Cost: ~$500/month at 500 queries/day with caching

Step 7: MONITORING
  - LangSmith traces for every request
  - Prometheus: latency, error rate, cache hit rate
  - Weekly: sample 50 queries → human eval of quality
  - Monthly: RAGAS automated eval on golden dataset
  - Alert: P99 > 15s, error > 3%, weekly quality < 7/10
```

### Additional ML System Design Questions for Practice

```
Question 4: "Design a content moderation system for user-generated text"
  Key: Multi-layer (fast classifier → LLM review for edge cases)
  Discuss: false positive rates, human review escalation, cultural sensitivity

Question 5: "Design a code review assistant integrated with GitHub"  
  Key: PR webhook → diff analysis → LLM review → comment posting
  Discuss: context window for large diffs, security review, cost per PR

Question 6: "Design a real-time fraud detection system using LLMs"
  Key: LLM too slow for real-time → use for pattern analysis offline
  Classic ML (XGBoost) for real-time → LLM for explanations and new pattern discovery

Question 7: "Design an AI-powered search engine for internal documents"
  Key: Hybrid search (BM25 + vector), query understanding, faceted search
  Discuss: indexing pipeline, stale documents, access control

Question 8: "Design a conversational analytics assistant (natural language to SQL)"
  Key: Schema understanding, SQL generation, result visualisation
  Discuss: schema injection in prompt, SQL validation, preventing DDL
```

> **End of Phase 2.** You now have advanced system design skills and deep production AI knowledge. Phase 3 goes deep on fine-tuning and optimisation.
