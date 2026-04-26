# LLMOps & Observability — Production AI Monitoring
### Phase 2 | August 2026 | Week 4

> **What is LLMOps?** LLMOps is MLOps adapted for LLM-powered applications. Traditional MLOps focuses on model training pipelines and data drift. LLMOps adds: prompt versioning, LLM-as-judge evaluation, token cost management, latency monitoring, and hallucination tracking. The challenge: LLMs produce unstructured text — traditional metrics (accuracy, F1) don't apply.

> 💡 **ELI5 (Explain Like I'm 5):**
> Running an LLM in production is like managing a brilliant but expensive contractor who never gives the same answer twice. LLMOps is the management system: track what they say (logging), rate how good the answers are (evaluation), watch how long they take and how much they cost (monitoring), and make sure the contract terms don't change on you (prompt versioning).

---

## Part 1 — The LLMOps Stack

```
┌─────────────────────────────────────────────────────────────┐
│                   LLMOps Architecture                        │
│                                                               │
│  ┌────────────┐   ┌────────────┐   ┌────────────────────┐   │
│  │  Prompts   │   │  LLM App   │   │  Monitoring        │   │
│  │  Versions  │──►│  FastAPI   │──►│  LangSmith         │   │
│  │  Git/DSPy  │   │  LangChain │   │  Prometheus/Grafana│   │
│  └────────────┘   └─────┬──────┘   └────────────────────┘   │
│                         │                                     │
│              ┌──────────▼──────────┐                         │
│              │  Evaluation Suite   │                         │
│              │  RAGAS / LLM-Judge  │                         │
│              └─────────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

**Key LLMOps tools:**
| Layer | Tools |
|---|---|
| Tracing | LangSmith, Arize Phoenix, PromptLayer, Helicone |
| Evaluation | RAGAS, DeepEval, Phoenix Evals, Promptfoo |
| Monitoring | Prometheus + Grafana, Datadog LLM Observability |
| Prompt management | LangSmith, PromptLayer, DSPy, Humanloop |
| Cost tracking | LangSmith, Helicone, custom token counting |

---

## Part 2 — LangSmith: Tracing & Observability

### 2.1 Setting Up LangSmith Tracing
```python
import os
from langsmith import Client
from langsmith.wrappers import wrap_openai
import openai

# Configure LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "production-rag-v2"

# Wrap OpenAI client — all calls automatically traced
client = wrap_openai(openai.OpenAI())

# Every call now traced with inputs, outputs, latency, tokens
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain transformers"}]
)
```

### 2.2 Custom Run Tracing with @traceable
```python
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

@traceable(name="RAG Pipeline", run_type="chain")
def run_rag(query: str, docs: list[str]) -> str:
    context = "\n".join(docs)
    
    prompt = ChatPromptTemplate.from_template(
        "Answer based on context:\nContext: {context}\nQuestion: {query}"
    )
    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm
    
    return chain.invoke({"context": context, "query": query}).content

@traceable(name="Retrieval", run_type="retriever")
def retrieve(query: str) -> list[str]:
    # Your vector DB retrieval here
    return ["doc1 content", "doc2 content"]

@traceable(name="Full Pipeline")
def answer_question(query: str) -> str:
    docs = retrieve(query)
    return run_rag(query, docs)

# This creates a nested trace: Full Pipeline → Retrieval + RAG Pipeline
result = answer_question("What is RAG?")
```

### 2.3 Adding Custom Metadata and Feedback
```python
from langsmith import Client
from langsmith.schemas import RunTypeEnum
import uuid

ls_client = Client()

# Log a run with custom metadata
run_id = str(uuid.uuid4())
ls_client.create_run(
    id=run_id,
    name="Production RAG",
    run_type=RunTypeEnum.chain,
    inputs={"query": "What is attention?"},
    tags=["production", "v2.1"],
    extra={
        "metadata": {
            "user_id": "user_123",
            "session_id": "sess_456",
            "model": "gpt-4o-mini",
        }
    }
)

# Attach feedback (e.g., from user thumbs up/down)
ls_client.create_feedback(
    run_id=run_id,
    key="user_rating",
    score=1,  # 1 = positive, 0 = negative
    comment="Answer was helpful and accurate"
)

# Automated LLM-as-judge feedback
ls_client.create_feedback(
    run_id=run_id,
    key="faithfulness",
    score=0.87,  # RAGAS score
    source_info={"evaluator": "ragas-v0.1"}
)
```

---

## Part 3 — Production Metrics & Prometheus Monitoring

### 3.1 Core LLM Application Metrics
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import functools

# Define metrics
REQUEST_COUNT = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['model', 'status', 'endpoint']
)
REQUEST_LATENCY = Histogram(
    'llm_request_latency_seconds',
    'LLM request latency',
    ['model', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)
TOKEN_COUNTER = Counter(
    'llm_tokens_total',
    'Total tokens used',
    ['model', 'type']  # type: input | output
)
ACTIVE_REQUESTS = Gauge(
    'llm_active_requests',
    'Currently active LLM requests',
    ['endpoint']
)
COST_COUNTER = Counter(
    'llm_cost_dollars_total',
    'Estimated LLM cost in USD',
    ['model']
)

# Token cost table (USD per 1M tokens, April 2026)
TOKEN_COSTS = {
    "gpt-4o":          {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":     {"input": 0.15,  "output": 0.60},
    "claude-opus-4-5": {"input": 3.00,  "output": 15.00},
    "claude-haiku-3":  {"input": 0.25,  "output": 1.25},
}

def track_llm_call(model: str, endpoint: str):
    """Decorator to track LLM metrics automatically."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            ACTIVE_REQUESTS.labels(endpoint=endpoint).inc()
            start = time.time()
            status = "success"
            try:
                result = await func(*args, **kwargs)
                # Extract token usage from result
                if hasattr(result, 'usage'):
                    in_tok = result.usage.prompt_tokens
                    out_tok = result.usage.completion_tokens
                    TOKEN_COUNTER.labels(model=model, type="input").inc(in_tok)
                    TOKEN_COUNTER.labels(model=model, type="output").inc(out_tok)
                    # Cost tracking
                    if model in TOKEN_COSTS:
                        cost = (in_tok * TOKEN_COSTS[model]["input"] +
                                out_tok * TOKEN_COSTS[model]["output"]) / 1_000_000
                        COST_COUNTER.labels(model=model).inc(cost)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                REQUEST_COUNT.labels(model=model, status=status, endpoint=endpoint).inc()
                REQUEST_LATENCY.labels(model=model, endpoint=endpoint).observe(
                    time.time() - start
                )
                ACTIVE_REQUESTS.labels(endpoint=endpoint).dec()
        return wrapper
    return decorator

# Usage:
@track_llm_call(model="gpt-4o-mini", endpoint="/api/chat")
async def call_llm(messages: list) -> dict:
    import openai
    client = openai.AsyncOpenAI()
    return await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
```

### 3.2 FastAPI + Prometheus Integration
```python
from fastapi import FastAPI, Request
from prometheus_client import make_asgi_app
import time

app = FastAPI()

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    REQUEST_LATENCY.labels(
        model="app",
        endpoint=request.url.path
    ).observe(duration)
    
    return response

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.1.0"}
```

### 3.3 Grafana Dashboard Alerts (YAML config)
```yaml
# grafana-alerts.yaml
groups:
  - name: llm-production-alerts
    rules:
      - alert: HighLLMLatency
        expr: histogram_quantile(0.99, llm_request_latency_seconds_bucket) > 5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "P99 LLM latency > 5s for 2 minutes"
      
      - alert: HighErrorRate
        expr: rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m]) > 0.05
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "LLM error rate > 5%"
      
      - alert: DailyBudgetExceeded
        expr: increase(llm_cost_dollars_total[24h]) > 100
        labels:
          severity: critical
        annotations:
          summary: "LLM cost exceeded $100 in last 24 hours"
```

---

## Part 4 — Prompt Versioning & Management

### 4.1 Git-based Prompt Versioning
```
prompts/
├── rag_system.v1.0.txt         # initial prompt
├── rag_system.v1.1.txt         # improved retrieval instruction  
├── rag_system.v2.0.txt         # major rewrite
├── rag_system.prod.txt         # symlink → current production prompt
└── rag_system.staging.txt      # symlink → candidate prompt
```

```python
# prompt_loader.py
from pathlib import Path
import hashlib

class PromptManager:
    def __init__(self, prompt_dir: str = "prompts"):
        self.prompt_dir = Path(prompt_dir)
    
    def load(self, name: str, version: str = "prod") -> str:
        path = self.prompt_dir / f"{name}.{version}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")
        
        content = path.read_text()
        # Log which prompt version is in use
        checksum = hashlib.md5(content.encode()).hexdigest()[:8]
        print(f"Loaded prompt: {name}@{version} [{checksum}]")
        return content
    
    def promote_to_prod(self, name: str, version: str):
        """After A/B test passes, promote version to production."""
        src = self.prompt_dir / f"{name}.{version}.txt"
        dst = self.prompt_dir / f"{name}.prod.txt"
        dst.write_text(src.read_text())
        print(f"Promoted {name}@{version} → prod")

pm = PromptManager()
system_prompt = pm.load("rag_system")  # loads prod version
```

### 4.2 LangSmith Prompt Hub
```python
from langsmith import Client
from langchain import hub

# Pull prompt from LangSmith hub (versioned)
prompt = hub.pull("my-org/rag-system:v2.1")

# Push new version
from langchain_core.prompts import ChatPromptTemplate

new_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI assistant. Answer only from context."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])
hub.push("my-org/rag-system", new_prompt, new_repo_is_public=False)
```

---

## Part 5 — Hallucination & Quality Monitoring

### 5.1 Sampling-based Production Evaluation
```python
import random
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

class ProductionEvaluator:
    def __init__(self, sample_rate: float = 0.01):
        self.sample_rate = sample_rate  # evaluate 1% of traffic
        self.buffer = []
    
    async def log_interaction(
        self,
        question: str,
        context: list[str],
        answer: str
    ):
        """Log for potential evaluation."""
        if random.random() < self.sample_rate:
            self.buffer.append({
                "question": question,
                "contexts": context,
                "answer": answer,
                "ground_truth": ""  # no ground truth in production
            })
            
            if len(self.buffer) >= 50:  # evaluate in batches
                await self._run_evaluation()
    
    async def _run_evaluation(self):
        dataset = Dataset.from_list(self.buffer)
        results = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy]
        )
        
        avg_faith = results["faithfulness"]
        avg_rel = results["answer_relevancy"]
        
        # Push to Prometheus
        QUALITY_GAUGE.labels(metric="faithfulness").set(avg_faith)
        QUALITY_GAUGE.labels(metric="answer_relevancy").set(avg_rel)
        
        # Alert if quality drops
        if avg_faith < 0.7:
            await send_alert(f"🚨 Faithfulness dropped to {avg_faith:.2f}")
        
        self.buffer = []
        print(f"Evaluated {len(dataset)} samples: faith={avg_faith:.2f}, rel={avg_rel:.2f}")

evaluator = ProductionEvaluator(sample_rate=0.02)
```

### 5.2 Data Drift Detection
```python
import numpy as np
from scipy.stats import ks_2samp

class EmbeddingDriftDetector:
    """Detect when input distribution shifts from training distribution."""
    
    def __init__(self, baseline_embeddings: np.ndarray):
        self.baseline = baseline_embeddings
        self.production_buffer = []
    
    def add_embedding(self, embedding: np.ndarray):
        self.production_buffer.append(embedding)
    
    def check_drift(self, threshold: float = 0.05) -> dict:
        if len(self.production_buffer) < 100:
            return {"drift_detected": False, "reason": "Insufficient data"}
        
        prod = np.array(self.production_buffer)
        
        # KS test per dimension (simplified: use first principal component)
        # In production: use Maximum Mean Discrepancy or cosine similarity distribution
        stat, p_value = ks_2samp(
            self.baseline.mean(axis=1),
            prod.mean(axis=1)
        )
        
        drift_detected = p_value < threshold
        if drift_detected:
            print(f"🚨 Input drift detected! KS stat={stat:.3f}, p={p_value:.4f}")
        
        return {
            "drift_detected": drift_detected,
            "ks_statistic": stat,
            "p_value": p_value
        }
```

---

## Part 6 — Cost Management

### 6.1 Token Budget Enforcement
```python
import tiktoken

class TokenBudgetManager:
    def __init__(self, daily_budget_usd: float = 50.0):
        self.daily_budget = daily_budget_usd
        self.today_spend = 0.0
        self.enc = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        costs = TOKEN_COSTS.get(model, {"input": 0, "output": 0})
        return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000
    
    def check_budget(self, estimated_cost: float) -> bool:
        if self.today_spend + estimated_cost > self.daily_budget:
            raise BudgetExceededError(
                f"Daily budget ${self.daily_budget} would be exceeded. "
                f"Spent: ${self.today_spend:.2f}"
            )
        return True
    
    def record_spend(self, actual_cost: float):
        self.today_spend += actual_cost

class BudgetExceededError(Exception):
    pass

# Model routing based on query complexity
def route_to_model(query: str) -> str:
    """Route simple queries to cheap model, complex to expensive."""
    words = query.split()
    
    # Heuristics (in production, use a classifier)
    if len(words) < 10 and "?" in query:
        return "gpt-4o-mini"      # simple factual query
    elif any(kw in query.lower() for kw in ["analyse", "compare", "design", "explain"]):
        return "claude-opus-4-5"  # complex reasoning
    else:
        return "gpt-4o-mini"      # default to cheap
```

---

## Part 7 — Complete LLMOps FastAPI Application

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import structlog

log = structlog.get_logger()

app = FastAPI(title="LLM API with Full Observability")

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float
    tokens_used: int
    cost_usd: float

budget_manager = TokenBudgetManager(daily_budget_usd=100.0)
prod_evaluator = ProductionEvaluator(sample_rate=0.01)

@app.post("/api/query", response_model=QueryResponse)
@track_llm_call(model="gpt-4o-mini", endpoint="/api/query")
async def query_endpoint(request: QueryRequest):
    start = time.time()
    
    log.info("query_received", query=request.query[:100], session=request.session_id)
    
    # 1. Estimate cost before calling
    input_tokens = budget_manager.count_tokens(request.query)
    estimated_cost = budget_manager.estimate_cost(input_tokens, 500, "gpt-4o-mini")
    
    try:
        budget_manager.check_budget(estimated_cost)
    except BudgetExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))
    
    # 2. Retrieve context
    docs = await retrieve_docs(request.query)
    
    # 3. Generate answer
    answer, usage = await generate_answer(request.query, docs)
    
    # 4. Record actual cost
    actual_cost = budget_manager.estimate_cost(
        usage["input_tokens"], usage["output_tokens"], "gpt-4o-mini"
    )
    budget_manager.record_spend(actual_cost)
    
    latency = (time.time() - start) * 1000
    log.info("query_completed", latency_ms=latency, cost_usd=actual_cost)
    
    # 5. Background evaluation
    await prod_evaluator.log_interaction(request.query, docs, answer)
    
    return QueryResponse(
        answer=answer,
        sources=docs[:3],
        latency_ms=latency,
        tokens_used=usage["input_tokens"] + usage["output_tokens"],
        cost_usd=actual_cost
    )
```

---

## Structured Logging Best Practices

```python
import structlog
import logging
import json
import sys

# Configure structured JSON logging (machine-readable)
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer()  # swap for JSONRenderer in prod
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

log = structlog.get_logger()

# Always include trace context
log = log.bind(
    service="rag-api",
    version="2.1.0",
    environment="production"
)

# Use structured fields, not string formatting
log.info("request_processed",
    query_length=len(query),
    retrieved_docs=len(docs),
    latency_p99_ms=234,
    model="gpt-4o-mini",
    session_id="sess_123"
)
# Output: {"level":"info","event":"request_processed","query_length":50,"latency_p99_ms":234,...}
```

---

## Interview Q&A

### Q1: How do you monitor an LLM application in production?
**A:** Three layers: (1) **Infrastructure metrics** — latency (TTFT, TBT, total), error rate, active connections, GPU utilisation (Prometheus + Grafana). (2) **LLM-specific metrics** — token usage, cost per request, cache hit rate, prompt/completion ratio (custom counters). (3) **Quality metrics** — faithfulness, answer relevancy, hallucination rate (RAGAS sampling on 1-5% of traffic, LLM-as-judge). Alert on P99 latency > threshold, error rate > 1%, daily cost > budget, faithfulness < 0.7.

### Q2: What is prompt versioning and why does it matter?
**A:** Prompts are as important as code — changing a prompt can dramatically affect output quality. Prompt versioning means tracking every prompt change (git, LangSmith Hub, or custom system) so you can: A/B test new prompts against old, roll back when quality drops, audit what prompt produced a problematic output, and reproduce any historical response.

### Q3: How do you handle LLM cost spiralling in production?
**A:** (1) **Token counting** before every call with tiktoken. (2) **Budget enforcement** — daily/monthly caps per user/team with hard cutoffs. (3) **Model routing** — classify query complexity, route simple queries to cheap models (GPT-4o-mini, Haiku). (4) **Caching** — exact match cache (Redis) saves 30-40% of calls. (5) **Prompt compression** — remove unnecessary context, use summarisation for long histories.

### Q4: What is data drift in the context of LLMs?
**A:** When the distribution of user queries in production significantly differs from what the system was tested on. Detected by monitoring embedding distributions (KS test, cosine similarity distribution shifts). Causes: seasonal changes, new user segments, changes in product that attract different queries. Response: trigger re-evaluation on golden dataset, consider prompt updates or fine-tuning.

### Q5: How do you evaluate LLM quality in production without ground truth?
**A:** (1) **LLM-as-Judge** — use a stronger model to rate outputs on faithfulness, relevance, coherence (G-Eval framework). (2) **RAGAS without GT** — faithfulness and answer_relevancy don't require ground truth. (3) **User signals** — thumbs up/down, follow-up questions, session abandonment rate. (4) **Reference-free metrics** — perplexity of outputs, repetition ratio, refusal rate.

---

## Further Resources

- **LangSmith Docs** — https://docs.smith.langchain.com/
- **Arize Phoenix** — https://px.arize.com/ (open-source LLM observability)
- **RAGAS Docs** — https://docs.ragas.io/
- **Chip Huyen — LLM monitoring** — https://huyenchip.com/2023/04/11/llm-engineering.html
- **"Designing Machine Learning Systems" Ch 8** — Data distribution shifts
