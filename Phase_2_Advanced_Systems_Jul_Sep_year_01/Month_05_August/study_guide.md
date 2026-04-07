# Month 5: LLM Internals Deep Dive + MCP Protocol + LLMOps
### Phase 2 | August 2026

---

## Week 1–2: LLM Internals — KV Cache, Quantisation, FlashAttention (Revisit + Deepen)

> 📖 **Why revisit this?** You covered these concepts in Month 1 Week 4. This month is about going deeper with *production implementation focus*. Month 1 was "understand the concepts". Month 5 is "calculate memory budgets, implement quantisation, profile throughput, justify architectural choices in a system design interview."
>
> **The practical stakes:** At FAANG scale, the difference between serving a 7B model at 100 requests/second vs 500 requests/second is worth millions of dollars. Every optimisation in this section — quantisation, speculative decoding, continuous batching — directly translates to cost savings and latency improvements.

### KV Cache Memory Budget Planning

```python
def kv_cache_memory_gb(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    max_batch_size: int,
    dtype_bytes: int = 2  # fp16
) -> float:
    """Calculate KV cache memory in GB."""
    # 2 = K and V
    bytes_total = (2 * num_layers * num_kv_heads * head_dim 
                   * max_seq_len * max_batch_size * dtype_bytes)
    return bytes_total / (1024**3)

# LLaMA 3 8B serving with batch=8, seq=4096
print(kv_cache_memory_gb(32, 8, 128, 4096, 8))  # ~8.6 GB
```

### INT8 Quantisation in Practice

```python
# bitsandbytes INT8 quantisation (LLM.int8())
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 8-bit loading: ~8B model fits in ~8 GB (vs 16 GB at fp16)
bnb_config_int8 = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config=bnb_config_int8,
    device_map="auto"
)

# What INT8 does internally:
# 1. Detects outlier channels (large magnitude activations)
# 2. Processes outliers in fp16 (small number of them)
# 3. Quantises remaining weights to int8: w_int8 = round(w_fp16 / scale)
# 4. scale = max(|w|) / 127
# Quality loss: ~1% on most benchmarks
```

### INT4 Quantisation (NF4 for QLoRA)

```python
bnb_config_nf4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # normal-float 4: quantile-based
    bnb_4bit_use_double_quant=True,   # quantise quantisation constants too
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config=bnb_config_nf4,
    device_map="auto"
)
# ~70B model in 4-bit ≈ 35 GB → fits on 2×A100 40GB!
```

### Speculative Decoding Implementation Concept

```python
# Small "draft" model generates k tokens quickly
# Large "target" model verifies all k tokens in parallel

def speculative_decode(target_model, draft_model, input_ids, gamma=4):
    """
    gamma: number of draft tokens to generate before verification
    Returns: verified output tokens
    """
    draft_tokens = draft_model.generate(input_ids, max_new_tokens=gamma)
    # draft_tokens shape: (batch, gamma)
    
    # Verify all gamma tokens in ONE parallel forward pass of target model
    # This is the key speedup: 1 target forward instead of gamma
    with torch.no_grad():
        target_logits = target_model(
            torch.cat([input_ids, draft_tokens], dim=1)
        ).logits
    
    accepted = []
    for i in range(gamma):
        draft_token = draft_tokens[:, i]
        target_prob = softmax(target_logits[:, len(input_ids)+i-1])
        draft_prob = softmax(draft_model_logits[:, len(input_ids)+i-1])
        
        # Accept/reject sampling
        r = torch.rand(1)
        accept_prob = min(1, target_prob[draft_token] / draft_prob[draft_token])
        if r < accept_prob:
            accepted.append(draft_token)
        else:
            # Reject: sample from corrected distribution and stop
            corrected = max(0, target_prob - draft_prob)
            accepted.append(sample(corrected))
            break
    
    return accepted
# Speedup: 2–3× for typical chat outputs (gamma=4, ~80% acceptance rate)
```

---

## Week 3: MCP Protocol (Model Context Protocol)

> 📖 **Big picture:** Imagine every AI assistant needing a custom adapter for every tool: a custom Slack integration, a custom GitHub integration, a custom database integration. That’s N×M integrations. MCP is the USB standard for AI: one protocol, one interface, and any LLM application can use any MCP-compatible tool.
>
> Released by Anthropic in November 2024 and rapidly adopted across the industry (including by OpenAI, Google, and Microsoft), MCP is becoming the standard way to give AI agents access to the real world. Understanding it deeply is a genuine differentiator in 2026 interviews.

### What Is MCP?

MCP is Anthropic's open protocol (Nov 2024) for connecting AI models to external data sources and tools in a **standardised way**.

**Problem before MCP:** Every AI application built custom integrations for each data source (Slack, GitHub, databases). N tools × M AI applications = N×M custom integrations.

**With MCP:** N tools × 1 MCP server = N integrations that work with ALL MCP-compatible AI clients.

```
Before MCP:                    After MCP:
Claude ──→ custom GitHub       Claude ──→ MCP Client ──→ MCP Server ──→ GitHub
Claude ──→ custom Slack                                             ──→ Slack
GPT ────→ custom GitHub                                            ──→ Postgres
GPT ────→ custom Slack
```

### MCP Architecture

```
┌─────────────────────────────────────────────────────┐
│                  MCP HOST                            │
│  (Claude Desktop, VS Code Copilot, custom app)       │
│                     │                               │
│  ┌─────────────────────────────────────────────┐    │
│  │              MCP CLIENT                     │    │
│  │  (manages connections to servers)           │    │
│  └──────────┬──────────┬───────────┬──────────┘    │
│             │          │           │               │
└─────────────┼──────────┼───────────┼───────────────┘
              │          │           │
     MCP Server  MCP Server   MCP Server
     (GitHub)   (Postgres)   (your GCP tools)
```

### MCP Components

**Server exposes:**
1. **Resources:** Read-only data (files, documents, database records)
2. **Tools:** Functions the AI can call (run query, send email, create file)
3. **Prompts:** Reusable prompt templates

**Transport:** JSON-RPC 2.0 over stdio (local) or HTTP+SSE (remote)

### Building an MCP Server in Python

```python
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server.stdio import stdio_server

# Create server
app = Server("gcp-tools-server")

# Tool: Query BigQuery
@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="query_bigquery",
            description="Execute a SQL query on Google BigQuery",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"},
                    "project_id": {"type": "string", "description": "GCP project ID"},
                    "max_rows": {"type": "integer", "default": 100}
                },
                "required": ["query", "project_id"]
            }
        ),
        types.Tool(
            name="list_gcs_buckets",
            description="List all GCS buckets in a GCP project",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"}
                },
                "required": ["project_id"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "query_bigquery":
        from google.cloud import bigquery
        client = bigquery.Client(project=arguments["project_id"])
        query_job = client.query(arguments["query"])
        results = query_job.result()
        rows = [dict(row) for row in results][:arguments.get("max_rows", 100)]
        return [types.TextContent(type="text", text=str(rows))]
    
    elif name == "list_gcs_buckets":
        from google.cloud import storage
        client = storage.Client(project=arguments["project_id"])
        buckets = [b.name for b in client.list_buckets()]
        return [types.TextContent(type="text", text="\n".join(buckets))]
    
    raise ValueError(f"Unknown tool: {name}")

# Resource: expose a GCS file
@app.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="gcs://my-bucket/config.json",
            name="Application Config",
            mimeType="application/json"
        )
    ]

@app.read_resource()
async def read_resource(uri: str) -> str:
    if uri.startswith("gcs://"):
        # Parse and fetch from GCS
        parts = uri.replace("gcs://", "").split("/", 1)
        bucket_name, blob_path = parts[0], parts[1]
        from google.cloud import storage
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_path)
        return blob.download_as_text()

# Run server
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream, write_stream,
            InitializationOptions(
                server_name="gcp-tools",
                server_version="1.0.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Configuring MCP in Claude Desktop

```json
// ~/.config/claude/claude_desktop_config.json
{
  "mcpServers": {
    "gcp-tools": {
      "command": "python",
      "args": ["/path/to/your/mcp_server.py"],
      "env": {
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/service-account.json"
      }
    }
  }
}
```

### MCP vs Tool Calling Comparison

| Aspect | Tool Calling (LangChain/OpenAI) | MCP |
|---|---|---|
| Standard | OpenAI function calling schema | Anthropic MCP protocol |
| Scope | Tools for one application | Tools for any MCP-compatible app |
| Transport | In-process / HTTP | stdio or HTTP+SSE |
| Ecosystem | LangChain, OpenAI | Claude Desktop, VS Code, custom |
| State | Stateless | Can maintain server-side state |
| Use when | Building custom agents | Building reusable tool servers |

---

## Week 4: LLMOps — Observability, Evaluation, Monitoring

> 📖 **Big picture:** You can’t improve what you can’t measure. A production LLM system without monitoring is flying blind. LLMOps is the practice of treating LLM pipelines like the software systems they are: with version control, testing, monitoring, alerting, and continuous improvement.
>
> **What specifically goes wrong in production LLM systems without monitoring:**
> - Prompt regressions: a prompt change that worked better in testing silently degrades quality in production
> - Cost explosions: token usage 10× higher than expected due to runaway chains
> - Latency spikes: P99 latency suddenly 5 seconds because one chain is retrying on errors
> - Quality drift: model provider updates their model, behaviour changes without notice
>
> LangSmith, RAGAS, Arize, and Weights & Biases are the tools that let you catch and fix these before users notice.

### The LLMOps Stack

```
Development → Staging → Production
     ↓              ↓          ↓
   Prompt     Integration   Real-time
   Testing      Testing     Monitoring
   (LangSmith) (RAGAS)     (Arize/W&B)
```

### LangSmith — Tracing in Production

```python
import langsmith
from langsmith import Client

client = Client()

# Programmatic evaluation
from langsmith.evaluation import evaluate, LangChainStringEvaluator

# Create dataset
dataset = client.create_dataset("rag-eval-v1")
for q, a in golden_qa_pairs:
    client.create_example(
        inputs={"question": q},
        outputs={"answer": a},
        dataset_id=dataset.id
    )

# Define evaluator
def qa_evaluator(run, example):
    # Use LLM as judge
    from langchain_openai import ChatOpenAI
    judge = ChatOpenAI(model="gpt-4o")
    verdict = judge.invoke(f"""
    Question: {example.inputs['question']}
    Expected: {example.outputs['answer']}
    Got: {run.outputs['output']}
    Score (1-5): """)
    return {"score": int(verdict.content.strip())}

# Run evaluation
results = evaluate(
    lambda inputs: rag_chain.invoke(inputs["question"]),
    data=dataset,
    evaluators=[qa_evaluator],
    experiment_prefix="rag-v2-bge-embeddings"
)
```

### Arize Phoenix — Local/OSS Alternative

```python
import phoenix as px
from phoenix.trace import LangChainInstrumentor

# Start Phoenix UI (local)
px.launch_app()

# Auto-instrument LangChain
LangChainInstrumentor().instrument()

# Now all LangChain calls are traced in Phoenix UI at localhost:6006
# Features: span viewer, token counts, latency, error rates
```

### Weights & Biases (W&B Weave)

```python
import weave
from weave import Model

weave.init("llm-production-monitoring")

@weave.op()
def rag_pipeline(question: str) -> dict:
    docs = retriever.invoke(question)
    answer = llm.invoke(f"Context: {docs}\n\nQuestion: {question}")
    return {"answer": answer, "sources": docs}

# W&B automatically tracks:
# - Inputs/outputs
# - Latency
# - Token usage  
# - Custom metrics you log

# Log custom eval metric
with weave.attributes({"faithfulness": 0.92, "relevancy": 0.88}):
    result = rag_pipeline("What is RAG?")
```

### Production Metrics to Track

```python
from dataclasses import dataclass
from prometheus_client import Counter, Histogram, Gauge

# Metrics to expose
request_count = Counter('llm_requests_total', 'Total LLM requests', ['model', 'status'])
latency = Histogram('llm_latency_seconds', 'LLM request latency', buckets=[0.1, 0.5, 1, 2, 5, 10])
active_requests = Gauge('llm_active_requests', 'Active LLM requests')
token_usage = Counter('llm_tokens_total', 'Total tokens used', ['type'])  # type: prompt/completion
cost_usd = Counter('llm_cost_usd_total', 'Total cost in USD', ['model'])

# Quality metrics (updated by eval pipeline)
faithfulness_score = Gauge('rag_faithfulness', 'Average faithfulness score (last 100 queries)')
answer_relevancy = Gauge('rag_answer_relevancy', 'Average answer relevancy (last 100 queries)')

# Alerts:
# - Latency P99 > 5s → page on-call
# - Faithfulness < 0.7 → alert AI team
# - Error rate > 1% → page
```

### Guardrails — Input/Output Safety

```python
# NeMo Guardrails
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.actions import action

# guardrails/config.yaml
"""
models:
  - type: main
    engine: openai
    model: gpt-4o-mini

rails:
  input:
    flows:
      - check input
  output:
    flows:
      - check output
"""

# guardrails/main.co (Colang language)
"""
define flow check input
  $is_harmful = execute check_harmful_input
  if $is_harmful
    bot refuse to respond

define flow check output
  $has_pii = execute check_pii
  if $has_pii
    bot redact pii
"""

@action(name="check_harmful_input")
async def check_harmful_input(context: dict) -> bool:
    # Use Lakera Guard API or NeMo NLP model
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.lakera.ai/v1/prompt_injection",
            json={"input": context["user_message"]},
            headers={"Authorization": "Bearer {LAKERA_KEY}"}
        )
        return resp.json()["results"][0]["flagged"]
```

### Interview Q&A — LLMOps

**Q1: What is LLMOps and how does it differ from MLOps?**
> MLOps manages ML model lifecycle: data pipelines, training, validation, deployment, monitoring. LLMOps adds: prompt versioning, LLM evaluation (faithfulness vs accuracy), context management, cost tracking (tokens), safety/guardrails, and the unique challenge that LLMs can "behave" correctly in testing but fail on edge cases in production due to their generative nature.

**Q2: How do you monitor LLM quality in production without ground truth?**
> Reference-free evaluation:
> 1. LLM-as-judge: use a separate model to score responses for helpfulness, safety, relevance
> 2. User signals: thumbs up/down, follow-up questions ("tell me more" = good, "that's wrong" = bad)
> 3. Retrieval quality proxies: context recall, context precision (don't require GT answer)
> 4. Semantic drift: compare response embeddings over time — sudden shifts indicate problems
> 5. Canary evaluation: small % of traffic uses a golden test set with known answers

**Q3: What is prompt versioning and why does it matter?**
> Prompts are code. They change model behaviour and must be: version controlled (git), tested before deployment, A/B testable (old vs new prompt), rollback-able. LangSmith, PromptLayer, and Langfuse provide prompt management. A prompt change can be as impactful as a code change — treat it as such.

**Q4: How do you detect LLM hallucination in production?**
> 1. Faithfulness scoring (LLM-as-judge): is every claim in response supported by retrieved context?
> 2. Hedging detector: "I'm not sure" / "As of my knowledge cutoff" (absence of hedging on uncertain topics is suspicious)
> 3. Source citation: require model to cite sources, verify citations are real
> 4. NLI (Natural Language Inference) models: does response contradict provided context?
> 5. SelfCheckGPT: generate same query multiple times; inconsistent answers indicate hallucination

---

## 📚 Further Resources

- **Anthropic MCP Documentation** — https://modelcontextprotocol.io/
- **MCP Python SDK** — https://github.com/modelcontextprotocol/python-sdk
- **LangSmith Documentation** — https://docs.smith.langchain.com/
- **Arize Phoenix** — https://phoenix.arize.com/
- **W&B Weave** — https://wandb.ai/site/weave
- **NeMo Guardrails** — https://github.com/NVIDIA/NeMo-Guardrails
- **DeepLearning.AI: Evaluating and Debugging GenAI** — https://learn.deeplearning.ai/courses/evaluating-debugging-generative-ai

> This month's project: **MCP Server** (Portfolio Project #5) — Build an MCP server that exposes your GCP resources (BigQuery, GCS, Cloud Monitoring) as tools accessible to Claude Desktop.

---

## Day-to-Day Work: MCP & LLMOps in Practice

### MCP in Your Daily Workflow

```
How MCP changes your day-to-day as an AI engineer:

BEFORE MCP:
  - Every AI app needs custom integrations for each data source
  - Claude can't access your BigQuery tables, GCS buckets, Jira tickets
  - You build bespoke REST APIs for each tool an agent needs

AFTER MCP:
  - Write ONE MCP server per data source
  - Claude Desktop, VS Code Copilot, and ANY MCP client can access it
  - New AI apps get all existing tools for free

At work, you'll build MCP servers for:
  1. BigQuery → query data, get schema info, run analytics
  2. GCS → list/read/write files, manage data pipelines
  3. Jira/Confluence → read tickets, search docs, create issues
  4. Internal APIs → expose business logic as tools
  5. Monitoring → query Prometheus, check alerts, get dashboards
```

### Building an Internal MCP Server (Step by Step)

```python
# Complete MCP server for internal data access
# This is the kind of thing you'll build in your first month

from mcp.server import Server
from mcp.types import Resource, Tool, TextContent
import mcp.server.stdio

server = Server("internal-data-server")

# Tool 1: Query your data warehouse
@server.tool()
async def query_bigquery(sql: str) -> str:
    """Execute a read-only SQL query against BigQuery.
    Only SELECT queries are allowed. No mutations.
    
    Args:
        sql: A valid BigQuery SQL SELECT query
    """
    # Security: validate query is read-only
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return "Error: Only SELECT queries are allowed"
    
    if any(keyword in sql_upper for keyword in ["DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER"]):
        return "Error: Mutation queries are not allowed"
    
    from google.cloud import bigquery
    client = bigquery.Client()
    query_job = client.query(sql)
    results = query_job.result()
    
    # Format as markdown table (Claude loves markdown)
    rows = [dict(row) for row in results]
    if not rows:
        return "Query returned no results."
    
    headers = list(rows[0].keys())
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows[:50]:  # limit to 50 rows
        table += "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n"
    
    return f"Query returned {len(rows)} rows (showing first 50):\n\n{table}"

# Tool 2: Get table schema
@server.tool()
async def get_table_schema(dataset: str, table: str) -> str:
    """Get the schema of a BigQuery table."""
    from google.cloud import bigquery
    client = bigquery.Client()
    table_ref = client.get_table(f"{dataset}.{table}")
    
    schema_info = f"Table: {dataset}.{table}\n"
    schema_info += f"Rows: {table_ref.num_rows:,}\n"
    schema_info += f"Size: {table_ref.num_bytes / 1e9:.2f} GB\n\n"
    schema_info += "Columns:\n"
    for field in table_ref.schema:
        schema_info += f"  - {field.name}: {field.field_type} ({field.mode})\n"
    
    return schema_info

# Resource: expose documentation as context
@server.resource("docs://runbooks/{topic}")
async def get_runbook(topic: str) -> str:
    """Expose internal runbooks as MCP resources."""
    runbook_path = f"/docs/runbooks/{topic}.md"
    with open(runbook_path) as f:
        return f.read()

# Run the server
async def main():
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(read, write)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### LLMOps Runbooks for Production

```
Runbook 1: MODEL DEGRADATION DETECTED
  Alert: Faithfulness score dropped below 0.7
  
  Steps:
  1. Check if embedding model changed or was updated → compare vector distributions
  2. Check if new documents were ingested → look for corrupted/malformed docs
  3. Check if prompt template changed → diff recent deployments
  4. Check if underlying LLM API changed (version update) → check release notes
  5. Run RAGAS eval on golden dataset → identify which metric dropped
  6. If retrieval issue → check vector DB health, index freshness
  7. If generation issue → try different temperature, check system prompt
  
Runbook 2: LATENCY SPIKE
  Alert: P99 > 10 seconds
  
  Steps:
  1. Check LLM API status page (OpenAI/Anthropic status)
  2. Check if context length increased (longer prompts = slower)
  3. Check concurrent request count (rate limiting?)
  4. Check reranker performance (cross-encoder can be slow)
  5. Check vector DB query time (index size growth?)
  6. If transient: API provider issue → wait or failover
  7. If persistent: optimize (smaller context, faster model, caching)

Runbook 3: COST SPIKE
  Alert: Daily spend > 2× historical average
  
  Steps:
  1. Check request volume (organic growth or anomaly?)
  2. Check average tokens/request (context growing?)
  3. Check for infinite loops in agents
  4. Check if expensive model being used where cheap one suffices
  5. Immediate: enable rate limiting, switch to cheaper model for low-priority
  6. Long-term: implement semantic caching, optimize prompts
```

### CI/CD for LLM Applications

```yaml
# .github/workflows/llm-deploy.yml
name: LLM Application Deployment

on:
  push:
    branches: [main]
    paths: ['src/**', 'prompts/**', 'configs/**']

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run unit tests
        run: pytest tests/unit/ -v
      
      - name: Run prompt regression tests
        run: |
          # Compare outputs against golden dataset
          python scripts/eval_prompts.py \
            --dataset golden_eval_set.json \
            --threshold-faithfulness 0.8 \
            --threshold-relevancy 0.75
      
      - name: Cost estimation
        run: |
          # Estimate deployment cost before shipping
          python scripts/estimate_cost.py \
            --model gpt-4o-mini \
            --avg-requests-per-day 50000 \
            --avg-context-tokens 1500
  
  deploy-staging:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: gcloud run deploy rag-service-staging --image $IMAGE
      
      - name: Smoke test staging
        run: python scripts/smoke_test.py --url https://staging.api/query
  
  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production  # requires manual approval
    steps:
      - name: Canary deploy (10% traffic)
        run: |
          gcloud run services update-traffic rag-service \
            --to-revisions LATEST=10
      
      - name: Monitor canary (5 min)
        run: python scripts/monitor_canary.py --duration 300
      
      - name: Full rollout
        run: |
          gcloud run services update-traffic rag-service \
            --to-revisions LATEST=100
```
