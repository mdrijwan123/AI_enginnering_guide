# MLOps Infrastructure — Complete Study Guide

> **Excel Curriculum Coverage**: CI/CD for ML, Experiment Tracking (MLflow, W&B), Model Registries, Monitoring, Infrastructure (Docker, K8s)
> **Interview Focus**: Production ML systems, infrastructure-as-code, reproducibility, monitoring
> **Day-to-Day**: Production AI systems require CI/CD, containerization, experiment tracking, and monitoring

---

## Table of Contents
1. [Docker for ML](#1-docker-for-ml)
2. [Kubernetes Essentials](#2-kubernetes-essentials)
3. [CI/CD for ML Pipelines](#3-cicd-for-ml-pipelines)
4. [Experiment Tracking](#4-experiment-tracking)
5. [Model Registry & Versioning](#5-model-registry--versioning)
6. [Data Versioning (DVC)](#6-data-versioning)
7. [Feature Stores](#7-feature-stores)
8. [Production Monitoring](#8-production-monitoring)
9. [Interview Questions (25 Q&As)](#9-interview-questions)
10. [Day-to-Day Work Applications](#10-day-to-day-work-applications)
11. [Resources](#11-resources)

---

## 1. Docker for ML

> 💡 **ELI5 (Explain Like I'm 5):**
> Docker is like a **standardised shipping container**. In the past, software was like shipping individual chairs and barrels — every ship had to be custom-packed, and things broke in transit ("It worked on my machine!"). Docker packs your code and all its dependencies into an identical, secure box. If it runs on your laptop, it will run exactly the same way in the cloud.

### Dockerfile for ML Applications

```dockerfile
# Multi-stage build — smaller final image
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application
COPY . .

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### GPU Docker

```dockerfile
# For GPU-based inference (vLLM, transformers)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "meta-llama/Llama-3-8B-Instruct", "--port", "8000"]
```

### Docker Compose for ML Stack

```yaml
version: "3.9"
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/mlapp
      - REDIS_URL=redis://redis:6379
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on: [postgres, redis, mlflow]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  postgres:
    image: postgres:16
    volumes: [pgdata:/var/lib/postgresql/data]
    environment:
      POSTGRES_DB: mlapp
      POSTGRES_PASSWORD: pass

  redis:
    image: redis:7-alpine

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.0
    ports: ["5000:5000"]
    command: mlflow server --host 0.0.0.0 --backend-store-uri postgresql://user:pass@postgres:5432/mlflow

  chromadb:
    image: chromadb/chroma:latest
    ports: ["8001:8000"]
    volumes: [chromadata:/chroma/chroma]

volumes:
  pgdata:
  chromadata:
```

### Key Docker Commands

```bash
# Build and run
docker build -t myml-app:v1.0 .
docker run -d -p 8000:8000 --gpus all myml-app:v1.0

# Docker Compose
docker compose up -d
docker compose logs -f api
docker compose down

# Debugging
docker exec -it <container> bash
docker stats  # Real-time resource usage
docker system prune -a  # Cleanup unused images
```

---

## 2. Kubernetes Essentials

> 💡 **ELI5 (Explain Like I'm 5):**
> If your Docker container is the shipping container, **Kubernetes is the intelligent port manager**. If your app suddenly gets a million users, Kubernetes automatically orders more ships (scaling). If a ship sinks, Kubernetes replaces it automatically (self-healing). You just tell the manager how you want the port to run, and it handles the chaos.

### Core Concepts

```yaml
# deployment.yaml — Deploy LLM API
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-api
  labels:
    app: llm-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-api
  template:
    metadata:
      labels:
        app: llm-api
    spec:
      containers:
        - name: api
          image: myregistry/llm-api:v1.0
          ports:
            - containerPort: 8000
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: api-secrets
                  key: openai-key
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2"
              nvidia.com/gpu: "1"  # GPU request
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
---
# service.yaml — Expose the deployment
apiVersion: v1
kind: Service
metadata:
  name: llm-api-service
spec:
  selector:
    app: llm-api
  ports:
    - port: 80
      targetPort: 8000
  type: LoadBalancer
---
# hpa.yaml — Auto-scaling
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

### Key K8s Commands

```bash
kubectl apply -f deployment.yaml
kubectl get pods
kubectl logs -f <pod-name>
kubectl scale deployment llm-api --replicas=5
kubectl rollout restart deployment llm-api
kubectl port-forward svc/llm-api-service 8000:80
```

---

## 3. CI/CD for ML Pipelines

### GitHub Actions for ML

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-test.txt
      
      - name: Run linting
        run: |
          ruff check .
          mypy src/
      
      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=src --cov-report=xml
      
      - name: Run integration tests
        run: pytest tests/integration/ -v
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      
      - name: Run prompt regression tests
        run: python tests/test_prompts.py
  
  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker image
        run: docker build -t myregistry/llm-api:${{ github.sha }} .
      
      - name: Push to registry
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker push myregistry/llm-api:${{ github.sha }}
  
  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/llm-api api=myregistry/llm-api:${{ github.sha }}
          kubectl rollout status deployment/llm-api
```

### ML-Specific Testing

```python
# tests/test_prompts.py — Prompt regression testing
import pytest

class TestPromptQuality:
    """Ensure prompt changes don't degrade output quality."""
    
    def test_system_prompt_follows_instructions(self):
        response = llm.generate("What is 2+2?", system_prompt=SYSTEM_PROMPT)
        assert "4" in response
        assert len(response) < 500  # Not too verbose
    
    def test_no_hallucination_on_unknown(self):
        response = llm.generate("What did XYZ Corp announce on March 32?")
        assert any(phrase in response.lower() for phrase in [
            "don't know", "not sure", "cannot find", "no information"
        ])
    
    def test_output_format_json(self):
        response = llm.generate("Extract entities", output_format="json")
        parsed = json.loads(response)  # Should not raise
        assert "entities" in parsed
    
    @pytest.mark.parametrize("question,expected_topic", [
        ("What is RAG?", "retrieval"),
        ("Explain LoRA", "fine-tuning"),
    ])
    def test_topic_relevance(self, question, expected_topic):
        response = llm.generate(question)
        assert expected_topic in response.lower()
```

---

## 4. Experiment Tracking

### MLflow

```python
import mlflow
from mlflow.tracking import MlflowClient

# Set tracking server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("rag-optimization")

# Log experiment
with mlflow.start_run(run_name="chunking-strategy-v2"):
    # Log parameters
    mlflow.log_params({
        "chunk_size": 512,
        "chunk_overlap": 50,
        "embedding_model": "text-embedding-3-small",
        "retriever": "hybrid-bm25-semantic",
        "top_k": 5,
        "llm_model": "gpt-4",
        "temperature": 0.7,
    })
    
    # Run evaluation
    results = evaluate_rag(config)
    
    # Log metrics
    mlflow.log_metrics({
        "faithfulness": results["faithfulness"],
        "answer_relevancy": results["answer_relevancy"],
        "context_precision": results["context_precision"],
        "latency_p50_ms": results["latency_p50"],
        "latency_p99_ms": results["latency_p99"],
        "cost_per_query": results["cost"],
    })
    
    # Log artifacts
    mlflow.log_artifact("prompts/system_prompt.txt")
    mlflow.log_artifact("eval_results.json")
    
    # Log the model
    mlflow.pyfunc.log_model("rag_pipeline", python_model=rag_model)

# Compare runs
client = MlflowClient()
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.faithfulness > 0.8",
    order_by=["metrics.answer_relevancy DESC"],
    max_results=10
)
```

### Weights & Biases

```python
import wandb

wandb.init(
    project="llm-fine-tuning",
    config={
        "model": "llama-3-8b",
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
        "epochs": 3,
    }
)

for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(batch)
        wandb.log({
            "train/loss": loss,
            "train/learning_rate": scheduler.get_last_lr()[0],
            "train/epoch": epoch,
        })
    
    # Evaluation
    eval_metrics = evaluate(model, eval_dataloader)
    wandb.log({
        "eval/loss": eval_metrics["loss"],
        "eval/perplexity": eval_metrics["perplexity"],
        "eval/rouge_l": eval_metrics["rouge_l"],
    })
    
    # Log sample outputs
    table = wandb.Table(columns=["prompt", "generated", "reference"])
    for sample in eval_samples:
        table.add_data(sample["prompt"], sample["generated"], sample["reference"])
    wandb.log({"eval/samples": table})

wandb.finish()
```

---

## 5. Model Registry & Versioning

```python
# MLflow Model Registry
import mlflow

# Register a model
model_uri = f"runs:/{run_id}/model"
model_version = mlflow.register_model(model_uri, "rag-pipeline-prod")

# Transition to production
client = MlflowClient()
client.transition_model_version_stage(
    name="rag-pipeline-prod",
    version=model_version.version,
    stage="Production"
)

# Load production model
model = mlflow.pyfunc.load_model("models:/rag-pipeline-prod/Production")

# Model versioning workflow
# v1: Initial RAG pipeline
# v2: Improved chunking (512→256 with overlap)
# v3: Added reranking step
# v4: Switched to hybrid retrieval
# Each version tracked with metrics, parameters, and artifacts
```

---

## 6. Data Versioning

```bash
# DVC (Data Version Control)
pip install dvc dvc-s3

# Initialize
dvc init
git add .dvc .dvcignore
git commit -m "init dvc"

# Track data files
dvc add data/training_set.jsonl
git add data/training_set.jsonl.dvc data/.gitignore
git commit -m "add training data v1"

# Remote storage
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push

# Reproduce pipeline
# dvc.yaml
# stages:
#   preprocess:
#     cmd: python preprocess.py
#     deps: [data/raw/]
#     outs: [data/processed/]
#   train:
#     cmd: python train.py
#     deps: [data/processed/, src/model.py]
#     outs: [models/model.pt]
#     metrics: [metrics.json]

dvc repro  # Reproduce full pipeline
dvc metrics show  # Show latest metrics
dvc diff  # Compare data/model versions
```

---

## 7. Feature Stores

```python
# Feast Feature Store
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo/")

# Define features
# feature_repo/features.py
from feast import Entity, FeatureView, Field
from feast.types import Float32, String

user = Entity(name="user_id", join_keys=["user_id"])

user_features = FeatureView(
    name="user_features",
    entities=[user],
    schema=[
        Field(name="avg_session_duration", dtype=Float32),
        Field(name="total_queries", dtype=Float32),
        Field(name="preferred_model", dtype=String),
    ],
    source=BigQuerySource(table="project.dataset.user_features"),
)

# Get features at inference time
features = store.get_online_features(
    features=["user_features:avg_session_duration", "user_features:total_queries"],
    entity_rows=[{"user_id": "user123"}]
).to_dict()
```

---

## 8. Production Monitoring

### Prometheus + Grafana

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
REQUEST_COUNT = Counter(
    'llm_requests_total', 
    'Total LLM API requests',
    ['model', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'llm_request_duration_seconds',
    'Request latency',
    ['model', 'endpoint'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)
TOKEN_USAGE = Counter(
    'llm_tokens_total',
    'Total tokens used',
    ['model', 'type']  # type: prompt or completion
)
ACTIVE_REQUESTS = Gauge(
    'llm_active_requests',
    'Currently processing requests'
)

# Instrument endpoints
@app.middleware("http")
async def metrics_middleware(request, call_next):
    ACTIVE_REQUESTS.inc()
    start = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start
    REQUEST_COUNT.labels(
        model="gpt-4", 
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(
        model="gpt-4",
        endpoint=request.url.path
    ).observe(duration)
    ACTIVE_REQUESTS.dec()
    
    return response

# Start metrics server on separate port
start_http_server(9090)
```

### Structured Logging

```python
import structlog
import logging

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

@app.post("/chat")
async def chat(request: ChatRequest):
    request_id = str(uuid4())
    log = logger.bind(request_id=request_id, model=request.model, user_id=user.id)
    
    log.info("chat_request_received", message_count=len(request.messages))
    
    try:
        result = await generate(request)
        log.info("chat_response_generated", 
                 tokens=result.total_tokens,
                 latency_ms=result.latency_ms,
                 cached=result.from_cache)
        return result
    except Exception as e:
        log.error("chat_request_failed", error=str(e))
        raise
```

### LLM-Specific Monitoring

```python
# Monitor for AI-specific issues
class LLMMonitor:
    def __init__(self):
        self.hallucination_rate = Gauge('llm_hallucination_rate', 'Estimated hallucination rate')
        self.toxicity_score = Histogram('llm_toxicity_score', 'Output toxicity scores')
        self.refusal_rate = Counter('llm_refusals_total', 'Model refusal count')
    
    async def check_output(self, query, response, context=None):
        # Check for hallucination (RAG)
        if context:
            grounded = await check_groundedness(response, context)
            self.hallucination_rate.set(1 - grounded)
        
        # Check toxicity
        toxicity = await classify_toxicity(response)
        self.toxicity_score.observe(toxicity)
        
        # Detect refusals
        if is_refusal(response):
            self.refusal_rate.inc()
```

---

## 9. Interview Questions

### Q1: Why is Docker important for ML?
**A**: Reproducibility. ML models depend on specific library versions, CUDA versions, and system libraries. Docker encapsulates everything: code, dependencies, and runtime. "Works on my machine" → "Works everywhere." Also enables: consistent CI/CD, easy scaling, GPU isolation, and multi-service architectures.

### Q2: Explain the difference between Docker image and container.
**A**: Image: Read-only template with code, libraries, and config (like a class). Container: Running instance of an image (like an object). One image → many containers. Images are built from Dockerfiles, stored in registries. Containers are created with `docker run` and can be started/stopped.

### Q3: What is Kubernetes and why use it for ML?
**A**: Container orchestration platform. For ML: (1) Auto-scaling based on request load. (2) Rolling deployments (zero-downtime updates). (3) GPU scheduling (assign GPUs to specific pods). (4) Health checks and self-healing (restart crashed pods). (5) Secret management. (6) Service discovery. Overkill for small projects; essential for production at scale.

### Q4: How would you set up CI/CD for an LLM application?
**A**: Pipeline: (1) Lint + type check. (2) Unit tests (mock LLM responses). (3) Integration tests (with LLM API). (4) **Prompt regression tests** (ensure prompts still work). (5) Build Docker image. (6) Push to registry. (7) Deploy to staging. (8) Run smoke tests. (9) Deploy to production with canary/gradual rollout. Key: test prompts as code.

### Q5: What is MLflow and why use it?
**A**: Open-source platform for ML lifecycle. Components: (1) Tracking: log parameters, metrics, artifacts per experiment. (2) Projects: reproducible runs with conda/docker. (3) Models: packaging format. (4) Registry: model versioning and staging. For LLM work: track prompt versions, RAG config, evaluation metrics, and model deployments.

### Q6: Compare MLflow vs Weights & Biases.
**A**: MLflow: Open-source, self-hosted, model registry, more enterprise. W&B: SaaS, better visualization, better collaboration, sweeps (hyperparameter search), tables for data exploration. Common pattern: W&B for research/experimentation, MLflow for production model registry. Both can coexist.

### Q7: What is DVC and when should you use it?
**A**: Data Version Control — Git for data. Tracks large files (datasets, models) without storing them in Git. Stores data in S3/GCS/Azure, tracks pointers in Git. Use when: datasets change over time, need to reproduce past results, team collaboration on data. Pipeline feature: define data→preprocess→train→evaluate as reproducible DAG.

### Q8: How do you monitor an ML model in production?
**A**: (1) Input monitoring: data drift, schema changes. (2) Output monitoring: prediction distribution shift, confidence scores. (3) Performance: latency p50/p99, throughput, error rates. (4) Business metrics: user satisfaction, task completion rate. (5) LLM-specific: hallucination rate, toxicity, refusals. (6) Cost: token usage, API spend per user.

### Q9: What is model drift and how do you detect it?
**A**: Model performance degrades over time because the data distribution changes. Types: (1) Data drift: input distribution changes. (2) Concept drift: relationship between input and output changes. Detection: monitor prediction distributions, track evaluation metrics on fresh data, statistical tests (KS test, PSI). Response: retrain, update prompts, or switch models.

### Q10: Explain blue-green vs canary deployments.
**A**: Blue-green: Two identical environments. Switch all traffic from old (blue) to new (green). Quick rollback. Canary: Route small % of traffic to new version. Monitor. Gradually increase if healthy. For LLM: canary is preferred — test on 5% of traffic, monitor hallucination rate and latency, then ramp up. Blue-green wastes resources (double infra).

### Q11: How would you handle secrets in a production ML system?
**A**: (1) Never in code or Git. (2) Environment variables (basic). (3) Kubernetes Secrets (encrypted at rest). (4) AWS Secrets Manager / GCP Secret Manager / HashiCorp Vault (best). (5) Rotate keys regularly. (6) Principle of least privilege (each service only accesses what it needs). (7) Audit access logs. For LLM: API keys, database credentials, encryption keys.

### Q12: What is a feature store?
**A**: Centralized repository for features used in ML models. Serves features for both training (batch) and inference (real-time). Benefits: (1) Consistency — same features in training and serving. (2) Reuse — share features across teams/models. (3) Freshness — automatic feature updates. Tools: Feast (open-source), Tecton (enterprise), Redis (simple real-time).

### Q13: How do you handle GPU resources in Kubernetes?
**A**: (1) NVIDIA device plugin for K8s (expose GPUs to pods). (2) Request GPUs in pod spec: `nvidia.com/gpu: 1`. (3) GPU scheduling: K8s assigns pods to nodes with available GPUs. (4) Multi-instance GPU (MIG) for sharing GPUs. (5) Node selectors/affinity for different GPU types (A100 vs T4). (6) Queue system for batch jobs.

### Q14: What is infrastructure as code and why does it matter for ML?
**A**: Define infrastructure (servers, networks, storage) in code files (Terraform, Pulumi). Benefits for ML: (1) Reproducibility — spin up identical environments. (2) Version control — track infra changes. (3) Automation — CI/CD deploys infrastructure. (4) Disaster recovery — recreate everything from code. (5) Cost optimization — teardown unused resources.

### Q15: How would you design a model serving system for high availability?
**A**: (1) Multiple replicas behind load balancer. (2) Health checks and auto-restart. (3) Auto-scaling (HPA based on CPU/GPU utilization or request queue depth). (4) Caching (Redis for repeated queries). (5) Circuit breaker (fallback to simpler model if primary fails). (6) Multi-region deployment. (7) Model warm-up on startup. (8) Graceful shutdown (drain in-flight requests).

### Q16: What is A/B testing for ML models?
**A**: Route different user segments to different models/configs. Measure: task completion, user satisfaction, latency, cost. For LLMs: test different prompts, temperatures, models, RAG configs. Key: ensure statistical significance (sample size calculator), control for confounders, avoid peeking at results too early. Use: chi-squared test, Bayesian A/B testing.

### Q17: How do you handle logging in a distributed ML system?
**A**: (1) Structured logging (JSON, not plaintext). (2) Correlation IDs (trace requests across services). (3) Centralized logging (ELK stack, Loki + Grafana). (4) Log levels (DEBUG for dev, INFO/WARN for prod). (5) Don't log PII or API keys. (6) Log request metadata: model, tokens, latency, user_id. (7) Searchable and alertable.

### Q18: What is GitOps?
**A**: Infrastructure and application deployment managed through Git. Changes to production happen only through Git commits. Tools: ArgoCD, Flux. Workflow: PR → review → merge → automated deploy. Benefits: audit trail, easy rollback (git revert), declarative (desired state, not imperative steps). Ideal for managing model deployments.

### Q19: How do you handle database migrations in production ML?
**A**: (1) Use migration tools (Alembic for SQLAlchemy). (2) Always forward-compatible (add columns, never remove). (3) Run migrations before deploying new code. (4) Test migrations on staging first. (5) Have rollback scripts. (6) For vector DBs: reindex in background, switch pointer when done. Avoid: schema changes that break running code.

### Q20: What is observability vs monitoring?
**A**: Monitoring: Predefined metrics and alerts (CPU usage, error rate). Tells you WHEN something is wrong. Observability: Ability to understand system state from external outputs (logs, metrics, traces). Tells you WHY something is wrong. Three pillars: metrics (Prometheus), logs (ELK), traces (Jaeger/OpenTelemetry). For LLM: add custom pillars — prompt traces, token analysis.

### Q21: How do you estimate compute costs for an LLM deployment?
**A**: Factors: (1) Model size × precision = VRAM need. (2) Expected QPS × avg tokens = throughput. (3) GPU cost/hour × utilization. (4) API costs (input tokens × rate + output tokens × rate). For self-hosted: match GPU to model (Llama-8B needs ~16GB for FP16, ~8GB for INT8). For API: track token usage per endpoint. Build cost dashboards.

### Q22: What is the sidecar pattern in K8s?
**A**: Additional container in the same pod that extends the main container's functionality. For ML: (1) Logging sidecar (collect and ship logs). (2) Monitoring sidecar (Prometheus metrics exporter). (3) Auth sidecar (handle authentication). (4) Model preloader (download model weights on startup). Benefits: separation of concerns, reusable components.

### Q23: How do you handle model rollbacks?
**A**: (1) Version all model artifacts (MLflow registry). (2) Keep previous versions deployable. (3) Canary + automatic rollback on metric degradation. (4) Blue-green switch back. (5) For prompt changes: version prompts in Git, rollback = git revert. (6) Feature flags to toggle between models. Key: always have a known-good fallback.

### Q24: What is chaos engineering for ML?
**A**: Intentionally inject failures to test system resilience. Examples: (1) Kill a model serving pod — does auto-scaling recover? (2) Inject latency in LLM API calls — does timeout handling work? (3) Corrupt input data — does validation catch it? (4) Exhaust GPU memory — does graceful degradation work? Tools: Chaos Monkey, Litmus. Build confidence in production robustness.

### Q25: How do you manage multiple environments (dev/staging/prod)?
**A**: (1) Environment-specific config files (`.env.dev`, `.env.prod`). (2) K8s namespaces (`-n staging`, `-n production`). (3) Terraform workspaces or separate state files. (4) CI/CD pipeline gates (auto-deploy to staging, manual promotion to prod). (5) Feature flags for gradual rollout. (6) Same Docker image across environments, different config.

---

## 10. Day-to-Day Work Applications

### As an AI/LLM Engineer

**Docker Every Day**: Package LLM services, RAG pipelines, and evaluation tools as containers. Consistent environments across dev/staging/prod. GPU containers for inference.

**CI/CD for Prompt Engineering**: Version prompts in Git. Automated tests catch regressions. Deploy prompt changes like code changes — with review, testing, and gradual rollout.

**Experiment Tracking**: Every RAG config change, every prompt variation, every model swap — track parameters and metrics. Compare configurations. Share results with team.

**Monitoring for Reliability**: Real-time dashboards showing latency, error rates, token costs. Alerts for anomalies. Essential for maintaining SLAs and controlling costs.

---

## 11. Resources

### Excel Curriculum Links
- MLOps Guide: https://ml-ops.org/
- MLflow Tutorial: https://mlflow.org/docs/latest/tutorials-and-examples/
- Docker for ML: https://www.youtube.com/watch?v=0H2miBK_gAk
- Kubernetes for ML: https://www.kubeflow.org/
- W&B: https://docs.wandb.ai/
- DVC: https://dvc.org/doc
- Feast: https://docs.feast.dev/
- Prometheus: https://prometheus.io/docs/
- Made with ML MLOps: https://madewithml.com/
