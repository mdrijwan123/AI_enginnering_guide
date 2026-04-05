# FastAPI & Streamlit for AI Applications — Complete Study Guide

> **Excel Curriculum Coverage**: FastAPI (Introduction, Path Operations, Request/Response, Dependencies, Database, Deployment) + Streamlit (Building Apps, Components, Deployment)
> **Interview Focus**: Building and serving AI applications — REST APIs for model serving, UIs for demos
> **Day-to-Day**: Every production LLM needs an API (FastAPI) and every demo needs a UI (Streamlit)

---

## Table of Contents
1. [FastAPI Fundamentals](#1-fastapi-fundamentals)
2. [Path Operations & Routing](#2-path-operations--routing)
3. [Request/Response Models (Pydantic)](#3-requestresponse-models)
4. [Dependencies & Middleware](#4-dependencies--middleware)
5. [Database Integration](#5-database-integration)
6. [Streaming & WebSockets](#6-streaming--websockets)
7. [LLM API Patterns](#7-llm-api-patterns)
8. [Deployment](#8-deployment)
9. [Streamlit for AI Demos](#9-streamlit-for-ai-demos)
10. [Interview Questions (25 Q&As)](#10-interview-questions)
11. [Day-to-Day Work Applications](#11-day-to-day-work-applications)
12. [Resources](#12-resources)

---

## 1. FastAPI Fundamentals

### Why FastAPI for AI?
- **Async support**: Handle 1000s of concurrent LLM API calls
- **Pydantic integration**: Automatic request validation and structured output
- **Auto-generated docs**: Swagger UI at `/docs`
- **Type-safe**: Catches bugs before production
- **FastAPI + Uvicorn**: Production-grade ASGI server

### Minimal App

```python
from fastapi import FastAPI

app = FastAPI(title="LLM API", version="1.0.0")

@app.get("/")
def read_root():
    return {"status": "healthy", "model": "gpt-4"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Run: uvicorn main:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

---

## 2. Path Operations & Routing

```python
from fastapi import FastAPI, HTTPException, Query, Path
from typing import Optional

app = FastAPI()

# --- GET with path parameters ---
@app.get("/models/{model_id}")
async def get_model(model_id: str = Path(..., description="The model identifier")):
    models = {"gpt-4": {"name": "GPT-4", "provider": "OpenAI"}}
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return models[model_id]

# --- GET with query parameters ---
@app.get("/search")
async def search_documents(
    query: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=5, ge=1, le=100),
    threshold: float = Query(default=0.7, ge=0, le=1),
):
    # RAG retrieval
    results = retriever.search(query, top_k=top_k, threshold=threshold)
    return {"query": query, "results": results}

# --- POST ---
@app.post("/chat", status_code=200)
async def chat(request: ChatRequest):
    response = await llm.generate(request.messages)
    return {"response": response}

# --- PUT ---
@app.put("/models/{model_id}/config")
async def update_model_config(model_id: str, config: ModelConfig):
    return {"model_id": model_id, "config": config.dict()}

# --- DELETE ---
@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    return {"deleted": session_id}

# --- Router for organization ---
from fastapi import APIRouter

chat_router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])
admin_router = APIRouter(prefix="/api/v1/admin", tags=["Admin"])

@chat_router.post("/completions")
async def create_completion(request: ChatRequest):
    return {"response": "Hello!"}

app.include_router(chat_router)
app.include_router(admin_router)
```

---

## 3. Request/Response Models

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"

class Message(BaseModel):
    role: Role
    content: str = Field(..., min_length=1, max_length=100000)

class ChatRequest(BaseModel):
    model: str = Field(default="gpt-4", description="Model to use")
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=128000)
    stream: bool = Field(default=False)
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello!"}],
                "temperature": 0.7
            }]
        }
    }

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    id: str
    model: str
    content: str
    usage: Usage
    created_at: float

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int

@app.post("/chat", response_model=ChatResponse, responses={400: {"model": ErrorResponse}})
async def chat_endpoint(request: ChatRequest):
    try:
        result = await generate_response(request)
        return ChatResponse(
            id=f"chat-{uuid4()}",
            model=request.model,
            content=result.content,
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens
            ),
            created_at=time.time()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

---

## 4. Dependencies & Middleware

```python
from fastapi import Depends, Request
from fastapi.middleware.cors import CORSMiddleware
import time

# --- Dependency Injection ---
async def get_llm_client():
    """Reusable dependency for LLM client."""
    client = LLMClient()
    try:
        yield client
    finally:
        await client.close()

async def get_current_user(authorization: str = Header(...)):
    """Authenticate requests."""
    token = authorization.replace("Bearer ", "")
    user = verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

@app.post("/chat")
async def chat(
    request: ChatRequest,
    llm: LLMClient = Depends(get_llm_client),
    user: User = Depends(get_current_user),
):
    return await llm.generate(request, user_id=user.id)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Custom Middleware (logging, timing) ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    print(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
    return response

# --- Rate Limiting ---
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/chat")
@limiter.limit("10/minute")
async def chat_limited(request: Request, body: ChatRequest):
    return await generate(body)
```

---

## 5. Database Integration

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import Column, String, Float, DateTime, select

# --- Async SQLAlchemy setup ---
DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/llm_app"
engine = create_async_engine(DATABASE_URL, pool_size=20)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    messages = Column(String)  # JSON string
    model = Column(String)
    total_tokens = Column(Float)
    created_at = Column(DateTime)

# Dependency
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@app.post("/conversations")
async def create_conversation(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    conv = Conversation(
        id=str(uuid4()),
        user_id=user.id,
        messages=json.dumps([m.dict() for m in request.messages]),
        model=request.model,
        created_at=datetime.utcnow()
    )
    db.add(conv)
    await db.commit()
    return {"id": conv.id}

@app.get("/conversations/{user_id}")
async def get_conversations(user_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Conversation).where(Conversation.user_id == user_id)
    )
    return result.scalars().all()
```

---

## 6. Streaming & WebSockets

```python
from fastapi.responses import StreamingResponse
from fastapi import WebSocket

# --- Server-Sent Events (SSE) for streaming LLM output ---
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        async for chunk in llm.stream(request.messages):
            data = json.dumps({"content": chunk, "done": False})
            yield f"data: {data}\n\n"
        yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

# --- WebSocket for real-time chat ---
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            messages = data.get("messages", [])
            
            async for chunk in llm.stream(messages):
                await websocket.send_json({"type": "chunk", "content": chunk})
            
            await websocket.send_json({"type": "done"})
    except Exception:
        await websocket.close()
```

---

## 7. LLM API Patterns

### Complete Production LLM API

```python
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
import asyncio

# --- Lifespan: Load model on startup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models
    app.state.llm = await load_model("gpt-4")
    app.state.embedder = await load_embedder("text-embedding-3-small")
    app.state.vectorstore = await connect_vectorstore()
    yield
    # Shutdown: cleanup
    await app.state.llm.close()

app = FastAPI(lifespan=lifespan)

# --- RAG endpoint ---
@app.post("/rag/query")
async def rag_query(
    query: str,
    top_k: int = 5,
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    # 1. Embed query
    query_embedding = await app.state.embedder.embed(query)
    
    # 2. Retrieve relevant documents
    docs = await app.state.vectorstore.search(query_embedding, top_k=top_k)
    
    # 3. Build prompt with context
    context = "\n".join([d.content for d in docs])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    # 4. Generate response
    response = await app.state.llm.generate(prompt)
    
    # 5. Log async (don't block response)
    background_tasks.add_task(log_query, query, response, docs)
    
    return {
        "answer": response,
        "sources": [{"title": d.title, "score": d.score} for d in docs]
    }

# --- Batch embedding endpoint ---
@app.post("/embeddings")
async def create_embeddings(texts: List[str]):
    # Process in chunks to avoid timeouts
    chunk_size = 100
    all_embeddings = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        embeddings = await app.state.embedder.embed_batch(chunk)
        all_embeddings.extend(embeddings)
    return {"embeddings": all_embeddings, "model": "text-embedding-3-small"}
```

---

## 8. Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user for security
RUN adduser --disabled-password appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://user:pass@db:5432/llmapp
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:16
    volumes:
      - pgdata:/var/lib/postgresql/data
    environment:
      POSTGRES_PASSWORD: pass
  
  redis:
    image: redis:7-alpine

volumes:
  pgdata:
```

### Production Configuration

```python
# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    database_url: str
    redis_url: str = "redis://localhost:6379"
    cors_origins: list = ["http://localhost:3000"]
    max_tokens: int = 4096
    rate_limit: str = "10/minute"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 9. Streamlit for AI Demos

### Basic LLM Chat App

```python
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="AI Chat", page_icon="🤖", layout="wide")
st.title("🤖 AI Chat Assistant")

# Sidebar configuration
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet"])
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.number_input("Max Tokens", 100, 4096, 1024)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        client = OpenAI()
        stream = client.chat.completions.create(
            model=model,
            messages=st.session_state.messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        response = st.write_stream(stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
```

### RAG Demo App

```python
import streamlit as st

st.title("📚 Document Q&A with RAG")

# File upload
uploaded_files = st.file_uploader(
    "Upload documents", 
    type=["pdf", "txt", "md"], 
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing documents..."):
        for file in uploaded_files:
            chunks = process_document(file)
            embed_and_store(chunks)
    st.success(f"Processed {len(uploaded_files)} documents!")

# Query interface
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Ask a question about your documents")
with col2:
    top_k = st.number_input("Sources", 1, 10, 3)

if query:
    with st.spinner("Searching..."):
        results = search(query, top_k=top_k)
        answer = generate_answer(query, results)
    
    st.markdown(f"### Answer\n{answer}")
    
    with st.expander("📄 Sources"):
        for i, doc in enumerate(results):
            st.markdown(f"**Source {i+1}** (Score: {doc.score:.3f})")
            st.text(doc.content[:500])
            st.divider()

# Metrics display
col1, col2, col3 = st.columns(3)
col1.metric("Documents", len(uploaded_files) if uploaded_files else 0)
col2.metric("Chunks", st.session_state.get("total_chunks", 0))
col3.metric("Avg Response Time", "1.2s")
```

### Streamlit Components Reference

```python
# --- Layout ---
st.columns([1, 2, 1])     # Column layout with proportions
st.tabs(["Tab 1", "Tab 2"])  # Tabbed interface
st.sidebar                   # Sidebar
st.expander("Details")       # Collapsible section
st.container()               # Container for grouping

# --- Input Widgets ---
st.text_input("Name")
st.text_area("Description")
st.number_input("Count", 0, 100)
st.slider("Temperature", 0.0, 2.0)
st.selectbox("Model", ["gpt-4", "claude"])
st.multiselect("Tags", ["tag1", "tag2", "tag3"])
st.checkbox("Enable streaming")
st.radio("Format", ["JSON", "Text"])
st.file_uploader("Upload", type=["pdf"])
st.chat_input("Message")

# --- Display ---
st.markdown("**Bold** and *italic*")
st.code("print('hello')", language="python")
st.json({"key": "value"})
st.dataframe(df)  # Interactive table
st.table(df)       # Static table
st.metric("Accuracy", "95.2%", "+2.1%")
st.progress(0.8)
st.spinner("Loading...")

# --- State Management ---
if "counter" not in st.session_state:
    st.session_state.counter = 0
st.session_state.counter += 1

# --- Caching ---
@st.cache_data  # Cache data (dataframes, API responses)
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource  # Cache resources (models, DB connections)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
```

---

## 10. Interview Questions

### Q1: Why FastAPI over Flask for AI applications?
**A**: (1) Native async support (essential for concurrent LLM API calls). (2) Pydantic integration (automatic request validation). (3) Auto-generated OpenAPI docs. (4) Type safety (catches errors early). (5) Performance — FastAPI is one of the fastest Python frameworks (comparable to Node.js/Go). Flask is synchronous by default, requiring extensions for async.

### Q2: Explain dependency injection in FastAPI.
**A**: `Depends()` lets you declare reusable functions that provide resources. FastAPI calls them automatically and passes results to your endpoint. Benefits: DRY (reuse auth, DB sessions), testability (swap dependencies in tests), lifecycle management (setup/teardown with yield). Example: `async def get_db(): session=...; yield session; session.close()`.

### Q3: How do you handle streaming LLM responses in FastAPI?
**A**: Use `StreamingResponse` with an async generator. The generator yields SSE-formatted chunks (`data: {...}\n\n`). Client uses EventSource API to consume the stream. Key: set `media_type="text/event-stream"` and headers for no-cache. Alternative: WebSocket for bidirectional communication.

### Q4: How would you deploy a FastAPI LLM service for production?
**A**: (1) Dockerfile with non-root user. (2) Uvicorn with multiple workers (--workers 4). (3) Nginx as reverse proxy with SSL. (4) Health check endpoint. (5) Environment variables for secrets (never hardcode). (6) Rate limiting. (7) CORS configuration. (8) Async database connections with connection pooling. (9) Monitoring (Prometheus metrics). (10) Docker Compose or K8s for orchestration.

### Q5: How does Streamlit handle state between reruns?
**A**: Streamlit reruns the entire script on every interaction. `st.session_state` is a dict that persists across reruns. Use it for: chat history, user settings, cached computations. Without session state, all variables reset each interaction. `st.cache_data` and `st.cache_resource` prevent expensive recomputation.

### Q6: What is the difference between `st.cache_data` and `st.cache_resource`?
**A**: `cache_data`: For serializable data (DataFrames, lists, API responses). Creates a new copy for each caller. Safe for data that shouldn't be shared. `cache_resource`: For non-serializable objects (ML models, DB connections). Returns the same object to all callers. Use for expensive-to-create resources that are thread-safe.

### Q7: How do you handle CORS in FastAPI?
**A**: Add CORSMiddleware with allowed origins, methods, and headers. Critical for web frontends calling your API. Restrict origins to specific domains in production (never `["*"]`). Allow credentials if using cookies/auth headers. CORS is enforced by browsers — not a server-side security measure by itself.

### Q8: Explain Pydantic validation in FastAPI.
**A**: Pydantic models define request/response schemas with type annotations and validators. FastAPI automatically: validates incoming requests (returns 422 if invalid), serializes responses, generates OpenAPI schema. Custom validators (`@validator`, `@field_validator`) add business logic. Benefits: single source of truth for API contract, auto-docs, type safety.

### Q9: How would you implement authentication for an LLM API?
**A**: (1) API key in header (`X-API-Key`): Simple, good for service-to-service. (2) JWT bearer tokens: Stateless, good for user auth. (3) OAuth2 scopes: For complex permissions. FastAPI has built-in OAuth2 support. Implement as a dependency: decode token → verify → return user. Rate limit per user. Log all API calls for audit.

### Q10: What is ASGI and how does it differ from WSGI?
**A**: WSGI (Flask, Django): Synchronous, handles one request at a time per worker. ASGI (FastAPI, Starlette): Asynchronous, can handle many concurrent connections per worker. Critical for LLM apps: while waiting for an LLM API response (seconds), ASGI handles other requests. WSGI would block entirely. Uvicorn is the standard ASGI server.

### Q11: Explain background tasks in FastAPI.
**A**: `BackgroundTasks` runs functions after the response is sent. Use for: logging, analytics, sending emails, updating caches — anything that shouldn't delay the response. Alternative: Celery for heavy/long-running tasks (separate worker process). Background tasks are simpler but run in the same process.

### Q12: How do you handle file uploads in FastAPI?
**A**: Use `UploadFile` parameter. Supports streaming large files (doesn't load all into memory). Access via `.file` (SpooledTemporaryFile), `.filename`, `.content_type`. For document processing: read chunks, validate file type, size limits. Store in object storage (S3), not local disk in production.

### Q13: How would you build a multi-page Streamlit app?
**A**: Use `st.page` (Streamlit 1.30+) or the `pages/` folder convention. Each Python file in `pages/` becomes a page. Shared state via `st.session_state`. Use `st.navigation()` for custom nav. Structure: main page for chat, pages for settings, document management, analytics.

### Q14: What are WebSockets and when to use them over SSE?
**A**: WebSockets: Full-duplex, bidirectional communication. SSE: Server → client only (one-way). Use WebSockets when: client needs to send data mid-stream (cancel generation, update context). Use SSE for: simple streaming output (chat responses). SSE is simpler, works through most proxies. WebSockets need special proxy config.

### Q15: How do you handle errors gracefully in FastAPI?
**A**: (1) Custom exception handlers: `@app.exception_handler(CustomError)`. (2) HTTPException for expected errors (404, 400). (3) Middleware for unexpected errors (500). (4) Pydantic validation errors return 422 automatically. (5) Return consistent error response format. (6) Don't expose internal details in production error messages.

### Q16: How would you implement caching for LLM responses?
**A**: (1) Semantic cache: Hash the prompt + params, cache in Redis with TTL. (2) Embedding-based cache: Embed the query, find similar cached queries. (3) Cache invalidation: TTL-based or manual purge. Consider: same prompt should return same response (temperature=0). Different temperatures = different cache keys. Cache hit rates can save 30-50% of API costs.

### Q17: How do you test FastAPI applications?
**A**: Use `TestClient` (synchronous) or `httpx.AsyncClient` for async. Test endpoints, validation, error cases. Mock dependencies with `app.dependency_overrides`. Test streaming responses by consuming the generator. Integration tests: test with real DB (use test database). Load tests: Locust or k6.

### Q18: What is uvicorn and how do you configure it for production?
**A**: Uvicorn is an ASGI server. Production config: `--workers N` (N = 2×CPU + 1), `--host 0.0.0.0`, `--port 8000`, `--no-access-log` (use middleware instead), `--limit-concurrency 100`. Use Gunicorn as process manager: `gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker`. Behind Nginx for SSL termination.

### Q19: How would you implement A/B testing for different prompts via API?
**A**: (1) Accept a `variant` parameter or use header-based routing. (2) Randomly assign users to variants (consistent via user_id hash). (3) Log all requests with variant info. (4) Endpoint returns response + variant metadata. (5) Analyze: compare completion quality, latency, cost per variant. (6) Use feature flags (LaunchDarkly, Unleash) for production.

### Q20: What is the difference between Streamlit and Gradio?
**A**: Streamlit: More flexible, full Python control, custom layouts, session state. Better for complex apps. Gradio: Quick ML demos, built-in components for inputs/outputs, easy sharing via Hugging Face Spaces. Better for model showcasing. For production AI tools: Streamlit. For quick model demos: Gradio. Both excellent for prototyping.

### Q21: How do you handle long-running tasks in FastAPI?
**A**: (1) Background tasks (simple, same process). (2) Celery + Redis/RabbitMQ (distributed, scalable). (3) Return task ID immediately, client polls for status. (4) WebSocket notification when complete. For LLM fine-tuning: use Celery. For batch embedding: background task or async processing. Never block the request for > 30 seconds.

### Q22: How would you version your API?
**A**: (1) URL path: `/api/v1/chat`, `/api/v2/chat` (most common). (2) Header: `API-Version: 2`. (3) Query param: `?version=2`. Use routers for clean separation: `v1_router`, `v2_router`. Support old versions during migration. Document deprecation timeline. Test backward compatibility.

### Q23: What security considerations for an LLM API?
**A**: (1) Authentication (API keys or JWT). (2) Rate limiting per user/IP. (3) Input validation (max length, content filtering). (4) Output filtering (prevent PII leakage). (5) CORS restriction. (6) HTTPS only. (7) No credentials in logs. (8) Prompt injection defense. (9) Cost controls (max tokens per request, per user budgets). (10) Audit logging.

### Q24: How do you handle database migrations with FastAPI?
**A**: Use Alembic (SQLAlchemy migration tool). `alembic init`, create migration scripts, apply with `alembic upgrade head`. Auto-generate migrations from model changes. In production: run migrations before deploying new code. Use transactions for safety. Test migrations on a copy of production data first.

### Q25: How would you monitor a FastAPI LLM service?
**A**: (1) Prometheus metrics (request count, latency, error rate, token usage). (2) Structured logging (JSON logs with request ID, user ID, model, tokens). (3) Health check endpoint. (4) OpenTelemetry for distributed tracing. (5) Dashboard: Grafana with alerts for error spikes, latency p99, cost anomalies. (6) LLM-specific: track hallucination rates, user satisfaction.

---

## 11. Day-to-Day Work Applications

### As an AI/LLM Engineer

**FastAPI for Model Serving**: Every production LLM system needs an API layer. FastAPI handles: authentication, rate limiting, request validation, response formatting, streaming, monitoring. Understanding async patterns is critical for handling thousands of concurrent LLM calls.

**Streamlit for Prototyping**: Quickly demo RAG systems, agent workflows, fine-tuning results. Build internal tools for prompt testing, dataset exploration, model comparison. Share with stakeholders who can't use Jupyter notebooks.

**Full Stack AI Applications**: FastAPI backend (model serving, RAG, agents) + React/Streamlit frontend. Understanding both sides makes you a more complete AI engineer. System design interviews often ask you to design the API layer.

---

## 12. Resources

### Excel Curriculum Links
- FastAPI Tutorial: https://fastapi.tiangolo.com/tutorial/
- FastAPI Video: https://www.youtube.com/watch?v=7t2alSnE2-I
- Streamlit Docs: https://docs.streamlit.io/
- Streamlit Tutorial: https://www.youtube.com/watch?v=JwSS70SZdyM
- Pydantic Docs: https://docs.pydantic.dev/
- Uvicorn: https://www.uvicorn.org/
- Docker for Python: https://docs.docker.com/language/python/
