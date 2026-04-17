# Month 8: vLLM, Inference Optimisation & LLM Serving at Scale
### Phase 3 | November 2026

---

## Week 1–2: LLM Inference Internals — vLLM & PagedAttention

> 📖 **Big picture:** You understand how LLMs work (transformer, KV cache). Now the question is: how do you serve them efficiently to thousands of concurrent users? This is one of the most practical topics for FAANG AI infrastructure roles.
>
> **The fundamental tension:** LLMs use a lot of GPU memory, and GPU memory is finite. The KV cache for each request grows with the sequence length. Without careful management, a single long-context request can exhaust memory and prevent other requests from being served. vLLM solves this with **PagedAttention** — a technique borrowed from operating system virtual memory management.
>
> **Why this matters in interviews:** "How would you design a system to serve a 70B LLM at production scale?" is a real FAANG system design question. The answer involves vLLM, PagedAttention, continuous batching, and quantisation — all of which are in this section.

### The Inference Memory Problem

When you serve an LLM, memory is allocated for:
```
1. Model weights (read-only, shared across requests)
2. KV cache (per-request, grows with sequence length)
3. Activations (during forward pass, temporary)

KV cache formula (per token per layer):
  memory = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element
  
For LLaMA 3 8B (fp16):
  = 2 × 32 × 8 × 128 × 2 bytes
  = 131,072 bytes = 128 KB per token
  
For 4096 token sequence: 128 KB × 4096 = 512 MB  (for ONE request!)
```

**The Problem Before vLLM:** Static memory allocation per request.
- Allocate max_sequence_length × kv_cache_size upfront
- Most of this is wasted (sequences don't fill the maximum)
- With 8 concurrent requests × 512 MB = 4 GB just for KV cache
- GPU memory fills up → low throughput

---

### PagedAttention: vLLM's Core Innovation

> 💡 **ELI5 (Explain Like I'm 5):**
> PagedAttention is like managing tables at a restaurant. Previously, restaurants would reserve an entire large table for a party of 8 even if only 2 showed up, wasting space (static KV cache). PagedAttention only assigns seats precisely when a person sits down, letting the restaurant serve instantly without wasting space.

**Insight:** KV cache is like virtual memory in an OS. Use paging!

```
Physical KV memory: divided into fixed-size blocks (e.g. 16 tokens × 128 KB = 2 MB/block)

Logical KV sequence: stored in blocks, not necessarily contiguous in physical memory

Block table: maps logical block idx → physical block address (like page table)

Example:
  Request A (400 tokens): uses blocks [3, 7, 12] (non-contiguous in physical)
  Request B (100 tokens): uses block [1]
  
Benefits:
  - No internal fragmentation (only last block is partially used)
  - Blocks shared between requests with same prefix (beam search, parallel sampling)
  - Memory utilisation: 96%+ vs <70% before PagedAttention
```

**Sequence sharing (copy-on-write):**
```python
# When sampling N sequences from same prompt (beam search / best-of-N):
# All share the KV blocks for the shared prefix!
# Only the diverged suffixes get new blocks
# Savings: N× reduction in KV cache for the shared prefix portion

# Also enables: caching system prompts across requests
# "Act as a helpful assistant. Today is {date}" — only date changes
```

---

### vLLM Architecture

> 📐 **End-to-End Production LLM Serving Pipeline (Annotated)**
>
> ```
>  CLIENT (Python / curl / any HTTP)
>      │
>      │  POST /v1/chat/completions
>      ▼
>  ┌────────────────────────────────────────────────────────────┐
>  │  FastAPI Server (OpenAI-compatible REST)                   │
>  │  • Validates request schema                                │
>  │  • Applies rate limiting                                   │
>  │  • Streams SSE tokens back to client                       │
>  └──────────────────────┬─────────────────────────────────────┘
>                         │ AsyncLLMEngine.add_request()
>                         ▼
>  ┌────────────────────────────────────────────────────────────┐
>  │  Scheduler  (the heart of vLLM)                           │
>  │  • Decides which requests run this iteration               │
>  │  • Splits requests into PREFILL (prompt) vs DECODE (gen)   │
>  │  • Implements continuous batching:                         │
>  │      → New requests join mid-flight, no waiting            │
>  │  • Block Allocator: allocates/frees KV cache pages         │
>  │      → Like OS virtual memory: paged, non-contiguous       │
>  └──────────────────────┬─────────────────────────────────────┘
>                         │ Batch of token_ids + KV block tables
>                         ▼
>  ┌────────────────────────────────────────────────────────────┐
>  │  Model Executor                                           │
>  │  • Distributes batch across GPUs (Tensor Parallelism)      │
>  │  • Runs ONE forward pass (NOT per-request, per-batch)      │
>  │  • Flash Attention reads/writes KV blocks in place         │
>  │  • Sampling: temperature / top-p / top-k → next tokens     │
>  └──────────────────────┬─────────────────────────────────────┘
>                         │ sampled_token_ids
>                         ▼
>  ┌────────────────────────────────────────────────────────────┐
>  │  Detokeniser + Output Handler                             │
>  │  • Maps token IDs → text strings                           │
>  │  • Checks stop conditions (max_tokens, stop sequences)     │
>  │  • Streams completed tokens back to FastAPI                │
>  └────────────────────────────────────────────────────────────┘
>
>  KEY INSIGHT: All requests share ONE forward pass per step.
>  GPU utilisation stays near 100% because the batch is always full.
>  This is why vLLM achieves 23× higher throughput than naive serving.
> ```

**Continuous Batching (vs Static Batching):**
```
Static batching (before vLLM):
  Request A: ████████████████████ (20 tokens)
  Request B: ████████             (8 tokens, done early → GPU idle)
  Request C: waiting...           (must wait for A to finish)
  
Continuous batching (vLLM):
  Request A: ████████████████████ (20 tokens, running)
  Request B: ★★★★★★★★             (done at step 8)
  Request C: ▓▓▓▓▓▓▓▓▓▓▓▓       (starts immediately after B finishes!)
  
Result: GPU is always working, no idle time
Throughput improvement: 23× vs static batching (from the vLLM paper)
```

---

### vLLM Usage

**Installation and basic serving:**
```bash
pip install vllm

# Start OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1    # Number of GPUs for tensor parallelism
```

**Query the vLLM server (OpenAI compatible):**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM doesn't require real key
)

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Explain RAG in 2 sentences"}],
    max_tokens=200,
    temperature=0.7
)
print(response.choices[0].message.content)
```

**Offline batch inference:**
```python
from vllm import LLM, SamplingParams

# Load model once
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    tensor_parallel_size=1,        # 1 GPU
    enable_prefix_caching=True,    # Cache KV for repeated prefixes
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=500,
    stop=["<|im_end|>"]
)

# Batch inference (all processed together with continuous batching)
prompts = [
    "Summarise the following in 3 points: ...",
    "Write a SQL query to find...",
    "Explain the difference between...",
    # Hundreds or thousands of prompts — vLLM handles batching automatically
]

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print(output.outputs[0].text)
```

---

### Tensor Parallelism & Pipeline Parallelism

> 💡 **ELI5 (Explain Like I'm 5):**
> Imagine carrying a piano. One person can't lift it alone. **Tensor Parallelism** is like splitting the piano in half and two people each carrying half simultaneously. **Pipeline Parallelism** is like a relay race where person 1 carries the piano up the first flight of stairs, then hands it to person 2 for the second flight.

**Tensor Parallelism (TP):** Split a single weight matrix across N GPUs.

```
Example: Attention weight W_q (4096 × 4096) split across 2 GPUs:
  GPU 0: W_q[:,0:2048]  (4096 × 2048)
  GPU 1: W_q[:,2048:]  (4096 × 2048)
  
Forward pass: each GPU does its half, then AllReduce to combine
Communication overhead: one AllReduce per layer
Use when: single layer doesn't fit in one GPU VRAM
```

**Pipeline Parallelism (PP):** Split model layers across GPUs.

```
GPU 0: Layers 0-15   (responsible for first half)
GPU 1: Layers 16-31  (responsible for second half)

Forward: sequence of activations passed GPU 0 → GPU 1
Backward: gradients flow GPU 1 → GPU 0

Use when: total model doesn't fit in one GPU
Problem: "pipeline bubble" — GPUs idle waiting for each other
Solution: micro-batching
```

**Combined (TP + PP) for very large models:**
```
LLaMA 3 70B serving strategy:
  TP = 4 (each layer split across 4 GPUs)
  PP = 2 (two halves, each on 4 GPUs)
  Total: 8 GPUs (2× 4× A100 nodes)
  
vLLM command:
  --tensor-parallel-size 4
  --pipeline-parallel-size 2
```

---

> 🃏 **Quick-Recall Card — vLLM & LLM Serving at Scale**
> | Concept | One-liner |
> |---|---|
> | PagedAttention | Allocates KV cache in fixed pages (like OS virtual memory) — no wasted reserved space. |
> | Continuous Batching | New requests join a running batch immediately, no waiting for the current batch to finish. |
> | TTFT | Time To First Token — latency of the Prefill phase. Watched by UX teams. |
> | TBT | Time Between Tokens — latency of each Decode step. Watched by streaming quality teams. |
> | Tensor Parallelism | Split each weight matrix across N GPUs (column-wise). AllReduce after each layer. |
> | Pipeline Parallelism | Split layers across GPUs (GPU0 → early layers, GPU1 → later layers). Relay-race model. |
> | Quantization INT8 | fp16 weights → int8. 2× memory reduction, ~5% accuracy cost. Use for serving 70B+ models. |
> | Speculative Decoding | Small draft model predicts K tokens; big model verifies in 1 pass. 2–3× decode speedup. |
> | KV cache key formula | `2 × layers × kv_heads × head_dim × bytes_per_element` per token |
>
> **Interview answer for "how would you serve a 70B model?":** *vLLM with PagedAttention + INT8 quantization + Tensor Parallelism across 4×A100 GPUs, tracking TTFT/throughput.*

### Benchmarking LLM Inference

**Key metrics:**
```
TTFT: Time To First Token — latency of prefill phase (affects perceived responsiveness)
TBT:  Time Between Tokens — latency of each decode step (affects streaming smoothness)
E2E:  End-to-end latency — total time for complete response
Throughput: tokens/second (across all concurrent requests)
RPS: Requests per second (for a fixed output length)
```

**vLLM benchmarking:**
```bash
# Clone vLLM
git clone https://github.com/vllm-project/vllm
cd vllm

# Run benchmark
python benchmarks/benchmark_serving.py \
  --backend vllm \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000 \
  --request-rate 10    # Requests per second
```

**Comparing backends:**

| Backend | Architecture | Best for | Weakness |
|---|---|---|---|
| vLLM | PagedAttention + continuous batching | High-throughput serving | Complex setup |
| TGI (HuggingFace) | Flash attention + continuous batching | Easy deployment, official HF support | Slower than vLLM at scale |
| Ollama | Quantised, llama.cpp backend | Local dev, Mac/CPU | Not for production |
| TensorRT-LLM (NVIDIA) | NVIDIA-optimised, CUDA kernels | Max throughput on NVIDIA | NVIDIA only, complex |
| LiteLLM | Proxy for 100+ LLM providers | Cost management, routing | Not a server itself |

**Typical benchmark numbers (LLaMA 3 8B, A100 80GB, float16):**
```
Single request (no concurrency):
  TTFT: ~200-400ms (for 512 token input)
  TBT: ~20-30ms/token (decode)
  
Under load (32 concurrent requests, vLLM):
  Throughput: ~2000-3000 tokens/sec
  TTFT P99: 1-3s (prefill queued)
  TBT P99: 50-100ms
```

---

### Week 2: Advanced Serving Patterns

### Speculative Decoding (Deep Dive)

> 💡 **ELI5 (Explain Like I'm 5):**
> Imagine a senior software engineer (the huge model) pair programming with a fast junior developer (the small model). The fast junior types out 5 lines of code instantly. The senior simply glances at the code and says "yes, looks good," approving it all at once rather than typing it themselves. The whole process is dramatically faster.

```
Problem: Decode phase is memory-bandwidth bound, not compute-bound
  GPU is underutilised generating one token at a time
  
Speculative decoding:
  1. Drafter (small model, e.g. LLaMA 3 1B): auto-regressively generates K tokens
  2. Verifier (large target model, e.g. LLaMA 3 70B): processes all K in ONE forward pass
  3. Accept/reject each drafter token based on target model's probability
  4. Guaranteed to match target model distribution (rejection sampling theorem)
  
Speedup: 2-3× for typical distribution (most tokens accepted)
Trade-off: Need compatible draft model (same tokeniser)
```

```python
# Speculative decoding with HuggingFace
from transformers import AutoTokenizer, TextStreamer

# Using HF transformers speculative decoding
model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
draft_model_id = "meta-llama/Meta-Llama-3-1B-Instruct"

target_model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
draft_model = AutoModelForCausalLM.from_pretrained(
    draft_model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

# Generate with speculative decoding
output = target_model.generate(
    **inputs,
    assistant_model=draft_model,    # HF built-in speculative decoding
    max_new_tokens=500,
)
```

### Prefix Caching

```python
# vLLM prefix caching (enabled by default for long system prompts)
# All requests with the same system prompt share KV blocks for that prefix

# This is critical for:
# - Same system prompt across all requests (save 90%+ KV memory for prefix)
# - Multi-turn conversations (cache turns 1-N when answering turn N+1)
# - RAG: cache the retrieved context if same documents reused across queries

# Enable in vLLM:
llm = LLM(
    model="...",
    enable_prefix_caching=True,
    preemption_mode="swap"  # Swap to CPU RAM if GPU full
)
```

### LLM Gateway / Router Architecture

```python
# Production LLM serving with routing
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "best",
            "litellm_params": {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_KEY")
            },
            "model_info": {"id": "gpt4o"}
        },
        {
            "model_name": "fast",  
            "litellm_params": {
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_KEY")
            }
        },
        {
            "model_name": "fast",  # Load balancing: two endpoints same model
            "litellm_params": {
                "model": "hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct",
                "api_base": "http://vllm-server:8000/v1"
            }
        }
    ],
    routing_strategy="least-busy",  # Or: "latency-based", "usage-based"
    fallbacks=[
        {"gpt4o": ["gpt-4o-mini"]},  # If GPT-4o fails, fallback to mini
    ],
    context_window_fallbacks=[       # If context too long for model:
        {"gpt-4o": ["claude-3-5-sonnet-20241022"]}
    ]
)

# Use router just like LiteLLM
response = await router.acompletion(
    model="fast",
    messages=[{"role": "user", "content": "Hello"}]
)
```

---

### Week 3–4: ML System Design — Advanced Serving Systems

### Design 6: Designing an LLM Inference Infrastructure (Senior IC Question)

**Prompt:** "You're joining a team that serves LLaMA 3 70B for 10,000 users. Design the infrastructure."

**Full answer framework:**

```
GIVEN:
  Model: LLaMA 3 70B (140 GB float16)
  Users: 10,000 concurrent → ~500 RPS at 20 req/user/day
  Latency SLA: TTFT < 2s, TBT < 50ms

GPU REQUIREMENTS:
  Model weights: 140 GB
  KV cache budget: 80% of remaining GPU memory
  Efficient serving: 2× A100 80GB per replica (tensor parallel=2)
  Memory per replica: 2 × 80 = 160 GB total, 140 weights + 20 GB KV
  KV supports ~40 concurrent sequences at 4096 tokens
  
  At 500 RPS, avg completion time 5s → 2,500 in-flight → need ~63 replicas
  63 replicas × 2 A100 = 126 A100 GPUs (use A100 80GB cluster)
  
  COST: ~$126/hour (A100 at $1/hr) = ~$1M/month ← motivates optimisation!

OPTIMISATIONS:
  1. INT8 quantisation: 140 GB → 70 GB → 1× A100 per replica (2× GPU savings!)
  2. Speculative decoding with LLaMA 3 8B draft: 2-3× throughput
  3. Prefix caching: reduces effective sequence length for common prefixes
  
  After optimisations: ~16-20 replicas needed ($16-20/hr = $12-15K/month)
  
ARCHITECTURE:
  Load Balancer (NGINX/Envoy)
    │
    ├── vLLM Cluster (GKE GPU node pool)
    │   ├── Node 1 (A100): vLLM instance serving LLaMA 3 70B INT8
    │   ├── Node 2 (A100): vLLM instance
    │   └── ...
    │
    ├── Semantic Cache (Redis + cosine similarity)
    │   └── ~30% cache hit rate for common queries
    │
    └── Monitoring (Prometheus + Grafana)
        ├── DCGM Exporter (GPU metrics)
        └── vLLM metrics (throughput, TTFT, TBT, queue depth)
        
SCALING:
  HPA (Horizontal Pod Autoscaler) based on GPU utilisation (target: 70%)
  Scale up trigger: queue depth > 10 for 30 seconds
  Scale down: queue depth = 0 for 5 minutes
  Pre-warm: always keep N+1 replicas ready (cold start = 2-3 min for 70B model)
```

---

## Interview Q&A — Inference & vLLM

**Q1: What is PagedAttention and what problem does it solve?**
> Before vLLM, KV cache was allocated contiguously per request, leading to severe memory fragmentation. PagedAttention borrows OS virtual memory concepts: KV cache is divided into fixed-size blocks (pages) with a logical-to-physical block table. Non-contiguous physical blocks can hold a logical sequence. Benefits: eliminates fragmentation (96%+ memory utilisation vs <70%), enables block sharing (prefix caching, beam search), enabling dramatically more concurrent requests.

**Q2: What is the difference between prefill and decode phases in LLM inference?**
> Prefill: the entire input prompt is processed in one forward pass (compute-bound, can be parallelised across tokens since all inputs are known). Decode: new tokens generated one at a time, each requiring a separate forward pass that reads the KV cache of all previous tokens (memory-bandwidth bound). TTFT measures prefill latency; TBT measures decode latency per step.

**Q3: Why is LLM inference memory-bandwidth bound rather than compute-bound?**
> During decode, we process only 1 token per forward pass but must read all KV cache weights + all model weights from GPU HBM. With a 70B model, that's 140 GB of data reads per forward step, but only a single token's worth of compute (matrix-vector multiplication, not matrix-matrix). The arithmetic intensity is too low to keep tensor cores busy. This is why throughput doesn't scale with more raw compute (FLOPS) but with memory bandwidth.

**Q4: How does continuous batching improve throughput compared to static batching?**
> Static: the engine waits for a full batch of requests, processes them together, returns all results together. Requests that finish early leave the GPU idle. Continuous batching: the scheduler dynamically adds new requests to the running batch as soon as any request finishes. The GPU is never idle waiting for stragglers, achieving much higher utilisation (23× throughput improvement in the vLLM paper).

**Q5: What is tensor parallelism and when would you use it vs pipeline parallelism?**
> Tensor parallelism (TP): split each weight matrix across N GPUs. Each GPU holds a horizontal slice; results combined via AllReduce per layer. Use when a single layer doesn't fit in one GPU (e.g. LLaMA 70B attention layer is 4096×4096 for GQA). Pipeline parallelism (PP): different GPUs run different groups of layers sequentially. Use when total model fits in one GPU but sequential layer depth requires parallelism. TP has lower latency (requires fast NVLink), PP has higher GPU utilisation with micro-batching.

**Q6: How would you reduce cost of serving LLaMA 3 70B by 50% without significant quality loss?**
> 1. INT8 quantisation: 2× memory reduction → 2× GPU utilisation → ~50% cost reduction. 2. Speculative decoding: 2-3× throughput with same GPU count → proportional cost reduction. 3. Prefix caching: reduces effective tokens to process. 4. Semantic cache in Redis: 30% cache hit rate on common queries → 30% fewer LLM calls. Combined, these can achieve 60-70% cost reduction with <5% quality degradation.

---

## Scaling Laws for LLMs

### Chinchilla Scaling Laws

The key insight: for a given compute budget, there's an **optimal balance** between model size and training data.

| Principle | Detail |
|---|---|
| **Kaplan et al. (2020)** | Loss scales as power law with model size, data, and compute |
| **Chinchilla (2022)** | Optimal: train tokens ≈ 20× model parameters |
| **Implication** | LLaMA 7B should train on ~140B tokens (actually: 1T tokens → overtrained but better at inference) |

```
Optimal Tokens ≈ 20 × Parameters

LLaMA 3 8B: 8B params → would need 160B tokens (optimal)
             Actually trained on 15T tokens → massively overtrained
             Result: much better quality per parameter at inference

GPT-4 (rumoured ~1.8T params): compute-optimal would be ~36T tokens
```

**Why Overtrain?**: Chinchilla-optimal minimizes training cost. But production cares about **inference cost**. A smaller model trained on more data → same quality as larger model → cheaper to serve.

**Interview Key**: "Modern practice is to overtrain smaller models well beyond Chinchilla-optimal, because inference cost matters more than training cost for production deployments."

---

## Local LLM Inference: GGUF & llama.cpp

### GGUF Format

GGUF (GPT-Generated Unified Format) is the standard format for running quantized LLMs locally with llama.cpp.

```
GGUF File Structure:
├── Header (magic number, version, metadata count)
├── Metadata (model architecture, tokenizer, quantization type)
└── Tensor Data (quantized weights)

Common quantization levels:
- Q2_K: 2-bit (tiny, low quality) — ~2.5 GB for 7B model
- Q4_K_M: 4-bit medium (good balance) — ~4.1 GB for 7B model ⭐ Best default
- Q5_K_M: 5-bit medium (higher quality) — ~4.8 GB for 7B model
- Q6_K: 6-bit (near FP16 quality) — ~5.5 GB for 7B model
- Q8_0: 8-bit (minimal quality loss) — ~7.2 GB for 7B model
- F16: Full 16-bit (original quality) — ~14 GB for 7B model
```

### llama.cpp Setup & Usage

```bash
# Install llama-cpp-python
pip install llama-cpp-python

# Or with GPU support (CUDA)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Download GGUF model
# From: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
# File: llama-2-7b-chat.Q4_K_M.gguf
```

```python
from llama_cpp import Llama

# Load model
llm = Llama(
    model_path="models/llama-3-8b-instruct.Q4_K_M.gguf",
    n_ctx=4096,        # Context window
    n_gpu_layers=-1,   # -1 = offload all layers to GPU
    n_threads=8,       # CPU threads for non-GPU layers
    verbose=False
)

# Generate
output = llm(
    "Q: What is retrieval augmented generation?\nA:",
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
    stop=["Q:", "\n\n"]
)
print(output["choices"][0]["text"])

# Chat completion (instruction format)
output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain RAG in 3 sentences."}
    ],
    max_tokens=256,
    temperature=0.7
)

# Streaming
for chunk in llm.create_chat_completion(
    messages=[{"role": "user", "content": "Explain attention"}],
    stream=True
):
    delta = chunk["choices"][0]["delta"]
    if "content" in delta:
        print(delta["content"], end="", flush=True)
```

### llama.cpp as OpenAI-Compatible Server

```bash
# Run as API server (drop-in replacement for OpenAI API)
python -m llama_cpp.server \
    --model models/llama-3-8b-instruct.Q4_K_M.gguf \
    --n_ctx 4096 \
    --n_gpu_layers -1 \
    --host 0.0.0.0 \
    --port 8000

# Now use with OpenAI client!
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="llama-3-8b-instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## GPTQ Quantization

### What is GPTQ?

GPTQ (GPT Quantization) is a **post-training quantization** method that achieves near-lossless 4-bit quantization using calibration data.

```
Process:
1. Take trained FP16 model
2. Feed calibration data (128 samples of C4 dataset)
3. Quantize weights layer-by-layer
4. Minimize quantization error per column using Hessian information
5. Output: 4-bit model with minimal quality loss
```

### GPTQ vs Other Quantization Methods

| Method | Bits | Speed | Quality | GPU Required | Best For |
|---|---|---|---|---|---|
| **GPTQ** | 4-bit | Fast (GPU) | High | Yes | GPU inference |
| **GGUF (llama.cpp)** | 2-8 bit | Good (CPU/GPU) | Varies | No | Local/CPU inference |
| **AWQ** | 4-bit | Fastest | High | Yes | Production GPU serving |
| **BitsAndBytes** | 4/8-bit | Moderate | Good | Yes | Fine-tuning (QLoRA) |
| **SqueezeLLM** | 3-4 bit | Good | High | Yes | Memory-constrained |

### Using GPTQ Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load GPTQ model (via auto-gptq integration)
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GPTQ",
    device_map="auto",
    trust_remote_code=False,
    revision="main"  # or "gptq-4bit-32g-actorder_True"
)
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-Chat-GPTQ")

# Or with vLLM (recommended for serving)
# vllm serve TheBloke/Llama-2-7B-Chat-GPTQ --quantization gptq
```

### AWQ (Activation-Aware Weight Quantization)

```python
# AWQ is becoming preferred over GPTQ for serving (faster kernels)
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-Chat-AWQ",
    fuse_layers=True  # Fused attention for speed
)

# Or serve with vLLM
# vllm serve TheBloke/Llama-2-7B-Chat-AWQ --quantization awq
```

**Interview Key**: "For production GPU serving, AWQ is fastest. For local/CPU inference, GGUF with llama.cpp. For fine-tuning, BitsAndBytes (QLoRA). GPTQ is the original but AWQ has surpassed it in inference speed."

---

## 📚 Further Resources

**Papers (must-read):**
- **"Efficient Memory Management for Large Language Model Serving with PagedAttention"** (Kwon et al., 2023) — https://arxiv.org/abs/2309.06180
- **"Fast Inference from Transformers via Speculative Decoding"** (Leviathan et al., 2022) — https://arxiv.org/abs/2211.17192

**Docs / Tools:**
- vLLM docs: https://docs.vllm.ai
- TGI docs: https://huggingface.co/docs/text-generation-inference
- LiteLLM docs: https://docs.litellm.ai (for routing and cost tracking)

**Courses:**
- **DeepLearning.AI: Efficiently Serving LLMs** — https://learn.deeplearning.ai/courses/efficiently-serving-llms
- **Full Stack Deep Learning: LLM Bootcamp** — https://fullstackdeeplearning.com

**Benchmarking:**
- vLLM benchmark scripts: https://github.com/vllm-project/vllm/tree/main/benchmarks
- LLMPerf (OpenAI/Anyscale benchmark): https://github.com/ray-project/llmperf

---

## Day-to-Day Work: LLM Serving in Production

### Your Role in LLM Serving (MLOps → AI Engineer)

```
As an AI/LLM engineer, you'll be responsible for:

1. MODEL DEPLOYMENT
   - Deploy models to GPU clusters (GKE, EKS, Cloud Run GPU)
   - Choose the right serving framework (vLLM, TGI, Triton)
   - Configure batching, memory, parallelism settings

2. PERFORMANCE OPTIMIZATION
   - Benchmark throughput and latency for different configurations
   - Tune batch size, max_model_len, GPU memory utilisation
   - Implement speculative decoding for latency-sensitive apps

3. COST MANAGEMENT
   - Right-size GPU instances  
   - Implement autoscaling based on request volume
   - Use spot/preemptible instances for non-critical workloads
   - Mix of API (OpenAI) + self-hosted based on volume and sensitivity

4. RELIABILITY
   - Health checks, graceful degradation, failover
   - Blue-green deployments for model updates
   - Canary releases with quality monitoring
```

### Production vLLM Deployment on GCP

```yaml
# kubernetes/vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama3-8b
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm-llama3
  template:
    metadata:
      labels:
        app: vllm-llama3
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command: ["python", "-m", "vllm.entrypoints.openai.api_server"]
        args:
          - "--model=meta-llama/Meta-Llama-3-8B-Instruct"
          - "--dtype=bfloat16"
          - "--max-model-len=8192"
          - "--gpu-memory-utilization=0.90"
          - "--enable-prefix-caching"  # reuse KV cache for common system prompts
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "32Gi"
            cpu: "4"
        ports:
          - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          periodSeconds: 10
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-l4  # or nvidia-tesla-a100
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm-llama3
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-llama3-8b
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "70"  # scale up at 70% GPU utilisation
```

### Cost Comparison: Self-Hosted vs API

```python
# This calculation is something you'll do regularly

def cost_comparison(
    daily_requests: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
):
    """Compare API vs self-hosted costs."""
    
    # API costs (GPT-4o-mini)
    api_input_cost = daily_requests * avg_input_tokens / 1e6 * 0.15
    api_output_cost = daily_requests * avg_output_tokens / 1e6 * 0.60
    api_daily = api_input_cost + api_output_cost
    api_monthly = api_daily * 30
    
    # Self-hosted costs (LLaMA 3 8B on L4 GPU)
    # L4 GPU on GKE: ~$0.70/hr (on-demand), ~$0.21/hr (spot)
    l4_monthly_ondemand = 0.70 * 24 * 30  # $504
    l4_monthly_spot = 0.21 * 24 * 30       # $151
    
    # Throughput: vLLM on L4 with LLaMA 3 8B ≈ 30 req/sec
    max_daily_capacity = 30 * 3600 * 24  # 2.59M requests/day
    gpus_needed = max(1, daily_requests // max_daily_capacity + 1)
    
    self_hosted_monthly = gpus_needed * l4_monthly_ondemand
    self_hosted_spot = gpus_needed * l4_monthly_spot
    
    print(f"Daily requests: {daily_requests:,}")
    print(f"\nAPI (GPT-4o-mini):     ${api_monthly:,.0f}/month")
    print(f"Self-hosted (L4 on-demand): ${self_hosted_monthly:,.0f}/month ({gpus_needed} GPU)")
    print(f"Self-hosted (L4 spot):      ${self_hosted_spot:,.0f}/month ({gpus_needed} GPU)")
    print(f"\nBreak-even: API is cheaper below ~{self_hosted_monthly/api_daily:.0f} requests/day")

# Example: moderate traffic
cost_comparison(10000, 500, 200)
# API: ~$126/month
# Self-hosted: ~$504/month (1 GPU)
# → API wins for <40K requests/day

# Example: high traffic
cost_comparison(500000, 500, 200)
# API: ~$6,300/month
# Self-hosted: ~$504/month (1 GPU handles it)
# → Self-hosted wins 12× cheaper
```

### Production Health Monitoring

```python
# Metrics to expose from your vLLM deployment

import aiohttp
import json

async def check_vllm_health(base_url: str = "http://vllm-service:80"):
    """Health check for vLLM — call from your monitoring service."""
    
    # 1. Check server is alive
    async with aiohttp.ClientSession() as session:
        resp = await session.get(f"{base_url}/health")
        if resp.status != 200:
            return {"status": "unhealthy", "reason": "health endpoint failed"}
    
    # 2. Check model is loaded and serving
    async with aiohttp.ClientSession() as session:
        resp = await session.get(f"{base_url}/v1/models")
        models = await resp.json()
        if not models.get("data"):
            return {"status": "unhealthy", "reason": "no models loaded"}
    
    # 3. Test inference latency
    import time
    start = time.time()
    async with aiohttp.ClientSession() as session:
        resp = await session.post(f"{base_url}/v1/chat/completions", json={
            "model": models["data"][0]["id"],
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5
        })
        latency_ms = (time.time() - start) * 1000
    
    return {
        "status": "healthy",
        "model": models["data"][0]["id"],
        "test_latency_ms": round(latency_ms),
        "warning": "high latency" if latency_ms > 5000 else None
    }
```
