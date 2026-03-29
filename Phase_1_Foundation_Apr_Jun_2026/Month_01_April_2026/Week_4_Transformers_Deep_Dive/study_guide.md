# Week 4: Transformers Deep Dive — KV Cache, FlashAttention, RoPE, GQA
### Phase 1 | Month 1 | April 28 – May 4, 2026

> This week goes under the hood of modern LLMs — the "how does it actually work at inference time" knowledge that separates senior AI engineers from junior ones.

---

## 🎯 Learning Objectives

By the end of this week you will:
- Explain the full inference computation step-by-step
- Understand KV cache internals and memory implications
- Be able to derive FlashAttention's memory advantage
- Know RoPE, GQA, MoE, and how LLaMA 3 / Mistral differ from GPT-2
- Calculate memory requirements for any model
- Answer deep architecture questions at FAANG L5/L6 level

---

## Part 1 — The Full Inference Pipeline

### 1.1 Two Phases: Prefill vs Decode

**Phase 1: Prefill (prompt processing)**
- Process all input tokens in parallel
- Fill the KV cache with K and V tensors for all input tokens
- Compute: O(n²) attention across the prompt
- Fast: parallel, GPU-efficient

**Phase 2: Decode (token generation)**
- Generate one new token at a time (autoregressive)
- For each step: compute attention of new token against ALL previous tokens (using KV cache)
- Compute: O(n) per step using cached KVs
- Slow: sequential, cannot parallelise across steps

```
Prefill:  [The][cat][sat][on] → process all 4 at once → KV cache filled
Decode:   [the] → attend to KV cache → sample "mat" → append
          [mat] → attend to KV cache → sample "." → append
          [.]   → attend to KV cache → <EOS> → stop
```

### 1.2 Memory Breakdown for LLM Inference

For a model like LLaMA 3 8B:
- Parameters: 8B × 2 bytes (fp16) = **16 GB**
- KV cache per token: 2 × L × d_head × n_heads × 2 bytes
  - LLaMA 3 8B: 2 × 32 layers × 128 dim × 8 heads × 2 bytes = 131,072 bytes/token = **128 KB/token**
- At 4096 context length: 4096 × 128 KB = **512 MB for KV cache**

```
Total GPU memory (inference): weights + KV_cache + activations
≈ 16 GB + 0.5 GB + ~0.5 GB = ~17 GB
→ Needs a single A100 80GB or two 3090s
```

**Key formula for KV cache memory:**
```
KV_memory = 2 × num_layers × num_kv_heads × head_dim × seq_len × bytes_per_element
```

---

## Part 2 — KV Cache: Deep Dive

### 2.1 What Is the KV Cache?

During autoregressive generation, step `t` needs to attend to all previous `t-1` tokens. Without caching, the model would recompute K and V for all previous tokens every step:

```
Step 5: Attention([token_1, ..., token_5])
   - Compute Q₅, K₁, K₂, K₃, K₄, K₅, V₁, V₂, V₃, V₄, V₅
   - This recomputes K₁..K₄, V₁..V₄ which were computed in steps 1–4!
```

With KV cache, K and V computed in previous steps are saved:
```
Step 1: Compute K₁, V₁ → save to cache
Step 2: Compute K₂, V₂ → save to cache; attend to [K₁,K₂], [V₁,V₂]
Step 3: Compute K₃, V₃ → save to cache; attend to [K₁,K₂,K₃], [V₁,V₂,V₃]
...
Step t: Only compute K_t, V_t; read [K₁..K_t] and [V₁..V_t] from cache
```

**Result:** Decode step is O(n) instead of O(n²).

### 2.2 Implementation

```python
class CachedAttention(nn.Module):
    def forward(self, x, kv_cache=None, step=None):
        B, T, C = x.shape
        
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        if kv_cache is not None:
            # Decode mode: append to cache
            kv_cache['k'] = torch.cat([kv_cache['k'], k], dim=1)
            kv_cache['v'] = torch.cat([kv_cache['v'], v], dim=1)
            # Attend to all cached keys and values
            k = kv_cache['k']
            v = kv_cache['v']
        
        # Compute attention: q is (B, 1, C) in decode mode
        # k, v are (B, full_seq_len, C) — full history
        attn = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        # No causal mask needed in decode (past tokens are already valid)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        return self.W_o(out), kv_cache
```

### 2.3 KV Cache Challenges

1. **Memory limits context length:** 128K context × 128 KB/token = 16 GB just for KV cache
2. **Memory fragmentation:** Different requests have different lengths → hard to batch → PagedAttention (vLLM)
3. **Quantisation of KV cache:** INT8 KV cache can halve memory (small quality loss)

---

## Part 3 — FlashAttention

### 3.1 The Problem with Standard Attention

Standard attention computes:
```
S = QK^T / √d_k        (batch, heads, seq, seq) — O(n²) memory!
P = softmax(S)
O = PV
```

For seq_len=8192 and 32 heads: S is 8192×8192 per head = 268M floats per head × 32 heads × 4 bytes = **32 GB** just for the attention matrix!

The bottleneck is **memory bandwidth** — reading and writing this huge matrix to HBM (GPU memory).

### 3.2 FlashAttention Solution: Tiling

FlashAttention (Dao et al., 2022) computes attention **in tiles** that fit in SRAM (fast on-chip memory), never materialising the full n×n attention matrix:

```
Algorithm:
1. Split Q into blocks Q₁, Q₂, ... Q_Br
2. Split K, V into blocks K₁, K₂, ... K_Bc and V₁, V₂, ... V_Bc
3. For each block:
   a. Load Qᵢ, Kⱼ, Vⱼ into SRAM
   b. Compute partial attention scores
   c. Use "online softmax" to update running max and sum
   d. Accumulate output incrementally
4. Never write the full n×n matrix to HBM
```

**Result:**
- Memory: O(n) instead of O(n²)
- Speed: 2–4× faster on A100s (memory bandwidth is the bottleneck)
- Numerically identical to standard attention (not an approximation!)

### 3.3 FlashAttention 2 & 3

- **FlashAttention-2 (2023):** Better work partitioning, 2× speedup over FA1 on A100
- **FlashAttention-3 (2024):** Optimised for H100 using WGMMA and TMA hardware features

```python
# PyTorch 2.0+: use scaled_dot_product_attention which calls FlashAttention
import torch.nn.functional as F

output = F.scaled_dot_product_attention(q, k, v, 
    attn_mask=None, 
    dropout_p=0.0,
    is_causal=True)   # enables FlashAttention automatically
```

---

## Part 4 — Modern Positional Encodings

### 4.1 RoPE (Rotary Position Embedding)

Used by: LLaMA, Mistral, Falcon, GPT-NeoX, Gemma

**Key idea:** Encode positions by rotating Q and K vectors by angle proportional to position.

$$\mathbf{q}_m = \mathbf{q} e^{i m \theta}$$

The dot product QᵀmKn naturally encodes **relative position (m-n)** because:
$$\text{Re}(\mathbf{q}_m \overline{\mathbf{k}}_n) = \text{Re}(\mathbf{q} \overline{\mathbf{k}} e^{i(m-n)\theta})$$

**Advantages:**
- Relative positions naturally captured
- Generalises to longer contexts than seen in training
- Efficient to compute: rotation applied before attention, no change to architecture

```python
def apply_rope(q, k, cos, sin):
    # Rotate pairs of dimensions
    q_rot = torch.cat([-q[..., 1::2], q[..., ::2]], dim=-1)
    k_rot = torch.cat([-k[..., 1::2], k[..., ::2]], dim=-1)
    q = q * cos + q_rot * sin
    k = k * cos + k_rot * sin
    return q, k
```

**Context extension with RoPE:** Methods like YaRN, LongRoPE adjust the rotation frequencies to extend the effective context window beyond training length.

### 4.2 ALiBi (Attention with Linear Biases)

Used by: MPT, BLOOM

Instead of adding positional embeddings, adds a **negative bias** proportional to key-query distance:

$$\text{softmax}(QK^T / \sqrt{d_k} + m \cdot [-(n-1), ..., -1, 0])$$

Where `m` is a head-specific slope. This naturally penalises attending to distant tokens.

**Advantage:** Extrapolates to longer sequences than training without any modification.

---

## Part 5 — Multi-Head Variants: MHA, MQA, GQA

### 5.1 Multi-Head Attention (MHA) — Original

```
n_heads Q heads
n_heads K heads (one per Q head)
n_heads V heads (one per Q head)

GPT-3: 96 heads → 96 K heads, 96 V heads
```

**Problem:** KV cache at inference is huge: n_heads × seq_len × head_dim

### 5.2 Multi-Query Attention (MQA)

```
n_heads Q heads
1 K head (shared by ALL Q heads)
1 V head (shared by ALL Q heads)

PaLM, Falcon use MQA
```

**KV cache reduction:** n_heads × smaller!
**Downside:** Quality drops slightly

### 5.3 Grouped Query Attention (GQA) — Current Standard

```
n_heads Q heads
n_kv_heads K heads (group of Q heads share one K head)
n_kv_heads V heads

LLaMA 3 8B:  32 Q heads, 8 KV heads → 4 Q heads per KV head
LLaMA 3 70B: 64 Q heads, 8 KV heads → 8 Q heads per KV head
Mistral 7B:  32 Q heads, 8 KV heads
```

**KV cache:** 4–8× smaller than MHA with minimal quality loss.

```python
# In LLaMA 3 inference, GQA implementation:
# Q: (batch, seq, n_heads, head_dim)
# K, V: (batch, seq, n_kv_heads, head_dim)

# Expand K, V to match Q heads
n_rep = n_heads // n_kv_heads  # = 4 for LLaMA 3 8B
k = k[:, :, :, None, :].expand(batch, seq, n_kv_heads, n_rep, head_dim)
k = k.reshape(batch, seq, n_heads, head_dim)  # now matches Q
```

---

## Part 6 — Mixture of Experts (MoE)

### 6.1 What Is MoE?

Instead of one FFN, use many "expert" FFNs. A **router** selects which experts process each token:

```
Input token
    ↓
Router (linear layer + softmax)
    ↓ selects top-k experts (typically k=2)
Expert 1: FFN₁
Expert 3: FFN₃
    ↓ weighted sum
Output
```

**Key insight:** Model has many parameters but **only activates a fraction per token** (sparse activation).

### 6.2 Examples

| Model | Total Params | Active Params | Experts | Top-k |
|---|---|---|---|---|
| Mixtral 8x7B | 46.7B | 12.9B | 8 × 7B | 2 |
| GPT-4 (rumoured) | ~1.8T | ~220B | ~16 | 2 |
| Mixtral 8x22B | 141B | 39B | 8 × 22B | 2 |

**Advantages:**
- Much larger model capacity (more parameters = more knowledge)
- Same inference cost as dense model of the "active params" size
- Scales better with compute than dense models

**Disadvantages:**
- All expert weights must be in GPU memory (expensive to host)
- Load balancing: router may prefer some experts (auxiliary load-balancing loss)
- Communication overhead in multi-GPU setups (expert parallelism)

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, n_experts=8, top_k=2):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([FFN(d_model) for _ in range(n_experts)])
        self.top_k = top_k
    
    def forward(self, x):
        # x: (batch, seq, d_model)
        router_logits = self.router(x)  # (batch, seq, n_experts)
        routing_weights = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = routing_weights.topk(self.top_k, dim=-1)
        
        # Normalise
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Route to experts and combine
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]  # (batch, seq)
            weight = top_k_weights[:, :, k:k+1]  # (batch, seq, 1)
            # Dispatch to expert (simplified — real impl uses scatter)
            for e in range(len(self.experts)):
                mask = (expert_idx == e)
                if mask.any():
                    output[mask] += weight[mask] * self.experts[e](x[mask])
        
        return output
```

---

## Part 7 — LLaMA 3 Architecture vs GPT-2: Diffs

| Component | GPT-2 | LLaMA 3 (8B) |
|---|---|---|
| Positional Encoding | Learned absolute | RoPE |
| Attention | MHA | GQA (32 Q, 8 KV) |
| FFN Activation | GELU | SwiGLU |
| Normalisation | Post-LayerNorm | Pre-RMSNorm |
| Vocabulary | 50,257 | 128,256 |
| Context | 1,024 | 8,192 (extendable) |
| Flash Attention | No | Yes |
| Tied Embeddings | Yes | No |

### SwiGLU (used in LLaMA)
```
SwiGLU(x) = SiLU(W₁x) ⊙ (W₂x)   (element-wise multiplication)

# vs original FFN:
FFN(x) = GELU(xW₁) W₂

# SwiGLU needs 3 matrices (W₁, W₂, W₃) but uses 2/3 × 4d_model = ~2.67d_model
# for same compute budget, giving slightly different parameter counts
```

### RMSNorm (simpler than LayerNorm)
```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight
        # Note: NO mean subtraction, NO bias — simpler and equally effective
```

---

## Part 8 — Calculating Model Memory Requirements

### 8.1 Weights Memory

```
# bytes per parameter:
fp32  = 4 bytes
bf16/fp16 = 2 bytes
int8  = 1 byte
int4  = 0.5 bytes

# Minimum GPU memory for inference (weights only):
LLaMA 3 8B  at fp16:  8B × 2 = 16 GB  → need 1× A100 40GB
LLaMA 3 70B at fp16: 70B × 2 = 140 GB → need 2× A100 80GB
GPT-4 (1.8T) at int8: 1800B × 1 = 1.8 TB → 24× A100 80GB
```

### 8.2 Training Memory

During training you need:
```
Weights:  P × 4 bytes (fp32) or P × 2 (bf16 mixed precision)
Gradients: P × 4 bytes 
Optimizer state (Adam): 2 × P × 4 bytes  (momentum + variance)
Total: ~16 bytes/param for fp32, ~18 bytes/param for mixed precision

LLaMA 3 8B training: 8B × 18 bytes = 144 GB minimum (weights+gradients+optimizer)
+ activations for backprop
→ typically need QLoRA or model parallelism
```

### 8.3 Quick Mental Math for Interviews

```
"How much memory to serve LLaMA 3 70B with batch_size=1, context=4K tokens?"

Weights: 70B × 2 bytes (fp16) = 140 GB
KV cache: 2 × 80 layers × 8 KV heads × 128 head_dim × 4096 tokens × 2 bytes
        = 2 × 80 × 8 × 128 × 4096 × 2 = 10.7 GB
Total: ~151 GB → need 2× A100 80GB minimum
```

---

## Part 9 — Interview Q&A (25 Questions)

**Q1: What is the computational complexity of self-attention?**
> O(n² × d) in time and O(n²) in space where n = sequence length, d = model dimension. The n² comes from the QKᵀ matrix: each of n queries attends to each of n keys. This is why context scaling is hard.

**Q2: What does FlashAttention improve and by how much?**
> Memory: from O(n²) to O(n). Speed: 2–4× on A100 (memory bandwidth savings). It does NOT change the mathematical result — same output as standard attention, just computed differently (tiling in SRAM). Required for long-context models.

**Q3: Explain the KV cache. Why doesn't the Q need caching?**
> Q (query) represents "what does this token want to know?" — it's unique per generation step and used immediately. K and V represent information from the sequence that must be compared to current Q. Before the next token, those K, V would be recomputed identically, so we cache them. Q is not reused.

**Q4: What is GQA and why is it preferred over MHA at inference?**
> Grouped Query Attention: multiple Q heads share a single K and V head. For LLaMA 3 8B, 32 Q heads share 8 KV heads (4:1 ratio). This reduces KV cache memory by 4×. Quality loss is < 1% on standard benchmarks vs full MHA.

**Q5: How does RoPE enable context length extrapolation?**
> Standard absolute positions can only extrapolate poorly (position 5000 was never seen during training if max was 4096). RoPE encodes relative positions via rotation — the model uses m-n distance, not absolute position m. With adjustments (YaRN, LongRoPE), the rotation frequencies can be rescaled to handle longer sequences.

**Q6: You have an A100 80GB GPU. What's the largest model you can serve?**
> At fp16: 80 GB / 2 bytes per param = 40B parameters maximum (weights only). Realistically ~35B accounting for KV cache and activations. In practice: LLaMA 3 70B (140 GB) needs 2× A100s. Run LLaMA 3 8B (16 GB) + 30B+ tokens of KV cache on single A100.

**Q7: What is quantisation and what are the trade-offs?**
> Representing weights in lower precision (fp16→int8→int4). Reduces memory by 2–4×, enables larger models or larger batches on same hardware. Trade-off: slight quality loss (typically < 2% on benchmarks for INT8). GGUF (llama.cpp), AWQ, GPTQ are quantisation formats. INT4 with grouping (Q4_K_M in GGUF) is widely used for local inference.

**Q8: What is the difference between model parallelism and tensor parallelism?**
> - **Pipeline parallelism:** Split model by layers across GPUs (GPU 1 has layers 1–12, GPU 2 has 13–24). Bubble problem: GPU 2 waits while GPU 1 is running.
> - **Tensor parallelism:** Split individual weight matrices across GPUs (row/column splitting). Each GPU has all layers but partial weights. Used in Megatron-LM.
> - **Data parallelism:** Each GPU holds full model copy, processes different data batches. Gradient synchronisation required.
> Typically combined: tensor × pipeline × data parallelism.

**Q9: What is PagedAttention? (Important for vLLM)**
> Inspired by OS virtual memory paging. KV cache blocks are allocated in fixed-size "pages" (typically 16 tokens). Physical memory is allocated on demand, not pre-allocated. Fragmentation drops from ~60% to < 4%. This is the key innovation in vLLM that enables much higher throughput.

**Q10: What happens during the prefill phase vs decode phase?**
> **Prefill:** Process entire prompt at once (batched matrix multiply — very GPU-efficient). Fills KV cache. Memory-bound for short prompts, compute-bound for long.
> **Decode:** Generate one token at a time. KV cache read + single token attention. Memory-bandwidth-bound (reading large KV cache for small computation). This is the bottleneck for throughput.

**Q11: What is speculative decoding?**
> Use a small "draft" model to quickly generate k tokens, then use the large model to verify them in a single parallel forward pass. If verified, accept; otherwise reject and resampling. Wall-clock speedup: 2–3× for chat-length outputs. Used by Claude, Gemini internally.

**Q12: Explain the "lost in the middle" problem.**
> Studies show LLMs perform best when relevant information is at the beginning or end of a long context. Information in the middle is less attended to. This is a practical concern for RAG: don't assume more context is always better. Reranking or position-aware chunking can help.

**Q13: What is Mixture of Experts (MoE) and when does it make sense?**
> MoE uses N expert FFNs but activates only k (typically 2) per token. Result: N× more parameters but same inference cost as a dense model with N/k experts' worth of parameters. Best when: you need high capability (large parameter count for knowledge), compute-efficient inference, and can afford all weights in memory. Mixtral 8x7B has quality close to a 46B dense model at the inference cost of ~13B.

**Q14: What is the difference between perplexity and accuracy for LLMs?**
> - **Perplexity:** The geometric mean inverse probability the model assigns to test tokens. Lower is better. Used to measure general language modeling quality. PPL = exp(average cross-entropy loss).
> - **Accuracy:** Fraction of correct answers on specific benchmarks (MMLU, HumanEval, etc.)
> For LLM evaluation, task-specific benchmarks (MMLU, HellaSwag, ARC) are more meaningful than perplexity for downstream use.

**Q15: What is context utilisation efficiency?**
> Some models are trained with long contexts but perform poorly beyond a fraction of that length. Check: Does retrieval accuracy drop for documents at position > 50% of context? If yes, use RAG with chunking rather than relying on naive long-context stuffing.

**Q16: What are the 3 most common LLM inference optimisations?**
> 1. **KV cache** — reduces decode from O(n²) to O(n), essential always
> 2. **FlashAttention** — O(n) memory attention, enables long context and large batches
> 3. **Quantisation (INT8/INT4)** — reduces memory 2–4×, enables larger models on same hardware
> Others: continuous batching, speculative decoding, tensor parallelism.

**Q17: How does continuous batching (iteration-level scheduling) work?**
> Traditional: batch all requests, wait for all to finish before new requests join (long requests block short ones). Continuous batching (vLLM): after each decode step, check if any request in the batch finished — immediately inject new requests to fill the gap. GPU utilisation improves from ~40% to ~85%.

**Q18: What is the TTFT and TBT metric for LLM serving?**
> - **TTFT (Time to First Token):** Latency from user request to first token generated. Measures prefill speed. User experience: the loading indicator duration.
> - **TBT (Time Between Tokens):** Latency per decode step. Determines the typing speed the user sees.
> FAANG metric: "P99 TTFT < 2 seconds, TBT < 50ms" is a reasonable target.

**Q19: What is a system prompt and how is it cached?**
> System prompt: instructions prepended to every request (e.g. "You are a helpful assistant"). For high-volume applications, pre-computing and caching the system prompt's KV vectors (called "prompt caching" in Anthropic/OpenAI APIs) saves prefill cost on every request. Anthropic charges 10% of normal cost for cached tokens.

**Q20: What is beam search and when do you use it?**
> Maintain k (beam width) best partial sequences instead of greedy single path. Each step expands all k beams with all vocabulary words, keeps top-k overall. Final output: best scoring complete sequence. Used in translation (accuracy > diversity). NOT used for chat/creative tasks (too expensive, produces overly safe/boring outputs). Greedy or sampling preferred for LLMs.

**Q21: What's the "softmax bottleneck" in language models?**
> Output logits are a product of the last hidden state and the embedding matrix: `logits = h W_e^T`. The rank of this product is at most min(d_model, V). For low d_model and large vocabulary V, the model cannot represent arbitrary probability distributions. This limits very small models' expressive power. Solutions: factorised softmax, mixture of softmax.

**Q22: Explain weight tying in language models.**
> The input embedding matrix and output projection matrix share the same weights (`lm_head.weight = embed.weight`). Intuition: the token's input representation and its output representation should be similar. This reduces parameters by `V × d_model` (e.g. 50K × 768 = 38.4M for GPT-2). Most models use this. LLaMA 3 does NOT (vocabulary too large at 128K, different representations needed).

**Q23: What is gradient checkpointing (activation recomputation)?**
> During backprop, activations from forward pass must be stored to compute gradients. For 70B model with long sequence, this can exceed GPU memory. Gradient checkpointing: discard activations during forward pass, recompute them when needed during backward. Trade-off: 30% more compute, ~3× less activation memory. Standard practice for training large models.

**Q24: What is the difference between INT8 and INT4 quantisation (QLoRA context)?**
> INT8: 8-bit integers. 2× compression from fp16. ~1% quality loss. Used in llm.int8() (bitsandbytes). INT4: 4-bit integers. 4× compression. Larger quality loss, mitigated by group quantisation (different scales per group of 64 weights). QLoRA uses NF4 (Normal Float 4) — a non-uniform quantisation that better represents normally distributed weights.

**Q25: You're designing a system to serve LLaMA 3 70B to 1000 concurrent users with < 3 second TTFT. What's your architecture?**
> 1. Model: LLaMA 3 70B in INT8 (~70 GB) on 2× A100 80GB, tensor parallel
> 2. Inference: vLLM with continuous batching and PagedAttention
> 3. Request routing: Load balancer → multiple vLLM replicas
> 4. Prefill optimisation: Common system prompt cached (KV prefix caching)
> 5. Scale: Monitor GPU utilisation + TTFT; add replicas as load grows
> 6. Fallback: 8B model for simple queries (routing by complexity classifier)

---

## Part 10 — Architecture Comparison Cheatsheet

| Model | Params | Context | Positional | Attention | Activation | Open? |
|---|---|---|---|---|---|---|
| GPT-2 | 1.5B | 1K | Learned | MHA | GELU | ✅ |
| GPT-3 | 175B | 4K | Learned | MHA | GELU | ❌ |
| LLaMA 1 | 7–65B | 2K | RoPE | MHA | SiLU | ✅ |
| LLaMA 2 | 7–70B | 4K | RoPE | GQA | SiLU | ✅ |
| LLaMA 3 8B | 8B | 8K | RoPE | GQA(32/8) | SwiGLU | ✅ |
| LLaMA 3 70B | 70B | 8K | RoPE | GQA(64/8) | SwiGLU | ✅ |
| Mistral 7B | 7B | 32K | RoPE | GQA(32/8) | SwiGLU | ✅ |
| Mixtral 8x7B | 47B | 32K | RoPE | GQA | SwiGLU + MoE | ✅ |
| Gemma 2 2B | 2B | 8K | RoPE | MQA | GELU | ✅ |
| Phi-3 Mini | 3.8B | 128K | RoPE | MHA | GELU | ✅ |

---

## Part 11 — Hugging Face Transformers Practical Guide

### 11.1 Loading & Using Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model + tokenizer
model_name = "meta-llama/Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",           # Auto-distribute across GPUs
    load_in_4bit=True,           # QLoRA quantization
    attn_implementation="flash_attention_2"
)

# Simple pipeline API
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = generator("Explain RAG in one sentence:", max_new_tokens=100, temperature=0.7)

# Chat template (instruction-tuned models)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is attention?"}
]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
output = model.generate(input_ids, max_new_tokens=256, do_sample=True, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### 11.2 Key HF Classes

| Class | Purpose | Example |
|---|---|---|
| `AutoModel` | Base model (embeddings only) | Feature extraction, similarity |
| `AutoModelForCausalLM` | Text generation | GPT, LLaMA, Mistral |
| `AutoModelForSequenceClassification` | Classification | Sentiment, toxicity |
| `AutoModelForTokenClassification` | NER | Entity extraction |
| `AutoModelForQuestionAnswering` | Extractive QA | BERT-based QA |
| `AutoTokenizer` | Tokenization | BPE, WordPiece |

### 11.3 Tokenizer Deep Dive

```python
# Tokenizer internals
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")

text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)        # ['Hello', ',', ' how', ' are', ' you', '?']
ids = tokenizer.encode(text)             # [15043, 11, 1268, 527, 499, 30]
decoded = tokenizer.decode(ids)          # "Hello, how are you?"

# Special tokens
print(tokenizer.bos_token)    # <|begin_of_text|>
print(tokenizer.eos_token)    # <|end_of_text|>
print(tokenizer.pad_token)    # None (set manually for batching)

# Batch encoding with padding
batch = tokenizer(
    ["Short text", "This is a longer piece of text"],
    padding=True, truncation=True, max_length=512, return_tensors="pt"
)
# batch.input_ids, batch.attention_mask
```

### 11.4 Model Hub & Datasets

```python
from datasets import load_dataset
from huggingface_hub import HfApi

# Load dataset
dataset = load_dataset("squad", split="train[:1000]")

# Push model to Hub
model.push_to_hub("my-org/my-fine-tuned-model")
tokenizer.push_to_hub("my-org/my-fine-tuned-model")

# Search models
api = HfApi()
models = api.list_models(filter="text-generation", sort="downloads", direction=-1, limit=10)
```

---

## Part 12 — Vision Transformers (ViT)

### 12.1 How ViT Works

Standard Transformers process sequences of tokens. ViT treats an image as a sequence of patches:

1. **Split image into patches**: 224×224 image → 16×16 patches → 196 patches
2. **Linear projection**: Each patch (16×16×3 = 768 pixels) → embedding vector via linear layer
3. **Add [CLS] token**: Prepend a learnable classification token
4. **Add positional embeddings**: Learnable 1D position embeddings for each patch
5. **Transformer encoder**: Standard multi-head attention + FFN blocks
6. **Classification head**: [CLS] token output → MLP → class prediction

```
Image (224×224) → Split into patches (14×14 grid of 16×16 patches)
  → Linear Projection (patch → embedding)
  → [CLS] + Patch Embeddings + Position Embeddings
  → Transformer Encoder (L layers)
  → [CLS] output → Classification Head → Prediction
```

### 12.2 ViT vs CNN

| Aspect | CNN (ResNet) | ViT |
|---|---|---|
| Inductive bias | Local patterns (convolutions) | Global attention (no locality bias) |
| Data efficiency | Better with small data | Needs large data (or pre-training) |
| Scalability | Limited by depth | Scales well with compute |
| Resolution | Fixed architecture | Flexible (change patch count) |
| Feature extraction | Hierarchical (local → global) | Global from layer 1 |

### 12.3 Modern Vision Models Using ViT

- **CLIP** (OpenAI): ViT + text encoder, trained on image-text pairs → zero-shot classification
- **DINO/DINOv2** (Meta): Self-supervised ViT → excellent features without labels
- **SAM** (Segment Anything): ViT-based image encoder for segmentation
- **LLaVA**: ViT encoder + LLM decoder → multimodal understanding
- **GPT-4V / Gemini**: Proprietary vision encoders integrated with LLMs

### 12.4 ViT in Practice

```python
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image

# Load pre-trained ViT
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Classify image
image = Image.open("cat.jpg")
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(-1).item()

# Use CLIP for zero-shot (multimodal)
from transformers import CLIPProcessor, CLIPModel

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"],
    images=image, return_tensors="pt", padding=True
)
outputs = clip(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)  # [0.95, 0.05]
```

---

## Part 13 — Advanced Mixture of Experts (MoE) Patterns

### 13.1 MoE Architecture Details

```
Input → Router (Gating Network) → Top-K Experts → Weighted Sum → Output

Router: softmax(W_gate · x) → select top-2 experts
Expert 1: FFN(x) × weight_1
Expert 2: FFN(x) × weight_2
Output: weight_1 × Expert_1(x) + weight_2 × Expert_2(x)
```

**Key Design Decisions:**
- **Number of experts**: 8 (Mixtral), 16, 64 (GPT-4 rumoured)
- **Top-K selection**: Usually K=2 (balance compute vs quality)
- **Load balancing loss**: Prevent expert collapse (all tokens → same expert)

### 13.2 MoE Models in Production

| Model | Experts | Active | Total Params | Active Params |
|---|---|---|---|---|
| Mixtral 8×7B | 8 | 2 | 47B | ~13B |
| Mixtral 8×22B | 8 | 2 | 141B | ~39B |
| DBRX | 16 | 4 | 132B | ~36B |
| Grok-1 | 8 | 2 | 314B | ~86B |

### 13.3 MoE Trade-offs

**Pros**: More parameters without proportional compute cost. Better quality per FLOP. Specialization (experts learn different skills).

**Cons**: Higher memory (all expert weights loaded). Load balancing challenge. Harder to quantize (some experts rarely activated). Tensor parallelism more complex.

**Interview Key Point**: "MoE gives you a larger model's quality at a smaller model's inference cost, because only a subset of experts activate per token."

---

## 📚 Further Resources

### Must Watch/Read This Week
| Resource | Link | Time |
|---|---|---|
| **Let's Build GPT** (Karpathy) — Week 3, but code it yourself this week | https://youtu.be/kCc8FmEb1nY | 2 hrs |
| **FlashAttention Paper** | https://arxiv.org/abs/2205.14135 | 30 min read |
| **GQA Paper** | https://arxiv.org/abs/2305.13245 | 20 min read |
| **Lilian Weng — Large Transformer Model** | https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/ | 1 hr |

### Books
- **"Build a Large Language Model (From Scratch)"** — Sebastian Raschka
  - Chapter 3: Attention mechanism from scratch
  - Chapter 4: GPT architecture implementation

### Project Task (This Weekend)
Build a minimal GPT from scratch (following Karpathy's video):
1. Implement multi-head attention with causal masking
2. Add positional embeddings
3. Stack transformer blocks with residuals + layer norm
4. Train on a small text dataset (Shakespeare)
5. Add KV cache for inference

```python
# Starter code structure:
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        ...
    def forward(self, tokens, kv_cache=None):
        # 1. Embed tokens + positions
        # 2. Run through N transformer blocks
        # 3. Final layer norm
        # 4. Project to vocabulary logits
```

> ✅ **End of Month 1 Core Content.** The sections below add depth for day-to-day production work.

---

## Part 10 — Day-to-Day Work: Transformer Knowledge in Practice

### 10.1 Production Model Debugging — When the Model Gives Wrong Answers

```python
# Debugging checklist for "model gives bad answers":
# 1. CHECK CONTEXT: Is the relevant info actually in the prompt?
# 2. CHECK TOKENS: Are you exceeding the context window? (silent truncation!)
# 3. CHECK TEMPERATURE: Is it too high (random) or too low (repetitive)?
# 4. CHECK SYSTEM PROMPT: Is the instruction clear and unambiguous?
# 5. CHECK FORMAT: Is the model outputting in the right format?

import tiktoken

def debug_prompt(messages, model="gpt-4o"):
    """Debug tool: check prompt before sending."""
    enc = tiktoken.encoding_for_model(model)
    
    total_tokens = 0
    for msg in messages:
        msg_tokens = len(enc.encode(msg["content"]))
        total_tokens += msg_tokens
        print(f"  [{msg['role']}]: {msg_tokens} tokens")
    
    model_limits = {"gpt-4o": 128000, "gpt-4o-mini": 128000, "gpt-3.5-turbo": 16385}
    limit = model_limits.get(model, 128000)
    
    print(f"\n  Total: {total_tokens} tokens ({total_tokens/limit*100:.1f}% of {limit} limit)")
    if total_tokens > limit * 0.9:
        print("  ⚠️  WARNING: >90% context used. May truncate or degrade quality.")
    if total_tokens > limit:
        print("  ❌ ERROR: Exceeds context window! Prompt will be truncated.")

# Usage before any LLM call:
# debug_prompt(messages, model="gpt-4o-mini")
```

### 10.2 Choosing the Right Attention/Architecture for Your Task

```
Decision tree for production model selection:

Q: Need text GENERATION (chat, completion, code)?
  → Decoder-only: GPT-4o, Claude, LLaMA 3, Mistral

Q: Need text UNDERSTANDING (classification, NER, search)?
  → Encoder-only: BERT, DistilBERT, RoBERTa (much faster, cheaper)
  → For production classification: DistilBERT + small head = <100ms inference

Q: Need text-to-text (translation, summarisation with fixed format)?
  → Encoder-decoder: T5, BART (less common now, decoder-only handles this too)

Q: Long documents (>32K tokens)?
  → Must have RoPE + FlashAttention: LLaMA 3 (128K), Claude (200K), GPT-4o (128K)

Q: Serving many concurrent users cheaply?
  → GQA models (LLaMA 3, Mistral) — smaller KV cache = more concurrent users per GPU
  → vLLM with PagedAttention (covered in Month 8)

Q: Very large model, limited GPUs?
  → MoE models: Mixtral 8x7B (46B total, 12.9B active — same cost as 13B dense)
```

### 10.3 Memory Planning for Production Deployments

```python
def plan_deployment(
    model_params_b: float,  # e.g. 8.0 for LLaMA 3 8B
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    max_batch_size: int,
    precision: str = "fp16"  # fp16, int8, int4
):
    """Calculate GPU memory needed — use this before every deployment."""
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
    bpp = bytes_per_param[precision]
    
    # Model weights
    weight_gb = model_params_b * 1e9 * bpp / (1024**3)
    
    # KV cache (fp16 always, even if model is quantized)
    kv_per_token = 2 * num_layers * num_kv_heads * head_dim * 2  # 2 bytes for fp16
    kv_total_gb = kv_per_token * max_seq_len * max_batch_size / (1024**3)
    
    # Overhead (activations, framework)
    overhead_gb = max(1.0, weight_gb * 0.1)
    
    total = weight_gb + kv_total_gb + overhead_gb
    
    print(f"Model weights ({precision}): {weight_gb:.1f} GB")
    print(f"KV cache ({max_batch_size} users × {max_seq_len} tokens): {kv_total_gb:.1f} GB")
    print(f"Overhead: {overhead_gb:.1f} GB")
    print(f"Total: {total:.1f} GB")
    print()
    
    gpus = {"A100-40": 40, "A100-80": 80, "H100": 80, "L4": 24, "T4": 16, "RTX4090": 24}
    for name, mem in gpus.items():
        n_needed = max(1, int(total / (mem * 0.9)) + (1 if total % (mem * 0.9) > 0 else 0))
        print(f"  {name} ({mem}GB): {n_needed} GPU(s)")

# Example: LLaMA 3 8B serving 16 concurrent users, 4K context
plan_deployment(8.0, 32, 8, 128, 4096, 16, "fp16")
# Weights: 16 GB, KV: ~8.6 GB, Total: ~26.2 GB → 1× A100-40 (tight) or 1× A100-80
```

### 10.4 A/B Testing LLM Configurations

```python
# Day-to-day: you'll constantly A/B test different configs
# Model A (gpt-4o, temp=0) vs Model B (gpt-4o-mini, temp=0.3)

import random
import time

class LLMExperiment:
    def __init__(self, variants):
        self.variants = variants  # {"control": config_a, "treatment": config_b}
        self.results = {"control": [], "treatment": []}
    
    def assign_variant(self, request_id: str) -> str:
        """Deterministic variant assignment using request_id hash."""
        return "treatment" if hash(request_id) % 2 == 0 else "control"
    
    def run(self, request_id, messages):
        variant = self.assign_variant(request_id)
        config = self.variants[variant]
        
        start = time.time()
        response = call_llm(messages, model=config["model"], temperature=config["temp"])
        latency = time.time() - start
        
        self.results[variant].append({
            "request_id": request_id,
            "latency": latency,
            "response_len": len(response),
            "variant": variant
        })
        return response, variant

# Usage:
experiment = LLMExperiment({
    "control": {"model": "gpt-4o", "temp": 0},
    "treatment": {"model": "gpt-4o-mini", "temp": 0.3}
})
```

### 10.5 Building an LLM Wrapper Service (Common First Task at a New Job)

```python
# Most AI engineering teams build an internal LLM service layer:
# - Unified API for all models (OpenAI, Anthropic, local)
# - Automatic retries, fallbacks, circuit breakers
# - Cost tracking, rate limiting, audit logging
# - PII masking before external API calls

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, messages: list, **kwargs) -> LLMResponse:
        pass

class OpenAIProvider(LLMProvider):
    def complete(self, messages, model="gpt-4o-mini", **kwargs):
        import openai, time
        start = time.time()
        client = openai.OpenAI()
        resp = client.chat.completions.create(model=model, messages=messages, **kwargs)
        latency = (time.time() - start) * 1000
        
        usage = resp.usage
        cost = self._calculate_cost(model, usage.prompt_tokens, usage.completion_tokens)
        
        return LLMResponse(
            content=resp.choices[0].message.content,
            model=model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_ms=latency,
            cost_usd=cost
        )
    
    def _calculate_cost(self, model, input_tok, output_tok):
        rates = {
            "gpt-4o": (2.50, 10.00),
            "gpt-4o-mini": (0.15, 0.60),
        }
        inp_rate, out_rate = rates.get(model, (0, 0))
        return (input_tok / 1e6 * inp_rate) + (output_tok / 1e6 * out_rate)

class LLMService:
    """Unified LLM service with fallbacks and cost tracking."""
    def __init__(self, primary: LLMProvider, fallback: LLMProvider = None):
        self.primary = primary
        self.fallback = fallback
        self.total_cost = 0.0
    
    def complete(self, messages, **kwargs) -> LLMResponse:
        try:
            resp = self.primary.complete(messages, **kwargs)
        except Exception as e:
            if self.fallback:
                resp = self.fallback.complete(messages, **kwargs)
            else:
                raise
        self.total_cost += resp.cost_usd
        return resp
```

---

> ✅ **End of Month 1!** You've covered core DSA patterns AND transformer internals. Month 2 begins with Trees, Binary Search, and Production RAG systems.
