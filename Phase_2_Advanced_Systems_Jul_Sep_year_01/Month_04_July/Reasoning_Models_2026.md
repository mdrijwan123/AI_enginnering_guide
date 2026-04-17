# Reasoning Models & 2026 AI Landscape
### Supplementary Guide | Phase 2 | Study Reference

> This guide covers the new class of "reasoning models" that have emerged in 2024–2026, which are critical knowledge for FAANG LLM engineer interviews and production AI systems.

---

## Part 1 — What Are Reasoning Models?

### 1.1 The Shift from Completion to Reasoning

> 💡 **ELI5 (Explain Like I'm 5):**
> If I ask you, "What is 2 + 2?", you instantly say "4" without thinking. That's a standard LLM. But if I ask you, "What is 17 × 23?", you have to pause, pull out a mental chalkboard, multiply the numbers step-by-step, and *then* give me the answer. That is a **Reasoning Model**. It doesn't just blurt out the first thing that comes to mind; it explicitly stops to "think" before answering.

Standard LLMs (GPT-4, Claude 3, Llama 3) generate tokens in a single forward pass — they produce an answer immediately without explicit "thinking" steps.

**Reasoning models** introduce a deliberate _thinking phase_ before producing the final answer:

```
Standard LLM flow:
  User: "What is 17 × 23?"
  Model: "391"   ← immediate output

Reasoning model flow:
  User: "What is 17 × 23?"
  <thinking>
    17 × 23 = 17 × 20 + 17 × 3
           = 340 + 51
           = 391
  </thinking>
  Model: "391"   ← final answer after thinking
```

The key innovation: the model **searches over a chain of reasoning steps** rather than committing to the first plausible continuation.

---

### 1.2 Chain-of-Thought (CoT) vs Reasoning Models

| Technique | Mechanism | Control | Training |
|---|---|---|---|
| Zero-shot CoT | Prompt: "Think step by step" | External | Standard LLM |
| Few-shot CoT | Provide examples with reasoning | External | Standard LLM |
| Self-consistency | Sample N chains, vote on answer | External | Standard LLM |
| **Reasoning Models (o1-style)** | Internal chain-of-thought trained via RL | Internal | RL on reasoning traces |
| **Extended Thinking (Claude)** | Visible thinking block before answer | Internal | RLHF + Constitutional AI |

**The key difference:** Reasoning models are **trained** to think, not just prompted — internal reasoning is reinforced via RL, not injected via prompting.

---

## Part 2 — Major Reasoning Models (2024–2026)

### 2.1 OpenAI o1 / o3 Series

**o1 (September 2024):** OpenAI's first reasoning model.
```
Architecture insight:
- "Test-time compute" scaling: more inference compute → better answers
- Trained with RL where the reward signal is answer correctness
- Internal chain-of-thought is NOT shown to users (opaque)
- Uses beam search / MCTS-like tree search during generation
- Performance: beats PhD-level on GPQA, competitive on competition math

Key tradeoff:
  + Much better at multi-step math, coding, science
  - Much slower (10–30s vs <1s for GPT-4o)
  - More expensive per token
```

**o3 (2025):** Significant improvement, ARC-AGI benchmark breakthrough.
```
ARC-AGI score:
  GPT-4o:   ~5%
  o1:       ~32%
  o3 high:  ~87.5%  ← near human level on novel reasoning tasks

Cost: o3 "high" mode uses ~172× more compute than o3 "low"
This demonstrates test-time compute scaling (vs training-time scaling)
```

**o3-mini / o4-mini:** Smaller, faster, cheaper — still reasoning capable.

---

### 2.2 DeepSeek R1 (January 2025)

**Why it's important:** Open-weights reasoning model, competitive with o1, trained via pure RL.

```
Training recipe (DeepSeek R1):
1. Start from DeepSeek-V3 base (mixture-of-experts)
2. Apply Group Relative Policy Optimisation (GRPO) — a PPO variant
3. Reward signal: correctness of final answer (no human preference data needed)
4. The model spontaneously developed "chain-of-thought" reasoning!

Key finding: You don't need human-annotated reasoning traces.
  Simply rewarding correct answers causes the model to develop reasoning.
```

**GRPO (Group Relative Policy Optimisation):**
```python
# GRPO vs PPO:
# PPO: needs a value function (critic model), separate from policy model
# GRPO: uses the group average reward as the baseline — simpler!

# For a set of G outputs {o1, o2, ..., oG} from one prompt:
reward_i = correctness(o_i)  # 1 if correct, 0 if wrong
baseline = mean(reward for all G outputs)
advantage_i = reward_i - baseline

# Gradient pushes policy toward outputs with above-average rewards
# No separate value model needed → saves memory, simpler to train
```

**R1 output format:**
```xml
<think>
  Let me work through this step by step.
  First, I notice that...
  Wait, that can't be right because...
  Actually, let me reconsider. The key insight is...
</think>

Final answer: ...
```

The `<think>` block is visible and often long (1000–10000 tokens).

---

### 2.3 Google Gemini 2.0 / 2.5 Flash Thinking

**Gemini 2.5 Flash (2025):** Google's reasoning model with "thinking budget" control.

```python
# Control thinking depth with budget tokens
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=question,
    config=GenerateContentConfig(
        thinking_config=ThinkingConfig(
            thinking_budget=2048  # tokens for internal reasoning
            # 0 = no thinking (fast), 24576 = maximum thinking
        )
    )
)
print(response.candidates[0].content.parts[0].text)  # final answer
print(response.usage_metadata.thoughts_token_count)   # thinking tokens used
```

---

### 2.4 Anthropic Claude Extended Thinking

**Claude 3.7 Sonnet (February 2025):** First Claude model with extended thinking.

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # up to 10K tokens of internal thinking
    },
    messages=[{
        "role": "user",
        "content": "Prove that sqrt(2) is irrational."
    }]
)

# Response has two blocks:
for block in response.content:
    if block.type == "thinking":
        print("THINKING:", block.thinking)   # internal reasoning
    elif block.type == "text":
        print("ANSWER:", block.text)          # final response
```

**When to use extended thinking:**
- Complex coding with multiple edge cases
- Multi-step mathematical proofs
- Strategic planning with many constraints
- Debugging subtle bugs (introspective reasoning helps)

**When NOT to use:**
- Simple Q&A or factual lookups
- High-throughput production (10–30× slower)
- Cost-sensitive applications

---

### 2.5 QwQ-32B (Alibaba Qwen, 2025)

Open-weights reasoning model from Alibaba.
```
Size: 32B parameters
Reasoning approach: visible chain-of-thought, self-questioning style
Performance: competitive with o1-mini on MATH and AIME benchmarks
License: Apache 2.0 (freely usable commercially)

Characteristic style:
  "Wait, let me reconsider..." 
  "Hmm, but that contradicts..."  
  "Actually, I think the key insight is..."
  → verbose but thorough self-correction
```

---

## Part 3 — Production Implications

### 3.1 When to Use Reasoning Models vs Standard LLMs

```
Problem characteristics → Model choice:

Multi-step math / proofs              → o3, R1, Claude extended thinking
Competitive coding (Hard LC)          → o3, R1
Scientific analysis                   → o1/o3
Strategic planning (complex)          → Claude 3.7 thinking
Simple classification / generation    → GPT-4o, Claude 3.5 Sonnet, Gemini 2.0 Flash
High-throughput (>100 req/s)          → Llama 3.3 70B, Gemini 2.0 Flash
Embedded / edge deployment            → Llama 3.2 3B/1B, Qwen 2.5 3B
RAG (knowledge retrieval)             → Claude 3.5 Sonnet, GPT-4o (no need for thinking)
```

### 3.2 Cost vs Reasoning Quality Trade-off

```
Model                    Cost/1M tokens   Reasoning quality   Speed
─────────────────────────────────────────────────────────────────────
GPT-4o                   $5 in/$15 out    ████░░  Good         Fast
o3-mini                  $1.1/$4.4        ████░░  Good         Fast
o1                       $15/$60          █████░  Excellent    Slow
o3                       ~$75/$??         ██████  Best         Very slow
Claude 3.5 Sonnet        $3/$15           ████░░  Good         Fast
Claude 3.7 (thinking)    $3/$15+thinking  █████░  Excellent    Slow when thinking
Gemini 2.0 Flash         $0.1/$0.4        ███░░░  Decent       Very fast
Gemini 2.5 Flash         $0.15/$0.6       ████░░  Good         Fast
DeepSeek R1 (API)        $0.55/$2.19      █████░  Excellent    Medium
Llama 3.3 70B (self-host) compute only   ████░░  Good         Depends on HW
```

### 3.3 Implementing Reasoning in Production RAG

```python
# Smart routing: use reasoning model only for complex queries
from anthropic import Anthropic
import re

client = Anthropic()

def classify_complexity(query: str) -> str:
    """Fast classification to route to reasoning vs standard model."""
    prompt = f"""Is this query simple (factual lookup) or complex (multi-step reasoning)?
    Query: {query}
    Answer only: simple or complex"""
    
    response = client.messages.create(
        model="claude-3-haiku-20240307",  # fast/cheap classifier
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip().lower()

def smart_rag_answer(query: str, context: str) -> str:
    complexity = classify_complexity(query)
    
    if complexity == "complex":
        # Use extended thinking for complex reasoning
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=8000,
            thinking={"type": "enabled", "budget_tokens": 5000},
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }]
        )
        # Extract only the text block for the user
        return next(b.text for b in response.content if b.type == "text")
    else:
        # Standard fast response
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }]
        )
        return response.content[0].text
```

---

## Part 4 — Test-Time Compute Scaling

### 4.1 The New Scaling Paradigm

```
2015–2023: Scale training compute → better models (Chinchilla scaling laws)
2024–2025: Scale inference compute → better answers (test-time compute scaling)

OpenAI o3 finding:
  Using 172× more compute at inference time achieves near-human ARC-AGI performance
  vs. previous models that couldn't improve regardless of inference compute
  
Key insight: There are two axes to scale:
  1. Model size (parameters): more expensive training
  2. Inference compute (thinking tokens): more expensive per query
  
For hard problems, scaling inference compute may be more efficient than scaling parameters!
```

### 4.2 Search Algorithms During Inference

Reasoning models use search to explore the solution space:

```
Beam Search:
  Maintain top-K partial solutions at each step
  Expand all K, score each, keep best K
  O(K × vocab_size) per step

Monte Carlo Tree Search (MCTS):
  Simulate many random completions (rollouts)
  Use UCB score: exploitation + exploration
  Tree structure — share computation for common prefixes
  
Best-of-N:
  Sample N complete solutions independently
  Score all N with a process reward model (PRM)
  Return the highest-scoring solution
  
Used in: o1/o3 (likely MCTS-inspired), R1 (best-of-N during RL training)
```

### 4.3 Process Reward Model (PRM) vs Outcome Reward Model (ORM)

```
ORM: Rewards only the final answer (correct/incorrect)
  + Simple to train
  - Can't guide step-by-step reasoning
  - Miss intermediate errors that lead to "right answer, wrong reasoning"

PRM: Rewards each intermediate step
  + Guides reasoning at each step
  + Can catch step-level errors (e.g., "this calculation is wrong")
  - Much harder to collect training data (must label each reasoning step)
  - OpenAI's "Let's Verify Step by Step" paper introduced this

Production use: PRM can re-rank reasoning chains in best-of-N
```

---

## Part 5 — 2026 Model Landscape Overview

### 5.1 Frontier Models (API-only, or Large Self-hosted)

| Model | Provider | Context | Strengths | Best For |
|---|---|---|---|---|
| GPT-4o | OpenAI | 128K | General, vision, speed | Most tasks |
| o3 | OpenAI | 200K | Best reasoning | Hard math, complex coding |
| Claude 3.7 Sonnet | Anthropic | 200K | Code, reasoning, safety | Code review, RAG, agents |
| Claude 3.5 Haiku | Anthropic | 200K | Fast, cheap, smart | High-throughput, classification |
| Gemini 2.0 Ultra | Google | 1M | Largest context, multimodal | Long document, video |
| Gemini 2.5 Flash | Google | 1M | Best price/performance | RAG, chatbots |

### 5.2 Open-Weights Models (Self-hosted / Fine-tunable)

| Model | Params | License | Context | Notable |
|---|---|---|---|---|
| Llama 3.3 70B | 70B | Meta (custom) | 128K | Best open 70B |
| Llama 3.2 3B/1B | 1–3B | Meta | 128K | Edge deployment |
| Mistral Small 3 | 24B | Apache 2.0 | 128K | Strong multilingual |
| Qwen 2.5 72B | 72B | Apache 2.0 | 128K | Code + math |
| DeepSeek R1 | 671B (MoE) | MIT | 128K | Best open reasoning |
| DeepSeek R1 Distill | 7–70B | MIT | 128K | Efficient reasoning |
| Gemma 3 27B | 27B | Gemma (commercial ok) | 128K | Google, strong perf |
| Phi-4 | 14B | MIT | 16K | Small but powerful |

### 5.3 Embedding Models

| Model | Dims | License | Notes |
|---|---|---|---|
| text-embedding-3-large | 3072 | OpenAI API | Best quality |
| text-embedding-3-small | 1536 | OpenAI API | Fast/cheap |
| nomic-embed-text | 768 | Apache 2.0 | Good open model |
| all-MiniLM-L6-v2 | 384 | Apache 2.0 | Fast/small |
| BAAI/bge-m3 | 1024 | MIT | Multilingual, long-context |
| voyage-3 | 1024 | Voyage API | Best retrieval |

---

## Part 6 — Interview Questions on Reasoning Models

**Q: What is "test-time compute scaling" and why does it matter?**

> Test-time compute scaling means using more computation during inference (rather than training) to improve output quality. OpenAI's o3 demonstrated that allocating more tokens for "thinking" — effectively searching over chains of reasoning — dramatically improves performance on hard reasoning tasks. This is significant because it opens a new axis for improvement beyond simply making models bigger or training them longer.

**Q: How does DeepSeek R1 train a reasoning model without human-annotated reasoning traces?**

> R1 uses Group Relative Policy Optimisation (GRPO), a reinforcement learning algorithm that only requires outcome correctness as the reward signal — no step-by-step annotation. During training, for each prompt, multiple responses are sampled. The model receives reward 1 for correct answers and 0 for wrong ones. The GRPO loss encourages the model to generate responses above the group average. Remarkably, through this process alone, the model spontaneously develops chain-of-thought reasoning and self-correction behaviours.

**Q: When would you NOT use a reasoning model in production?**

> Reasoning models incur significantly higher latency (10–30 seconds) and cost (typically 5–15× standard models). They should be avoided for: (1) high-throughput APIs where latency is critical, (2) simple factual queries that don't require multi-step reasoning, (3) cost-sensitive applications, and (4) real-time streaming use cases. The right approach is to route queries by complexity — use reasoning models only when the problem genuinely benefits from extended deliberation.

**Q: What is the difference between a Process Reward Model and an Outcome Reward Model?**

> An ORM scores only the final answer (1 if correct, 0 if wrong). It's simple to train but provides no guidance on intermediate reasoning steps. A PRM scores each step in the reasoning chain, requiring annotators to label intermediate steps as correct or incorrect. PRMs enable better search (you can prune wrong reasoning paths early) and can help identify where a chain of thought went wrong even if the final answer happens to be correct. The trade-off is that PRM training data is expensive and hard to collect.

**Q: How does "extended thinking" in Claude 3.7 differ from prompting "think step by step"?**

> "Think step by step" is a prompting technique that nudges the model to produce visible reasoning in its output — the reasoning and the final answer are interleaved in a single response stream. Claude's extended thinking uses a dedicated `thinking` block with a token budget, separating internal deliberation from the final response. The thinking is trained behaviour reinforced with RL, not just prompted behaviour. Additionally, the thinking block can be shown to users or hidden, and the token budget can be tuned for the complexity/cost trade-off.

---

*Reasoning Models 2026 Guide | Phase 2 Supplementary | Added April 2026*
