# Month 7: Fine-Tuning (SFT, LoRA, QLoRA) + ML System Design Practice
### Phase 3 | October 2026

---

## Week 1–2: Fine-Tuning Theory & Implementation

### Why Fine-Tune vs Prompt Engineer vs RAG?

| Approach | When to Use | Cost | Latency | Privacy |
|---|---|---|---|---|
| Prompt Engineering | Behaviour nudging, no data | $0 | Same | Good (no training data stored) |
| RAG | Knowledge grounding, updatable facts | Low-Med | +100-500ms | Depends on vector DB |
| Fine-Tuning (SFT) | Domain style/format, proprietary behaviour | High upfront | Same (post-train) | Excellent |
| LoRA/QLoRA | SFT with limited GPU | Medium | Same | Excellent |
| RLHF/DPO | Alignment, preference learning | Very High | Same | Excellent |

**Decision framework:**
- External knowledge needed → RAG
- Specific output format/domain tone → Fine-tuning
- Safety/alignment → RLHF or DPO
- All of the above → RAG + Fine-tuned model

---

### Supervised Fine-Tuning (SFT) Fundamentals

**What is SFT?**
Continue training a pre-trained LLM on a curated dataset of (instruction, response) pairs using standard cross-entropy loss.

```
Pre-trained model: P(next_token | all previous tokens) — self-supervised
SFT model: P(response | instruction) — supervised on human-curated data

Example dataset row:
{
  "instruction": "Summarise this into 3 bullet points: [document]",
  "response": "• Key finding 1\n• Key finding 2\n• Key finding 3"
}
```

**SFT Data Format (ChatML format):**
```
<|im_start|>system
You are a helpful assistant specialised in data engineering.<|im_end|>
<|im_start|>user
Write a PySpark job to read parquet files from GCS and aggregate by date.<|im_end|>
<|im_start|>assistant
Here is a PySpark job:
```python
from pyspark.sql import SparkSession
...
```
<|im_end|>
```

**Alpaca format (simpler):**
```python
PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}"""
```

---

### LoRA: Low-Rank Adaptation — Deep Dive

**The Core Problem:** Full fine-tuning of a 7B model requires:
- 7B parameters × 4 bytes (float32) = 28 GB just to store weights
- Optimizer states (Adam): 2× model size = 56 GB
- Gradients: another 28 GB
- Total: ~112 GB. Even 8× A100 80GB is tight.

**LoRA's insight:** The update to a weight matrix W during fine-tuning has a **low intrinsic rank**.

Instead of updating W (d × k matrix), train two small matrices:
```
ΔW = B × A

where:
  A: (r × k), initialised to random Gaussian
  B: (d × r), initialised to zero (so ΔW=0 at start)
  r = rank (typically 4, 8, 16, 64)

Forward pass: h = W₀x + ΔWx = W₀x + BAx
```

**Parameter savings example (LLaMA 3 8B attention layer):**
```
d_model = 4096
Full W_q: 4096 × 4096 = 16.7M parameters
LoRA rank=8: A(8×4096) + B(4096×8) = 32768 + 32768 = 65536 = 0.065M
Savings: 99.6% fewer trainable parameters!
```

**Where to apply LoRA:**
```python
target_modules = [
    "q_proj",   # Query projection
    "k_proj",   # Key projection
    "v_proj",   # Value projection
    "o_proj",   # Output projection
    # Often also:
    "gate_proj",  # MLP gate
    "up_proj",    # MLP up
    "down_proj",  # MLP down
]
# Note: NOT applied to embedding layers or LM head typically
```

**LoRA hyperparameters:**
```
r (rank): 
  - 4-8: light domain adaptation, format learning
  - 16-64: heavier task-specific learning
  - Rule of thumb: start with r=16

alpha (lora_alpha):
  - Scaling factor: ΔW = (alpha/r) × BA
  - Often set equal to r (effective scale = 1.0)
  - Higher alpha = more influence of LoRA on base model

dropout: 0.05-0.1 (regularisation)
```

---

### QLoRA: Quantised LoRA

**QLoRA stacks LoRA on top of a 4-bit quantised base model:**

```
Memory breakdown for LLaMA 3 8B with QLoRA:
  Base model (NF4): 8B × 0.5 bytes = 4 GB
  LoRA adapters (float16): ~0.1 GB (r=16, all attention)
  Activations + gradients: ~2-3 GB
  Total: ~7-8 GB → fits on 1× RTX 4090 (24 GB) or T4 (16 GB)!
```

**QLoRA key innovations:**
1. **NF4 (Normal Float 4):** Special 4-bit quantisation that preserves normal distribution of weights better than INT4
2. **Double quantisation:** Quantise the quantisation constants themselves (saves ~0.5 GB)
3. **Paged optimisers:** Use CPU RAM for Adam optimizer states to prevent GPU OOM spikes

**Full QLoRA Implementation:**
```python
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import torch

# Step 1: Configure 4-bit quantisation
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,    # Double quantisation
    bnb_4bit_quant_type="nf4",          # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16  # Compute in BF16 for speed
)

# Step 2: Load model in 4-bit
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Step 3: Prepare for k-bit training (enables gradient computation on quantised base)
model = prepare_model_for_kbit_training(model)

# Step 4: Configure LoRA
lora_config = LoraConfig(
    r=16,                        # Rank
    lora_alpha=32,               # alpha
    target_modules=[             # Which layers get LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",                 # Don't train bias terms
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Trainable params: 41,943,040 || All params: 8,031,051,776 || Trainable%: 0.5224

# Step 5: Prepare dataset
dataset = load_dataset("json", data_files="your_data.jsonl")

def format_sample(sample):
    return f"""<|im_start|>system
You are a helpful data engineering assistant.<|im_end|>
<|im_start|>user
{sample['instruction']}<|im_end|>
<|im_start|>assistant
{sample['response']}<|im_end|>"""

# Step 6: Training arguments
training_args = TrainingArguments(
    output_dir="./llama3-8b-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,    # Effective batch = 2×4 = 8
    warmup_ratio=0.03,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,                        # BF16 training
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",                # W&B tracking
    optim="paged_adamw_8bit",         # Paged optimiser (QLoRA key feature!)
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
)

# Step 7: SFT Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    formatting_func=format_sample,
    max_seq_length=2048,
    args=training_args,
    packing=False,                    # Set True for efficiency if samples are short
)

trainer.train()

# Step 8: Save adapter (NOT full model — just the delta!)
trainer.model.save_pretrained("./llama3-8b-adapter")
# Takes ~160 MB vs 16 GB for full model

# Step 9: Merge adapter for deployment
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
merged_model = PeftModel.from_pretrained(base_model, "./llama3-8b-adapter")
merged_model = merged_model.merge_and_unload()  # Merge LoRA into base weights
merged_model.save_pretrained("./llama3-8b-merged")
```

---

### Data Preparation for SFT

**How much data do you need?**
```
Task complexity         | Data needed | Notes
------------------------|-------------|-------
Simple format learning  | 500–1K      | e.g. "always answer as JSON"
Domain style/tone       | 1K–5K       | e.g. legal writing style
Complex task learning   | 5K–50K      | e.g. code generation in domain
Full capability         | 100K+       | GPT-level from scratch (not SFT)
```

**Data quality > data quantity.** 500 high-quality examples often beat 10K noisy samples.

**Synthetic data generation:**
```python
from openai import OpenAI

client = OpenAI()

def generate_synthetic_qa(document: str, num_questions: int = 5) -> list:
    """Generate instruction-response pairs from a document."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""
Generate {num_questions} diverse question-answer pairs from this document.
Return as JSON list: [{{"instruction": str, "response": str}}]

Document: {document}
"""
        }],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# Process your documents
training_data = []
for doc in your_documents:
    pairs = generate_synthetic_qa(doc, num_questions=5)
    training_data.extend(pairs)

# Then filter: use reward model or LLM-as-judge to remove low quality
def quality_filter(sample: dict, threshold: float = 0.7) -> bool:
    score_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"""
Rate this Q&A pair on a scale of 0.0-1.0 for:
- Answer accuracy
- Answer completeness
- Instruction clarity

Q: {sample['instruction']}
A: {sample['response']}

Respond with just a float score.
"""}]
    )
    score = float(score_response.choices[0].message.content.strip())
    return score >= threshold
```

---

### Evaluation: How to Know Fine-tuning Worked

```python
from evaluate import load
import json

# 1. Domain-specific benchmark (build yourself!)
test_set = [
    {"instruction": "...", "expected_keywords": ["pipeline", "medallion", "bronze"]},
]

# 2. LLM-as-Judge evaluation
def evaluate_with_llm(instruction, expected, generated):
    judge = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"""
Compare the generated response to the expected response.
Score 1-5 for: Accuracy, Completeness, Style.
Return JSON: {{"accuracy": int, "completeness": int, "style": int, "reasoning": str}}

Instruction: {instruction}
Expected: {expected}
Generated: {generated}
"""}],
        response_format={"type": "json_object"}
    )
    return json.loads(judge.choices[0].message.content)

# 3. Check catastrophic forgetting!
# Run base model benchmarks (MMLU, HellaSwag) on fine-tuned model
# If scores drop >5%, your LoRA rank may be too high or LR too high
```

---

## Week 3–4: ML System Design Practice (5 Full Designs)

### Design 4: Feature Store

**"Design a feature store for a ML platform serving 50 model teams"**

```
Components:
  Offline store: Delta Lake on S3/GCS (historical features, training)
  Online store: Redis / Feast + DynamoDB (low-latency serving, <10ms)
  Feature registry: metadata, lineage, schema
  Materialisation jobs: Spark/Dataflow to sync offline→online
  SDK: feast.get_online_features(entity_rows, features)

Key challenges:
  - Point-in-time correctness (training-serving skew!)
    → Always use timestamp-aware joins in training
    → Log serving features at inference time and rejoin
  - Feature freshness: streaming features via Flink/Kafka
  - Schema evolution: versioned features, backward compatibility

Stack: Feast (open source) or Tecton (managed)
```

### Design 5: Real-Time Fraud Detection with ML

```
Requirements: <100ms decision, 99.99% uptime, 1M transactions/min

Architecture:
  Kafka → Flink (feature extraction) → Redis (features) → Model API → decision
  
Features (Flink, computed in real-time):
  - merchant_7d_fraud_rate
  - user_30d_avg_transaction_amount  
  - user_current_hour_txn_count
  - device_id_first_seen_days_ago
  - velocity: txn count last 60 seconds

Model: Gradient Boosted Trees (XGBoost, <1ms inference)
Fallback: Heuristic rules (always on, even if model fails)

Key design decisions:
  - No LLM here: too slow, too expensive, explainability issues
  - GBDT: fast, explainable (SHAP), robust to missing features
  - Real-time features via Flink: sub-second freshness
  - Model updated weekly: retrain on last 90 days, monitor drift
  - Shadow mode: new model runs in parallel before promotion
```

---

## Interview Q&A — Fine-Tuning

**Q1: What is catastrophic forgetting and how do you mitigate it?**
> When fine-tuning causes the model to lose previously learned knowledge. Mitigations: use low learning rate (1e-4 to 2e-4), small number of epochs (1-3), LoRA (base weights frozen), replay buffer (mix original pretraining data), evaluate on benchmark like MMLU before/after.

**Q2: When would you choose LoRA rank 8 vs rank 64?**
> Rank 8: adapting writing style, output format, specific domain tone — light adaptation needed, fewer trainable params, less overfitting risk. Rank 64: learning complex new capabilities (domain-specific reasoning, new programming language syntax) where the task is further from pretraining distribution.

**Q3: How does QLoRA differ from LoRA? What's the memory saving?**
> QLoRA additionally quantises the base model to 4-bit (NF4), reducing its memory by 8× vs float32. For LLaMA 3 8B: full fine-tuning ~112 GB → LoRA ~40 GB → QLoRA ~8-10 GB, enabling training on a single consumer GPU.

**Q4: How would you build a training dataset for fine-tuning an LLM on your company's data engineering domain?**
> 1. Collect 200-500 real high-quality Q&A from senior engineers. 2. Use GPT-4 to generate 3-5× synthetic variations per real example. 3. Filter with LLM-as-judge (quality score ≥ 0.75). 4. Ensure diversity: different task types, difficulty levels, formats. 5. Hold out 10-15% for evaluation. 6. Check dataset for PII before training.

**Q5: How do you evaluate a fine-tuned model before deployment?**
> 1. Domain benchmark (curated golden set of 200+ Q&A). 2. LLM-as-judge (GPT-4o grades accuracy, completeness 1-5). 3. Human eval on 50-100 examples. 4. Regression: run MMLU / HellaSwag to check no catastrophic forgetting. 5. A/B test in shadow mode before full rollout.

**Q6: What is PEFT and what strategies does it include?**
> PEFT (Parameter-Efficient Fine-Tuning) is a HuggingFace library containing approaches that train a small fraction of parameters: LoRA, QLoRA, Prefix Tuning, Prompt Tuning, IA3, AdaLoRA, LongLoRA. LoRA/QLoRA are most popular because they add no latency overhead at inference after merging.

**Q7: What are paged optimisers and why are they needed in QLoRA?**
> Standard Adam keeps optimizer states (first + second moments) for all trainable parameters in GPU VRAM permanently. Paged Adam transfers optimizer states to CPU RAM when GPU is under pressure (using CUDA unified memory), preventing OOM crashes during training while accepting small CPU↔GPU transfer overhead.

**Q8: Walk me through the math of LoRA.**
> Pre-trained weight matrix W₀ ∈ ℝ^(d×k). Instead of full update ΔW ∈ ℝ^(d×k), LoRA factorises: ΔW = BA where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k). A is Gaussian init, B is zero init (so ΔW = 0 at start, training begins from pre-trained behaviour). During forward pass: h = W₀x + (α/r)BAx. Only A and B are trained. At inference, merge: W = W₀ + (α/r)BA — zero added latency.

---

## 📚 Further Resources

**Books:**
- **"Natural Language Processing with Transformers" by Lewis Tunstall, Leandro von Werra, Thomas Wolf** (HuggingFace team) — Best practical book on fine-tuning
- **"Build a Large Language Model (from scratch)" by Sebastian Raschka** — Deep understanding

**Courses:**
- **DeepLearning.AI: Finetuning Large Language Models** — https://learn.deeplearning.ai/courses/finetuning-large-language-models
- **DeepLearning.AI: Efficiently Serving LLMs** — https://learn.deeplearning.ai/courses/efficiently-serving-llms

**Papers (must-read):**
- **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021) — https://arxiv.org/abs/2106.09685
- **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023) — https://arxiv.org/abs/2305.14314
- **LIMA: Less Is More for Alignment** (Zhou et al., 2023) — https://arxiv.org/abs/2305.11206 (1000 examples beat larger datasets)

**Practice:**
- HuggingFace PEFT docs: https://huggingface.co/docs/peft
- TRL (Transformer Reinforcement Learning): https://huggingface.co/docs/trl
- Axolotl (training framework): https://github.com/axolotl-ai-cloud/axolotl

---

## Day-to-Day Work: Fine-Tuning in Production

### When to Fine-Tune at Work (Decision Framework)

```
START HERE:
  "My LLM application isn't performing well enough"
  
  Q1: Is the model giving WRONG FACTS?
    YES → RAG (inject correct knowledge), NOT fine-tuning
    NO → continue
  
  Q2: Is the model ignoring FORMAT instructions?
    YES → Try structured outputs (JSON mode) first
    Still bad → Fine-tune with 500 format examples
  
  Q3: Is the model's TONE/STYLE wrong for our domain?
    YES → Fine-tune with 1K domain-specific examples
    Example: Legal language, medical reports, retail analytics
  
  Q4: Is the model too SLOW or EXPENSIVE?
    YES → Fine-tune smaller model (7B) to match GPT-4 quality on YOUR task
    This is the #1 reason companies fine-tune: cost reduction
  
  Q5: Is there a PRIVACY concern?
    YES → Fine-tune open-source model → serve on-prem
    No customer data leaves your infrastructure
```

### Real Fine-Tuning Scenarios You'll Execute

```python
# Scenario 1: Train a model to write in your company's analytics style

# Step 1: Collect examples from existing reports
training_examples = [
    {
        "instruction": "Summarize the beverage category performance for Q3 2025",
        "response": """## Beverage Category Performance — Q3 2025

**Key Headline:** Beverages grew +3.2% YoY driven by energy drinks (+12.4%) 
offsetting soft drink decline (-2.1%).

**Top Movers:**
- Energy drinks: +12.4% driven by Monster and Red Bull innovation
- RTD coffee: +8.7% driven by Starbucks canned range
- Soft drinks: -2.1% due to sugar tax impact and own-label switching

**Recommendation:** Increase shelf space allocation for energy drinks by 15%."""
    },
    # ... 1000+ examples
]

# Step 2: Fine-tune LLaMA 3 8B with QLoRA
# (use the QLoRA code from earlier in this file)

# Step 3: Compare outputs
# Base model: generic, verbose, no specific format
# Fine-tuned: matches company style, uses correct metrics, proper format

# Scenario 2: Reduce cost by replacing GPT-4o with fine-tuned 8B
# Before: GPT-4o at $7.50/1M tokens × 100K requests/day = $225/day
# After: Fine-tuned LLaMA 3 8B self-hosted on 1× A100 = $3/hr = $72/day
# Savings: 68% cost reduction with comparable quality on YOUR specific task
```

### Post-Fine-Tuning Deployment Checklist

```
□ Run automated eval suite (RAGAS + custom metrics)
□ Compare against base model on 100 test cases
□ Check for catastrophic forgetting (run MMLU benchmark)
□ Test edge cases: empty input, very long input, adversarial input
□ Benchmark inference speed (tokens/sec) 
□ Calculate serving cost (GPU memory, expected throughput)
□ Merge LoRA adapters (if using LoRA)
□ Upload to model registry (HuggingFace Hub or internal registry)
□ Deploy to staging → canary → production
□ Set up monitoring: quality metrics + operational metrics
□ Document: base model, training data description, hyperparams, eval results
```
