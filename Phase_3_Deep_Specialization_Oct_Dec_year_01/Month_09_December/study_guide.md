# Month 9: General System Design + RLHF / Alignment + Multimodal Models
### Phase 3 | December 2026

---

## Week 1–2: General System Design for AI Engineers

> 📖 **Big picture:** FAANG AI engineer interviews include general system design rounds alongside ML system design. You might be asked to design a rate limiter, a distributed cache, a notification service, or a URL shortener. These aren't ML problems — they're infrastructure problems. But as an AI engineer, you're expected to be fluent in both layers. The databases, queues, and caches you design here are the exact same components that appear in ML feature stores, model serving infrastructure, and RAG pipelines.
>
> **Key insight:** The same distributed systems principles (CAP theorem, consistency vs availability trade-offs, horizontal scaling) govern both traditional databases and vector databases. Understanding them deeply means you can design *any* system, not just ML-specific ones.

### CAP Theorem

> 💡 **ELI5 (Explain Like I'm 5):**
> Suppose you and your partner share a bank account but have two debit cards. The network goes down (Partition). You go to the ATM. Should the ATM show you your balance? If it says "Sorry, network down," it's Consistent (CP) but unavailable. If it gives you the last known balance, it's Available (AP) but maybe incorrect. You can't have both when the network breaks.

**Consistency, Availability, Partition Tolerance — pick two in a distributed system.**

```
Consistency  = Every read gets the most recent write (or an error)
Availability = Every request receives a response (may not be most recent)
Partition Tolerance = System works despite network partitions

CAP states: when a network partition occurs, you must CHOOSE between C and A.
(P is always required in distributed systems — networks fail)

CA (No partition tolerance): Only possible in single-node systems. Not realistic for prod.

CP (Consistent + Partition Tolerant):
  Examples: HBase, Zookeeper, MongoDB (default config), etcd
  Behaviour: Returns error or timeout when partition detected
  Use when: Financial transactions, distributed locks, leader election
  "Correct result or no result"

AP (Available + Partition Tolerant):
  Examples: DynamoDB, Cassandra, CouchDB, DNS
  Behaviour: Returns potentially stale data
  Use when: Social media feeds, shopping carts, product catalogues
  "Always responds, maybe stale"
```

**PACELC (extension of CAP):**
```
Even without partition, there's a trade-off between Latency and Consistency:
  PA/EL: Dynamo, Cassandra — prioritise availability and low latency
  PC/EC: HBase, Spanner — prioritise consistency even at cost of latency
```

**Interview answer pattern:**
> "For [X system], I'd choose [CP/AP] because [reason tied to requirements]. This means [trade-off] which is acceptable because [justification]."

---

### Database Selection Guide

| Database Type | Examples | When to Use |
|---|---|---|
| Relational (SQL) | PostgreSQL, MySQL, CockroachDB | Structured data, ACID, complex joins |
| Document | MongoDB, Firestore | Flexible schema, nested data, prototyping |
| Wide-column | Cassandra, HBase, BigTable | Time-series, write-heavy, massive scale |
| Key-Value | Redis, DynamoDB | Caching, sessions, simple lookups (<1ms) |
| Search | Elasticsearch, OpenSearch | Full-text search, log analysis |
| Vector | Pinecone, Weaviate, pgvector | Embedding similarity search, RAG |
| Time-series | InfluxDB, TimescaleDB | Metrics, IoT, telemetry |
| Graph | Neo4j, Neptune | Relationships, recommendations, knowledge graphs |

**Decision framework:**
```
1. Need ACID + joins? → PostgreSQL
2. Need sub-millisecond reads? → Redis (cache layer)
3. Need to handle billions of events? → Cassandra or BigTable
4. Need full-text search? → Elasticsearch
5. Need vector similarity? → Pinecone / pgvector / Weaviate
6. Simple key-value at massive scale? → DynamoDB
```

---

### Kafka: Event Streaming for ML Systems

**Kafka fundamentals:**
```
Topic: named channel for events (e.g. "user_events", "model_predictions")
Partition: parallelism unit within a topic; each has sorted, immutable log
Producer: writes events to topic
Consumer: reads events from topic (tracks offset, can replay)
Consumer Group: share partitions among consumers for parallel processing
Offset: position in partition log (durable — consumers resume after failure)
Retention: default 7 days (configurable, can be infinite)
```

**Why Kafka in ML systems?**
```
1. Feature pipeline: user_events (Kafka) → Flink (transform) → Feature Store
2. Model prediction logging: App → Kafka("predictions") → Storage → Training data
3. Dead letter queue: failed/rejected requests → Kafka("dlq") → alerting
4. Exactly-once semantics: critical for fraud detection, financial ML
5. Replay: re-process past events with new model version
```

**Kafka for ML — code example:**
```python
from kafka import KafkaProducer, KafkaConsumer
import json

# Producer: log model predictions
producer = KafkaProducer(
    bootstrap_servers=['kafka:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    # Reliability settings
    acks='all',          # Wait for all replicas to acknowledge
    retries=3,
    max_in_flight_requests_per_connection=1  # Prevent reordering
)

def log_prediction(request_id, input_text, prediction, model_version):
    producer.send('model_predictions', {
        'request_id': request_id,
        'timestamp': time.time(),
        'input': input_text,
        'prediction': prediction,
        'model_version': model_version,
        'latency_ms': ...
    })

# Consumer: read predictions for monitoring/retraining
consumer = KafkaConsumer(
    'model_predictions',
    bootstrap_servers=['kafka:9092'],
    group_id='monitoring_service',
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    auto_offset_reset='earliest',       # Start from beginning if no offset
    enable_auto_commit=False            # Manual commit for exactly-once
)

for message in consumer:
    prediction = message.value
    # Check for drift, log to monitoring, etc.
    process_prediction(prediction)
    consumer.commit()  # Only commit after successful processing
```

---

### Redis Patterns for ML Systems

```python
import redis
import pickle
import hashlib, json

r = redis.Redis(host='redis', port=6379, db=0)

# Pattern 1: Semantic Cache for LLM queries
def get_cache_key(messages: list, model: str) -> str:
    canonical = json.dumps({"model": model, "messages": messages}, sort_keys=True)
    return f"llm_cache:{hashlib.md5(canonical.encode()).hexdigest()}"

def cached_llm_call(messages, model="gpt-4o-mini", ttl=3600):
    key = get_cache_key(messages, model)
    cached = r.get(key)
    if cached:
        return pickle.loads(cached)
    
    response = openai.chat.completions.create(model=model, messages=messages)
    result = response.choices[0].message.content
    r.setex(key, ttl, pickle.dumps(result))
    return result

# Pattern 2: Rate limiting (token bucket)
def is_rate_limited(user_id: str, limit: int = 100, window: int = 3600) -> bool:
    key = f"rate_limit:{user_id}"
    pipe = r.pipeline()
    pipe.incr(key)
    pipe.expire(key, window)
    result = pipe.execute()
    count = result[0]
    return count > limit

# Pattern 3: Online feature store
def set_user_features(user_id: str, features: dict, ttl: int = 86400):
    key = f"user_features:{user_id}"
    r.setex(key, ttl, pickle.dumps(features))

def get_user_features(user_id: str) -> dict:
    key = f"user_features:{user_id}"
    data = r.get(key)
    return pickle.loads(data) if data else {}

# Pattern 4: Pub/Sub for streaming results
def stream_completion(request_id: str, channel: str):
    pubsub = r.pubsub()
    pubsub.subscribe(channel)
    for message in pubsub.listen():
        if message['type'] == 'message':
            token = message['data'].decode()
            if token == '[DONE]':
                break
            yield token
```

---

## Week 3: RLHF, DPO & Alignment

> 📖 **Big picture:** Pre-trained models generate text that is statistically likely, not necessarily helpful or safe. A model trained on the internet might learn to produce toxic, deceptive, or harmful content because that content exists in the training data. RLHF and DPO are the alignment techniques that transform a capable but potentially dangerous base model into a helpful, harmless assistant.
>
> **The core insight:** It’s easier for humans to compare two responses ("which is better?") than to write the perfect response from scratch. RLHF exploits this: collect human preferences, train a model to predict preferences (reward model), then optimise the LLM to maximise the reward. DPO is a simplified version that skips the separate reward model entirely.
>
> **Why this matters:** This is why ChatGPT behaves helpfully and refuses harmful requests. Understanding RLHF/DPO is essential for any role working on LLM safety, quality, or alignment at FAANG.

### RLHF: Full Pipeline

> 💡 **ELI5 (Explain Like I'm 5):**
> Imagine an AI chef who knows 10,000 recipes but doesn't know what TASTES good. RLHF is having humans taste the food and say "this one is better." The AI learns a mental model of human taste (the Reward Model) and then optimizes all its cooking to please that palate.

**Step 1: Supervised Fine-Tuning (SFT)**
```python
# Already covered in Month 7
# Fine-tune base model on high-quality (instruction, response) pairs
# Creates SFT model — the starting point for RLHF
```

**Step 2: Reward Model Training**
```python
# Dataset: (prompt, chosen_response, rejected_response) pairs
# Human labellers choose which of two responses is better
# Train classifier to predict score: high = humans prefer it

from transformers import AutoModelForSequenceClassification

reward_model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    num_labels=1  # Single score output
)

# Bradley-Terry loss: probability chosen > rejected
def reward_model_loss(chosen_reward, rejected_reward):
    # Maximise P(chosen preferred) = sigmoid(r_chosen - r_rejected)
    return -F.logsigmoid(chosen_reward - rejected_reward).mean()
```

**Step 3: PPO Fine-tuning**
```python
# Optimise: E[R(response)] - β * KL(π_θ || π_ref)
# Where:
#   R = reward from reward model
#   KL = divergence from SFT model (prevents too much drift)
#   β = KL penalty coefficient (typically 0.1-0.2)

# Libraries: trl (Transformer Reinforcement Learning by HuggingFace)
from trl import PPOConfig, PPOTrainer

ppo_config = PPOConfig(
    model_name="sft_model_path",
    learning_rate=1.41e-5,
    batch_size=512,
    mini_batch_size=128,
    gradient_accumulation_steps=1,
    optimize_device_cache=True,
    early_stopping=True,
    target_kl=0.1,              # Stop if KL divergence too high
    kl_penalty="kl",            # KL penalty type
    seed=0,
    use_score_norm=True,        # Normalise rewards
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=actor_model,
    ref_model=ref_model,        # Frozen SFT model (for KL computation)
    tokenizer=tokenizer,
    reward_model=reward_model,
)

for batch in ppo_trainer.dataloader:
    # Generate responses
    query_tensors = batch["input_ids"]
    response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
    
    # Compute rewards
    texts = [tokenizer.decode(r) for r in response_tensors]
    rewards = [reward_model.score(t) for t in texts]
    
    # PPO update
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

---

### DPO: Direct Preference Optimisation (2023)

> 💡 **ELI5 (Explain Like I'm 5):**
> RLHF/PPO is like hiring a full-time food critic (reward model) to rate every dish the AI chef cooks, then paying the chef based on those ratings. **DPO** is simpler: you just show the chef two versions of a dish and say "people preferred this one over that one." The chef learns directly from comparisons, no full-time critic needed, saving you enormous effort.

**Key insight:** You don't need a separate reward model. The optimal policy IS the reward model, up to a reparameterisation.

**DPO Loss:**
```
L_DPO(π_θ) = -E[(x, y_w, y_l)~D] [ log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x))) ]

where:
  y_w = chosen (winner) response
  y_l = rejected (loser) response
  π_ref = reference SFT model (frozen)
  β = temperature (typically 0.1)
```

**Implementation:**
```python
from trl import DPOConfig, DPOTrainer

dpo_config = DPOConfig(
    model_name_or_path="sft_model_path",
    beta=0.1,                       # Temperature (higher = closer to ref model)
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    output_dir="./dpo_model",
    bf16=True,
    report_to="wandb",
    generate_during_eval=True,      # Sample responses and evaluate during training
)

# Dataset format: {"prompt": str, "chosen": str, "rejected": str}
dpo_trainer = DPOTrainer(
    model=model,                    # Model to train
    ref_model=ref_model,            # Frozen reference model
    args=dpo_config,
    train_dataset=train_dataset,    # With prompt, chosen, rejected columns
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

dpo_trainer.train()
```

**DPO vs RLHF comparison:**

| Aspect | RLHF (PPO) | DPO |
|---|---|---|
| Reward model | Required (train separately) | Not needed |
| Training stability | Unstable (many hyperparameters) | Stable (supervised loss) |
| Compute | 3× SFT compute | 1.5× SFT compute |
| Quality | Frontier models (GPT-4) | Strong (Zephyr, OpenHermes) |
| Implementation | Complex (PPO, KL, sampling) | Simple |
| Used by | OpenAI, Anthropic | Meta, MISTRAL, community |

---

### Constitutional AI (Anthropic's Approach)

```
RLHF bottleneck: Human labellers are slow and expensive
Constitutional AI (CAI) automates the preference data generation:

1. Generate initial response
2. AI (Claude) critiques it against Constitutional principles
3. AI revises the response
4. Repeat 3-4 times (chain of thought revision)
5. Use (original draft, revised response) as preference pair
6. Train reward model on these AI-generated preferences
7. Proceed with PPO as normal

Advantage: 
  - Scalable (AI generates its own training signal)
  - More consistent than human labellers
  - Principles can be explicitly specified (not hidden in human preferences)
```

---

> 🃏 **Quick-Recall Card — RLHF, PPO & DPO**
> | Concept | One-liner |
> |---|---|
> | RLHF goal | Align a pre-trained LLM to human preferences beyond simple instructions |
> | SFT (Step 1) | Fine-tune on curated (prompt, ideal response) pairs — sets baseline behaviour |
> | Reward Model (Step 2) | Trained on human comparisons (A > B) to score response quality. Outputs a scalar. |
> | PPO (Step 3) | RL algorithm that nudges the policy to maximise reward, penalised by KL from SFT model |
> | KL penalty | Prevents model from drifting too far from SFT — stops reward hacking |
> | PPO complexity | Requires 4 models in memory: SFT ref, actor, reward model, critic. Very expensive. |
> | DPO | Skips reward model entirely. Directly optimises on (chosen, rejected) pairs. ~3× cheaper. |
> | DPO β | Temperature: higher = stay closer to SFT reference. Typically 0.1. |
> | Constitutional AI | Anthropic's method: AI self-critiques responses using principles → generates preference pairs automatically |
>
> **Key trade-off:** PPO = more powerful/flexible; DPO = simpler, cheaper, often similar quality for instruction following.

## Week 4: Multimodal Models

> 📖 **Big picture:** Text-only LLMs are becoming the baseline, not the frontier. Modern AI systems at FAANG process images, audio, video, and code alongside text. Understanding multimodal models is essential for roles working on the next generation of AI products: image generation (DALL-E, Stable Diffusion), visual question answering (GPT-4V), speech (Whisper), and visual search.
>
> **The key conceptual bridge:** Multimodal models work by projecting different modalities (images, audio) into the same embedding space as text. Once in the same space, the transformer’s attention mechanism can relate visual features to text tokens naturally. CLIP and similar models pioneered this — everything else builds on it.

### CLIP: Contrastive Language-Image Pretraining (OpenAI, 2021)

> 💡 **ELI5 (Explain Like I'm 5):**
> Imagine putting all the world's text in one room, and all the images in another. CLIP is a translator who stands between the rooms, learning that when someone shouts "Golden Retriever!" in the text room, a specific picture lights up in the image room. It builds a universal dictionary connecting words and pictures.

**Architecture:**
```
Image Encoder (ViT or ResNet) → image_embedding (512-dim)
Text Encoder (Transformer) → text_embedding (512-dim)

Both embedded in SHARED space
Cosine similarity = relevance of (image, text) pair
```

**Training objective (contrastive loss):**
```
Given N (image, text) pairs in a batch:
- N matching pairs (positive)
- N² - N non-matching pairs (negative)

InfoNCE loss: maximise similarity of matching pairs, minimise non-matching
This is the same idea as SimCLR contrastive learning

Result: without any labels, CLIP learns rich vision-language alignment
Training data: 400M (image, alt-text) pairs from the web (WIT dataset)
```

```python
from transformers import CLIPModel, CLIPProcessor
import torch
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Zero-shot image classification
image = Image.open("cat.jpg")
labels = ["a cat", "a dog", "a bird", "a car"]

inputs = processor(
    text=labels,
    images=image,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # (1, num_labels)
    probs = logits_per_image.softmax(dim=1)
    
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob.item():.3f}")
# a cat: 0.987, a dog: 0.008, a bird: 0.003, a car: 0.002
```

---

### Vision-Language Models (VLMs) — LLaVA Architecture

```
LLaVA (Large Language and Vision Assistant) architecture:
  
  Image → CLIP Visual Encoder → Image features (N×D)
        → MLP Projection → Visual tokens
                                          ↓
  Text prompt → Tokenizer → Text tokens [Visual tokens + Text tokens]
                                                      ↓
                                            LLM (LLaMA / Mistral)
                                                      ↓
                                               Text response
                                               
Key insight: treat image regions as "visual tokens" that can be interleaved with text tokens
Training:
  1. Pre-train only the MLP projection (align image and text spaces)
  2. Fine-tune MLP + LLM on visual instruction data (image, question, answer) triples

Models in this family:
  - LLaVA 1.6 (open source)
  - GPT-4V / GPT-4o (OpenAI)
  - Gemini Vision (Google)
  - Claude 3 Vision (Anthropic)
  - Qwen-VL (Alibaba)
```

**Using GPT-4o for vision:**
```python
import base64
from openai import OpenAI

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image('chart.png')}",
                        "detail": "high"   # "low" (85 tokens) or "high" (up to 1445 tokens)
                    }
                },
                {
                    "type": "text",
                    "text": "What trend does this chart show? Identify any anomalies."
                }
            ]
        }
    ],
    max_tokens=500
)
```

---

### Stable Diffusion (Text-to-Image) — Architecture Overview

```
For AI engineers, you need to know the concepts:

Latent Diffusion Model (LDM):
  1. VAE Encoder: image (512×512×3) → latent (64×64×4) [smaller space!]
  2. Forward Diffusion: add Gaussian noise gradually over T steps → noisy latent
  3. U-Net (denoiser): learns to predict noise given (noisy_latent, timestep, text_embedding)
  4. CLIP Text Encoder: text prompt → text embedding (conditioning signal for U-Net)
  5. VAE Decoder: denoised latent → generated image

Inference (text-to-image generation):
  1. Sample random Gaussian noise z_T
  2. For t = T, T-1, ..., 1:
       noise_pred = U-Net(z_t, t, text_embedding)
       z_{t-1} = DDPM/DDIM step (z_t, noise_pred, t)
  3. VAE decode z_0 → final image

SDXL (Stable Diffusion XL):
  - Two text encoders: CLIP ViT-L/14 + OpenCLIP ViT-bigG
  - Base model (1024×1024) + Refiner model for high-frequency details
  - More parameters: 2.6B vs 0.9B for SD 1.5
```

---

## Interview Q&A — RLHF & Multimodal

**Q1: Walk me through RLHF step by step.**
> 1. SFT: fine-tune base LLM on (instruction, good response) pairs. 2. Preference data: collect (prompt, chosen response, rejected response) tuples from human comparisons. 3. Reward model: train a model to score responses (contrastive loss: chosen score > rejected score). 4. PPO training: use the policy (SFT model) to generate responses, score with reward model, update policy with PPO while adding KL penalty to prevent divergence from SFT model. 5. Monitor and iterate.

**Q2: What's the difference between DPO and PPO-based RLHF? When would you use each?**
> DPO: simpler, stable, no explicit RM, supervised loss on preference pairs. Works well for fine-tuning OSS models at moderate scale. PPO: more complex, unstable, but has explicit separation between RM and policy. Allows online data collection (sample new responses, get scored, retrain). Better for very large models at frontier labs. Use DPO: limited compute, need stability, fine-tuning community models. Use PPO: frontier model, need online RL to handle distribution shift.

**Q3: What is the KL penalty in RLHF and why is it important?**
> Without a KL penalty, PPO would exploit the reward model — generate text that maximises reward score but is degenerate (e.g., gibberish that the reward model happens to assign high scores to). The KL penalty limits how far the policy can drift from the SFT reference model. The objective is: maximise R(response) - β × KL(policy || SFT). β controls the strength of the regularisation (typically 0.1-0.5).

**Q4: How does CLIP enable zero-shot image classification?**
> CLIP trains an image encoder and text encoder jointly with contrastive loss on 400M (image, text) pairs, aligning their embedding spaces. At inference, embed query image + all class names as text. The class with highest cosine similarity to the image embedding is the prediction — no task-specific training needed ("zero-shot").

**Q5: What is the key architectural difference between CLIP and LLaVA?**
> CLIP: dual-encoder (image encoder + text encoder), no generation, just comparison/retrieval. Used for classification, image-text matching, semantic search. LLaVA: extends a generative LLM with a visual encoder. A projection MLP maps CLIP visual features into visual tokens that are prepended to the text token sequence, then fed to a causal language model. Enables dialogue and generation conditioned on images.

**Q6: How would you design a production vision-language pipeline for processing receipts (OCR + extraction)?**
> Option 1: GPT-4o with vision — easy, high quality, handles complex layouts. Cost: ~$0.01/image. Option 2: LLaVA local — lower cost, privacy-preserving for sensitive financial data, but needs GPU. Pipeline: Image → resize & encode to base64 → VLM (extract structured JSON: merchant, date, items, total) → validate with Pydantic → store. Evaluation: 200 ground-truth receipts with hand-annotated fields, measure field-level accuracy.

---

## 📚 Further Resources

**RLHF / Alignment:**
- **"RLHF: Reinforcement Learning from Human Feedback"** — Hugging Face blog: https://huggingface.co/blog/rlhf
- **"DPO: Direct Preference Optimization"** (Rafailov et al., 2023) — https://arxiv.org/abs/2305.18290
- **"Constitutional AI: Harmlessness from AI Feedback"** (Bai et al., 2022) — https://arxiv.org/abs/2212.08073
- **TRL library**: https://huggingface.co/docs/trl (PPO, DPO, GRPO trainers)

**Multimodal:**
- **"Learning Transferable Visual Models From Natural Language Supervision"** (Radford et al., 2021) — CLIP paper: https://arxiv.org/abs/2103.00020
- **"Visual Instruction Tuning"** (Liu et al., 2023) — LLaVA paper: https://arxiv.org/abs/2304.08485
- **GPT-4V System Card** (OpenAI, 2023)

**Books:**
- **"Hands-On Large Language Models" by Jay Alammar & Maarten Grootendorst** — Excellent diagrams of CLIP, transformers, fine-tuning

**Courses:**
- **DeepLearning.AI: Reinforcement Learning from Human Feedback** — https://learn.deeplearning.ai/courses/reinforcement-learning-from-human-feedback
- **Google: Introduction to Stable Diffusion** — https://www.cloudskillsboost.google

> **End of Phase 3 Core Content.** The sections below add day-to-day work depth.

---

## Day-to-Day Work: System Design, RLHF & Multimodal in Practice

### System Design Knowledge at Work

```
You'll use system design knowledge DAILY as an AI engineer:

KAFKA IN YOUR WORKFLOW:
  - Log every LLM request/response to Kafka topic "llm_events"
  - Consumer 1: Real-time monitoring dashboard
  - Consumer 2: Evaluation pipeline (sample 1% → RAGAS scores)
  - Consumer 3: Training data pipeline (positive examples → future fine-tuning)
  - Consumer 4: Cost tracking (aggregate token usage → billing)

REDIS IN YOUR WORKFLOW:
  - Cache LLM responses (40-60% hit rate for support bots)
  - Rate limiting per user/team (prevent cost spikes)
  - Session memory for conversational agents
  - Feature store for real-time ML features

DATABASE DECISIONS YOU'LL MAKE:
  - Postgres: user data, config, evaluation results
  - Pinecone/pgvector: embeddings for RAG
  - Redis: cache + rate limits
  - BigQuery: analytics, training data, batch processing
  - Elasticsearch: full-text search + log aggregation
```

### RLHF/DPO at Work — Practical Applications

```python
# You probably won't train RLHF from scratch at work.
# But you WILL use preference data to improve model outputs.

# Practical scenario: Improving your RAG chatbot with user feedback

# Step 1: Collect preference data from production
# User gives thumbs up/down → create preference pairs
def collect_preferences(query, response_a, response_b, user_choice):
    """Log user preference for DPO training."""
    return {
        "prompt": query,
        "chosen": response_a if user_choice == "a" else response_b,
        "rejected": response_b if user_choice == "a" else response_a,
    }

# Step 2: DPO fine-tuning (simpler than full RLHF)
from trl import DPOConfig, DPOTrainer

training_args = DPOConfig(
    output_dir="./dpo-model",
    per_device_train_batch_size=4,
    learning_rate=5e-7,  # very low LR for alignment
    num_train_epochs=1,
    beta=0.1,  # KL penalty — prevents model diverging too far from base
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # uses implicit reference (base model copy)
    args=training_args,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)
trainer.train()

# Step 3: A/B test DPO model vs base model
# Route 50% of traffic to each → compare user satisfaction scores
```

### Multimodal at Work — Vision + Language

```python
# Multimodal LLMs are becoming standard at work:

# Use Case 1: Product image analysis (retail)
from openai import OpenAI
import base64

client = OpenAI()

def analyze_product_image(image_path: str) -> dict:
    """Extract product info from shelf photo."""
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": """Analyze this retail shelf photo:
                1. List all visible products with brand names
                2. Estimate shelf share percentage per brand
                3. Identify any out-of-stock positions
                4. Note any promotional displays
                Return as JSON."""},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }}
            ]
        }],
        response_format={"type": "json_object"},
        max_tokens=1000
    )
    return json.loads(response.choices[0].message.content)

# Use Case 2: Chart/graph understanding for analytics
def analyze_chart(image_path: str, question: str) -> str:
    """Ask questions about charts and graphs."""
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Look at this chart and answer: {question}"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }}
            ]
        }]
    )
    return response.choices[0].message.content

# Use Case 3: Document understanding (OCR + analysis)
# Process invoices, receipts, contracts — extract structured data from images

# Use Case 4: Multimodal RAG
# Index product images alongside text descriptions
# Query: "Show me products similar to this photo" → image embedding + text search
```

### Building a Multimodal RAG Pipeline

```python
# Emerging pattern: combine text + image embeddings for richer retrieval

from transformers import CLIPProcessor, CLIPModel
import torch

class MultimodalRetriever:
    def __init__(self):
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def embed_text(self, text: str):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            return self.clip.get_text_features(**inputs).numpy()[0]
    
    def embed_image(self, image):
        inputs = self.processor(images=[image], return_tensors="pt")
        with torch.no_grad():
            return self.clip.get_image_features(**inputs).numpy()[0]
    
    def search(self, query_text: str = None, query_image=None, k=5):
        """Search by text, image, or both."""
        if query_text and query_image:
            text_emb = self.embed_text(query_text)
            image_emb = self.embed_image(query_image)
            query_emb = 0.5 * text_emb + 0.5 * image_emb  # weighted combination
        elif query_text:
            query_emb = self.embed_text(query_text)
        else:
            query_emb = self.embed_image(query_image)
        
        # Search in vector DB (both text and image embeddings stored together)
        return self.vector_store.similarity_search_by_vector(query_emb, k=k)
```

> **End of Phase 3.** You've covered fine-tuning, inference optimisation, system design, RLHF and multimodal. Phase 4 is pure interview prep — DSA sprint, behavioural stories, applications.
