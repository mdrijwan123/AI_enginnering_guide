# Knowledge Distillation — Complete Guide
### Phase 3 Supplementary | October 2026 Reference

Fine-tuning and quantisation are not the only ways to make a large model smaller and cheaper to run. Knowledge distillation takes a fundamentally different approach: instead of compressing an existing model, it *trains a new, smaller model to mimic the behaviour of the large one*. The result is a compact model that punches well above its weight class — because it learned from a teacher that already solved the hard problem of generalisation.

Understanding distillation is essential in 2026 because the entire Small Language Model revolution (Phi-3, Gemma 2, DeepSeek-R1-Distill, LLaMA 3 8B) is powered by it. FAANG interviewers routinely ask: "How would you deploy this system cheaply at scale?" Distillation is one of the most powerful answers available.

---

## Table of Contents
1. [The Core Idea — Soft Targets vs Hard Targets](#1)
2. [The Mathematics of Distillation](#2)
3. [DistilBERT — A Clinical Case Study](#3)
4. [Distilling Large Language Models](#4)
5. [Task-Specific vs General Distillation](#5)
6. [Distillation vs Quantisation vs Pruning](#6)
7. [Implementation — DistilTrainer in Practice](#7)
8. [Interview Q&A](#8)

---

## Part 1 — The Core Idea: Soft Targets vs Hard Targets

Imagine you are training to be a doctor. The fastest path is to memorise the answer key: question 23 = "C: appendicitis." This is training with **hard targets** — a one-hot distribution where the correct answer is 1 and everything else is 0.

Now imagine instead of the answer key, you received a senior consultant's differential diagnosis: "Appendicitis: 82%, mesenteric adenitis: 12%, ovarian cyst: 4%, other: 2%." This is far richer information. You learn not just *what* the answer is, but *how certain* the expert was, *which conditions were plausible alternatives*, and *how the problem space is structured*. This is training with **soft targets** — the teacher model's full output probability distribution.

Hinton, Vinyals & Dean (2015) formalised this intuition. A large pretrained model (the **teacher**) generates a probability distribution over all output classes for every training example. A smaller model (the **student**) is then trained to reproduce that distribution, not just the majority vote. The probability "mass" given to non-winning classes encodes the teacher's understanding of similarity relationships between concepts — what the authors called the **dark knowledge** of the model.

```
Teacher model (large):                Student model (small):
  "cat" = 0.85                          learns from teacher's
  "kitten" = 0.08                       full distribution,
  "dog" = 0.05                          not just the label "cat"
  "lion" = 0.02                         
```

The student learns that "cat" and "kitten" are similar, that "cat" and "dog" are more similar than "cat" and "car" — all from the teacher's soft probabilities, without needing explicit similarity supervision.

---

## Part 2 — The Mathematics of Distillation

### Temperature Scaling

A standard softmax produces very "peaked" distributions for a well-trained model — the correct class might get 0.998 probability and everything else is near zero. To make the soft targets more informative, Hinton introduced a **temperature parameter** $T$:

$$\text{softmax}_T(z_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

At $T = 1$ (standard softmax), the distribution is peaked. As $T$ increases, the distribution becomes softer — more "dark knowledge" leaks into the non-winning classes. At $T = 4$ (common in practice), the tails are rich enough for the student to learn meaningful relationships.

```python
import torch
import torch.nn.functional as F

def soft_targets(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Convert logits to soft probability distribution at temperature T."""
    return F.softmax(logits / temperature, dim=-1)

# High T → softer, more informative distribution
# Low T  → harder, more peaked distribution
# T = 1  → standard inference
```

### The Distillation Loss

The student is trained with a combination of two losses:

$$\mathcal{L}_{\text{distill}} = \alpha \cdot T^2 \cdot \text{KL}(p_{\text{teacher}}^T \,\|\, p_{\text{student}}^T) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(y, p_{\text{student}}^1)$$

Where:
- $p_{\text{teacher}}^T$ = teacher's soft probabilities at temperature $T$
- $p_{\text{student}}^T$ = student's soft probabilities at temperature $T$  
- $\mathcal{L}_{\text{CE}}$ = standard cross-entropy against hard ground-truth labels $y$
- $\alpha$ = weight balancing the two losses (typically 0.7 for distillation term)
- $T^2$ = scaling factor to normalise the gradient magnitudes (KL gradients are $1/T^2$ smaller)

Intuitively: the first term teaches the student to **think like the teacher**; the second term ensures the student still **gets the right answer** on labelled data.

```python
import torch
import torch.nn.functional as F

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> torch.Tensor:
    """
    Combined distillation + cross-entropy loss.
    
    Args:
        student_logits: raw logits from student model [batch, classes]
        teacher_logits: raw logits from teacher model [batch, classes]
        labels: hard ground-truth labels [batch]
        temperature: softmax temperature for soft targets
        alpha: weight for distillation loss (1-alpha for CE loss)
    """
    # Soft targets at temperature T
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    
    # KL divergence: how different is student from teacher?
    distill_loss = F.kl_div(
        input=soft_student,
        target=soft_teacher,
        reduction="batchmean"
    ) * (temperature ** 2)  # scale by T² to normalise gradients
    
    # Standard cross-entropy on hard labels (student at T=1)
    ce_loss = F.cross_entropy(student_logits, labels)
    
    return alpha * distill_loss + (1 - alpha) * ce_loss
```

### Feature Distillation (Beyond Output Layer)

Advanced distillation also matches **intermediate representations**, not just the final output. The student is trained to produce hidden states similar to the teacher's at intermediate layers:

$$\mathcal{L}_{\text{feature}} = \sum_l \text{MSE}(W_l \cdot h_l^{\text{student}}, h_l^{\text{teacher}})$$

Where $W_l$ is a learned projection matrix (because teacher and student may have different hidden dimensions). This is how DistilBERT works — the student's hidden states are trained to mimic the teacher's.

---

## Part 3 — DistilBERT: A Clinical Case Study

DistilBERT (Sanh et al., 2019 — Hugging Face) is the canonical example of distillation applied to a large pretrained language model. Understanding it gives you a concrete anchor for interview discussions.

### The Setup

| | BERT-Base | DistilBERT |
|---|---|---|
| Parameters | 110M | 66M (40% smaller) |
| Layers | 12 | 6 (every other layer) |
| Hidden dim | 768 | 768 (same!) |
| Attention heads | 12 | 12 (same!) |
| Training speed | baseline | 60% faster |
| GLUE score | 82.1 | 79.5 (97% of teacher) |
| Inference speed | baseline | ~2× faster |

The key architectural choice: DistilBERT keeps the same hidden dimension and attention heads as BERT — it simply removes every other transformer layer. This is deliberate. It means the student's internal representations are comparable in dimension to the teacher's, making feature matching straightforward.

### Three Training Signals

DistilBERT was trained with three simultaneous losses:

1. **MLM Loss** — standard masked language modelling on the token logits (hard label CE)
2. **Distillation Loss** — KL divergence between student and teacher MLM probability distributions (at $T=8$ — quite high, to extract maximum dark knowledge)
3. **Cosine Embedding Loss** — cosine similarity between student and teacher hidden states, maximised to encourage the student's internal representations to align with the teacher's

```python
# Simplified DistilBERT-style training loop
for batch in dataloader:
    student_output = student_model(**batch)
    with torch.no_grad():
        teacher_output = teacher_model(**batch)
    
    # 1. Standard MLM loss
    mlm_loss = student_output.loss
    
    # 2. Distillation loss on masked token logits
    dist_loss = distillation_loss(
        student_logits=student_output.logits,
        teacher_logits=teacher_output.logits,
        labels=batch["labels"],
        temperature=8.0,
        alpha=0.9,  # heavy weight on teacher signal
    )
    
    # 3. Cosine loss on hidden states
    cos_loss = 1 - F.cosine_similarity(
        student_output.hidden_states[-1],
        teacher_output.hidden_states[-1],
        dim=-1
    ).mean()
    
    total_loss = 1.0 * dist_loss + 1.0 * mlm_loss + 4.0 * cos_loss
    total_loss.backward()
    optimiser.step()
```

---

## Part 4 — Distilling Large Language Models

Distilling a model with 70B parameters into one with 7B is the same principle, evolved for the autoregressive generation setting.

### Sequence-Level Distillation

For generative models, you cannot easily compute token-level soft targets over a vocabulary of 128,000+ tokens per position in a long sequence (memory would be enormous). Instead, you use **sequence-level KD**: the teacher *generates* complete sequences, and the student is trained on those generated sequences as if they were hard labels.

```python
# Offline distillation: pre-generate teacher outputs
def generate_teacher_dataset(
    teacher_model,
    prompts: list[str],
    output_path: str,
    temperature: float = 0.8,
    n_samples: int = 3,
):
    """
    Generate multiple high-quality teacher responses per prompt.
    The student trains on the best ones (filtered by reward model score).
    """
    import json
    data = []
    
    for prompt in prompts:
        responses = []
        for _ in range(n_samples):
            response = teacher_model.generate(
                prompt,
                max_new_tokens=512,
                temperature=temperature,
            )
            responses.append(response)
        
        data.append({"prompt": prompt, "responses": responses})
    
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
```

### Real-World Examples: How Small Models Are Built

The "small but smart" model trend is almost entirely distillation-powered:

| Model | Teacher | Method | Size Reduction |
|---|---|---|---|
| DistilBERT | BERT-Base | Feature + output KD | 40% smaller |
| DistilGPT-2 | GPT-2 | Output KD | 50% smaller |
| DeepSeek-R1-Distill-7B | DeepSeek-R1 (671B) | Sequence-level KD on CoT traces | 96% smaller |
| Phi-3-mini (3.8B) | GPT-4 + web data | Sequence-level KD + RLHF | ~98% smaller |
| Gemma 2 (9B) | Gemini family | Output KD + distillation data | — |
| LLaMA 3 8B | LLaMA 3 70B + frontier models | Preference distillation | 88% smaller |

**DeepSeek-R1-Distill** is particularly instructive: the teacher model generates thousands of chain-of-thought reasoning traces (the scratchpad). The student is trained on those traces via standard SFT. The student never needs to learn to reason from scratch — it learns by imitating the teacher's reasoning patterns. This is why the 7B distilled model scores close to the 671B teacher on reasoning benchmarks.

### On-Policy vs Off-Policy Distillation

- **Off-policy (offline):** Teacher generates responses to a fixed dataset; student trains on them. Simple, scalable, but the student never learns to recover from its own mistakes.
- **On-policy:** Student generates responses; teacher scores or corrects them. More expensive but produces better-calibrated models (used in iterative distillation setups like Orca 2).

---

## Part 5 — Task-Specific vs General Distillation

### General Distillation

Distil the model before any fine-tuning — compress the general-purpose model. DistilBERT and DistilGPT-2 are both general distillations: the resulting models can then be fine-tuned on any downstream task.

**Use when:** You need a smaller backbone for many different downstream tasks. Amortises the distillation cost across all use cases.

### Task-Specific Distillation

Distil the already-fine-tuned teacher on a specific task's data only. The student model may be smaller *and* specialised.

```python
# Task-specific: teacher is already fine-tuned on your task
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Teacher: fine-tuned large model
teacher = AutoModelForSequenceClassification.from_pretrained("bert-large-finetuned-sentiment")
teacher.eval()

# Student: small model (initialise randomly or from small pretrained)
student = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Custom trainer with distillation loss
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=4.0, alpha=0.7, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher_model
        self.temp = temperature
        self.alpha = alpha
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
        
        loss = distillation_loss(
            student_logits=outputs.logits,
            teacher_logits=teacher_outputs.logits,
            labels=inputs["labels"],
            temperature=self.temp,
            alpha=self.alpha,
        )
        
        return (loss, outputs) if return_outputs else loss

# Train student
trainer = DistillationTrainer(
    teacher_model=teacher,
    model=student,
    args=TrainingArguments(
        output_dir="./distilled-sentiment",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        learning_rate=5e-5,
        fp16=True,
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

**Use when:** You only care about one task and want maximum compression for that specific capability. Often achieves better student performance per parameter than general distillation.

---

## Part 6 — Distillation vs Quantisation vs Pruning

These three techniques are often confused. They attack the compression problem from completely different angles. In practice, the best systems combine multiple techniques.

| Technique | What It Does | Key Mechanism | Requires Retraining? | Best For |
|---|---|---|---|---|
| **Distillation** | Trains a new smaller model | Student learns from teacher's soft outputs | Yes (new model training) | New model creation; architecture flexibility |
| **Quantisation** | Reduces numeric precision of existing weights | FP16/INT8/INT4 representation | Usually not (post-training QA) | Cheaply shrinking an existing model |
| **Pruning** | Removes near-zero weights | Magnitude-based or gradient-based zeroing | Yes (fine-tuning after pruning) | Structured sparsity; hardware-friendly |
| **LoRA** | Trains only a low-rank delta | Frozen base + trainable decomposition | Yes (adapter training only) | Cheap task-specific adaptation |

### When to Use Which

```
You have a large model and want to deploy it cheaply on existing hardware → Quantisation (quickest path)
  
You want a permanently smaller architecture for long-term production → Distillation (bigger upfront cost, bigger long-term win)

You need to adapt a model to a specific task with limited resources → LoRA/QLoRA (best cost/quality for adaptation)

You want inference on mobile / edge with minimal latency → Pruning + Quantisation

You want to create a new small model that reasons well on complex tasks → Distillation (teach CoT reasoning traces)
```

### Stacking Techniques

These methods compose well. All four can be applied to the same model:

```
Base: LLaMA 3 70B (140GB FP16)
         │
         │ Step 1: Distillation
         ▼
     8B student model (16GB FP16)
         │
         │ Step 2: LoRA fine-tuning for your task
         ▼
     8B with 160MB LoRA adapter (16GB + 0.16GB)
         │
         │ Step 3: QLoRA (NF4 quantisation of base)
         ▼
     8B in 4-bit (4GB) + LoRA adapter
         │
         │ Step 4: Merge + post-training quantisation
         ▼
     8B merged model at INT4 (~4.5GB)
         └─→ Fits on a single RTX 3090 for serving
```

---

## Part 7 — Full Implementation: DistilTrainer in Practice

A complete, production-ready training script for distilling any classification or generation model. This uses the general-purpose 2-loss approach (KL divergence + cross-entropy). For DistilBERT's specialised 3-loss approach (which adds a cosine embedding loss on hidden states), see Part 3:

```python
#!/usr/bin/env python3
"""
Knowledge distillation training script.
Works for classification, token classification, and generation tasks.

Usage:
    python distil_train.py \
        --teacher_model microsoft/deberta-v3-large-finetuned-mnli \
        --student_model google/electra-base-discriminator \
        --task classification \
        --output_dir ./distilled_model \
        --temperature 4.0 \
        --alpha 0.7
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_dataset


def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """Combined KD + CE loss."""
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    
    kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
    ce = F.cross_entropy(student_logits, labels)
    
    return alpha * kl + (1 - alpha) * ce


class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=4.0, alpha=0.7, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher_model
        self.teacher.eval()  # teacher always in eval mode
        self.temperature = temperature
        self.alpha = alpha
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        student_outputs = model(**inputs)
        
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
        
        loss = distillation_loss(
            student_logits=student_outputs.logits,
            teacher_logits=teacher_outputs.logits,
            labels=labels,
            temperature=self.temperature,
            alpha=self.alpha,
        )
        return (loss, student_outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, required=True)
    parser.add_argument("--student_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./distilled")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    
    teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher_model)
    teacher = teacher.to(device)
    
    student = AutoModelForSequenceClassification.from_pretrained(
        args.student_model,
        num_labels=teacher.config.num_labels,  # match teacher's output space
    )
    student = student.to(device)
    
    # Parameter efficiency report
    teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    student_params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"Teacher: {teacher_params:.1f}M params")
    print(f"Student: {student_params:.1f}M params ({student_params/teacher_params*100:.1f}% of teacher)")
    
    # Dataset (replace with your own)
    dataset = load_dataset("glue", "sst2")
    
    def tokenize_fn(examples):
        return tokenizer(examples["sentence"], truncation=True, max_length=512)
    
    tokenized = dataset.map(tokenize_fn, batched=True)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        report_to="wandb",
    )
    
    trainer = DistillationTrainer(
        teacher_model=teacher,
        temperature=args.temperature,
        alpha=args.alpha,
        model=student,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\nDistilled model saved to {args.output_dir}")
    print(f"Compression: {teacher_params:.1f}M → {student_params:.1f}M parameters")


if __name__ == "__main__":
    main()
```

---

## Part 8 — Interview Q&A

### Q1: Explain knowledge distillation and why "soft targets" are more informative than hard labels.

In standard supervised learning, each example is paired with a one-hot label — the correct class gets probability 1, everything else gets 0. Soft targets are the full probability distribution output by a pre-trained teacher model, computed at a raised temperature $T$ to amplify the tail probabilities. The non-winning class probabilities encode the similarity structure the teacher has learned: output class "cat" gets a small probability for "kitten" and even smaller for "dog", revealing that cats and kittens are conceptually closer than cats and cars. The student learning from this distribution absorbs that relational knowledge without any additional supervised signal. Hinton called this the "dark knowledge" — it's the generalisation insight the teacher acquired during training, compressed into per-example probability distributions.

---

### Q2: Walk me through the DistilBERT architecture and training procedure.

DistilBERT is BERT-Base with every other transformer layer removed — 12 layers become 6. Crucially, the hidden dimension (768) and attention heads (12) are kept identical to the teacher. This is intentional: it makes intermediate-layer feature matching tractable. Training uses three simultaneous losses. The MLM loss is standard cross-entropy against masked token labels. The distillation loss is KL divergence between the student's and teacher's soft probability distributions at $T=8$, scaled by $T^2$ to normalise gradient magnitudes. The cosine embedding loss maximises the cosine similarity between the student's and teacher's hidden states at the last transformer layer, encouraging the student's internal representations to mirror the teacher's. The student is initialised from the even-numbered layers of BERT (layers 0, 2, 4, 6, 8, 10), giving it a warm start. The result: 40% smaller, 60% faster at training, 97% of BERT's GLUE performance.

---

### Q3: How is distillation being used to create small reasoning models like Phi-3 and DeepSeek-R1-Distill?

These models use sequence-level distillation rather than token-level soft targets. For Phi-3, Microsoft used GPT-4 to generate thousands of high-quality chain-of-thought reasoning examples on diverse problems, then trained a 3.8B model on those generated traces via standard SFT. The 3.8B student never needs to derive the reasoning ability from scratch — it learns to imitate GPT-4's reasoning patterns. For DeepSeek-R1-Distill, the 671B teacher generates extended CoT scratchpads on math and coding problems, and the 7B and 14B students are SFT-trained on those scratchpad completions. This is why the distilled models score disproportionately well on reasoning benchmarks — they are trained on the teacher's own reasoning process, not just its final answers.

---

### Q4: What is the distillation loss formula and what does each component do?

$$\mathcal{L} = \alpha \cdot T^2 \cdot \text{KL}(p_{\text{teacher}}^T \,\|\, p_{\text{student}}^T) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(y, p_{\text{student}})$$

The first term is the distillation signal: KL divergence between teacher and student soft-target distributions, both computed at temperature $T$. The $T^2$ factor compensates for the fact that KL gradients are naturally $1/T^2$ smaller when computed at elevated temperature. The second term is ordinary cross-entropy against the ground-truth hard label $y$, evaluated at $T=1$. The hyperparameter $\alpha$ controls the balance: high $\alpha$ (0.7–0.9) emphasises learning from the teacher; lower $\alpha$ emphasises correct answer accuracy. The temperature $T$ typically ranges from 2–20; higher values produce softer distributions with richer dark knowledge.

---

### Q5: When would you choose distillation over quantisation for deploying an LLM?

Choose distillation when you have the compute budget to train and need long-term architectural flexibility — you get a genuinely smaller model, not a compressed one. A distilled 7B model loads faster, uses less memory at inference, and can be further fine-tuned or quantised independently. The distilled model is also architecturally smaller, which matters for edge deployment where weight file size and layer count are constraints, not just precision.

Choose quantisation when you have an existing fine-tuned model and need to make it cheaper immediately — INT4 quantisation with bitsandbytes or GGUF takes hours and requires no training. You lose a small amount of quality (1–3% on most tasks at INT4) but the turnaround time is incomparable.

In practice, the best production pipelines combine both: distil first to get a smaller architecture, then quantise for efficient inference.

---

### Q6: An LLM application is costing $15,000/month to serve via API. How would you reduce this using distillation?

This is a cost-reduction design question. Step one is characterisation: capture 30 days of production prompts and responses. Analyse the task distribution — are 80% of requests actually the same 3 task types? Step two is teacher generation: run the expensive model on your production prompt dataset, generating 3-5 responses per prompt, and score them with a reward model or LLM-as-judge to keep only the top-quality examples. Step three is student training: fine-tune a small open-weights model (LLaMA 3 8B or Mistral 7B) on the teacher-generated examples with the distillation loss — or purely with SFT if you cannot easily get token-level logits from the teacher API. Step four is evaluation: benchmark the student on your domain-specific test set; common outcome is 85–95% of teacher quality at 4–8% of the cost. Step five is shadow deployment: route 5% of live traffic to the student, compare quality metrics, and gradually increase. Outcome: from $15K/month (GPT-4o API) to approximately $1–2K/month (self-hosted 8B model on 2× A10G instances).

---

## 📚 Further Resources

- [Distilling the Knowledge in a Neural Network (Hinton 2015)](https://arxiv.org/abs/1503.02531) — the original paper
- [DistilBERT (Sanh et al. 2019)](https://arxiv.org/abs/1910.01108) — BERT distillation
- [DeepSeek-R1 paper (2025)](https://arxiv.org/abs/2501.12948) — distillation of reasoning traces at scale
- [Orca 2 (Microsoft 2023)](https://arxiv.org/abs/2311.11045) — on-policy distillation for reasoning
- [Knowledge Distillation Survey (Gou et al. 2021)](https://arxiv.org/abs/2006.05525) — comprehensive taxonomy
- [HuggingFace DistilBERT model card](https://huggingface.co/distilbert-base-uncased) — training details and benchmarks
