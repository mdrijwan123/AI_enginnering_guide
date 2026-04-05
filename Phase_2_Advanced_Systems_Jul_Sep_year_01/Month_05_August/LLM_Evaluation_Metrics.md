# LLM Evaluation & Metrics — Complete Guide
### Phase 2 Supplementary | Critical for Production & Interviews

> This is one of the top 5 topics tested in LLM/AI engineer interviews in 2026. Every production LLM system needs evaluation — and most candidates only know RAGAS. This guide covers the full spectrum.

---

## Part 1 — Why LLM Evaluation Is Hard

"Does the model give a good answer?" is a deceptively simple question with no clean answer. In traditional ML, evaluation is straightforward: you have a fixed label space, you compute accuracy or F1, and you're done. LLMs break every assumption that makes this easy.

With classical models, outputs are deterministic and come from a finite label space — you can always compare a prediction to a ground truth label. With LLMs, outputs are stochastic (run the same query twice and you get different text), correctness is often subjective or context-dependent, and sometimes there is simply no single "right" answer. An LLM that politely declines to help with a harmful request is producing a *correct* output — but a naive accuracy metric would score it as a failure.

This means LLM quality is fundamentally **multi-dimensional**. You need to assess at least four independent axes simultaneously:

| Dimension | Question it answers |
|---|---|
| **Correctness** | Is the information factually accurate? Did the model hallucinate? |
| **Relevance** | Does the response actually address what was asked? |
| **Coherence** | Is it logically structured, internally consistent, and readable? |
| **Safety/Ethics** | Is it free from harmful, biased, or inappropriate content? |

No single number captures all four. This is why production LLM systems need a *suite* of complementary metrics — not a single score — and why this guide covers everything from classical overlap metrics to LLM-as-judge patterns.

---

## Part 2 — Automatic Text Generation Metrics

### 2.1 Perplexity

Perplexity is the foundational metric of language modelling. The intuition is this: a well-trained language model should assign high probability to fluent, natural text in its training domain. If you show it well-formed English prose and the model is consistently "surprised" by each next word, something is wrong. Perplexity quantifies that surprise — lower perplexity means the model found the text predictable and well-formed. A well-trained GPT-2 achieves roughly 20–50 perplexity on English Wikipedia text; random character sequences produce values in the thousands.

**What it measures:** How "surprised" the model is by a text, expressed as the exponentiated average cross-entropy loss per token. Lower = the model assigns higher probability to the text (less surprised).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

def calculate_perplexity(text: str, model_name: str = "gpt2") -> float:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss  # cross-entropy loss
    
    perplexity = math.exp(loss.item())
    return perplexity

# Usage:
# Good English text → perplexity ~50-100 (for GPT-2)
# Random tokens → perplexity >> 1000
# Model's own outputs → perplexity ≈ 10-30

# Formula:
# PPL = exp(-1/N × Σ log P(token_i | context))
# Where N = number of tokens
```

**When to use:** Pre-training evaluation, comparing model variants trained on the same data distribution, or detecting distribution shift in production (rising perplexity signals that incoming queries differ from what the model was trained on).  
**Limitations:** Perplexity tells you nothing about whether the model is actually *helpful* or *factually correct*. A model trained to reproduce convincing-sounding nonsense can have very low perplexity. Never use it as a standalone quality signal for production systems.

---

### 2.2 BLEU Score (Bilingual Evaluation Understudy)

BLEU was introduced in 2002 and immediately became the standard for automatic machine translation evaluation — because for the first time it correlated reasonably well with human judgements without requiring expensive human review. The intuition is simple: a good translation should share word sequences (n-grams) with professional human reference translations. BLEU computes precision across 1-, 2-, 3-, and 4-gram overlaps, then combines them geometrically with a penalty for outputs that are too short.

**Designed for:** Machine translation. Measures n-gram precision overlap between generated text and one or more reference translations.

```python
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

# BLEU-4: geometric mean of 1-, 2-, 3-, 4-gram precisions × brevity penalty
references = [["the", "cat", "sat", "on", "the", "mat"]]
hypothesis = ["the", "cat", "is", "on", "the", "mat"]

# Individual sentence BLEU
score = sentence_bleu(
    references, 
    hypothesis,
    smoothing_function=SmoothingFunction().method1
)
print(f"BLEU: {score:.3f}")

# Corpus-level (use for proper evaluation)
all_references = [[ref1], [ref2], ...]
all_hypotheses = [hyp1, hyp2, ...]
corpus_score = corpus_bleu(all_references, all_hypotheses)
```

**BLEU formula:**
$$\text{BLEU} = \text{BP} \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Where $p_n$ = n-gram precision, BP = brevity penalty.

**Limitations:** BLEU is blind to semantics and paraphrasing. The sentence "The cat sat on the mat" vs "A feline rested on the rug" scores 0 BLEU despite identical meaning. This makes it largely inappropriate for evaluating modern LLMs, which routinely express the same idea in many valid ways. In 2026, BLEU is still seen occasionally on translation benchmarks, but for open-ended generation, Q&A, or summarisation, BERTScore or LLM-as-judge gives far more reliable signal.

---

### 2.3 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Where BLEU focuses on precision (how much of the generated text appears in the reference), ROUGE flips the lens to ask: how much of the reference's important content appears in the generated output? For summarisation, this framing is often more appropriate — a good summary should *cover* the key information, so recall matters more than precision. ROUGE-1 counts unigram overlap, ROUGE-2 counts bigram overlap, and ROUGE-L finds the longest common subsequence, capturing sentence-level structure.

**Designed for:** Summarisation tasks where you want to measure information coverage. Also the standard metric for the CNN/DailyMail and XSum benchmarks.

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

reference = "The cat sat on the mat near the window."
hypothesis = "A cat was sitting on the mat."

scores = scorer.score(reference, hypothesis)
print(f"ROUGE-1: P={scores['rouge1'].precision:.3f} R={scores['rouge1'].recall:.3f} F={scores['rouge1'].fmeasure:.3f}")
print(f"ROUGE-2: F={scores['rouge2'].fmeasure:.3f}")  # bigram overlap
print(f"ROUGE-L: F={scores['rougeL'].fmeasure:.3f}")  # longest common subsequence
```

| Metric | What it measures | Best for |
|---|---|---|
| ROUGE-1 | Unigram (word) overlap | General content coverage |
| ROUGE-2 | Bigram overlap | Phrase-level coherence |
| ROUGE-L | Longest common subsequence | Sentence structure |

**When to use:** Summarisation benchmarks, evaluating extractive QA systems, or comparing summarisation models where you have human-written reference summaries.

---

### 2.4 BERTScore

BERTScore was a significant step forward in automatic evaluation quality because it moves beyond surface-level word matching to semantic similarity. Instead of checking whether the same words appear in both texts, it encodes both the reference and the candidate with a pretrained BERT model and measures how closely their token embeddings align in vector space. Two sentences that mean the same thing using different words will score high, even if they share no n-grams.

**Designed for:** Generation tasks where paraphrasing is expected. Uses contextual BERT embeddings to measure semantic similarity rather than lexical overlap.

```python
from bert_score import score

references = ["The cat sat on the mat"]
candidates = ["A feline rested on the rug"]

P, R, F1 = score(candidates, references, lang="en", verbose=False)
print(f"BERTScore F1: {F1.mean().item():.3f}")
# Much higher than BLEU — captures that cat=feline, mat≈rug
```

**How it works:**
1. Encode both the reference and candidate sentence using a pretrained BERT model (each word becomes a contextual embedding vector)
2. For each token in the *candidate*, find the most similar token in the *reference* using cosine similarity
3. **Precision**: average of the maximum similarity scores for each candidate token
4. **Recall**: average of the maximum similarity scores for each reference token
5. **F1**: harmonic mean of precision and recall

**Advantage over BLEU:** Captures semantic equivalence naturally. "Very good" and "excellent" receive high BERTScore F1 because their BERT embeddings are close in vector space, even though they share zero n-gram overlap.

---

## Part 3 — RAG-Specific Evaluation: RAGAS

### 3.1 RAGAS Metrics Overview

RAGAS (Retrieval-Augmented Generation Assessment) is designed to evaluate the full RAG pipeline rather than just the final answer. A RAG system can fail in at least two distinct places: the *retriever* might not find the right context, or the *generator* might not faithfully use the context it was given. RAGAS measures both failure modes with four complementary metrics, each of which answers a different diagnostic question about where the system is breaking down.

| Metric | Diagnostic question |
|---|---|
| `faithfulness` | Did the generator hallucinate, or is the answer grounded in the retrieved context? |
| `answer_relevancy` | Does the answer actually address the question that was asked? |
| `context_precision` | Are the retrieved chunks useful, or is the retriever returning noise? |
| `context_recall` | Did the retriever find *all* the information needed to answer completely? |

```
Faithfulness checks:
  For each statement in the answer:
    → Can this statement be inferred from the retrieved context?
    → Yes: faithful. No: hallucination.
  
  faithfulness = # statements supported by context / # total statements
```

### 3.2 RAGAS Implementation

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,     # requires ground truth
    answer_correctness,    # requires ground truth
)
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": ["What is RAG?", "How does attention work?"],
    "answer": ["RAG retrieves documents...", "Attention weights tokens..."],
    "contexts": [
        ["RAG stands for Retrieval-Augmented Generation..."],   # retrieved chunks
        ["In transformer models, attention..."],
    ],
    "ground_truth": ["RAG stands for...", "Attention mechanism..."],  # optional
}
dataset = Dataset.from_dict(eval_data)

# Run evaluation (uses LLM as judge internally)
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=openai_llm,          # judge LLM (GPT-4o recommended)
    embeddings=openai_embed, # for answer_relevancy
)
print(results)
# {'faithfulness': 0.87, 'answer_relevancy': 0.83, 'context_precision': 0.91, 'context_recall': 0.78}
```

### 3.3 RAGAS Metrics Deep-Dive

Here is the precise definition of each metric — understanding the exact calculation helps you debug low scores and explain the metrics confidently in interviews:

**`context_precision`** (signal-to-noise ratio in retrieved chunks): Measures what fraction of the retrieved chunks were actually relevant to answering the question. A high score means the retriever is surgical — it returns mostly useful context. A low score indicates a "noisy" retriever where irrelevant chunks are crowding out the useful ones, which in turn confuses the generator.

**`context_recall`** (coverage completeness): Measures how much of the ground-truth information was present in the retrieved context. A high score means the retriever successfully found all the information needed. A low score means the retriever missed important sections of the knowledge base — the generator couldn't produce a complete answer because the facts simply weren't provided to it.

**`answer_relevancy`** (does the answer address the question?): Computed as the cosine similarity between the original question embedding and the embeddings of *N hypothetical questions* generated from the answer. The intuition: if your answer fully addresses the question, then questions generated from your answer should closely resemble the original question. A low score means the answer veered off-topic or was evasive.

**`faithfulness`** (groundedness, anti-hallucination): The most important metric for RAG. Decompose the answer into atomic factual claims, then check each claim against the retrieved context using an LLM. `faithfulness = claims_supported_by_context / total_claims_in_answer`. A score of 0.87 means 87% of claims in the answer were supported by the retrieved documents. The remaining 13% were hallucinated — not present in any retrieved chunk.

---

## Part 4 — LLM-as-Judge Pattern

### 4.1 What is LLM-as-Judge?

LLM-as-Judge uses a capable frontier model (typically GPT-4o or Claude 3.5 Sonnet) to score or compare outputs produced by the model under evaluation. The appeal is significant: human-quality evaluation at automated speed and cost. A well-prompted judge model can assess nuanced dimensions like tone, helpfulness, factual grounding, and completeness in ways that no rule-based metric can match. In 2026, LLM-as-judge has become the de-facto standard for evaluating production LLM systems, used in everything from nightly regression tests to continuous A/B evaluation in shadow mode.

```python
import openai
import json

def llm_judge_response(
    question: str,
    response: str,
    context: str = None,
    judge_model: str = "gpt-4o"
) -> dict:
    """Score an LLM response on 4 dimensions using GPT-4o as judge."""
    
    system_prompt = """You are an expert evaluator. Score the given response on:
    - factual_accuracy: Is the information correct? (1=wrong, 5=perfectly accurate)
    - relevance: Does it address the question? (1=off-topic, 5=directly answers)
    - completeness: Does it cover all important aspects? (1=incomplete, 5=comprehensive)
    - clarity: Is it clear and well-structured? (1=confusing, 5=crystal clear)
    
    Return ONLY a JSON with these 4 scores and a brief justification."""
    
    context_block = f"\n\nContext: {context}" if context else ""
    
    user_message = f"""Question: {question}{context_block}

Response to evaluate: {response}

Return JSON with keys: factual_accuracy, relevance, completeness, clarity, justification."""

    completion = openai.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        response_format={"type": "json_object"}
    )
    
    return json.loads(completion.choices[0].message.content)

# Example:
scores = llm_judge_response(
    question="What is the capital of France?",
    response="Paris is the capital of France and is known for the Eiffel Tower.",
)
# → {"factual_accuracy": 5, "relevance": 5, "completeness": 4, "clarity": 5, ...}
```

### 4.2 Pairwise Comparison (A/B Evaluation)

```python
def pairwise_judge(question: str, response_a: str, response_b: str) -> str:
    """Compare two responses and pick the better one."""
    
    prompt = f"""Compare these two responses to the question:
    
    Question: {question}
    Response A: {response_a}
    Response B: {response_b}
    
    Which is better? Answer: "A", "B", or "TIE". Then briefly explain why.
    Format: {{"winner": "A/B/TIE", "reason": "..."}}"""
    
    result = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(result.choices[0].message.content)

# Use cases:
# - A/B test two prompt versions
# - Compare fine-tuned vs base model
# - Compare RAG vs no-RAG responses
```

### 4.3 G-Eval (NLP Lab, 2023)

G-Eval addresses a subtle but important flaw in simple LLM scoring prompts: if you just ask a model to "rate this response 1–5", it tends to anchor to 3–4 regardless of actual quality (a form of central tendency bias). G-Eval forces the judge to think step-by-step through an explicit evaluation rubric *before* assigning a score — the chain-of-thought process surfaces reasoning that anchors the final score more accurately to the actual quality of the output.

```python
# G-Eval uses a longer system prompt with explicit evaluation steps
g_eval_prompt = """
You will evaluate a summary of a news article.

Evaluation Steps:
1. Read the source article carefully
2. Read the generated summary
3. Check for missing key information
4. Check for factual errors vs the article
5. Assess coherence and readability
6. Assign a score 1-5 based on your analysis

Assign score: """

# G-Eval outperforms simple "rate 1-5" because the chain-of-thought
# forces the judge to actually evaluate before scoring
```

---

## Part 5 — Academic Benchmarks for LLM Capability

### 5.1 Overview Table

| Benchmark | What it Tests | Score Type | Top Models (2026) |
|---|---|---|---|
| MMLU | 57-subject multiple choice (undergrad to expert) | % accuracy | Gemini Ultra: 90.0% |
| MT-Bench | Multi-turn conversation quality | LLM score 1-10 | GPT-4o: 9.0 |
| MATH | Competition math problems (AMC/AIME) | % accuracy | o3: 96.7% |
| HumanEval | Python coding (pass@k) | pass@1 % | o3: 92.4% |
| TruthfulQA | Truthfulness + avoiding false beliefs | % truthful | Best ~85% |
| HellaSwag | Commonsense reasoning, sentence completion | % accuracy | Top models: 95%+ |
| ARC-Challenge | Science questions (Grade 3-9) | % accuracy | Top models: 95%+ |
| ARC-AGI | Novel visual reasoning tasks | % correct | o3 high: 88% |
| GPQA | PhD-level science (biology, chemistry, physics) | % accuracy | o3: ~87% |
| DROP | Reading comprehension + discrete reasoning | F1 score | GPT-4: ~83% |
| GSM8K | Grade school math word problems | % accuracy | Top models: 99%+ |

### 5.2 MMLU in Practice

```python
# MMLU tests 57 subjects: law, medicine, history, CS, statistics, physics...
# Example question:
{
    "question": "What is the time complexity of building a binary heap?",
    "choices": ["O(n log n)", "O(n)", "O(log n)", "O(n²)"],
    "answer": "B"  # O(n) — Floyd's heapify
}

# How to run MMLU evaluation:
from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model="hf",
    model_args="pretrained=meta-llama/Llama-3.2-8B",
    tasks=["mmlu"],
    num_fewshot=5,   # 5-shot prompting standard
)
print(results["results"]["mmlu"])  # {"acc": 0.742, "acc_stderr": 0.003}
```

### 5.3 HumanEval (Code Evaluation)

```python
# HumanEval: 164 programming problems from OpenAI
# Metric: pass@k — probability at least 1 of k samples passes all unit tests

# Example problem:
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other
    than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    # Model must generate this implementation correctly

# pass@1 scores (2026):
# GPT-4o: 90.2%
# Claude 3.5 Sonnet: 92.0%
# Llama 3.3 70B: 83.7%
# DeepSeek R1: 95.3%

# Why it matters for interviews: measures whether the model can generate 
# functionally correct code, not just syntactically correct.
```

---

## Part 6 — Production LLM Evaluation Pipeline

### 6.1 The Evaluation Pyramid

```
                    ┌───────────────────────────────┐
                    │   Human Evaluation (slower)    │ ← Ground truth, periodic
                    │   50-100 examples / 2 weeks    │
                    ├───────────────────────────────┤
                    │   LLM-as-Judge (automated)     │ ← Daily, continuous
                    │   GPT-4o scores every output   │
                    ├───────────────────────────────┤
                    │   Automated Metrics             │ ← Every request
                    │   RAGAS, BERTScore, length,     │
                    │   latency, cost, error rate     │
                    └───────────────────────────────┘
```

### 6.2 Building a Production Eval Pipeline

```python
import asyncio
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class EvalResult:
    question: str
    response: str
    faithfulness: Optional[float] = None
    relevance: Optional[float] = None
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    flagged: bool = False

class ProductionEvaluator:
    def __init__(self, rag_pipeline, judge_llm):
        self.rag = rag_pipeline
        self.judge = judge_llm
    
    async def evaluate_single(self, question: str) -> EvalResult:
        import time
        
        start = time.time()
        response, context = await self.rag.invoke(question)
        latency = (time.time() - start) * 1000
        
        # Automated metrics
        faithfulness_score = await self.check_faithfulness(response, context)
        relevance_score = await self.check_relevance(question, response)
        
        # Safety check
        flagged = await self.safety_check(response)
        
        return EvalResult(
            question=question,
            response=response,
            faithfulness=faithfulness_score,
            relevance=relevance_score,
            latency_ms=latency,
            flagged=flagged
        )
    
    async def check_faithfulness(self, response: str, context: list[str]) -> float:
        """Decompose response into claims, check each against context."""
        prompt = f"""Context: {' '.join(context)}
        
        Response: {response}
        
        List each factual claim in the response. For each claim, state whether
        it is fully supported by the context (yes/no).
        Return JSON: {{"claims": [{{"claim": "...", "supported": true/false}}]}}"""
        
        result = self.judge(prompt)
        data = json.loads(result)
        supported = sum(1 for c in data["claims"] if c["supported"])
        total = len(data["claims"])
        return supported / total if total > 0 else 1.0
    
    async def safety_check(self, response: str) -> bool:
        """Check for harmful content using moderation API."""
        # OpenAI Moderation API (free)
        import openai
        mod = openai.moderations.create(input=response)
        return mod.results[0].flagged

# Logging to LangSmith / MLflow
def log_eval_result(result: EvalResult, experiment_name: str):
    import mlflow
    with mlflow.start_run(run_name=experiment_name):
        mlflow.log_metrics({
            "faithfulness": result.faithfulness,
            "relevance": result.relevance,
            "latency_ms": result.latency_ms,
        })
```

### 6.3 Continuous Evaluation Strategy

```
Daily automated runs:
  1. Run 50 golden questions from test set
  2. Compute RAGAS metrics for all
  3. Alert if faithfulness drops below 0.80
  4. Alert if P99 latency exceeds SLA
  5. Log all to LangSmith / W&B

Weekly human review:
  1. Sample 20 flagged/low-scoring responses
  2. Human rater completes structured rubric
  3. Update golden test set with new edge cases discovered

Monthly model review:
  1. Run full golden set (200+ questions) before any model change
  2. A/B test new model/prompt version vs current
  3. Require: faithfulness ≥ 0.85, no regression on any metric
  4. Deploy only if stats. significant improvement on target metric
```

---

## Part 7 — Custom Domain Evaluation

### 7.1 Building Your Golden Test Set

```python
# Structure of a golden evaluation dataset
test_cases = [
    {
        "id": "001",
        "question": "What was our Q3 2025 revenue?",
        "ground_truth": "Q3 2025 revenue was $2.3B, up 12% YoY",
        "source_docs": ["earnings_q3_2025.pdf"],
        "difficulty": "easy",
        "category": "financial"
    },
    {
        "id": "002",
        "question": "Explain our return policy for electronics",
        "ground_truth": "Electronics must be returned within 30 days...",
        "source_docs": ["policy_handbook_2025.pdf"],
        "difficulty": "medium",
        "category": "policy"
    }
]

# Guidelines for building golden set:
# 1. Cover all categories/domains the system handles
# 2. Include easy, medium, and hard questions proportionally
# 3. Include questions with no answer (should say "I don't know")
# 4. Include adversarial questions (jailbreak attempts, trick questions)
# 5. Maintain at least 200 examples for statistical significance
# 6. Update quarterly with real user failures
```

### 7.2 A/B Testing LLM Versions

```python
# Shadow mode A/B: run new model in parallel, compare offline
class ABEvaluator:
    def __init__(self, model_a, model_b, judge):
        self.a = model_a
        self.b = model_b
        self.judge = judge
    
    def run_ab_test(self, questions: list[str], n_per_question: int = 5) -> dict:
        results = {"a_wins": 0, "b_wins": 0, "ties": 0}
        
        for question in questions:
            # Generate multiple times to account for stochasticity
            a_responses = [self.a.generate(question) for _ in range(n_per_question)]
            b_responses = [self.b.generate(question) for _ in range(n_per_question)]
            
            # Best-of-N evaluation
            best_a = self.judge.pick_best(a_responses, question)
            best_b = self.judge.pick_best(b_responses, question)
            
            winner = self.judge.compare(question, best_a, best_b)
            results[f"{winner.lower()}_wins"] += 1
        
        # Run statistical significance test
        from scipy.stats import binom_test
        n_trials = results["a_wins"] + results["b_wins"]
        p_value = binom_test(results["b_wins"], n_trials, 0.5)
        
        results["p_value"] = p_value
        results["significant"] = p_value < 0.05
        return results
```

---

## Part 8 — Interview Questions & Answers

**Q1: What is RAGAS faithfulness and how is it calculated?**

> Faithfulness measures whether the generated answer is grounded in the retrieved context — i.e., the model didn't hallucinate facts not in the source documents. Calculation: (1) Decompose the answer into atomic factual claims using an LLM. (2) For each claim, check whether it can be inferred from the retrieved context using another LLM call. (3) Faithfulness = claims_supported / total_claims. A score of 0.87 means 87% of claims in the answer were supported by retrieved documents.

**Q2: When would you use BLEU vs ROUGE vs BERTScore?**

> BLEU is designed for translation tasks and measures n-gram precision against a reference. Use it when you have an exact reference and care about word-for-word accuracy. ROUGE measures recall (how much of the reference appears in the output) and is standard for summarization. BERTScore uses semantic embeddings to compare outputs — it correctly assigns high similarity to "excellent" vs "very good" whereas BLEU/ROUGE give 0. For most LLM evaluation in 2026, BERTScore or LLM-as-judge is preferred over BLEU/ROUGE because LLMs generate equivalent answers in many ways.

**Q3: What is "LLM-as-Judge" and what are its limitations?**

> LLM-as-Judge uses a capable model (typically GPT-4o or Claude 3.5 Sonnet) to score or compare outputs from the model under evaluation. Strengths: captures nuanced quality dimensions that rule-based metrics miss, scales cheaply, aligns well with human preferences. Limitations: (1) Positional bias — judges tend to favour the first option presented in pairwise comparisons; mitigate by swapping order and averaging. (2) Verbosity bias — longer responses often score higher even if not better. (3) Self-enhancement bias — GPT-4o may favour GPT-4o outputs. (4) LLM judge can itself hallucinate. Mitigations: use calibrated prompts, compare against human annotations on a sample, use multiple judges.

**Q4: How do you evaluate an LLM fine-tuned for a specific domain?**

> A multi-level evaluation: (1) **Automated benchmark**: run MMLU or a domain-specific benchmark before/after fine-tuning to check for catastrophic forgetting. (2) **Golden test set**: curate 200+ domain-specific Q&A pairs with ground truth; compute faithfulness, answer correctness, and BERTScore. (3) **LLM-as-judge**: have GPT-4o score accuracy, completeness, and tone compliance on 1-5 scale. (4) **Human eval**: have domain experts review 50-100 examples with a structured rubric. (5) **A/B test**: shadow-deploy the new model, compare pairwise against current production model on real user queries.

**Q5: What metrics would you track in a production LLM system dashboard?**

> I monitor two categories: **Quality metrics**: RAGAS faithfulness (target ≥ 0.85), answer relevancy (target ≥ 0.80), hallucination rate estimated by a classifier, thumbs up/down from users. **Operational metrics**: P50/P95/P99 latency per request, tokens per second, cost per query, error rate (timeouts, context length exceeded), retry rate, and cache hit rate. I set alerting thresholds — e.g., if faithfulness drops below 0.75 for 50+ queries in a rolling hour, page on-call with a summary of failing examples.

**Q6: What is perplexity and when is it a useful metric?**

> Perplexity measures how well a language model predicts a sample: $\text{PPL} = \exp\!\left(-\dfrac{1}{N}\sum_{i=1}^{N} \log P(\text{token}_i \mid \text{context})\right)$. Lower perplexity means the model assigns higher probability to the text — it "expected" the text. Useful for: (1) Comparing model variants during pre-training (lower PPL = better language model). (2) Detecting distribution shift — if model's perplexity on production inputs increases, the input distribution has changed. (3) Detecting AI-generated text (generated text has anomalously low perplexity under the generating model). Not useful for: measuring factual accuracy, helpfulness, or safety — a hallucinating model can have low perplexity.

**Q7: Design an evaluation framework for a customer support RAG chatbot.**

```
Framework for customer support RAG evaluation:

1. Automated (runs every 30 min):
   - RAGAS faithfulness on sampled conversations
   - Answer relevancy score (LLM-as-judge)
   - Response time P95
   - Escalation rate (% of chats handed to human agent)
   - Resolution rate (% of issues resolved without escalation)

2. Daily:
   - Run 100-question golden test set
   - Compare faithfulness vs prior 7-day average
   - Monitor for drift in question distribution (new topics emerging)

3. Weekly:
   - Human review of 20 lowest-scoring conversations
   - Human review of 20 conversations where customer expressed frustration
   - Update golden set with newly discovered failure modes

4. Monthly:
   - Full A/B test before any model/prompt change
   - Customer satisfaction survey correlation with automated metrics
   - Cost per resolved ticket tracking
   
Alerting:
   - faithfulness < 0.75 → immediate on-call alert
   - escalation rate > 30% → alert engineering + product
   - P95 latency > 5s → alert infrastructure team
```

---

*LLM Evaluation & Metrics Guide | Phase 2 Supplementary | Added April 2026*
