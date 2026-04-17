# Week 3: LLM Fundamentals — How Language Models Work
### Phase 1 | Month 1 | April 21–27, 2026

> This week shifts from DSA to AI/LLM theory — the core knowledge tested in FAANG's LLM/GenAI interview rounds.  
> Continue DSA mornings (review Week 1–2 problems). Use evenings for this content.

---

## 🎯 Learning Objectives

By the end of this week you will be able to:
- Explain the complete lifecycle of a language model (pre-training → fine-tuning → inference)
- Describe tokenization, embeddings, and why they matter
- Explain attention and transformer architecture at an intuitive level
- Contrast BERT vs GPT architectures and use cases
- Answer 30+ FAANG-level LLM interview questions with confidence

---

## Part 1 — What Is a Language Model?

> 📖 **Read this first — the big picture before any maths.**

Imagine you're texting a friend. You type "Happy birth" and your phone suggests "day". You type "Let's grab" and it offers "coffee" or "lunch". That suggestion feature on your phone is a tiny, primitive language model.

A modern LLM (Large Language Model) like ChatGPT does exactly the same thing — just vastly more sophisticated. It reads everything you've written and predicts what word (strictly: what *token*) should come next. That's it. That's the entire job.

The magic is: when you train this simple prediction task on **trillions** of words from the internet — books, Wikipedia, code, conversations, research papers — the model is forced to understand *everything*: grammar, facts about the world, reasoning patterns, cause and effect, how to write code, how to explain ideas. All of that knowledge is a side effect of getting really, really good at "guess the next word."

Think of it like this: a student who reads every book ever written and is then tested by filling in blanks — to do well, they would have to deeply understand every topic. That's how LLMs learn.

### 1.1 The Core Task: Next-Token Prediction

A language model (LM) is trained to predict the next token given a sequence of previous tokens:

$$P(\text{token}_t \mid \text{token}_1, \text{token}_2, \ldots, \text{token}_{t-1})$$

That's it. Everything — translation, summarisation, code generation, question answering — emerges from training this one objective on massive amounts of text.

**Why does next-token prediction lead to general intelligence?**

Here's a concrete way to think about it: imagine you're playing a fill-in-the-blank game with these sentences:

- "The capital of France is ___" → To fill in *Paris*, you need to know geography.
- "def fibonacci(n): return ___" → To fill in the right code, you need to understand recursion.
- "She was angry because he forgot their ___" → To predict *anniversary*, you need to understand human relationships.
- "The patient was given a dose of ___" → To predict *medication*, you need medical context.

Every single one of these requires *genuine understanding*. And the model sees hundreds of billions of such examples. World knowledge and reasoning capabilities are a byproduct of learning to predict text very well. This is why the same model can write poetry, debug code, and explain quantum physics — it has been tested on all of them through next-token prediction.

### 1.2 Brief History

| Year | Model | Organisation | Key Innovation |
|---|---|---|---|
| 2017 | Transformer | Google | "Attention is All You Need" paper |
| 2018 | BERT | Google | Bidirectional encoder, masked LM |
| 2018 | GPT-1 | OpenAI | Decoder-only, next-token prediction |
| 2019 | GPT-2 | OpenAI | Scale (1.5B params), "too dangerous to release" |
| 2020 | GPT-3 | OpenAI | Few-shot learning emerges at 175B params |
| 2022 | InstructGPT | OpenAI | RLHF — align model to follow instructions |
| 2022 | ChatGPT | OpenAI | RLHF + dialogue fine-tuning |
| 2023 | LLaMA | Meta | Open-source, efficient smaller models |
| 2023 | Mistral 7B | Mistral AI | SLM, grouped query attention |
| 2024 | LLaMA 3 | Meta | State-of-SOTA open-source |
| 2024 | Claude 3 | Anthropic | Long context (200K), constitutional AI |
| 2025 | GPT-4o, Gemini 1.5 | OpenAI, Google | Multimodal, very long context |

---

## Part 2 — Tokenization

> 📖 **The big picture:** Before a model can read your text, it needs to convert it into numbers. Computers only understand numbers. Tokenization is the process of splitting text into small pieces called *tokens*, then mapping each token to a number (an ID). Think of it like translating English letters into Morse code — dots and dashes are what the telegraph understands, just as token IDs are what the model understands.

Here's a concrete example. When you type:
> "I love pizza"

The tokenizer might split it like:
> `["I", " love", " pizza"]` → `[40, 2928, 17953]`

Now the model works with numbers `[40, 2928, 17953]` — never the original letters. Everything the model does internally is matrix multiplication on these numbers.

**So why not just give each word its own number?** And why not just use letters? Great questions — here's why neither extreme works well:

### 2.1 Why Not Characters or Words?

| Approach | Problem |
|---|---|
| Character-level | Very long sequences (slow attention), no semantic meaning units |
| Word-level | Huge vocabulary (100K+ words), OOV problem for new/rare words |
| Subword (BPE) | ✅ Balance: reasonable vocabulary (~50K), handles any word |

### 2.2 Byte-Pair Encoding (BPE) — GPT's Tokenizer

> 💡 **ELI5 (Explain Like I'm 5):** BPE is like having a really smart autocomplete that notices which letter combinations appear together all the time. If the letters "i" and "n" and "g" always appear together at the end of words, it makes sense to treat "ing" as one unit rather than three separate letters. BPE formalises this idea: start with individual characters, then keep merging the most common pairs until you've built a vocabulary of ~50,000 units. Common words become single tokens; rare words are broken into recognisable sub-pieces.

BPE starts with individual characters and iteratively merges the most frequent adjacent pairs:

```
Step 0: corpus = "low lower lowest slower slowest"
Step 0: tokens: l o w _ l o w e r _ l o w e s t _ s l o w e r _ s l o w e s t

Step 1: Most frequent pair: ('l', 'o') → merge to 'lo'
        lo w _ lo w e r _ lo w e s t _ s lo w e r _ s lo w e s t

Step 2: Most frequent pair: ('lo', 'w') → merge to 'low'
        low _ low e r _ low e s t _ s low e r _ s low e s t

... continues until vocabulary size reached
```

Result: Common words are single tokens. Rare words split into common subwords. "unbelievable" → "un", "believ", "able".

### 2.3 WordPiece — BERT's Tokenizer

Similar to BPE but uses `##` to indicate continuation:
- "playing" → "play", "##ing"
- "unbelievable" → "un", "##believ", "##able"

### 2.4 Important Tokenization Facts for Interviews

```python
from tiktoken import encoding_for_model

# GPT-4 tokenizer
enc = encoding_for_model("gpt-4")
tokens = enc.encode("Hello, how are you today?")
print(tokens)        # [9906, 11, 1268, 527, 499, 3432, 30]
print(len(tokens))   # 7 tokens for 7 words (roughly 1 token ≈ 0.75 words)

# Key rule of thumb: 1 token ≈ 4 characters ≈ 0.75 English words
# "ChatGPT is great" → roughly 4 tokens
# 1000 tokens ≈ 750 words ≈ ~3 pages
```

**Interview Q:** "Why do models sometimes fail on simple arithmetic like 9.9 > 9.11?"
> Numbers are tokenized unexpectedly. "9.11" might be one token or multiple ("9", ".", "11"). Models reason over tokens, not characters — they don't see individual digits unless each digit is its own token.

**Interview Q:** "What's a context window?"
> The maximum number of tokens the model can process in a single forward pass. GPT-4 = 128K tokens. Claude 3 = 200K tokens. Everything outside the context window is completely invisible to the model.

---

## Part 3 — Embeddings

> 📖 **The big picture:** Earlier we said the tokenizer converts words into numbers (IDs) like `"pizza" → 17953`. But here's the problem: that number 17953 is meaningless on its own. The number 17952 (the word "pasta") doesn't feel "related" to 17953 just because they're one apart. We need a smarter way to represent words — one where *similar words are mathematically close together.*
>
> That's what embeddings do. An embedding is a list of numbers (a *vector*) that captures the meaning of a word. Think of it like GPS coordinates: two cities that are geographically close have similar coordinates. Two words that are semantically similar should have similar coordinates in "meaning space."

**The map analogy in detail:**

Imagine a 2D map where we place words based on their meaning:
- "king" and "queen" are placed very close together in a "royalty" region of the map
- "dog" and "cat" are close together in a "household pets" region
- "Paris" and "London" are close in a "European capitals" region
- But "king" and "dog" are far apart — they have nothing in common

An embedding is exactly this, but instead of 2 dimensions on a flat map, the model uses 768 or 4096 dimensions. More dimensions means more nuance — it can capture many aspects of meaning simultaneously (is it a person? is it royalty? is it female? is it alive? etc.).

### 3.1 What Is an Embedding?

An embedding is a **dense vector representation** learned to capture semantic meaning.

```
"king"  → [0.25, -0.77, 0.14, ..., 0.89]  # 768-dimensional vector
"queen" → [0.24, -0.75, 0.18, ..., 0.91]  # similar direction = similar meaning
"dog"   → [-0.42, 0.31, -0.67, ..., 0.12] # different direction = different meaning
```

**Famous property:** king - man + woman ≈ queen (Word2Vec, 2013)

> 💡 **What does "king - man + woman ≈ queen" actually mean?**
> Think of it as algebra with meanings. The difference between "king" and "man" represents the concept of "royalty-without-gender". Add "woman" to that and you get "female + royalty" = "queen". This isn't magic — the model learned this purely by predicting text. It observed that king/queen appear in similar contexts to man/woman and encoded this relationship in the numbers.

### 3.2 How Token Embeddings Work in LLMs

> 💡 **ELI5 (Explain Like I'm 5):** Before a word enters the transformer, it gets converted from a token ID (just a number) into a rich vector. This happens via an *embedding table* — think of it like a very large dictionary where every word has an entry, and instead of a text definition, the entry is a list of ~768 numbers that encode meaning.
>
> So the model looks up "cat" → finds row 3797 in the embedding table → gets back a 768-number vector. Now "cat" has been translated into a language the transformer can work with — pure maths.

```
Input: "The cat sat"
       ↓ tokenize
       [464, 3797, 3332]  (token IDs)
       ↓ embedding lookup (like an index into a learned matrix)
       [[0.1, 0.3, ...],   # embedding for "The"
        [0.5, -0.2, ...],  # embedding for "cat"
        [0.2, 0.8, ...]]   # embedding for "sat"
        
Each vector: d_model dimensions (e.g. 768 for BERT-base, 4096 for LLaMA 7B)
```

### 3.3 Positional Embeddings

> 📖 **The problem they solve:** Imagine you receive a bag of scrabble tiles with the words "cat", "the", "sat". You know all the words but not their order. "The cat sat" and "Cat the sat" have the exact same tiles. For language understanding, *order matters enormously*. "Dog bites man" ≠ "Man bites dog."
>
> Here's the catch: the Transformer processes **all tokens at once in parallel** (unlike an RNN which reads left-to-right one step at a time). This is great for speed, but it means the model has no built-in sense of position — all tokens arrive simultaneously. Positional embeddings solve this by adding a "position tag" to each token's vector before it enters the transformer.
>
> Think of it like seat numbers at a cinema. Two people with identical appearance can be told apart by their seat number (position 1, position 2, etc.). We add a little "seat number signal" to each word's embedding so the model always knows where it sits in the sentence.

Transformers process all tokens **in parallel** — they have no inherent notion of sequence order. Positional embeddings add position information to token embeddings.

**Approach 1: Sinusoidal (original Transformer paper)**
$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**Approach 2: Learned positional embeddings** — just another embedding table indexed by position. Simple, works well.

**Approach 3: RoPE (Rotary Position Embedding)** — used in LLaMA, Mistral, GPT-NeoX.
- Encodes position by **rotating** the query and key vectors in attention
- Key advantage: relative positions are naturally captured, generalises better to longer sequences than seen in training

**Approach 4: ALiBi (Attention with Linear Biases)** — used in MPT, BLOOM.
- Adds a linear bias to attention scores based on distance between tokens
- Penalises attending to distant tokens, naturally handles long sequences

### 3.4 Sentence/Document Embeddings (for RAG)

Different from token embeddings — represent entire sentences/paragraphs as single vectors.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim
embeddings = model.encode(["The cat is on the mat", "A feline rests on the rug"])
# Cosine similarity ≈ 0.85 (semantically similar but different words)
```

**Models to know:**
- `all-MiniLM-L6-v2` — fast, 384-dim, good for most RAG tasks
- `text-embedding-3-small` / `text-embedding-3-large` — OpenAI APIs
- `BAAI/bge-large-en` — strong open-source option
- `cohere-embed-v3` — multilingual, strong reranking

---

## Part 4 — The Transformer Architecture

> 📖 **The big picture:** Now we understand what goes *into* a transformer (tokenized, embedded, position-tagged numbers). But what happens *inside* the black box? This is where things get beautifully designed.
>
> A Transformer is like an assembly line for understanding language. Your sentence enters at one end as raw token embeddings, passes through a series of identical "understanding stations" (called *Transformer Blocks*), and comes out the other end as rich, context-aware representations that the model uses to predict the next token.
>
> Each block has two jobs:
> 1. **Attention** — let each word "talk to" every other word and update its own meaning based on context
> 2. **Feed-Forward Network** — independently process each word's updated representation to extract deeper features
>
> The same block structure repeats N times (e.g. 12 times for BERT-base, 32 times for LLaMA 7B, 96 times for GPT-4). Each repetition allows the model to build increasingly abstract understanding — early layers handle syntax, later layers handle semantics and facts.
>
> Let's walk through each component carefully.

### 4.1 High-Level Architecture

```
Input Tokens
    ↓
Token Embeddings + Positional Embeddings
    ↓
┌─────────────────────────────────────────────┐
│  Transformer Block (×N, e.g. 12, 24, 32)   │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │   Multi-Head Self-Attention          │   │
│  │   (with residual connection)         │   │
│  └────────────────┬────────────────────┘   │
│                   │ + residual              │
│  ┌────────────────▼────────────────────┐   │
│  │   Layer Normalisation               │   │
│  └────────────────┬────────────────────┘   │
│                   │                         │
│  ┌────────────────▼────────────────────┐   │
│  │   Feed-Forward Network (FFN)         │   │
│  │   (with residual connection)         │   │
│  └────────────────┬────────────────────┘   │
│                   │ + residual              │
│  ┌────────────────▼────────────────────┐   │
│  │   Layer Normalisation               │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
    ↓ (×N times)
Linear Projection + Softmax
    ↓
Next-token probability distribution
```

### 4.2 Encoder vs Decoder vs Encoder-Decoder

| Architecture | Models | Attention Type | Best For |
|---|---|---|---|
| **Encoder-only** | BERT, RoBERTa, DistilBERT | Bidirectional (sees all tokens) | Classification, NER, embeddings |
| **Decoder-only** | GPT, LLaMA, Mistral, Claude | Causal (only sees past tokens) | Text generation, chat, completion |
| **Encoder-Decoder** | T5, BART, Marian | Enc: bidirectional, Dec: causal | Translation, summarisation |

### 4.3 Self-Attention (The Core)

> 📖 **The intuition — let's really understand this:**
>
> Consider the sentence: **"The animal didn't cross the street because *it* was too tired."**
>
> What does "it" refer to? The animal or the street? Humans immediately know it's the animal — because animals get tired, streets don't.
>
> Now read: **"The animal didn't cross the street because *it* was too wide."**
>
> Now "it" means the street — because streets can be wide, animals aren't.
>
> The word "it" appears in the same position in both sentences, but its meaning completely changes based on *other words in the sentence*. This is the core problem that self-attention solves.
>
> **Self-attention lets each word look at all other words in the sentence and update its own meaning based on what's relevant.** The word "it" will "look at" both "animal" and "street", decide that "animal" is more relevant (when "tired" is present), and update its representation accordingly.
>
> Here's a human analogy: imagine you're in a meeting room with 10 people. You have a question. You briefly make eye contact with each person and judge "does this person know something relevant to my question?" Then you absorb information proportionally from each person — a lot from the relevant expert, a little from everyone else. That's exactly what self-attention does for every word in a sentence.
>
> **The Query, Key, Value framework:**
>
> The maths uses three concepts borrowed from information retrieval (like a database search):
> - **Query (Q):** What you're searching for. "I am the word 'it' — I need to find what I refer to."
> - **Key (K):** What each item advertises. "I am 'animal', here's my description." "I am 'street', here's mine."
> - **Value (V):** The actual content to return. "I am 'animal', here's my full information to share."
>
> The attention score between "it" and "animal" is computed as: how well does "it's Query" match "animal's Key"? If the match is high, "it" borrows a lot of "animal's Value". Low match = borrow very little.

**Intuition:** For each token, compute a weighted average of all other tokens' information, where the weights represent "how relevant is each token to the current one?"

**Math:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where:
- **Q (Query):** "What am I looking for?" — current token's question
- **K (Key):** "What do I offer?" — each token's description
- **V (Value):** "What information do I provide?" — each token's content
- $d_k$ = dimension of keys (scaling prevents softmax from saturating for large $d_k$)

```python
import torch
import torch.nn.functional as F

def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)  # (batch, heads, seq, seq)
    weights = F.softmax(scores, dim=-1)
    return weights @ V
```

**Why divide by $\sqrt{d_k}$?**
For large $d_k$, dot products grow large (variance $= d_k$ for unit vectors). The softmax would then create very peaked distributions (near one-hot), causing vanishing gradients. Dividing by $\sqrt{d_k}$ normalises the variance back to $\approx 1$.

### 4.4 Multi-Head Attention

> 📖 **Why run attention multiple times?**
>
> Think about reading a complex legal contract. To fully understand it, you might read it multiple times — once looking for obligations ("who must do what"), once for dates and deadlines, once for financial clauses, once for exceptions and caveats. Each reading focuses on a different *type* of relationship.
>
> Multi-head attention does the same thing for language. Instead of running attention once, it runs it `h` times in parallel (e.g. 12 times for BERT, 32 times for LLaMA 7B). Each "head" learns to focus on different types of linguistic relationships:
> - One head might specialise in tracking pronouns back to their antecedents ("it → animal")
> - Another might track subject-verb agreement ("birds → fly", not "birds → flies")
> - Another might capture positional patterns (words close together are often related)
> - Another might handle syntactic dependencies (adjective modifying a noun)
>
> None of this is programmed in — the heads discover these specialisations automatically during training. After all heads run independently, their outputs are concatenated and mixed back together with a final linear layer.

Run attention multiple times in parallel with different learned projections, then concatenate.

```
Input: seq_len × d_model

Split into h heads, each with dimension d_k = d_model / h
Head 1: Q₁, K₁, V₁ → output₁  (might learn syntactic relationships)
Head 2: Q₂, K₂, V₂ → output₂  (might learn coreference)
...
Head h: Qₕ, Kₕ, Vₕ → outputₕ  (might learn positional patterns)

Concatenate: [output₁ | output₂ | ... | outputₕ]
Linear projection: W_O × concatenated → d_model
```

**Why multiple heads?** Each head can attend to different relationships simultaneously. One head might learn to resolve pronouns, another to track subject-verb agreement.

### 4.5 Feed-Forward Network (FFN)

> 📖 **What does it do?** After attention lets words "talk to each other" and gather context, the FFN is each word's chance to "think in isolation" about what that context means. It's a simple 2-layer neural network applied independently to every token's vector.
>
> A useful analogy: imagine each employee in a company has just finished a team meeting (= attention step) and gathered input from colleagues. Now they go back to their own desk to independently process that input and decide what to do next (= FFN step). The meeting is about communication; the desk work is about individual computation.
>
> **A key insight:** Research has shown that the FFN layers are where the model's "factual knowledge" is primarily stored. When someone asks "What's the capital of France?" — the attention layers identify what's being asked and the FFN layers retrieve the answer "Paris" from what feels like a key-value memory in the weights.

After attention, each position goes through an identical 2-layer FFN:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

In modern LLMs, `ReLU` is replaced with `GELU` or `SiLU (Swish)`, and dimension expands 4×:

```python
class FFN(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.w2(self.act(self.w1(x)))
```

**Insight:** While attention aggregates information across tokens, FFN processes each token independently. FFN is thought to store factual knowledge ("Paris is the capital of France" → key-value memories in the weight matrices).

### 4.6 Layer Normalisation

> 📖 **Why do we need it?** As data flows through many layers, the numbers can easily spiral out of control — some activations explode to millions, others collapse to near-zero. When this happens, gradients (which drive learning) become either enormous or vanishing, and training breaks down.
>
> Layer normalisation is like a standardising rinse between each layer. After each attention or FFN block, it resets the numbers to have a mean of 0 and a standard deviation of 1, then lets the model apply its own learned scaling. Think of it as a volume control that always resets the gain to a sensible baseline so nothing blows out your speakers (the gradients).

**Pre-Norm vs Post-Norm:**

Original paper used **Post-Norm**: residual → add → layer norm  
Modern LLMs (GPT-3, LLaMA) use **Pre-Norm**: layer norm → sublayer → add residual

```python
# Post-Norm (original)
x = LayerNorm(x + Attention(x))

# Pre-Norm (modern — better training stability)
x = x + Attention(LayerNorm(x))
```

**Why Pre-Norm is better:** At initialisation, the identity path (residual) dominates. The model starts close to identity mappings, making gradients flow better and training more stable at scale.

---

## Part 5 — BERT vs GPT: Know Both Cold

> 📖 **The big picture first:**
>
> There are two fundamentally different ways to "read" a sentence, and BERT and GPT represent these two approaches.
>
> **BERT's approach — reading the whole book at once (bidirectional):**
> Imagine you're given a test where you see an entire paragraph with one word blanked out: "The _____ sat on the mat." You get to read the full sentence — words before AND after the blank. With "on the mat" as context, you confidently fill in "cat." This is how BERT works. It reads the full context in both directions before making any judgment.
>
> **GPT's approach — reading left-to-right, covering what comes next (causal):**
> Imagine instead you're reading a story word by word, with a piece of paper covering the next word. You can only see what came before. "The cat ___" → you predict "sat" or "jumped". Then "The cat sat ___" → you predict "on". You never peek ahead. This is exactly how GPT generates text — one token at a time, only ever looking backward.
>
> These two different "reading styles" make BERT and GPT suited for very different tasks:
> - **BERT** is great at understanding tasks (classification, NER, question answering from a document) because it can see the full picture at once.
> - **GPT** is great at generation tasks (chatting, writing, coding) because it naturally "writes" the next word by looking only at what came before.

### 5.1 BERT (Bidirectional Encoder Representations from Transformers)

```
Input: "The [MASK] sat on the mat"

   The  [MASK]  sat   on   the  mat
    ↓     ↓     ↓     ↓     ↓    ↓
  (bidirectional: each token attends to ALL other tokens)
    ↓     ↓     ↓     ↓     ↓    ↓
  emb   emb   emb   emb   emb  emb
    
Predict: [MASK] = "cat"  (Masked Language Model objective)
```

**Pre-training tasks:**
1. **Masked Language Modeling (MLM):** Mask 15% of tokens, predict them
2. **Next Sentence Prediction (NSP):** Given sentence A and B, predict if B follows A (controversial — later shown to not help much)

**Use BERT for:**
- Text classification
- Named entity recognition (NER)
- Semantic search / embeddings
- Question answering (extractive)

### 5.2 GPT (Generative Pre-trained Transformer)

```
Input: "The cat sat"
                         ← [MASK] (can't see future)
   The  cat   sat [next]
    ↓    ↓     ↓    ↓
  (causal: each token attends only to LEFT/previous tokens)
    ↓    ↓     ↓    ↓
   emb  emb   emb  emb

Predict: [next] = "on"  (Causal Language Model objective)
```

**Causal mask (upper triangular):**
```
        The  cat  sat  on
The  [  1    0    0    0 ]
cat  [  1    1    0    0 ]   0 = masked (cannot attend)
sat  [  1    1    1    0 ]   1 = can attend
on   [  1    1    1    1 ]
```

**Use GPT/Decoder-only for:**
- Text generation
- Chat / instruction following
- Code generation
- Anything involving generating new text

### 5.3 Scaling Laws

> 📖 **Plain English:** Bigger models trained on more data are better — but how much bigger, and how much more data? Scaling laws give us the answer. They're essentially "rules of thumb with maths behind them" for how to spend a training budget wisely.
>
> **The Chinchilla insight (2022):** Before Chinchilla, companies were racing to build bigger models (GPT-3 was 175B parameters). But DeepMind discovered that GPT-3 was actually *undertrained* — they used a massive model but didn't give it enough data to match the model's capacity. It's like hiring 175 PhD students but only giving them one textbook to read.
>
> The optimal rule: **train model parameters ≈ number-of-training-tokens / 20.** If you have 1 trillion tokens of text, your ideal model size is about 50 billion parameters. More importantly: a smaller model trained on lots of data often beats a bigger model trained on little data.
>
> **Practical implication:** This is why LLaMA 3 8B (only 8 billion parameters) can outperform GPT-3 (175B parameters) on many tasks — it was trained on 15 trillion tokens, massively more than GPT-3's 300 billion.

Chinchilla (DeepMind, 2022) finding:
> For a fixed compute budget, the optimal model size and training tokens satisfy:  
> **N_optimal ≈ tokens / 20**, and training tokens ≈ 20 × model parameters

Key takeaway: **LLaMA 7B trained on 1 trillion tokens** is better than GPT-3 175B trained on 300B tokens (Chinchilla-optimal).

```
Model         Params   Tokens       Chinchilla-optimal?
GPT-3         175B     300B         NO (undertrained)
LLaMA-1 65B   65B      1.4T         YES
LLaMA-3 8B    8B       15T          OVER-trained (intentional for inference efficiency)
Chinchilla    70B      1.4T         YES (the benchmark)
```

---

## Part 6 — The LLM Lifecycle

> 📖 **The big picture — think of it like a student's education:**
>
> Building a production LLM is a three-stage journey, and it maps surprisingly well to how a human expert is trained:
>
> | Stage | Human Analogy | LLM Equivalent |
> |---|---|---|
> | **Pre-training** | 18 years of school, university — absorbing huge amounts of general knowledge | Training on trillions of tokens from the internet; model learns language, facts, reasoning |
> | **Fine-tuning (SFT)** | Professional certification / job training — learning to apply knowledge in a specific way | Training on (instruction, response) examples; model learns to *be helpful* |
> | **RLHF** | Performance reviews + feedback — guided by managers to correct bad habits | Human raters rank outputs; model learns to be helpful, harmless, and honest |
> | **Inference** | The employee working on real tasks | Taking user queries and generating responses |
>
> A raw pre-trained model is like a brilliant but unsocialised student — it knows everything but acts strangely. It might complete "Tell me how to bake a cake" by writing "...and then my grandmother said..." rather than actually giving you a recipe. Fine-tuning and RLHF teach it to be a helpful assistant.

### 6.1 Pre-training

> 📖 **What actually happens:** The model is shown enormous amounts of text (think: most of the internet, millions of books, all of GitHub). For every piece of text, it tries to predict each next word. When it's wrong, the error is measured and the weights are adjusted slightly — billions of times, across trillions of examples. After months of this on thousands of GPUs, the model has compressed the statistical patterns of virtually all human knowledge into its weights.
>
> **Key insight:** The model never "stores" facts like a database. It learns *patterns*. "Paris" follows "The capital of France is" because that pattern appeared millions of times. This is why models can also hallucinate — they're generating statistically likely continuations, not retrieving ground-truth facts.

```
Massive text corpus (Common Crawl, books, GitHub, Wikipedia...)
         ↓
Next-token prediction (unsupervised)
         ↓
Base model (knows language, facts, code — but doesn't "chat")
```

**Cost:** GPT-4 estimated ~$100M. LLaMA 3 8B: ~$5M. Mistral 7B: ~$1–2M.

### 6.2 Fine-tuning

> 📖 **Why is this needed?** The base (pre-trained) model is brilliant but not useful. If you ask it "How do I make pasta?", it might continue with "...was a question my mother always asked me when I was young..." — because on the internet, that's a plausible continuation of a conversational setup. It's completing text, not answering your question.
>
> Fine-tuning fixes this with a much smaller dataset of *correct examples*: thousands of (question, ideal answer) pairs written by humans. Now the model learns "when someone asks me a question, I should *answer it*." This transforms the base model into an instruction-following assistant.

**Supervised Fine-tuning (SFT):**
```
(instruction, response) pairs from human curators
         ↓
Train model to produce human-like responses
         ↓
"Instruction-tuned" model (like LLaMA-3-Instruct)
```

**RLHF (Reinforcement Learning from Human Feedback):**

> 📖 **Why SFT alone isn't enough:** A fine-tuned model can follow instructions, but it still doesn't know exactly what "quality" looks like. Two different answers to the same question might both be grammatically fine, but one is more accurate, helpful, and safe. SFT can't easily capture this because writing a perfect answer is hard, but *ranking two answers* is easy for humans.
>
> RLHF exploits this: show human raters two model outputs and ask "which is better?" Collect thousands of these comparisons, train a *reward model* to predict human preference scores, then train the main LLM to score highly on that reward model. It's like game training — the model keeps trying to get a higher score (human approval).
>
> **Why this matters:** This is what made ChatGPT dramatically better than GPT-3 overnight. Same architecture, same data, but RLHF made it genuinely helpful and safe.

```
Multiple model responses
         ↓
Human rankers rank responses (best → worst)
         ↓
Train reward model to predict human rankings
         ↓
PPO (Proximal Policy Optimisation) to maximise reward
         ↓
Aligned model (helpful, harmless, honest)
```

### 6.3 Inference

> 📖 **How text generation actually works — step by step:**
>
> People often think ChatGPT "writes the whole reply at once." It doesn't. It generates **one token at a time** in a loop:
>
> 1. You send: "What is the capital of France?"
> 2. The model reads all tokens, runs a forward pass through all layers, and produces a **probability distribution** over its 100K-word vocabulary: "Paris" gets 72%, "Lyon" gets 4%, "France" gets 2%, etc.
> 3. It picks a token (e.g. "Paris") based on this distribution
> 4. "Paris" is appended to the sequence. Now the model reads: "What is the capital of France? Paris"
> 5. It runs *another* full forward pass and predicts the next token: " is" → `.` → end
> 6. This loop repeats until a stop token (like `<|endoftext|>`) is generated or `max_tokens` is hit
>
> **This is called *autoregressive* generation** — each output becomes part of the input for the next step. And it's why the KV cache matters (section 4 in advanced topics) — you don't want to recompute all previous tokens on every step.
>
> **This is also why LLMs can't "go back and fix" an early mistake** — once "Paris" is in the sequence, the model conditions all future tokens on it. There's no editing, no drafting — just left-to-right generation.

```
User prompt (tokenise)
    ↓
Forward pass through transformer (autoregressive: one token at a time)
    ↓
Sample next token from distribution
    ↓
Append to sequence → repeat until stop token or max length
```

**Key inference parameters:**

> 📖 **Temperature — the creativity dial:**
> After the model generates its probability distribution ("Paris: 72%, Lyon: 4%..."), temperature controls how confident vs. random the final sampling is.
>
> Imagine you're asking a very confident expert a question. At **temperature 0**, they always give their top answer with 100% conviction — like a lookup table. At **temperature 2**, they become a creative brainstormer who entertains unlikely ideas. At **temperature 1**, they respond naturally.
>
> - Use **low temperature (0–0.3)** for: factual questions, code, data extraction — you want the most likely correct answer
> - Use **high temperature (0.7–1.2)** for: creative writing, brainstorming, generating diverse options
>
> **Top-p (nucleus sampling) — the vocabulary filter:**
> Instead of always sampling from all 100K possible next tokens, top-p says "only sample from the *smallest set of top tokens* whose combined probability adds up to p%." For top-p = 0.9: find the fewest tokens that together total 90% probability, then pick from those. This cuts out the very unlikely "tail" tokens that would feel random and incoherent.

| Parameter | Default | Effect |
|---|---|---|
| `temperature` | 1.0 | 0 = deterministic (argmax), >1 = more random |
| `top_p` (nucleus) | 1.0 | Only sample from top p% probability mass |
| `top_k` | 50 | Only sample from top k most likely tokens |
| `max_tokens` | varies | Hard limit on output length |
| `repetition_penalty` | 1.0 | >1 discourages repeating tokens |

```python
# Temperature effect
logits = [2.0, 1.0, 0.5]   # raw scores for tokens A, B, C

# temperature=1.0: normal softmax
probs = softmax(logits / 1.0)  # [0.55, 0.33, 0.12]

# temperature=0.1: near-greedy (very confident)
probs = softmax(logits / 0.1)  # [0.97, 0.03, 0.00]

# temperature=2.0: more uniform (creative/diverse)
probs = softmax(logits / 2.0)  # [0.42, 0.34, 0.24]
```

---

## Part 7 — Interview Q&A (30 Questions)

### Tokenization & Embeddings

**Q1: What is tokenization and why is it important?**
> Tokenization converts text to numerical IDs the model can process. The choice of tokenizer affects vocabulary size, context efficiency, and model behaviour. Subword tokenization (BPE, WordPiece) balances vocabulary size and handles out-of-vocabulary words.

**Q2: "The model gets 100 tokens per second throughput. How many words per minute is that?"**
> 100 tokens/sec × 60 sec = 6,000 tokens/min. At ~0.75 words/token → ~4,500 words/min. (Real throughput is usually reported in tokens/sec for this reason — it's model-agnostic.)

**Q3: Why can't you just use word-level tokenization?**
> - Vocabulary would need to include every word in the training corpus (100K–1M+)
> - New words (slang, brand names, code) would be unknown (OOV)
> - Morphological variants ("run", "running", "runs") would be separate tokens with no shared information

**Q4: What is embedding dimensionality and how is it chosen?**
> Larger dimensionality → more capacity to represent nuances, but more parameters and compute. Common values: 384 (MiniLM), 768 (BERT-base), 1024 (BERT-large), 4096 (LLaMA 7B). Chosen empirically — typically scales with model size.

### Architecture

**Q5: Explain attention in your own words.**
> Attention lets each token "look at" every other token and decide how much to "borrow" information from each. For "The bank by the river was steep", the word "bank" uses attention to look at "river" and update its representation to mean "riverbank" not "financial bank".

**Q6: Why is the attention score divided by √d_k?**
> For large d_k, dot products grow proportionally to d_k, pushing softmax into regions with tiny gradients. Dividing by √d_k keeps dot products at unit variance regardless of d_k.

**Q7: What is causal (masked) self-attention?**
> In decoder-only models (GPT), each token can only attend to itself and previous tokens, not future ones. This is enforced by setting attention scores for future positions to -∞ before softmax (masking them to ~0 weight).

**Q8: What is the difference between self-attention and cross-attention?**
> - Self-attention: Q, K, V all come from the same sequence (e.g. encoder processing its own tokens)
> - Cross-attention: Q comes from one sequence (decoder), K and V come from another (encoder output). Used in encoder-decoder models so the decoder can "attend to" the source.

**Q9: Why is the FFN dimension usually 4× the model dimension?**
> Empirical scaling — the original paper used 512 model dim and 2048 FFN dim. The 4× ratio has been kept largely because it works well. The FFN is thought to store factual memories, so more capacity helps. LLaMA uses a slightly different variant with SwiGLU that uses ~2.67× but with 3 weight matrices.

**Q10: What is a residual connection and why is it critical?**
> Adding the input directly to the sublayer output: `output = x + sublayer(x)`. This creates a "highway" for gradients to flow directly to earlier layers, solving the vanishing gradient problem in deep networks. Without residuals, training transformers with 24+ layers would be unstable.

**Q11: What is the role of LayerNorm?**
> Normalises activations to have zero mean and unit variance (across the feature dimension), then applies learned scale (γ) and shift (β). This stabilises training by preventing activation magnitudes from exploding or vanishing. In LLMs, RMSNorm (simpler — just scale, no mean) is often used instead.

**Q12: How many parameters does a transformer have?**
> For a model with `L` layers, `d_model` dimension, `d_ff` FFN dimension, vocabulary size `V`:
> - Embeddings: V × d_model
> - Per layer: 4 × d_model² (attention Q,K,V,O) + 2 × d_model × d_ff (FFN)
> - Total ≈ V × d_model + L × (4d² + 8d × d_ff/4 × 4)
>
> LLaMA 7B: 7B params. GPT-3: 175B params. Rough check: "7B params at fp16 = 14GB RAM minimum"

**Q13: Explain pre-norm vs post-norm. Which is better for large models?**
> Pre-norm (modern): `output = x + sublayer(LayerNorm(x))` — gradients flow through the residual path unobstructed; training is more stable.
> Post-norm (original paper): `output = LayerNorm(x + sublayer(x))` — harder to train but sometimes better final performance on small models.
> For large models (>1B params), pre-norm is standard because training stability is critical.

### BERT vs GPT

**Q14: Can you use GPT for classification?**
> Yes — add a classification head (linear layer) on top of the last token's hidden state. OpenAI's original GPT paper did this. However, BERT-style models generally perform better for classification tasks because bidirectional attention gives richer representations.

**Q15: Why is BERT not used for text generation?**
> BERT saw all tokens during training (bidirectional). To generate text, you need to predict tokens one at a time without "peeking" at future tokens. BERT's architecture doesn't support causal masking, so generated text is incoherent.

**Q16: What is RoBERTa and how does it improve on BERT?**
> RoBERTa (Liu et al., 2019): Same architecture as BERT but trained:
> - On 10× more data
> - Without NSP objective (showed NSP hurt performance)
> - With larger batch sizes and longer sequences
> - With dynamic masking (different masks each epoch)
> Result: significantly better performance on GLUE/SQuAD benchmarks.

### Scaling & Inference

**Q17: What are scaling laws?**
> Kaplan et al. (OpenAI, 2020) showed model performance (measured by loss) scales predictably as a power law with: number of parameters, training compute, and dataset size. Chinchilla (2022) refined this to show the optimal compute allocation is roughly 20 tokens per parameter.

**Q18: What is context length and what are its limitations?**
> Context length = max tokens in one forward pass. Limitations:
> 1. Attention is O(n²) in memory and time (quadratic with context length)
> 2. Models trained with shorter contexts may not generalise well to longer ones ("lost in the middle" problem)
> 3. KV cache (memory) grows linearly with context length during inference
> Solutions: FlashAttention (memory-efficient O(n²)), RoPE (better position generalisation), sliding window attention (Mistral).

**Q19: What is the KV cache and why does it matter for inference?**
> During autoregressive generation, each new token's attention needs K and V from all previous tokens. Without caching, this requires recomputing all previous K and V on every forward pass: O(n²) total work. The KV cache stores computed K, V tensors per layer, so each step is O(n) — only the new token's KV must be computed.

**Q20: What is temperature in language model sampling and how would you set it?**
> Temperature T divides logits before softmax: `probs = softmax(logits / T)`.
> - T < 1: More peaked distribution → repetitive but accurate (use for code, factual tasks)
> - T = 1: Default distribution
> - T > 1: Flatter distribution → more diverse/creative but less accurate (use for creative writing, brainstorming)
> For production systems: typically T=0.1–0.3 for factual, T=0.7–1.0 for creative.

**Q21: What is top-p (nucleus) sampling?**
> Instead of sampling from all tokens, only sample from the smallest set of tokens whose cumulative probability ≥ p. E.g. top-p=0.9: find the fewest tokens that together have 90% probability mass, then sample from just those. Dynamically adjusts vocabulary size — uses many tokens when distribution is flat (high uncertainty), few when peaked (high confidence).

**Q22: Why do LLMs hallucinate?**
> 1. Not trained to say "I don't know" — trained to always predict a plausible next token
> 2. Memorisation vs inference distinction — may have seen many similar sentences but not the exact fact
> 3. Autoregressive nature — once a wrong token is generated, the model conditions on it and continues the error
> 4. Training data: if 100 documents say X and 1 correct document says Y, the model may "vote" for X
> 5. Long-range context: can "forget" important context in very long documents

**Q23: What is few-shot learning (in-context learning)?**
> Providing examples in the prompt so the model learns the task format without weight updates:
> ```
> Q: What is 2+2? A: 4.
> Q: What is 3+3? A: 6.
> Q: What is 5+5? A:  ← model completes: 10
> ```
> Emerged strongly at GPT-3 scale (175B). Below ~10B params, few-shot learning is unreliable.

**Q24: What is chain-of-thought (CoT) prompting?**
> Include reasoning steps in the examples: "Let's think step by step". Model learns to produce intermediate reasoning before the answer. Significantly improves multi-step reasoning tasks. Key paper: Wei et al., 2022. CoT is most effective at 100B+ scale.

**Q25: What is instruction tuning?**
> Fine-tuning on (instruction, correct-response) pairs to make the model follow instructions rather than just complete text. Examples: FLAN (Google), InstructGPT (OpenAI), Alpaca (Stanford). Transform "base model" (completes text) → "instruction model" (follows commands).

**Q26: Explain the difference between parameters and activations.**
> - **Parameters (weights):** Learned numbers stored in the model, fixed after training (W_q, W_k, W_v etc.)
> - **Activations:** Intermediate computations during forward pass (attention scores, hidden states). Activations are recomputed every forward pass and proportional to batch_size × sequence_length.
> Memory: parameters = fixed, activations = varies with batch size. For inference: activations often dominate. For training: activations must be stored for backprop (gradient checkpointing trades compute for memory).

**Q27: What is the difference between fine-tuning and prompt engineering?**
> - **Prompt engineering:** Change the input (no model updates). Fast but limited — can't change model knowledge or capabilities.
> - **Fine-tuning:** Update model weights on new data. Teaches new knowledge/behaviour but requires data and compute.
> - **When to use each:** Prompt engineering first (cheap). Fine-tune if: consistent behaviour needed, tasks require specific format/style, model lacks domain knowledge.

**Q28: What is RLHF and why is it necessary?**
> Reinforcement Learning from Human Feedback. Pre-trained models predict likely text, not helpful/safe text. RLHF trains a reward model on human preferences, then uses PPO to optimise the LLM to generate text that maximises that reward. Result: helpful, harmless, honest assistants.

**Q29: What's the difference between a 7B and a 70B model for your specific use case?**
> Factors: task complexity (7B is great for factual QA, 70B needed for multi-step reasoning), latency requirements (7B ~5× faster), cost (7B ~10× cheaper to serve), quality delta (70B generally better but gap narrowing). For FAANG interview: "Start with smaller model, measure quality gap, scale up only if needed."

**Q30: You're asked to reduce LLM API costs by 50%. What are your options?**
> 1. **Prompt compression** — remove redundant context tokens (LLMLingua)
> 2. **Smaller model** — GPT-4o-mini vs GPT-4o
> 3. **Caching** — cache responses for identical/similar queries (exact match or semantic cache)
> 4. **Batching** — batch requests (higher throughput, lower cost/token)
> 5. **Self-hosted** — run open-source model (LLaMA 3 8B) if volume justifies GPU cost
> 6. **Token-level control** — reduce max_tokens, reduce few-shot examples, use shorter system prompts

---

## 📚 Further Resources

### Essential This Week
| Resource | Link | Time |
|---|---|---|
| **Let's Build GPT from Scratch** (Karpathy) | https://youtu.be/kCc8FmEb1nY | 2 hrs |
| **How Transformer LLMs Work** (DeepLearning.AI) | https://learn.deeplearning.ai/courses/how-transformer-llms-work | 2 hrs |
| **The Illustrated Transformer** (Jay Alammar) | http://jalammar.github.io/illustrated-transformer/ | 45 min |

### Go Deeper
| Resource | What It Covers |
|---|---|
| **Attention is All You Need** (original paper) | https://arxiv.org/abs/1706.03762 |
| **BERT paper** | https://arxiv.org/abs/1810.04805 |
| **Chinchilla scaling laws** | https://arxiv.org/abs/2203.15556 |
| **Lilian Weng — Attention? Attention!** | https://lilianweng.github.io/posts/2018-06-24-attention/ |

### Books
- **"Build a Large Language Model (From Scratch)"** by Sebastian Raschka — best practical book to code a GPT from scratch
- **"Hands-On Large Language Models"** by Jay Alammar & Maarten Grootendorst — visual, practical, 2024

> ✅ **End of Week 3 Core Content.** The sections below add depth for day-to-day work and advanced interview prep.

---

## Part 8 — Day-to-Day Work Applications: Using LLMs as an AI Engineer

> This section bridges theory → daily engineering work. Every concept above maps to a practical task you'll do regularly.

### 8.1 Model Selection for Production Tasks

```
Decision Framework: Which model for which task?

┌──────────────────────────────────────────────────────────────┐
│ Task                        │ Recommended Model              │
├──────────────────────────────────────────────────────────────┤
│ Internal chatbot (low cost) │ Mistral 7B / LLaMA 3 8B       │
│ Customer-facing chatbot     │ GPT-4o / Claude 3.5 Sonnet     │
│ Code generation             │ GPT-4o / Claude 3.5 Sonnet     │
│ Text classification         │ Fine-tuned BERT / DistilBERT   │
│ Embeddings for RAG          │ text-embedding-3-small (OpenAI) │
│                             │ or BGE-large / E5-large-v2     │
│ Summarisation (bulk)        │ GPT-4o-mini / Claude 3 Haiku   │
│ JSON extraction             │ GPT-4o-mini (structured output)│
│ On-prem / air-gapped        │ LLaMA 3 / Mistral via Ollama   │
│ Multilingual                │ GPT-4o / Cohere Command-R+     │
└──────────────────────────────────────────────────────────────┘
```

### 8.2 LLM API Usage Patterns (You'll Write This Daily)

```python
# Pattern 1: Simple completion with retry logic
import openai
import time
from tenacity import retry, stop_after_attempt, wait_exponential

client = openai.OpenAI()  # reads OPENAI_API_KEY from env

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def call_llm(messages, model="gpt-4o-mini", temperature=0, max_tokens=1000):
    """Production-grade LLM call with retry and error handling."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

# Pattern 2: Structured output extraction (JSON mode)
def extract_product_info(text: str) -> dict:
    """Extract structured data from unstructured text — common retail/CPG task."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": """Extract product information as JSON:
            {"product_name": str, "brand": str, "category": str, "price": float, "sentiment": "positive"|"negative"|"neutral"}"""},
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    import json
    return json.loads(response.choices[0].message.content)

# Pattern 3: Batch processing with cost awareness
def process_batch_with_budget(texts, max_budget_usd=5.0):
    """Process texts while tracking costs. Essential for production."""
    COST_PER_1K_INPUT = 0.00015   # gpt-4o-mini input
    COST_PER_1K_OUTPUT = 0.0006   # gpt-4o-mini output
    
    total_cost = 0.0
    results = []
    
    for text in texts:
        est_input_tokens = len(text) / 4  # rough estimate
        est_cost = (est_input_tokens / 1000) * COST_PER_1K_INPUT + 0.5 * COST_PER_1K_OUTPUT
        
        if total_cost + est_cost > max_budget_usd:
            print(f"Budget limit reached at {len(results)}/{len(texts)} items")
            break
        
        result = call_llm([{"role": "user", "content": f"Summarize: {text}"}])
        results.append(result)
        total_cost += est_cost
    
    print(f"Total estimated cost: ${total_cost:.4f}")
    return results
```

### 8.3 Prompt Engineering for Daily Work

```python
# The CRAFT framework for production prompts:
# C - Context: Set the role and background
# R - Role: Define expertise
# A - Action: Specify the exact task
# F - Format: Define output structure
# T - Tone: Set communication style

# Example: Building a data quality checker for retail data
SYSTEM_PROMPT = """You are a senior data engineer at a retail analytics company.

CONTEXT: You are reviewing incoming product data feeds from suppliers.

TASK: Analyse the data row provided and identify quality issues.

RULES:
- Flag missing required fields (product_name, ean, price)
- Flag price outliers (>£1000 or <£0.01 for grocery)
- Flag suspicious patterns (same price for all items, future dates)
- Be specific about which field has the issue

OUTPUT FORMAT (JSON):
{
  "status": "pass" | "fail",
  "issues": [{"field": str, "issue": str, "severity": "critical"|"warning"}],
  "confidence": float  // 0-1
}"""

# Few-shot prompting — give examples for consistent output
FEW_SHOT_EXAMPLES = """
Example 1:
Input: {"product_name": "Heinz Beans 400g", "ean": "5000157024674", "price": 0.85, "category": "canned_goods"}
Output: {"status": "pass", "issues": [], "confidence": 0.95}

Example 2:
Input: {"product_name": "", "ean": "5000157024674", "price": -1.50, "category": "canned_goods"}
Output: {"status": "fail", "issues": [{"field": "product_name", "issue": "Empty required field", "severity": "critical"}, {"field": "price", "issue": "Negative price", "severity": "critical"}], "confidence": 0.99}
"""

# Chain-of-thought prompting — for complex reasoning tasks
COT_PROMPT = """Analyse this customer shopping basket for cross-sell opportunities.

Think step by step:
1. Identify the product categories present
2. Identify what's MISSING that typically co-occurs with these categories
3. Rank recommendations by historical attachment rate
4. Provide your top 3 recommendations with reasoning

Basket: {basket_items}"""
```

### 8.4 Cost Estimation for LLM Projects (Dunnhumby Context)

```
Real-world cost planning:

Scenario: Customer support chatbot processing 50K queries/day

Inputs:
  Average query: ~200 tokens input + 300 tokens output
  Model: GPT-4o-mini ($0.15/1M input, $0.60/1M output)

Daily cost:
  Input:  50,000 × 200 / 1,000,000 × $0.15 = $1.50/day
  Output: 50,000 × 300 / 1,000,000 × $0.60 = $9.00/day
  Total: $10.50/day = $315/month

With RAG context (adds ~1000 tokens per query):
  Input:  50,000 × 1,200 / 1,000,000 × $0.15 = $9.00/day
  Total: $18.00/day = $540/month

With GPT-4o instead:
  Input:  50,000 × 1,200 / 1,000,000 × $2.50 = $150/day
  Output: 50,000 × 300 / 1,000,000 × $10.00 = $150/day
  Total: $300/day = $9,000/month ← 17× more expensive!

LESSON: Model selection is the BIGGEST cost lever.
  - Start with cheapest model that meets quality bar
  - Use GPT-4o only for complex reasoning, fallback from smaller model
  - Cache identical/similar queries (Redis → 40-60% cost reduction)
  - Batch non-urgent requests (off-peak pricing, async processing)
```

### 8.5 Running Local Models (Ollama — For Development & Testing)

```bash
# Install Ollama (works on Mac, Linux, Windows)
# https://ollama.ai

# Pull and run a model locally — zero API cost
ollama pull llama3:8b
ollama run llama3:8b "Explain gradient descent in 3 sentences"

# Serve as OpenAI-compatible API (use same code as above!)
# Ollama automatically starts server at localhost:11434
```

```python
# Use Ollama with OpenAI client — same code, local model
from openai import OpenAI

local_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # not used, but required
)

response = local_client.chat.completions.create(
    model="llama3:8b",
    messages=[{"role": "user", "content": "Write a Python function to deduplicate a list"}],
    temperature=0
)
print(response.choices[0].message.content)

# Benefits for day-to-day:
# 1. No API costs during development/testing
# 2. No data leaves your machine (privacy)
# 3. No rate limits
# 4. Works offline (flights, restricted networks)
```

### 8.6 Embedding Models for Semantic Search at Work

```python
# You'll use embeddings constantly for:
# - RAG document search
# - Duplicate detection
# - Semantic clustering of support tickets / product descriptions
# - Finding similar code, documents, or customer queries

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')  # Local, free, fast

# Real work example: finding similar product descriptions
products = [
    "Heinz Tomato Ketchup 460g",
    "Heinz Tomato Sauce 460g",
    "Fairy Original Washing Up Liquid 900ml",
    "HP Brown Sauce 450g",
    "Hellmann's Real Mayonnaise 400g"
]

embeddings = model.encode(products)

# Find most similar to "tomato ketchup"
query = model.encode(["ketchup sauce"])
similarities = np.dot(embeddings, query.T).flatten()
ranked = sorted(zip(products, similarities), key=lambda x: -x[1])
# Output: [("Heinz Tomato Ketchup 460g", 0.82), ("Heinz Tomato Sauce 460g", 0.79), ...]
```

### 8.7 Evaluating LLM Outputs (Production Quality Checks)

```python
# You'll need to evaluate LLM outputs for quality, safety, and correctness

# Method 1: LLM-as-Judge (GPT-4o evaluates GPT-4o-mini outputs)
def evaluate_response(query, response, criteria="helpfulness, accuracy, safety"):
    eval_prompt = f"""Rate this AI response on a scale of 1-5 for: {criteria}

    User query: {query}
    AI response: {response}
    
    Provide JSON: {{"helpfulness": int, "accuracy": int, "safety": int, "reasoning": str}}"""
    
    return call_llm([
        {"role": "system", "content": "You are an expert AI output evaluator."},
        {"role": "user", "content": eval_prompt}
    ], model="gpt-4o")

# Method 2: Automated metrics
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
score = scorer.score(
    "The cat sat on the mat",  # reference
    "A cat is sitting on a mat"  # generated
)
# ROUGE-L: 0.67 (measures overlap — useful for summarisation)

# Method 3: Embedding similarity (semantic correctness)
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
ref_emb = model.encode("Paris is the capital of France")
gen_emb = model.encode("The capital city of France is Paris")
similarity = util.cos_sim(ref_emb, gen_emb)  # ~0.95 (semantically identical)
```

---

## Part 9 — Advanced Topics for Deeper Interviews

### 9.1 Tokenizer Pitfalls in Production

```python
# Issue 1: Token counting for context window management
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Always count tokens before sending to API — avoid context overflow."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def truncate_to_fit(text: str, max_tokens: int = 3000, model: str = "gpt-4o") -> str:
    """Truncate text to fit within token budget."""
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])

# Issue 2: Different models use different tokenizers!
# GPT-4: cl100k_base (100K vocab)
# LLaMA 3: custom BPE (128K vocab)
# Same text = different token count depending on model
# "Hello world" = 2 tokens on GPT-4, could be 2-3 on LLaMA

# Issue 3: Multilingual tokenization is expensive
# English: ~1 token per word
# Chinese/Japanese: ~2-3 tokens per character
# Hindi/Arabic: ~3-4 tokens per word
# Cost implication: non-English queries cost 2-4× more tokens!
```

### 9.2 Context Window Strategies

```python
# Strategy 1: Sliding window with summarisation
def process_long_document(document: str, question: str, chunk_size: int = 3000):
    """Process documents longer than context window."""
    chunks = split_into_chunks(document, chunk_size)
    
    # Map: process each chunk
    chunk_answers = []
    for chunk in chunks:
        answer = call_llm([
            {"role": "system", "content": "Answer based only on this excerpt."},
            {"role": "user", "content": f"Excerpt: {chunk}\n\nQuestion: {question}"}
        ])
        chunk_answers.append(answer)
    
    # Reduce: combine chunk answers
    combined = "\n".join(chunk_answers)
    final = call_llm([
        {"role": "system", "content": "Synthesize these partial answers into one coherent answer."},
        {"role": "user", "content": f"Partial answers:\n{combined}\n\nOriginal question: {question}"}
    ])
    return final

# Strategy 2: Hierarchy — summarize first, detail on demand
# Step 1: Summarize each section → fits in context
# Step 2: If question needs detail, retrieve specific section
# This is essentially what RAG does!
```

### 9.3 Structured Outputs (Production Must-Have)

```python
# Modern APIs support guaranteed JSON schema outputs

from pydantic import BaseModel
from openai import OpenAI

class ProductReview(BaseModel):
    sentiment: str  # "positive", "negative", "neutral"
    key_themes: list[str]
    quality_score: float  # 1-10
    would_recommend: bool

client = OpenAI()

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Analyse the product review."},
        {"role": "user", "content": "Great ketchup, thick consistency, love the glass bottle. A bit pricey though."}
    ],
    response_format=ProductReview,
)

review = response.choices[0].message.parsed
print(review.sentiment)      # "positive"
print(review.key_themes)     # ["consistency", "packaging", "price"]
print(review.quality_score)  # 8.0
```

### 9.4 LLM Safety & Guardrails for Production

```python
# You MUST implement guardrails in production — day 1 requirement

# Guardrail 1: Input validation
def validate_input(user_message: str) -> bool:
    """Block prompt injection and harmful inputs."""
    injection_patterns = [
        "ignore previous instructions",
        "you are now",
        "system prompt",
        "jailbreak",
    ]
    lower = user_message.lower()
    return not any(pattern in lower for pattern in injection_patterns)

# Guardrail 2: Output validation
def validate_output(response: str, forbidden_topics: list) -> str:
    """Check LLM output before returning to user."""
    for topic in forbidden_topics:
        if topic.lower() in response.lower():
            return "I'm sorry, I can't discuss that topic. Please rephrase your question."
    return response

# Guardrail 3: PII detection (critical for retail/customer data)
import re

def mask_pii(text: str) -> str:
    """Mask personally identifiable information before sending to LLM."""
    # Email
    text = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL]', text)
    # Phone (UK format)
    text = re.sub(r'\b0\d{10}\b', '[PHONE]', text)
    # Credit card
    text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD]', text)
    # Postcode (UK)
    text = re.sub(r'\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b', '[POSTCODE]', text, flags=re.I)
    return text

# Always mask PII before sending to external LLM APIs!
masked = mask_pii("Contact john@email.com or call 07912345678")
# "Contact [EMAIL] or call [PHONE]"
```

---

## Part 10 — Further Resources

| Resource | URL |
|---|---|
| **Attention is All You Need** (original paper) | https://arxiv.org/abs/1706.03762 |
| **BERT paper** | https://arxiv.org/abs/1810.04805 |
| **Chinchilla scaling laws** | https://arxiv.org/abs/2203.15556 |
| **Lilian Weng — Attention? Attention!** | https://lilianweng.github.io/posts/2018-06-24-attention/ |

### Books
- **"Build a Large Language Model (From Scratch)"** by Sebastian Raschka — best practical book to code a GPT from scratch
- **"Hands-On Large Language Models"** by Jay Alammar & Maarten Grootendorst — visual, practical, 2024

> ✅ **End of Week 3.** Next week we go deeper: KV cache, FlashAttention, RoPE, GQA, and the full attention math you'll need to answer "describe the exact computation" interviews.
