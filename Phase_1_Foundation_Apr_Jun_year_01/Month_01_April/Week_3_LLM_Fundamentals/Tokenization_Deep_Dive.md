# Tokenization — Deep Dive Guide
### Phase 1 Supplementary | Week 3–4 Reference | Critical for Interviews

> Tokenization is tested in nearly every LLM engineer interview. Understanding it deeply separates candidates who just use LLMs from those who understand them.

---

## Part 1 — What Is Tokenization?

> 💡 **ELI5 (Explain Like I'm 5):**
> Imagine you want to feed a whole book to a computer, but the computer only understands numbers. You could give every single letter a number (a=1, b=2), but reading letter-by-letter is too slow. You could give every whole word a number ("apple" = 500), but there are too many unique words in the world! Tokenization is the Goldilocks solution: we break text into chunks (often common syllables or word parts) and give *those* chunks numbers. "Unbelievable" might become ["Un", "believ", "able"].

Tokenization converts raw text into the discrete units (tokens) that an LLM processes. It sits at the boundary between human language and the model's mathematical world.

```
Raw text:  "Hello, I love LLMs!"
           ↓ tokenizer
Tokens:    ["Hello", ",", " I", " love", " LL", "Ms", "!"]
Token IDs: [15496, 11, 314, 1842, 22178, 58, 0]
           ↓ embedding lookup
Vectors:   [[0.12, -0.34, ...], [0.89, 0.10, ...], ...]   ← input to transformer
```

**Why does tokenization matter for engineers?**
```
1. Cost: Pricing is per token. "GPT-4o at $5/1M input tokens" — your cost depends
   on how many tokens your text produces.

2. Context window: 128K tokens ≠ 128K words. English text is ~1.3 tokens/word on avg.
   Code is denser (~2-3 tokens/word). Chinese is ~2 tokens/character.

3. Performance: Models sometimes struggle with individual characters because
   they've never seen them as single tokens in training.

4. Rare words: Medical/legal/financial terms may be split into many subword tokens,
   making them harder to process accurately.
```

---

## Part 2 — The Three Major Tokenization Algorithms

### 2.1 Byte Pair Encoding (BPE)

**Used by:** GPT-2, GPT-3, GPT-4 (all OpenAI), LLaMA, Mistral, Falcon

**Algorithm:**
```python
# BPE Training (simplified):

# 1. Start: split all words into individual characters + special </w> (end-of-word)
# Vocabulary: {"h", "e", "l", "o", " ", "w", "r", "d", "</w>"}

# 2. Count all adjacent pair frequencies in training corpus:
# "hello world" → ("h","e"):1, ("e","l"):2, ("l","l"):1, ("l","o"):1, etc.

# 3. Merge the most frequent pair into a new token:
# Merge ("e","l") → "el"
# New vocab: {"h", "el", "l", "o", " ", "w", "r", "d", "</w>", "el"}

# 4. Repeat steps 2-3 for N merge operations (e.g., N=50,000 for GPT-2)

# Result after many merges:
# Common sequences become single tokens:
# "ing" → single token (very frequent in English)
# "tion" → single token
# " the" → single token (space+word is common pattern)

def get_vocab_bpe(text: str, num_merges: int) -> dict:
    """Simplified BPE training."""
    # Split into characters with end-of-word marker
    vocab = {}
    for word in text.split():
        word_chars = " ".join(list(word)) + " </w>"
        vocab[word_chars] = vocab.get(word_chars, 0) + 1
    
    def get_pairs(vocab):
        pairs = {}
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pair = (symbols[i], symbols[i+1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs
    
    for _ in range(num_merges):
        pairs = get_pairs(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)  # most frequent pair
        
        # Merge best pair in all vocab words
        new_vocab = {}
        bigram = " ".join(best_pair)
        replacement = "".join(best_pair)
        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]
        vocab = new_vocab
    
    return vocab
```

**Key properties — and why they matter:**
- **Frequency-driven**: Common subwords like "ing", "tion", and " the" get merged into single tokens early. This means frequently occurring word endings and prefixes are efficiently represented with a single token ID, while rare words are gracefully decomposed into components the model has already seen during training. The model never encounters a completely unknown token.
- **Greedy**: Merges are performed in order of frequency, with no backtracking. This makes BPE fast to train and fully deterministic, but the resulting segmentation isn't guaranteed to be linguistically optimal. The word "misunderstanding" might be split as `["mis", "under", "stand", "ing"]` or `["misunder", "standing"]` depending entirely on which substrings happened to be most common in the training corpus.
- **Language-agnostic at the byte level**: Byte-level BPE (used by GPT-4) starts from the 256 possible byte values, meaning it can represent any Unicode character without ever producing an `[UNK]` token. All text in any language, script, or encoding — including emoji — is always representable.

**GPT-4 tokenizer (tiktoken):**
```python
import tiktoken

# GPT-4o uses "cl100k_base" tokenizer (100,277 tokens)
enc = tiktoken.get_encoding("cl100k_base")

# Tokenize examples:
text1 = "Hello, world!"
tokens1 = enc.encode(text1)
print(tokens1)          # [9906, 11, 1917, 0]
print(len(tokens1))     # 4 tokens

# LLM-related terms:
text2 = "Tokenization is fundamental to LLMs"
print(enc.encode(text2))
# [5161, 2065, 374, 16188, 311, 445, 11237, 82]  → 8 tokens

# Code tends to more tokens:
code = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1)+fibonacci(n-2)"
print(f"Code: {len(enc.encode(code))} tokens")  # ~22 tokens for this line

# Roughly: 1 token ≈ 4 characters ≈ 0.75 words in English
print(f"Token ratio: {len(text2.split())} words, {len(enc.encode(text2))} tokens")
```

---

### 2.2 WordPiece

**Used by:** BERT, DistilBERT, ALBERT, all Google BERT-family models

WordPiece is BPE's close cousin, but with a smarter merging criterion. While BPE greedily picks the most *frequently occurring* adjacent pair, WordPiece picks the pair whose merger produces the greatest improvement in language model likelihood over the training corpus. Formally, it maximises $P(AB) \;/\; (P(A) \times P(B))$ — a mutual information-style score that asks: *"how much more likely does this combined token appear than you'd expect if A and B were independent?"* This preference for statistically surprising co-occurrences tends to produce subwords that are linguistically meaningful (common morphemes like "-ing", "-tion", "un-") rather than just high-frequency character sequences.

Think of it this way: in a corpus with many occurrences of both "un" and "lock", BPE would merge them simply because the pair `(un, lock)` is frequent. WordPiece first asks: is "unlock" appearing *more* often than chance would predict given how common "un" and "lock" each are independently? Only if the answer is yes does it commit to the merge.

**Difference from BPE:**
- BPE objective: $\max(\text{count}(AB))$ — raw frequency
- WordPiece objective: $\max\!\left(P(AB) \;/\; (P(A) \times P(B))\right)$ — likelihood / mutual information

**WordPiece markers:**
```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# WordPiece uses "##" prefix for continuation subwords
text = "tokenization is fundamental"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['token', '##ization', 'is', 'fundamental']
#            ^ "##" means this is a continuation of the previous token
#              (i.e., "ization" continues "token")

# vs BPE (GPT-style): uses leading space to signal word boundary
# ["token", "ization", " is", " fundamental"]
# ^ space is part of the token (encodes word boundary)

# Encode to IDs:
ids = tokenizer.encode(text)
print(ids)  # [101, 19204, 3989, 2003, 8918, 102]
# 101 = [CLS], 102 = [SEP] — special tokens added by BERT
```

**Key special tokens in BERT:**
```
[CLS] (ID 101): Classification token — placed at beginning of every sequence
                Its embedding is used for classification tasks
[SEP] (ID 102): Separator — marks end of sentence or boundary between two sentences
[MASK] (ID 103): Used during masked language model (MLM) pretraining
[PAD] (ID 0):   Padding to make same length in batches
[UNK] (ID 100): Unknown token for characters not in vocabulary
```

---

### 2.3 SentencePiece

**Used by:** LLaMA (v1 & v2), T5, ALBERT, mT5, PaLM, Gemma

Here is the problem SentencePiece was designed to solve: both BPE and WordPiece assume you can pre-split text on whitespace. This works fine for English, but fails for languages like Japanese, Chinese, Thai, or Arabic, which don't use spaces as word boundaries in the same way. If a tokeniser's very first step is "split on spaces", it's already making a language-specific assumption that excludes a large fraction of the world's written text.

SentencePiece sidesteps this entirely by treating the *whole raw input* — spaces and all — as a single stream of characters. It learns subwords directly from the byte stream with no pre-splitting step. Spaces are treated as just another character, represented as "\u2581" (U+2581) when they mark a word boundary. The result is a tokeniser that behaves identically across all languages, making it the preferred choice for multilingual models:

- `"Hello world"` → `["\u2581Hello", "\u2581world"]`
- `"Helloworld"` → `["\u2581Hello", "world"]` — the \u2581 only appears at a genuine word start, so the model can always recover word boundaries from the token sequence

**SentencePiece in practice:**
```python
import sentencepiece as spm
from transformers import LlamaTokenizer

# LLaMA uses SentencePiece
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

text = "The LLaMA model uses SentencePiece tokenization"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['▁The', '▁LL', 'AMA', '▁model', '▁uses', '▁Sentence', 'Piece', '▁token', 'ization']

ids = tokenizer.encode(text)
print(ids)
print(f"Token count: {len(ids)}")

# Vocabulary size:
print(f"LLaMA vocab: {tokenizer.vocab_size}")  # 32,000

# BOS/EOS tokens:
print(f"BOS token: {tokenizer.bos_token}")  # <s> — beginning of sequence
print(f"EOS token: {tokenizer.eos_token}")  # </s> — end of sequence
```

---

## Part 3 — Vocabulary Size and Its Impact

### 3.1 Vocabulary Size Comparison

| Model | Tokenizer | Vocab Size | Notes |
|---|---|---|---|
| BERT-base | WordPiece | 30,522 | English-focused |
| GPT-2 | BPE | 50,257 | Byte-level BPE |
| GPT-3/4 | BPE (tiktoken) | 100,277 | cl100k_base |
| LLaMA 1/2 | SentencePiece | 32,000 | Multilingual |
| LLaMA 3 | BPE (tiktoken-based) | 128,256 | Extended for code + multilingual |
| Mistral 7B | SentencePiece | 32,000 | Same as LLaMA 2 |
| Qwen 2.5 | BPE | 152,064 | Optimised for Chinese+English |
| Gemma | SentencePiece | 256,000 | Very large vocab |

### 3.2 The Vocab Size Trade-off

Vocabulary size is a core architectural decision that shapes memory usage, inference speed, and model quality in ways that aren't always obvious. It's worth understanding this trade-off deeply both for selecting base models and for explaining design choices under interview conditions.

A **smaller vocabulary** (e.g., 32K tokens like LLaMA 2) keeps the embedding matrix compact and lookup fast. The downside is that rare or technical words get split into many subword tokens — "tokenization" might become 3–4 tokens, and a medical term like "antidisestablishmentarianism" could be 8 or more. More tokens per word means longer sequences consuming more of the context window and increasing inference cost proportionally.

A **larger vocabulary** (e.g., 100K–256K like GPT-4 or Gemma) dedicates tokens to common technical terms, code constructs, and characters from many scripts. The same text produces *fewer* tokens, letting you fit more meaningful content into the context window for the same cost. The trade-off is a bigger embedding table and the fact that rare tokens appear less frequently during training, potentially resulting in weaker representations for those tokens.

**Rule of thumb for model selection:** Larger vocabulary models are generally better for code generation, multilingual tasks, and domains with dense technical vocabulary. Smaller vocabulary models are lighter to serve and simpler to fine-tune on constrained hardware.

---

> 🃏 **Quick-Recall Card — Tokenization**
> | Concept | One-liner |
> |---|---|
> | BPE (Byte Pair Encoding) | Merge most frequent adjacent byte/character pairs iteratively. Used by GPT-2/3/4, LLaMA. |
> | WordPiece | Like BPE but merges pairs that maximise language model likelihood, not raw frequency. Used by BERT. |
> | SentencePiece | Language-agnostic, treats raw text as bytes. Good for multilingual. Used by LLaMA 1/2, T5. |
> | Byte-level BPE | Starts from 256 raw bytes → never produces `[UNK]`. Used by GPT-4 (tiktoken). |
> | Vocab size trade-off | Small vocab (32K) → more tokens per word, longer sequences. Large vocab (128K+) → fewer tokens, bigger embedding table. |
> | ~1.3 tokens/word | Rule of thumb for English. Code is ~2–3 tokens/word. Chinese ≈ 2 tokens/char. |
> | Special tokens | `[BOS]`, `[EOS]`, `[PAD]`, `[MASK]` — always occupy fixed IDs in the vocabulary |
> | Temperature ≠ tokenization | Token probability at sampling uses softmax temperature. Separate from tokenization. |
>
> **Why this matters in interviews:** "How do you estimate API cost for your pipeline?" → Count tokens, not words. "Why does this SQL query use 3× more tokens?" → Code tokenizes inefficiently.

## Part 4 — Byte-Level BPE (GPT-4 Approach)

Every tokeniser built on characters faces an uncomfortable edge case: what happens when a character simply isn't in the vocabulary? Standard tokenisers fall back to `[UNK]`, which throws away all information about that part of the input. For a production LLM that handles user-generated content from hundreds of countries in dozens of scripts, hitting `[UNK]` constantly is a real quality problem.

GPT-4 solves this permanently with **byte-level BPE**. Instead of starting with a character vocabulary, it starts with the 256 possible byte values (0–255). Since every string in any language or encoding is ultimately a sequence of bytes, there is no concept of an unknown token. A Japanese kanji, a Russian letter, an obscure mathematical symbol, an emoji — all are representable as one or more byte tokens. The model has always seen those individual byte values during training, so it can handle any input without degradation.

```python
# Byte-level BPE: any character → UTF-8 bytes → BPE merges
# This means no "unknown" token — every string can be encoded

# Regular BPE might fail on:
# "This emoji 🚀 is cool" → if 🚀 not in vocab → [UNK]

# Byte-level BPE handles it:
# 🚀 = bytes [0xF0, 0x9F, 0x9A, 0x80]
# These bytes can always be represented, merged if frequent together

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

# Emoji example
emoji_text = "AI is 🚀🔥 amazing!"
tokens = enc.encode(emoji_text)
decoded = [enc.decode([t]) for t in tokens]
print(decoded)
# ['AI', ' is', ' 🚀', '🔥', ' amazing', '!']
# No UNK! Each emoji gets its own token (or bytes if really rare)
```

---

## Part 5 — Context Window and Token Counting

### 5.1 Tokens vs Words vs Characters

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> dict:
    tokens = enc.encode(text)
    return {
        "characters": len(text),
        "words": len(text.split()),
        "tokens": len(tokens),
        "ratio_tokens_per_word": round(len(tokens) / len(text.split()), 2)
    }

# Different text types:
examples = {
    "English prose": "The quick brown fox jumps over the lazy dog",
    "Technical code": "def fibonacci(n): return n if n<=1 else fibonacci(n-1)+fibonacci(n-2)",
    "JSON": '{"name": "Alice", "age": 30, "city": "London"}',
    "Chinese": "人工智能是计算机科学的一个重要分支",
}

for name, text in examples.items():
    stats = count_tokens(text)
    print(f"{name}: {stats}")
    
# Results (approximate):
# English prose:  9 words → 10 tokens  (1.1 tokens/word)
# Technical code: 14 "words" → 20 tokens (1.4 tokens/word)
# JSON:           varies → ~20 tokens (punctuation = extra tokens)
# Chinese:        10 chars → 13 tokens (1.3 tokens/char)
```

### 5.2 Estimating Cost

```python
def estimate_cost(
    prompt: str,
    expected_completion_words: int = 100,
    model: str = "gpt-4o"
) -> dict:
    """Estimate API cost for a single request."""
    enc = tiktoken.get_encoding("cl100k_base")
    
    input_tokens = len(enc.encode(prompt))
    output_tokens = int(expected_completion_words * 1.3)  # ~1.3 tokens/word
    
    # 2026 pricing (approximate, check current API pricing page)
    pricing = {
        "gpt-4o":           {"input": 5.0,  "output": 15.0},  # per 1M tokens
        "gpt-4o-mini":      {"input": 0.15, "output": 0.6},
        "claude-3-5-sonnet":{"input": 3.0,  "output": 15.0},
        "claude-3-haiku":   {"input": 0.25, "output": 1.25},
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    }
    
    if model not in pricing:
        return {"error": "Unknown model"}
    
    p = pricing[model]
    input_cost = (input_tokens / 1_000_000) * p["input"]
    output_cost = (output_tokens / 1_000_000) * p["output"]
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
        "cost_at_10k_requests": round((input_cost + output_cost) * 10000, 2)
    }
```

---

## Part 6 — Special Tokens and Chat Templates

### 6.1 Chat Templates (Critical for Fine-tuning)

Different models use different special tokens to separate roles in a conversation:

```python
# ChatML format (OpenAI, LLaMA 3, Qwen)
chatML_example = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is RAG?<|im_end|>
<|im_start|>assistant
RAG stands for Retrieval-Augmented Generation...<|im_end|>"""

# LLaMA 2 format
llama2_example = """<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

What is RAG? [/INST] RAG stands for... </s>"""

# LLaMA 3 format
llama3_example = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is RAG?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

RAG stands for...<|eot_id|>"""

# Mistral format (no system prompt support natively)
mistral_example = """<s>[INST] What is RAG? [/INST] RAG stands for... </s>"""

# Apply chat template programmatically:
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is RAG?"},
]

# apply_chat_template handles the special tokens automatically:
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,  # return string first to inspect
    add_generation_prompt=True  # add the assistant header to prompt generation
)
print(formatted)
```

---

## Part 7 — Tokenization Failure Modes

### 7.1 Arithmetic and Non-Latin Scripts

```python
# Problem 1: Arithmetic reasoning
# Numbers are split into ambiguous subwords:
# "2023" → ["20", "23"] — model processes them as two separate tokens
# "2047" → ["20", "47"] — looks same as "2023" to many model layers
# This is one reason LLMs struggle with arithmetic!

# Problem 2: Whitespace sensitivity
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode("cat"))     # [9508]
print(enc.encode(" cat"))    # [8996] ← different token! (leading space)
print(enc.encode("  cat"))   # [257, 8996] ← two tokens (extra space)

# Problem 3: Tokenisation differs for same phonetic content
print(enc.encode("SolidGoldMagikarp"))   # known GPT "glitch token"
# Some rare token sequences cause unexpected model behaviour

# Problem 4: Long rare words
print(enc.encode("antidisestablishmentarianism"))
# ['ant', 'id', 'is', 'est', 'abl', 'ishment', 'arian', 'ism']  ~ 8 tokens
# compared to common word "cat" = 1 token
```

### 7.2 Cross-lingual Tokenisation Disparity

```python
# English is over-represented in training → better tokenisation

# "Hello" → 1 token (English)
# "Hola"  → 2 tokens (Spanish)
# "Привет" → 4 tokens (Russian)
# "안녕"   → 6 tokens (Korean)

# Practical implication: same semantic content costs 3-6x more tokens
# for non-English languages → higher cost, less content fits in context

# This is why Qwen (Chinese-focused) uses 152K vocab —
# better Chinese tokenisation means fewer tokens per Chinese character
```

---

## Part 8 — Interview Q&A

**Q1: What is the difference between BPE and WordPiece tokenization?**

> Both are subword tokenization algorithms that build a vocabulary of subword units. BPE learns merge rules by greedily merging the most frequent adjacent pair of symbols, driven by raw count. WordPiece also learns merges but uses a likelihood-based objective: it merges pairs that maximise the language model's probability over the training corpus — specifically, it merges pairs where $P(AB) / (P(A) \times P(B))$ is maximised. In practice, WordPiece tends to produce more linguistically coherent subwords. BPE is used in GPT-family models; WordPiece is used in BERT-family models.

**Q2: Why does the choice of tokenizer matter when fine-tuning a model?**

> The model's weights encode patterns for its specific tokenizer's vocabulary. If you fine-tune on data prepared with a different tokenizer than the one the model was trained on, token IDs won't match the learned embeddings — the fine-tuning will fail silently or produce garbage. This also means you can't swap tokenizers without retraining from scratch. For production, ensure your fine-tuning pipeline, inference serving, and any pre/post-processing all use the exact same tokenizer version.

**Q3: How does tokenisation affect RAG pipeline cost and context window planning?**

> Long documents must be split into chunks that fit within the LLM's context window, measured in tokens — not words. Since code-heavy docs, JSON, or non-English text produce more tokens per character, you need to measure token count during chunking, not word count. For cost estimation: an average English document of 1,000 words ≈ 1,300 tokens. At GPT-4o pricing ($5/1M input tokens), processing 10,000 such documents = 13M tokens ≈ $65. You'd pre-compute token counts during ingestion to avoid expensive context-window-exceeded errors at query time.

**Q4: What is SentencePiece and why is it preferred for multilingual models?**

> SentencePiece operates on the raw character stream without pre-tokenising on whitespace. This makes it fully language-agnostic — ideal for models like mT5 or LLaMA that handle Japanese, Chinese, or Arabic, which don't use spaces as word boundaries. Standard BPE/WordPiece break text on spaces first, which fails for these languages. SentencePiece uses a ▁ character to denote word beginnings and learns subwords purely from the byte stream.

**Q5: Why do LLMs sometimes struggle with reversing strings or counting letters?**

> This is fundamentally a tokenisation issue. "strawberry" as a token sequence might be ["str", "awb", "erry"] — the model sees 3 tokens, not 10 characters. It has no representation of individual characters unless they happen to be their own tokens. Counting letters requires character-level reasoning, but the model was trained on token-level patterns. Models with byte-level tokenisation or special character tokens handle this better, and modern reasoning models use extended thinking to decompose the problem into explicit character-by-character steps.

---

*Tokenization Deep Dive | Phase 1 Supplementary | Added April 2026*
