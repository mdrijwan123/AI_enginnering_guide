# Natural Language Processing (NLP) — Complete Study Guide

> **Excel Curriculum Coverage**: Text Preprocessing, Word Embeddings, Text Classification, Named Entity Recognition, Sequence Models
> **Interview Focus**: Classical NLP → modern NLP pipeline understanding → why Transformers replaced RNNs
> **Day-to-Day**: Every LLM pipeline has NLP at its core — tokenization, preprocessing, evaluation, classification

---

## Table of Contents
1. [Text Preprocessing](#1-text-preprocessing)
2. [TF-IDF & Bag of Words](#2-tf-idf--bag-of-words)
3. [Word Embeddings — Word2Vec, GloVe, FastText](#3-word-embeddings)
4. [Text Classification](#4-text-classification)
5. [Named Entity Recognition (NER)](#5-named-entity-recognition)
6. [Sequence Models — RNN, LSTM, GRU](#6-sequence-models)
7. [Seq2Seq & Attention for NLP](#7-seq2seq--attention)
8. [BERT vs GPT Deep Dive](#8-bert-vs-gpt-deep-dive)
9. [Interview Questions (40 Q&As)](#9-interview-questions)
10. [Day-to-Day Work Applications](#10-day-to-day-work-applications)
11. [Resources](#11-resources)

---

## 1. Text Preprocessing

> 💡 **ELI5 (Explain Like I'm 5):**
> Imagine trying to organise a messy filing cabinet. Before you can sort anything, you need to remove the paperclips, flatten the crumpled pages, and rewrite messy cursive into neat print. Text preprocessing is exactly this: stripping away punctuation, making everything lowercase, and turning "running" into "run", so the AI doesn't get confused thinking "Run" and "running" are different words.

> 📖 **Big picture:** Raw text from the internet, emails, documents, or user input is messy — inconsistent capitalisation, punctuation everywhere, meaningless filler words ("the", "a", "is"), and words that mean the same thing but are written differently ("running", "runs", "ran"). Before any ML model can learn from text, you need to clean and normalise it.
>
> **The pipeline analogy:** Think of text preprocessing like washing and chopping vegetables before cooking. You could cook them raw and dirty, but the end result would be worse. Similarly, training a model on messy raw text gives worse results than training on clean, normalised text.
>
> **When it matters less now:** Modern LLMs (GPT-4, LLaMA 3) handle raw text well and don't need most preprocessing. But for classical NLP (TF-IDF, Naive Bayes, small task-specific models) and for embedding pipelines where you want consistent representations, these steps still matter.

### Why It Matters
Every NLP pipeline starts with text preprocessing. Raw text is messy — different cases, punctuation, stop words, inflections. Preprocessing normalizes text so models can learn meaningful patterns instead of noise.

### Key Steps in a Preprocessing Pipeline

```python
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# --- 1. Lowercasing ---
text = "The Quick Brown FOX jumped Over the Lazy DOG!"
lower = text.lower()
# "the quick brown fox jumped over the lazy dog!"

# --- 2. Removing Punctuation & Special Characters ---
cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', lower)
# "the quick brown fox jumped over the lazy dog"

# --- 3. Tokenization ---
# Word tokenization
tokens = word_tokenize(cleaned)
# ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']

# Sentence tokenization
sentences = sent_tokenize("Hello world. This is NLP. It's amazing!")
# ['Hello world.', 'This is NLP.', "It's amazing!"]

# --- 4. Stop Word Removal ---
stop_words = set(stopwords.words('english'))
filtered = [w for w in tokens if w not in stop_words]
# ['quick', 'brown', 'fox', 'jumped', 'lazy', 'dog']

# --- 5. Stemming (rule-based, aggressive) ---
stemmer = PorterStemmer()
stemmed = [stemmer.stem(w) for w in filtered]
# ['quick', 'brown', 'fox', 'jump', 'lazi', 'dog']

# --- 6. Lemmatization (dictionary-based, preserves meaning) ---
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w, pos='v') for w in filtered]
# ['quick', 'brown', 'fox', 'jump', 'lazy', 'dog']
```

### Stemming vs Lemmatization

| Feature | Stemming | Lemmatization |
|---------|----------|---------------|
| Method | Rule-based suffix stripping | Dictionary/morphological analysis |
| Speed | Faster | Slower |
| Output | May not be real words ("studi") | Always real words ("study") |
| Example | "running" → "run" | "running" → "run" |
| Example | "better" → "better" | "better" → "good" |
| Use Case | Search engines, IR | Text classification, NLU |

### Tokenization Strategies

| Method | How It Works | Example |
|--------|-------------|---------|
| **Whitespace** | Split on spaces | "New York" → ["New", "York"] |
| **Word-level** (NLTK) | Grammar-aware splitting | "don't" → ["do", "n't"] |
| **Subword BPE** | Merge frequent byte pairs | "unhappiness" → ["un", "happiness"] |
| **WordPiece** | Similar to BPE, used in BERT | "playing" → ["play", "##ing"] |
| **SentencePiece** | Language-agnostic, used in LLaMA | Handles any language without pre-tokenization |

### N-grams

```python
from nltk.util import ngrams

tokens = ['natural', 'language', 'processing', 'is', 'fun']
bigrams = list(ngrams(tokens, 2))
# [('natural', 'language'), ('language', 'processing'), ('processing', 'is'), ('is', 'fun')]
trigrams = list(ngrams(tokens, 3))
# [('natural', 'language', 'processing'), ('language', 'processing', 'is'), ('processing', 'is', 'fun')]
```

### Regular Expressions for Text Cleaning

```python
import re

text = "Email me at user@example.com or call +1-555-0123. Visit https://example.com"

# Extract emails
emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
# ['user@example.com']

# Extract URLs
urls = re.findall(r'https?://\S+', text)
# ['https://example.com']

# Extract phone numbers
phones = re.findall(r'\+?\d[\d\-]{8,}\d', text)
# ['+1-555-0123']

# Remove HTML tags
html = "<p>Hello <b>World</b></p>"
clean = re.sub(r'<[^>]+>', '', html)
# "Hello World"
```

---

## 2. TF-IDF & Bag of Words

> 📖 **Big picture:** Before embeddings existed, how did machines understand text? The simplest approach: count words. If a document contains the word "cancer" 15 times, it’s probably about medicine. If it contains "bitcoin" 20 times, it’s about cryptocurrency. **Bag of Words** does exactly this: treats each document as an unordered collection of word counts, ignoring grammar and word order.
>
> **The problem BoW solves:** It converts variable-length text into fixed-length numerical vectors that machine learning algorithms can process. "I love NLP" becomes `[0, 0, 1, 0, 1, 0, 1, 0...]` (1 for each word that appears).
>
> **TF-IDF improves on raw counts:** Some words ("the", "and", "is") appear in *every* document and carry no information. TF-IDF downweights these common words and upweights rare words that are distinctive for specific documents. It’s the foundation of keyword-based search (BM25, used in Elasticsearch).

### Bag of Words (BoW)

Simple document representation: count word occurrences. Ignores order entirely.

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Cats and dogs are friends"
]

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
# ['and', 'are', 'cat', 'cats', 'dog', 'dogs', 'friends', 'log', 'mat', 'on', 'sat', 'the']
print(bow_matrix.toarray())
# [[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 2],
#  [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 2],
#  [1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0]]
```

### TF-IDF (Term Frequency — Inverse Document Frequency)

Weighs terms by how important they are in a document relative to the corpus.

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

Where:
- $\text{TF}(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total terms in } d}$
- $\text{IDF}(t) = \log\frac{N}{1 + \text{df}(t)}$ where $N$ = total documents, $\text{df}(t)$ = documents containing $t$

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "Machine learning is great",
    "Deep learning is a subset of machine learning",
    "NLP uses machine learning techniques"
]

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(corpus)

# Get feature names and scores for first document
feature_names = tfidf.get_feature_names_out()
scores = tfidf_matrix[0].toarray().flatten()
for name, score in sorted(zip(feature_names, scores), key=lambda x: -x[1]):
    if score > 0:
        print(f"{name}: {score:.4f}")
# great: 0.6316 (unique to doc 0 → high IDF)
# is: 0.4481 
# learning: 0.3556
# machine: 0.3556
```

### When to Use TF-IDF vs Embeddings

| Scenario | TF-IDF | Neural Embeddings |
|----------|--------|-------------------|
| Small dataset (<10K docs) | ✅ Preferred | ⚠️ May overfit |
| Keyword-heavy search | ✅ Exact match | ❌ Misses keywords |
| Semantic similarity | ❌ No semantics | ✅ Captures meaning |
| Speed & interpretability | ✅ Fast, sparse | ❌ Dense, slower |
| Production RAG systems | ✅ BM25 (TF-IDF variant) | ✅ Hybrid search |

---

## 3. Word Embeddings

> 💡 **ELI5 (Explain Like I'm 5):** 
> Imagine a massive city map. We place all the pet stores in the north, and all the banks in the south. The word "Dog" gets coordinates (North 10, East 5). "Cat" gets (North 11, East 4). They are close together! "Mortgage" gets (South 5, West 2). An embedding is simply giving every word a set of GPS coordinates in "meaning space." Words with similar meanings get similar coordinates.

> 📖 **Big picture:** BoW and TF-IDF have a fatal flaw: "happy" and "joyful" are completely unrelated in a BoW vocabulary, even though they mean almost the same thing. Word embeddings fix this by representing each word as a dense vector in a "meaning space" where similar words are mathematically close.
>
> **The evolution of word representations shows the history of NLP:**
> - **One-hot encoding (pre-2013):** Each word is a binary vector; "king" = [0,0,1,0,0...]. No meaning captured, vectors are sparse and orthogonal.
> - **Word2Vec (2013):** Trains on the principle "you shall know a word by the company it keeps." Words that appear in similar contexts get similar vectors. "king" and "queen" end up near each other.
> - **GloVe (2014):** Similar to Word2Vec but uses global co-occurrence statistics instead of local context windows.
> - **Contextual embeddings / BERT (2018+):** The same word gets *different* vectors depending on context. "bank" in "river bank" vs "bank account" gets different representations. This is the foundation of modern LLMs.

### The Evolution

```
One-Hot Encoding → BoW/TF-IDF → Word2Vec/GloVe → Contextual (ELMo/BERT)
  (sparse, no semantics)   (dense, fixed)    (dense, context-aware)
```

### Word2Vec (Google, 2013)

Two architectures for learning word vectors from large corpora:

**CBOW (Continuous Bag of Words)**: Predict center word from context
```
Context: ["The", "cat", ___, "on", "the"] → Predict: "sat"
```

**Skip-gram**: Predict context words from center word
```
Center: "sat" → Predict: ["The", "cat", "on", "the"]
```

```python
from gensim.models import Word2Vec

# Training data: list of tokenized sentences
sentences = [
    ["king", "queen", "prince", "princess", "royal", "palace"],
    ["man", "woman", "boy", "girl", "person"],
    ["cat", "dog", "pet", "animal", "veterinarian"],
    ["deep", "learning", "neural", "network", "training"],
    ["natural", "language", "processing", "text", "nlp"],
]

# Train Word2Vec
model = Word2Vec(
    sentences, 
    vector_size=100,  # Embedding dimension
    window=5,         # Context window size
    min_count=1,      # Minimum word frequency
    sg=1,             # 1=Skip-gram, 0=CBOW
    workers=4,
    epochs=100
)

# Get word vector
vector = model.wv['king']  # 100-dimensional numpy array

# Famous analogy: king - man + woman ≈ queen
result = model.wv.most_similar(
    positive=['king', 'woman'], 
    negative=['man'], 
    topn=3
)
# [('queen', 0.85), ('princess', 0.72), ...]

# Word similarity
sim = model.wv.similarity('cat', 'dog')  # ~0.78
```

**Key Properties of Word2Vec**:
- Captures semantic relationships: king-man+woman ≈ queen
- Fixed vector per word (no polysemy handling)
- Efficient training with negative sampling
- Typical dimensions: 100-300

### GloVe (Stanford, 2014)

**Global Vectors for Word Representation** — combines global matrix factorization with local context windows.

Key idea: The ratio of co-occurrence probabilities encodes meaning.

$$J = \sum_{i,j=1}^{V} f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

Where $X_{ij}$ is the co-occurrence count of words $i$ and $j$.

```python
# Using pre-trained GloVe embeddings
import numpy as np

def load_glove(path, dim=100):
    """Load GloVe embeddings from file."""
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Load pre-trained (download from https://nlp.stanford.edu/projects/glove/)
# glove = load_glove('glove.6B.100d.txt')

# Using gensim to load GloVe
import gensim.downloader as api
glove_model = api.load("glove-wiki-gigaword-100")

# Analogy: Paris - France + Italy = Rome
result = glove_model.most_similar(
    positive=['paris', 'italy'], 
    negative=['france'], 
    topn=3
)
# [('rome', 0.88), ...]
```

### FastText (Facebook, 2016)

Extension of Word2Vec that uses **subword information** (character n-grams).

Key advantage: Can generate embeddings for **out-of-vocabulary (OOV) words**.

```python
from gensim.models import FastText

# FastText breaks words into character n-grams
# "apple" with n=3: ["<ap", "app", "ppl", "ple", "le>"]
# Embedding("apple") = sum of all n-gram embeddings

model = FastText(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,           # Skip-gram
    min_n=3,        # Min n-gram length
    max_n=6,        # Max n-gram length
    epochs=100
)

# Can handle OOV words!
vector = model.wv['unseen_word']  # Works! (summing subword vectors)
```

### Comparison Table

| Feature | Word2Vec | GloVe | FastText |
|---------|----------|-------|----------|
| Training | Local context (window) | Global co-occurrence matrix | Local context + subwords |
| OOV Words | ❌ No | ❌ No | ✅ Yes |
| Subword Info | ❌ No | ❌ No | ✅ Yes |
| Training Speed | Fast | Medium | Slower |
| Morphologically Rich Languages | Poor | Poor | ✅ Excellent |
| Best For | General-purpose | Analogy tasks | Rare words, morphology |

### From Static to Contextual Embeddings

```
Word2Vec "bank" → same vector always
ELMo    "bank" → different vector in "river bank" vs "bank account"  
BERT    "bank" → deeply contextual across all layers
```

---

## 4. Text Classification

> 📖 **Big picture:** Text classification is the task of assigning a label to a piece of text. It's one of the most common NLP tasks in production: spam detection, sentiment analysis, content moderation, intent classification for chatbots, ticket routing in customer support.
>
> **The evolution of approaches:** In 2015, you'd use TF-IDF + Logistic Regression. In 2019, you'd fine-tune BERT. In 2024, you'd either fine-tune a smaller model for cost efficiency OR use a large LLM with a well-crafted prompt for zero/few-shot classification. Knowing all three approaches and *when to use each* is what FAANG interviewers want to hear.
>
> **Interview insight:** When asked "how would you build a text classifier?", the right answer isn't just one approach — it's a decision tree: "For low data, start with zero-shot LLM. For high accuracy with labeled data, fine-tune. For production at scale, distil to a small model."

### Problem Types
- **Binary**: Spam vs Not Spam, Positive vs Negative
- **Multi-class**: News categories (Sports, Politics, Tech, Business)
- **Multi-label**: Movie tags (Action + Comedy + Romance)

### Classical Approaches

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Example: Sentiment Classification
texts = [
    "This movie was absolutely wonderful and amazing",
    "Terrible film, worst I've ever seen", 
    "Great acting and beautiful cinematography",
    "Boring plot, waste of time",
    "Outstanding performance by the lead actor",
    "Awful, I walked out of the theater",
]
labels = [1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

# --- Naive Bayes Pipeline ---
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('clf', MultinomialNB())
])
nb_pipeline.fit(X_train, y_train)
predictions = nb_pipeline.predict(X_test)

# --- Logistic Regression Pipeline ---
lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 3))),
    ('clf', LogisticRegression(max_iter=1000, C=1.0))
])

# --- SVM Pipeline ---
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, sublinear_tf=True)),
    ('clf', LinearSVC(C=1.0))
])

print(classification_report(y_test, predictions))
```

### Zero-Shot Classification with Transformers

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The new iPhone has an amazing camera and long battery life"
candidate_labels = ["technology", "sports", "politics", "entertainment"]

result = classifier(text, candidate_labels)
# {'labels': ['technology', 'entertainment', 'politics', 'sports'],
#  'scores': [0.92, 0.04, 0.02, 0.02]}
```

### Sentiment Analysis with Transformers

```python
from transformers import pipeline

sentiment = pipeline("sentiment-analysis")

results = sentiment([
    "I love this product! Best purchase ever.",
    "Terrible experience. Never buying again.",
    "It's okay, nothing special."
])
# [{'label': 'POSITIVE', 'score': 0.9998},
#  {'label': 'NEGATIVE', 'score': 0.9994},
#  {'label': 'NEGATIVE', 'score': 0.6823}]
```

---

## 5. Named Entity Recognition

> 💡 **ELI5 (Explain Like I'm 5):** 
> Imagine reading a book with three highlighters. Every time you see a person's name, you highlight it in yellow. Every time you see a city, green. Every time you see a date, pink. NER is just teaching an AI to hold those digital highlighters and automatically color-code the important nouns in a sentence.

> 📖 **Big picture:** Named Entity Recognition (NER) finds and classifies *specific things* mentioned in text: people's names, organisations, locations, dates, dollar amounts. It's the "highlighter pass" over a document — identify all the important nouns and label what type they are.
>
> **Why it matters in production:** NER is a core pre-processing step in many AI pipelines. Before you can answer "what did Apple announce last quarter?", you need to identify "Apple" as an organisation and "last quarter" as a time period. RAG systems, knowledge graph construction, and document intelligence all depend on reliable NER.
>
> **Modern approach:** Fine-tuned transformer models (BERT, RoBERTa) achieve near-human accuracy. For most production use cases, SpaCy or Hugging Face pipelines give you pre-trained models you can use immediately or fine-tune on domain-specific entities (medical terms, legal entities, company-specific product names).

### What is NER?
Identifying and classifying named entities in text into predefined categories:
- **PER** (Person): "Elon Musk"
- **ORG** (Organization): "Google"
- **LOC** (Location): "San Francisco"
- **DATE**: "January 2026"
- **MONEY**: "$100 million"

### BIO Tagging Scheme

```
Token:  "Barack"  "Obama"  "visited"  "Berlin"  "in"  "2024"
Tag:    B-PER     I-PER    O          B-LOC     O     B-DATE
```

- **B-XXX**: Beginning of entity XXX
- **I-XXX**: Inside (continuation of) entity XXX
- **O**: Outside any entity

### spaCy NER

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976")

for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_:10s} {spacy.explain(ent.label_)}")
# Apple Inc.           ORG        Companies, agencies, institutions
# Steve Jobs           PERSON     People, including fictional
# Cupertino            GPE        Countries, cities, states
# California           GPE        Countries, cities, states
# 1976                 DATE       Absolute or relative dates or periods
```

### Transformer-based NER

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

text = "Hugging Face is based in New York and was founded by Clément Delangue"
entities = ner(text)

for entity in entities:
    print(f"{entity['word']:20s} {entity['entity_group']:10s} {entity['score']:.4f}")
# Hugging Face         ORG        0.9987
# New York             LOC        0.9993
# Clément Delangue     PER        0.9989
```

### Custom NER with spaCy

```python
import spacy
from spacy.training import Example

# Create blank model
nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

# Add custom entity labels
ner.add_label("TECH_FRAMEWORK")
ner.add_label("AI_MODEL")

# Training data
train_data = [
    ("LangChain is a framework for LLMs", {
        "entities": [(0, 9, "TECH_FRAMEWORK")]
    }),
    ("GPT-4 can generate code", {
        "entities": [(0, 5, "AI_MODEL")]
    }),
    ("We deployed using vLLM for inference", {
        "entities": [(17, 21, "TECH_FRAMEWORK")]
    }),
]

# Train
optimizer = nlp.begin_training()
for epoch in range(30):
    losses = {}
    for text, annotations in train_data:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], drop=0.3, losses=losses)
    print(f"Epoch {epoch}: {losses}")
```

### CRF (Conditional Random Fields) for NER

CRFs model the entire label sequence, not just individual labels:

```python
# Using sklearn-crfsuite
import sklearn_crfsuite
from sklearn_crfsuite import metrics

def word_features(sent, i):
    """Extract features for a word in a sentence."""
    word = sent[i]
    features = {
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        prev_word = sent[i-1]
        features['prev_word.lower()'] = prev_word.lower()
        features['prev_word.istitle()'] = prev_word.istitle()
    if i < len(sent)-1:
        next_word = sent[i+1]
        features['next_word.lower()'] = next_word.lower()
        features['next_word.istitle()'] = next_word.istitle()
    return features

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100
)
# crf.fit(X_train, y_train)
# predictions = crf.predict(X_test)
```

---

## 6. Sequence Models

> 📖 **Big picture:** This section explains *why transformers exist* — by showing the problems that the models before them (RNNs, LSTMs) had. Understanding the history makes the transformer design choices obvious rather than arbitrary.
>
> **The problem with text:** Text is a sequence. "John loves Mary" has the same words as "Mary loves John" but opposite meanings. BoW lost order entirely. RNNs were the first serious attempt to model sequential structure: process text one word at a time, maintaining a running "memory" of what came before.
>
> **Why RNNs failed at scale:** The memory is compressed into a single vector (the hidden state), regardless of how long the sequence is. A 1000-word document must compress all information into a fixed-size vector. Long-range dependencies are lost — the model "forgets" what was said 50 words ago. LSTMs improved this with gating mechanisms, but couldn’t truly parallelize (each step depends on the previous). Transformers solved all of this.

### Why Sequences Matter
Text is inherently sequential — word order carries meaning:
- "Dog bites man" ≠ "Man bites dog"
- Classical BoW/TF-IDF lose this order

### Recurrent Neural Networks (RNN)

Process sequences one element at a time, maintaining a hidden state:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$
$$y_t = W_{hy} h_t + b_y$$

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)           # (batch, seq_len, embed_dim)
        output, hidden = self.rnn(embedded)     # output: (batch, seq_len, hidden_dim)
        # Use last hidden state for classification
        last_hidden = hidden.squeeze(0)         # (batch, hidden_dim)
        return self.fc(last_hidden)             # (batch, output_dim)

model = SimpleRNN(vocab_size=10000, embed_dim=128, hidden_dim=256, output_dim=2)
```

### The Vanishing/Exploding Gradient Problem

During backpropagation through time (BPTT), gradients are multiplied at each step:

$$\frac{\partial L}{\partial h_0} = \frac{\partial L}{\partial h_T} \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

- If $\|\frac{\partial h_t}{\partial h_{t-1}}\| < 1$: gradients vanish → can't learn long-range dependencies
- If $\|\frac{\partial h_t}{\partial h_{t-1}}\| > 1$: gradients explode → unstable training

**Solutions**: Gradient clipping, LSTM, GRU

### LSTM (Long Short-Term Memory)

Addresses vanishing gradients with **gates** that control information flow:

```
       ┌──────────────────────────────┐
       │         LSTM Cell             │
       │                               │
 ──────┤ Forget Gate: σ(Wf·[h,x]+bf)  │──── What to forget from cell state
       │ Input Gate:  σ(Wi·[h,x]+bi)  │──── What new info to store
       │ Cell Update: tanh(Wc·[h,x])  │──── Candidate values
       │ Output Gate: σ(Wo·[h,x]+bo)  │──── What to output
       │                               │
       │ c_t = f_t * c_{t-1} + i_t * ĉ_t │── Cell state update
       │ h_t = o_t * tanh(c_t)        │──── Hidden state
       └──────────────────────────────┘
```

**Gate equations**:
- **Forget gate**: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
- **Input gate**: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
- **Cell candidate**: $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
- **Cell state**: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
- **Output gate**: $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
- **Hidden state**: $h_t = o_t \odot \tanh(C_t)$

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True  # Read forward AND backward
        )
        self.dropout = nn.Dropout(dropout)
        # *2 because bidirectional
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))  # (batch, seq_len, embed_dim)
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch, hidden*2)
        return self.fc(self.dropout(hidden))

model = LSTMClassifier(
    vocab_size=25000, embed_dim=300, hidden_dim=256, 
    output_dim=2, n_layers=2
)
```

### GRU (Gated Recurrent Unit)

Simplified version of LSTM — fewer parameters, often similar performance:

```
GRU has 2 gates (vs LSTM's 3):
- Update gate z_t: How much of past info to keep
- Reset gate r_t:  How much of past info to forget

z_t = σ(Wz · [h_{t-1}, x_t])
r_t = σ(Wr · [h_{t-1}, x_t])
h̃_t = tanh(W · [r_t * h_{t-1}, x_t])
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
```

```python
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(hidden)
```

### RNN vs LSTM vs GRU

| Feature | RNN | LSTM | GRU |
|---------|-----|------|-----|
| Gates | 0 | 3 (forget, input, output) | 2 (update, reset) |
| Parameters | Fewest | Most | Middle |
| Long-range deps | ❌ Poor | ✅ Excellent | ✅ Good |
| Training speed | Fast | Slowest | Faster than LSTM |
| Memory | Low | High | Medium |
| When to use | Short sequences | Long sequences, complex tasks | Good default choice |

---

## 7. Seq2Seq & Attention

> 📖 **Big picture:** Sequence-to-Sequence (Seq2Seq) models solve tasks where both input and output are sequences — translation (English → French), summarisation (long text → short text), question answering (question + context → answer). The key architecture: an **encoder** reads the input and compresses it to a fixed vector, a **decoder** generates the output from that vector.
>
> **The bottleneck problem:** The encoder's summary vector has to capture *everything* about the input. For long inputs, this single vector isn't enough — it forgets early parts of the sequence. **Attention** fixes this: instead of a single summary, the decoder can "attend" to every encoder state and choose which parts to focus on for each output word. This was the precursor to self-attention and the transformer.
>
> **Why learn this if transformers replaced it?** Because attention is the core innovation, and understanding Bahdanau attention (2014) makes transformer self-attention (2017) immediately intuitive. They're the same idea, geneneralised.

### Encoder-Decoder Architecture

Used for sequence-to-sequence tasks: translation, summarization, Q&A.

```
Encoder: "How are you?" → [h1, h2, h3] → context vector c
Decoder: c → "Comment allez-vous?"
```

**Problem**: The entire input is compressed into a single context vector (information bottleneck).

### Attention Mechanism (Bahdanau, 2014)

Instead of using only the last encoder state, attend to all encoder states:

$$\alpha_{t,s} = \frac{\exp(e_{t,s})}{\sum_{s'} \exp(e_{t,s'})}$$
$$c_t = \sum_s \alpha_{t,s} h_s$$

Where $e_{t,s} = \text{score}(h_t^{dec}, h_s^{enc})$ is the alignment score.

```python
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        # hidden: (batch, hidden_dim) - decoder hidden state
        # encoder_outputs: (batch, src_len, hidden_dim)
        
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # (batch, src_len, hidden)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # (batch, src_len)
        
        return torch.softmax(attention, dim=1)
```

### From Attention to Transformers

The key insight that led to Transformers:
1. RNNs process **sequentially** → cannot parallelize
2. Attention can relate any two positions **directly**
3. **Self-attention**: Use attention without RNNs (Transformer paper, 2017)

```
RNN + Attention → slow (sequential), good accuracy
Self-Attention Only → fast (parallel), better accuracy = TRANSFORMER
```

---

## 8. BERT vs GPT Deep Dive

> 📖 **Big picture:** By 2018, "transformer" had become the default architecture for NLP. But two very different families emerged from it, based on which part of the transformer they use:
>
> - **BERT (encoder-only):** Reads the whole sentence at once, understanding each word in context of the words before *and after* it. Great for understanding tasks: classification, NER, question answering, similarity.
> - **GPT (decoder-only):** Generates text left-to-right, predicting the next word. Great for generation tasks: completion, summarisation, instruction following.
>
> **The analogy:** BERT reads an essay and answers questions about it (comprehension). GPT writes the essay (generation). For FAANG AI roles, you need to know both cold: when to use each, how they're pre-trained, and what fine-tuning looks like for each family.

### BERT (Bidirectional Encoder Representations from Transformers)

**Architecture**: Encoder-only Transformer
**Pre-training Tasks**:
1. **Masked Language Model (MLM)**: Mask 15% of tokens, predict them
   ```
   Input:  "The [MASK] sat on the [MASK]"
   Output: "The cat sat on the mat"
   ```
2. **Next Sentence Prediction (NSP)**: Is sentence B the actual next sentence?

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
print(inputs['input_ids'])
# tensor([[  101,  7592,  1010,  2129,  2024,  2017,  1029,   102]])
#          [CLS]  Hello   ,     how    are   you    ?    [SEP]

# Get embeddings
outputs = model(**inputs)
last_hidden = outputs.last_hidden_state  # (1, 8, 768) - contextual embeddings
pooler_output = outputs.pooler_output    # (1, 768) - [CLS] token embedding

# BERT for Classification
from transformers import BertForSequenceClassification
classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
```

### GPT (Generative Pre-trained Transformer)

**Architecture**: Decoder-only Transformer
**Pre-training**: Autoregressive language modeling (predict next token)
**Key**: Uses **causal masking** — each token can only attend to previous tokens

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generate text
input_ids = tokenizer.encode("The future of AI is", return_tensors="pt")
output = model.generate(
    input_ids,
    max_length=50,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Architectural Comparison

| Feature | BERT | GPT |
|---------|------|-----|
| **Architecture** | Encoder-only | Decoder-only |
| **Attention** | Bidirectional (full) | Causal (left-to-right) |
| **Pre-training** | MLM + NSP | Next token prediction |
| **Best For** | Understanding tasks | Generation tasks |
| **Tasks** | Classification, NER, QA | Text generation, chat, code |
| **Context** | Sees all tokens | Sees only past tokens |
| **Fine-tuning** | Add task-specific head | Prompt engineering or SFT |

### T5 (Text-to-Text Transfer Transformer)

Encoder-decoder that treats ALL tasks as text-to-text:
```
Classification: "classify: This movie is great" → "positive"
Translation:    "translate English to French: Hello" → "Bonjour"
Summarization:  "summarize: [long text]" → "[summary]"
```

---

## 9. Interview Questions

### Q1: What is tokenization and why does it matter?
**A**: Tokenization splits text into smaller units (tokens) for model processing. It matters because it defines the model's vocabulary and affects how well it handles different languages, rare words, and context length. Modern models use subword tokenization (BPE, WordPiece) to balance vocabulary size with coverage.

### Q2: Stemming vs Lemmatization — when to use each?
**A**: Stemming is faster but produces non-words ("studies" → "studi"). Lemmatization is slower but produces real words ("studies" → "study") by using dictionary lookups. Use stemming for search/IR where exact form doesn't matter. Use lemmatization for NLU tasks where meaning matters.

### Q3: Explain TF-IDF intuitively.
**A**: TF-IDF measures how important a word is to a document in a corpus. TF (Term Frequency) measures how often a word appears in a document. IDF (Inverse Document Frequency) penalizes words that appear in many documents (like "the", "is"). A word with high TF-IDF is frequent in its document but rare across the corpus — making it a good discriminator.

### Q4: How does Word2Vec learn word representations?
**A**: Word2Vec uses a shallow neural network trained on predicting words from context (CBOW) or context from words (Skip-gram). Words appearing in similar contexts get similar vectors. The hidden layer weights become the word embeddings. Training uses negative sampling to efficiently approximate the softmax over the entire vocabulary.

### Q5: What's the difference between Word2Vec and GloVe?
**A**: Word2Vec uses local context windows (predictive model). GloVe uses the global word co-occurrence matrix (count-based model). GloVe directly factorizes the log co-occurrence matrix, capturing global statistics. Word2Vec often performs better on syntactic tasks, GloVe on semantic tasks. In practice, both produce similar quality embeddings.

### Q6: Why is FastText better for morphologically rich languages?
**A**: FastText represents words as bags of character n-grams. "playing" → ["<pl", "pla", "lay", "ayi", "yin", "ing", "ng>"]. This means it can generate embeddings for unseen words by summing their n-gram vectors. Languages like Turkish, Finnish, or Arabic with complex morphology benefit greatly because related word forms share n-grams.

### Q7: Explain the vanishing gradient problem in RNNs.
**A**: During BPTT, gradients are multiplied through each time step. If the gradient of the recurrence relation is < 1, repeated multiplication makes it exponentially small (vanishing). This means the network can't learn dependencies between distant tokens. For a 100-token sentence, the gradient from token 100 to token 1 passes through 99 multiplications, approaching zero.

### Q8: How does LSTM solve the vanishing gradient problem?
**A**: LSTM introduces a cell state that acts as a "highway" for gradients. The forget gate controls what to erase from the cell state, the input gate controls what to write, and the output gate controls what to expose. Because the cell state update is additive (not multiplicative), gradients can flow through many time steps without vanishing.

### Q9: When would you use GRU over LSTM?
**A**: GRU has fewer parameters (2 gates vs 3), trains faster, and often performs similarly on many tasks. Use GRU when: (1) dataset is small and you need fewer parameters to avoid overfitting, (2) training speed matters, (3) the task doesn't require very long-range dependencies. Use LSTM when maximum representational power is needed.

### Q10: What is bidirectional LSTM and when is it useful?
**A**: Bidirectional LSTM runs two LSTMs — one forward, one backward — and concatenates their outputs. This gives each position context from both past and future tokens. Essential for NER, POS tagging, and classification where the full context matters. Not suitable for generation tasks (can't see future tokens during generation).

### Q11: Explain the attention mechanism in seq2seq models.
**A**: Instead of compressing the entire input into a single vector, attention computes a weighted sum of all encoder states at each decoder step. The weights (attention scores) indicate which input tokens are most relevant for the current output token. This solves the information bottleneck problem and enables direct connections between input and output positions.

### Q12: What is BIO tagging in NER?
**A**: BIO (Beginning, Inside, Outside) is a tagging scheme for sequence labeling. B-PER marks the start of a person entity, I-PER continues it, O is non-entity. Example: "Barack Obama visited Berlin" → B-PER I-PER O B-LOC. It handles adjacent entities of the same type and multi-word entities cleanly.

### Q13: How does BERT's masked language model work?
**A**: BERT randomly masks 15% of input tokens. Of those, 80% are replaced with [MASK], 10% with random words, 10% unchanged. The model predicts the original token at masked positions. The 80/10/10 split prevents the model from only learning to predict [MASK] tokens, making it more generalizable to downstream tasks.

### Q14: Why is BERT bidirectional while GPT is unidirectional?
**A**: BERT uses full self-attention — each token attends to all other tokens. GPT uses causal (masked) self-attention — tokens only attend to previous tokens. BERT is bidirectional because it's designed for understanding (classification, NER). GPT is unidirectional because it's designed for generation (each token is predicted from left context).

### Q15: What's the difference between encoder-only, decoder-only, and encoder-decoder models?
**A**: Encoder-only (BERT): Bidirectional, good for understanding tasks (classification, NER). Decoder-only (GPT): Autoregressive, good for generation. Encoder-decoder (T5, BART): Combines both, good for seq2seq tasks (translation, summarization). Modern LLMs are mostly decoder-only as they can handle all tasks with appropriate prompting.

### Q16: What is perplexity and why does it matter?
**A**: Perplexity measures how well a language model predicts text. $PPL = 2^{H}$ where $H$ is cross-entropy. Lower perplexity = better predictions. A perplexity of 100 means the model is as confused as if it had to choose uniformly among 100 words at each step. Used to evaluate language models during pre-training.

### Q17: Explain zero-shot vs few-shot classification.
**A**: Zero-shot: Model classifies without any labeled examples, using only the task description. "Classify this review as positive/negative." Few-shot: Model sees a few examples in the prompt. "Positive: Great movie! Negative: Terrible film. Classify: Amazing acting →". Zero-shot requires strong pre-trained understanding; few-shot leverages in-context learning.

### Q18: What is the difference between soft attention and hard attention?
**A**: Soft attention computes weighted averages over all positions (differentiable, used in Transformers). Hard attention selects specific positions (categorical, requires reinforcement learning). Soft attention is standard because it's fully differentiable and trainable with backpropagation. Hard attention is computationally cheaper but harder to train.

### Q19: How would you handle class imbalance in text classification?
**A**: (1) Oversampling minority class or undersampling majority, (2) class-weighted loss functions, (3) focal loss for extreme imbalance, (4) data augmentation (synonym replacement, back-translation, paraphrase), (5) stratified train/test splits, (6) evaluation with F1/AUC instead of accuracy.

### Q20: What is back-translation for text augmentation?
**A**: Translate text to another language, then back to the original. "Great movie!" → French: "Super film!" → English: "Awesome movie!". This creates natural paraphrases for data augmentation. Useful for low-resource text classification to increase training set diversity.

### Q21: Explain cross-attention vs self-attention.
**A**: Self-attention: Q, K, V all come from the same sequence (within encoder or decoder). Cross-attention: Q comes from the decoder, K and V from the encoder output. Cross-attention is how the decoder "reads" the encoder — used in translation, summarization, and encoder-decoder models like T5.

### Q22: What is the CRF layer in NER and why is it needed?
**A**: CRF (Conditional Random Field) models dependencies between output labels. Without CRF, each token is classified independently, allowing invalid sequences (e.g., I-PER after O). CRF learns transition probabilities (B-PER is likely followed by I-PER, not I-ORG) and finds the globally optimal label sequence using the Viterbi algorithm.

### Q23: How do subword tokenizers handle unknown words?
**A**: BPE and WordPiece break unknown words into known subword units. "unhappiness" → "un" + "happiness" or "un" + "hap" + "pi" + "ness". This means there are NO truly unknown words — any word can be represented as a sequence of subwords (worst case: individual characters). This is why modern models don't need a fixed vocabulary.

### Q24: What is teacher forcing in seq2seq training?
**A**: During training, instead of feeding the model's own predictions as the next input, we feed the ground truth target token. This speeds up training and prevents error propagation. However, it creates exposure bias — at inference time, the model must use its own (potentially wrong) predictions, which it never saw during training.

### Q25: How does BERT handle multiple sentences?
**A**: BERT uses segment embeddings (segment A, segment B) and special tokens. Input: [CLS] Sentence A [SEP] Sentence B [SEP]. Each token gets: token embedding + position embedding + segment embedding. This enables tasks like sentence pair classification, QA (question + passage), and natural language inference.

### Q26: Explain the difference between autoregressive and autoencoding models.
**A**: Autoregressive (GPT): Generate text left-to-right, each token conditioned on all previous tokens. P(x) = ∏ P(x_i | x_<i). Autoencoding (BERT): Reconstruct corrupted input. P(x_masked | x_unmasked). Autoregressive excels at generation, autoencoding at understanding. They represent fundamentally different pre-training paradigms.

### Q27: What is BM25 and how does it relate to TF-IDF?
**A**: BM25 is a probabilistic ranking function that extends TF-IDF with term frequency saturation and document length normalization. TF-IDF: TF grows linearly. BM25: TF saturates (a term appearing 100 times isn't 10x more important than appearing 10 times). BM25 is the standard for keyword search and is used in hybrid RAG retrieval systems.

### Q28: How would you evaluate an NER model?
**A**: Token-level: precision, recall, F1 per entity type. Entity-level: exact match (entire entity span and type must be correct) and partial match. Key metric: micro-averaged F1 across all entity types. Also evaluate: boundary detection accuracy, confusion between similar entity types (PER vs ORG), and performance on rare entities.

### Q29: What is the difference between word-level and token-level NER evaluation?
**A**: Word-level: evaluate if each word's entity label is correct independently. Token-level: evaluate after subword tokenization — "New York" might be ["new", "york"] or ["new", "yor", "##k"]. For subword models, we typically take the first subword's prediction as the word's label and aggregate at the word level for final metrics.

### Q30: What are contextual embeddings and why are they better than static?
**A**: Static embeddings (Word2Vec/GloVe) assign one vector per word regardless of context. "bank" has the same embedding in "river bank" and "bank account." Contextual embeddings (BERT/GPT) generate different vectors for the same word depending on context. This captures polysemy and word sense, significantly improving task performance.

### Q31: Explain position embeddings in Transformers.
**A**: Transformers process all tokens in parallel (unlike RNNs), losing sequence order. Position embeddings inject order information. Sinusoidal (fixed): use sin/cos functions of position and dimension. Learned: train an embedding matrix indexed by position. RoPE: rotate embeddings to encode relative positions. ALiBi: add linear bias to attention scores based on distance.

### Q32: What is the information bottleneck in seq2seq models?
**A**: In vanilla seq2seq, the entire input sequence is compressed into a single fixed-size vector (the last encoder hidden state). For long inputs, important information is lost because the vector can't capture everything. Attention solves this by allowing the decoder to access all encoder states, creating a dynamic summary at each step.

### Q33: How does beam search work for text generation?
**A**: Beam search maintains the top-k (beam width) most likely partial sequences at each step. At each step, each beam is extended by all vocabulary tokens, scored, and only the top-k survive. This avoids greedy decoding's myopia without the exponential cost of exhaustive search. Trade-off: larger beam = better quality but slower and less diverse.

### Q34: Explain nucleus (top-p) sampling vs top-k sampling.
**A**: Top-k: Sample from the k most likely tokens. Problem: k is fixed regardless of probability distribution shape. Nucleus (top-p): Sample from the smallest set of tokens whose cumulative probability exceeds p. Adapts to the distribution — when the model is confident, fewer tokens are considered; when uncertain, more tokens are included.

### Q35: What is knowledge distillation for NLP models?
**A**: Train a smaller "student" model to mimic a larger "teacher" model. The student learns from the teacher's soft probability distribution over vocabulary (dark knowledge), not just hard labels. DistilBERT is 60% smaller than BERT with 97% of its performance. Useful for deploying models on edge devices or reducing inference cost.

### Q36: How do you handle multi-label text classification vs multi-class?
**A**: Multi-class: Use softmax (mutually exclusive classes, one label per sample). Multi-label: Use sigmoid per class (independent binary decisions, multiple labels per sample). Multi-label uses binary cross-entropy loss per label. Threshold selection matters — use precision-recall curves per label to find optimal thresholds.

### Q37: What is the role of the [CLS] token in BERT?
**A**: [CLS] (Classification) is prepended to every BERT input. During pre-training, it's used for the Next Sentence Prediction task. For fine-tuning on classification, the [CLS] token's final hidden state serves as the aggregate sequence representation. It's passed through a classification head to produce predictions. However, mean pooling of all tokens often works better.

### Q38: Explain co-reference resolution.
**A**: Identifying when different expressions refer to the same entity. "John went to the store. He bought milk." → "He" = "John". Important for document understanding, dialogue systems, and information extraction. Modern approaches use span-based models that score pairs of mentions. Critical for building coherent multi-turn chatbots and document Q&A systems.

### Q39: What is constituency vs dependency parsing?
**A**: Constituency parsing: Breaks sentences into nested phrases (NP, VP, PP). "The cat sat" → [S [NP [The cat]] [VP [sat]]]. Dependency parsing: Identifies head-modifier relationships between words. "cat" → nsubj → "sat". Dependency parsing is more common in modern NLP because it directly captures grammatical relationships useful for information extraction.

### Q40: How has NLP evolved from rule-based to LLMs?
**A**: Evolution: (1) Rule-based: hand-crafted patterns and grammars, (2) Statistical: HMMs, CRFs, Naive Bayes with hand-engineered features, (3) Neural: Word2Vec + RNN/LSTM with learned representations, (4) Pre-trained: BERT/GPT with transfer learning from massive corpora, (5) LLMs: GPT-4, Claude with in-context learning, few-shot capabilities, and emergent abilities. Each step reduced human effort and increased capability.

---

## 10. Day-to-Day Work Applications

### As an AI/LLM Engineer

**Text Preprocessing in Production Pipelines**:
- Cleaning user inputs before sending to LLMs (removing PII, normalizing text)
- Building preprocessing pipelines for RAG indexing (chunking, cleaning HTML, handling tables)
- Tokenization analysis for context window budgeting ("How many tokens will this prompt use?")

**NER in Real Products**:
- Extracting structured data from unstructured documents (invoices, contracts, resumes)
- Building entity-based search filters ("Find all documents mentioning Google or Microsoft")
- Custom NER for domain-specific entities (product names, medical terms, financial tickers)

**Text Classification in Production**:
- Intent classification for chatbots ("Is this a billing question, technical issue, or general inquiry?")
- Content moderation (flagging toxic, inappropriate, or off-topic content)
- Routing customer queries to appropriate agents/departments
- Sentiment monitoring on social media or product reviews

**Embeddings & Similarity**:
- RAG retrieval fundamentally relies on embedding similarity
- Duplicate detection (finding similar tickets, documents, or queries)
- Recommendation systems based on content similarity
- Clustering documents for topic discovery

**Sequence Model Knowledge**:
- Understanding why Transformers replaced RNNs helps you explain architecture decisions
- Debugging attention patterns in RAG systems
- Evaluating when simpler models (TF-IDF + Logistic Regression) might outperform LLMs
- Cost-benefit analysis: "Do we need GPT-4 or would a fine-tuned BERT work?"

**In Interviews**:
- System design: "Design a content moderation system" → NER + classification + embeddings
- Coding: Implement TF-IDF from scratch, basic tokenizer, sentiment pipeline
- ML depth: Explain RNN → LSTM → Transformer evolution, why attention works
- Trade-offs: When to use classical NLP vs LLMs (cost, latency, accuracy)

---

## 11. Resources

### Excel Curriculum Links
- NLP Basics: https://www.youtube.com/watch?v=fNxaJsNG3-s
- NLTK Tutorial: https://www.youtube.com/watch?v=X2vAabgKiuM
- spaCy NLP Course: https://course.spacy.io/
- Word2Vec Explained: https://www.youtube.com/watch?v=viZrOnJclY0
- GloVe Paper: https://nlp.stanford.edu/projects/glove/
- FastText Paper: https://arxiv.org/abs/1607.04606
- Stanford NLP with Deep Learning (CS224N): https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ
- Hugging Face NLP Course: https://huggingface.co/learn/nlp-course/chapter1/1
- Text Classification: https://www.youtube.com/watch?v=VtRLB_w1ReE
- NER Tutorial: https://www.youtube.com/watch?v=NLPat1oH1CU
- Seq2Seq: https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
- Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- BERT Paper: https://arxiv.org/abs/1810.04805
- GPT-2 Paper: https://openai.com/research/better-language-models

### Additional Resources
- Natural Language Processing with Python (NLTK Book): https://www.nltk.org/book/
- Stanford CS224N: NLP with Deep Learning lectures
- Real-world NLP by Masato Hagiwara
- Speech and Language Processing by Jurafsky & Martin: https://web.stanford.edu/~jurafsky/slp3/
