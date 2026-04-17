# Python for AI Engineers — Complete Study Guide

> **Excel Curriculum Coverage**: Python Basics, Data Structures, Functions & Modules, OOP, File Handling & APIs
> **Interview Focus**: Python fluency is assumed — master advanced patterns, OOP design, API integration
> **Day-to-Day**: Every AI/ML pipeline is in Python — OOP for production code, APIs for serving, file I/O for data

---

## Table of Contents
1. [Advanced Data Structures](#1-advanced-data-structures)
2. [Functions, Decorators & Generators](#2-functions-decorators--generators)
3. [Object-Oriented Programming](#3-oop)
4. [File Handling & Data Formats](#4-file-handling--data-formats)
5. [REST APIs & HTTP](#5-rest-apis--http)
6. [Async Python](#6-async-python)
7. [Type Hints & Pydantic](#7-type-hints--pydantic)
8. [Testing & Debugging](#8-testing--debugging)
9. [Package Management & Virtual Environments](#9-package-management)
10. [Interview Questions (30 Q&As)](#10-interview-questions)
11. [Day-to-Day Work Applications](#11-day-to-day-work-applications)
12. [Resources](#12-resources)

---

## 1. Advanced Data Structures

> 💡 **ELI5 (Explain Like I'm 5):** Think of normal Python lists and dicts like a basic hammer and screwdriver. They work for almost everything, but sometimes you need a power drill. Advanced data structures are power tools designed for specific jobs—like counting items instantly or preventing errors when a key doesn't exist.

> 📖 **Big picture:** Standard Python lists and dicts get you far, but they’re not always the right tool. Python’s `collections` module gives you purpose-built data structures that make common patterns much cleaner and often faster. `defaultdict` removes the "handle missing key" boilerplate you’d otherwise write every time. `Counter` replaces a 3-line frequency loop with one call. `deque` gives you O(1) insertions at both ends (lists are O(n) for left-end insertions). In production AI code, you’ll use all of these constantly.

### Collections Module

```python
from collections import defaultdict, Counter, OrderedDict, deque, namedtuple

# --- defaultdict: no KeyError ---
word_counts = defaultdict(int)
for word in ["apple", "banana", "apple", "cherry"]:
    word_counts[word] += 1
# {'apple': 2, 'banana': 1, 'cherry': 1}

# Group items
groups = defaultdict(list)
data = [("Math", 90), ("English", 85), ("Math", 95), ("English", 78)]
for subject, score in data:
    groups[subject].append(score)
# {'Math': [90, 95], 'English': [85, 78]}

# --- Counter: frequency analysis ---
text = "the quick brown fox jumps over the lazy dog the fox"
word_freq = Counter(text.split())
print(word_freq.most_common(3))  # [('the', 3), ('fox', 2), ('quick', 1)]

# Counter arithmetic
c1 = Counter("aabbcc")
c2 = Counter("abcdef")
print(c1 - c2)  # Counter({'a': 1, 'b': 1, 'c': 1})

# --- deque: O(1) operations at both ends ---
dq = deque(maxlen=5)  # Fixed-size buffer
for i in range(10):
    dq.append(i)
print(dq)  # deque([5, 6, 7, 8, 9], maxlen=5)

dq.appendleft(0)  # O(1)
dq.rotate(2)       # Rotate right by 2

# --- namedtuple: lightweight data class ---
Point = namedtuple('Point', ['x', 'y', 'z'])
p = Point(1, 2, 3)
print(p.x, p.y)  # 1 2
```

### Advanced Dictionary Patterns

```python
# Dict comprehension with filtering
data = {"a": 1, "b": 2, "c": 3, "d": 4}
filtered = {k: v for k, v in data.items() if v > 2}
# {'c': 3, 'd': 4}

# Merging dicts (Python 3.9+)
d1 = {"a": 1, "b": 2}
d2 = {"b": 3, "c": 4}
merged = d1 | d2  # {'a': 1, 'b': 3, 'c': 4}

# setdefault
graph = {}
edges = [(1, 2), (1, 3), (2, 3)]
for u, v in edges:
    graph.setdefault(u, []).append(v)
    graph.setdefault(v, []).append(u)

# Sorting dict by value
sorted_dict = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
```

### Sets for Efficient Lookups

```python
# O(1) lookup, intersection, union
s1 = {1, 2, 3, 4, 5}
s2 = {4, 5, 6, 7, 8}

print(s1 & s2)   # Intersection: {4, 5}
print(s1 | s2)   # Union: {1, 2, 3, 4, 5, 6, 7, 8}
print(s1 - s2)   # Difference: {1, 2, 3}
print(s1 ^ s2)   # Symmetric difference: {1, 2, 3, 6, 7, 8}

# Deduplicate while preserving order
items = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
unique = list(dict.fromkeys(items))  # [3, 1, 4, 5, 9, 2, 6]
```

---

## 2. Functions, Decorators & Generators

> 💡 **ELI5 (Explain Like I'm 5):** 
> * **Decorators** are like add-ons for your functions. Imagine a function makes a sandwich. A decorator is a machine that automatically puts the sandwich in a bag and labels it without you having to change the sandwich-making function.
> * **Generators** are like a water filter pitcher. Instead of trying to filter a whole lake at once (which takes too much memory/space), it filters one glass of water at a time as you ask for it.

> 📖 **Big picture:** These three features let you write production-quality Python. **Decorators** wrap functions to add behaviour (retry logic, logging, caching, auth checks) without modifying the function itself — every LLM API call in production uses a retry decorator. **Generators** let you process huge datasets lazily (one item at a time) rather than loading everything into memory — critical when processing millions of documents. **Closures** let functions remember state, enabling factory patterns.

### Lambda & Higher-Order Functions

```python
# Lambda
square = lambda x: x ** 2
add = lambda x, y: x + y

# map, filter, reduce
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))          # [1, 4, 9, 16, 25]
evens = list(filter(lambda x: x % 2 == 0, numbers))   # [2, 4]

from functools import reduce
product = reduce(lambda x, y: x * y, numbers)          # 120

# sorted with key
students = [("Alice", 90), ("Bob", 85), ("Charlie", 92)]
sorted_students = sorted(students, key=lambda s: s[1], reverse=True)
```

### Decorators

```python
import time
import functools

# Basic decorator
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def train_model(epochs):
    time.sleep(0.1)
    return "model trained"

# Decorator with arguments
def retry(max_attempts=3, delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt+1} failed: {e}. Retrying...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=2)
def call_api(url):
    # Make API call
    pass

# Class-based decorator
class CacheResult:
    def __init__(self, func):
        self.func = func
        self.cache = {}
    
    def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = self.func(*args)
        return self.cache[args]

@CacheResult
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# functools.lru_cache (built-in memoization)
@functools.lru_cache(maxsize=128)
def expensive_computation(n):
    return sum(i**2 for i in range(n))
```

### Generators

```python
# Generator function (lazy evaluation)
def read_large_file(file_path):
    """Read file line by line without loading all into memory."""
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip()

# Process 10GB file with constant memory
for line in read_large_file("huge_dataset.txt"):
    process(line)

# Generator expression
squares = (x**2 for x in range(1000000))  # No memory allocation
first_10 = [next(squares) for _ in range(10)]

# Infinite generator
def count(start=0, step=1):
    n = start
    while True:
        yield n
        n += step

# Generator pipeline (composable)
def tokenize(lines):
    for line in lines:
        for token in line.split():
            yield token

def filter_stopwords(tokens, stopwords):
    for token in tokens:
        if token not in stopwords:
            yield token

# Chain generators (memory-efficient pipeline)
lines = read_large_file("corpus.txt")
tokens = tokenize(lines)
filtered = filter_stopwords(tokens, {"the", "a", "is"})
```

### Context Managers

```python
from contextlib import contextmanager

@contextmanager
def timer_context(name="Operation"):
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"{name} took {elapsed:.4f}s")

with timer_context("Training"):
    model.train()

# Context manager for GPU memory
@contextmanager
def torch_inference():
    """Disable gradient computation for inference."""
    import torch
    with torch.no_grad():
        torch.cuda.empty_cache()
        yield
        torch.cuda.empty_cache()
```

---

## 3. Object-Oriented Programming

> 💡 **ELI5 (Explain Like I'm 5):** OOP is like using blueprints to build houses. Instead of building a house from scratch by throwing wood and nails together (procedural code), you create a blueprint (`Class`) that describes what a house has (doors, windows) and what it can do (open garage). Then you can effortlessly build 100 actual houses (`Objects`) from that one blueprint.

> 📖 **Big picture:** OOP is how you structure production AI systems. A well-designed class hierarchy means you can swap out model providers without changing business logic, add new tools to an agent without touching existing code, and write tests without mocking entire systems.
>
> For AI engineers, the most important OOP patterns are:
> - **Abstract base class** — defines a common interface (e.g. `BaseModel.predict()`) so different models (GPT-4, LLaMA, Claude) are interchangeable
> - **Context manager** (`with` statement) — ensures resources (API sessions, database connections) are always cleaned up, even on errors
> - **Dataclass** — clean, type-annotated data containers without boilerplate `__init__`

### Classes & Inheritance

```python
from abc import ABC, abstractmethod

# Abstract base class
class BaseModel(ABC):
    def __init__(self, model_name: str, version: str = "1.0"):
        self.model_name = model_name
        self.version = version
        self._is_loaded = False
    
    @abstractmethod
    def predict(self, input_data):
        """Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def load(self):
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.model_name}, v={self.version})"

# Concrete implementation
class LLMModel(BaseModel):
    def __init__(self, model_name, version="1.0", temperature=0.7):
        super().__init__(model_name, version)
        self.temperature = temperature
        self._model = None
    
    def load(self):
        print(f"Loading {self.model_name}...")
        self._model = f"loaded_{self.model_name}"
        self._is_loaded = True
    
    def predict(self, input_data):
        if not self._is_loaded:
            self.load()
        return f"Prediction for: {input_data}"
    
    @property
    def is_ready(self):
        return self._is_loaded
    
    @staticmethod
    def supported_models():
        return ["gpt-4", "claude-3", "llama-3"]

# Multiple inheritance
class RAGModel(LLMModel):
    def __init__(self, model_name, retriever, **kwargs):
        super().__init__(model_name, **kwargs)
        self.retriever = retriever
    
    def predict(self, query):
        context = self.retriever.search(query)
        prompt = f"Context: {context}\nQuery: {query}"
        return super().predict(prompt)
```

### Dunder (Magic) Methods

```python
class Vector:
    def __init__(self, *components):
        self.components = components
    
    def __repr__(self):
        return f"Vector({', '.join(map(str, self.components))})"
    
    def __len__(self):
        return len(self.components)
    
    def __getitem__(self, index):
        return self.components[index]
    
    def __add__(self, other):
        return Vector(*(a + b for a, b in zip(self.components, other.components)))
    
    def __mul__(self, scalar):
        return Vector(*(c * scalar for c in self.components))
    
    def __eq__(self, other):
        return self.components == other.components
    
    def __hash__(self):
        return hash(self.components)
    
    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

v1 = Vector(1, 2, 3)
v2 = Vector(4, 5, 6)
print(v1 + v2)        # Vector(5, 7, 9)
print(v1 * 2)         # Vector(2, 4, 6)
print(v1.dot(v2))     # 32
```

### Dataclasses

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
    model_name: str
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: List[str] = field(default_factory=list)
    api_key: Optional[str] = field(default=None, repr=False)  # Hidden from repr
    
    def __post_init__(self):
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")

config = ModelConfig("gpt-4", temperature=0.5)
print(config)  # ModelConfig(model_name='gpt-4', max_tokens=4096, temperature=0.5, ...)

@dataclass(frozen=True)  # Immutable
class Embedding:
    text: str
    vector: tuple
    model: str = "text-embedding-3-small"
```

### Design Patterns for AI Systems

```python
# Singleton (for model loading)
class ModelSingleton:
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

# Strategy Pattern (for different LLM providers)
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class OpenAIProvider(LLMProvider):
    def generate(self, prompt):
        return openai_client.chat.completions.create(...)

class AnthropicProvider(LLMProvider):
    def generate(self, prompt):
        return anthropic_client.messages.create(...)

class LLMService:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
    
    def answer(self, question):
        return self.provider.generate(question)

# Switch providers easily
service = LLMService(OpenAIProvider())
service = LLMService(AnthropicProvider())

# Factory Pattern
class ModelFactory:
    _registry = {}
    
    @classmethod
    def register(cls, name):
        def decorator(model_class):
            cls._registry[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def create(cls, name, **kwargs):
        model_class = cls._registry.get(name)
        if not model_class:
            raise ValueError(f"Unknown model: {name}")
        return model_class(**kwargs)

@ModelFactory.register("gpt4")
class GPT4Model:
    def __init__(self, **kwargs):
        self.config = kwargs

model = ModelFactory.create("gpt4", temperature=0.5)
```

---

## 4. File Handling & Data Formats

```python
import json
import csv
import yaml
from pathlib import Path

# --- JSON ---
data = {"model": "gpt-4", "metrics": {"accuracy": 0.95, "latency_ms": 120}}

# Write
with open("config.json", "w") as f:
    json.dump(data, f, indent=2)

# Read
with open("config.json", "r") as f:
    loaded = json.load(f)

# Handle large JSON files (streaming)
import ijson  # pip install ijson
with open("large_file.json", "rb") as f:
    for item in ijson.items(f, "item"):
        process(item)

# --- CSV ---
rows = [["name", "score"], ["Alice", 90], ["Bob", 85]]
with open("scores.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

with open("scores.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row["name"], row["score"])

# --- YAML ---
config = {
    "model": {"name": "llama-3", "params": {"temperature": 0.7}},
    "data": {"path": "/data/train.jsonl", "max_samples": 10000}
}
with open("config.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)

# --- JSONL (JSON Lines) — standard for ML datasets ---
def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)

def write_jsonl(path, records):
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

# --- Pathlib (modern file handling) ---
data_dir = Path("data/processed")
data_dir.mkdir(parents=True, exist_ok=True)

for json_file in data_dir.glob("*.json"):
    print(json_file.stem)  # filename without extension

# Check file exists
if (data_dir / "embeddings.npy").exists():
    import numpy as np
    embeddings = np.load(data_dir / "embeddings.npy")
```

---

## 5. REST APIs & HTTP

> 📖 **Big picture:** Every LLM interaction in production is an HTTP request. OpenAI, Anthropic, Cohere, HuggingFace — they all expose REST APIs. You’ll use `requests` for synchronous calls and `aiohttp` for async. The most important habit: always handle errors explicitly (`raise_for_status()`), always set timeouts, and always have retry logic for transient failures.

```python
import requests

# --- GET request ---
response = requests.get(
    "https://api.example.com/models",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    params={"limit": 10, "status": "active"},
    timeout=30
)
response.raise_for_status()  # Raise on 4xx/5xx
data = response.json()

# --- POST request ---
response = requests.post(
    "https://api.example.com/predict",
    json={"text": "Hello world", "model": "gpt-4"},
    headers={"Authorization": "Bearer YOUR_TOKEN", "Content-Type": "application/json"},
    timeout=60
)

# --- Session (reuse connection) ---
session = requests.Session()
session.headers.update({"Authorization": "Bearer YOUR_TOKEN"})

for i in range(100):
    resp = session.get(f"https://api.example.com/items/{i}")

# --- Error handling ---
def safe_api_call(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limited
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
    raise Exception(f"Failed after {max_retries} attempts")

# --- Streaming response (for LLM APIs) ---
response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    json={"model": "gpt-4", "messages": [...], "stream": True},
    headers={"Authorization": "Bearer KEY"},
    stream=True
)
for chunk in response.iter_lines():
    if chunk:
        data = json.loads(chunk.decode().removeprefix("data: "))
        print(data["choices"][0]["delta"].get("content", ""), end="")
```

---

## 6. Async Python

> 📖 **Big picture:** LLM API calls are I/O-bound — you send a request and wait for the model server to respond. With synchronous code, your program sits idle during that wait. Async Python lets you fire off 100 LLM calls simultaneously and process responses as they arrive. For bulk embedding generation, document classification at scale, or any batch LLM pipeline, async is 10-50× faster than sequential.
>
> **The taxi analogy:** Synchronous code is a taxi driver who takes one passenger, drives them to their destination, comes back, then picks up the next. Async is Uber: take many requests, handle them all concurrently, respond as each completes.

```python
import asyncio
import aiohttp

# Basic async
async def fetch_embedding(session, text):
    async with session.post(
        "https://api.openai.com/v1/embeddings",
        json={"input": text, "model": "text-embedding-3-small"},
        headers={"Authorization": "Bearer KEY"}
    ) as response:
        data = await response.json()
        return data["data"][0]["embedding"]

async def batch_embed(texts):
    """Embed many texts concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_embedding(session, text) for text in texts]
        return await asyncio.gather(*tasks)

# Run
embeddings = asyncio.run(batch_embed(["Hello", "World", "AI"]))

# Semaphore for rate limiting
async def rate_limited_fetch(texts, max_concurrent=10):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_embed(session, text):
        async with semaphore:
            return await fetch_embedding(session, text)
    
    async with aiohttp.ClientSession() as session:
        tasks = [limited_embed(session, t) for t in texts]
        return await asyncio.gather(*tasks)
```

---

## 7. Type Hints & Pydantic

```python
from typing import List, Dict, Optional, Union, Tuple, Callable
from pydantic import BaseModel, Field, validator

# Type hints
def process_batch(
    texts: List[str],
    model: str = "gpt-4",
    max_tokens: Optional[int] = None,
    callback: Optional[Callable[[str], None]] = None
) -> List[Dict[str, Union[str, float]]]:
    ...

# Pydantic models (validation + serialization)
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str = Field(..., min_length=1, max_length=100000)

class ChatRequest(BaseModel):
    model: str = Field(default="gpt-4")
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=128000)
    
    @validator("messages")
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages cannot be empty")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello!"}],
                "temperature": 0.7
            }
        }

# Usage
request = ChatRequest(messages=[ChatMessage(role="user", content="Hello!")])
print(request.model_dump_json(indent=2))
```

---

## 8. Testing & Debugging

```python
import pytest

# Basic test
def test_tokenize():
    assert tokenize("hello world") == ["hello", "world"]
    assert tokenize("") == []
    assert tokenize("word") == ["word"]

# Parametrized tests
@pytest.mark.parametrize("input_text,expected", [
    ("hello world", ["hello", "world"]),
    ("", []),
    ("one", ["one"]),
    ("a  b", ["a", "b"]),
])
def test_tokenize_parametrized(input_text, expected):
    assert tokenize(input_text) == expected

# Fixtures
@pytest.fixture
def model_config():
    return ModelConfig(model_name="test-model", temperature=0.5)

@pytest.fixture
def mock_llm(monkeypatch):
    def mock_generate(prompt):
        return "mocked response"
    monkeypatch.setattr("myapp.llm.generate", mock_generate)

def test_rag_pipeline(model_config, mock_llm):
    result = rag_pipeline("test query", model_config)
    assert "mocked response" in result

# Testing async code
@pytest.mark.asyncio
async def test_async_embed():
    result = await fetch_embedding("test text")
    assert len(result) == 1536

# Debugging with breakpoint
def complex_function(data):
    processed = transform(data)
    breakpoint()  # Python 3.7+ built-in debugger
    return analyze(processed)
```

---

## 9. Package Management

```bash
# Virtual environments
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# pip
pip install torch transformers langchain
pip install -r requirements.txt
pip freeze > requirements.txt

# Poetry (modern dependency management)
poetry init
poetry add torch transformers
poetry install
poetry run python train.py

# Conda (for complex ML dependencies)
conda create -n ml python=3.11
conda activate ml
conda install pytorch torchvision -c pytorch
```

---

## 10. Interview Questions

### Q1: What is the difference between a list and a tuple?
**A**: Lists are mutable (can modify in-place), tuples are immutable. Tuples are hashable (can be dict keys/set elements). Tuples are slightly faster and use less memory. Use tuples for fixed collections (coordinates, function returns, dict keys), lists for dynamic collections.

### Q2: Explain Python's GIL.
**A**: The Global Interpreter Lock allows only one thread to execute Python bytecode at a time. This means CPU-bound multithreaded code doesn't benefit from multiple cores. Solutions: use multiprocessing for CPU-bound work, threading for I/O-bound work (API calls, file I/O), or use numpy/PyTorch (release GIL in C extensions).

### Q3: How do decorators work?
**A**: Decorators are functions that take a function as input and return a modified function. @decorator is syntactic sugar for func = decorator(func). They wrap the original function, adding behavior before/after it executes. Use functools.wraps to preserve the original function's name and docstring.

### Q4: What's the difference between `__str__` and `__repr__`?
**A**: `__repr__` is for developers (unambiguous, ideally eval()-able). `__str__` is for users (readable). `print()` calls `__str__`, falling back to `__repr__`. The REPL shows `__repr__`. Best practice: always implement `__repr__`; implement `__str__` only if you need a different human-readable format.

### Q5: Explain generators and why they're useful for ML pipelines.
**A**: Generators use `yield` to produce values lazily — only compute the next value when requested. Memory-efficient for large datasets (don't load entire dataset into RAM). Can chain generators into pipelines: read_file() → tokenize() → filter() → batch(). Essential for streaming massive training datasets. DataLoaders in PyTorch use a similar concept.

### Q6: What is `*args` and `**kwargs`?
**A**: `*args` collects positional arguments into a tuple. `**kwargs` collects keyword arguments into a dict. Used for flexible function signatures: `def func(*args, **kwargs)` accepts any arguments. Common in decorators and wrapper functions. In ML: `model.forward(**batch)` passes a dict of tensors as keyword arguments.

### Q7: How does Python's garbage collection work?
**A**: Reference counting (primary): objects are freed when ref count drops to 0. Generational GC (backup): detects reference cycles (A→B→A). Three generations — young objects checked frequently, old objects rarely. Manually trigger with `gc.collect()`. In ML: large tensors are freed when no longer referenced — del tensor; torch.cuda.empty_cache().

### Q8: Explain list comprehension vs generator expression.
**A**: List comprehension `[x**2 for x in range(n)]` creates the entire list in memory. Generator expression `(x**2 for x in range(n))` produces values lazily. For large n, generators use O(1) memory vs O(n). Use list comp when you need random access or multiple iterations. Use generator when processing sequentially once.

### Q9: What is a context manager and when should you use one?
**A**: Objects with `__enter__` and `__exit__` methods, used with `with` statement. Guarantees cleanup even if exceptions occur. Use for: file handles, database connections, GPU memory management, locks, temporary directories. Example: `with open('file') as f:` ensures file is closed. Use @contextmanager decorator for simple cases.

### Q10: Explain Python's MRO (Method Resolution Order).
**A**: Python uses C3 linearization to determine which method to call in multiple inheritance. MRO defines the order classes are searched. Check with `ClassName.__mro__`. Follows left-to-right depth-first, but respects "a class cannot precede its parent." Important for understanding how frameworks like PyTorch's nn.Module works with mixins.

### Q11: What are metaclasses?
**A**: Classes of classes — control how classes are created. type is the default metaclass. Use `class Meta(type)` to customize class creation. Practical uses: ORM field registration (Django), abstract method enforcement, singleton pattern, automatic registration. Rarely needed in day-to-day code — prefer class decorators or `__init_subclass__`.

### Q12: How does `@property` work?
**A**: Turns a method into a computed attribute, providing getter/setter/deleter. Allows validation on attribute setting and lazy computation on getting. Example: `model.is_ready` calls a method but looks like attribute access. Useful for: encapsulation, computed properties, validation, backwards compatibility.

### Q13: What is monkey patching?
**A**: Dynamically modifying a module, class, or object at runtime. Example: replacing a function in a library for testing. `module.function = mock_function`. Useful for testing (monkeypatch fixture in pytest) but dangerous in production (fragile, hard to debug). Never use for production code — prefer dependency injection or strategy pattern.

### Q14: Explain the difference between shallow and deep copy.
**A**: Shallow copy (`copy.copy()`, `list.copy()`, `[:]`): creates new container but references same nested objects. Deep copy (`copy.deepcopy()`): recursively copies all nested objects. For ML: shallow copying a list of tensors shares tensor data. Deep copy duplicates everything. Tensor `.clone()` creates a new tensor with same data.

### Q15: How would you make a class iterable?
**A**: Implement `__iter__` (returns iterator) and `__next__` (returns next element, raises StopIteration). Or `__iter__` can yield values directly (generator). In ML: custom Dataset classes implement `__getitem__` and `__len__` for DataLoader iteration.

### Q16: What are slots in Python classes?
**A**: `__slots__ = ('x', 'y')` restricts instance attributes to listed names, preventing dynamic attribute creation. Benefits: 20-30% less memory per instance, slightly faster attribute access. Use for: classes with many instances (data objects, tokens, graph nodes). Don't use when you need dynamic attributes or inheritance flexibility.

### Q17: Explain async/await in Python.
**A**: `async def` creates a coroutine. `await` suspends execution until the awaited coroutine completes, allowing other coroutines to run. The event loop manages scheduling. For ML: concurrent API calls (embedding 1000 texts), parallel file I/O, serving multiple requests. Not useful for CPU-bound work (use multiprocessing).

### Q18: What is Pydantic and why do AI engineers use it?
**A**: Data validation and settings management using Python type annotations. Validates input data at runtime, serializes to/from JSON. AI engineers use it for: API request/response models (FastAPI), configuration management, structured LLM output parsing (LangChain uses Pydantic), data pipeline schemas. Pydantic v2 is written in Rust — very fast.

### Q19: How do you handle environment variables and secrets?
**A**: Never hardcode API keys. Use: (1) `os.environ.get("API_KEY")` for env vars, (2) `.env` files with python-dotenv, (3) Secret managers (AWS Secrets Manager, GCP Secret Manager) for production. In code: validate at startup, fail fast if missing. Pydantic `BaseSettings` can auto-load from env vars and .env files.

### Q20: What is the walrus operator (:=)?
**A**: Assignment expression (Python 3.8+). Assigns and returns value in one expression. Example: `if (n := len(data)) > 10: print(f"Processing {n} items")`. Useful in while loops: `while (chunk := f.read(8192)): process(chunk)`. Reduces redundant function calls.

### Q21: Explain Python's descriptor protocol.
**A**: Objects with `__get__`, `__set__`, `__delete__` methods. When placed as class attributes, they intercept attribute access. `@property` is a descriptor. Django model fields are descriptors. PyTorch `nn.Parameter` uses descriptors. The underlying mechanism for many "magic" behaviors in Python frameworks.

### Q22: How do you profile Python code?
**A**: (1) `time.time()` — basic timing. (2) `cProfile` — function-level profiling. (3) `line_profiler` — line-by-line timing. (4) `memory_profiler` — memory usage. (5) `py-spy` — sampling profiler (no code changes). For ML: PyTorch profiler for GPU operations, `torch.cuda.memory_summary()` for VRAM. Profile before optimizing — don't guess bottlenecks.

### Q23: What is the difference between `is` and `==`?
**A**: `==` checks value equality (`__eq__`). `is` checks identity (same object in memory). Use `is` for: None checks (`if x is None`), singleton comparison. Use `==` for everything else. Gotcha: Python caches small integers (-5 to 256) and short strings, so `a is b` may accidentally work for small values but breaks for large ones.

### Q24: How do dataclasses compare to Pydantic models?
**A**: Dataclasses: standard library, no validation, faster, no serialization. Pydantic: runtime validation, JSON serialization, env var loading, more features. Use dataclasses for internal data structures with trusted data. Use Pydantic for: API boundaries, configuration, LLM output parsing, anything that needs validation. In AI: Pydantic is preferred at system boundaries.

### Q25: Explain Python's asyncio event loop.
**A**: The event loop is the core of async Python — it runs coroutines, handles I/O events, and schedules callbacks. `asyncio.run()` creates a loop, runs a coroutine until complete. `await` yields control back to the loop. Multiple coroutines are interleaved cooperatively. FastAPI uses uvicorn's event loop to handle thousands of concurrent requests.

### Q26: What are abstract base classes and when to use them?
**A**: ABC defines an interface — any subclass MUST implement abstract methods. `from abc import ABC, abstractmethod`. Use when: multiple implementations of the same interface (different LLM providers), enforcing a contract, framework extension points. Don't overuse — Python's duck typing means interfaces are often implicit.

### Q27: How does Python handle memory for large numpy/torch arrays?
**A**: Numpy arrays and PyTorch tensors store data in contiguous C-allocated memory (outside Python's heap). The Python object is a thin wrapper. When the Python object is garbage collected, the C memory is freed. Views (slices) share the same memory. `.copy()` or `.clone()` creates independent copies. GPU tensors live in VRAM — explicitly free with `del tensor; torch.cuda.empty_cache()`.

### Q28: What is structural pattern matching (Python 3.10+)?
**A**: `match`/`case` statements for elegant branching on data structures. Like switch/case but much more powerful — can destructure and match complex patterns. Useful for: parsing ASTs, handling API responses, routing messages. Example: `match event: case {"type": "message", "text": str(t)}: process(t)`.

### Q29: Explain the `__init_subclass__` hook.
**A**: Called when a class is subclassed. Alternative to metaclasses for simpler use cases. Use for: auto-registration (register all subclasses in a registry), validation of subclass attributes, injecting behavior. Example: auto-registering all model types: `class BaseModel: def __init_subclass__(cls): registry[cls.__name__] = cls`.

### Q30: How would you optimize Python code for production ML?
**A**: (1) Use vectorized operations (numpy/torch) instead of loops. (2) Use async for I/O-bound operations (API calls, DB queries). (3) Cache expensive computations (lru_cache, Redis). (4) Profile first — optimize bottlenecks. (5) Use C extensions (Cython, pybind11) for hot loops. (6) Batch API calls. (7) Use generators for memory efficiency. (8) Consider Rust/C++ for critical paths.

---

## 11. Day-to-Day Work Applications

### As an AI/LLM Engineer

**OOP for Production Code**: Design model services with interfaces, factories, and clean architecture. Use abstract classes for swappable LLM providers. Dataclasses and Pydantic for configuration management.

**API Integration**: Every LLM interaction is an API call. Master requests, error handling, retries, and streaming. Build robust clients with exponential backoff and circuit breakers.

**File I/O for Data Pipelines**: Read/write JSONL training data, manage config files (YAML), handle large datasets with generators. Log experiments and metrics to structured files.

**Async for Throughput**: Embed thousands of documents concurrently. Serve multiple users simultaneously with FastAPI. Fan-out to multiple LLM calls in agent systems.

**Testing for Reliability**: Test prompt templates, RAG pipelines, and agent behavior. Mock LLM responses for deterministic tests. Parametrize across different model configs.

---

## 12. Resources

### Excel Curriculum Links
- Python Basics: https://www.youtube.com/watch?v=_uQrJ0TkZlc
- Python OOP: https://www.youtube.com/watch?v=Ej_02ICOIgs
- Python Advanced: https://www.youtube.com/watch?v=p15xzjzR9j0
- Automate the Boring Stuff: https://automatetheboringstuff.com/
- Real Python: https://realpython.com/
- Python Design Patterns: https://refactoring.guru/design-patterns/python
