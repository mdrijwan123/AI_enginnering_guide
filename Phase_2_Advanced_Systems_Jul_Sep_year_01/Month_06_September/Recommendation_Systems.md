# Recommendation Systems — Complete Guide
### Phase 2 Supplementary | ML System Design Reference

> Recommendation systems are asked in almost every ML System Design round. This guide covers classical and modern LLM-powered recsys from first principles.

---

## Part 1 — Why Recommendation Systems Matter in Production AI

If you've ever wondered why Netflix always seems to know what you want to watch next, or why Amazon's "Customers also bought" section is eerily accurate, you've experienced the power of recommendation systems in action. These systems sit at the heart of engagement at every major technology company — and the impact numbers are staggering:

- **Netflix** — 80% of streaming hours come from recommendations, not search
- **YouTube** — 70% of total watch time is driven by the recommendation feed
- **Amazon** — approximately 35% of revenue is attributed to personalised product suggestions
- **Spotify** — Discover Weekly reaches over 30 million users every week and has become a defining product feature in its own right
- **Meta** — recommendations power the news feed, friend suggestions, and ad targeting simultaneously

This makes recommender systems one of the highest-impact ML domains in industry, and almost every ML System Design round will include a question along the lines of *"Design YouTube video recommendations"*, *"Design a news feed ranking system"*, or *"Design Spotify's Discover Weekly."*

**The fundamental challenge:** A user could theoretically see any of Netflix's 17,000 titles or Amazon's 350 million products — but they'll only see around 20 recommendations. The system must identify, with milliseconds of latency, the specific items most likely to be useful or engaging for *this* person *right now*. What makes this genuinely hard is that users have only interacted with a tiny fraction of available items, meaning the interaction matrix is extremely sparse — sometimes less than 0.1% filled. This sparsity is what makes recommendation systems such a rich engineering and machine learning challenge.

---

## Part 2 — Types of Recommendation Approaches

There is no single "best" approach to recommendations — every production recommendation system is a hybrid, blending multiple strategies that each handle different aspects of the problem. Before jumping into code, it's worth building a clear mental model of the four core paradigms and what each one is genuinely good at. The right choice depends on whether you have user history, whether you have item features, and what failure modes you can tolerate.

### 2.1 The 4 Approaches

| Approach | Uses | Pros | Cons |
|---|---|---|---|
| Collaborative Filtering | User-item interactions | Captures latent preferences, no item features needed | Cold start, scalability |
| Content-Based Filtering | Item features | No user history needed, transparent | Feature engineering, no serendipity |
| Knowledge-Based | Domain rules | Explainable, works without data | Manual rules, doesn't personalise |
| **Hybrid** | All of the above | Best quality | Complex engineering |

---

## Part 3 — Collaborative Filtering

Collaborative filtering is the oldest and most widely studied approach to recommendations. The key insight is elegant: you don't need to understand *what* an item is — you only need to observe *who* liked it. If two users have similar interaction patterns, items one user loved are strong candidates for the other. This approach works even for highly subjective preferences like music taste or film genre, where content features alone can't capture why someone would enjoy something.

### 3.1 User-Based Collaborative Filtering

> 💡 **ELI5 (Explain Like I'm 5):**
> This is pure **word of mouth**. If you and a stranger both loved *The Matrix* and *Inception*, and the stranger also gave 5 stars to *Interstellar*, the system recommends *Interstellar* to you. The system knows absolutely nothing about space or sci-fi; it only knows that people with your taste loved it.

The most intuitive form of collaborative filtering: find users whose historical ratings most closely resemble yours, then recommend items they enjoyed that you haven't seen yet. It's essentially "word of mouth" implemented algorithmically at scale.

**Idea:** "Users similar to you liked X, so you might like X."

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# User-item ratings matrix
# Rows = users, Columns = items, 0 = not rated
ratings = np.array([
    [5, 3, 0, 1, 4],   # User 0
    [4, 0, 4, 0, 2],   # User 1
    [0, 1, 5, 2, 0],   # User 2
    [2, 0, 0, 4, 3],   # User 3
])

def user_based_recommendations(user_id: int, ratings: np.ndarray, top_k: int = 3):
    # Compute cosine similarity between users (ignoring zeros)
    # Replace 0 with mean to not penalise missing ratings
    filled = ratings.copy().astype(float)
    for i, row in enumerate(filled):
        non_zero_mean = row[row > 0].mean() if row[row > 0].any() else 0
        row[row == 0] = non_zero_mean
    
    similarities = cosine_similarity(filled)    # shape: (n_users, n_users)
    user_similarities = similarities[user_id]   # similarity to all users
    
    # Find similar users (exclude self)
    similar_users = np.argsort(user_similarities)[::-1][1:]  
    
    # Items not yet seen by target user
    unseen_items = np.where(ratings[user_id] == 0)[0]
    
    scores = {}
    for item in unseen_items:
        numerator = sum(user_similarities[u] * ratings[u][item]
                        for u in similar_users if ratings[u][item] > 0)
        denominator = sum(abs(user_similarities[u])
                          for u in similar_users if ratings[u][item] > 0)
        if denominator > 0:
            scores[item] = numerator / denominator
    
    return sorted(scores, key=scores.get, reverse=True)[:top_k]

print(user_based_recommendations(user_id=0, ratings=ratings))
```

**Why this doesn't scale:** The time complexity is O(n_users² × n_items) — computing pairwise similarities across even 10 million users means 100 trillion pair comparisons. Even with approximate nearest-neighbour shortcuts, user-based CF struggles beyond tens of millions of users. This is the fundamental scalability wall that drove the industry toward model-based approaches like matrix factorisation and, eventually, two-tower neural networks.

---

### 3.2 Matrix Factorization

Matrix factorisation reframes the recommendation problem entirely. Instead of computing explicit similarity scores between users, you ask: *what if every user and every item could be described by a small set of hidden (latent) characteristics?* For a film platform, these latent dimensions might loosely correspond to concepts like "action intensity", "comedic tone", or "critically acclaimed" — though the model learns them purely from data, without predefined labels. A user's tastes are encoded as a vector in this latent space, and an item's characteristics are encoded as another vector. The predicted rating for a user-item pair is the dot product of these vectors — a measure of how well they align in the shared latent space.

The PyTorch implementation below adds bias terms to account for structural patterns: some users consistently give high ratings regardless of content, and some items are universally popular. These biases, combined with the interaction term, give significantly better predictions than a vanilla dot product.

```python
import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    """Classic MF: learns user and item embeddings minimising reconstruction error."""
    
    def __init__(self, n_users: int, n_items: int, n_factors: int = 64):
        super().__init__()
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)
        
        # Bias terms (account for users who rate everything high, items that are always popular)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialise with small random values
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embeddings(user_ids)  # (batch, n_factors)
        item_emb = self.item_embeddings(item_ids)  # (batch, n_factors)
        
        # Dot product = predicted interaction score
        dot_product = (user_emb * item_emb).sum(dim=1)  # (batch,)
        
        # Add biases
        u_bias = self.user_bias(user_ids).squeeze()
        i_bias = self.item_bias(item_ids).squeeze()
        
        return dot_product + u_bias + i_bias + self.global_bias
    
    def recommend(self, user_id: int, top_k: int = 10) -> list:
        """Get top-k recommendations for a user."""
        user_tensor = torch.tensor([user_id])
        all_items = torch.arange(self.item_embeddings.num_embeddings)
        user_repeated = user_tensor.repeat(len(all_items))
        
        with torch.no_grad():
            scores = self.forward(user_repeated, all_items)
        
        return scores.argsort(descending=True)[:top_k].tolist()

# Training
def train_mf(model, train_pairs, n_epochs=20, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()   # For explicit ratings. Use BCE for implicit (clicks)
    
    for epoch in range(n_epochs):
        total_loss = 0
        for user_id, item_id, rating in train_pairs:
            u = torch.tensor([user_id])
            i = torch.tensor([item_id])
            r = torch.tensor([float(rating)])
            
            pred = model(u, i)
            loss = loss_fn(pred, r)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
```

**Where this breaks down:** Pure matrix factorisation is blind to item content — it knows nothing about an item's title, genre, description, or any other feature. This means it cannot recommend a brand-new item that has zero interactions, no matter how compelling that item might be. A film uploaded today with no watches is completely invisible to a collaborative filtering model. This is the cold-start problem, and it is one of the primary motivations for the two-tower architecture we'll see in Part 4.

---

## Part 4 — Two-Tower Model (Modern Production Standard)

**Used by:** YouTube, Google Play, Pinterest, LinkedIn, TikTok

> 💡 **ELI5 (Explain Like I'm 5):**
> Imagine an exclusive **bespoke matchmaking agency**. You have one department ("user tower") whose only job is to interview people and summarise their personality into a profile. You have another department ("item tower") that watches every movie and summarises its vibe. The magic happens when both departments are forced to write their summaries in the exact same language (a shared embedding space). To find a match, you just look for a user profile and a movie profile that are almost identical.

The two-tower model is the dominant architecture for recommendation retrieval at scale, and understanding it deeply is one of the most important technical topics for an AI/ML system design interview. The core idea is elegant: build two separate neural networks — one for users, one for items — and train them to project their respective inputs into a shared embedding space where a user's vector should land close to the items they'd enjoy.

What makes this so powerful in production is the **online/offline decomposition**. Item embeddings can be computed offline for the entire catalogue and stored in a vector index. At serving time, you only need to run a single forward pass through the user tower (milliseconds), then perform an approximate nearest-neighbour search to retrieve candidates. This is how YouTube serves personalised recommendations from 800 million videos with sub-200ms latency — the heavy computation has already been done.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    """
    Two separate neural networks (towers):
    - User tower: encodes user features → user embedding
    - Item tower: encodes item features → item embedding
    
    Recommendation score = cosine_similarity(user_emb, item_emb)
    
    Key advantage: Item embeddings can be pre-computed offline!
    At inference: only encode user, then ANN search in item embedding space.
    """
    
    def __init__(
        self,
        user_feature_dim: int = 128,   # user features (demographics, history, etc.)
        item_feature_dim: int = 256,   # item features (title, category, etc.)
        embedding_dim: int = 64,       # shared embedding space
    ):
        super().__init__()
        
        # User tower: MLP
        self.user_tower = nn.Sequential(
            nn.Linear(user_feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
        )
        
        # Item tower: MLP
        self.item_tower = nn.Sequential(
            nn.Linear(item_feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
        )
    
    def encode_user(self, user_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_tower(user_features), dim=-1)  # L2-normalise
    
    def encode_item(self, item_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_tower(item_features), dim=-1)
    
    def forward(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor
    ) -> torch.Tensor:
        user_emb = self.encode_user(user_features)
        item_emb = self.encode_item(item_features)
        
        # Dot product of normalised vectors = cosine similarity
        return (user_emb * item_emb).sum(dim=-1)  # (batch,)

# Training with in-batch negatives (efficient!)
def compute_in_batch_negative_loss(user_embs: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
    """
    For a batch of (user, positive_item) pairs:
    - Positive: diagonal entries (user_i paired with item_i)  
    - Negatives: all off-diagonal entries (user_i vs item_j where j≠i)
    
    This is called "in-batch negative sampling" — very efficient!
    Effective batch_size negatives per example with no extra forward passes.
    """
    # (batch, batch) similarity matrix
    similarities = torch.matmul(user_embs, item_embs.T)  # cosine sim (both normalised)
    
    # Temperature scaling
    temperature = 0.07
    similarities = similarities / temperature
    
    # Labels: positive pairs are on the diagonal
    labels = torch.arange(len(user_embs))
    
    # Cross-entropy: for each user, predict which item was their positive
    loss = F.cross_entropy(similarities, labels)
    return loss

# Deployment architecture:
"""
Offline (pre-computed nightly):
  1. Encode ALL items → item embeddings
  2. Build ANN index (HNSW in Qdrant/Faiss)
  
Online (real-time per request):
  1. Encode user (user features only) → user embedding  [fast: one forward pass]
  2. ANN query: top-K nearest items in embedding space  [fast: O(log N)]
  
Scale: 1B items × 64-dim fp32 = 256 GB → need Faiss with PQ compression
       1B items × 64-dim int8 = 64 GB  → manageable on GPU cluster
"""
```

### 4.1 Feature Engineering for Two-Tower

The quality of a two-tower model is often more determined by feature engineering than by model architecture. The user tower needs features that capture both stable long-term preferences (genres they consistently enjoy) and real-time context (what time of day is it? what device are they on?). The item tower needs features that describe content and its performance history. Here's a practical breakdown for a video platform:

```python
# User Features:
user_features = {
    # Identity
    "user_id": "embedding lookup (learned)",  
    "age_bucket": "[18-24, 25-34, 35-44, 45+]",
    "country": "embeddings for top-100 countries + 'other'",
    
    # Historical behaviour
    "watched_categories_last_30d": "multi-hot vector, top 50 categories",
    "average_watch_duration": "float, normalised",
    "recently_watched_items": "average of last 10 item embeddings (from item tower)",
    "time_of_day": "sin/cos encoding of hour (cyclical)",
    "device_type": "[mobile, tablet, desktop, TV]",
    
    # Preferences
    "genre_preferences": "average of liked items' genre embeddings",
    "language": "one-hot",
}

# Item Features (for video platform):
item_features = {
    "item_id": "embedding lookup (learned)",
    "title_embedding": "BERT embed of title (768-dim, frozen or fine-tuned)",
    "category": "one-hot or learned embedding",
    "duration": "float, log-normalised",
    "upload_date_days_ago": "log-normalised",
    "creator_id": "embedding lookup",
    "watch_count": "log-normalised",
    "avg_watch_pct": "float 0-1",
    "thumbnail_embedding": "ViT/ResNet embed of thumbnail",
}
```

---

## Part 5 — The Full Recommendation Pipeline

A real-world recommendation system is never a single model — it's a carefully orchestrated multi-stage pipeline. The reason is straightforward: you can't run an expensive, high-quality ranking model over millions of candidates and return results in 200ms. The solution is to progressively filter the candidate pool with increasingly sophisticated (but slower) models at each stage, trading breadth for quality.

Understanding this pipeline is essential for system design interviews. Interviewers want to hear you articulate *why* each stage exists, not just list them.

### 5.1 Production Architecture (Netflix/YouTube Scale)

```
Stage 1: CANDIDATE GENERATION (speed: ~100ms, recall quality, hundreds)
──────────────────────────────────────────────────────────────────────
  Two-tower ANN retrieval → top 500 candidates
  + Trending items (popularity-based)
  + New items (freshness boost)
  + Collaborative filtering recall

Stage 2: FILTERING (business rules)
──────────────────────────────────
  Remove: already watched
  Remove: content policy violations  
  Remove: geography-restricted content
  Apply: diversity constraints (max N per creator)

Stage 3: RANKING (quality: ~10-50ms, top 20-50)
──────────────────────────────────────────────── 
  Wide & Deep model / LambdaRank / XGBoost
  Features: user + item + context (time, device, previous in session)
  Objective: P(watch > 30 seconds | user sees item)
  
Stage 4: RE-RANKING & DIVERSITY
────────────────────────────────
  Ensure diversity (not all same genre/creator)
  Apply A/B test assignment
  Business rules (promoted content, sponsored content placement)
  
Stage 5: SERVE (final N results in order to UI)
```

```python
class RecommendationPipeline:
    def __init__(self, retrieval_model, ranking_model, filters, reranker):
        self.retrieval = retrieval_model
        self.ranker = ranking_model
        self.filters = filters
        self.reranker = reranker
    
    def recommend(self, user: User, n: int = 20, context: dict = None) -> list[Item]:
        # Stage 1: Retrieval
        candidates = self.retrieval.get_candidates(user, top_k=500)
        
        # Stage 2: Filtering
        for f in self.filters:
            candidates = f.apply(candidates, user)
        
        # Stage 3: Ranking
        ranked = self.ranker.rank(user, candidates, context or {})
        
        # Stage 4: Diversity re-rank
        final = self.reranker.diversify(ranked, max_same_creator=3)
        
        return final[:n]
```

---

## Part 6 — Cold Start Problem

> 💡 **ELI5 (Explain Like I'm 5):**
> This is the **"new kid at school"** problem. A brand new movie uploads to YouTube with zero views. Because collaborative filtering relies wholly on user behaviour, the movie is completely invisible to the recommendation engine. To fix this, the system relies on the movie's "resume" (its title, thumbnail, category) to guess who might like it until enough real people have watched it.

The cold start problem is one of the most common discussion topics in recommendation system interviews — and a genuine pain point in production systems. The name captures the dilemma perfectly: your recommendation model learns to surface good items by training on interaction history, but you can only gather that interaction history by serving recommendations in the first place. New users arrive with zero history, and newly published items have zero engagement. Without explicit handling, both would be invisible to a standard collaborative filtering or matrix factorisation model.

There are three distinct cold-start scenarios worth handling differently:
- **New user cold start** — A user just signed up. You have their demographics and maybe device type, but no behavioural history.
- **New item cold start** — An item was just published. It may have rich metadata (title, description, category), but zero interactions.
- **System-level cold start** — You're launching a recommender from scratch with no historical data at all (rare except for startups — solve with content-based methods initially).

### 6.1 Solutions

```python
# NEW USER cold start:
def handle_new_user(user: User) -> list[Item]:
    strategies = {
        "popularity": get_trending_items(time_window="7d", limit=20),
        "onboarding": ask_user_preferences_onboarding(),  # "Select 5 topics you like"
        "demographics": get_popular_for_demographics(user.age, user.location),
        "context": get_popular_in_context(time_of_day=now.hour, device=user.device),
    }
    
    # After 5+ interactions → switch to personalised model
    if user.interaction_count < 5:
        return strategies["popularity"] + strategies["onboarding"]
    return personalised_recommendations(user)

# NEW ITEM cold start:
def handle_new_item(item: Item) -> None:
    """
    Assign item to embedding space using content features alone
    (no interaction data yet).
    """
    # Content-based: use item tower with its features
    item_embedding = item_tower.encode(item.features)  # from two-tower model
    
    # "Warm up" period:
    # 1. Explore phase: show to diverse users, gather initial interactions (10-100)
    # 2. Thompson sampling: balance exploration vs exploitation
    # 3. After enough data: train item tower, refresh embeddings
    
    # Item index in vector DB updated nightly
    vector_db.upsert(item_id=item.id, vector=item_embedding)
```

---

## Part 7 — LLM-Powered Recommendation

### 7.1 When to Use LLMs for Recommendations

Traditional recommendation models and LLMs have genuinely complementary strengths. Understanding when to use each — and when to combine them — is exactly the nuance interviewers are listening for.

**Where traditional recommenders win:** They excel at raw scale. Two-tower models routinely serve from catalogues of hundreds of millions of items with millisecond latency. They also excel at learning subtle implicit preferences from behavioural signals like watch time, scroll depth, and repeat visits. These patterns are difficult to articulate in words, but a neural model trained on interaction data picks them up naturally.

**Where LLMs add genuine value:** They shine when a user's intent can be expressed semantically — *"find a cosy mystery set in Scotland"* or *"something like Inception but shorter."* No embedding model trained purely on click data will understand that query as well as an LLM that comprehends natural language. LLMs are also powerful for the cold start problem (an LLM can reason about a new item's description before any user has interacted with it), for generating natural-language explanations (*"We recommended this because you loved the atmosphere in your last three reads"*), and for conversational refinement where the user gives feedback across multiple turns.

### 7.2 Hybrid Architecture

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage

class LLMHybridRecommender:
    """
    Stage 1: Traditional two-tower retrieval (fast, scalable)
    Stage 2: LLM re-ranking and explanation (quality, understand nuance)
    """
    
    def __init__(self, retriever, llm: ChatOpenAI):
        self.retriever = retriever
        self.llm = llm
    
    def recommend_with_explanation(
        self,
        user_query: str,      # natural language preference
        user_history: list,   # previously interacted items (titles/descriptions)
        n: int = 5
    ) -> list[dict]:
        
        # Stage 1: Retrieve 50 candidates using embedding similarity
        candidates = self.retriever.retrieve(query=user_query, top_k=50)
        
        # Stage 2: LLM re-ranks + explains
        candidate_descriptions = "\n".join(
            [f"{i+1}. {c.title}: {c.description[:200]}" 
             for i, c in enumerate(candidates)]
        )
        user_history_str = ", ".join([item.title for item in user_history[-10:]])
        
        prompt = f"""You are a recommendation expert. Based on the user's request
        and their history, select the {n} best items from the candidates and explain 
        why each is recommended.
        
        User request: "{user_query}"
        User's recent history: {user_history_str}
        
        Candidates:
        {candidate_descriptions}
        
        Return JSON with: 
        [{{"rank": 1, "item_number": N, "title": "...", "reason": "..."}}]"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content  # parse JSON
    
    def conversational_recommend(self, conversation: list[dict]) -> str:
        """Handle multi-turn recommendation conversation."""
        messages = [
            SystemMessage(content="""You are a helpful recommendation assistant.
            Ask clarifying questions to understand preferences.
            Suggest items with clear explanations.""")
        ]
        messages.extend([
            HumanMessage(content=m["content"]) if m["role"] == "user"
            else SystemMessage(content=m["content"])
            for m in conversation
        ])
        return self.llm.invoke(messages).content
```

---

## Part 8 — Evaluation Metrics for RecSys

Evaluating a recommendation system is a two-phase process: offline evaluation during development (fast and cheap) and online evaluation with real traffic (the ground truth, but expensive). The offline metrics below tell you whether your model is *potentially* good; the online metrics tell you whether it *actually* improves the user experience. A model can score well offline but flop in production if it optimises the wrong proxy metric.

```python
import numpy as np

# OFFLINE EVALUATION (fast, during development):

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """P@K: of the top-K recommendations, what fraction are relevant?"""
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k

def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """R@K: of all relevant items, what fraction were in top-K?"""
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant) if relevant else 0

def average_precision(recommended: list, relevant: set) -> float:
    """AP: average of precision values at each relevant item position."""
    precisions = []
    hits = 0
    for i, item in enumerate(recommended):
        if item in relevant:
            hits += 1
            precisions.append(hits / (i + 1))
    return sum(precisions) / len(relevant) if relevant else 0

def mean_average_precision(all_recommended: list, all_relevant: list) -> float:
    """MAP: mean AP across all users."""
    return np.mean([average_precision(r, s) 
                    for r, s in zip(all_recommended, all_relevant)])

def ndcg_at_k(recommended: list, relevant: dict, k: int) -> float:
    """
    NDCG@K: Normalised Discounted Cumulative Gain.
    Accounts for position — items ranked higher contribute more.
    relevant = {item_id: relevance_score} (can be graded, not just binary)
    """
    def dcg(items, scores, k):
        gain = 0
        for i, item in enumerate(items[:k]):
            if item in scores:
                gain += scores[item] / np.log2(i + 2)  # +2 because log2(1) = 0
        return gain
    
    actual_dcg = dcg(recommended, relevant, k)
    ideal_order = sorted(relevant, key=relevant.get, reverse=True)
    ideal_dcg = dcg(ideal_order, relevant, k)
    
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0

# ONLINE EVALUATION (requires live traffic):
"""
A/B Test Metrics:
  - Click-through rate (CTR): % of recommendations that get clicked
  - Engagement rate: watch time, session duration
  - Conversion rate: purchases, sign-ups
  - User satisfaction: explicit ratings, retention
  - Diversity: % unique creators/genres in recommendations
  - Serendipity: how "surprising" but still relevant recommendations are
  - Coverage: % of item catalog ever recommended
"""
```

---

## Part 9 — Interview Questions & Answers

**Q1: Design a recommendation system for YouTube at scale.**

> Walk interviewers through this in clear structured steps. They're evaluating your thought process as much as the final architecture — narrate your reasoning out loud as you go.

```
Step 1: Clarify requirements
  - Scale: 1B users, 800M videos, 500 hours uploaded/minute
  - Latency: < 200ms for 20 recommendations
  - Goals: maximise watch time, session length, user satisfaction

Step 2: High-level architecture
  1. Candidate Generation (Two-Tower + collaborative filtering): 500 items
  2. Filtering (watched, inappropriate): 400 items  
  3. Ranking (deep neural network, 200+ features): 50 items
  4. Re-ranking (diversity, freshness, ads): 20 items

Step 3: Two-Tower model details
  User tower: user_id embedding + watch history avg + demographics + context
  Item tower: video_id embedding + title BERT + category + creator + duration
  Training signal: positive = video watched >50%, negative = in-batch negatives
  
Step 4: Ranking model
  Features: user_item_cross_features (interaction history with similar videos),
            item freshness, watch_probability, like_probability
  Architecture: Deep & Cross Network (DCN) — best for feature interactions at scale
  
Step 5: Online serving
  Pre-computed: all item embeddings in Faiss index (updated nightly)
  Real-time: encode user (10ms) → ANN search (50ms) → rank (100ms) → serve

Step 6: Data pipeline
  Kafka → feature store (Redis for real-time, Feast for batch) → model serving
```

**Q2: What is the cold start problem and how would you solve it for a new item?**

> Cold start for items: a new item has no interaction data, so collaborative filtering can't recommend it. Solutions: (1) **Content-based features**: use the item's metadata (title, description, category, thumbnail) to compute an embedding via the item tower, enabling candidate retrieval immediately. (2) **Exploration policy**: during a "warm-up" phase (first 24-48 hours), show the item to diverse users who have similar taste profiles to early engagers — Thompson sampling balances explore/exploit. (3) **Creator heuristic**: if a known creator uploads a new video, bootstrap the new video's popularity from the creator's historical performance. (4) **Similarity to existing items**: embed the new item and place it in the ANN index — recommendation by neighbour items handles retrieval even before training iterations.

**Q3: Explain the difference between collaborative filtering and content-based filtering.**

> Collaborative filtering: "Users similar to you liked X." Makes recommendations based on the interaction patterns of similar users or the behaviour of users who interacted with similar items. It doesn't need item features — only the interaction matrix. Strength: discovers non-obvious preferences. Weakness: cold start, scalability, popularity bias. Content-based filtering: "You liked item A, so here's item B which has similar features." Uses item features (description, genre, tags) to find similar items. Strength: works without user history, transparent. Weakness: "feature prison" — can only recommend items similar to what you've already seen, no serendipity. Production systems use hybrid: collaborative for personalisation at scale, content-based for cold start and diversity.

**Q4: What loss function do you use for implicit feedback recommendation training?**

> For implicit feedback (clicks, views, purchases — no explicit ratings), binary cross-entropy (BCE) is standard: positive = interacted, negative = sampled non-interacted items. The challenge: you have many more negatives than positives. Solutions: (1) **Negative sampling**: randomly sample N negatives per positive rather than using all. (2) **In-batch negatives**: treat other users' items in the batch as hard negatives (efficient for two-tower models). (3) **Hard negative mining**: sample negatives that are near-positives in embedding space — makes the model learn finer distinctions. For ranking-style losses, you can use BPR (Bayesian Personalised Ranking): directly optimise pairwise preference, i.e., P(score_positive > score_negative).

**Q5: How would you handle the popularity bias in recommendation systems?**

> Popularity bias occurs when popular items dominate recommendations, reducing diversity and hurting discovery of long-tail content. Solutions: (1) **Inverse propensity weighting**: weight the loss function by 1/P(item_shown) — rare items being logged as exposed have a high weight. (2) **Correction in ranking**: add an inverse popularity signal as a feature and penalise over-recommendation of already popular items. (3) **Calibration**: ensure the distribution of categories in recommendations matches the user's historical preferences rather than global popularity. (4) **Explore-exploit**: reserve 10-20% of recommendations for exploration (less popular items), track engagement, and feed results back into the model. (5) **Long-tail boosting**: add a freshness/novelty feature that boosts items not widely seen yet.

---

*Recommendation Systems Guide | Phase 2 Supplementary | Added April 2026*
