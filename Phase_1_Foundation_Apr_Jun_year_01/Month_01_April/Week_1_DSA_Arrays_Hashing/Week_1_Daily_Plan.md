# Week 1 Daily Plan — DSA: Arrays & Hashing
### Phase 1 | April 6–13, 2026

---

## 🎯 Week Goals
- Solve 8–10 LeetCode problems on Arrays & Hashing patterns
- Internalize `HashMap` / `HashSet` as the go-to O(1) lookup tool
- Be able to explain time/space complexity for every solution
- Start the Q&A Chatbot project skeleton this weekend

---

## 📅 Day-by-Day Schedule

### Monday April 6 (Today) — Warm-Up
| Session | Time | Activity |
|---|---|---|
| Morning | 6:00–7:00 AM | LC #217 Contains Duplicate + LC #242 Valid Anagram |
| Evening | 8:00–9:30 PM | Read `Week_1_DSA_Arrays_Hashing/study_guide.md` (Part 1–2) |

**Problems:**
- ✅ / ⏳ [LC #217](https://leetcode.com/problems/contains-duplicate/) — Contains Duplicate (Easy)
- ✅ / ⏳ [LC #242](https://leetcode.com/problems/valid-anagram/) — Valid Anagram (Easy)

**Concepts to understand:**
- Why `set` gives O(1) average lookup
- Hash collision handling (chaining vs open addressing)

---

### Tuesday April 7 — Core Hashing
| Session | Activity |
|---|---|
| 6:00–7:00 AM | LC #1 Two Sum + LC #49 Group Anagrams |
| 8:00–9:30 PM | Study Guide continued (Part 3–4: Two Sum variants, concatenation) |

**Problems:**
- ✅ / ⏳ [LC #1](https://leetcode.com/problems/two-sum/) — Two Sum (Easy)
- ✅ / ⏳ [LC #49](https://leetcode.com/problems/group-anagrams/) — Group Anagrams (Medium)

**Pattern:** `HashMap` storing `{value: index}` — scan once, check complement.

```python
# Two Sum template — commit to memory
def twoSum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        complement = target - n
        if complement in seen:
            return [seen[complement], i]
        seen[n] = i
```

---

### Wednesday April 8 — Top K & Encoding
| Session | Activity |
|---|---|
| 6:00–7:00 AM | LC #347 Top K Frequent Elements |
| 8:00–9:30 PM | LC #271 Encode/Decode Strings + review |

**Problems:**
- ✅ / ⏳ [LC #347](https://leetcode.com/problems/top-k-frequent-elements/) — Top K Frequent (Medium)
- ✅ / ⏳ [LC #271](https://leetcode.com/problems/encode-and-decode-strings/) — Encode/Decode Strings (Medium)

**Pattern:** Bucket sort for Top K (avoid full sort: O(n) vs O(n log n)).

```python
# Bucket Sort top-K — key insight
def topKFrequent(nums, k):
    count = {}
    freq = [[] for _ in range(len(nums) + 1)]  # index = frequency
    for n in nums:
        count[n] = count.get(n, 0) + 1
    for n, c in count.items():
        freq[c].append(n)
    result = []
    for i in range(len(freq)-1, 0, -1):
        for n in freq[i]:
            result.append(n)
            if len(result) == k:
                return result
```

---

### Thursday April 9 — Product & Longest Consecutive
| Session | Activity |
|---|---|
| 6:00–7:00 AM | LC #238 Product of Array Except Self |
| 8:00–9:30 PM | LC #128 Longest Consecutive Sequence |

**Problems:**
- ✅ / ⏳ [LC #238](https://leetcode.com/problems/product-of-array-except-self/) — Product Except Self (Medium)
- ✅ / ⏳ [LC #128](https://leetcode.com/problems/longest-consecutive-sequence/) — Longest Consecutive Sequence (Medium)

**Pattern LC #238:** Prefix products + suffix products, no division, O(n) space optimization.

**Pattern LC #128:** HashSet + only start counting from sequence starts (`n-1 not in set`).

```python
# LC #128 — Only check sequence starts (key trick)
def longestConsecutive(nums):
    num_set = set(nums)
    longest = 0
    for n in num_set:
        if n - 1 not in num_set:   # sequence start
            length = 1
            while n + length in num_set:
                length += 1
            longest = max(longest, length)
    return longest
# O(n) — each number visited at most twice
```

---

### Friday April 10 — Mixed Review
| Session | Activity |
|---|---|
| 6:00–7:00 AM | LC #36 Valid Sudoku |
| 8:00–9:30 PM | Review all week's problems. Re-type solutions from scratch. |

**Problems:**
- ✅ / ⏳ [LC #36](https://leetcode.com/problems/valid-sudoku/) — Valid Sudoku (Medium)

**Pattern:** Nested `defaultdict(set)` for tracking rows, cols, boxes simultaneously.

**Evening review:**
- Can you code Two Sum without notes? ✅ / ❌
- Can you code Top K Frequent from memory? ✅ / ❌
- Can you explain Longest Consecutive O(n) trick? ✅ / ❌

---

### Saturday April 11 — Deep Work Session (3–4 hrs)
| Time | Activity |
|---|---|
| 10:00 AM–12:00 PM | Attempt 2 harder problems: LC #560 Subarray Sum = K + LC #525 Contiguous Array |
| 12:00–1:00 PM | Start Portfolio #1 skeleton: `uv init chatbot && cd chatbot` |
| 1:00–2:00 PM | Set up: FastAPI hello-world + Ollama pull llama3.2:3b + basic LangChain chain |

**Project Skeleton:**
```bash
# Bootstrap Q&A Chatbot project
mkdir ai-chatbot && cd ai-chatbot
python -m venv .venv
pip install langchain langchain-community fastapi uvicorn ollama

# Pull local model
ollama pull llama3.2:3b

# Basic chain test  
python -c "from langchain_community.llms import Ollama; llm = Ollama(model='llama3.2:3b'); print(llm.invoke('What is RAG?'))"
```

**Extra LC (if time allows):**
- ✅ / ⏳ LC #560 — Subarray Sum Equals K (prefix sum + hashmap)
- ✅ / ⏳ LC #525 — Contiguous Array (transform: 0→-1, find equal prefix sums)

---

### Sunday April 12 — Mock Interview Day (3 hrs)
| Time | Activity |
|---|---|
| 10:00–11:00 AM | Timed mock: 45 min, 2 problems (no looking at notes) |
| 11:00 AM–12:00 PM | Review solutions, classify any mistakes (conceptual / implementation / careless) |
| 12:00–1:00 PM | Read `classical_ml_fundamentals.md` sections 1–3 (Linear Regression, Gradient Descent, Bias-Variance) |

**Mock problems (pick two):**
- LC #49 Group Anagrams  
- LC #347 Top K Frequent  
- LC #128 Longest Consecutive  
- LC #238 Product Except Self

**Sunday Q&A drill** (say answers aloud, 2 min each):
1. What is the time/space complexity of a HashMap lookup?
2. Explain the difference between a HashSet and a HashMap.
3. What is the key trick in Longest Consecutive Sequence to achieve O(n)?
4. Why does bucket sort beat heap sort for Top K Frequent when k is close to n?

---

### Monday April 13 — Polish Day / Buffer
- Review any problems below your comfort level
- Fill in the PROGRESS_TRACKER.md with Week 1 completions
- Plan Week 2 (Two Pointers & Sliding Window) — scan study guide

---

## 📊 Week 1 Completion Checklist

### LeetCode Problems
- [ ] LC #217 Contains Duplicate
- [ ] LC #242 Valid Anagram
- [ ] LC #1 Two Sum
- [ ] LC #49 Group Anagrams
- [ ] LC #347 Top K Frequent Elements
- [ ] LC #271 Encode/Decode Strings
- [ ] LC #238 Product of Array Except Self
- [ ] LC #128 Longest Consecutive Sequence
- [ ] LC #36 Valid Sudoku
- [ ] LC #560 Subarray Sum Equals K *(stretch)*
- [ ] LC #525 Contiguous Array *(stretch)*

### Study Materials
- [ ] `Week_1_DSA_Arrays_Hashing/study_guide.md` — full read
- [ ] `Math_Foundations/study_guide.md` — sections 1–2 (Linear Algebra, Calculus)
- [ ] `classical_ml_fundamentals.md` — sections 1–3

### Project
- [ ] Chatbot project folder created with venv
- [ ] FastAPI + Ollama hello-world running locally

---

## 💡 Key Patterns to Remember

| Pattern | When to Use | Complexity Gain |
|---|---|---|
| HashSet | Check membership, deduplicate | O(n) lookup → O(1) |
| HashMap `{val: idx}` | Two Sum style: "find complement" | Eliminates nested loop |
| HashMap `{val: count}` | Frequency tracking | O(n) scan |
| Bucket sort by frequency | Top K (when K small) | O(n) vs O(n log n) |
| `n-1 not in set` | Sequence start detection | Avoids redundant counting |
| Prefix sum + HashMap | Subarray sum = K | O(n) vs O(n²) |

---

## 🔑 Template: Arrays Interview Problem Framework

```
1. Understand: What are we looking for? Single element? Pair? Subarray? Count?
2. Naive: What's the brute-force O(n²) or O(n³) solution?
3. Optimize: Can a HashSet/HashMap reduce to O(n)?
4. Edge cases: Empty array. Single element. All duplicates. Negative numbers.
5. Code: Write clean, readable solution.
6. Test: Walk through 2-3 examples manually.
7. Complexity: State O(n) time, O(n) space — explain both.
```

---

*Week 1 Daily Plan | Phase 1 | April 2026*
