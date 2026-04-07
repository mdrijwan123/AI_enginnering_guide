# Week 1: Arrays & Hash Tables
### Phase 1 | Month 1 | April 7–13, 2026

> **Daily plan:** 6–7 AM — solve 1–2 LeetCode problems from this guide. 8–9:30 PM — study the theory sections below.

---

## 🎯 Learning Objectives

By the end of this week you will be able to:
- Explain time/space complexity for all common array operations
- Implement a hash table from scratch and explain collision resolution
- Recognise and apply the key array+hashing patterns: frequency counting, complement lookup, grouping by key
- Solve all 15 LeetCode problems in this guide within the time limits
- Answer any FAANG-level interview question on arrays and hash tables with confidence

---

## Part 1 — Arrays: Deep Dive

> 📖 **Big picture:** An array is the simplest data structure imaginable — a row of boxes in memory, each holding one value, each with a numbered label (index 0, 1, 2...). Almost every algorithm problem in FAANG interviews starts with an array. Before jumping to tricks and patterns, you need to understand *why* certain operations are fast and others are slow — this understanding is what lets you choose the right approach under interview pressure.
>
> Think of memory like a very long street of houses, each with a unique address. An array is like booking a row of consecutive houses so you can find your neighbour by just adding 1 to the address. This consecutive layout is what makes **random access O(1)** — the CPU can jump directly to any index without searching.

### 1.1 What Is an Array?

An array is a **contiguous block of memory** where each element is stored at a fixed offset from the base address.

```
Index:  0     1     2     3     4
Value:  10   25     7    42    18
Addr: 0x100 0x104 0x108 0x10C 0x110  (4 bytes per int)
```

Because of this layout, accessing element at index `i` is always **O(1)** — the CPU computes `base_address + i * element_size` directly.

### 1.2 Time & Space Complexity — Arrays

| Operation | Time | Notes |
|---|---|---|
| Access by index | O(1) | Direct memory address calculation |
| Search (unsorted) | O(n) | Must scan every element |
| Search (sorted) | O(log n) | Binary search |
| Insert at end | O(1) amortized | Dynamic array doubles capacity |
| Insert at position i | O(n) | Shift elements right |
| Delete at end | O(1) | Just decrement length |
| Delete at position i | O(n) | Shift elements left |
| Append to Python list | O(1) amortized | Dynamic array under the hood |

> **Interview trap:** "What's the time complexity of `list.append()` in Python?"  
> Answer: **O(1) amortized**. Occasionally it's O(n) when the array needs to resize, but averaged over many operations it's O(1). This is called **amortised analysis**.

### 1.3 Dynamic Arrays (Python `list`)

Python's `list` is a dynamic array. Internally it:
1. Allocates memory for some initial capacity (e.g. 4 elements)
2. When full, allocates a **new, larger block** (~1.5–2× size)
3. Copies all elements to the new block (O(n) one-time cost)
4. Frees old memory

```python
import sys

a = []
for i in range(10):
    print(f"len={len(a)}, allocated memory={sys.getsizeof(a)} bytes")
    a.append(i)
# Output shows memory jumps at 0, 4, 8, 16, 25... (growth factor ~1.125 in CPython)
```

### 1.4 Key Array Patterns for FAANG

> 📖 **Before the code:** These three array patterns solve a huge portion of interview questions. The underlying thread in all three is: *avoid recomputing the same thing twice*. Prefix sums avoid re-summing ranges. Kadane's avoids trying every subarray. Two-pass avoids using division. Each is a lesson in "precompute now, answer cheaply later."

#### Pattern 1: Prefix Sum
Pre-compute cumulative sums to answer range queries in O(1).

```python
def build_prefix(nums):
    prefix = [0] * (len(nums) + 1)
    for i, n in enumerate(nums):
        prefix[i+1] = prefix[i] + n
    return prefix

# Range sum from index l to r (inclusive)
def range_sum(prefix, l, r):
    return prefix[r+1] - prefix[l]

nums = [2, 3, 1, 5, 4]
prefix = build_prefix(nums)  # [0, 2, 5, 6, 11, 15]
print(range_sum(prefix, 1, 3))  # 3+1+5 = 9
```

**When to use:** "Find sum of all subarrays", "Count subarrays with sum = k", range queries.

#### Pattern 2: Kadane's Algorithm (Maximum Subarray)
Track the best subarray ending at current position.

```python
def maxSubArray(nums):
    max_sum = nums[0]
    current_sum = nums[0]
    
    for n in nums[1:]:
        # Either extend current subarray or start fresh
        current_sum = max(n, current_sum + n)
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
# Answer: 6 (subarray [4,-1,2,1])
```

#### Pattern 3: Two-Pass / Multiple Pointers on Same Array

```python
# Product of Array Except Self — no division, O(n) time O(1) extra space
def productExceptSelf(nums):
    n = len(nums)
    ans = [1] * n
    
    # Left pass: ans[i] = product of everything LEFT of i
    prefix = 1
    for i in range(n):
        ans[i] = prefix
        prefix *= nums[i]
    
    # Right pass: multiply by product of everything RIGHT of i
    suffix = 1
    for i in range(n-1, -1, -1):
        ans[i] *= suffix
        suffix *= nums[i]
    
    return ans
```

---

## Part 2 — Hash Tables: Deep Dive

> 📖 **Big picture:** A hash table is arguably the single most important data structure for coding interviews. It lets you trade a small amount of extra memory for dramatically faster lookups.
>
> Imagine you're working in a library with millions of books. Finding a specific book by scanning every shelf would take forever — O(n). But if you have a catalogue (a hash table), you can look up "Moby Dick" → shelf B12 → walk directly there. That's O(1).
>
> In Python, `dict` and `set` are both hash tables. Whenever you find yourself wanting to ask "have I seen this before?" or "how many times have I seen X?", your instinct should immediately be: **hash table**.

### 2.1 What Is a Hash Table?

A hash table (Python: `dict`, `set`) maps **keys → values** with average O(1) for insert, delete, and lookup.

**How it works:**
1. Compute `hash(key)` → integer
2. Map to a bucket: `bucket_index = hash(key) % capacity`
3. Store key-value pair in that bucket

```
Key "alice"  → hash("alice") = 1234567 → 1234567 % 8 = 7 → bucket 7
Key "bob"    → hash("bob")   = 987654  → 987654  % 8 = 6 → bucket 6
Key "charlie"→ hash("charlie")= 1234575 → 1234575 % 8 = 7 → bucket 7  ← COLLISION
```

### 2.2 Collision Resolution

**Chaining (Python's approach):**
Each bucket holds a linked list of all (key, value) pairs that hash to that bucket.

```
bucket 7: → ("alice", 25) → ("charlie", 30) → None
```

**Open Addressing (Linear Probing):**
If bucket is occupied, try next bucket: `(h + 1) % cap`, `(h + 2) % cap`, etc.

| Method | Pros | Cons |
|---|---|---|
| Chaining | Simple, handles high load | Extra memory for pointers |
| Open Addressing | Cache-friendly, no extra allocs | Clustering degrades to O(n) |

### 2.3 Load Factor & Rehashing

```
load_factor = n_elements / n_buckets
```

- Python dicts resize when load factor > **0.67**
- On resize: allocate ~2× buckets, **rehash all keys** (O(n) one-time)
- Average case is still O(1) due to amortised analysis

### 2.4 Python `dict` vs `collections.defaultdict` vs `Counter`

```python
from collections import defaultdict, Counter

# dict — raises KeyError if key missing
d = {}
d["x"] += 1  # ❌ KeyError

# defaultdict — creates default value if key missing
dd = defaultdict(int)
dd["x"] += 1  # ✅ dd["x"] = 1

dd2 = defaultdict(list)
dd2["x"].append(1)  # ✅ dd2["x"] = [1]

# Counter — dict subclass for counting
c = Counter("aabbbcc")  # Counter({'b': 3, 'c': 2, 'a': 2})
c.most_common(2)         # [('b', 3), ('c', 2)]
c["a"]                   # 2 (no KeyError for missing keys — returns 0)
```

### 2.5 Key Hash Table Patterns

> 📖 **Why these patterns matter:** Most FAANG array problems reduce to one of these three templates. Once you see the pattern name, the code almost writes itself. The goal is to get fast enough at recognising "oh, this is a frequency counting problem" that you start explaining your approach within 30 seconds of reading the problem.

#### Pattern 1: Frequency Counting
```python
def topKFrequent(nums, k):
    count = Counter(nums)
    # Bucket sort: bucket[freq] = [nums with that freq]
    freq_buckets = [[] for _ in range(len(nums) + 1)]
    for num, cnt in count.items():
        freq_buckets[cnt].append(num)
    
    result = []
    for i in range(len(freq_buckets) - 1, 0, -1):
        result.extend(freq_buckets[i])
        if len(result) >= k:
            return result[:k]
```

#### Pattern 2: Complement / Pair Lookup
```python
def twoSum(nums, target):
    seen = {}  # value -> index
    for i, n in enumerate(nums):
        complement = target - n
        if complement in seen:
            return [seen[complement], i]
        seen[n] = i
```

#### Pattern 3: Grouping by Hash Key
```python
def groupAnagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))  # or use frozenset/tuple of char counts
        groups[key].append(s)
    return list(groups.values())

# Faster: use character frequency array as key (no sort)
def groupAnagrams_fast(strs):
    groups = defaultdict(list)
    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        groups[tuple(count)].append(s)
    return list(groups.values())
```

---

## Part 3 — LeetCode Problems (15 Problems, Week 1)

> **Strategy:** Read problem → Think 1–2 mins → If stuck, think about which pattern applies → Code → Check edge cases.

---

### 🟢 Easy

#### Problem 1: Contains Duplicate (LC #217)
**Pattern:** Simple hash set

```python
def containsDuplicate(nums):
    seen = set()
    for n in nums:
        if n in seen:
            return True
        seen.add(n)
    return False
    # One-liner: return len(nums) != len(set(nums))
```
- **Time:** O(n) | **Space:** O(n)
- **Edge cases:** Empty array → False, single element → False

---

#### Problem 2: Valid Anagram (LC #242)
**Pattern:** Frequency counting

```python
def isAnagram(s, t):
    if len(s) != len(t): return False
    return Counter(s) == Counter(t)

# Without Counter (more interview-friendly to write):
def isAnagram_v2(s, t):
    if len(s) != len(t): return False
    count = [0] * 26
    for c in s: count[ord(c) - ord('a')] += 1
    for c in t: count[ord(c) - ord('a')] -= 1
    return all(x == 0 for x in count)
```
- **Time:** O(n) | **Space:** O(1) — only 26 chars

---

#### Problem 3: Two Sum (LC #1)
**Pattern:** Complement lookup

```python
def twoSum(nums, target):
    seen = {}  # num -> index
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
```
- **Time:** O(n) | **Space:** O(n)
- **Follow-up:** What if array is sorted? → Use two pointers, O(1) space

---

### 🟡 Medium

#### Problem 4: Group Anagrams (LC #49)
**Pattern:** Group by hash key

```python
def groupAnagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())
```
- **Time:** O(n × k log k) where k = max string length | **Space:** O(n × k)

---

#### Problem 5: Top K Frequent Elements (LC #347)
**Pattern:** Frequency counting + bucket sort

```python
def topKFrequent(nums, k):
    count = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, freq in count.items():
        buckets[freq].append(num)
    
    result = []
    for i in range(len(buckets) - 1, 0, -1):
        result.extend(buckets[i])
        if len(result) >= k:
            return result[:k]
```
- **Time:** O(n) bucket sort is faster than O(n log n) heap | **Space:** O(n)
- **Alternative:** Use `heapq.nlargest(k, count, key=count.get)` — O(n log k)

---

#### Problem 6: Product of Array Except Self (LC #238)
**Pattern:** Prefix + suffix products (no division!)

```python
def productExceptSelf(nums):
    n = len(nums)
    ans = [1] * n
    prefix = 1
    for i in range(n):
        ans[i] = prefix
        prefix *= nums[i]
    suffix = 1
    for i in range(n - 1, -1, -1):
        ans[i] *= suffix
        suffix *= nums[i]
    return ans
```
- **Time:** O(n) | **Space:** O(1) output array doesn't count
- **Key insight:** ans[i] = (product of everything left of i) × (product of everything right of i)

---

#### Problem 7: Valid Sudoku (LC #36)
**Pattern:** Multiple hash sets

```python
def isValidSudoku(board):
    rows = defaultdict(set)
    cols = defaultdict(set)
    boxes = defaultdict(set)
    
    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val == '.': continue
            
            box_key = (r // 3, c // 3)
            if val in rows[r] or val in cols[c] or val in boxes[box_key]:
                return False
            rows[r].add(val)
            cols[c].add(val)
            boxes[box_key].add(val)
    return True
```
- **Time:** O(81) = O(1) | **Space:** O(81) = O(1) fixed board size

---

#### Problem 8: Encode and Decode Strings (LC #271 — LintCode #659)
**Pattern:** Custom serialisation

```python
def encode(strs):
    # Format: "length#string" for each string
    return ''.join(f'{len(s)}#{s}' for s in strs)

def decode(s):
    result = []
    i = 0
    while i < len(s):
        j = s.index('#', i)
        length = int(s[i:j])
        result.append(s[j+1 : j+1+length])
        i = j + 1 + length
    return result

# Test
strings = ["hello", "world#test", "42"]
encoded = encode(strings)   # "5#hello10#world#test2#42"
decoded = decode(encoded)   # ["hello", "world#test", "42"]
```
- **Why not use `','`?** Strings can contain commas. Length prefix is unambiguous.

---

#### Problem 9: Longest Consecutive Sequence (LC #128)
**Pattern:** Hash set for O(1) lookup

```python
def longestConsecutive(nums):
    num_set = set(nums)
    best = 0
    
    for n in num_set:
        # Only start counting from the beginning of a sequence
        if n - 1 not in num_set:  # n is the start of a sequence
            length = 1
            while n + length in num_set:
                length += 1
            best = max(best, length)
    
    return best
```
- **Time:** O(n) — each number visited at most twice | **Space:** O(n)
- **Key insight:** Skip any `n` where `n-1` exists — it's not a sequence start

---

#### Problem 10: Maximum Subarray (LC #53) — Kadane's
**Pattern:** Kadane's algorithm

```python
def maxSubArray(nums):
    max_sum = current = nums[0]
    for n in nums[1:]:
        current = max(n, current + n)
        max_sum = max(max_sum, current)
    return max_sum
```
- **Time:** O(n) | **Space:** O(1)

---

#### Problem 11: Find All Numbers Disappeared in an Array (LC #448)
**Pattern:** Array as hash map (index marking trick)

```python
def findDisappearedNumbers(nums):
    # Mark visited indices by negating the value
    for n in nums:
        idx = abs(n) - 1
        nums[idx] = -abs(nums[idx])
    
    # Indices that are still positive indicate missing numbers
    return [i + 1 for i in range(len(nums)) if nums[i] > 0]
```
- **Time:** O(n) | **Space:** O(1) — modifies in-place

---

#### Problem 12: Sort Colors (Dutch National Flag — LC #75)
**Pattern:** Three-way partition

```python
def sortColors(nums):
    low, mid, high = 0, 0, len(nums) - 1
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1; mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:  # nums[mid] == 2
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
```
- **Time:** O(n) single pass | **Space:** O(1)

---

#### Problem 13: Subarray Sum Equals K (LC #560)
**Pattern:** Prefix sum + hash map

```python
def subarraySum(nums, k):
    count = 0
    prefix_sum = 0
    prefix_counts = defaultdict(int)
    prefix_counts[0] = 1  # empty subarray has sum 0
    
    for n in nums:
        prefix_sum += n
        # If prefix_sum - k exists, those subarrays sum to k
        count += prefix_counts[prefix_sum - k]
        prefix_counts[prefix_sum] += 1
    
    return count
```
- **Time:** O(n) | **Space:** O(n)
- **Key insight:** sum(i..j) = prefix[j] - prefix[i-1] = k → prefix[i-1] = prefix[j] - k

---

### 🔴 Hard

#### Problem 14: First Missing Positive (LC #41)
**Pattern:** Array as hash map — O(n) time, O(1) space

```python
def firstMissingPositive(nums):
    n = len(nums)
    
    # Step 1: Place each number in its "correct" position
    # nums[i] should equal i+1
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            # Swap nums[i] with nums[nums[i]-1]
            correct_idx = nums[i] - 1
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
    
    # Step 2: Find first position where nums[i] != i+1
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    
    return n + 1  # All 1..n present
```
- **Time:** O(n) | **Space:** O(1)

---

#### Problem 15: Minimum Window Substring (LC #76) — Preview for Week 2
**Pattern:** Sliding window + frequency hash

```python
def minWindow(s, t):
    if not t or not s: return ""
    need = Counter(t)       # chars we still need
    missing = len(t)        # total chars still needed
    best = ""
    left = 0
    
    for right, c in enumerate(s):
        if need[c] > 0:
            missing -= 1
        need[c] -= 1
        
        if missing == 0:  # valid window found
            # Shrink from left
            while need[s[left]] < 0:
                need[s[left]] += 1
                left += 1
            
            window = s[left:right+1]
            if not best or len(window) < len(best):
                best = window
            
            # Break window to continue searching
            need[s[left]] += 1
            missing += 1
            left += 1
    
    return best
```
- **Time:** O(|s| + |t|) | **Space:** O(|s| + |t|)

---

## Part 4 — Interview Q&A (20 Questions)

### Theory Questions

**Q1: What is the time complexity of Python's `in` operator for list vs set/dict?**
> - `x in list` → **O(n)** — scans every element
> - `x in set` → **O(1)** average — hash lookup
> - Never use `in list` in a loop if you care about performance. Convert to set first.

**Q2: Why does Python's dict maintain insertion order since Python 3.7?**
> Python 3.7+ guarantees dict iteration maintains insertion order. Internally, CPython uses a compact hash table that stores indices separately from the key-value store, which naturally preserves insertion order as a side effect.

**Q3: What happens in the worst case for a hash table?**
> If all keys hash to the same bucket (adversarial input or bad hash function), every operation becomes **O(n)**. Python defends against this with hash randomisation (`PYTHONHASHSEED`) for strings — the hash changes every process run. This prevents DoS attacks on web servers.

**Q4: Explain amortised O(1) for `list.append()`.**
> Most appends are O(1). Occasionally (when capacity is full), a resize triggers copying all elements — O(n). But this doubling strategy means the total cost for n appends is O(n), so **per-append cost averages to O(1)** (amortised). *Banker's analogy:* you "save" credits on cheap ops to pay for the expensive resize.

**Q5: What's the difference between `list.sort()` and `sorted()`?**
> - `list.sort()` — in-place, modifies the list, returns `None`
> - `sorted()` — returns a new sorted list, works on any iterable
> Both use **Timsort**: O(n log n) worst case, O(n) best case (nearly sorted data). Stable sort.

**Q6: When would you use an array over a hash map?**
> - When order matters (arrays preserve order naturally)
> - When you need range queries or index-based access
> - When keys are dense integers (use array as direct address table)
> - When memory is critical (arrays have ~5× less overhead than dicts)

**Q7: What's a frozen set and when is it useful?**
> `frozenset` is an immutable set. It's hashable, so it can be used as a dict key or stored inside a set. Useful when you need a set as a hash map key (e.g., grouping anagrams with char sets).

**Q8: How would you implement an LRU Cache?**
> Use an `OrderedDict` (doubly linked list + hash map). The hash map gives O(1) lookup; the linked list gives O(1) insertion/deletion to maintain LRU order. This is LC #146 — know this cold.

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache: return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

### Coding Interview Tips

**Q9: How do you handle edge cases in array problems?**
> Always check: empty array, single element, all duplicates, negative numbers, integer overflow (in Java/C++ but less common in Python), sorted/reverse sorted input.

**Q10: When interviewer says "constant space", what does that mean?**
> O(1) extra space — you cannot use a hash map or extra array proportional to input size. You must use the input array itself (index tricks, negation tricks) or a few variables.

**Q11: What is the "sliding window" pattern and when do you use it?**
> A sliding window maintains a subset of array elements between two pointers (`left`, `right`). Expand right to grow, contract left to shrink. Use when the problem asks for "subarray/substring with property X" and the property is monotonic (adding elements can only make it "more valid" or "less valid").

**Q12: What's the difference between a hash set and hash map in Python?**
> - `set` — stores only keys, O(1) membership test
> - `dict` — stores key-value pairs, O(1) lookup by key
> Internally both use the same hash table mechanism. `set` is essentially a `dict` with no values.

**Q13: Explain the "two-sum" pattern and its variants.**
> Core: use a hash map to track "what have I seen?" Key idea: for pair `(a, b)` where `a + b = target`, when you see `b`, check if `target - b = a` is in the map.
>
> Variants: Three Sum (sort first, then two-pointer for each fixed element), Four Sum (two nested loops + two-pointer).

**Q14: Why is `nums.sort()` usually O(n log n) but counting sort is O(n)?**
> Comparison-based sorting has a theoretical lower bound of O(n log n). Counting sort bypasses this by not doing comparisons — it counts occurrences and reconstructs. Works only for integers in a known range.

**Q15: What does "in-place" mean? Does it matter for interviews?**
> In-place means O(1) extra space. Often interviewers ask for in-place solutions to assess whether you understand memory. In Python, even "in-place" array modifications use O(1) extra space since Python lists are objects with fixed references.

---

### Behavioural / Approach Questions

**Q16: Walk me through your approach when you see a new array problem.**
> 1. Clarify: input constraints (size, value range, sorted?), edge cases, output format
> 2. Think about brute force first (never skip this step — it shows process)
> 3. Identify the pattern: prefix sum? sliding window? hash map?
> 4. State time/space complexity before coding
> 5. Code cleanly, explain as you go
> 6. Test with provided examples, then edge cases

**Q17: How do you choose between sorting and hashing to solve an array problem?**
> - **Hashing:** O(n) time, O(n) space — best when you need O(1) lookups and don't need order
> - **Sorting:** O(n log n) time, O(1) space — best when O(n) space is not allowed or you need sorted order for two-pointer
> When sorting is required, two pointers can often eliminate the need for O(n) hash space.

---

## Part 5 — Common Mistakes to Avoid

| Mistake | Fix |
|---|---|
| Using `list.index()` in a loop (O(n²)) | Convert to `dict` for O(1) lookup |
| Forgetting that `set` has no order | If you need sorted output, sort after |
| Off-by-one in prefix sum (`prefix[r+1] - prefix[l]` vs `prefix[r] - prefix[l-1]`) | Draw it out, test with small example |
| Mutating array while iterating | Iterate over a copy or use two-pass |
| Not handling duplicates in "unique pair" problems | Sort + skip duplicates |
| Python `dict` vs `defaultdict` confusion | Use `defaultdict` when accessing missing keys is expected |
| Integer overflow in Java/C++ (not Python) | In Python ints are arbitrary precision — no overflow |

---

## 📚 Further Resources — What I Can't Cover in Depth Here

### Must Read
- **[NeetCode Arrays & Hashing Playlist](https://www.youtube.com/playlist?list=PLot-Xpze53letfIu9dMzIIO7na_sqvl0y)** — Video walkthroughs of all the patterns above
- **[NeetCode.io Practice](https://neetcode.io/practice)** — Best curated problem set with video solutions

### Go Deeper
- **Python's internal hash table implementation:** Read CPython source `Objects/dictobject.c` if you want to understand the actual implementation
- **"Cracking the Coding Interview" Chapter 1** (Arrays and Strings) by Gayle McDowell — classic FAANG interview prep
- **"Programming Pearls" by Jon Bentley** — Column 1 covers the "use the right data structure" mindset
- **LeetCode Hash Table tag** — sort by "Most Liked" for high-quality problems

### Practice Tracker

| Problem | Difficulty | Pattern | Solved? | Time |
|---|---|---|---|---|
| LC #217 Contains Duplicate | Easy | Hash Set | ⬜ | |
| LC #242 Valid Anagram | Easy | Frequency Count | ⬜ | |
| LC #1 Two Sum | Easy | Complement Lookup | ⬜ | |
| LC #49 Group Anagrams | Medium | Group by Key | ⬜ | |
| LC #347 Top K Frequent | Medium | Bucket Sort | ⬜ | |
| LC #238 Product Except Self | Medium | Prefix/Suffix | ⬜ | |
| LC #36 Valid Sudoku | Medium | Multi Hash Set | ⬜ | |
| LC #128 Longest Consecutive | Medium | Hash Set Start | ⬜ | |
| LC #53 Maximum Subarray | Medium | Kadane's | ⬜ | |
| LC #560 Subarray Sum = K | Medium | Prefix + Hash | ⬜ | |
| LC #41 First Missing Positive | Hard | Array as Hash | ⬜ | |
| LC #76 Min Window Substring | Hard | Sliding Window | ⬜ | |

---

> ✅ **End of Week 1.** When you can consistently solve the Medium problems in < 20 minutes and the Hard problems in < 35 minutes, you're ready for FAANG coding rounds on these topics.

---

## Day-to-Day Work: Where Arrays & Hashing Appear in AI Engineering

```
ARRAYS & HASHING IN YOUR DAILY WORK:

1. EMBEDDING OPERATIONS
   - Embeddings ARE arrays (768-dim vectors)
   - Cosine similarity = dot product of normalised arrays
   - Batch operations on embedding matrices → numpy/torch array ops

2. TOKEN COUNTING & MANAGEMENT
   - Context window = array of token IDs
   - Truncation = array slicing
   - Token budget allocation = prefix sum / greedy allocation

3. DATA DEDUPLICATION (Hash Maps)
   - Dedup documents before RAG indexing → hash content
   - Semantic caching for LLM → hash(messages) → cached response
   - Request ID tracking → hash map for idempotency

4. FREQUENCY COUNTING (Hash Maps)
   - Token frequency analysis for tokenizer training
   - Query analytics: most common user questions (Counter)
   - Error categorisation: group errors by type

5. FEATURE ENCODING
   - One-hot encoding = sparse arrays
   - Feature hashing (hashing trick) for high-cardinality features
   - Product ID → embedding lookup (array indexing)

6. BATCH PROCESSING
   - Process LLM requests in batches → array of prompts
   - Vectorised operations on DataFrames (pandas is arrays under the hood)
```

```python
# Real example: semantic cache using hashing
import hashlib

def get_cache_key(messages: list) -> str:
    """Hash messages for LLM response caching — classic hash map thinking."""
    canonical = str(sorted([(m["role"], m["content"]) for m in messages]))
    return hashlib.sha256(canonical.encode()).hexdigest()
```
