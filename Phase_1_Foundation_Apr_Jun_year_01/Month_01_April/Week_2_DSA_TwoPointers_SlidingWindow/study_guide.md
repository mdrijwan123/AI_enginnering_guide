# Week 2: Two Pointers & Sliding Window
### Phase 1 | Month 1 | April 14–20, 2026

> **Daily plan:** 6–7 AM — 1–2 LeetCode problems. 8–9:30 PM — study the sections below.

---

## 🎯 Learning Objectives

By the end of this week you will be able to:
- Identify when a two-pointer approach applies vs when not
- Distinguish fixed vs variable sliding window problems
- Recognise the "shrink from left when valid" and "expand right then shrink" patterns
- Solve 10 problems (including Hard) on these patterns
- Explain why two pointers is often O(n) instead of O(n²)

---

## Part 1 — Two Pointers

### 1.1 The Core Idea

Two pointers uses two indices (`left`, `right` or `slow`, `fast`) to avoid the O(n²) nested loop.

**When to use:**
- Array/string problems asking for pairs, triplets, or subarrays
- Array is **sorted** (or you can sort it)
- "Find pair with sum X", "remove duplicates in-place", "check palindrome"

### 1.2 Pattern A: Opposite Ends (Left + Right Converging)

Both pointers start at opposite ends and move toward each other.

```
Input:  [1, 3, 5, 7, 9, 11]   target = 12
Left ↑                ↑ Right
       →           ←
```

```python
# Classic: Two Sum (sorted array)
def twoSumSorted(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        s = nums[left] + nums[right]
        if s == target:
            return [left + 1, right + 1]  # 1-indexed
        elif s < target:
            left += 1   # need larger sum → move left right
        else:
            right -= 1  # need smaller sum → move right left
    return []
```

**Why this works:** At any point, if `nums[left] + nums[right] < target`, every pair with `right` is too small (since `left` is already the largest valid left pointer for that `right`). So we can safely increment `left`.

### 1.3 Pattern B: Same Direction (Fast + Slow)

Both pointers move in the same direction, often at different speeds.

**Use case:** In-place removal, linked list cycle detection, finding middle of list.

```python
# Remove duplicates from sorted array in-place
def removeDuplicates(nums):
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1  # new length

# Generalisation: keep at most k duplicates
def removeDuplicatesK(nums, k=2):
    slow = 0
    for fast in range(len(nums)):
        if slow < k or nums[fast] != nums[slow - k]:
            nums[slow] = nums[fast]
            slow += 1
    return slow
```

### 1.4 3Sum — The Classic Hard Two-Pointer Problem

```python
def threeSum(nums):
    nums.sort()
    result = []
    
    for i in range(len(nums) - 2):
        # Skip duplicates for the first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        left, right = i + 1, len(nums) - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s == 0:
                result.append([nums[i], nums[left], nums[right]])
                # Skip duplicates for left and right
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif s < 0:
                left += 1
            else:
                right -= 1
    
    return result
```

- **Time:** O(n²) — O(n log n) sort + O(n²) main loop | **Space:** O(1) ignoring output
- **Key:** Sort first. Fix one element, use two pointers on the rest.

### 1.5 Container with Most Water (LC #11)

```
height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
                                    
         8               8     7   
```

```python
def maxArea(height):
    left, right = 0, len(height) - 1
    max_water = 0
    
    while left < right:
        h = min(height[left], height[right])
        width = right - left
        max_water = max(max_water, h * width)
        
        # Move the shorter side (moving the taller side can never help)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_water
```

- **Greedy insight:** The water is limited by the shorter wall. Moving the shorter wall inward might find a taller wall (and could increase area). Moving the taller wall can only decrease width while the height stays limited.

### 1.6 Trapping Rain Water (LC #42) — Hard

**Approach 1: Two-pointer (O(1) space)**

```python
def trap(height):
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0
    
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    
    return water
```

**Insight:** Water at position `i` = `min(max_left, max_right) - height[i]`. The two-pointer processes each bar once; if `height[left] < height[right]`, then `left_max` is the binding constraint regardless of what's to the right.

---

## Part 2 — Sliding Window

### 2.1 The Core Idea

A sliding window maintains a contiguous subarray between `left` and `right` indices. You slide it across the array to ask: "What's the best window satisfying condition X?"

**Why it's O(n):** Each element is added (right pointer advances) and removed (left pointer advances) **at most once** — O(2n) = O(n).

### 2.2 Pattern A: Fixed-Size Window

Window size `k` is given. Right pointer runs, left follows exactly `k` behind.

```python
# Maximum sum of subarray of size k
def maxSumSubarrayK(nums, k):
    window_sum = sum(nums[:k])
    max_sum = window_sum
    
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]  # slide: add right, remove left
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

### 2.3 Pattern B: Variable-Size Window (Most Common at FAANG)

Expand `right` to include more elements.  
When window becomes **invalid**, advance `left` until valid again.  
Track best valid window.

**Template:**
```python
def slidingWindowTemplate(nums, condition):
    left = 0
    state = {}  # track window state (e.g. freq counts)
    best = 0
    
    for right in range(len(nums)):
        # 1. Add nums[right] to window
        state[nums[right]] = state.get(nums[right], 0) + 1
        
        # 2. Shrink from left while window is invalid
        while not is_valid(state):
            state[nums[left]] -= 1
            if state[nums[left]] == 0:
                del state[nums[left]]
            left += 1
        
        # 3. Window [left..right] is valid — update best
        best = max(best, right - left + 1)
    
    return best
```

### 2.4 Longest Substring Without Repeating Characters (LC #3)

```python
def lengthOfLongestSubstring(s):
    char_idx = {}   # last seen index of each char
    left = 0
    max_len = 0
    
    for right, c in enumerate(s):
        # If c was seen inside current window, move left past it
        if c in char_idx and char_idx[c] >= left:
            left = char_idx[c] + 1
        
        char_idx[c] = right
        max_len = max(max_len, right - left + 1)
    
    return max_len
```

- **Time:** O(n) | **Space:** O(min(n, 26)) for lowercase letters, O(128) for ASCII

### 2.5 Longest Repeating Character Replacement (LC #424)

Find length of longest substring where you can replace at most `k` characters to make all characters the same.

```python
def characterReplacement(s, k):
    count = defaultdict(int)
    max_count = 0  # count of most frequent char in window
    left = 0
    result = 0
    
    for right in range(len(s)):
        count[s[right]] += 1
        max_count = max(max_count, count[s[right]])
        
        # Window size - max_count = chars to replace
        # If replacements needed > k, shrink window
        if (right - left + 1) - max_count > k:
            count[s[left]] -= 1
            left += 1
        
        result = max(result, right - left + 1)
    
    return result
```

**Key insight:** We're looking for the largest window where `window_size - max_freq_char_count <= k`. We only need to shrink by 1 at a time (not re-search max_count) because the result can only grow if max_count grows.

### 2.6 Minimum Window Substring (LC #76) — Hard

Find the smallest window in `s` containing all characters of `t`.

```python
def minWindow(s, t):
    if not t: return ""
    
    need = Counter(t)      # chars needed
    have = defaultdict(int)
    formed = 0             # how many chars have required frequency
    required = len(need)   # how many distinct chars needed
    
    left = 0
    best = ""
    
    for right, c in enumerate(s):
        have[c] += 1
        if c in need and have[c] == need[c]:
            formed += 1
        
        # Try to shrink window while it's valid
        while formed == required:
            window = s[left:right+1]
            if not best or len(window) < len(best):
                best = window
            
            have[s[left]] -= 1
            if s[left] in need and have[s[left]] < need[s[left]]:
                formed -= 1
            left += 1
    
    return best
```

- **Time:** O(|s| + |t|) | **Space:** O(|s| + |t|)

### 2.7 Sliding Window Maximum (LC #239) — Hard (using Deque)

Find the maximum in every window of size `k`.

```python
from collections import deque

def maxSlidingWindow(nums, k):
    dq = deque()  # stores INDICES, values in decreasing order
    result = []
    
    for i, n in enumerate(nums):
        # Remove elements outside current window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove elements smaller than current (they'll never be max)
        while dq and nums[dq[-1]] < n:
            dq.pop()
        
        dq.append(i)
        
        # Window is full: add max to result
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

- **Time:** O(n) — each element added/removed from deque once | **Space:** O(k)
- **Data structure:** Monotonic decreasing deque (front = max of window)

---

## Part 3 — All 10 LeetCode Problems

### Two Pointers Problems

#### Problem 1: Valid Palindrome (LC #125)
```python
def isPalindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True
```
- **Time:** O(n) | **Space:** O(1)

---

#### Problem 2: Two Sum II (LC #167) — Sorted Array
```python
def twoSum(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        s = nums[left] + nums[right]
        if s == target: return [left+1, right+1]
        elif s < target: left += 1
        else: right -= 1
```

---

#### Problem 3: 3Sum (LC #15)
See code in Part 1.4 above.

---

#### Problem 4: Container With Most Water (LC #11)
See code in Part 1.5 above.

---

#### Problem 5: Trapping Rain Water (LC #42) — Hard
See code in Part 1.6 above.

---

### Sliding Window Problems

#### Problem 6: Best Time to Buy and Sell Stock (LC #121)
> Not strictly sliding window but same mental model: track min price seen so far.

```python
def maxProfit(prices):
    min_price = float('inf')
    max_profit = 0
    for p in prices:
        min_price = min(min_price, p)
        max_profit = max(max_profit, p - min_price)
    return max_profit
```
- **Time:** O(n) | **Space:** O(1)

---

#### Problem 7: Longest Substring Without Repeating Chars (LC #3)
See code in Part 2.4 above.

---

#### Problem 8: Longest Repeating Character Replacement (LC #424)
See code in Part 2.5 above.

---

#### Problem 9: Permutation in String (LC #567)
Check if any permutation of `p` exists as a substring of `s`.

```python
def checkInclusion(p, s):
    if len(p) > len(s): return False
    
    p_count = Counter(p)
    window = Counter(s[:len(p)])
    
    if window == p_count: return True
    
    for i in range(len(p), len(s)):
        # Add new char on right
        window[s[i]] += 1
        # Remove old char on left
        old = s[i - len(p)]
        window[old] -= 1
        if window[old] == 0:
            del window[old]
        if window == p_count:
            return True
    
    return False
```
- **Time:** O(26 × n) = O(n) | **Space:** O(26) = O(1)
- **Optimisation:** Track a `matches` count instead of comparing full dicts each step.

---

#### Problem 10: Minimum Window Substring (LC #76) — Hard
See code in Part 2.6 above.

---

## Part 4 — Interview Q&A (15 Questions)

**Q1: When should you use two pointers vs sliding window?**
> - **Two pointers:** You have exactly two elements interacting (a pair, triplet with outer loop). Often used on sorted arrays or when looking for pairs/triples summing to a value.
> - **Sliding window:** You're looking for a contiguous subarray/substring satisfying a condition. The "window" has a clear meaning (frequency of characters, sum of elements, etc.).
> Both are O(n) and both avoid nested loops. The distinction is mostly about framing.

**Q2: Why does sorting enable two pointers?**
> When the array is sorted, moving `left` right always increases the sum, and moving `right` left always decreases it. This creates a deterministic relationship — you always know which pointer to move to get closer to the target. Without sorting, there's no such guarantee.

**Q3: What's the time complexity of the 3Sum algorithm?**
> O(n²). Sorting is O(n log n). The outer loop runs n times, and for each iteration the two-pointer inner loop runs O(n). So O(n log n + n²) = O(n²). 4Sum extends this to O(n³).

**Q4: In the sliding window, when should I use a hash map vs an integer counter?**
> - Hash map: when tracking frequencies of arbitrary characters/elements
> - Single integer: when your condition is just "count of elements exceeding a threshold" (e.g. "at most k distinct characters") — track `distinct_chars` as a counter and adjust as window changes

**Q5: Explain the monotonic deque in Sliding Window Maximum.**
> A monotonic deque maintains elements in a specific order (decreasing here). By removing from the back any element smaller than the new one, we ensure the front always has the current window's maximum. This gives O(1) max query per window step.

**Q6: How would you find the smallest subarray with sum ≥ target?**
```python
def minSubArrayLen(target, nums):
    left = 0
    curr_sum = 0
    min_len = float('inf')
    
    for right in range(len(nums)):
        curr_sum += nums[right]
        while curr_sum >= target:
            min_len = min(min_len, right - left + 1)
            curr_sum -= nums[left]
            left += 1
    
    return 0 if min_len == float('inf') else min_len
```
Pattern: expand right until valid, shrink left while still valid.

**Q7: Why does Valid Palindrome skip non-alphanumeric characters?**
> The problem defines "palindrome" ignoring case and non-alphanumeric characters. This is a practical test of string manipulation — always clarify what "valid palindrome" means in the problem statement.

**Q8: Can sliding window handle negative numbers?**
> Variable-size sliding window assumes that adding elements to the window makes it "more valid" (or that the metric is monotonic with window size). Negative numbers break this — subtracting a negative increases the sum. For negative numbers, use prefix sum + hash map instead (LC #560 Subarray Sum Equals K).

**Q9: What's the difference between LC #3 (Longest Without Repeating) and LC #424 (Longest With K Replacements)?**
> In LC #3, the constraint is strict (no duplicates → 0 replacements). In LC #424, you can "fix" up to k characters. The key insight in #424 is that we only care about the most frequent character in the window — everything else needs replacing.

**Q10: Walk through the "at most K distinct characters" → "exactly K" trick.**
> `exactly(k) = atMost(k) - atMost(k-1)`. This technique converts "exactly k" problems to two "at most k" problems, which are easier to solve with sliding window.
```python
def subarraysWithKDistinct(nums, k):
    def atMost(k):
        count = defaultdict(int)
        left = 0
        result = 0
        for right in range(len(nums)):
            count[nums[right]] += 1
            while len(count) > k:
                count[nums[left]] -= 1
                if count[nums[left]] == 0:
                    del count[nums[left]]
                left += 1
            result += right - left + 1
        return result
    return atMost(k) - atMost(k - 1)
```

**Q11: Give 3 examples of fixed-size window problems.**
> 1. Maximum sum of subarray of size k
> 2. Find all anagrams of a pattern in a string (LC #438)
> 3. Maximum average subarray of size k (LC #643)

**Q12: What are the typical symptoms that tell you a problem needs two pointers?**
> - Sorted array or you can sort without losing information
> - "Find pair / triplet that sums to X"
> - "Remove duplicates in-place" (fast/slow pointer)
> - "Check if string is palindrome"
> - "Find two numbers" or "find adjacent pair"

**Q13: Trapping Rain Water — explain the two-pointer intuition.**
> If `height[left] < height[right]`, then:
> - The water at `left` is determined solely by `left_max` (since `height[right]` ≥ `height[left]`, the right boundary is guaranteed to be at least `left_max`)
> - So we can compute water at `left` and advance it
> The same logic applies symmetrically to the right side.

**Q14: What's the space complexity of the sliding window (minimum window substring)?**
> O(|s| + |t|) — we store character frequencies for both `need` and `have` dictionaries. In practice, bounded by O(128) if ASCII or O(26) if lowercase only, so often cited as O(1).

**Q15: How would you extend 3Sum to find the closest sum (LC #16: 3Sum Closest)?**
```python
def threeSumClosest(nums, target):
    nums.sort()
    closest = float('inf')
    for i in range(len(nums) - 2):
        l, r = i + 1, len(nums) - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if abs(s - target) < abs(closest - target):
                closest = s
            if s < target: l += 1
            elif s > target: r -= 1
            else: return s  # exact match
    return closest
```

---

## Part 5 — Common Patterns Summary

```
Problem Type                    → Pattern
─────────────────────────────────────────────────────
Find pair summing to X          → Two pointers (sorted) or hash map
Find triplet summing to 0       → Sort + outer loop + two pointers
Longest valid substring         → Sliding window (variable, expand right)
All windows of size k           → Sliding window (fixed)
Minimum window containing all   → Sliding window + hash map (frequency match)
Check palindrome                → Two pointers from outside in
Max in every k-window           → Monotonic deque
Remove duplicates in-place      → Fast/slow pointers
```

---

## 📚 Further Resources

### Must Complete This Week
- **[NeetCode Two Pointers Playlist](https://www.youtube.com/playlist?list=PLot-Xpze53lf5C3HSjCnyFghlW0G1HHXo)** — watch after solving each problem
- **[NeetCode Sliding Window Playlist](https://www.youtube.com/playlist?list=PLot-Xpze53leOBgcVsJBEGrHPd_7x_koV)**

### Practice Tracker

| Problem | Difficulty | Pattern | Solved? | Mins |
|---|---|---|---|---|
| LC #125 Valid Palindrome | Easy | Two Ptr | ⬜ | |
| LC #167 Two Sum II | Easy | Two Ptr (sorted) | ⬜ | |
| LC #121 Best Time Buy Stock | Easy | Slide Window | ⬜ | |
| LC #3 Longest Without Repeat | Medium | Slide Window | ⬜ | |
| LC #15 3Sum | Medium | Sort + Two Ptr | ⬜ | |
| LC #11 Container Most Water | Medium | Two Ptr | ⬜ | |
| LC #424 Char Replacement | Medium | Slide Window | ⬜ | |
| LC #567 Permutation in String | Medium | Fixed Window | ⬜ | |
| LC #42 Trapping Rain Water | Hard | Two Ptr | ⬜ | |
| LC #76 Min Window Substring | Hard | Slide Window | ⬜ | |

> ✅ **End of Week 2.** You've now covered all the core array patterns. Week 3 shifts to LLM theory.

---

## Day-to-Day Work: Two Pointers & Sliding Window in AI Engineering

```
WHERE THESE PATTERNS APPEAR AT WORK:

1. SLIDING WINDOW FOR STREAMING DATA
   - Monitor LLM latency: rolling average over last 100 requests
   - Token rate limiting: max N tokens per sliding time window
   - Anomaly detection: compare current window stats vs historical

2. TWO POINTERS FOR DATA PROCESSING  
   - Merge sorted data sources (merge step in merge sort)
   - Diff two sorted lists: old documents vs new → find changes
   - Dedup sorted arrays (in-place, for embedding index maintenance)

3. CONTEXT WINDOW MANAGEMENT (literally a sliding window)
   - Chat history: keep last N messages within token budget
   - Document processing: slide over text with overlap

4. RATE LIMITING
   - Token bucket = sliding window of API calls
```

```python
# Real example: rolling latency monitor using sliding window
from collections import deque
import time

class LatencyMonitor:
    """Sliding window latency tracker — pure two-pointer/deque thinking."""
    def __init__(self, window_seconds=300):
        self.window = window_seconds
        self.latencies = deque()  # (timestamp, latency_ms)
    
    def record(self, latency_ms: float):
        now = time.time()
        self.latencies.append((now, latency_ms))
        # Evict old entries (shrink left pointer)
        while self.latencies and now - self.latencies[0][0] > self.window:
            self.latencies.popleft()
    
    def p50(self):
        values = sorted([l for _, l in self.latencies])
        return values[len(values)//2] if values else 0
    
    def p99(self):
        values = sorted([l for _, l in self.latencies])
        idx = int(len(values) * 0.99)
        return values[idx] if values else 0
```
