# Dynamic Programming — 1D & 2D Complete Guide
### Phase 2 | July 2026 | Week 1–2 DSA

> **Why DP is the hardest interview topic:** DP problems look impossible until you see the pattern. There is no single DP algorithm — it's a *problem-solving technique*: break a hard problem into subproblems, solve each once, reuse the result. Every DP problem follows the same 4-step recipe.

> 💡 **ELI5 (Explain Like I'm 5):**
> Imagine you're climbing stairs and someone asks: "How many ways can you reach step 10?" Instead of counting all paths from scratch, you realize: to reach step 10, you came from step 9 or step 8. So the answer is ways(9) + ways(8). Build up from step 1, store each answer, and you never recompute anything twice. That's dynamic programming.

---

## The 4-Step DP Recipe (Apply to Every Problem)

```
Step 1: DEFINE the subproblem
         dp[i] = "what does this cell mean?"
         
Step 2: RECURRENCE — how does dp[i] relate to smaller subproblems?
         dp[i] = f(dp[i-1], dp[i-2], ...)

Step 3: BASE CASES — what are the smallest answers you know directly?
         dp[0] = ..., dp[1] = ...

Step 4: ANSWER — which cell contains the final answer?
         return dp[n], dp[n][m], max(dp), etc.
```

---

## Part 1 — 1D Dynamic Programming

### 1.1 Climbing Stairs (The Hello World of DP)
**Problem:** You can climb 1 or 2 stairs at a time. How many ways to reach step n?

```python
def climbStairs(n: int) -> int:
    # Step 1: dp[i] = number of ways to reach step i
    # Step 2: dp[i] = dp[i-1] + dp[i-2]  (came from i-1 via 1 step, or i-2 via 2 steps)
    # Step 3: dp[1] = 1, dp[2] = 2
    # Step 4: return dp[n]

    if n <= 2:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

# Space-optimised (only need last 2):
def climbStairsOptimized(n: int) -> int:
    if n <= 2:
        return n
    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        prev2, prev1 = prev1, prev1 + prev2
    return prev1
```
**Time:** O(n) | **Space:** O(1) optimised

---

### 1.2 House Robber — Classic DP Pattern
**Problem:** Row of houses with money. Can't rob adjacent houses. Max money?

```python
def rob(nums: list[int]) -> int:
    # dp[i] = max money stealing from houses 0..i
    # dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    # "skip house i" vs "rob house i (skip i-1)"
    
    if not nums: return 0
    if len(nums) == 1: return nums[0]
    
    prev2 = nums[0]
    prev1 = max(nums[0], nums[1])
    
    for i in range(2, len(nums)):
        curr = max(prev1, prev2 + nums[i])
        prev2, prev1 = prev1, curr
    
    return prev1

# House Robber II — houses in a circle (rob first XOR last)
def robII(nums: list[int]) -> int:
    def rob_linear(arr):
        prev2, prev1 = 0, 0
        for n in arr:
            prev2, prev1 = prev1, max(prev1, prev2 + n)
        return prev1
    
    return max(nums[0], rob_linear(nums[1:]), rob_linear(nums[:-1]))
```

---

### 1.3 Longest Increasing Subsequence (LIS)
**Problem:** Find length of longest strictly increasing subsequence.

```python
def lengthOfLIS(nums: list[int]) -> int:
    # dp[i] = LIS length ending at index i
    # dp[i] = max(dp[j] + 1) for all j < i where nums[j] < nums[i]
    
    n = len(nums)
    dp = [1] * n  # every element is LIS of length 1
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

# O(n log n) with binary search (patience sorting):
from bisect import bisect_left

def lengthOfLIS_fast(nums: list[int]) -> int:
    tails = []  # tails[i] = smallest tail of IS with length i+1
    for n in nums:
        pos = bisect_left(tails, n)
        if pos == len(tails):
            tails.append(n)
        else:
            tails[pos] = n
    return len(tails)
```
**Time:** O(n²) basic / O(n log n) optimised

---

### 1.4 Coin Change — Unbounded Knapsack Template
**Problem:** Given coins, find minimum coins to make amount.

```python
def coinChange(coins: list[int], amount: int) -> int:
    # dp[i] = min coins to make amount i
    # dp[i] = min(dp[i - coin] + 1) for each valid coin
    # Base: dp[0] = 0
    
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Coin Change II — count number of combinations:
def change(amount: int, coins: list[int]) -> int:
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:          # outer loop: coins (order matters for combinations!)
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    
    return dp[amount]
```

> 💡 **Key insight:** Coin Change = "unbounded knapsack". Each coin can be used unlimited times. If you want *combinations* (order doesn't matter), loop coins on the outside. If you want *permutations* (order matters), loop amounts on outside.

---

### 1.5 Word Break
**Problem:** Given string s and dictionary wordDict, can s be segmented into dict words?

```python
def wordBreak(s: str, wordDict: list[str]) -> bool:
    words = set(wordDict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True  # empty string is breakable
    
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break
    
    return dp[n]
```

---

### 1.6 Decode Ways
**Problem:** '1'->'A', '2'->'B', ..., '26'->'Z'. Count decodings of digit string.

```python
def numDecodings(s: str) -> int:
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1           # empty string = 1 way
    dp[1] = 0 if s[0] == '0' else 1
    
    for i in range(2, n + 1):
        one_digit = int(s[i-1])
        two_digit = int(s[i-2:i])
        
        if one_digit != 0:          # single digit decode
            dp[i] += dp[i-1]
        if 10 <= two_digit <= 26:   # two digit decode
            dp[i] += dp[i-2]
    
    return dp[n]
```

---

### 1.7 Palindromic Substrings / Longest Palindromic Substring

```python
# Count palindromic substrings
def countSubstrings(s: str) -> int:
    count = 0
    n = len(s)
    
    def expand(left, right):
        nonlocal count
        while left >= 0 and right < n and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
    
    for i in range(n):
        expand(i, i)      # odd length
        expand(i, i + 1)  # even length
    
    return count

# Longest Palindromic Substring — DP approach:
def longestPalindrome(s: str) -> str:
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    start, max_len = 0, 1
    
    # Every char is palindrome
    for i in range(n):
        dp[i][i] = True
    
    # Length 2
    for i in range(n - 1):
        if s[i] == s[i+1]:
            dp[i][i+1] = True
            start, max_len = i, 2
    
    # Length 3+
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and dp[i+1][j-1]:
                dp[i][j] = True
                if length > max_len:
                    start, max_len = i, length
    
    return s[start:start + max_len]
```

---

### 1.8 Maximum Product Subarray
```python
def maxProduct(nums: list[int]) -> int:
    # Track both max AND min (negative × negative = positive)
    max_prod = min_prod = result = nums[0]
    
    for n in nums[1:]:
        candidates = (n, max_prod * n, min_prod * n)
        max_prod = max(candidates)
        min_prod = min(candidates)
        result = max(result, max_prod)
    
    return result
```

---

## Part 2 — 2D Dynamic Programming

### 2.1 Unique Paths
**Problem:** Robot in m×n grid. Can only move right/down. Count paths to bottom-right.

```python
def uniquePaths(m: int, n: int) -> int:
    # dp[i][j] = paths to reach cell (i,j)
    # dp[i][j] = dp[i-1][j] + dp[i][j-1]  (came from above or from left)
    
    dp = [[1] * n for _ in range(m)]  # first row & col = 1 (only one path)
    
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
    return dp[m-1][n-1]

# Unique Paths II — with obstacles:
def uniquePathsWithObstacles(grid: list[list[int]]) -> int:
    m, n = len(grid), len(grid[0])
    if grid[0][0] == 1: return 0
    
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1
    
    for i in range(1, m):
        dp[i][0] = 0 if grid[i][0] == 1 else dp[i-1][0]
    for j in range(1, n):
        dp[0][j] = 0 if grid[0][j] == 1 else dp[0][j-1]
    
    for i in range(1, m):
        for j in range(1, n):
            if grid[i][j] == 1:
                dp[i][j] = 0
            else:
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
    return dp[m-1][n-1]
```

---

### 2.2 Longest Common Subsequence (LCS) — Core 2D Pattern
**Problem:** Find length of longest common subsequence of two strings.

```python
def longestCommonSubsequence(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1   # chars match: extend LCS
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])  # skip one char
    
    return dp[m][n]

# Variant: Shortest Common Supersequence
def shortestCommonSupersequence(str1: str, str2: str) -> str:
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Backtrack to build result
    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if str1[i-1] == str2[j-1]:
            result.append(str1[i-1])
            i -= 1; j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            result.append(str1[i-1]); i -= 1
        else:
            result.append(str2[j-1]); j -= 1
    result.extend(str1[:i][::-1])
    result.extend(str2[:j][::-1])
    return ''.join(reversed(result))
```

---

### 2.3 Edit Distance (Levenshtein)
**Problem:** Min operations (insert/delete/replace) to convert word1 → word2.

```python
def minDistance(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases: convert to/from empty string
    for i in range(m + 1): dp[i][0] = i  # delete all of word1
    for j in range(n + 1): dp[0][j] = j  # insert all of word2
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]     # chars match, no cost
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # delete from word1
                    dp[i][j-1],    # insert into word1
                    dp[i-1][j-1]   # replace
                )
    
    return dp[m][n]
```

> 💡 **Real-world use:** Edit distance powers spell checkers, DNA sequence alignment, and git diff. Many LLM tokenizer implementations use variations of this.

---

### 2.4 0/1 Knapsack — The Classic
**Problem:** Items with weight and value. Max value with capacity W. Each item used once.

```python
def knapsack(weights: list[int], values: list[int], W: int) -> int:
    n = len(weights)
    # dp[i][w] = max value using first i items with capacity w
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(W + 1):
            # Don't take item i
            dp[i][w] = dp[i-1][w]
            # Take item i (if it fits)
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])
    
    return dp[n][W]

# Space-optimised 1D version (iterate w in reverse!):
def knapsack_1d(weights, values, W):
    dp = [0] * (W + 1)
    for i in range(len(weights)):
        for w in range(W, weights[i] - 1, -1):  # reverse! prevents reuse
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[W]
```

---

### 2.5 Partition Equal Subset Sum
**Problem:** Can array be split into two subsets with equal sum?

```python
def canPartition(nums: list[int]) -> bool:
    total = sum(nums)
    if total % 2 != 0:
        return False
    
    target = total // 2
    # dp[j] = can we form sum j using some subset?
    dp = [False] * (target + 1)
    dp[0] = True  # empty subset = sum 0
    
    for num in nums:
        for j in range(target, num - 1, -1):  # reverse! 0/1 knapsack
            dp[j] = dp[j] or dp[j - num]
    
    return dp[target]
```

---

### 2.6 Maximal Square
**Problem:** Find the largest square of 1s in a binary matrix.

```python
def maximalSquare(matrix: list[list[str]]) -> int:
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * n for _ in range(m)]
    max_side = 0
    
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    # Largest square ending here = min of 3 neighbours + 1
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                max_side = max(max_side, dp[i][j])
    
    return max_side * max_side
```

---

### 2.7 Burst Balloons (Hard — Interval DP)
**Problem:** Burst balloons to maximise coins. Coins = left × current × right balloon values.

```python
def maxCoins(nums: list[int]) -> int:
    # Add boundaries: [1] + nums + [1]
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    
    # length of interval
    for length in range(2, n):
        for left in range(0, n - length):
            right = left + length
            for k in range(left + 1, right):  # k = last balloon to burst in interval
                dp[left][right] = max(
                    dp[left][right],
                    dp[left][k] + dp[k][right] + nums[left] * nums[k] * nums[right]
                )
    
    return dp[0][n-1]
```

> 📖 **Interval DP pattern:** Think about the *last* element processed, not the first. This transforms a hard combinatorial problem into a clean recurrence.

---

## Part 3 — Advanced Patterns

### 3.1 DP on Strings — Distinct Subsequences
**Problem:** Count distinct subsequences of s that equal t.

```python
def numDistinct(s: str, t: str) -> int:
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = 1  # empty t = 1 subsequence (choose nothing)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j]  # skip s[i]
            if s[i-1] == t[j-1]:
                dp[i][j] += dp[i-1][j-1]  # use s[i] to match t[j]
    
    return dp[m][n]
```

---

### 3.2 Buy/Sell Stock Series (State Machine DP)
```python
# At most 2 transactions (LC 123):
def maxProfit_k2(prices: list[int]) -> int:
    buy1 = buy2 = float('-inf')
    sell1 = sell2 = 0
    
    for p in prices:
        buy1  = max(buy1,  -p)
        sell1 = max(sell1,  buy1  + p)
        buy2  = max(buy2,   sell1 - p)
        sell2 = max(sell2,  buy2  + p)
    
    return sell2

# With cooldown (LC 309):
def maxProfit_cooldown(prices: list[int]) -> int:
    held = float('-inf')   # holding stock
    sold = 0               # just sold (cooldown next)
    rest = 0               # resting
    
    for p in prices:
        prev_held = held
        held = max(held, rest - p)   # buy (from rest state)
        rest = max(rest, sold)        # rest or continue resting
        sold = prev_held + p          # sell
    
    return max(sold, rest)
```

---

## Part 4 — Interview Q&A

### Q1: What's the difference between memoization and tabulation?
**A:**
- **Memoization (top-down):** Recursive + cache. Compute only what's needed. More intuitive but has recursion overhead.
- **Tabulation (bottom-up):** Iterative. Fill table from base cases up. No recursion overhead, typically faster in practice.
- **When to use memoization:** Not all subproblems are needed. When recursion structure is naturally clear.
- **When to use tabulation:** Need all subproblems. Want O(1) space optimisation.

### Q2: How do you identify a DP problem?
**A:** Two signals:
1. **Optimal substructure** — optimal solution is built from optimal solutions to subproblems.
2. **Overlapping subproblems** — the same subproblem is solved multiple times.
Common cue words: "minimum/maximum number of ways", "count distinct ways", "longest/shortest", "can you achieve X".

### Q3: What's the difference between 0/1 knapsack and unbounded knapsack?
**A:** In 0/1, each item used **at most once** → loop weight/capacity in **reverse**. In unbounded, items can be reused → loop **forward**. Example: coin change = unbounded (same coin multiple times), partition subset sum = 0/1 (each number once).

### Q4: Explain LCS and its applications.
**A:** LCS finds longest subsequence common to two sequences. Applications: diff tools (git diff), DNA alignment, plagiarism detection, spell correction. LCS(s, reverse(s)) = longest palindromic subsequence.

### Q5: How would you optimise space in a 2D DP?
**A:** If dp[i][j] only depends on the previous row (i-1), reduce to 1D array. If it depends on dp[i][j-1] as well, need to be careful about update order. For 0/1 knapsack: iterate j in reverse. For unbounded: iterate j forward.

### Q6: What is interval DP? Give an example.
**A:** Interval DP solves problems on subarrays/substrings. State: dp[i][j] = answer for interval [i, j]. Recurrence: try each split point k. Examples: Burst Balloons, Matrix Chain Multiplication, Palindrome Partitioning. Key trick: think about the **last** operation on the interval.

### Q7: Walk me through a tree DP problem.
**A:** House Robber III (rob/no-rob each tree node). State: pair (rob, not_rob) at each node. Post-order traversal: solve children first, then combine.
```python
def rob_tree(root) -> int:
    def dfs(node):
        if not node: return (0, 0)  # (rob, skip)
        left = dfs(node.left)
        right = dfs(node.right)
        # rob this: can't rob children
        rob = node.val + left[1] + right[1]
        # skip this: take best of each child
        skip = max(left) + max(right)
        return (rob, skip)
    return max(dfs(root))
```

### Q8: How does DP differ from greedy?
**A:** Greedy makes locally optimal choice each step without reconsidering. DP explores all subproblems and keeps optimal. Greedy works when local optimum = global optimum (e.g., activity selection). DP is needed when greedy fails (e.g., 0/1 knapsack, coin change with arbitrary denominations).

---

## LeetCode Problem List

| # | Problem | Pattern | Difficulty |
|---|---|---|---|
| 70 | Climbing Stairs | 1D DP | Easy |
| 198 | House Robber | 1D DP | Medium |
| 213 | House Robber II | 1D DP | Medium |
| 300 | Longest Increasing Subsequence | 1D DP | Medium |
| 322 | Coin Change | Unbounded Knapsack | Medium |
| 518 | Coin Change II | Unbounded Knapsack | Medium |
| 139 | Word Break | 1D DP | Medium |
| 91 | Decode Ways | 1D DP | Medium |
| 647 | Palindromic Substrings | Expand Centre | Medium |
| 5 | Longest Palindromic Substring | 2D DP / Expand | Medium |
| 152 | Maximum Product Subarray | 1D DP | Medium |
| 62 | Unique Paths | 2D DP | Medium |
| 63 | Unique Paths II | 2D DP | Medium |
| 1143 | Longest Common Subsequence | 2D DP | Medium |
| 1092 | Shortest Common Supersequence | 2D DP | Hard |
| 72 | Edit Distance | 2D DP | Medium |
| 416 | Partition Equal Subset Sum | 0/1 Knapsack | Medium |
| 221 | Maximal Square | 2D DP | Medium |
| 312 | Burst Balloons | Interval DP | Hard |
| 115 | Distinct Subsequences | DP on Strings | Hard |
| 123 | Best Time to Buy/Sell Stock III | State Machine | Hard |
| 309 | Buy/Sell with Cooldown | State Machine | Medium |
| 337 | House Robber III | Tree DP | Medium |
| 494 | Target Sum | 0/1 Knapsack | Medium |
| 1312 | Min Insertions for Palindrome | 2D DP | Hard |

---

## Further Resources

- **NeetCode DP Playlist** — https://neetcode.io/roadmap (Dynamic Programming section)
- **Aditya Verma DP Playlist (YouTube)** — best visual explanation of knapsack variants
- **"Grokking Dynamic Programming" (Educative.io)** — pattern-based approach
- **MIT 6.006 Lecture 19–21** — formal treatment of DP
