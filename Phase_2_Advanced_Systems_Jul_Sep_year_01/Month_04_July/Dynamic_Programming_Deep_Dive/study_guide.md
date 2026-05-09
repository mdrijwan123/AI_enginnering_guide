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
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# Input:  n = 2
# Output: 2   (1+1 or 2)

# Input:  n = 3
# Output: 3   (1+1+1, 1+2, 2+1)

# Input:  n = 5
# Output: 8

def climbStairsOptimized(n: int) -> int:
    if n <= 2: return n
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
    if not nums: return 0
    if len(nums) == 1: return nums[0]
    prev2 = nums[0]
    prev1 = max(nums[0], nums[1])
    for i in range(2, len(nums)):
        curr = max(prev1, prev2 + nums[i])
        prev2, prev1 = prev1, curr
    return prev1

# Input:  nums = [1, 2, 3, 1]
# Output: 4   (rob house 0 and house 2: 1+3=4)

# Input:  nums = [2, 7, 9, 3, 1]
# Output: 12  (rob house 0,2,4: 2+9+1=12)

def robII(nums: list[int]) -> int:
    def rob_linear(arr):
        prev2, prev1 = 0, 0
        for n in arr:
            prev2, prev1 = prev1, max(prev1, prev2 + n)
        return prev1
    return max(nums[0], rob_linear(nums[1:]), rob_linear(nums[:-1]))

# Input:  nums = [2, 3, 2]
# Output: 3   (circular: can't rob both 0 and 2)

# Input:  nums = [1, 2, 3, 1]
# Output: 4
```

---

### 1.3 Longest Increasing Subsequence (LIS)
**Problem:** Find length of longest strictly increasing subsequence.

```python
def lengthOfLIS(nums: list[int]) -> int:
    n = len(nums)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

# Input:  nums = [10, 9, 2, 5, 3, 7, 101, 18]
# Output: 4   ([2, 3, 7, 101] or [2, 5, 7, 101])

# Input:  nums = [0, 1, 0, 3, 2, 3]
# Output: 4   ([0, 1, 2, 3])

# Input:  nums = [7, 7, 7, 7, 7]
# Output: 1   (strictly increasing, so no pair qualifies)

from bisect import bisect_left

def lengthOfLIS_fast(nums: list[int]) -> int:
    tails = []
    for n in nums:
        pos = bisect_left(tails, n)
        if pos == len(tails): tails.append(n)
        else: tails[pos] = n
    return len(tails)

# Same inputs/outputs as above, O(n log n)
```
**Time:** O(n²) basic / O(n log n) optimised

---

### 1.4 Coin Change — Unbounded Knapsack Template
**Problem:** Given coins, find minimum coins to make amount.

```python
def coinChange(coins: list[int], amount: int) -> int:
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1

# Input:  coins = [1, 5, 6, 9],  amount = 11
# Output: 2   (6+5=11, 2 coins)

# Input:  coins = [1, 2, 5],  amount = 11
# Output: 3   (5+5+1=11)

# Input:  coins = [2],  amount = 3
# Output: -1  (impossible)

def change(amount: int, coins: list[int]) -> int:
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    return dp[amount]

# Input:  amount = 5,  coins = [1, 2, 5]
# Output: 4   ([5], [2+2+1], [2+1+1+1], [1+1+1+1+1])

# Input:  amount = 3,  coins = [2]
# Output: 0   (impossible with only 2s)
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
    dp[0] = True
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break
    return dp[n]

# Input:  s = "leetcode",  wordDict = ["leet", "code"]
# Output: True   ("leet" + "code")

# Input:  s = "applepenapple",  wordDict = ["apple", "pen"]
# Output: True   ("apple" + "pen" + "apple")

# Input:  s = "catsandog",  wordDict = ["cats", "dog", "sand", "and", "cat"]
# Output: False
```

---

### 1.6 Decode Ways
**Problem:** '1'->'A', '2'->'B', ..., '26'->'Z'. Count decodings of digit string.

```python
def numDecodings(s: str) -> int:
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 0 if s[0] == '0' else 1
    for i in range(2, n + 1):
        one_digit = int(s[i-1])
        two_digit = int(s[i-2:i])
        if one_digit != 0: dp[i] += dp[i-1]
        if 10 <= two_digit <= 26: dp[i] += dp[i-2]
    return dp[n]

# Input:  s = "12"
# Output: 2   ("AB" or "L")

# Input:  s = "226"
# Output: 3   ("BZ", "VF", "BBF")

# Input:  s = "06"
# Output: 0   (leading zero → invalid)
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

# Input:  s = "abc"
# Output: 3   ("a","b","c" are palindromes)

# Input:  s = "aaa"
# Output: 6   ("a","a","a","aa","aa","aaa")

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

# Input:  s = "babad"
# Output: "bab" (or "aba")

# Input:  s = "cbbd"
# Output: "bb"
```

---

### 1.8 Maximum Product Subarray
```python
def maxProduct(nums: list[int]) -> int:
    max_prod = min_prod = result = nums[0]
    for n in nums[1:]:
        candidates = (n, max_prod * n, min_prod * n)
        max_prod = max(candidates)
        min_prod = min(candidates)
        result = max(result, max_prod)
    return result

# Input:  nums = [2, 3, -2, 4]
# Output: 6   ([2,3] = 6)

# Input:  nums = [-2, 0, -1]
# Output: 0   (0 is the max subarray product)

# Input:  nums = [-2, 3, -4]
# Output: 24  (all three: -2*3*-4=24)
```

---

## Part 2 — 2D Dynamic Programming

### 2.1 Unique Paths
**Problem:** Robot in m×n grid. Can only move right/down. Count paths to bottom-right.

```python
def uniquePaths(m: int, n: int) -> int:
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]

# Input:  m = 3,  n = 7
# Output: 28

# Input:  m = 3,  n = 2
# Output: 3

def uniquePathsWithObstacles(grid: list[list[int]]) -> int:
    m, n = len(grid), len(grid[0])
    if grid[0][0] == 1: return 0
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1
    for i in range(1, m): dp[i][0] = 0 if grid[i][0] == 1 else dp[i-1][0]
    for j in range(1, n): dp[0][j] = 0 if grid[0][j] == 1 else dp[0][j-1]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = 0 if grid[i][j] == 1 else dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]

# Input:  grid = [[0,0,0],[0,1,0],[0,0,0]]
# Output: 2   (obstacle at center blocks one path)
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

# Input:  text1 = "abcde",  text2 = "ace"
# Output: 3   ("ace" is the LCS)

# Input:  text1 = "abc",  text2 = "abc"
# Output: 3

# Input:  text1 = "abc",  text2 = "def"
# Output: 0   (no common subsequence)

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
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

# Input:  word1 = "horse",  word2 = "ros"
# Output: 3
# Why:    horse → rorse (replace h→r) → rose (remove r) → ros (remove e)

# Input:  word1 = "intention",  word2 = "execution"
# Output: 5

# Input:  word1 = "",  word2 = "a"
# Output: 1   (insert 'a')
```

> 💡 **Real-world use:** Edit distance powers spell checkers, DNA sequence alignment, and git diff. Many LLM tokenizer implementations use variations of this.

---

### 2.4 0/1 Knapsack — The Classic
**Problem:** Items with weight and value. Max value with capacity W. Each item used once.

```python
def knapsack(weights: list[int], values: list[int], W: int) -> int:
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])
    return dp[n][W]

# Input:  weights=[2,3,4,5], values=[3,4,5,6], W=8
# Output: 10  (items with weight 3+5=8, value 4+6=10)

# Input:  weights=[1,3,4,5], values=[1,4,5,7], W=7
# Output: 9   (items weight 3+4=7, value 4+5=9)

def knapsack_1d(weights, values, W):
    dp = [0] * (W + 1)
    for i in range(len(weights)):
        for w in range(W, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[W]
```

---

### 2.5 Partition Equal Subset Sum
**Problem:** Can array be split into two subsets with equal sum?

```python
def canPartition(nums: list[int]) -> bool:
    total = sum(nums)
    if total % 2 != 0: return False
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    return dp[target]

# Input:  nums = [1, 5, 11, 5]
# Output: True   ([1,5,5] = 11 = [11])

# Input:  nums = [1, 2, 3, 5]
# Output: False  (no equal partition exists)
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
                if i == 0 or j == 0: dp[i][j] = 1
                else: dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                max_side = max(max_side, dp[i][j])
    return max_side * max_side

# Input:  matrix = [["1","0","1","0","0"],
#                   ["1","0","1","1","1"],
#                   ["1","1","1","1","1"],
#                   ["1","0","0","1","0"]]
# Output: 4   (2x2 square found)

# Input:  matrix = [["0","1"],["1","0"]]
# Output: 1
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

## Part 5 — Hard DP Problems (Gist Q265–282)

### Q266. Wildcard Matching (LC #44) — Hard

> `?` matches any single character, `*` matches any sequence (including empty).

```python
def isMatch_wildcard(s: str, p: str) -> bool:
    m, n = len(s), len(p)
    # dp[i][j] = does s[:i] match p[:j]?
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # '*' in pattern can match empty string
    for j in range(1, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-1]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                # '*' matches empty (dp[i][j-1]) OR matches s[i] (dp[i-1][j])
                dp[i][j] = dp[i][j-1] or dp[i-1][j]
            elif p[j-1] == '?' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]

    return dp[m][n]

# s="adceb", p="*a*b" → True
# s="acdcb", p="a*c?b" → False
```
**Time:** O(m×n) | **Space:** O(m×n)

---

### Q267. Regex Matching with `.` and `*` (LC #10) — Hard

> `.` matches any single character, `*` matches zero or more of the preceding element.

```python
def isMatch_regex(s: str, p: str) -> bool:
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # Pattern like "a*b*c*" can match empty string
    for j in range(2, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                # Zero occurrences of p[j-2]
                dp[i][j] = dp[i][j-2]
                # One or more: p[j-2] must match s[i-1]
                if p[j-2] == '.' or p[j-2] == s[i-1]:
                    dp[i][j] = dp[i][j] or dp[i-1][j]
            elif p[j-1] == '.' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]

    return dp[m][n]

# s="aa", p="a*"  → True
# s="ab", p=".*"  → True
# s="aab", p="c*a*b" → True
```

> 💡 **Key difference from wildcard:** `*` in regex refers to the *preceding element*, not itself. `a*` means zero or more `a`s. So when you see `*`, look back one character in the pattern.

---

### Q270 & Q271. Word Break II (LC #140) — Hard

> Return all ways to segment s into dictionary words.

```python
from functools import lru_cache

def wordBreak_II(s: str, wordDict: list[str]) -> list[str]:
    words = set(wordDict)

    @lru_cache(None)
    def dp(start):
        """Return list of all valid sentences from s[start:]."""
        if start == len(s):
            return [""]
        result = []
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in words:
                for rest in dp(end):
                    result.append(word + (" " + rest if rest else ""))
        return result

    return dp(0)

# s="catsanddog", wordDict=["cat","cats","and","sand","dog"]
# → ["cats and dog", "cat sand dog"]
```

---

### Q272. Interleaving String (LC #97) — Medium/Hard

> Check if s3 is formed by interleaving s1 and s2 in order.

```python
def isInterleave(s1: str, s2: str, s3: str) -> bool:
    m, n = len(s1), len(s2)
    if m + n != len(s3):
        return False
    # dp[i][j] = can s3[:i+j] be formed by s1[:i] and s2[:j]?
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (
                (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or
                (dp[i][j-1] and s2[j-1] == s3[i+j-1])
            )

    return dp[m][n]

# s1="aab", s2="axy", s3="aaxaby" → True
```

---

### Q273. Longest Arithmetic Subsequence (LC #1027)

```python
def longestArithSeqLength(nums: list[int]) -> int:
    # dp[i][diff] = length of longest arithmetic subsequence ending at i with difference diff
    from collections import defaultdict
    n = len(nums)
    dp = [defaultdict(int) for _ in range(n)]
    result = 2

    for i in range(1, n):
        for j in range(i):
            diff = nums[i] - nums[j]
            dp[i][diff] = dp[j][diff] + 1
            result = max(result, dp[i][diff] + 1)

    return result

# [3,6,9,12] → 4 (diff=3)
# [9,4,7,2,10] → 3 ([4,7,10], diff=3)
```
**Time:** O(n²) | **Space:** O(n²)

---

### Q275. Weighted Job Scheduling (LC #1235)

> Jobs with [start, end, profit]. Pick non-overlapping jobs to maximize profit.

```python
from bisect import bisect_right

def jobScheduling(startTime, endTime, profit) -> int:
    jobs = sorted(zip(startTime, endTime, profit), key=lambda x: x[1])
    n = len(jobs)
    # dp[i] = max profit using first i jobs (sorted by end time)
    dp = [0] * (n + 1)
    end_times = [0] + [j[1] for j in jobs]

    for i in range(1, n + 1):
        start, end, p = jobs[i-1]
        # Find latest job that ends <= start of current job
        j = bisect_right(end_times, start, 0, i) - 1
        # Take current job OR skip it
        dp[i] = max(dp[i-1], dp[j] + p)

    return dp[n]

# startTime=[1,2,3,3], endTime=[3,4,5,6], profit=[50,10,40,70] → 120
```

---

### Q277. Matrix Chain Multiplication — Interval DP

> Find minimum cost to multiply a chain of matrices.

```python
def matrixChainOrder(dims: list[int]) -> int:
    """
    dims[i-1] x dims[i] = dimensions of matrix i.
    Cost of multiplying A(p×q) × B(q×r) = p*q*r operations.
    """
    n = len(dims) - 1   # number of matrices
    # dp[i][j] = min cost to multiply matrices i..j
    dp = [[0] * n for _ in range(n)]

    for length in range(2, n + 1):      # chain length
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):       # split point
                cost = dp[i][k] + dp[k+1][j] + dims[i] * dims[k+1] * dims[j+1]
                dp[i][j] = min(dp[i][j], cost)

    return dp[0][n-1]

# dims = [10,30,5,60]  → matrices: 10×30, 30×5, 5×60
# Optimal: (A×B)×C → 10*30*5 + 10*5*60 = 1500+3000 = 4500
```

---

### Q278. Boolean Parenthesization

> Count ways to parenthesize expression of T/F and &, |, ^ to get True.

```python
def countWays(expr: str) -> int:
    symbols = expr[0::2]   # T/F at even indices
    operators = expr[1::2] # &/|/^ at odd indices
    n = len(symbols)

    # dp_T[i][j] = ways to make symbols[i..j] True
    # dp_F[i][j] = ways to make symbols[i..j] False
    dp_T = [[0]*n for _ in range(n)]
    dp_F = [[0]*n for _ in range(n)]

    for i in range(n):
        dp_T[i][i] = 1 if symbols[i] == 'T' else 0
        dp_F[i][i] = 1 if symbols[i] == 'F' else 0

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                op = operators[k]
                lt, lf = dp_T[i][k], dp_F[i][k]
                rt, rf = dp_T[k+1][j], dp_F[k+1][j]
                if op == '&':
                    dp_T[i][j] += lt * rt
                    dp_F[i][j] += lt*rf + lf*rt + lf*rf
                elif op == '|':
                    dp_T[i][j] += lt*rt + lt*rf + lf*rt
                    dp_F[i][j] += lf * rf
                elif op == '^':
                    dp_T[i][j] += lt*rf + lf*rt
                    dp_F[i][j] += lt*rt + lf*rf

    return dp_T[0][n-1]
```

---

### Q279. Longest Increasing Path in a Matrix (LC #329)

```python
from functools import lru_cache

def longestIncreasingPath(matrix: list[list[int]]) -> int:
    m, n = len(matrix), len(matrix[0])
    DIRS = [(0,1),(0,-1),(1,0),(-1,0)]

    @lru_cache(None)
    def dfs(r, c):
        best = 1
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and matrix[nr][nc] > matrix[r][c]:
                best = max(best, 1 + dfs(nr, nc))
        return best

    return max(dfs(r, c) for r in range(m) for c in range(n))

# [[9,9,4],[6,6,8],[2,1,1]] → 4 ([1,2,6,9])
```
**Time:** O(m×n) with memoization | Each cell computed once.

---

### Q280. Cherry Pickup II (LC #1463) — 2-Agent Grid DP

> Two robots start at top-left and top-right of a grid. Both move down simultaneously. Maximize total cherries collected.

```python
def cherryPickup(grid: list[list[int]]) -> int:
    m, n = len(grid), len(grid[0])
    from functools import lru_cache

    @lru_cache(None)
    def dp(row, col1, col2):
        """Both robots at same row. col1=robot1 col, col2=robot2 col."""
        if col1 < 0 or col1 >= n or col2 < 0 or col2 >= n:
            return float('-inf')

        cherries = grid[row][col1]
        if col1 != col2:
            cherries += grid[row][col2]   # don't double-count same cell

        if row == m - 1:
            return cherries

        best = float('-inf')
        for d1 in [-1, 0, 1]:
            for d2 in [-1, 0, 1]:
                best = max(best, dp(row + 1, col1 + d1, col2 + d2))

        return cherries + best

    return dp(0, 0, n - 1)

# grid=[[3,1,1],[2,5,1],[1,5,5],[2,1,1]] → 24
```

---

### Q281. Maximum Subarray Sum With One Deletion (LC #1186)

```python
def maximumSum(arr: list[int]) -> int:
    n = len(arr)
    # no_del[i] = max subarray sum ending at i with 0 deletions
    # one_del[i] = max subarray sum ending at i with 1 deletion used
    no_del = arr[0]
    one_del = 0
    result = arr[0]

    for i in range(1, n):
        one_del = max(one_del + arr[i], no_del)   # delete arr[i] or extend with deletion
        no_del = max(no_del + arr[i], arr[i])      # extend or restart
        result = max(result, no_del, one_del)

    return result

# [1,-2,0,3] → 4 (delete -2: [1,0,3])
# [1,-2,-2,3] → 4 (delete one -2: [1,-2,3] or [1,3])
```

---

## Updated LeetCode Problem List (Extended)

| # | Problem | Pattern | Difficulty |
|---|---|---|---|
| 44 | Wildcard Matching | 2D DP | Hard |
| 10 | Regular Expression Matching | 2D DP | Hard |
| 140 | Word Break II | DP + Backtrack | Hard |
| 97 | Interleaving String | 2D DP | Medium |
| 1027 | Longest Arithmetic Subsequence | DP + Hash | Medium |
| 1235 | Maximum Profit in Job Scheduling | DP + Binary Search | Hard |
| 329 | Longest Increasing Path in Matrix | DFS + Memo | Hard |
| 1463 | Cherry Pickup II | 3D DP | Hard |
| 1186 | Maximum Subarray Sum With One Deletion | 1D DP | Medium |

---

## Further Resources

- **NeetCode DP Playlist** — https://neetcode.io/roadmap (Dynamic Programming section)
- **Aditya Verma DP Playlist (YouTube)** — best visual explanation of knapsack variants
- **"Grokking Dynamic Programming" (Educative.io)** — pattern-based approach
- **MIT 6.006 Lecture 19–21** — formal treatment of DP
