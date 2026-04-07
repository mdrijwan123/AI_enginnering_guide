# DSA Missing Patterns — Backtracking, Tries, Intervals, Greedy & Bit Manipulation
### Gap-Filler Study Guide for FAANG Coding Interviews

> **Why this file exists:** The main study guides cover arrays, hashing, two pointers, sliding window, stack, binary search, linked lists, trees, heaps, graphs, and DP. But FAANG consistently asks these additional patterns too. Missing ANY of these in an interview round is a fail.

---

## TABLE OF CONTENTS

1. [Backtracking — The Complete Guide](#part-1)
2. [Tries (Prefix Trees)](#part-2)
3. [Intervals](#part-3)
4. [Greedy Algorithms](#part-4)
5. [Bit Manipulation](#part-5)
6. [String Algorithms (Bonus)](#part-6)
7. [Interview Q&A](#part-7)

---

<a name="part-1"></a>
## Part 1 — Backtracking

> 📖 **Big picture:** Backtracking is how you solve problems that require *exploring all possibilities* — generating all combinations, all permutations, all valid paths, all valid placements on a grid (like N-Queens). The naive way is to just try everything, which is what backtracking does — but *intelligently*. It builds the solution one step at a time, and the moment a partial solution can't possibly lead to a valid answer, it backtracks (undoes the last step) and tries the next option.
>
> **The maze analogy:** Navigating a maze, you try a path, walk as far as you can. Dead end? You backtrack to the last junction and take a different turn. You explore *depth-first*, always committing to a direction until proven wrong.
>
> **The key technique that trips people up:** When you store a solution with `result.append(path)`, you’re appending a *reference* to the list, not a copy. Since you mutate `path` on every step, you’ll end up with a list of identical items at the end. Always use `result.append(path[:])` or `result.append(path.copy())`.

### 1.1 What Is Backtracking?

Backtracking is a systematic way to explore ALL possible solutions by building them incrementally and abandoning ("pruning") partial solutions as soon as you determine they can't lead to a valid answer.

```
Think of it like navigating a maze:
  - At each junction, pick a path
  - If you hit a dead end, BACKTRACK to the last junction and try another path
  - Continue until you find the exit (or have tried all paths)

The pattern:
  1. Choose: pick an element to add to the current solution
  2. Explore: recurse to add more elements
  3. Unchoose: remove the element and try the next option (BACKTRACK)
```

### 1.2 The Universal Backtracking Template

```python
def backtrack(candidates, path, result, start):
    """
    candidates: the set of choices available
    path: the current partial solution being built
    result: accumulates all valid complete solutions
    start: index to avoid duplicates / control where to look next
    """
    # Base case: is the current path a valid solution?
    if is_valid_solution(path):
        result.append(path[:])   # IMPORTANT: append a COPY, not reference
        return                   # return (if only one solution) or continue
    
    for i in range(start, len(candidates)):
        # Skip duplicates (when candidates has repeats and we need unique solutions)
        if i > start and candidates[i] == candidates[i-1]:
            continue
        
        # Pruning: if this choice can't lead to a valid solution, skip
        if not is_promising(candidates[i], path):
            continue
        
        # CHOOSE: add candidate to the path
        path.append(candidates[i])
        
        # EXPLORE: recurse
        backtrack(candidates, path, result, i + 1)  # i+1: each element used once
        # For unlimited reuse: pass i instead of i+1
        
        # UNCHOOSE (backtrack): remove the candidate
        path.pop()
    
    return result
```

**Time complexity:** Backtracking is inherently exponential — O(2ⁿ) or O(n!) depending on the problem. Pruning reduces the constant factor but doesn't change the asymptote.

### 1.3 Subsets (LC #78) — The Foundation

> Generate all subsets of a given array. Start here — this is the simplest backtracking problem and the foundation for all others.

```python
def subsets(nums):
    result = []
    
    def backtrack(start, path):
        # Every path is a valid subset (including empty)
        result.append(path[:])
        
        for i in range(start, len(nums)):
            path.append(nums[i])        # choose
            backtrack(i + 1, path)       # explore (i+1: move forward, no reuse)
            path.pop()                   # unchoose
    
    backtrack(0, [])
    return result

# nums = [1, 2, 3]
# Output: [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
#
# Decision tree:
#     []
#    / | \
#  [1] [2] [3]
#  /\    \
# [1,2] [1,3] [2,3]
#  |
# [1,2,3]
```

**Time:** O(n × 2ⁿ) — there are 2ⁿ subsets, each takes O(n) to copy.
**Space:** O(n) recursion depth.

### 1.4 Subsets II — With Duplicates (LC #90)

```python
def subsetsWithDup(nums):
    nums.sort()  # MUST sort to group duplicates together
    result = []
    
    def backtrack(start, path):
        result.append(path[:])
        
        for i in range(start, len(nums)):
            # Skip duplicates at the same recursion level
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result

# nums = [1, 2, 2]
# Without duplicate skip: [[],[1],[1,2],[1,2,2],[1,2],[2],[2,2],[2]]  ← duplicates!
# With skip:             [[],[1],[1,2],[1,2,2],[2],[2,2]]  ✓
```

**Key insight:** `i > start and nums[i] == nums[i-1]` — at the same level of the decision tree, don't pick the same value twice. But at DEEPER levels (recursive calls), duplicates are fine.

### 1.5 Permutations (LC #46) — All Orderings

```python
def permute(nums):
    result = []
    
    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        
        for i in range(len(remaining)):
            path.append(remaining[i])
            # Exclude used element from remaining
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()
    
    backtrack([], nums)
    return result

# nums = [1, 2, 3]
# Output: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
# Count: 3! = 6

# Alternative using a visited set (more efficient):
def permute_v2(nums):
    result = []
    used = [False] * len(nums)
    
    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path)
            path.pop()
            used[i] = False
    
    backtrack([])
    return result
```

**Time:** O(n × n!) — n! permutations, each O(n) to copy.

### 1.6 Permutations II — With Duplicates (LC #47)

```python
def permuteUnique(nums):
    nums.sort()
    result = []
    used = [False] * len(nums)
    
    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for i in range(len(nums)):
            if used[i]:
                continue
            # Skip duplicate: same value at same level AND previous duplicate wasn't used
            if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                continue
            
            used[i] = True
            path.append(nums[i])
            backtrack(path)
            path.pop()
            used[i] = False
    
    backtrack([])
    return result

# nums = [1, 1, 2]
# Output: [[1,1,2], [1,2,1], [2,1,1]]  (not [[1,1,2], [1,2,1], [1,1,2], ...])
```

### 1.7 Combinations (LC #77)

> Given n and k, return all combinations of k numbers chosen from [1, n].

```python
def combine(n, k):
    result = []
    
    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        
        # Pruning: if remaining numbers aren't enough to fill k slots, stop
        remaining_needed = k - len(path)
        for i in range(start, n + 1 - remaining_needed + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(1, [])
    return result

# combine(4, 2) → [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
```

### 1.8 Combination Sum (LC #39) — Unlimited Reuse

```python
def combinationSum(candidates, target):
    result = []
    candidates.sort()
    
    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return
        
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break  # prune: all future candidates are also > remaining
            
            path.append(candidates[i])
            backtrack(i, path, remaining - candidates[i])  # i, NOT i+1 (reuse allowed)
            path.pop()
    
    backtrack(0, [], target)
    return result

# candidates = [2, 3, 6, 7], target = 7
# Output: [[2,2,3], [7]]
```

### 1.9 Combination Sum II — Each Number Used Once (LC #40)

```python
def combinationSum2(candidates, target):
    candidates.sort()
    result = []
    
    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return
        
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break
            # Skip duplicates at same level
            if i > start and candidates[i] == candidates[i-1]:
                continue
            
            path.append(candidates[i])
            backtrack(i + 1, path, remaining - candidates[i])  # i+1: no reuse
            path.pop()
    
    backtrack(0, [], target)
    return result
```

### 1.10 Palindrome Partitioning (LC #131)

```python
def partition(s):
    result = []
    
    def is_palindrome(sub):
        return sub == sub[::-1]
    
    def backtrack(start, path):
        if start == len(s):
            result.append(path[:])
            return
        
        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]
            if is_palindrome(substring):
                path.append(substring)
                backtrack(end, path)
                path.pop()
    
    backtrack(0, [])
    return result

# s = "aab"
# Output: [["a","a","b"], ["aa","b"]]
```

### 1.11 N-Queens (LC #51) — The Classic

```python
def solveNQueens(n):
    result = []
    # Track which columns and diagonals are under attack
    cols = set()
    pos_diag = set()  # (row + col) identifies a positive diagonal
    neg_diag = set()  # (row - col) identifies a negative diagonal
    board = [['.' for _ in range(n)] for _ in range(n)]
    
    def backtrack(row):
        if row == n:
            # All queens placed successfully
            result.append([''.join(r) for r in board])
            return
        
        for col in range(n):
            # Check if this position is safe
            if col in cols or (row + col) in pos_diag or (row - col) in neg_diag:
                continue
            
            # Place queen
            board[row][col] = 'Q'
            cols.add(col)
            pos_diag.add(row + col)
            neg_diag.add(row - col)
            
            # Recurse to next row
            backtrack(row + 1)
            
            # Remove queen (backtrack)
            board[row][col] = '.'
            cols.remove(col)
            pos_diag.remove(row + col)
            neg_diag.remove(row - col)
    
    backtrack(0)
    return result

# n = 4
# Output: [[".Q..","...Q","Q...","..Q."], ["..Q.","Q...","...Q",".Q.."]]
# Two valid configurations for 4-Queens
```

**How N-Queens pruning works:**
```
Row-by-row placement: One queen per row, guaranteed by recursion structure.
Column check: set `cols` — O(1) lookup
Diagonal checks:
  Positive diagonal (↗): all cells where (row + col) is the same
  Negative diagonal (↘): all cells where (row - col) is the same
  
Without pruning: O(n!) placements to try
With pruning: dramatically fewer — only valid positions explored
```

### 1.12 Word Search (LC #79) — Grid Backtracking

```python
def exist(board, word):
    rows, cols = len(board), len(board[0])
    
    def backtrack(r, c, idx):
        if idx == len(word):
            return True
        
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            board[r][c] != word[idx]):
            return False
        
        # Mark as visited (modify board temporarily)
        temp = board[r][c]
        board[r][c] = '#'
        
        # Explore all 4 directions
        found = (backtrack(r + 1, c, idx + 1) or
                 backtrack(r - 1, c, idx + 1) or
                 backtrack(r, c + 1, idx + 1) or
                 backtrack(r, c - 1, idx + 1))
        
        # Unmark (backtrack)
        board[r][c] = temp
        
        return found
    
    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True
    return False

# board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = "ABCCED" → True
```

### 1.13 Generate Parentheses (LC #22)

```python
def generateParenthesis(n):
    result = []
    
    def backtrack(path, open_count, close_count):
        if len(path) == 2 * n:
            result.append(''.join(path))
            return
        
        # Can always add open paren if we haven't used all n
        if open_count < n:
            path.append('(')
            backtrack(path, open_count + 1, close_count)
            path.pop()
        
        # Can add close paren only if it won't exceed open count
        if close_count < open_count:
            path.append(')')
            backtrack(path, open_count, close_count + 1)
            path.pop()
    
    backtrack([], 0, 0)
    return result

# n = 3
# Output: ["((()))","(()())","(())()","()(())","()()()"]
```

### 1.14 Letter Combinations of a Phone Number (LC #17)

```python
def letterCombinations(digits):
    if not digits:
        return []
    
    phone_map = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }
    result = []
    
    def backtrack(idx, path):
        if idx == len(digits):
            result.append(''.join(path))
            return
        
        for char in phone_map[digits[idx]]:
            path.append(char)
            backtrack(idx + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result

# digits = "23"
# Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

### 1.15 Backtracking Cheat Sheet

| Problem | Key Differences |
|---|---|
| Subsets | Collect path at every node |
| Combinations | Collect path only when len(path) == k |
| Permutations | Use `used` array, don't pass `start` |
| Combination Sum | Pass `i` (reuse) or `i+1` (no reuse) |
| With Duplicates | Sort + skip `if i > start and nums[i] == nums[i-1]` |
| N-Queens | Track cols, positive/negative diagonals |
| Grid search | Mark cell visited, explore 4 dirs, unmark |

---

<a name="part-2"></a>
## Part 2 — Tries (Prefix Trees)

> 📖 **Big picture:** A trie is a specialised tree for storing words where each level represents one character. Its superpower is prefix search: "give me all words starting with 'app'" — a hash table can’t do this efficiently, a trie does it in O(prefix length).
>
> **The dictionary analogy:** Think of a physical dictionary. To find all words beginning with "pre", you open to the P section, then P-R, then P-R-E — and now you’re at the start of all "pre" words. You navigate letter by letter along a shared path. That *shared path* is the key idea: all words starting with "app" share the nodes a-p-p in the trie, which is why prefix search is efficient.
>
> **When to use:** AutoComplete, spell check, IP routing tables, finding longest common prefix, word search in a grid.

### 2.1 What Is a Trie?

A trie (pronounced "try") is a tree-like data structure for storing strings where each node represents a character. It enables O(L) lookup, insertion, and prefix search (L = length of the word).

```
Insert: "apple", "app", "ape", "bat"

           root
          /    \
         a      b
        /        \
       p          a
      / \          \
     p   e($)      t($)
     |
     l
     |
     e($)

($) marks "end of word" — "app" ends at the 2nd 'p', "apple" ends at 'e'
```

### 2.2 Trie Implementation

```python
class TrieNode:
    def __init__(self):
        self.children = {}     # char → TrieNode
        self.is_end = False    # marks end of a complete word

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        """Insert a word into the trie. O(L) time."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word: str) -> bool:
        """Return True if word is in trie. O(L) time."""
        node = self._find_node(word)
        return node is not None and node.is_end
    
    def startsWith(self, prefix: str) -> bool:
        """Return True if any word starts with prefix. O(L) time."""
        return self._find_node(prefix) is not None
    
    def _find_node(self, prefix: str) -> TrieNode:
        """Navigate to the node at the end of prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

# Usage
trie = Trie()
trie.insert("apple")
trie.insert("app")
print(trie.search("apple"))      # True
print(trie.search("app"))        # True
print(trie.search("ap"))         # False (not a complete word)
print(trie.startsWith("ap"))     # True (prefix exists)
print(trie.startsWith("bat"))    # False
```

### 2.3 Trie vs Hash Set — When to Use Each

| Operation | Trie | Hash Set |
|---|---|---|
| Exact word lookup | O(L) | O(L) average |
| Prefix search | O(L) ✓ | O(n×L) — must scan all words |
| Autocomplete | O(L + k) | Not possible efficiently |
| Space (sparse) | Higher (node per char) | Lower (hash per word) |
| Space (many shared prefixes) | Lower ✓ | Higher |

**Use a Trie when:** prefix search, autocomplete, word dictionary with prefix queries.
**Use a Hash Set when:** only exact lookups needed, simpler implementation.

### 2.4 Autocomplete / Find Words with Prefix

```python
def find_words_with_prefix(trie, prefix):
    """Return all words in trie that start with prefix."""
    node = trie._find_node(prefix)
    if node is None:
        return []
    
    result = []
    
    def dfs(node, current_word):
        if node.is_end:
            result.append(current_word)
        for char, child in node.children.items():
            dfs(child, current_word + char)
    
    dfs(node, prefix)
    return result

# trie contains: "apple", "app", "application", "ape", "bat"
# find_words_with_prefix(trie, "app") → ["app", "apple", "application"]
```

### 2.5 Word Search II (LC #212) — Trie + Backtracking (Hard)

Google's favourite. Given a board of characters and a list of words, find all words on the board.

```python
def findWords(board, words):
    # Step 1: Build trie from all target words
    root = TrieNode()
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.word = word  # store the full word at the end node
    
    rows, cols = len(board), len(board[0])
    result = set()
    
    def backtrack(r, c, node):
        char = board[r][c]
        if char not in node.children:
            return
        
        next_node = node.children[char]
        
        # Found a word!
        if hasattr(next_node, 'word') and next_node.word:
            result.add(next_node.word)
            next_node.word = None  # avoid duplicates
        
        # Mark visited
        board[r][c] = '#'
        
        # Explore 4 directions
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != '#':
                backtrack(nr, nc, next_node)
        
        # Unmark
        board[r][c] = char
        
        # Optimisation: prune empty branches
        if not next_node.children:
            del node.children[char]
    
    # Step 2: Start backtracking from every cell
    for r in range(rows):
        for c in range(cols):
            backtrack(r, c, root)
    
    return list(result)

# board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]
# words = ["oath","pea","eat","rain"]
# Output: ["eat","oath"]
```

**Why Trie + Backtracking?**
- Without Trie: for each word, do Word Search from every cell → O(words × rows × cols × 4^L)
- With Trie: single backtracking pass, shared prefix matching → dramatically faster

### 2.6 Design Add and Search Words (LC #211)

```python
class WordDictionary:
    def __init__(self):
        self.root = TrieNode()
    
    def addWord(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        """Search with '.' as wildcard (matches any character)."""
        def dfs(node, i):
            if i == len(word):
                return node.is_end
            
            char = word[i]
            if char == '.':
                # Wildcard: try every child
                for child in node.children.values():
                    if dfs(child, i + 1):
                        return True
                return False
            else:
                if char not in node.children:
                    return False
                return dfs(node.children[char], i + 1)
        
        return dfs(self.root, 0)

# wd = WordDictionary()
# wd.addWord("bad"); wd.addWord("dad"); wd.addWord("mad")
# wd.search("pad") → False
# wd.search("bad") → True
# wd.search(".ad") → True (matches "bad", "dad", "mad")
# wd.search("b..") → True
```

---

<a name="part-3"></a>
## Part 3 — Intervals

> 📖 **Big picture:** Interval problems are about managing time slots, ranges, or spans. They come up constantly in real life (meeting scheduler, calendar conflicts, merging date ranges from a database) and in FAANG interviews.
>
> **The calendar analogy:** Imagine you have a list of meeting time slots on a calendar. You want to find if any meetings overlap, or merge all adjacent ones, or find the first free slot. Sorting the meetings by start time is the key move — once sorted, overlapping intervals are always adjacent, so you can process them in a simple left-to-right sweep.
>
> **The single most important rule:** Sort intervals by start time first. Almost every interval problem becomes simple after sorting.

### 3.1 The Intervals Pattern

Interval problems involve ranges [start, end] and operations like merging, inserting, or finding overlaps.

**Key technique:** Sort by start time, then process sequentially.

**When do two intervals overlap?**
```
Interval A: [a_start, a_end]
Interval B: [b_start, b_end]

Overlap if: a_start < b_end AND b_start < a_end
No overlap if: a_end <= b_start OR b_end <= a_start
```

### 3.2 Merge Intervals (LC #56) — Most Common

```python
def merge(intervals):
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for start, end in intervals[1:]:
        # Does current interval overlap with the last merged interval?
        if start <= merged[-1][1]:
            # Merge: extend the end of the last merged interval
            merged[-1][1] = max(merged[-1][1], end)
        else:
            # No overlap: add as new interval
            merged.append([start, end])
    
    return merged

# intervals = [[1,3],[2,6],[8,10],[15,18]]
# After sort: [[1,3],[2,6],[8,10],[15,18]]
# Step 1: [1,3] + [2,6] → overlap (2 ≤ 3) → merge to [1,6]
# Step 2: [1,6] + [8,10] → no overlap (8 > 6) → add [8,10]
# Step 3: [8,10] + [15,18] → no overlap → add [15,18]
# Output: [[1,6],[8,10],[15,18]]
```

**Time:** O(n log n) for sort. **Space:** O(n) for output.

### 3.3 Insert Interval (LC #57)

```python
def insert(intervals, newInterval):
    result = []
    i = 0
    n = len(intervals)
    
    # Step 1: Add all intervals that come BEFORE newInterval (no overlap)
    while i < n and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1
    
    # Step 2: Merge all overlapping intervals with newInterval
    while i < n and intervals[i][0] <= newInterval[1]:
        newInterval[0] = min(newInterval[0], intervals[i][0])
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1
    result.append(newInterval)
    
    # Step 3: Add all intervals that come AFTER newInterval
    while i < n:
        result.append(intervals[i])
        i += 1
    
    return result

# intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]]
# newInterval = [4,8]
# Before: [1,2]
# Merge [3,5],[6,7],[8,10] with [4,8] → [3,10]
# After: [12,16]
# Output: [[1,2],[3,10],[12,16]]
```

### 3.4 Meeting Rooms (LC #252 — Can attend all?)

```python
def canAttendMeetings(intervals):
    intervals.sort(key=lambda x: x[0])
    
    for i in range(1, len(intervals)):
        # If current meeting starts before previous ends → conflict
        if intervals[i][0] < intervals[i-1][1]:
            return False
    
    return True

# [[0,30],[5,10],[15,20]] → False (5 < 30, conflict!)
# [[7,10],[2,4]] → True (no overlap after sorting)
```

### 3.5 Meeting Rooms II — Minimum Rooms Needed (LC #253)

```python
import heapq

def minMeetingRooms(intervals):
    if not intervals:
        return 0
    
    intervals.sort(key=lambda x: x[0])
    
    # Min-heap tracks the END times of ongoing meetings
    heap = []
    
    for start, end in intervals:
        # If earliest ending meeting ends before this one starts,
        # that room is free — reuse it
        if heap and heap[0] <= start:
            heapq.heappop(heap)
        
        # Allocate a room for this meeting
        heapq.heappush(heap, end)
    
    # Number of rooms = number of ongoing meetings
    return len(heap)

# [[0,30],[5,10],[15,20]]
# Sort: [[0,30],[5,10],[15,20]]
# heap: [30] → [10,30] (5<30, need new room) → [20,30] (15>10, reuse room)
# Answer: 2 rooms
```

**Alternative approach: sweep line**
```python
def minMeetingRooms_sweep(intervals):
    events = []
    for start, end in intervals:
        events.append((start, 1))   # meeting starts: +1 room
        events.append((end, -1))    # meeting ends:   -1 room
    
    events.sort()
    max_rooms = current = 0
    for _, delta in events:
        current += delta
        max_rooms = max(max_rooms, current)
    
    return max_rooms
```

### 3.6 Non-Overlapping Intervals (LC #435) — Greedy

```python
def eraseOverlapIntervals(intervals):
    # Greedy: sort by END time, keep intervals with earliest end
    intervals.sort(key=lambda x: x[1])
    
    count = 0
    prev_end = float('-inf')
    
    for start, end in intervals:
        if start >= prev_end:
            # No overlap: keep this interval
            prev_end = end
        else:
            # Overlap: remove this interval (count it)
            count += 1
    
    return count

# intervals = [[1,2],[2,3],[3,4],[1,3]]
# Sort by end: [[1,2],[2,3],[1,3],[3,4]]
# Keep [1,2] → keep [2,3] (2≥2) → remove [1,3] (1<3) → keep [3,4]
# Removed: 1
```

### 3.7 Interval List Intersections (LC #986)

```python
def intervalIntersection(firstList, secondList):
    result = []
    i, j = 0, 0
    
    while i < len(firstList) and j < len(secondList):
        # Find the intersection of two intervals
        start = max(firstList[i][0], secondList[j][0])
        end = min(firstList[i][1], secondList[j][1])
        
        if start <= end:
            result.append([start, end])
        
        # Advance the pointer with the earlier end time
        if firstList[i][1] < secondList[j][1]:
            i += 1
        else:
            j += 1
    
    return result
```

---

<a name="part-4"></a>
## Part 4 — Greedy Algorithms

> 📖 **Big picture:** Greedy algorithms are deceptively simple: at each step, make the choice that looks best *right now*, without considering future consequences. The tricky part is knowing *when this is safe*. If a locally optimal choice always leads to a globally optimal answer, greedy works. If not, you need dynamic programming.
>
> **The coin change intuition:** Making change for 36 cents with coins [25, 10, 5, 1]. Greedy says: always use the largest coin that fits. 25 + 10 + 1 = 36. This works!
> But with coins [1, 3, 4] and amount 6: Greedy picks 4 + 1 + 1 = 3 coins. Optimal is 3 + 3 = 2 coins. Greedy fails here! Use DP instead.
>
> **The test:** Greedy works for "activity selection" type problems (scheduling, interval problems, minimum spanning trees). DP is needed when past choices constrain future choices in complex ways.

### 4.1 What Is Greedy?

A greedy algorithm makes the **locally optimal choice at each step**, hoping it leads to a globally optimal solution. Unlike DP, greedy doesn't revisit past choices.

**Greedy works when:** the problem has the **greedy choice property** — a locally optimal choice leads to a globally optimal solution.

**Greedy fails when:** a short-term good choice makes the long-term result worse (then use DP).

### 4.2 Jump Game (LC #55) — Can You Reach the End?

```python
def canJump(nums):
    # Greedy: track the FARTHEST position reachable
    max_reach = 0
    
    for i in range(len(nums)):
        if i > max_reach:
            return False  # stuck! can't reach this position
        max_reach = max(max_reach, i + nums[i])
        if max_reach >= len(nums) - 1:
            return True
    
    return True

# nums = [2,3,1,1,4] → True
# Step 0: max_reach = max(0, 0+2) = 2
# Step 1: max_reach = max(2, 1+3) = 4 ≥ 4 → True!

# nums = [3,2,1,0,4] → False
# Step 0: max_reach = 3
# Step 1: max_reach = 3
# Step 2: max_reach = 3
# Step 3: max_reach = 3 (stuck at index 3, nums[3]=0)
# Step 4: i=4 > max_reach=3 → False
```

### 4.3 Jump Game II — Minimum Jumps (LC #45)

```python
def jump(nums):
    jumps = 0
    current_end = 0    # farthest we can go with current number of jumps
    farthest = 0       # farthest we've seen so far
    
    for i in range(len(nums) - 1):  # don't need to jump from last position
        farthest = max(farthest, i + nums[i])
        
        if i == current_end:
            # We MUST jump now (reached the end of current jump range)
            jumps += 1
            current_end = farthest
            
            if current_end >= len(nums) - 1:
                break
    
    return jumps

# nums = [2,3,1,1,4] → 2 jumps
# Jump 1: from 0 to 1 (land on 3)
# Jump 2: from 1 to 4 (land on end)
```

### 4.4 Gas Station (LC #134)

```python
def canCompleteCircuit(gas, cost):
    # If total gas < total cost, impossible
    if sum(gas) < sum(cost):
        return -1
    
    # Greedy: start from the first station where tank never goes negative
    tank = 0
    start = 0
    
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:
            # Can't start from `start` — try starting from next station
            start = i + 1
            tank = 0
    
    return start

# gas = [1,2,3,4,5], cost = [3,4,5,1,2]
# Total gas = 15, total cost = 15 → possible
# Try starting from 0: tank goes -2 at station 0 → reset
# Starting from 3: tank = 3, 6, 4, 2, 0 → works!
# Answer: 3
```

### 4.5 Task Scheduler (LC #621)

```python
def leastInterval(tasks, n):
    from collections import Counter
    
    freq = Counter(tasks)
    max_freq = max(freq.values())
    # Count how many tasks have the maximum frequency
    max_count = sum(1 for f in freq.values() if f == max_freq)
    
    # Minimum intervals = (max_freq - 1) × (n + 1) + max_count
    # Or the number of tasks (if no idle needed)
    result = max(len(tasks), (max_freq - 1) * (n + 1) + max_count)
    return result

# tasks = ["A","A","A","B","B","B"], n = 2
# max_freq = 3 (both A and B appear 3 times)
# max_count = 2
# result = max(6, (3-1) × (2+1) + 2) = max(6, 8) = 8
# Schedule: A B _ A B _ A B → 8 intervals
```

### 4.6 Partition Labels (LC #763)

```python
def partitionLabels(s):
    # Record the LAST occurrence of each character
    last = {c: i for i, c in enumerate(s)}
    
    result = []
    start = end = 0
    
    for i, c in enumerate(s):
        end = max(end, last[c])  # extend partition to include last occurrence
        
        if i == end:
            # All characters in [start, end] don't appear after end
            result.append(end - start + 1)
            start = end + 1
    
    return result

# s = "ababcbacadefegdehijhklij"
# Output: [9, 7, 8]
# "ababcbaca" (a,b,c all contained), "defegde" (d,e,f,g), "hijhklij"
```

### 4.7 Best Time to Buy and Sell Stock II (LC #122) — Multiple Transactions

```python
def maxProfit(prices):
    # Greedy: capture EVERY upward movement
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            profit += prices[i] - prices[i-1]
    return profit

# prices = [7,1,5,3,6,4]
# Day 1→2: +4 (buy 1, sell 5)
# Day 3→4: +3 (buy 3, sell 6)
# Total: 7
```

### 4.8 Greedy vs DP Decision Guide

```
Ask yourself:
1. Can I get the optimal answer by always making the locally best choice?
   YES → Try Greedy first (simpler, faster)
   
2. Does the problem have overlapping subproblems?
   YES → DP is needed
   
3. Can I prove the greedy choice property?
   YES → Greedy is correct
   NOT SURE → Use DP (always correct if substructure exists)

Examples:
  - Coin change (US coins: 25,10,5,1): Greedy works (always pick largest coin)
  - Coin change (arbitrary coins: 6,4,1, target=8): Greedy fails (6+1+1=8 vs 4+4=8)
    → Need DP
  - Activity selection: Greedy (sort by end time) → optimal
  - 0/1 Knapsack: Greedy fails → need DP
  - Fractional Knapsack: Greedy works (pick highest value/weight ratio)
```

---

<a name="part-5"></a>
## Part 5 — Bit Manipulation

> 📖 **Big picture:** Computers store all data in binary (1s and 0s). Bit manipulation operates on these raw binary representations directly, bypassing normal arithmetic. The reward is speed (bitwise ops are a single CPU instruction) and elegant O(1) solutions to problems that look hard with normal maths.
>
> **Why interviewers love it:** Bit manipulation problems have compact, beautiful solutions. A problem like "find the single non-repeating element in an array" takes O(n) space with a hash map, but XOR gives you an O(1) space O(n) time solution in one line.
>
> **The mental model:** Think of each integer as 32 light switches (bits), each either ON (1) or OFF (0). Bitwise AND, OR, XOR, and shifts are operations on these switches. Mastering 6-7 key tricks will cover 95% of bit manipulation problems you’ll ever see in an interview.

### 5.1 Bit Basics

```python
# Bitwise operators in Python:
a = 0b1010  # 10 in binary
b = 0b1100  # 12 in binary

a & b    # AND:  1010 & 1100 = 1000 (8)   — both bits 1
a | b    # OR:   1010 | 1100 = 1110 (14)  — either bit 1
a ^ b    # XOR:  1010 ^ 1100 = 0110 (6)   — bits differ
~a       # NOT:  ~1010 = ...10101 (complement — careful with Python's int size)
a << 2   # Left shift:  1010 → 101000 (40) — multiply by 2²
a >> 1   # Right shift: 1010 → 0101 (5)    — divide by 2

# Key XOR properties:
# a ^ a = 0   (anything XOR itself is 0)
# a ^ 0 = a   (anything XOR 0 is itself)
# a ^ b ^ a = b  (XOR is self-inverse — cancels out)
# XOR is commutative and associative
```

### 5.2 Single Number (LC #136) — XOR Magic

```python
def singleNumber(nums):
    """Every element appears twice except one. Find the unique one."""
    result = 0
    for n in nums:
        result ^= n
    return result

# nums = [4, 1, 2, 1, 2]
# 0 ^ 4 = 4
# 4 ^ 1 = 5
# 5 ^ 2 = 7
# 7 ^ 1 = 6  (1 cancels out)
# 6 ^ 2 = 4  (2 cancels out)
# Answer: 4

# Why: all pairs cancel (a ^ a = 0), leaving only the unique element
# Time: O(n), Space: O(1) — can't beat this!
```

### 5.3 Number of 1 Bits (LC #191) — Hamming Weight

```python
def hammingWeight(n):
    count = 0
    while n:
        count += n & 1    # check last bit
        n >>= 1           # shift right
    return count

# Brian Kernighan's trick (faster — only loops once per 1-bit):
def hammingWeight_fast(n):
    count = 0
    while n:
        n &= (n - 1)     # removes the lowest set bit!
        count += 1
    return count

# n = 11 (binary: 1011)
# 1011 & 1010 = 1010 (removed lowest 1) → count=1
# 1010 & 1001 = 1000 (removed next 1)  → count=2
# 1000 & 0111 = 0000 (removed last 1)  → count=3
# Answer: 3
```

### 5.4 Counting Bits (LC #338)

```python
def countBits(n):
    """For each i in [0, n], count number of 1-bits."""
    result = [0] * (n + 1)
    for i in range(1, n + 1):
        # DP: number of 1s in i = number of 1s in (i >> 1) + (i & 1)
        result[i] = result[i >> 1] + (i & 1)
    return result

# countBits(5) → [0, 1, 1, 2, 1, 2]
# 0=0000, 1=0001, 2=0010, 3=0011, 4=0100, 5=0101
```

### 5.5 Power of Two (LC #231)

```python
def isPowerOfTwo(n):
    # A power of 2 has exactly one 1-bit: 1, 10, 100, 1000, ...
    # n & (n-1) removes the lowest 1-bit
    # If result is 0, there was only one 1-bit → power of 2
    return n > 0 and (n & (n - 1)) == 0

# 8 = 1000, 7 = 0111 → 1000 & 0111 = 0000 → True
# 6 = 0110, 5 = 0101 → 0110 & 0101 = 0100 → False
```

### 5.6 Reverse Bits (LC #190)

```python
def reverseBits(n):
    result = 0
    for i in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result
```

### 5.7 Missing Number (LC #268)

```python
def missingNumber(nums):
    """Given [0, 1, ..., n] with one number missing, find it."""
    # XOR approach: XOR all indices and all values
    # Pairs cancel, leaving the missing number
    n = len(nums)
    result = n  # start with n (since indices go 0 to n-1)
    for i in range(n):
        result ^= i ^ nums[i]
    return result

# Or simply: sum approach
def missingNumber_sum(nums):
    n = len(nums)
    expected = n * (n + 1) // 2
    return expected - sum(nums)
```

### 5.8 Bit Manipulation Cheat Sheet

| Trick | Code | What It Does |
|---|---|---|
| Check if bit i is set | `(n >> i) & 1` | Returns 1 if bit i is 1 |
| Set bit i | `n | (1 << i)` | Makes bit i = 1 |
| Clear bit i | `n & ~(1 << i)` | Makes bit i = 0 |
| Toggle bit i | `n ^ (1 << i)` | Flips bit i |
| Remove lowest set bit | `n & (n-1)` | Brian Kernighan's trick |
| Isolate lowest set bit | `n & (-n)` | Gets the lowest 1-bit |
| Check power of 2 | `n > 0 and n & (n-1) == 0` | True if exactly one 1-bit |
| Get all 1s mask (k bits) | `(1 << k) - 1` | e.g., k=4 → 1111 |

---

<a name="part-6"></a>
## Part 6 — String Algorithms (Bonus)

### 6.1 Valid Palindrome (LC #125) — Two Pointer

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

### 6.2 Longest Palindromic Substring (LC #5) — Expand Around Centre

```python
def longestPalindrome(s):
    result = ""
    
    def expand(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]
    
    for i in range(len(s)):
        # Odd-length palindrome (centre = single char)
        odd = expand(i, i)
        # Even-length palindrome (centre = between two chars)
        even = expand(i, i + 1)
        
        if len(odd) > len(result):
            result = odd
        if len(even) > len(result):
            result = even
    
    return result

# s = "babad" → "bab" or "aba"
# Time: O(n²), Space: O(1) — better than DP for this problem
```

### 6.3 Longest Common Prefix (LC #14)

```python
def longestCommonPrefix(strs):
    if not strs:
        return ""
    
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    
    return prefix

# Or using zip:
def longestCommonPrefix_zip(strs):
    if not strs:
        return ""
    
    result = []
    for chars in zip(*strs):
        if len(set(chars)) == 1:    # all same character at this position
            result.append(chars[0])
        else:
            break
    
    return ''.join(result)
```

---

<a name="part-7"></a>
## Interview Q&A

**Q1: When should you use backtracking?**
> When you need to explore all possible combinations/permutations/configurations and the solution space has a tree structure. Classic signals: "find ALL", "generate ALL", "count the number of ways". The key optimization is pruning — skipping branches that can't lead to valid solutions. Examples: N-Queens, Sudoku solver, all subsets/permutations, word search on a grid.

**Q2: What's the time complexity of generating all subsets?**
> O(n × 2ⁿ). There are 2ⁿ subsets of an n-element set (each element is either included or excluded). Copying each subset to the result takes O(n). This is optimal since you must output all 2ⁿ subsets. For permutations: O(n × n!).

**Q3: What's the difference between subsets, combinations, and permutations?**
> Subsets: all possible groups (any size, order doesn't matter). Combinations: groups of fixed size k (order doesn't matter). Permutations: all orderings (order MATTERS). Example with [1,2,3]: Subsets include [], [1], [1,2], [1,2,3], etc. Combinations of size 2: [1,2], [1,3], [2,3]. Permutations: [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1].

**Q4: What is a Trie and when would you use one over a hash set?**
> A trie is a tree where each node represents a character, enabling O(L) operations where L is the word length. Advantage over hash set: prefix operations (startsWith, autocomplete, wildcard matching) in O(L) — a hash set requires O(n×L) scanning. Disadvantage: higher memory per word due to node overhead. Use trie when prefix-based lookups are needed.

**Q5: How do you approach interval problems?**
> Almost always sort by start time (or end time for greedy selection). Then scan left-to-right, tracking the current interval's end. Three cases per new interval: (1) no overlap → add new interval, (2) overlap → merge by extending end, (3) contained → skip. For "minimum rooms" problems, use a min-heap of end times. For "maximum non-overlapping" problems, sort by END time (greedy: pick earliest ending).

**Q6: What's the greedy choice property?**
> A problem has the greedy choice property when making the locally optimal choice at each step leads to a globally optimal solution. Example: activity selection — always choosing the activity that ends earliest leaves maximum room for future activities, and this provably gives the optimal schedule. When unsure, try to prove greedy works via exchange argument: show that swapping a greedy choice for a non-greedy one never improves the result.

**Q7: How does `n & (n-1)` work and what is it useful for?**
> It removes the lowest set bit from n. Example: n=12 (1100), n-1=11 (1011), n & (n-1) = 1000 (8). Useful for: counting 1-bits (Brian Kernighan's — loop until n becomes 0), checking power of 2 (only one set bit → n & (n-1) == 0), and fast bit counting.

---

## Practice Problem Log

### Backtracking (12 problems)
| # | Problem | Difficulty | Status |
|---|---|---|---|
| 78 | Subsets | Med | ☐ |
| 90 | Subsets II | Med | ☐ |
| 46 | Permutations | Med | ☐ |
| 47 | Permutations II | Med | ☐ |
| 77 | Combinations | Med | ☐ |
| 39 | Combination Sum | Med | ☐ |
| 40 | Combination Sum II | Med | ☐ |
| 131 | Palindrome Partitioning | Med | ☐ |
| 51 | N-Queens | Hard | ☐ |
| 79 | Word Search | Med | ☐ |
| 22 | Generate Parentheses | Med | ☐ |
| 17 | Letter Combinations | Med | ☐ |

### Tries (4 problems)
| # | Problem | Difficulty | Status |
|---|---|---|---|
| 208 | Implement Trie | Med | ☐ |
| 211 | Add and Search Word | Med | ☐ |
| 212 | Word Search II | Hard | ☐ |
| 14 | Longest Common Prefix | Easy | ☐ |

### Intervals (6 problems)
| # | Problem | Difficulty | Status |
|---|---|---|---|
| 56 | Merge Intervals | Med | ☐ |
| 57 | Insert Interval | Med | ☐ |
| 252 | Meeting Rooms | Easy | ☐ |
| 253 | Meeting Rooms II | Med | ☐ |
| 435 | Non-Overlapping Intervals | Med | ☐ |
| 986 | Interval List Intersections | Med | ☐ |

### Greedy (8 problems)
| # | Problem | Difficulty | Status |
|---|---|---|---|
| 55 | Jump Game | Med | ☐ |
| 45 | Jump Game II | Med | ☐ |
| 134 | Gas Station | Med | ☐ |
| 621 | Task Scheduler | Med | ☐ |
| 763 | Partition Labels | Med | ☐ |
| 122 | Buy/Sell Stock II | Med | ☐ |
| 435 | Non-Overlapping Intervals | Med | ☐ |
| 846 | Hand of Straights | Med | ☐ |

### Bit Manipulation (6 problems)
| # | Problem | Difficulty | Status |
|---|---|---|---|
| 136 | Single Number | Easy | ☐ |
| 191 | Number of 1 Bits | Easy | ☐ |
| 338 | Counting Bits | Easy | ☐ |
| 231 | Power of Two | Easy | ☐ |
| 190 | Reverse Bits | Easy | ☐ |
| 268 | Missing Number | Easy | ☐ |

---

## 📚 Further Resources

- **NeetCode Roadmap** (covers all these patterns with video explanations): https://neetcode.io/roadmap
- **"Cracking the Coding Interview" by Gayle Laakmann McDowell** — Chapter on backtracking and bit manipulation
- **AlgoMonster** — Pattern-based teaching with these exact categories: https://algo.monster
- **Blind 75** — The original curated list covering all patterns: https://neetcode.io/practice

> **With this file + the main study guides, you now have complete coverage of every DSA pattern tested at FAANG.** No gaps remain.
