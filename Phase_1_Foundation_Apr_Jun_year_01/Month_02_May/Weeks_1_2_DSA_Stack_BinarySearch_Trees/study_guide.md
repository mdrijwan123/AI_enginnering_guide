# Weeks 1–2: Stack, Binary Search & Trees
### Phase 1 | Month 2 | May 5–18, 2026

> **DSA mornings:** Continue daily LeetCode 6–7 AM. New topics: Stack, Binary Search, Linked Lists, Trees.

---

## 🎯 Learning Objectives

By the end of these two weeks you will be able to:
- Implement and use a stack for expression parsing, monotonic problems, and backtracking
- Apply binary search to arrays, rotated arrays, and answer-space problems
- Traverse linked lists with slow/fast pointers
- Traverse trees with DFS (recursive + iterative) and BFS (level-order)
- Solve all 20 problems listed here

---

## Part 1 — Stack

### 1.1 Stack LIFO Operations
Stack = Last-In First-Out. Python `list` works perfectly as a stack.

```python
stack = []
stack.append(1)  # push
stack.append(2)
stack.append(3)
top = stack[-1]  # peek: 3
stack.pop()      # pop: removes 3
```
All O(1).

### 1.2 Pattern: Monotonic Stack

A monotonic stack maintains elements in a strictly increasing (or decreasing) order. Used to find **next/previous greater/smaller element** in O(n).

```python
# Next Greater Element for each position
def nextGreaterElement(nums):
    result = [-1] * len(nums)
    stack = []  # indices of elements without a "next greater" yet
    
    for i, n in enumerate(nums):
        # Pop all elements smaller than current → current is their "next greater"
        while stack and nums[stack[-1]] < n:
            idx = stack.pop()
            result[idx] = n
        stack.append(i)
    
    return result

# nums = [2, 1, 5, 3, 6]
# result = [5, 5, 6, 6, -1]
```

### 1.3 Key Stack Problems

#### Valid Parentheses (LC #20)
```python
def isValid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for c in s:
        if c in mapping:
            top = stack.pop() if stack else '#'
            if top != mapping[c]:
                return False
        else:
            stack.append(c)
    return not stack
```

#### Min Stack (LC #155)
```python
class MinStack:
    def __init__(self):
        self.stack = []       # (val, current_min) tuples
    
    def push(self, val):
        curr_min = min(val, self.stack[-1][1]) if self.stack else val
        self.stack.append((val, curr_min))
    
    def pop(self): self.stack.pop()
    def top(self): return self.stack[-1][0]
    def getMin(self): return self.stack[-1][1]
```
All O(1).

#### Evaluate Reverse Polish Notation (LC #150)
```python
def evalRPN(tokens):
    stack = []
    ops = {'+', '-', '*', '/'}
    for t in tokens:
        if t in ops:
            b, a = stack.pop(), stack.pop()
            if t == '+': stack.append(a + b)
            elif t == '-': stack.append(a - b)
            elif t == '*': stack.append(a * b)
            else: stack.append(int(a / b))  # truncate toward zero
        else:
            stack.append(int(t))
    return stack[0]
```

#### Daily Temperatures (LC #739) — Monotonic Stack
```python
def dailyTemperatures(temperatures):
    result = [0] * len(temperatures)
    stack = []  # indices
    for i, t in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < t:
            idx = stack.pop()
            result[idx] = i - idx
        stack.append(i)
    return result
```

#### Largest Rectangle in Histogram (LC #84) — Hard
```python
def largestRectangleArea(heights):
    stack = []
    max_area = 0
    heights = heights + [0]  # sentinel to flush remaining stack
    
    for i, h in enumerate(heights):
        start = i
        while stack and stack[-1][1] > h:
            idx, height = stack.pop()
            max_area = max(max_area, height * (i - idx))
            start = idx
        stack.append((start, h))
    
    return max_area
```

---

## Part 2 — Binary Search

### 2.1 The Template

Binary search finds a target in a **sorted** space (array or "answer space") in O(log n).

**Universal Template (avoid off-by-one errors):**
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # avoid overflow (relevant in Java/C++)
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # not found
```

**When to suspect binary search:**
- Array is sorted
- Problem says "O(log n)" or "can you do better than O(n)?"
- "Find minimum/maximum value satisfying a condition"
- Any monotonic predicate (true/false transitions exactly once)

### 2.2 Binary Search on Answer Space

This is a key FAANG pattern: binary search on the **answer** rather than array indices.

```python
# Koko Eating Bananas (LC #875)
def minEatingSpeed(piles, h):
    def canFinish(speed):
        hours = sum((pile + speed - 1) // speed for pile in piles)  # ceil division
        return hours <= h
    
    left, right = 1, max(piles)
    while left <= right:
        mid = (left + right) // 2
        if canFinish(mid):
            right = mid - 1   # try smaller speed
        else:
            left = mid + 1    # need faster speed
    return left
```

**Pattern recognition:** "Find the minimum X such that condition(X) is True" → Binary search on X.

### 2.3 Find Minimum in Rotated Array (LC #153)
```python
def findMin(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            # Min is in right half
            left = mid + 1
        else:
            # Min is in left half (or mid is min)
            right = mid
    return nums[left]
```

### 2.4 Search in Rotated Sorted Array (LC #33)
```python
def search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        # Determine which half is sorted
        if nums[left] <= nums[mid]:  # left half is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
```

---

## Part 3 — Linked Lists

### 3.1 Core Operations

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Build a linked list
def build(arr):
    dummy = ListNode(0)
    curr = dummy
    for val in arr:
        curr.next = ListNode(val)
        curr = curr.next
    return dummy.next  # [1, 2, 3, 4, 5]
```

### 3.2 Fast + Slow Pointers

#### Find Middle (and base for many problems)
```python
def findMiddle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow  # slow is at middle
```

#### Detect Cycle (LC #141 / #142)
```python
def hasCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

def detectCycleStart(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # no cycle
    
    # Move one pointer to head; both advance at speed 1
    # They meet at cycle start (Floyd's proof)
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    return slow
```

#### LRU Cache (LC #146) — Hash Map + Doubly Linked List
```python
class LRUCache:
    class Node:
        def __init__(self, key=0, val=0):
            self.key = key; self.val = val
            self.prev = self.next = None
    
    def __init__(self, capacity):
        self.cap = capacity
        self.map = {}
        # Dummy head (oldest) and tail (newest)
        self.head = self.Node(); self.tail = self.Node()
        self.head.next = self.tail; self.tail.prev = self.head
    
    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _insert_tail(self, node):
        node.prev = self.tail.prev; node.next = self.tail
        self.tail.prev.next = node; self.tail.prev = node
    
    def get(self, key):
        if key not in self.map: return -1
        node = self.map[key]
        self._remove(node); self._insert_tail(node)
        return node.val
    
    def put(self, key, val):
        if key in self.map: self._remove(self.map[key])
        node = self.Node(key, val)
        self.map[key] = node; self._insert_tail(node)
        if len(self.map) > self.cap:
            lru = self.head.next
            self._remove(lru); del self.map[lru.key]
```

---

## Part 4 — Trees

### 4.1 Tree Traversals

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val; self.left = left; self.right = right

# Recursive traversals
def inorder(root):   # Left → Root → Right (BST: gives sorted order)
    return inorder(root.left) + [root.val] + inorder(root.right) if root else []

def preorder(root):  # Root → Left → Right
    return [root.val] + preorder(root.left) + preorder(root.right) if root else []

def postorder(root): # Left → Right → Root
    return postorder(root.left) + postorder(root.right) + [root.val] if root else []

# Iterative inorder (common interview ask)
def inorderIterative(root):
    result, stack = [], []
    curr = root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        result.append(curr.val)
        curr = curr.right
    return result

# BFS / Level-order
from collections import deque
def levelOrder(root):
    if not root: return []
    result, queue = [], deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result
```

### 4.2 Key Tree Problems

#### Invert Binary Tree (LC #226)
```python
def invertTree(root):
    if not root: return None
    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root
```

#### Maximum Depth (LC #104)
```python
def maxDepth(root):
    if not root: return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
```

#### Validate BST (LC #98)
```python
def isValidBST(root, lo=float('-inf'), hi=float('inf')):
    if not root: return True
    if not (lo < root.val < hi): return False
    return (isValidBST(root.left, lo, root.val) and
            isValidBST(root.right, root.val, hi))
```

#### Lowest Common Ancestor (LC #236)
```python
def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    return root if left and right else left or right
```

#### Binary Tree Maximum Path Sum (LC #124) — Hard
```python
def maxPathSum(root):
    max_sum = [float('-inf')]
    
    def dfs(node):
        if not node: return 0
        left = max(dfs(node.left), 0)   # ignore negative contributions
        right = max(dfs(node.right), 0)
        max_sum[0] = max(max_sum[0], node.val + left + right)
        return node.val + max(left, right)  # can only extend one side to parent
    
    dfs(root)
    return max_sum[0]
```

#### Kth Smallest in BST (LC #230)
```python
def kthSmallest(root, k):
    # Inorder traversal gives sorted order for BST
    stack = []
    curr = root
    while curr or stack:
        while curr:
            stack.append(curr); curr = curr.left
        curr = stack.pop()
        k -= 1
        if k == 0: return curr.val
        curr = curr.right
```

#### Serialize and Deserialize Binary Tree (LC #297) — Hard
```python
class Codec:
    def serialize(self, root):
        # Preorder with 'N' for null
        if not root: return 'N'
        return f'{root.val},{self.serialize(root.left)},{self.serialize(root.right)}'
    
    def deserialize(self, data):
        vals = iter(data.split(','))
        def build():
            v = next(vals)
            if v == 'N': return None
            node = TreeNode(int(v))
            node.left = build()
            node.right = build()
            return node
        return build()
```

---

## Part 5 — Interview Q&A (20 Questions)

**Q1: When do you use a stack vs a queue for tree traversal?**
> Stack → DFS (depth-first: go deep before wide). Queue → BFS (breadth-first: level by level). Recursive DFS implicitly uses the call stack. For iterative DFS, explicit stack. For BFS, always a queue (deque).

**Q2: What's the time complexity of tree DFS and BFS?**
> Both O(n) — you visit every node exactly once. Space: O(h) for DFS (h = tree height, worst case O(n) for skewed tree), O(w) for BFS (w = max width, worst case O(n/2) for complete tree's last level).

**Q3: Explain why the slow/fast pointer finds the middle of a linked list.**
> Fast moves 2 nodes per step, slow moves 1. When fast reaches the end (null or last node), slow has traveled exactly half the distance → middle. For even-length lists, you get the first or second middle depending on the termination condition.

**Q4: What is Floyd's cycle detection and why does the meeting point work?**
> After slow and fast meet inside a cycle, resetting one pointer to head and advancing both at speed 1 makes them meet at the cycle entry. Proof: Let `F` = steps to cycle start, `C` = cycle length. When they first meet, slow took `F + a` steps, fast took `F + a + k×C` steps. Since fast = 2×slow: `2(F+a) = F+a+kC → F = kC - a`. Starting from head (distance F) and from meeting point (distance kC - a from start = F steps forward in cycle) → they meet at cycle start.

**Q5: What are the 4 BST properties?**
> 1. All nodes in left subtree < root
> 2. All nodes in right subtree > root
> 3. Left and right subtrees are also BSTs
> 4. In-order traversal yields sorted sequence
> BST operations (insert, search, delete) are O(h): O(log n) balanced, O(n) degenerate (sorted input → linked list).

**Q6: What is the difference between a balanced BST and a tree in general?**
> Balanced BST (AVL, Red-Black) maintains h = O(log n) through rotations on insert/delete. Python's `SortedList` from sortedcontainers uses this. General BST can degenerate to O(n) height with sorted input. In interviews, assume BST = balanced unless stated otherwise.

**Q7: How would you do binary search on a predicate (boolean function)?**
> Identify: what is the monotonic property? E.g., "can complete in time T" is False for small T, True for large T. Then binary search on T: find the exact boundary.
> Template: `while left < right: mid = ...; if predicate(mid): right = mid; else: left = mid + 1; return left`

**Q8: What is the time complexity of building a BST from sorted data? How do you fix it?**
> O(n²) if you insert 1, 2, 3, ... — each insertion runs down the entire tree (skewed). Fix: use divide-and-conquer to build from sorted array. Take middle as root, recurse on left and right halves → O(n log n), balanced tree.

**Q9: What are the different ways to detect a cycle in a linked list?**
> 1. Floyd's algorithm (fast/slow pointers) — O(n), O(1)
> 2. Hash set of visited nodes — O(n), O(n)
> Floyd's is preferred in interviews for O(1) space.

**Q10: How does iterative inorder traversal work? Why would you need iterative over recursive?**
> Push all left nodes, then pop and visit, then go right. Needed when: recursion depth could cause stack overflow (very deep tree), or when you need to pause/resume traversal (generator pattern, e.g. for merge of two BSTs).

---

## Practice Tracker

| Problem | Difficulty | Topic | Solved? | Mins |
|---|---|---|---|---|
| LC #20 Valid Parentheses | Easy | Stack | ⬜ | |
| LC #155 Min Stack | Medium | Stack | ⬜ | |
| LC #150 Eval RPN | Medium | Stack | ⬜ | |
| LC #739 Daily Temperatures | Medium | Monotonic Stack | ⬜ | |
| LC #84 Largest Rectangle | Hard | Monotonic Stack | ⬜ | |
| LC #153 Find Min Rotated | Medium | Binary Search | ⬜ | |
| LC #33 Search Rotated | Medium | Binary Search | ⬜ | |
| LC #875 Koko Bananas | Medium | Binary Search on Answer | ⬜ | |
| LC #141 Linked List Cycle | Easy | Fast/Slow | ⬜ | |
| LC #146 LRU Cache | Medium | Linked List + Hash | ⬜ | |
| LC #226 Invert Tree | Easy | Tree DFS | ⬜ | |
| LC #104 Max Depth | Easy | Tree DFS | ⬜ | |
| LC #102 Level Order | Medium | Tree BFS | ⬜ | |
| LC #98 Validate BST | Medium | Tree DFS + Bounds | ⬜ | |
| LC #230 Kth Smallest BST | Medium | Iterative Inorder | ⬜ | |
| LC #236 LCA Binary Tree | Medium | Tree DFS | ⬜ | |
| LC #124 Max Path Sum | Hard | Tree DFS | ⬜ | |
| LC #297 Serialize/Deserialize | Hard | Tree Preorder | ⬜ | |
| LC #875 Koko Eating Bananas | Medium | Binary Search | ⬜ | |
| LC #142 Linked List Cycle II | Medium | Floyd's | ⬜ | |

---

## 📚 Further Resources

- **[NeetCode Stack Playlist](https://www.youtube.com/playlist?list=PLot-Xpze53lclxHFBaEdNhHBRiRrRFdSq)**
- **[NeetCode Trees Playlist](https://www.youtube.com/playlist?list=PLot-Xpze53lfoI7Q88nIH77HXlR_RZLpJ)**
- **"Cracking the Coding Interview"** Chapters 3 (Stacks/Queues) and 4 (Trees/Graphs)
- **[CS61B Trees lectures (Berkeley)](https://inst.eecs.berkeley.edu/~cs61b)**

---

## Day-to-Day Work: Stack, Binary Search & Trees in AI Engineering

```
WHERE THESE PATTERNS APPEAR AT WORK:

STACK:
  - JSON/XML parsing for document processing (balanced brackets)
  - Expression evaluation in computed features
  - Undo/redo operations in AI-assisted editors
  - Call stack debugging for agent tool chains

BINARY SEARCH:
  - Finding optimal hyperparameters (bisection-like)
  - Threshold tuning: find the cosine similarity cutoff for "relevant"
  - Log analysis: binary search on timestamps to find error window
  - Quantile computation in streaming data

TREES:
  - Decision trees in ML (literally this data structure)
  - DOM/AST parsing for code analysis agents
  - Trie for autocomplete suggestions (covered in DSA Missing Patterns)
  - Hierarchical category navigation (product taxonomy)
  - Agent decision trees: parse and execute nested plans
```

```python
# Real example: binary search for optimal similarity threshold
def find_optimal_threshold(query_results: list, ground_truth: list, metric_fn):
    """Binary search for best cosine similarity threshold for RAG retrieval."""
    low, high = 0.0, 1.0
    best_threshold, best_score = 0.5, 0.0
    
    for _ in range(20):  # 20 iterations → precision of 1e-6
        mid = (low + high) / 2
        filtered = [r for r in query_results if r["score"] >= mid]
        score = metric_fn(filtered, ground_truth)
        
        if score > best_score:
            best_score = score
            best_threshold = mid
        
        # If too many results → raise threshold; too few → lower it
        if len(filtered) > len(ground_truth) * 2:
            low = mid
        else:
            high = mid
    
    return best_threshold, best_score
```
