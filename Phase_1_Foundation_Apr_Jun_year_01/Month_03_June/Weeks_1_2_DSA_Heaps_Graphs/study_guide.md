# Weeks 1–2: Heaps & Graph Algorithms (BFS/DFS)
### Phase 1 | Month 3 | June 2–15, 2026

---

## 🎯 Learning Objectives

- Implement a min/max heap and know when to use it
- Apply BFS for shortest path, level-order, connected components
- Apply DFS for cycle detection, topological sort, island problems
- Solve 20 LeetCode problems on these topics

---

## Part 1 — Heaps (Priority Queue)

> 📖 **Big picture:** A heap is a tree-based structure that answers one question extremely efficiently: "What’s the current minimum (or maximum)?"
>
> **The analogy — hospital triage:** In A&E, patients don’t get seen in the order they arrive. The most critical case gets seen first. As new patients arrive, the queue reorders. A heap is the data structure behind this: you push items in any order, and you always pop the highest-priority item (smallest number = most urgent) in O(log n).
>
> **When to use:** "K largest/smallest/closest" problems, "always process the minimum next" (Dijkstra’s), "merge K sorted lists", "median of a stream". If you see K + something in the problem, try a heap.
>
> **Python’s `heapq` is a min-heap.** To simulate a max-heap, negate all values.

### 1.1 Heap Properties

A **min-heap** is a complete binary tree where every parent ≤ its children. Stored as an array:
```
parent(i) = (i-1) // 2
left(i)   = 2*i + 1
right(i)  = 2*i + 2

Array: [1, 3, 2, 7, 4, 5, 6]
Tree:      1
         /   \
        3     2
       / \   / \
      7   4 5   6
```

Python's `heapq` is a **min-heap**. For max-heap, negate values.

```python
import heapq

# Min-heap operations
h = []
heapq.heappush(h, 3)
heapq.heappush(h, 1)
heapq.heappush(h, 4)
print(heapq.heappop(h))  # 1 (minimum)
print(h[0])              # 3 (peek without removal)

# Build heap from list: O(n) — faster than pushing one by one O(n log n)
h = [3, 1, 4, 1, 5, 9, 2]
heapq.heapify(h)

# Max-heap trick (negate values)
max_heap = []
for n in [3, 1, 4, 1, 5]:
    heapq.heappush(max_heap, -n)
print(-heapq.heappop(max_heap))  # 5

# Heap with tuples (sort by first element)
tasks = [(3, "low"), (1, "high"), (2, "medium")]
heapq.heapify(tasks)
priority, name = heapq.heappop(tasks)  # (1, "high")
```

| Operation | Time |
|---|---|
| Push | O(log n) |
| Pop (min/max) | O(log n) |
| Peek | O(1) |
| Build heap | O(n) |
| heapq.nlargest(k) | O(n log k) |

### 1.2 Kth Largest Element (LC #215)

```python
import heapq

# Approach 1: Min-heap of size k
def findKthLargest(nums, k):
    heap = []
    for n in nums:
        heapq.heappush(heap, n)
        if len(heap) > k:
            heapq.heappop(heap)  # remove smallest
    return heap[0]  # kth largest = smallest in k-heap

# O(n log k) time, O(k) space — better if k << n

# Approach 2: QuickSelect O(n) average (see below)
```

### 1.3 K Closest Points to Origin (LC #973)

```python
def kClosest(points, k):
    # negate distance for max-heap trick
    heap = [(-x**2 - y**2, x, y) for x, y in points[:k]]
    heapq.heapify(heap)
    
    for x, y in points[k:]:
        dist = -x**2 - y**2
        if dist > heap[0][0]:  # closer than current worst
            heapq.heapreplace(heap, (dist, x, y))
    
    return [(x, y) for _, x, y in heap]
```

### 1.4 Merge K Sorted Lists (LC #23) — Heap

```python
def mergeKLists(lists):
    heap = []
    result_dummy = ListNode(0)
    curr = result_dummy
    
    # Initialise heap with first node of each list
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))
    
    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    
    return result_dummy.next
```
- Time: O(N log k) where N = total nodes, k = number of lists

### 1.5 Find Median from Data Stream (LC #295) — Hard

```python
class MedianFinder:
    def __init__(self):
        self.small = []   # max-heap (negate) — smaller half
        self.large = []   # min-heap — larger half
    
    def addNum(self, num):
        # Always push to small first
        heapq.heappush(self.small, -num)
        
        # Rebalance: small's max <= large's min
        if self.large and -self.small[0] > self.large[0]:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        
        # Keep sizes balanced (differ by at most 1)
        if len(self.small) > len(self.large) + 1:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        elif len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))
    
    def findMedian(self):
        if len(self.small) == len(self.large):
            return (-self.small[0] + self.large[0]) / 2
        return -self.small[0]  # odd: small has one extra
```

---

## Part 2 — Graphs: BFS

> 📖 **Big picture:** A graph is the most general data structure — nodes connected by edges. Trees are a special case of graphs (connected, no cycles). Real-world problems modelled as graphs: maps (road networks), social networks (friendships), dependency graphs (task scheduling), web pages (hyperlinks).
>
> **BFS (Breadth-First Search)** explores level by level, like ripples spreading out from a stone thrown in water. It uses a queue (FIFO). It’s the right choice for **shortest path in an unweighted graph**, because it guarantees the first time it reaches a node is via the shortest path.
>
> **The key pattern:** BFS always needs a `visited` set to avoid re-processing nodes in graphs with cycles (unlike trees which have no cycles). Forgetting visited is one of the most common interview bugs.

### 2.1 Graph Representations

```python
# Adjacency List (most common)
graph = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 4],
    3: [1],
    4: [2]
}

# For grid problems: implicit graph (cells are nodes, edges connect adjacent cells)
# directions = [(0,1), (0,-1), (1,0), (-1,0)]
```

### 2.2 BFS Template

BFS explores level by level — guarantees **shortest path** in unweighted graphs.

```python
from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result
```

### 2.3 Number of Islands (LC #200) — BFS/DFS on Grid

```python
def numIslands(grid):
    if not grid: return 0
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    def bfs(r, c):
        queue = deque([(r, c)])
        grid[r][c] = '0'  # mark visited
        while queue:
            row, col = queue.popleft()
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = row+dr, col+dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                    queue.append((nr, nc))
                    grid[nr][nc] = '0'
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                bfs(r, c)
                count += 1
    
    return count
```

### 2.4 Shortest Path (BFS with Distance)

```python
def shortestPath(graph, start, end):
    queue = deque([(start, 0)])   # (node, distance)
    visited = {start}
    
    while queue:
        node, dist = queue.popleft()
        if node == end: return dist
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    
    return -1  # no path

# Word Ladder (LC #127) — BFS on word graph
def ladderLength(beginWord, endWord, wordList):
    word_set = set(wordList)
    if endWord not in word_set: return 0
    
    queue = deque([(beginWord, 1)])
    visited = {beginWord}
    
    while queue:
        word, steps = queue.popleft()
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i+1:]
                if next_word == endWord: return steps + 1
                if next_word in word_set and next_word not in visited:
                    visited.add(next_word)
                    queue.append((next_word, steps + 1))
    return 0
```

---

## Part 3 — Graphs: DFS

> 📖 **When to use DFS vs BFS:** BFS finds shortest paths in unweighted graphs. DFS dives deep, exploring as far as possible before backtracking. Use DFS for: detecting cycles, topological sort (ordering dependencies), finding all paths, checking connectivity, and "island" counting problems (count connected components in a grid).
>
> **Recursive DFS is elegant but dangerous for very large inputs** (Python’s default recursion limit is 1000). For production code or large graphs, use iterative DFS with an explicit stack.

### 3.1 DFS Template

```python
def dfs(graph, node, visited=None):
    if visited is None: visited = set()
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited
```

### 3.2 Topological Sort (Course Schedule — LC #207, #210)

```python
def canFinish(numCourses, prerequisites):
    # Build adjacency list
    adj = defaultdict(list)
    for a, b in prerequisites: adj[b].append(a)
    
    # 0 = unvisited, 1 = in-progress (cycle detection), 2 = done
    state = [0] * numCourses
    
    def dfs(node):
        if state[node] == 1: return False  # cycle!
        if state[node] == 2: return True   # already processed
        state[node] = 1
        for neighbor in adj[node]:
            if not dfs(neighbor): return False
        state[node] = 2
        return True
    
    return all(dfs(i) for i in range(numCourses))

# LC #210: Return actual topological order
def findOrder(numCourses, prerequisites):
    adj = defaultdict(list)
    for a, b in prerequisites: adj[b].append(a)
    
    state = [0] * numCourses
    order = []
    
    def dfs(node):
        if state[node] == 1: return False
        if state[node] == 2: return True
        state[node] = 1
        for neighbor in adj[node]:
            if not dfs(neighbor): return False
        state[node] = 2
        order.append(node)
        return True
    
    if not all(dfs(i) for i in range(numCourses)): return []
    return order[::-1]
```

### 3.3 Pacific Atlantic Water Flow (LC #417)

```python
def pacificAtlantic(heights):
    rows, cols = len(heights), len(heights[0])
    pacific = set()
    atlantic = set()
    
    def dfs(r, c, visited, prev_height):
        if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols:
            return
        if heights[r][c] < prev_height: return
        visited.add((r, c))
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            dfs(r+dr, c+dc, visited, heights[r][c])
    
    # Reverse DFS: start from ocean borders, find cells that can reach them
    for r in range(rows):
        dfs(r, 0, pacific, heights[r][0])        # Pacific left
        dfs(r, cols-1, atlantic, heights[r][cols-1])  # Atlantic right
    
    for c in range(cols):
        dfs(0, c, pacific, heights[0][c])        # Pacific top
        dfs(rows-1, c, atlantic, heights[rows-1][c])  # Atlantic bottom
    
    return list(pacific & atlantic)
```

### 3.4 Clone Graph (LC #133)

```python
def cloneGraph(node):
    if not node: return None
    cloned = {}
    
    def dfs(n):
        if n in cloned: return cloned[n]
        copy = Node(n.val)
        cloned[n] = copy
        for neighbor in n.neighbors:
            copy.neighbors.append(dfs(neighbor))
        return copy
    
    return dfs(node)
```

### 3.5 Union-Find (Disjoint Set Union)

Essential for connected components and cycle detection in undirected graphs.

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return False  # already connected
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.components -= 1
        return True

# Number of Connected Components (LC #323)
def countComponents(n, edges):
    uf = UnionFind(n)
    for a, b in edges:
        uf.union(a, b)
    return uf.components
```

---

## Part 4 — Dijkstra's Algorithm (Weighted Shortest Path)
> 📖 **When BFS isn’t enough:** BFS finds shortest paths in *unweighted* graphs (all edges have equal cost). But what if roads have different travel times? Dijkstra uses a min-heap to always process the cheapest unvisited node next. It’s like BFS, but instead of a regular queue, it uses a priority queue so the "closest" node is always processed first.
>
> **The key invariant:** Once a node is popped from the heap, its shortest distance is final. Why? Because all future paths through the heap will be at least as costly (heap always pops the minimum).
```python
import heapq

def dijkstra(graph, start):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    heap = [(0, start)]   # (cost, node)
    
    while heap:
        cost, node = heapq.heappop(heap)
        if cost > dist[node]: continue  # stale entry
        
        for neighbor, weight in graph[node]:
            new_cost = cost + weight
            if new_cost < dist[neighbor]:
                dist[neighbor] = new_cost
                heapq.heappush(heap, (new_cost, neighbor))
    
    return dist

# Network Delay Time (LC #743)
def networkDelayTime(times, n, k):
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))
    
    dist = dijkstra(graph, k)
    max_dist = max(d for d in dist.values() if d != float('inf'))
    return max_dist if len(dist) == n else -1
```
- Time: O((V + E) log V) with binary heap

---

## Part 5 — Interview Q&A (15 Questions)

**Q1: When do you use BFS vs DFS?**
> BFS: shortest path in unweighted graph, level-order traversal, minimum steps. DFS: topological sort, cycle detection, connected components, maze solving, exploring all paths. BFS uses a queue; DFS uses a stack (or recursion).

**Q2: What's the time complexity of DFS/BFS on a graph?**
> O(V + E) where V = vertices, E = edges. Each vertex and edge is visited at most once.

**Q3: When does recursion fail for DFS?**
> Very deep graphs can cause stack overflow (Python default recursion limit: 1000). Fix: `sys.setrecursionlimit(10000)` or use iterative DFS with an explicit stack.

**Q4: Explain Union-Find path compression.**
> When finding root of x, set every node on the path to point directly to the root. This flattens the tree, making future find operations O(1) amortised. Combined with union by rank: O(α(n)) ≈ O(1) per operation.

**Q5: Why is Dijkstra wrong for negative edge weights?**
> Once a node is "settled" (popped from heap), Dijkstra assumes no shorter path exists. With negative edges, a later path through negative-weight edges could be shorter. Use Bellman-Ford instead (O(VE)).

**Q6: How do you detect a cycle in a directed vs undirected graph?**
> Directed: DFS with 3-state coloring (unvisited/in-progress/done). In-progress node encountered = cycle.
> Undirected: DFS tracking parent — if we reach a visited node that isn't the parent = cycle. Or Union-Find: cycle if two endpoints already share a root.

**Q7: What is a topological sort and when is it needed?**
> Linear ordering of vertices where for every edge u→v, u comes before v. Requires DAG (directed acyclic graph). Used for: task scheduling, build systems, course prerequisites. Algorithms: DFS-based (Tarjan's), Kahn's algorithm (BFS-based, iteratively removes in-degree-0 nodes).

**Q8: What's the most efficient way to check if a graph is bipartite?**
> BFS/DFS coloring: assign alternating colours. If you ever try to assign the same colour to a node as its neighbour, it's not bipartite. O(V+E).

**Q9: When would you use Dijkstra vs A*?**
> Dijkstra: no heuristic, finds shortest path to ALL nodes, O((V+E) log V). A*: uses domain-specific heuristic (e.g. Euclidean distance for grid) to guide search toward target — faster in practice for single target. A* = Dijkstra + heuristic. heuristic must be admissible (never overestimates).

**Q10: How does a heap-based priority queue differ from a sorted array?**
> Heap: O(log n) insert and extract-min, O(1) peek. Sorted array: O(n) insert (find position), O(1) extract-min. Heap is better when you have many inserts/removes. Sorted array is better when you mostly query and rarely insert.

---

## Practice Tracker

| Problem | Difficulty | Topic | Solved? |
|---|---|---|---|
| LC #215 Kth Largest Element | Medium | Heap | ⬜ |
| LC #973 K Closest Points | Medium | Heap | ⬜ |
| LC #295 Find Median Stream | Hard | Heap (2 heaps) | ⬜ |
| LC #23 Merge K Sorted Lists | Hard | Heap | ⬜ |
| LC #200 Number of Islands | Medium | BFS/DFS Grid | ⬜ |
| LC #133 Clone Graph | Medium | DFS | ⬜ |
| LC #207 Course Schedule | Medium | Topological Sort | ⬜ |
| LC #210 Course Schedule II | Medium | Topological Sort | ⬜ |
| LC #417 Pacific Atlantic | Medium | DFS from borders | ⬜ |
| LC #323 Connected Components | Medium | Union-Find | ⬜ |
| LC #743 Network Delay Time | Medium | Dijkstra | ⬜ |
| LC #127 Word Ladder | Hard | BFS | ⬜ |
| LC #787 Cheapest Flights K Stops | Medium | Bellman-Ford/Dijkstra | ⬜ |
| LC #684 Redundant Connection | Medium | Union-Find cycle | ⬜ |
| LC #130 Surrounded Regions | Medium | DFS/BFS border | ⬜ |

## 📚 Further Resources
- **[NeetCode Graphs Playlist](https://www.youtube.com/playlist?list=PLot-Xpze53lcBFp-e2nRpJVN9hEPZkBe3)**
- **[CLRS Algorithm Book]** — Chapters 22–24 (Graph algorithms)

---

## Day-to-Day Work: Heaps & Graphs in AI Engineering

```
WHERE THESE PATTERNS APPEAR AT WORK:

HEAPS / PRIORITY QUEUES:
  - Top-K retrieval: retrieve K most similar documents (min-heap of scores)
  - Task scheduling: priority queue for agent task execution
  - vLLM scheduler: priority queue for request batching
  - A* search in agent planning (priority queue of partial solutions)
  - Stream processing: maintain top-N items in streaming data

GRAPHS:
  - Knowledge graphs: entity relationships for enhanced RAG
  - DAG scheduling: Airflow/dbt dependency resolution = topological sort!
  - Multi-agent communication: agents as nodes, messages as edges
  - Dependency resolution: model dependencies, feature dependencies
  - Social network analysis for recommendation systems
  - BFS for web crawling / document link traversal for RAG

UNION-FIND:
  - Entity resolution: group duplicate products/customers
  - Clustering: connected components in similarity graphs
  - Network connectivity: is the agent communication graph fully connected?
```

```python
# Real example: top-K retrieval using a heap
import heapq
import numpy as np

def top_k_similar(query_embedding, all_embeddings, k=5):
    """Retrieve top-K most similar documents using a min-heap.
    This is literally what vector databases do internally."""
    heap = []  # min-heap of (-score, index)
    
    for i, emb in enumerate(all_embeddings):
        score = np.dot(query_embedding, emb)  # cosine sim (if normalised)
        
        if len(heap) < k:
            heapq.heappush(heap, (score, i))
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, i))
    
    # Return sorted by score (highest first)
    return sorted(heap, reverse=True)

# Real example: topological sort for pipeline dependencies (= Airflow)
from collections import defaultdict, deque

def execution_order(tasks: dict) -> list:
    """Given tasks with dependencies, find valid execution order.
    tasks = {"embed": [], "chunk": [], "index": ["chunk", "embed"], "query": ["index"]}
    """
    in_degree = defaultdict(int)
    graph = defaultdict(list)
    
    for task, deps in tasks.items():
        in_degree.setdefault(task, 0)
        for dep in deps:
            graph[dep].append(task)
            in_degree[task] += 1
    
    queue = deque([t for t in in_degree if in_degree[t] == 0])
    order = []
    
    while queue:
        task = queue.popleft()
        order.append(task)
        for next_task in graph[task]:
            in_degree[next_task] -= 1
            if in_degree[next_task] == 0:
                queue.append(next_task)
    
    return order  # ["chunk", "embed", "index", "query"]
```
