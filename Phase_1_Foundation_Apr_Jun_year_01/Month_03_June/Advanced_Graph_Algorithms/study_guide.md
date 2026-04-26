# Advanced Graph Algorithms — BFS, Dijkstra, Union-Find, Topo Sort
### Phase 1 | June 2026 | Weeks 1–2 Extension

> **Why graphs are critical:** Almost every real distributed system can be modelled as a graph — service dependencies, knowledge graphs, social networks, road networks. Advanced graph algorithms are tested in senior coding interviews because they demand both conceptual clarity and clean implementation under pressure.

> 💡 **ELI5 (Explain Like I'm 5):**
> A graph is just a map. Cities are nodes, roads are edges. BFS explores the nearest cities first (layer by layer). Dijkstra finds the *cheapest* route. Union-Find tracks which cities are connected. Topological Sort figures out which city must be visited before which (like prerequisites in a course).

---

## Part 1 — BFS for Shortest Path

### 1.1 Standard BFS Template
```python
from collections import deque

def bfs(graph: dict, start: int, end: int) -> int:
    """Returns shortest path length (unweighted)."""
    queue = deque([(start, 0)])   # (node, distance)
    visited = {start}
    
    while queue:
        node, dist = queue.popleft()
        if node == end:
            return dist
        for neighbour in graph[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, dist + 1))
    
    return -1  # unreachable
```

### 1.2 Word Ladder (BFS on Implicit Graph)
**Problem:** Transform word A → word B, changing one letter at a time. All intermediates must be in wordList.

```python
def ladderLength(beginWord: str, endWord: str, wordList: list[str]) -> int:
    word_set = set(wordList)
    if endWord not in word_set:
        return 0
    
    queue = deque([(beginWord, 1)])
    visited = {beginWord}
    
    while queue:
        word, steps = queue.popleft()
        
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i+1:]
                if new_word == endWord:
                    return steps + 1
                if new_word in word_set and new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, steps + 1))
    
    return 0

# Bidirectional BFS (much faster for large graphs):
def ladderLength_bidir(beginWord, endWord, wordList):
    word_set = set(wordList)
    if endWord not in word_set:
        return 0
    
    begin_set = {beginWord}
    end_set = {endWord}
    visited = set()
    steps = 1
    
    while begin_set and end_set:
        # Always expand smaller frontier
        if len(begin_set) > len(end_set):
            begin_set, end_set = end_set, begin_set
        
        next_set = set()
        for word in begin_set:
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    new_word = word[:i] + c + word[i+1:]
                    if new_word in end_set:
                        return steps + 1
                    if new_word in word_set and new_word not in visited:
                        next_set.add(new_word)
                        visited.add(new_word)
        begin_set = next_set
        steps += 1
    
    return 0
```

### 1.3 0-1 BFS (Deque with Weights 0 or 1)
```python
from collections import deque

def minCostPath(grid, start, end):
    """BFS where some edges cost 0 and some cost 1."""
    # Use deque: cost-0 edges go to front, cost-1 go to back
    dist = [[float('inf')] * len(grid[0]) for _ in range(len(grid))]
    dist[start[0]][start[1]] = 0
    dq = deque([start])
    
    while dq:
        r, c = dq.popleft()
        for nr, nc, cost in get_neighbors(r, c, grid):
            new_dist = dist[r][c] + cost
            if new_dist < dist[nr][nc]:
                dist[nr][nc] = new_dist
                if cost == 0:
                    dq.appendleft((nr, nc))  # free — go to front
                else:
                    dq.append((nr, nc))       # cost 1 — go to back
    
    return dist[end[0]][end[1]]
```

---

## Part 2 — Dijkstra's Algorithm

### 2.1 Standard Dijkstra (Min-Heap)
```python
import heapq

def dijkstra(graph: dict, start: int) -> dict:
    """
    graph = {node: [(weight, neighbour), ...]}
    Returns dist dict: shortest distance from start to all nodes.
    """
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    heap = [(0, start)]   # (distance, node)
    
    while heap:
        d, node = heapq.heappop(heap)
        
        if d > dist[node]:   # stale entry — skip
            continue
        
        for weight, neighbour in graph[node]:
            new_dist = d + weight
            if new_dist < dist[neighbour]:
                dist[neighbour] = new_dist
                heapq.heappush(heap, (new_dist, neighbour))
    
    return dist

# LC 743: Network Delay Time
def networkDelayTime(times: list[list[int]], n: int, k: int) -> int:
    graph = {i: [] for i in range(1, n + 1)}
    for u, v, w in times:
        graph[u].append((w, v))
    
    dist = dijkstra(graph, k)
    max_dist = max(dist.values())
    return max_dist if max_dist < float('inf') else -1
```

**Time:** O((V + E) log V) | **Space:** O(V + E)

### 2.2 Dijkstra with Path Reconstruction
```python
def dijkstra_with_path(graph, start, end):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    prev = {node: None for node in graph}
    heap = [(0, start)]
    
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        if node == end:
            break
        for weight, neighbour in graph[node]:
            new_dist = d + weight
            if new_dist < dist[neighbour]:
                dist[neighbour] = new_dist
                prev[neighbour] = node
                heapq.heappush(heap, (new_dist, neighbour))
    
    # Reconstruct path
    path = []
    curr = end
    while curr is not None:
        path.append(curr)
        curr = prev[curr]
    return list(reversed(path)), dist[end]
```

### 2.3 Swim in Rising Water (Dijkstra on Grid)
```python
def swimInWater(grid: list[list[int]]) -> int:
    n = len(grid)
    dist = [[float('inf')] * n for _ in range(n)]
    dist[0][0] = grid[0][0]
    heap = [(grid[0][0], 0, 0)]
    
    while heap:
        t, r, c = heapq.heappop(heap)
        if t > dist[r][c]:
            continue
        if r == n-1 and c == n-1:
            return t
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                new_t = max(t, grid[nr][nc])  # bottleneck: max of path
                if new_t < dist[nr][nc]:
                    dist[nr][nc] = new_t
                    heapq.heappush(heap, (new_t, nr, nc))
    
    return dist[n-1][n-1]
```

---

## Part 3 — Union-Find (Disjoint Set Union)

### 3.1 Full Implementation with Path Compression + Union by Rank
```python
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # already connected
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.components -= 1
        return True
    
    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)
```

**Time:** Nearly O(1) amortised per operation (O(α(n)) — inverse Ackermann)

### 3.2 Number of Connected Components
```python
def countComponents(n: int, edges: list[list[int]]) -> int:
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    return uf.components
```

### 3.3 Kruskal's Minimum Spanning Tree
```python
def minCostConnectPoints(points: list[list[int]]) -> int:
    n = len(points)
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
            edges.append((dist, i, j))
    
    edges.sort()  # sort by weight
    uf = UnionFind(n)
    total_cost = 0
    edges_used = 0
    
    for cost, u, v in edges:
        if uf.union(u, v):
            total_cost += cost
            edges_used += 1
            if edges_used == n - 1:  # MST has n-1 edges
                break
    
    return total_cost
```

### 3.4 Detect Cycle in Undirected Graph
```python
def hasCycle(n: int, edges: list[list[int]]) -> bool:
    uf = UnionFind(n)
    for u, v in edges:
        if not uf.union(u, v):  # already connected = cycle!
            return True
    return False
```

---

## Part 4 — Topological Sort

### 4.1 Kahn's Algorithm (BFS-based)
```python
from collections import deque

def topologicalSort(n: int, prerequisites: list[list[int]]) -> list[int]:
    """
    Returns topological order or [] if cycle exists.
    """
    in_degree = [0] * n
    graph = [[] for _ in range(n)]
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # Start with all nodes that have no prerequisites
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    order = []
    
    while queue:
        node = queue.popleft()
        order.append(node)
        
        for neighbour in graph[node]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)
    
    return order if len(order) == n else []  # [] means cycle

# LC 207: Course Schedule
def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:
    return len(topologicalSort(numCourses, prerequisites)) == numCourses

# LC 210: Course Schedule II
def findOrder(numCourses: int, prerequisites: list[list[int]]) -> list[int]:
    return topologicalSort(numCourses, prerequisites)
```

### 4.2 DFS-based Topological Sort
```python
def topologicalSort_dfs(n, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
    
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    result = []
    has_cycle = [False]
    
    def dfs(node):
        if has_cycle[0]: return
        color[node] = GRAY  # in progress
        for neighbour in graph[node]:
            if color[neighbour] == GRAY:
                has_cycle[0] = True  # back edge = cycle
                return
            if color[neighbour] == WHITE:
                dfs(neighbour)
        color[node] = BLACK
        result.append(node)  # append AFTER visiting all descendants
    
    for i in range(n):
        if color[i] == WHITE:
            dfs(i)
    
    return [] if has_cycle[0] else result[::-1]
```

### 4.3 Alien Dictionary (Advanced Topo Sort)
**Problem:** Given sorted alien words, derive character ordering.

```python
def alienOrder(words: list[str]) -> str:
    # Build adjacency: for each adjacent pair of words, find first diff char
    adj = {c: set() for word in words for c in word}
    in_degree = {c: 0 for c in adj}
    
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i+1]
        min_len = min(len(w1), len(w2))
        
        # Invalid: "abc" before "ab" (longer word first)
        if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
            return ""
        
        for j in range(min_len):
            if w1[j] != w2[j]:
                if w2[j] not in adj[w1[j]]:
                    adj[w1[j]].add(w2[j])
                    in_degree[w2[j]] += 1
                break
    
    # Kahn's BFS
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    result = []
    
    while queue:
        c = queue.popleft()
        result.append(c)
        for neighbour in adj[c]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)
    
    return "".join(result) if len(result) == len(in_degree) else ""
```

---

## Part 5 — Advanced Graph Patterns

### 5.1 Graph Colouring / Bipartite Check
```python
def isBipartite(graph: list[list[int]]) -> bool:
    color = [-1] * len(graph)
    
    for start in range(len(graph)):
        if color[start] != -1:
            continue
        queue = deque([start])
        color[start] = 0
        
        while queue:
            node = queue.popleft()
            for neighbour in graph[node]:
                if color[neighbour] == -1:
                    color[neighbour] = 1 - color[node]  # flip color
                    queue.append(neighbour)
                elif color[neighbour] == color[node]:
                    return False  # same color = not bipartite
    
    return True
```

### 5.2 Tarjan's Algorithm — Strongly Connected Components
```python
def tarjan_scc(graph: dict, n: int) -> list[list[int]]:
    """Find all SCCs in directed graph."""
    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    on_stack = {}
    sccs = []
    
    def strongconnect(v):
        index[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        
        for w in graph.get(v, []):
            if w not in index:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif on_stack[w]:
                lowlinks[v] = min(lowlinks[v], index[w])
        
        if lowlinks[v] == index[v]:  # v is root of SCC
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)
    
    for v in range(n):
        if v not in index:
            strongconnect(v)
    
    return sccs
```

---

## Interview Q&A

### Q1: When do you use Dijkstra vs BFS?
**A:** BFS for **unweighted** graphs (shortest path = fewest edges). Dijkstra for **weighted** graphs with **non-negative** weights. For negative weights, use Bellman-Ford. For grids with 0/1 weights, 0-1 BFS (deque) is faster than Dijkstra.

### Q2: What is Union-Find used for?
**A:** Tracking connected components dynamically as edges are added. Key applications: detecting cycles in undirected graphs, Kruskal's MST, network connectivity, percolation problems. Near-O(1) per operation with path compression + union by rank.

### Q3: Why is Kahn's algorithm preferred for cycle detection in directed graphs?
**A:** Kahn's naturally detects cycles: if the topological sort doesn't include all nodes, a cycle exists (nodes with cyclic dependencies never reach in-degree 0). It's also iterative (no recursion stack overflow risk) and produces the actual sort order simultaneously.

### Q4: What is the difference between an SCC and a connected component?
**A:** In an **undirected** graph, a connected component is a set of nodes reachable from each other. In a **directed** graph, an SCC (Strongly Connected Component) is a maximal set where every node is reachable from every other node *in both directions*. Tarjan's and Kosaraju's find SCCs.

### Q5: How do you detect a negative cycle?
**A:** Run Bellman-Ford for V iterations (instead of V-1). If any distance still decreases on the V-th iteration, a negative cycle exists. Dijkstra cannot detect negative cycles and gives wrong answers with negative weights.

### Q6: Design a system to check if any two services in a microservices dependency graph have a circular dependency.
**A:** Model as directed graph (service A → B means A depends on B). Run topological sort (Kahn's or DFS). If sort includes all N services, no cycle. Otherwise, services that never reach in-degree 0 are in a cycle. Time O(V+E). For incremental updates (new dependency added), use Union-Find to quickly check if adding edge u→v creates cycle.

---

## LeetCode Problem List

| # | Problem | Algorithm | Difficulty |
|---|---|---|---|
| 127 | Word Ladder | BFS | Hard |
| 752 | Open the Lock | BFS | Medium |
| 743 | Network Delay Time | Dijkstra | Medium |
| 778 | Swim in Rising Water | Dijkstra | Hard |
| 1631 | Path With Minimum Effort | Dijkstra | Medium |
| 787 | Cheapest Flights K Stops | Dijkstra/Bellman-Ford | Medium |
| 207 | Course Schedule | Topo Sort | Medium |
| 210 | Course Schedule II | Topo Sort | Medium |
| 269 | Alien Dictionary | Topo Sort | Hard |
| 684 | Redundant Connection | Union-Find | Medium |
| 1584 | Min Cost to Connect All Points | Kruskal's (UF) | Medium |
| 323 | Number of Connected Components | Union-Find | Medium |
| 547 | Number of Provinces | BFS/Union-Find | Medium |
| 785 | Is Graph Bipartite? | BFS Colouring | Medium |
| 1319 | Number of Operations to Connect Network | Union-Find | Medium |

---

## Further Resources

- **CP Algorithms** — https://cp-algorithms.com/ (Dijkstra, SCC, MST with proofs)
- **William Fiset Graph Series (YouTube)** — best visual explanation of all graph algorithms
- **"Algorithm Design" by Kleinberg & Tardos** — Chapter 3-4 for formal graph theory
