# DSA — Custom Data Structure Design
### Phase 1 | Month 2 | Gap-Fill Study Guide

> **Why this file exists:** "Design X data structure" problems are a dedicated interview round at Google, Meta, Amazon, and Microsoft. They test whether you understand hash maps, doubly-linked lists, heaps, and how to compose them. Every problem here has appeared in real FAANG interviews.

> 💡 **ELI5 (Explain Like I'm 5):**
> You're designing tools for a chef. An LRU cache is like a kitchen counter — you keep the most recently used ingredients on the counter, and when it's full, you push the oldest one back to the pantry. Each "custom data structure" problem asks you to build a specialised tool with specific O(1) operations using the basic building blocks you already know.

---

## The Key Building Blocks

| Building Block | Use For |
|---|---|
| Hash Map (dict) | O(1) lookup by key |
| Doubly Linked List | O(1) insert/delete at any position |
| Min/Max Heap | O(log n) access to smallest/largest |
| Deque | O(1) insert/delete at both ends |
| Sorted container (SortedList) | O(log n) rank / order queries |

**The most powerful combo:** `dict + doubly linked list` = LRU/LFU cache. Almost every "O(1) all operations" design problem uses this combination.

---

## Q303. LRU Cache (LC #146) — The Most Important Design Problem

> Least Recently Used: when cache is full, evict the item used least recently.

**Approach:** `dict` for O(1) lookup + `doubly linked list` to track recency order.

```python
class LRUCache:
    class Node:
        def __init__(self, key=0, val=0):
            self.key = key
            self.val = val
            self.prev = self.next = None

    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = {}   # key → Node
        # Sentinel head/tail: head ↔ [LRU...MRU] ↔ tail
        self.head = self.Node()
        self.tail = self.Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _insert_at_tail(self, node):
        """Insert node just before tail (most recently used position)."""
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev.next = node
        self.tail.prev = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._insert_at_tail(node)   # mark as most recently used
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])
        node = self.Node(key, value)
        self.cache[key] = node
        self._insert_at_tail(node)
        if len(self.cache) > self.cap:
            # Evict LRU: node just after head
            lru = self.head.next
            self._remove(lru)
            del self.cache[lru.key]

# cache = LRUCache(2)
# cache.put(1, 1); cache.put(2, 2); cache.get(1)  # → 1
# cache.put(3, 3)  # evicts key 2
# cache.get(2)     # → -1 (evicted)
```
**All operations: O(1)** | **Space: O(capacity)**

> 🔑 **Why doubly linked list?** We need O(1) delete from the middle. Singly linked list requires O(n) to find the prev node. Doubly linked gives us prev instantly.

---

## Q304. Modified LRU — Values Are Lists, Get Consumes from Front

```python
from collections import deque, OrderedDict

class ListLRUCache:
    """LRU where each key maps to a queue; get() pops from the front of the queue."""
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = OrderedDict()   # key → deque

    def put(self, key: int, val: int) -> None:
        if key in self.cache:
            self.cache[key].append(val)
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.cap:
                self.cache.popitem(last=False)   # evict LRU
            self.cache[key] = deque([val])

    def get(self, key: int) -> int:
        if key not in self.cache or not self.cache[key]:
            return -1
        val = self.cache[key].popleft()   # consume from front
        if not self.cache[key]:
            del self.cache[key]
        else:
            self.cache.move_to_end(key)
        return val
```

> 💡 **Python shortcut:** `collections.OrderedDict` gives you `move_to_end()` and `popitem(last=False)` which implement LRU in ~10 lines instead of building a doubly linked list manually.

---

## Q306. LFU Cache (LC #460) — Least Frequently Used

> When cache is full, evict the least frequently accessed item. On ties, evict the LRU among equal-frequency items.

```python
from collections import defaultdict, OrderedDict

class LFUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.key_to_val = {}                     # key → value
        self.key_to_freq = {}                    # key → frequency
        self.freq_to_keys = defaultdict(OrderedDict)  # freq → {key: None} (ordered)
        self.min_freq = 0

    def _update_freq(self, key):
        freq = self.key_to_freq[key]
        del self.freq_to_keys[freq][key]
        if not self.freq_to_keys[freq]:
            del self.freq_to_keys[freq]
            if self.min_freq == freq:
                self.min_freq += 1
        self.key_to_freq[key] = freq + 1
        self.freq_to_keys[freq + 1][key] = None

    def get(self, key: int) -> int:
        if key not in self.key_to_val:
            return -1
        self._update_freq(key)
        return self.key_to_val[key]

    def put(self, key: int, value: int) -> None:
        if self.cap == 0:
            return
        if key in self.key_to_val:
            self.key_to_val[key] = value
            self._update_freq(key)
        else:
            if len(self.key_to_val) >= self.cap:
                # Evict: LRU among min_freq keys
                evict_key, _ = self.freq_to_keys[self.min_freq].popitem(last=False)
                del self.key_to_val[evict_key]
                del self.key_to_freq[evict_key]
            self.key_to_val[key] = value
            self.key_to_freq[key] = 1
            self.freq_to_keys[1][key] = None
            self.min_freq = 1

# All operations: O(1)
```

---

## Q307. All O(1) Data Structure (LC #432)

> Support: `inc(key)`, `dec(key)`, `getMaxKey()`, `getMinKey()` — all O(1).

```python
from collections import defaultdict

class AllOne:
    """
    Maintain a doubly linked list where each node = one frequency bucket.
    Nodes are in ascending frequency order.
    Each bucket stores all keys with that frequency.
    """
    class Bucket:
        def __init__(self, freq):
            self.freq = freq
            self.keys = set()
            self.prev = self.next = None

    def __init__(self):
        self.key_to_bucket = {}
        self.head = self.Bucket(0)   # sentinel min
        self.tail = self.Bucket(float('inf'))  # sentinel max
        self.head.next = self.tail
        self.tail.prev = self.head

    def _insert_after(self, node, new_node):
        new_node.prev = node
        new_node.next = node.next
        node.next.prev = new_node
        node.next = new_node

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def inc(self, key: str) -> None:
        if key not in self.key_to_bucket:
            bucket = self.head
            freq = 0
        else:
            bucket = self.key_to_bucket[key]
            freq = bucket.freq
        # Move key to freq+1 bucket
        next_b = bucket.next
        if next_b.freq != freq + 1:
            new_b = self.Bucket(freq + 1)
            self._insert_after(bucket, new_b)
            next_b = new_b
        next_b.keys.add(key)
        self.key_to_bucket[key] = next_b
        if freq > 0:
            bucket.keys.discard(key)
            if not bucket.keys:
                self._remove(bucket)

    def dec(self, key: str) -> None:
        if key not in self.key_to_bucket:
            return
        bucket = self.key_to_bucket[key]
        freq = bucket.freq
        bucket.keys.discard(key)
        if freq == 1:
            del self.key_to_bucket[key]
        else:
            prev_b = bucket.prev
            if prev_b.freq != freq - 1:
                new_b = self.Bucket(freq - 1)
                self._insert_after(prev_b, new_b)
                prev_b = new_b
            prev_b.keys.add(key)
            self.key_to_bucket[key] = prev_b
        if not bucket.keys:
            self._remove(bucket)

    def getMaxKey(self) -> str:
        if self.tail.prev == self.head:
            return ""
        return next(iter(self.tail.prev.keys))

    def getMinKey(self) -> str:
        if self.head.next == self.tail:
            return ""
        return next(iter(self.head.next.keys))
```

---

## Q308. RandomizedSet (LC #380) — O(1) Insert, Delete, GetRandom

```python
import random

class RandomizedSet:
    def __init__(self):
        self.val_to_idx = {}   # value → index in list
        self.vals = []

    def insert(self, val: int) -> bool:
        if val in self.val_to_idx:
            return False
        self.vals.append(val)
        self.val_to_idx[val] = len(self.vals) - 1
        return True

    def remove(self, val: int) -> bool:
        if val not in self.val_to_idx:
            return False
        # Swap with last element, then pop last (O(1) delete)
        idx = self.val_to_idx[val]
        last = self.vals[-1]
        self.vals[idx] = last
        self.val_to_idx[last] = idx
        self.vals.pop()
        del self.val_to_idx[val]
        return True

    def getRandom(self) -> int:
        return random.choice(self.vals)  # O(1) random access by index

# All operations O(1) average
```

> 🔑 **The trick:** Random access requires an array (O(1) by index). Delete from array is O(n) unless you swap with the last element and pop — that's O(1).

---

## Q309. RandomizedCollection — Allows Duplicates (LC #381)

```python
class RandomizedCollection:
    def __init__(self):
        self.val_to_idxs = {}   # value → set of indices
        self.vals = []

    def insert(self, val: int) -> bool:
        is_new = val not in self.val_to_idxs or not self.val_to_idxs[val]
        if val not in self.val_to_idxs:
            self.val_to_idxs[val] = set()
        self.vals.append(val)
        self.val_to_idxs[val].add(len(self.vals) - 1)
        return is_new

    def remove(self, val: int) -> bool:
        if val not in self.val_to_idxs or not self.val_to_idxs[val]:
            return False
        # Pick any index of val, swap with last
        idx = next(iter(self.val_to_idxs[val]))
        last = self.vals[-1]
        self.vals[idx] = last
        self.val_to_idxs[val].remove(idx)
        self.val_to_idxs[last].add(idx)
        self.val_to_idxs[last].discard(len(self.vals) - 1)
        self.vals.pop()
        return True

    def getRandom(self) -> int:
        return random.choice(self.vals)
```

---

## Q310. Time-Based Key-Value Store (LC #981)

> `set(key, value, timestamp)` and `get(key, timestamp)` — return value at the largest timestamp ≤ given timestamp.

```python
from collections import defaultdict
from bisect import bisect_right

class TimeMap:
    def __init__(self):
        self.store = defaultdict(list)   # key → [(timestamp, value)]

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.store[key].append((timestamp, value))   # timestamps arrive in order

    def get(self, key: str, timestamp: int) -> str:
        if key not in self.store:
            return ""
        entries = self.store[key]
        # Binary search for the largest timestamp ≤ target
        lo, hi = 0, len(entries) - 1
        result = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            if entries[mid][0] <= timestamp:
                result = entries[mid][1]   # valid candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return result

# tm = TimeMap()
# tm.set("foo", "bar", 1); tm.set("foo", "bar2", 4)
# tm.get("foo", 4) → "bar2"
# tm.get("foo", 3) → "bar"
# tm.get("foo", 5) → "bar2"
```
**set:** O(1) | **get:** O(log n) | **Space:** O(total sets)

---

## Q311. Hit Counter Over Moving Time Window (LC #362)

> Count hits in the last 300 seconds. Operations: `hit(timestamp)`, `getHits(timestamp)`.

```python
from collections import deque

class HitCounter:
    def __init__(self):
        self.hits = deque()   # (timestamp, count)

    def hit(self, timestamp: int) -> None:
        if self.hits and self.hits[-1][0] == timestamp:
            # Merge same-timestamp hits
            self.hits[-1] = (timestamp, self.hits[-1][1] + 1)
        else:
            self.hits.append((timestamp, 1))

    def getHits(self, timestamp: int) -> int:
        # Remove hits older than 300 seconds
        while self.hits and self.hits[0][0] <= timestamp - 300:
            self.hits.popleft()
        return sum(count for _, count in self.hits)

# Amortized O(1) for hit, O(hits in window) for getHits
```

---

## Q312. Browser History (LC #1472)

```python
class BrowserHistory:
    def __init__(self, homepage: str):
        self.history = [homepage]
        self.current = 0

    def visit(self, url: str) -> None:
        # Truncate forward history
        self.history = self.history[:self.current + 1]
        self.history.append(url)
        self.current += 1

    def back(self, steps: int) -> str:
        self.current = max(0, self.current - steps)
        return self.history[self.current]

    def forward(self, steps: int) -> str:
        self.current = min(len(self.history) - 1, self.current + steps)
        return self.history[self.current]
```

---

## Q313. Circular Deque (LC #641)

```python
class MyCircularDeque:
    def __init__(self, k: int):
        self.data = [0] * k
        self.front = 0
        self.rear = -1
        self.size = 0
        self.cap = k

    def insertFront(self, value: int) -> bool:
        if self.isFull(): return False
        self.front = (self.front - 1) % self.cap
        self.data[self.front] = value
        self.size += 1
        if self.size == 1: self.rear = self.front
        return True

    def insertLast(self, value: int) -> bool:
        if self.isFull(): return False
        self.rear = (self.rear + 1) % self.cap
        self.data[self.rear] = value
        self.size += 1
        return True

    def deleteFront(self) -> bool:
        if self.isEmpty(): return False
        self.front = (self.front + 1) % self.cap
        self.size -= 1
        return True

    def deleteLast(self) -> bool:
        if self.isEmpty(): return False
        self.rear = (self.rear - 1) % self.cap
        self.size -= 1
        return True

    def getFront(self) -> int:
        return -1 if self.isEmpty() else self.data[self.front]

    def getRear(self) -> int:
        return -1 if self.isEmpty() else self.data[self.rear]

    def isEmpty(self) -> bool: return self.size == 0
    def isFull(self) -> bool: return self.size == self.cap
```

---

## Q314. Snapshot Array (LC #1146)

> `set(index, val)`, `snap()` returns snap_id, `get(index, snap_id)` → value at that snapshot.

```python
from bisect import bisect_right
from collections import defaultdict

class SnapshotArray:
    def __init__(self, length: int):
        # For each index, store [(snap_id, value)] pairs
        self.data = [[(0, 0)] for _ in range(length)]
        self.snap_id = 0

    def set(self, index: int, val: int) -> None:
        if self.data[index][-1][0] == self.snap_id:
            self.data[index][-1] = (self.snap_id, val)  # update current snap
        else:
            self.data[index].append((self.snap_id, val))

    def snap(self) -> int:
        self.snap_id += 1
        return self.snap_id - 1

    def get(self, index: int, snap_id: int) -> int:
        entries = self.data[index]
        # Binary search: find latest snap_id <= given snap_id
        lo, hi = 0, len(entries) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if entries[mid][0] <= snap_id:
                lo = mid
            else:
                hi = mid - 1
        return entries[lo][1]

# Space efficient: only stores changes, not full copies
```

---

## Q316. Leaderboard (LC #1244)

> `addScore(playerId, score)`, `top(K)` returns sum of top K, `reset(playerId)`.

```python
from sortedcontainers import SortedList

class Leaderboard:
    def __init__(self):
        self.scores = {}           # playerId → score
        self.sorted_scores = SortedList()

    def addScore(self, playerId: int, score: int) -> None:
        if playerId in self.scores:
            self.sorted_scores.remove(self.scores[playerId])
        self.scores[playerId] = self.scores.get(playerId, 0) + score
        self.sorted_scores.add(self.scores[playerId])

    def top(self, K: int) -> int:
        return sum(self.sorted_scores[-K:])

    def reset(self, playerId: int) -> None:
        if playerId in self.scores:
            self.sorted_scores.remove(self.scores[playerId])
            del self.scores[playerId]
```

> 💡 Without `sortedcontainers`, use a heap: `top(K)` → `heapq.nlargest(K, self.scores.values())` — O(n) but simpler.

---

## Q317. Stock Price Fluctuation (LC #2034)

> `update(timestamp, price)`, `current()`, `maximum()`, `minimum()`.

```python
from sortedcontainers import SortedList

class StockPrice:
    def __init__(self):
        self.prices = {}       # timestamp → price
        self.sorted_prices = SortedList()
        self.latest_ts = 0

    def update(self, timestamp: int, price: int) -> None:
        if timestamp in self.prices:
            self.sorted_prices.remove(self.prices[timestamp])
        self.prices[timestamp] = price
        self.sorted_prices.add(price)
        self.latest_ts = max(self.latest_ts, timestamp)

    def current(self) -> int:
        return self.prices[self.latest_ts]

    def maximum(self) -> int:
        return self.sorted_prices[-1]

    def minimum(self) -> int:
        return self.sorted_prices[0]
```

---

## Q318. Moving Average from Data Stream (LC #346)

```python
from collections import deque

class MovingAverage:
    def __init__(self, size: int):
        self.window = deque()
        self.size = size
        self.total = 0

    def next(self, val: int) -> float:
        if len(self.window) == self.size:
            self.total -= self.window.popleft()
        self.window.append(val)
        self.total += val
        return self.total / len(self.window)

# ma = MovingAverage(3)
# ma.next(1) → 1.0
# ma.next(10) → 5.5
# ma.next(3) → 4.67
# ma.next(5) → 6.0  (window: [10,3,5])
```

---

## Pattern Summary

| Problem Type | Key Insight |
|---|---|
| LRU Cache | dict + doubly linked list — O(1) all ops |
| LFU Cache | dict + freq_to_OrderedDict — track min_freq |
| All O(1) freq structure | Doubly linked list of frequency buckets |
| RandomizedSet | dict + list — swap-with-last for O(1) delete |
| Time-Based KV | dict of sorted lists + binary search |
| Moving window counter | Deque + running sum |
| Snapshot array | Per-index sorted list of (snap_id, val) pairs |
| Leaderboard / Stock | SortedList or heap for dynamic ordering |

---

## Interview Q&A

### Q1: Why is a doubly linked list used instead of singly linked?
**A:** O(1) delete from an arbitrary position requires knowing the previous node. In a singly linked list, finding the previous node takes O(n). Doubly linked gives us `node.prev` directly, making delete O(1).

### Q2: Explain the LFU cache eviction strategy.
**A:** Always evict the key with the lowest access frequency. On ties, evict the least recently used among equal-frequency keys. Implementation: maintain a `min_freq` variable and a `freq → OrderedDict` map. When frequency changes, move the key to the next bucket. O(1) all ops.

### Q3: How do you design for O(1) random element selection?
**A:** Use a list for O(1) random index access. For O(1) delete, swap the target with the last element, update the dict to point the swapped element to its new position, then pop the last. The dict maps each value to its current index in the list.

### Q4: When would you use a Fenwick Tree vs Segment Tree vs SortedList?
**A:**
- **SortedList (Python):** Simplest for dynamic ordered data (leaderboard, stock prices). O(log n) insert/delete/query.
- **Fenwick Tree / BIT:** Prefix sum queries with point updates. Very memory-efficient.
- **Segment Tree:** Range queries with range updates. More powerful but more code.

---

## LeetCode Problem List

| Q# | LC# | Problem | Difficulty |
|---|---|---|---|
| Q303 | 146 | LRU Cache | Medium |
| Q306 | 460 | LFU Cache | Hard |
| Q307 | 432 | All O(1) Data Structure | Hard |
| Q308 | 380 | Insert Delete GetRandom O(1) | Medium |
| Q309 | 381 | Insert Delete GetRandom O(1) - Duplicates | Hard |
| Q310 | 981 | Time Based Key-Value Store | Medium |
| Q311 | 362 | Design Hit Counter | Medium |
| Q312 | 1472 | Design Browser History | Medium |
| Q313 | 641 | Design Circular Deque | Medium |
| Q314 | 1146 | Snapshot Array | Medium |
| Q316 | 1244 | Design A Leaderboard | Medium |
| Q317 | 2034 | Stock Price Fluctuation | Medium |
| Q318 | 346 | Moving Average from Data Stream | Easy |

---

## Further Resources

- **NeetCode Design Problems playlist** — https://neetcode.io/roadmap
- **"Design Data-Intensive Applications"** — Kleppmann (for production data structures)
- **SortedContainers library** — https://grantjenks.com/docs/sortedcontainers/
