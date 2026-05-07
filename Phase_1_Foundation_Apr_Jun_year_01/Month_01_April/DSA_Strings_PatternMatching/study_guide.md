# DSA — Strings & Pattern Matching
### Phase 1 | Month 1 | Gap-Fill Study Guide

> **Why this file exists:** Strings appear in every FAANG interview. KMP, Z-algorithm, sliding window on strings, and palindrome problems are asked at Google, Meta, and Amazon constantly. This guide covers all 24 string problems from the gist that were missing from the curriculum.

> 💡 **ELI5 (Explain Like I'm 5):**
> Strings are just arrays of characters. Most string problems are really array problems in disguise. The same sliding window, two-pointer, and hash map tricks apply — just with characters instead of numbers. The only truly new ideas are KMP (for fast substring search) and expand-around-center (for palindromes).

---

## TABLE OF CONTENTS

1. [Basic String Operations](#part-1)
2. [Sliding Window on Strings](#part-2)
3. [Palindrome Problems](#part-3)
4. [String Encoding / Decoding](#part-4)
5. [KMP & Z-Algorithm](#part-5)
6. [Interview Q&A](#part-6)

---

<a name="part-1"></a>
## Part 1 — Basic String Operations

### Q25. Longest Common Prefix (LC #14)

```python
def longestCommonPrefix(strs: list[str]) -> str:
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        # Shrink prefix until it matches the start of s
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix

# strs = ["flower","flow","flight"] → "fl"
# strs = ["dog","racecar","car"]   → ""
```
**Time:** O(S) where S = total characters. **Space:** O(1)

---

### Q26. Valid Palindrome (LC #125)

```python
def isPalindrome(s: str) -> bool:
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

# "A man, a plan, a canal: Panama" → True
# "race a car" → False
```

---

### Q27. Valid Palindrome II — At Most One Deletion (LC #680)

```python
def validPalindrome(s: str) -> bool:
    def is_palindrome(left, right):
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1; right -= 1
        return True

    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            # Try skipping left OR skipping right
            return is_palindrome(left + 1, right) or is_palindrome(left, right - 1)
        left += 1; right -= 1
    return True

# "abca" → True (delete 'b' or 'c')
# "abc"  → False
```

---

### Q28. First Non-Repeating Character (LC #387)

```python
from collections import Counter

def firstUniqChar(s: str) -> int:
    count = Counter(s)
    for i, c in enumerate(s):
        if count[c] == 1:
            return i
    return -1

# "leetcode" → 0 ('l')
# "aabb"     → -1
```
**Time:** O(n) | **Space:** O(1) (26 letters max)

---

### Q29. String Compression (LC #443)

```python
def compress(chars: list[str]) -> int:
    write = 0
    read = 0
    while read < len(chars):
        char = chars[read]
        count = 0
        # Count consecutive same chars
        while read < len(chars) and chars[read] == char:
            read += 1
            count += 1
        chars[write] = char
        write += 1
        if count > 1:
            for c in str(count):
                chars[write] = c
                write += 1
    return write

# ['a','a','b','b','c','c','c'] → 6, array becomes ['a','2','b','2','c','3']
```

---

### Q30 & Q31. Anagram Check & Group Anagrams (LC #242, #49)

```python
def isAnagram(s: str, t: str) -> bool:
    return Counter(s) == Counter(t)

def groupAnagrams(strs: list[str]) -> list[list[str]]:
    groups = {}
    for s in strs:
        key = tuple(sorted(s))   # "eat","tea","ate" all sort to ('a','e','t')
        groups.setdefault(key, []).append(s)
    return list(groups.values())

# ["eat","tea","tan","ate","nat","bat"]
# → [["eat","tea","ate"],["tan","nat"],["bat"]]
```

---

### Q32. Lexicographically Smallest String After One Removal

```python
def removeOneChar(s: str) -> str:
    # Remove the first character where s[i] > s[i+1]
    for i in range(len(s) - 1):
        if s[i] > s[i + 1]:
            return s[:i] + s[i+1:]
    # If string is non-decreasing, remove the last character
    return s[:-1]

# "abcda" → "abca" (remove 'd')
# "abcd"  → "abc"  (remove last)
```

---

### Q35. Reverse String in Blocks of Size k (LC #541)

```python
def reverseStr(s: str, k: int) -> str:
    s = list(s)
    for i in range(0, len(s), 2 * k):
        # Reverse first k chars of each 2k block
        s[i:i+k] = s[i:i+k][::-1]
    return ''.join(s)

# s = "abcdefg", k = 2 → "bacdfeg"
```

---

### Q36. Count Distinct Strings After Exactly One Swap

```python
def countDistinctStrings(s: str) -> int:
    """Brute-force: generate all strings from exactly one swap."""
    n = len(s)
    seen = {s}   # include original (swap identical chars = original)
    for i in range(n):
        for j in range(i + 1, n):
            t = list(s)
            t[i], t[j] = t[j], t[i]
            seen.add(''.join(t))
    return len(seen)

# "abc"  → 4  (original + "bac", "cba", "acb")
# "aab"  → 2  ("aab" and "baa" — swapping the two 'a's gives back "aab")
```
**Time:** O(n² × n) | **Space:** O(n² × n) — fine for interview lengths

> 💡 **Optimised O(n²):** Count unique swaps = total_pairs − same_char_pairs. Add 1 if any duplicate chars exist (swapping identical chars = original). But the brute-force above is cleaner for interviews.


---

### Q37. Reorganize String (LC #767)

> No two adjacent characters can be the same. Use a max-heap on frequencies.

```python
import heapq
from collections import Counter

def reorganizeString(s: str) -> str:
    counts = Counter(s)
    # Max-heap: negate counts for Python's min-heap
    heap = [(-cnt, char) for char, cnt in counts.items()]
    heapq.heapify(heap)
    result = []

    while len(heap) >= 2:
        cnt1, char1 = heapq.heappop(heap)
        cnt2, char2 = heapq.heappop(heap)
        result.extend([char1, char2])
        if cnt1 + 1 < 0: heapq.heappush(heap, (cnt1 + 1, char1))
        if cnt2 + 1 < 0: heapq.heappush(heap, (cnt2 + 1, char2))

    if heap:
        cnt, char = heap[0]
        if -cnt > 1:
            return ""   # Impossible
        result.append(char)

    return ''.join(result)

# "aab" → "aba"
# "aaab" → ""
```
**Time:** O(n log k) where k = distinct chars (≤ 26)

---

### Q38. Roman to Integer (LC #13)

```python
def romanToInt(s: str) -> int:
    val = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
    result = 0
    for i in range(len(s)):
        # If current < next, subtract (e.g. IV = 5-1 = 4)
        if i + 1 < len(s) and val[s[i]] < val[s[i+1]]:
            result -= val[s[i]]
        else:
            result += val[s[i]]
    return result

# "MCMXCIV" → 1994
```

---

### Q44. Count Subsequences Starting with One Char, Ending with Another

```python
def countSubseq(s: str, first: str, last: str) -> int:
    # Count pairs (i, j) where s[i]==first, s[j]==last, i<=j
    count_first = 0
    result = 0
    for c in s:
        if c == last:
            result += count_first
            if first == last:  # count_first includes current position
                pass  # already counted
        if c == first:
            count_first += 1
    return result

# s="aab", first='a', last='b' → 2 (("a","b") twice)
```

---

### Q48. Validate Abbreviations (LC #408)

```python
def validWordAbbreviation(word: str, abbr: str) -> bool:
    i = j = 0
    while i < len(word) and j < len(abbr):
        if abbr[j].isdigit():
            if abbr[j] == '0':  # no leading zeros
                return False
            num = 0
            while j < len(abbr) and abbr[j].isdigit():
                num = num * 10 + int(abbr[j])
                j += 1
            i += num  # skip `num` characters in word
        else:
            if word[i] != abbr[j]:
                return False
            i += 1; j += 1
    return i == len(word) and j == len(abbr)

# word="internationalization", abbr="i18n" → True
# word="apple", abbr="a2e" → False
```

---

<a name="part-2"></a>
## Part 2 — Sliding Window on Strings

### Q41. Longest Substring Without Repeating Characters (LC #3)

```python
def lengthOfLongestSubstring(s: str) -> int:
    char_idx = {}   # char → last seen index
    left = 0
    max_len = 0

    for right, c in enumerate(s):
        if c in char_idx and char_idx[c] >= left:
            left = char_idx[c] + 1   # shrink window
        char_idx[c] = right
        max_len = max(max_len, right - left + 1)

    return max_len

# "abcabcbb" → 3 ("abc")
# "pwwkew"   → 3 ("wke")
```
**Time:** O(n) | **Space:** O(min(n, alphabet))

---

### Q42. Longest Substring with At Most k Distinct Characters (LC #340)

```python
from collections import defaultdict

def lengthOfLongestSubstringKDistinct(s: str, k: int) -> int:
    if k == 0: return 0
    window = defaultdict(int)
    left = 0
    max_len = 0

    for right, c in enumerate(s):
        window[c] += 1
        while len(window) > k:
            left_char = s[left]
            window[left_char] -= 1
            if window[left_char] == 0:
                del window[left_char]
            left += 1
        max_len = max(max_len, right - left + 1)

    return max_len

# s="eceba", k=2 → 3 ("ece")
```

---

### Q43. Minimum Window Substring (LC #76) — Hard

```python
from collections import Counter

def minWindow(s: str, t: str) -> str:
    if not t or not s:
        return ""

    need = Counter(t)
    missing = len(t)     # total chars still needed
    best = ""
    left = 0

    for right, c in enumerate(s):
        if need[c] > 0:
            missing -= 1
        need[c] -= 1

        if missing == 0:
            # Window is valid — shrink from left
            while need[s[left]] < 0:
                need[s[left]] += 1
                left += 1
            # Record if best
            window = s[left:right+1]
            if not best or len(window) < len(best):
                best = window
            # Expand: move left out to find next valid window
            need[s[left]] += 1
            missing += 1
            left += 1

    return best

# s="ADOBECODEBANC", t="ABC" → "BANC"
```
**Time:** O(|s| + |t|) | **Space:** O(|t|)

> 💡 **Pattern:** Sliding window with a "missing" counter is the key trick. When `missing == 0`, the window is valid — then shrink from the left while it stays valid.

---

<a name="part-3"></a>
## Part 3 — Palindrome Problems

### Q39. Longest Palindromic Substring (LC #5)

```python
def longestPalindrome(s: str) -> str:
    best = ""

    def expand(left, right):
        nonlocal best
        while left >= 0 and right < len(s) and s[left] == s[right]:
            if right - left + 1 > len(best):
                best = s[left:right+1]
            left -= 1
            right += 1

    for i in range(len(s)):
        expand(i, i)      # odd-length palindromes
        expand(i, i + 1)  # even-length palindromes

    return best

# "babad" → "bab" or "aba"
# "cbbd"  → "bb"
```
**Time:** O(n²) | **Space:** O(1)

> Manacher's algorithm solves this in O(n) but is rarely asked — expand-around-center is the expected interview answer.

---

### Q40. Longest Palindromic Subsequence (LC #516)

```python
def longestPalindromeSubseq(s: str) -> int:
    n = len(s)
    # dp[i][j] = longest palindromic subsequence in s[i..j]
    dp = [[0] * n for _ in range(n)]

    for i in range(n):
        dp[i][i] = 1   # single char is palindrome

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])

    return dp[0][n-1]

# "bbbab" → 4 ("bbbb")
# "cbbd"  → 2 ("bb")
```
**Time:** O(n²) | **Space:** O(n²)

---

<a name="part-4"></a>
## Part 4 — String Encoding & Decoding

### Q33. Decode Ways (LC #91)

```python
def numDecodings(s: str) -> int:
    if not s or s[0] == '0':
        return 0
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1

    for i in range(2, n + 1):
        one = int(s[i-1])
        two = int(s[i-2:i])
        if one != 0:
            dp[i] += dp[i-1]
        if 10 <= two <= 26:
            dp[i] += dp[i-2]

    return dp[n]

# "12" → 2 ("AB" or "L")
# "226" → 3
# "06" → 0
```

---

### Q47. Decode Bracket-Encoded String (LC #394)

```python
def decodeString(s: str) -> str:
    stack = []
    current_str = ""
    current_num = 0

    for c in s:
        if c.isdigit():
            current_num = current_num * 10 + int(c)
        elif c == '[':
            stack.append((current_str, current_num))
            current_str = ""
            current_num = 0
        elif c == ']':
            prev_str, num = stack.pop()
            current_str = prev_str + current_str * num
        else:
            current_str += c

    return current_str

# "3[a]2[bc]"    → "aaabcbc"
# "3[a2[c]]"     → "accaccacc"
# "2[abc]3[cd]ef" → "abcabccdcdcdef"
```
**Time:** O(output length) | **Space:** O(depth of nesting)

---

### Q29. Compress String / Run-Length Encoding

See Q29 above. The decode is the reverse: expand `a3b2` → `aaabb`.

---

<a name="part-5"></a>
## Part 5 — KMP & Z-Algorithm

### Q45. KMP — Substring Search (LC #28)

> **Big picture:** Naive substring search is O(n×m). KMP preprocesses the pattern to skip redundant comparisons, achieving O(n+m).

```python
def strStr_KMP(haystack: str, needle: str) -> int:
    if not needle:
        return 0
    n, m = len(haystack), len(needle)

    # Step 1: Build the LPS (Longest Proper Prefix which is also Suffix) array
    lps = [0] * m
    length = 0
    i = 1
    while i < m:
        if needle[i] == needle[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length != 0:
            length = lps[length - 1]  # fall back — DON'T increment i
        else:
            lps[i] = 0
            i += 1

    # Step 2: Search using LPS to skip characters
    i = j = 0  # i = haystack pointer, j = needle pointer
    while i < n:
        if haystack[i] == needle[j]:
            i += 1; j += 1
        if j == m:
            return i - j   # found at index i-j
        elif i < n and haystack[i] != needle[j]:
            if j != 0:
                j = lps[j - 1]  # skip using LPS
            else:
                i += 1

    return -1

# haystack="mississippi", needle="issip" → 4
```

**How LPS works:**
```
needle = "AAACAAAA"
lps    = [0,1,2,0,1,2,3,3]
         ↑               ↑
    "A" has no prefix   "AAAA" has "AAA" as both prefix and suffix → lps=3
```
**Time:** O(n + m) | **Space:** O(m)

---

### Q46. Z-Algorithm — Pattern Search

> Builds a Z-array where Z[i] = length of the longest substring starting from s[i] that is also a prefix of s.

```python
def z_function(s: str) -> list[int]:
    n = len(s)
    z = [0] * n
    z[0] = n
    l = r = 0

    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l, r = i, i + z[i]

    return z

def z_search(text: str, pattern: str) -> list[int]:
    """Find all occurrences of pattern in text."""
    combined = pattern + '#' + text   # '#' is a separator not in text/pattern
    z = z_function(combined)
    m = len(pattern)
    matches = []
    for i in range(m + 1, len(combined)):
        if z[i] == m:
            matches.append(i - m - 1)  # index in original text
    return matches

# text="aabxaa", pattern="aa" → [0, 4]
```
**Time:** O(n + m) | **Space:** O(n + m)

---

### Rolling Hash (Rabin-Karp)

> Alternative to KMP. Hash the pattern, slide a hash window over text.

```python
def rabinKarp(text: str, pattern: str) -> list[int]:
    n, m = len(text), len(pattern)
    BASE, MOD = 31, 10**9 + 7
    matches = []

    def char_val(c): return ord(c) - ord('a') + 1

    # Compute pattern hash and first window hash
    p_hash = 0
    t_hash = 0
    power = 1

    for i in range(m):
        p_hash = (p_hash + char_val(pattern[i]) * power) % MOD
        t_hash = (t_hash + char_val(text[i]) * power) % MOD
        if i < m - 1:
            power = (power * BASE) % MOD

    if p_hash == t_hash and text[:m] == pattern:
        matches.append(0)

    for i in range(1, n - m + 1):
        # Slide window: remove leftmost, add rightmost
        t_hash = (t_hash - char_val(text[i-1])) % MOD
        t_hash = (t_hash * pow(BASE, MOD-2, MOD)) % MOD   # modular inverse
        t_hash = (t_hash + char_val(text[i+m-1]) * power) % MOD

        if p_hash == t_hash and text[i:i+m] == pattern:   # verify (hash collision)
            matches.append(i)

    return matches
```

---

<a name="part-6"></a>
## Part 6 — Interview Q&A

### Q1: When do you use KMP vs a hash-based approach vs Python's `in` operator?
**A:**
- **Python `in` / `str.find()`:** For most interview problems — Python's built-in is already optimised. Use this unless you're explicitly asked to implement search.
- **KMP:** When asked to implement O(n+m) search. Also useful when you need to search the same pattern in multiple texts (reuse the LPS table).
- **Rabin-Karp / Rolling Hash:** When searching for multiple patterns simultaneously, or when the pattern set is large.
- **Z-Algorithm:** Elegant alternative to KMP; equally fast, sometimes cleaner to reason about.

### Q2: How does the sliding window technique apply to strings?
**A:** Sliding window works on strings when you need the optimal (longest/shortest) contiguous substring satisfying a constraint. Two patterns:
- **Variable window (grow + shrink):** Longest substring with ≤ k distinct chars. Grow right, shrink left when constraint violated.
- **Fixed window:** Anagram search (LC #438) — window = len(p), slide character by character.

### Q3: Explain the LPS array in KMP.
**A:** LPS[i] = length of the longest proper prefix of `pattern[0..i]` that is also a suffix. When a mismatch happens at position j in the pattern, instead of restarting from j=0, we jump to j=lps[j-1]. This avoids re-comparing characters we know matched.

### Q4: What's the difference between a substring and a subsequence?
**A:** Substring = contiguous (`"ace"` is NOT a substring of `"abcde"`, but `"abc"` is). Subsequence = not necessarily contiguous, order preserved (`"ace"` IS a subsequence of `"abcde"`). Most sliding window problems are about substrings; LCS/LPS problems are about subsequences.

### Q5: How to check if s2 is a rotation of s1?
**A:** Concatenate `s1 + s1` and check if `s2` is a substring. O(n).
```python
def isRotation(s1, s2): return len(s1) == len(s2) and s2 in s1 + s1
```

---

## LeetCode Problem List

| Q# | LC# | Problem | Pattern | Difficulty |
|---|---|---|---|---|
| Q25 | 14 | Longest Common Prefix | String | Easy |
| Q26 | 125 | Valid Palindrome | Two Pointers | Easy |
| Q27 | 680 | Valid Palindrome II | Two Pointers | Easy |
| Q28 | 387 | First Unique Character | Hash Map | Easy |
| Q29 | 443 | String Compression | Two Pointers | Medium |
| Q30 | 242 | Valid Anagram | Hash Map | Easy |
| Q31 | 49 | Group Anagrams | Hash Map | Medium |
| Q33 | 91 | Decode Ways | DP | Medium |
| Q35 | 541 | Reverse String II | String | Easy |
| Q37 | 767 | Reorganize String | Heap | Medium |
| Q38 | 13 | Roman to Integer | String | Easy |
| Q39 | 5 | Longest Palindromic Substring | Expand | Medium |
| Q40 | 516 | Longest Palindromic Subsequence | DP | Medium |
| Q41 | 3 | Longest Substring w/o Repeating | Sliding Window | Medium |
| Q42 | 340 | Longest Substring K Distinct | Sliding Window | Medium |
| Q43 | 76 | Minimum Window Substring | Sliding Window | Hard |
| Q45 | 28 | Find Index of First Occurrence | KMP | Easy |
| Q47 | 394 | Decode String | Stack | Medium |
| Q48 | 408 | Valid Word Abbreviation | Two Pointers | Easy |

---

## Further Resources

- **KMP Visualizer** — https://people.ok.ubc.ca/ylucet/DS/KnuthMorrisPratt.html
- **NeetCode Sliding Window playlist** — https://neetcode.io/roadmap
- **"Algorithms" by Sedgewick** — Chapter 5 for string algorithms
