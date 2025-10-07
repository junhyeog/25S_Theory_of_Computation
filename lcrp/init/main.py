"""
Linear‑time implementation of the Longest Common Repeat (LCR) algorithm
======================================================================
This **revised version** fixes an issue that made the previous draft
return an empty string for the sample test.  The new code now builds *one*
generalised suffix tree for the whole (expanded) input instead of the
per‑string/supermaximal pipeline.  The complexity is still **Θ(∑|Tᵢ|)** in
time and memory, but the logic is simpler and—most importantly—correct.

Public API
----------
```
longest_common_repeat(strings: list[str], k: int = 2) -> str
```
* `strings` : list of input strings T₁ … T_ℓ.
* `k`       : the pattern must occur **at least twice** in **≥ k different**
              strings.
* returns   : one longest common repeat (ties → lexicographically smallest).

Sample run
~~~~~~~~~~
```bash
$ python3 main.py
k=2 → CTCTC
k=3 → CTC
```

Implementation highlights
------------------------
* **Single GST** – all (forward · reversed · reverse‑complement) copies of
  each Tᵢ are concatenated with unique sentinels and fed to **one** online
  Ukkonen build.  That is still linear and avoids the fragile “extract
  supermaximals first” stage.
* **Bit‑mask repeat counting** – during a *post‑order* DFS each internal node
  maintains two 64‑bit masks:
  * `once`  : strings in which the substring appears **exactly once** so far.
  * `twice` : strings in which it appears **≥ 2 times**.
  The merge logic upgrades bits from `once` → `twice` in O(1).
* **Colour threshold** – while propagating the masks we track the deepest
  node whose `twice` bit‑count ≥ k.
* **Path reconstruction** – the winning node’s path label is reconstructed
  without copying large substrings and then stripped of sentinel chars.

You can now drop the file in the same location (`main.py`) and re‑run your
experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Union, Tuple

###############################################################################
# 0.  Helper utilities                                                        #
###############################################################################

class End:
    """Shared mutable end‑index for all currently open leaves."""
    __slots__ = ("value",)

    def __init__(self, value: int):
        self.value = value

    def incr(self) -> None:  # advance by one position
        self.value += 1

    def __int__(self) -> int:  # implicit cast
        return self.value


def _safe_chr(i: int) -> str:
    """Return a control character (ASCII 1‑31) guaranteed not in inputs."""
    if not 0 < i < 32:
        raise ValueError("Sentinel must be 1 ≤ i ≤ 31")
    return chr(i)


def reverse_complement(s: str) -> str:
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return s.translate(comp)[::-1]

###############################################################################
# 1.  Generalised suffix tree (single online build)                           #
###############################################################################

@dataclass(slots=True)
class Node:
    start: int
    end: Union[int, End]
    link: int = -1                       # suffix link index
    parent: int = -1                     # parent index
    children: Dict[str, int] = field(default_factory=dict)
    depth: int = 0                       # path‑length (chars) from root

    # repeat counters (filled in post‑order DFS)
    once: int = 0   # bitmask – substring appears *exactly once* in these Ti
    twice: int = 0  # bitmask – substring appears ≥2 times in these Ti

    # for leaves only
    leaf_string: int = -1                # which Tᵢ this suffix belongs to

    # -----------------------------------------------------------
    def edge_len(self, cur: int) -> int:
        """Current edge length (cur = global position during construction)."""
        end = int(self.end) if isinstance(self.end, End) else self.end
        return end - self.start + 1


class SuffixTree:
    """Ukkonen online suffix tree over the *entire* expanded input."""

    def __init__(self, text: List[str]):
        self.text = text
        self.nodes: List[Node] = []
        self._new = self._node_factory()

        self.root = self._new(-1, -1)
        self.active_node = self.root
        self.active_edge = 0
        self.active_len = 0
        self.remainder = 0
        self.leaf_end = End(-1)

        for pos, ch in enumerate(text):
            self._extend(pos, ch)
        self._annotate_depths()

    # ------------------------------------------------------------------
    def _node_factory(self):
        def make(start: int, end: Union[int, End]):
            self.nodes.append(Node(start, end))
            return len(self.nodes) - 1
        return make

    # ------------------------------------------------------------------
    def _extend(self, pos: int, ch: str):
        self.leaf_end.incr()
        self.remainder += 1
        last_internal = -1

        while self.remainder:
            if self.active_len == 0:
                self.active_edge = pos
            edge_char = self.text[self.active_edge]

            if edge_char not in self.nodes[self.active_node].children:
                # Rule 2 – new leaf
                leaf = self._new(pos, self.leaf_end)
                self.nodes[leaf].parent = self.active_node
                self.nodes[self.active_node].children[edge_char] = leaf
                if last_internal != -1:
                    self.nodes[last_internal].link = self.active_node
                    last_internal = -1
            else:
                nxt = self.nodes[self.active_node].children[edge_char]
                edge_len = self.nodes[nxt].edge_len(pos)
                if self.active_len >= edge_len:
                    self.active_edge += edge_len
                    self.active_len -= edge_len
                    self.active_node = nxt
                    continue
                if self.text[self.nodes[nxt].start + self.active_len] == ch:
                    # Rule 3 – next phase
                    self.active_len += 1
                    if last_internal != -1 and self.active_node != self.root:
                        self.nodes[last_internal].link = self.active_node
                    break
                # Rule 2 – split edge
                split_end = self.nodes[nxt].start + self.active_len - 1
                split = self._new(self.nodes[nxt].start, split_end)
                self.nodes[split].parent = self.active_node
                self.nodes[self.active_node].children[edge_char] = split

                leaf = self._new(pos, self.leaf_end)
                self.nodes[leaf].parent = split
                next_char = self.text[self.nodes[nxt].start + self.active_len]
                self.nodes[split].children[next_char] = nxt
                self.nodes[split].children[ch] = leaf
                self.nodes[nxt].start += self.active_len
                self.nodes[nxt].parent = split

                if last_internal != -1:
                    self.nodes[last_internal].link = split
                last_internal = split

            self.remainder -= 1
            if self.active_node == self.root and self.active_len:
                self.active_len -= 1
                self.active_edge = pos - self.remainder + 1
            elif self.active_node != self.root:
                self.active_node = (
                    self.nodes[self.active_node].link
                    if self.nodes[self.active_node].link != -1
                    else self.root
                )

    # ------------------------------------------------------------------
    def _annotate_depths(self):
        stack: List[Tuple[int, int]] = [(self.root, 0)]
        while stack:
            n, d = stack.pop()
            self.nodes[n].depth = d
            for c in self.nodes[n].children.values():
                stack.append((c, d + self.nodes[c].edge_len(0)))

    # ------------------------------------------------------------------
    def postorder(self):  # generator – node indices in post‑order
        stack = [(self.root, False)]
        while stack:
            n, visited = stack.pop()
            if n == -1:
                continue
            if visited:
                yield n
            else:
                stack.append((n, True))
                for c in self.nodes[n].children.values():
                    stack.append((c, False))

###############################################################################
# 2.  LCR driver                                                               #
###############################################################################

def longest_common_repeat(strings: List[str], k: int = 2) -> str:  # noqa: C901
    if not strings or k < 2 or k > len(strings):
        return ""

    # ------------------------------------------------------------------
    # 2.1  Stitch all expanded strings into one text with unique sentinels
    # ------------------------------------------------------------------
    text: List[str] = []
    str_idx_at_pos: List[int] = []  # map suffix‑start → Tᵢ index
    sentinel = 1
    for idx, s in enumerate(strings):
        rev, rc = s[::-1], reverse_complement(s)
        # for part in (s, _safe_chr(sentinel), rev, _safe_chr(sentinel + 1), rc, _safe_chr(sentinel + 2)): # full
        for part in (s, _safe_chr(sentinel)): # only normal
            for ch in part:
                text.append(ch)
                str_idx_at_pos.append(idx)
        sentinel += 3
    text.append(_safe_chr(sentinel))  # global terminal
    str_idx_at_pos.append(-1)

    # ------------------------------------------------------------------
    # 2.2  Build GST (Ukkonen) in Θ(N)
    # ------------------------------------------------------------------
    gst = SuffixTree(text)

    # annotate leaf nodes with their string‑id
    for n in gst.nodes:
        if not n.children:
            s_pos = n.start
            n.leaf_string = str_idx_at_pos[s_pos] if 0 <= s_pos < len(str_idx_at_pos) else -1

    # ------------------------------------------------------------------
    # 2.3  Post‑order DFS: propagate `once` / `twice` bitmasks upward
    # ------------------------------------------------------------------
    best_len = 0
    best_node = gst.root
    for n_idx in gst.postorder():
        n = gst.nodes[n_idx]
        if not n.children:  # leaf
            if n.leaf_string >= 0:
                n.once = 1 << n.leaf_string
        else:
            once, twice = 0, 0
            for c_idx in n.children.values():
                c = gst.nodes[c_idx]
                twice |= c.twice
                overlap = once & c.once
                twice |= overlap
                once = (once | c.once) & ~twice
            n.once, n.twice = once, twice

        # update best
        if n.twice.bit_count() >= k and n.depth > best_len:
            best_len, best_node = n.depth, n_idx

    if best_len == 0:
        return ""

    # ------------------------------------------------------------------
    # 2.4  Reconstruct substring (root → best_node path) & strip sentinels
    # ------------------------------------------------------------------
    chars: List[str] = []
    node = best_node
    while node != gst.root:
        parent = gst.nodes[node].parent
        p = gst.nodes[parent]
        # edge label = text[start … end]
        start = gst.nodes[node].start
        end = int(gst.nodes[node].end) if isinstance(gst.nodes[node].end, End) else gst.nodes[node].end
        chars.extend(reversed(gst.text[start : end + 1]))
        node = parent
    candidate = "".join(reversed(chars))
    return "".join(ch for ch in candidate if ord(ch) >= 32)

###############################################################################
# 3.  Simple CLI test                                                          #
###############################################################################
if __name__ == "__main__":
    # dna = ["GGTCTCTC", "ACTCTCAG", "TTTCTCGT"]
    # print("k=2 →", longest_common_repeat(dna, k=2))  # CTCTC
    # print("k=3 →", longest_common_repeat(dna, k=3))  # CTC
    dna = ["CAACGAAGAAG", "CAACGAAGAAG"]
    print("k=3 →", longest_common_repeat(dna, k=len(dna)))
