from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

########## ! Env
sys.setrecursionlimit(20000)
DEBUG = False


########## ! Sentinels
_ALPHABET = {"A", "C", "G", "T", "%", "#", "$"}
_SENTINELS: list[str] = [chr(c) for c in range(33, 127) if chr(c) not in _ALPHABET]  # readable ASCII 33-126 (for debug)
USED_SENTINEL_IDX = 0  # index of the next sentinel to use


def _safe_chr() -> str:
    global USED_SENTINEL_IDX
    if USED_SENTINEL_IDX >= len(_SENTINELS):
        raise ValueError(f"Sentinel id {USED_SENTINEL_IDX} out of range (max {len(_SENTINELS)})")
    sentinel = _SENTINELS[USED_SENTINEL_IDX]
    USED_SENTINEL_IDX += 1
    return sentinel, USED_SENTINEL_IDX - 1


########## ! Misc
def reverse_complement(s: str) -> str:
    comp = str.maketrans("ACGT", "TGCA")
    return s.translate(comp)[::-1]


def expand_string(s: str, normal_only=False) -> List[str]:
    if normal_only:
        return s
    rev = s[::-1]
    rc = reverse_complement(s)
    return s + "%" + rev + "#" + rc


########## ! Suffix Tree
class End:
    __slots__ = ("value",)

    def __init__(self, value: int):
        self.value = value

    def incr(self) -> None:
        self.value += 1

    def __int__(self) -> int:
        return self.value


@dataclass(slots=True)
class Node:
    start: int
    end: Union[int, End]
    link: int = -1  # suffix link index
    parent: int = -1
    children: Dict[str, int] = field(default_factory=dict)
    depth: int = 0  # path-length (chars) from root
    leaf_string: int = -1  # string id (sid)

    # * for step 2 (j, j')
    rep_leaf: int = -1
    lca_pair: tuple[int, int] | None = None  # (j, j′)

    # * for step 4
    is_sm_leaf: bool = False  # S$ 리프 여부
    marked: bool = False  # 슈퍼맥시멀 GST에 포함되는지

    def edge_len(self) -> int:
        end = int(self.end) if isinstance(self.end, End) else self.end
        return end - self.start + 1


# Ukkonen suffix tree
class SuffixTree:
    def __init__(self, text: List[str]):
        self.text = text
        self.nodes: List[Node] = []

        self.root = self._new(-1, -1)
        self.active_node = self.root
        self.active_edge = 0
        self.active_len = 0
        self.remainder = 0
        self.leaf_end = End(-1)

        for pos, ch in enumerate(text):
            self._extend(pos, ch)
        self._annotate_depths()
        self._finalize_pairs()

    def _new(self, start: int, end: Union[int, End]):
        self.nodes.append(Node(start, end))
        return len(self.nodes) - 1

    def _extend(self, pos: int, ch: str):
        self.leaf_end.incr()
        self.remainder += 1
        last_internal = -1

        while self.remainder:
            if self.active_len == 0:
                self.active_edge = pos
            edge_char = self.text[self.active_edge]

            if edge_char not in self.nodes[self.active_node].children:
                # Rule 2 – new leaf
                leaf = self._new(pos, self.leaf_end)
                self.nodes[leaf].parent = self.active_node
                self.nodes[self.active_node].children[edge_char] = leaf

                # * step 2 (j, j′)
                self.nodes[leaf].rep_leaf = leaf

                if last_internal != -1:
                    self.nodes[last_internal].link = self.active_node
                    last_internal = -1
            else:
                nxt = self.nodes[self.active_node].children[edge_char]
                edge_len = self.nodes[nxt].edge_len()
                if self.active_len >= edge_len:
                    self.active_edge += edge_len
                    self.active_len -= edge_len
                    self.active_node = nxt
                    continue
                if self.text[self.nodes[nxt].start + self.active_len] == ch:
                    # Rule 3 – next phase
                    self.active_len += 1
                    if last_internal != -1 and self.active_node != self.root:
                        self.nodes[last_internal].link = self.active_node
                    break
                # Rule 2 – split edge
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

                # * step 2 (j, j′)
                self.nodes[split].rep_leaf = leaf

                if last_internal != -1:
                    self.nodes[last_internal].link = split
                last_internal = split

            self.remainder -= 1
            if self.active_node == self.root and self.active_len:
                self.active_len -= 1
                self.active_edge = pos - self.remainder + 1
            elif self.active_node != self.root:
                self.active_node = (
                    self.nodes[self.active_node].link if self.nodes[self.active_node].link != -1 else self.root
                )

    def _annotate_depths(self):
        stack: List[Tuple[int, int]] = [(self.root, 0)]
        while stack:
            n, d = stack.pop()
            self.nodes[n].depth = d
            for c in self.nodes[n].children.values():
                stack.append((c, d + self.nodes[c].edge_len()))

    def postorder(self):
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

    # * step 2 (j, j′)
    def _finalize_pairs(self) -> None:
        for v in self.postorder():
            node = self.nodes[v]

            if not node.children:
                node.rep_leaf = v
                continue

            reps = [self.nodes[c].rep_leaf for c in node.children.values() if self.nodes[c].rep_leaf != -1]

            node.rep_leaf = reps[0]

            if len(reps) >= 2:
                if node.lca_pair is None or len(set(node.lca_pair)) < 2:
                    node.lca_pair = (reps[0], reps[1])


########## ! Debug


def path_to_root(st: SuffixTree, n: int) -> list[int]:
    p = []
    while n != -1:
        p.append(n)
        n = st.nodes[n].parent
    return p


def check_lca_property(st: SuffixTree) -> bool:
    for vidx, vnode in enumerate(st.nodes):
        if vidx == st.root or not vnode.children:
            continue

        assert vnode.lca_pair is not None, f"Node {vidx} has no (j,j') recorded"

        leaf1, leaf2 = vnode.lca_pair

        assert 0 <= leaf1 < len(st.nodes) and 0 <= leaf2 < len(st.nodes), f"(j,j') out of range at node {vidx}"
        assert (
            not st.nodes[leaf1].children and not st.nodes[leaf2].children
        ), f"(j,j') at node {vidx} are not both leaves"

        path1 = path_to_root(st, leaf1)
        path2 = path_to_root(st, leaf2)
        lca = next(u for u in path2 if u in path1)

        assert lca == vidx, f"LCA({leaf1},{leaf2}) is node {lca}, but stored at node {vidx}"

    return True


def _visualize_node_recursive(
    st: "SuffixTree", node_idx: int, current_indent_str: str, is_last_child_of_parent: bool, global_edge_end_val: int
):
    node = st.nodes[node_idx]
    edge_label_display = ""
    if node_idx == st.root:
        edge_label_display = "<ROOT>"
    else:
        actual_node_end = node.end if isinstance(node.end, int) else global_edge_end_val

        if node.start == -1:
            edge_label_display = "''(no edge/root)"
        elif node.start > actual_node_end:
            edge_label_display = f"''(empty edge: start={node.start} > end={actual_node_end})"
        else:
            label_chars_list = st.text[node.start : actual_node_end + 1]
            edge_label_display = f"'{''.join(label_chars_list)}'"

    line_prefix = current_indent_str
    if node_idx != st.root:
        line_prefix += "└─ " if is_last_child_of_parent else "├─ "

    node_info_str = f"Node {node_idx} {edge_label_display}"

    details = []
    raw_end_display = str(node.end) if isinstance(node.end, int) else f"End@{node.end.value}"
    details.append(f"S/E:({node.start},{raw_end_display})")  # 간선의 시작/끝 인덱스
    details.append(f"Depth:{node.depth}")

    if node.link != -1:
        details.append(f"Link:{node.link}")
    if node.parent != -1:
        details.append(f"Parent:{node.parent}")

    if not node.children:
        details.append(f"LEAF(sid:{node.leaf_string})")
        if getattr(node, "is_sm_leaf", False):
            sid = getattr(node, "leaf_string", -1)
            details.append(f"SM_LEAF(sid:{sid})")

    if getattr(node, "marked", False):
        details.append("MARKED")

    if node.lca_pair:
        details.append(f"LCA_Pair:{node.lca_pair}")

    print(f"{line_prefix}{node_info_str} [{', '.join(details)}]")

    children_items = sorted(list(node.children.items()))

    for i, (edge_first_char, child_idx) in enumerate(children_items):
        indent_for_child_level = current_indent_str
        if node_idx != st.root:
            indent_for_child_level += "    " if is_last_child_of_parent else "│   "

        _visualize_node_recursive(
            st,
            child_idx,
            indent_for_child_level,
            i == len(children_items) - 1,
            global_edge_end_val,
        )


def visualize_suffix_tree(st: "SuffixTree", title: str = "Suffix Tree Visualization"):
    print(f"\n--- {title} ---")

    text_str = "".join(st.text)
    print(f'Text: "{text_str}" (Length: {len(st.text)})')

    global_edge_end_val = st.leaf_end.value

    print(f"Root Node Index: {st.root}, Global End for Edges: {global_edge_end_val}")

    # 루트 노드에서부터 재귀적 시각화 시작
    _visualize_node_recursive(
        st,
        st.root,
        current_indent_str="",  # 초기 들여쓰기는 없음
        is_last_child_of_parent=True,  # 루트는 개념상 마지막 자식 (다음 레벨 들여쓰기 위함)
        global_edge_end_val=global_edge_end_val,
    )
    print()


########## ! LCA
def offline_lca_tarjan(gst: SuffixTree, queries: list[Tuple[int, int]]) -> list[int]:
    parent = list(range(len(gst.nodes)))
    rank = [0] * len(gst.nodes)
    anc = [None] * len(gst.nodes)
    visited = [False] * len(gst.nodes)
    res = [None] * len(queries)

    adj = [[] for _ in range(len(gst.nodes))]
    for idx, (u, v) in enumerate(queries):
        adj[u].append((v, idx))
        adj[v].append((u, idx))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        x, y = find(x), find(y)
        if x == y:
            return
        if rank[x] < rank[y]:
            parent[x] = y
        else:
            parent[y] = x
            if rank[x] == rank[y]:
                rank[x] += 1

    def dfs(u):
        parent[u] = u
        anc[u] = u
        for c in gst.nodes[u].children.values():
            dfs(c)
            union(u, c)
            anc[find(u)] = u
        visited[u] = True
        for v, qidx in adj[u]:
            if visited[v]:
                res[qidx] = anc[find(v)]

    dfs(gst.root)
    return res


########## ! Steps


def supermax_nodes(st: SuffixTree) -> list[int]:
    supers = []
    for v_idx in st.postorder():
        node = st.nodes[v_idx]
        if not node.children:
            continue
        if node.parent == -1:
            continue

        # * all children are leaves
        if any(st.nodes[c].children for c in node.children.values()):
            continue
        # * all children have distinct left chars
        lc_set = set()
        for c_i in node.children.values():
            left_char_index = len(st.text) - st.nodes[c_i].depth - 1
            assert left_char_index >= -1
            if left_char_index < 0:
                lc = ""
            else:
                lc = st.text[left_char_index]
            lc_set.add(lc)
        if len(lc_set) >= len(node.children):
            supers.append(v_idx)
    return supers


# * step 3
def get_Vs_with_Ms(
    gst: SuffixTree,
    suffix_trees: SuffixTree,
    M_lists: list[list[int]],
    gst_depth_offsets: list[int],
) -> list[int]:
    gst_depth_to_gst_node_index = {n.depth: i for i, n in enumerate(gst.nodes) if not n.children}

    queries = []
    for i, M_list in enumerate(M_lists):
        st = suffix_trees[i]
        depth_offset = gst_depth_offsets[i]
        for v in M_list:
            l1, l2 = st.nodes[v].lca_pair
            gst_depth1 = st.nodes[l1].depth + depth_offset
            gst_depth2 = st.nodes[l2].depth + depth_offset
            gst_node1 = gst_depth_to_gst_node_index[gst_depth1]
            gst_node2 = gst_depth_to_gst_node_index[gst_depth2]
            queries.append((gst_node1, gst_node2))

    # M to V mapping
    lca_nodes = offline_lca_tarjan(gst, queries)

    # slice lca_nodes to match M_lists shape
    V_lists = []
    cnt = 0
    for i, M_list in enumerate(M_lists):
        V_list = lca_nodes[cnt : cnt + len(M_list)]
        V_lists.append(V_list)
        cnt += len(M_list)

    # ? debug
    if DEBUG:
        for q_idx, (l1, l2) in enumerate(queries):
            path1 = path_to_root(gst, l1)  # leaf1 → root
            path2 = path_to_root(gst, l2)  # leaf2 → root
            lca = next(u for u in path2 if u in path1)  # 첫 공통 노드
            assert (
                lca == lca_nodes[q_idx]
            ), f"LCA({l1},{l2}) is node {lca}, but stored at index {q_idx} as {lca_nodes[q_idx]}"

    return V_lists


# * step 4
def insert_supermax_leaves(
    gst: SuffixTree, strings: List[str], V_lists: List[List[int]], sentinel_list: List[str], gst_offsets: List[int]
) -> List[List[int]]:
    n_strings = len(strings)
    N_lists: List[List[int]] = [[] for _ in range(n_strings)]

    for sid, V in enumerate(V_lists):
        sentinel_char = sentinel_list[sid]
        sentinel_pos = gst_offsets[sid] + len(strings[sid])

        for v in V:
            cur = v
            while cur != -1:
                if gst.nodes[cur].is_sm_leaf:
                    break
                parent = gst.nodes[cur]
                leaf = gst._new(sentinel_pos, sentinel_pos)
                gst.nodes[leaf].parent = cur
                gst.nodes[leaf].leaf_string = sid
                gst.nodes[leaf].is_sm_leaf = True
                parent.children[sentinel_char] = leaf
                N_lists[sid].append(leaf)
                cur = gst.nodes[cur].link
    return N_lists


# * step 4
def mark_supermax_paths(gst: SuffixTree, N_lists: List[List[int]]) -> None:
    for leaves in N_lists:
        for leaf in leaves:
            cur = leaf
            while cur != -1 and not gst.nodes[cur].marked:
                gst.nodes[cur].marked = True
                cur = gst.nodes[cur].parent


# * step 5
def collect_sm_leaves(gst: SuffixTree) -> List[int]:
    stack = [gst.root]
    sm_leaves = []
    while stack:
        v = stack.pop()
        node = gst.nodes[v]
        for c in node.children.values():
            stack.append(c)
        if node.is_sm_leaf:
            sm_leaves.append(v)
    return sm_leaves


def find_lcr_node(gst: SuffixTree, strings, k: int) -> Tuple[int, int]:
    sm_leaves = collect_sm_leaves(gst)
    n_strings = len(strings)

    leaves_by_color: List[List[int]] = [[] for _ in range(n_strings)]
    for leaf in sm_leaves:
        sid = gst.nodes[leaf].leaf_string
        leaves_by_color[sid].append(leaf)

    # LCA queries (leaf_i, leaf_{i+1})
    queries = []
    for L in leaves_by_color:
        for a, b in zip(L, L[1:]):
            queries.append((a, b))

    # count duplicates in each color
    lcas = offline_lca_tarjan(gst, queries)
    dup_local = [0] * len(gst.nodes)
    for v in lcas:
        dup_local[v] += 1

    # aggregate dup_cnt and # of leaves
    best_len = 0
    best_node = gst.root
    leaf_cnt = [0] * len(gst.nodes)
    dup_cnt = [0] * len(gst.nodes)

    for v in gst.postorder():
        node = gst.nodes[v]
        if not node.marked:
            continue

        if node.is_sm_leaf:
            leaf_cnt[v] = 1
        for c in node.children.values():
            leaf_cnt[v] += leaf_cnt[c]
            dup_cnt[v] += dup_cnt[c]

        dup_cnt[v] += dup_local[v]
        distinct = leaf_cnt[v] - dup_cnt[v]

        if distinct >= k and node.depth > best_len:
            best_len, best_node = node.depth, v

    return best_len, best_node


########## ! Main
def longest_common_repeat(strings: List[str], k: int = 2, normal_only=False, verbose=False) -> str:
    if not strings or k > len(strings):
        return ""

    global USED_SENTINEL_IDX
    USED_SENTINEL_IDX = 0  # reset sentinel index for each run
    sentinel_list: List[str] = []

    # ! Step 1
    new_strings: List[str] = []
    for i in range(len(strings)):
        new_strings.append(expand_string(strings[i], normal_only=normal_only))
    strings = new_strings

    text: List[str] = []
    str_idx_at_pos: List[int] = []
    gst_offsets = []
    gst_depth_offsets = []
    for idx, s in enumerate(strings):
        gst_offsets.append(len(text))
        for ch in s:
            text.append(ch)
            str_idx_at_pos.append(idx)

        # add a unique sentinel for each string
        _sen, _sen_idx = _safe_chr()
        text.append(_sen)
        sentinel_list.append(_sen)
        str_idx_at_pos.append(idx)

    if verbose:
        print(f"[+] strings: {strings}")
        print(f"[+] text: {text}")
        print(f"[+] k: {k}")

    for i, o in enumerate(gst_offsets):
        gst_depth_offsets.append(len(text) - o - len(strings[i]) - 1)

    # ! Step 2: Build STs and GST
    suffix_trees = [SuffixTree(list(s + sentinel_list[i])) for i, s in enumerate(strings)]

    if DEBUG:
        for i, st in enumerate(suffix_trees):
            assert check_lca_property(st), "LCA property check failed for one of the suffix trees."
    if verbose:
        for i, st in enumerate(suffix_trees):
            print(f"Suffix Tree for string {i} ({st.text}):")
            for v_i in range(len(st.nodes)):
                print(f"  {v_i}: {st.nodes[v_i]}")
            print()

            visualize_suffix_tree(st, title=f"Suffix Tree for string {i} ({st.text})")

    gst = SuffixTree(text)
    # annotate leaf nodes with their string-id
    for n in gst.nodes:
        if not n.children:
            s_pos = n.start
            n.leaf_string = str_idx_at_pos[s_pos] if 0 <= s_pos < len(str_idx_at_pos) else -1

    if verbose:
        print(f"GST {gst.text}:")
        for v_i in range(len(gst.nodes)):
            print(f"  {v_i}: {gst.nodes[v_i]}")
        print()

        visualize_suffix_tree(gst, title="Generalized Suffix Tree (GST)")

    # ! Step 3: Find supermaximal repeats
    M_lists = []  # internal and supermaximal nodes
    for i, st in enumerate(suffix_trees):
        supermax_nodes_list = supermax_nodes(st)
        M_lists.append(supermax_nodes_list)

    if verbose:
        print(f"M_lists: {M_lists}")

    V_lists = get_Vs_with_Ms(gst, suffix_trees, M_lists, gst_depth_offsets)
    if verbose:
        print(f"V_lists: {V_lists}")

    # ! Step 4
    N_lists = insert_supermax_leaves(gst, strings, V_lists, sentinel_list, gst_offsets)
    if verbose:
        print(f"N_lists: {N_lists}")

    mark_supermax_paths(gst, N_lists)

    if verbose:
        visualize_suffix_tree(gst, title="Generalized Suffix Tree (GST)")

    # ! Step 5
    best_len, best_node = find_lcr_node(gst, strings, k)

    # ! Result
    if best_len == 0:
        return ""

    chars = []
    node = best_node
    while node != gst.root:
        parent = gst.nodes[node].parent
        start = gst.nodes[node].start
        end = int(gst.nodes[node].end) if isinstance(gst.nodes[node].end, End) else gst.nodes[node].end
        chars.extend(reversed(gst.text[start : end + 1]))
        node = parent
    ans = "".join(reversed(chars))
    return ans


if __name__ == "__main__":
    DEBUG = True  # Enable debug mode

    # dna = ["AACTGAACGA", "GCAACTAACT"]
    # dna = ["AACTGAACGA", "GCAACTAACT", "GAATAACAAG"]

    # dna = ["CAACGAAGAAG"]
    # dna = ["CAACGAAGAAG", "AAA", "A", "AAAA"]
    dna = ["CAACGAAGAAG", "CAACGAAGAAG"]
    # dna = ["GCCCA", "CTGCA"]
    # dna = ["AACTG", "ACTGCTG"]
    # dna = ["CCCC", "CCC"]
    print("=", longest_common_repeat(dna, k=len(dna), verbose=True, normal_only=DEBUG))
