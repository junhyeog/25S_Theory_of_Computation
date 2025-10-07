from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

# ! utils

_ALPHABET = {"A", "C", "G", "T", "%", "#", "$"}
_SENTINELS: list[str] = [chr(c) for c in range(33, 127) if chr(c) not in _ALPHABET]  # readable ASCII 33-126 (for debug)
USED_SENTINEL_IDX = 0  # index of the next sentinel to use


# new _safe_chr function using yeild
def _safe_chr() -> str:
    global USED_SENTINEL_IDX
    if USED_SENTINEL_IDX >= len(_SENTINELS):
        raise ValueError(f"Sentinel id {USED_SENTINEL_IDX} out of range (max {len(_SENTINELS)})")
    sentinel = _SENTINELS[USED_SENTINEL_IDX]
    USED_SENTINEL_IDX += 1
    return sentinel, USED_SENTINEL_IDX - 1


def reverse_complement(s: str) -> str:
    comp = str.maketrans("ACGT", "TGCA")
    return s.translate(comp)[::-1]


###############################################################################
# 1.  Generalised suffix tree (single online build)                           #
###############################################################################


class End:
    """Shared mutable end-index for all currently open leaves."""

    __slots__ = ("value",)

    def __init__(self, value: int):
        self.value = value

    def incr(self) -> None:  # advance by one position
        self.value += 1

    def __int__(self) -> int:  # implicit cast
        return self.value


@dataclass(slots=True)
class Node:
    start: int
    end: Union[int, End]
    link: int = -1  # suffix link index
    parent: int = -1  # parent index
    children: Dict[str, int] = field(default_factory=dict)
    depth: int = 0  # path-length (chars) from root

    # repeat counters (filled in post-order DFS)
    once: int = 0  # bitmask – substring appears *exactly once* in these Ti
    twice: int = 0  # bitmask – substring appears ≥2 times in these Ti

    # for leaves only
    leaf_string: int = -1  # which Tᵢ this suffix belongs to

    # * for step 2 (j, j')
    rep_leaf: int = -1
    lca_pair: tuple[int, int] | None = None  # (j, j′)

    # * for step 4
    is_sm_leaf: bool = False  # S$ 리프 여부
    marked: bool = False  # 슈퍼맥시멀 GST에 포함되는지

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
        self._finalize_pairs()

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
                # Rule 2 – new leaf
                leaf = self._new(pos, self.leaf_end)
                self.nodes[leaf].parent = self.active_node
                self.nodes[self.active_node].children[edge_char] = leaf

                # * step 2 (j, j′)
                self.nodes[leaf].rep_leaf = leaf
                # * end of step 2 (j, j′)

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
                # * step 2 (j, j′)

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

    # ------------------------------------------------------------------
    def _annotate_depths(self):
        stack: List[Tuple[int, int]] = [(self.root, 0)]
        while stack:
            n, d = stack.pop()
            self.nodes[n].depth = d
            for c in self.nodes[n].children.values():
                stack.append((c, d + self.nodes[c].edge_len(0)))

    # ------------------------------------------------------------------
    def postorder(self):  # generator – node indices in post-order
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
        """
        후위 DFS 한 번으로
        • rep_leaf(v)  = v 서브트리에서 임의 리프의 *노드 ID*
        • lca_pair(v) = 서로 다른 자식 쪽 두 리프의 (leaf-ID, leaf-ID)
        를 채워 논문의 (j, j′) 조건을 확정한다.
        """
        for v in self.postorder():  # leaves → root
            node = self.nodes[v]

            # 1) 리프   → 대표 = 자기 자신
            if not node.children:
                node.rep_leaf = v
                continue

            # 2) 내부노드 → 각 자식의 대표 취합
            reps = [self.nodes[c].rep_leaf for c in node.children.values() if self.nodes[c].rep_leaf != -1]  # -1 필터링

            # 대표 리프는 아무거나 하나
            node.rep_leaf = reps[0]

            # (j, j′) : 서로 다른 자식에서 온 ‘첫 두 개’ 리프
            if len(reps) >= 2:
                if node.lca_pair is None or len(set(node.lca_pair)) < 2:
                    node.lca_pair = (reps[0], reps[1])


###############################################################################
# 2.  LCR driver                                                               #
###############################################################################
def expand_string(s: str) -> List[str]:
    # rev = s[::-1]
    # rc = reverse_complement(s)
    # return s + "%" + rev + "#" + rc
    return s


def check_lca_property(st: SuffixTree) -> bool:
    """
    모든 내부 노드 v(≠root)에 대해
      • v.lca_pair = (leafID1, leafID2) 가 존재하고
      • 그 두 리프의 LCA 가 정확히 v
    임을 확인한다.
    실패 시 AssertionError를 던지며, 끝까지 통과하면 True를 리턴.
    """
    # ───────────────────────────────────────────────────────────────
    # 1) 각 내부 노드 검사
    for vidx, vnode in enumerate(st.nodes):
        if vidx == st.root or not vnode.children:  # root 또는 leaf
            continue

        assert vnode.lca_pair is not None, f"Node {vidx} has no (j,j') recorded"

        leaf1, leaf2 = vnode.lca_pair

        # 1-a. 두 값이 실제 ‘리프 노드 ID’인지 확인
        assert 0 <= leaf1 < len(st.nodes) and 0 <= leaf2 < len(st.nodes), f"(j,j') out of range at node {vidx}"
        assert (
            not st.nodes[leaf1].children and not st.nodes[leaf2].children
        ), f"(j,j') at node {vidx} are not both leaves"

        # 1-b. 두 리프의 LCA 계산 (경로 비교 – O(h))
        def path_to_root(n: int) -> list[int]:
            p = []
            while n != -1:
                p.append(n)
                n = st.nodes[n].parent
            return p

        path1 = path_to_root(leaf1)  # leaf1 → root
        path2 = path_to_root(leaf2)  # leaf2 → root
        lca = next(u for u in path2 if u in path1)  # 첫 공통 노드

        assert lca == vidx, f"LCA({leaf1},{leaf2}) is node {lca}, but stored at node {vidx}"

    return True


def supermax_nodes(st: SuffixTree) -> list[int]:
    supers = []
    for v_idx in st.postorder():
        node = st.nodes[v_idx]
        if not node.children:  # leaf X
            continue
        if node.parent == -1:  # root X
            continue
        # (a) 모든 자식이 리프
        if any(st.nodes[c].children for c in node.children.values()):
            continue
        # (b) left-symbol 서로 다름
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


def _offline_lca_tarjan(gst: SuffixTree, queries: list[Tuple[int, int]]) -> list[int]:
    """
    gst   : suffix tree
    queries[i] = (leaf_u, leaf_v)
    return: lca_node_id for each query (same order)
    """
    parent = list(range(len(gst.nodes)))  # DSU
    rank = [0] * len(gst.nodes)
    anc = [None] * len(gst.nodes)
    visited = [False] * len(gst.nodes)
    res = [None] * len(queries)

    # adj[u] = [(v, qidx), …]
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


def path_to_root(st, n: int) -> list[int]:
    p = []
    while n != -1:
        p.append(n)
        n = st.nodes[n].parent
    return p


def get_Vs_with_Ms(gst, suffix_trees, M_lists: list[list[int]], gst_depth_offsets: list[int]) -> list[int]:
    target_gst_depth_pairs = []  # (l1, l2) 쌍의 GST 깊이
    gst_depth_to_gst_node_index = {n.depth: i for i, n in enumerate(gst.nodes) if not n.children}
    queries = []  # (gst_leaf1, gst_leaf2)
    for i, M_list in enumerate(M_lists):
        st = suffix_trees[i]
        depth_offset = gst_depth_offsets[i]
        for v in M_list:
            l1, l2 = st.nodes[v].lca_pair
            gst_depth1 = st.nodes[l1].depth + depth_offset
            gst_depth2 = st.nodes[l2].depth + depth_offset
            # print(st.nodes[l1])
            # print(st.nodes[l2])
            # print(l1, l2, "→", gst_depth1, gst_depth2)
            target_gst_depth_pairs.append((gst_depth1, gst_depth2))
            gst_node1 = gst_depth_to_gst_node_index[gst_depth1]
            gst_node2 = gst_depth_to_gst_node_index[gst_depth2]
            queries.append((gst_node1, gst_node2))

    # * M to V mapping
    # for v_i in range(len(gst.nodes)):
    # print(f"GST node {v_i}: {gst.nodes[v_i]}")
    lca_nodes = _offline_lca_tarjan(gst, queries)

    # slice lca_nodes to match M_lists shape
    V_lists = []
    cnt = 0
    for i, M_list in enumerate(M_lists):
        V_list = lca_nodes[cnt : cnt + len(M_list)]
        V_lists.append(V_list)
        cnt += len(M_list)

    # TODO: debug: check LCA property for each V_list
    for q_idx, (l1, l2) in enumerate(queries):
        path1 = path_to_root(gst, l1)  # leaf1 → root
        path2 = path_to_root(gst, l2)  # leaf2 → root
        lca = next(u for u in path2 if u in path1)  # 첫 공통 노드
        assert (
            lca == lca_nodes[q_idx]
        ), f"LCA({l1},{l2}) is node {lca}, but stored at index {q_idx} as {lca_nodes[q_idx]}"

    return V_lists


################################################## !


def _insert_supermax_leaves(
    gst: SuffixTree, strings: List[str], V_lists: List[List[int]], sentinel_list: List[str], gst_offsets: List[int]
) -> List[List[int]]:
    """Step 4-(1): 각 V_i 경로에 S$ 리프를 달고 N_i 반환."""
    n_strings = len(strings)

    visited = [False] * len(gst.nodes)  # 노드별 전역 방문표
    N_lists: List[List[int]] = [[] for _ in range(n_strings)]
    v_to_sid = {}
    for sid, V in enumerate(V_lists):
        for v in V:
            if v not in v_to_sid:
                v_to_sid[v] = [sid]
            else:
                v_to_sid[v].append(sid)
    print(v_to_sid)
    for v, sids in v_to_sid.items():
        cur = v
        while cur != -1 and not visited[cur]:
            # visited[cur] = True
            parent = gst.nodes[cur]
            for sid in sids:
                sentinel_char = sentinel_list[sid]
                sentinel_pos = gst_offsets[sid] + len(strings[sid])

                leaf = gst._new(sentinel_pos, sentinel_pos)
                gst.nodes[leaf].parent = cur
                gst.nodes[leaf].leaf_string = sid
                gst.nodes[leaf].is_sm_leaf = True
                parent.children[sentinel_char] = leaf
                N_lists[sid].append(leaf)
            cur = gst.nodes[cur].link

    return N_lists


def _mark_supermax_paths(gst: SuffixTree, N_lists: List[List[int]]) -> None:
    """Step 4-(2): N_i에서 루트까지 올라가며 노드.marked = True."""
    for leaves in N_lists:
        for leaf in leaves:
            cur = leaf
            while cur != -1 and not gst.nodes[cur].marked:
                gst.nodes[cur].marked = True
                cur = gst.nodes[cur].parent


def _visualize_node_recursive(
    st: "SuffixTree", node_idx: int, current_indent_str: str, is_last_child_of_parent: bool, global_edge_end_val: int
):
    """
    SuffixTree의 개별 노드와 그 자식들을 재귀적으로 시각화하는 헬퍼 함수.
    """
    node = st.nodes[node_idx]

    # 1. 현재 노드로 들어오는 간선의 레이블 결정
    edge_label_display = ""
    if node_idx == st.root:
        edge_label_display = "<ROOT>"
    else:
        # node.start와 node.end는 이 노드로 "들어오는" 간선의 레이블을 정의합니다.
        # node.end가 End 객체일 수 있으므로 global_edge_end_val를 사용해 실제 끝 위치를 결정합니다.
        actual_node_end = node.end if isinstance(node.end, int) else global_edge_end_val

        if node.start == -1:  # 일반적으로 루트만 해당
            edge_label_display = "''(no edge/root)"
        elif node.start > actual_node_end:  # start가 end보다 큰 비정상적인 경우 (예: 초기화 안된 노드)
            edge_label_display = f"''(empty edge: start={node.start} > end={actual_node_end})"
        else:
            # st.text에서 간선 레이블 문자열을 가져옵니다.
            label_chars_list = st.text[node.start : actual_node_end + 1]
            edge_label_display = f"'{''.join(label_chars_list)}'"

    # 2. 트리 구조를 위한 접두사(커넥터) 문자열 생성
    line_prefix = current_indent_str
    if node_idx != st.root:  # 루트는 커넥터가 필요 없습니다.
        line_prefix += "└─ " if is_last_child_of_parent else "├─ "

    # 3. 현재 노드의 기본 정보 문자열 준비
    node_info_str = f"Node {node_idx} {edge_label_display}"

    # 4. 현재 노드의 상세 정보 리스트 준비
    details = []
    raw_end_display = str(node.end) if isinstance(node.end, int) else f"End@{node.end.value}"
    # details.append(f"S/E:({node.start},{raw_end_display})")  # 간선의 시작/끝 인덱스
    details.append(f"Depth:{node.depth}")  # 문자열 깊이

    if node.link != -1:  # 접미사 링크
        details.append(f"Link:{node.link}")
    # if node.parent != -1: details.append(f"Parent:{node.parent}") # 부모 정보 (디버깅 시 유용)

    # 5. 리프 상태 및 특정 노드 타입 정보 추가
    if not node.children:  # 현재 GST 구조에서 물리적인 리프인가?
        details.append(f"LEAF(sid:{node.leaf_string})")
        # details.append("PHY_LEAF")
        if getattr(node, "is_sm_leaf", False):  # Step 4.1에서 추가된 S$ 리프인가?
            sid = getattr(node, "leaf_string", -1)  # is_sm_leaf라면 leaf_string은 sid를 가짐
            details.append(f"SM_LEAF(sid:{sid})")
        # elif node.leaf_string != -1:  # T_i'의 끝 센티널을 나타내는 원래 GST의 리프인가?
        #     details.append(f"OrigGST_LEAF(sid:{node.leaf_string})")

    # Step 4 관련 정보
    if getattr(node, "marked", False):  # Step 4.2에서 마킹되었는가?
        details.append("MARKED")

    # 기타 Node 클래스에 있는 디버깅용 정보
    # if node.lca_pair:  # Step 2에서 계산된 (j, j') 쌍
    #     details.append(f"LCA_Pair:{node.lca_pair}")
    # if node.rep_leaf != -1:  # Step 2에서 계산된 대표 리프 노드 인덱스
    #     details.append(f"RepLeaf_idx:{node.rep_leaf}")

    # once, twice 필드도 필요시 추가 가능
    # details.append(f"Once:{bin(node.once)}, Twice:{bin(node.twice)}")

    # 현재 노드 정보 출력
    print(f"{line_prefix}{node_info_str} [{', '.join(details)}]")

    # 6. 자식 노드들에 대해 재귀 호출
    # 자식들을 간선 시작 문자로 정렬하여 일관된 순서로 출력 (선택 사항)
    children_items = sorted(list(node.children.items()))

    for i, (edge_first_char, child_idx) in enumerate(children_items):
        # 다음 레벨의 자식 노드를 위한 들여쓰기 문자열 준비
        indent_for_child_level = current_indent_str
        if node_idx != st.root:  # 현재 노드가 루트가 아니라면, 그 줄의 커넥터 모양에 맞춰 다음 들여쓰기 결정
            indent_for_child_level += "    " if is_last_child_of_parent else "│   "

        _visualize_node_recursive(
            st,
            child_idx,
            indent_for_child_level,
            i == len(children_items) - 1,  # 현재 자식이 마지막 자식인지 여부
            global_edge_end_val,
        )


def visualize_suffix_tree(st: "SuffixTree", title: str = "Suffix Tree Visualization"):
    """
    주어진 SuffixTree 객체의 구조를 텍스트 형태로 시각화하여 출력합니다.

    호출 전제조건:
    - st (SuffixTree): 시각화할 SuffixTree 객체.
      - st.nodes: Node 객체들의 리스트.
      - st.root: 루트 노드의 인덱스.
      - st.text: 전체 텍스트 (문자 리스트).
      - st.leaf_end: 전역 끝 위치를 나타내는 End 객체 (또는 그 .value).
    - Node 클래스: 필요한 속성들(start, end, link, children, depth, leaf_string,
                   lca_pair, rep_leaf, is_sm_leaf, marked 등)을 가지고 있어야 합니다.
                   is_sm_leaf, marked는 getattr을 사용하여 안전하게 접근합니다.
    """
    print(f"\n--- {title} ---")
    if not st.nodes:
        print("트리가 비어 있거나 초기화되지 않았습니다.")
        return

    # 전체 텍스트 정보 출력
    # st.text가 문자 리스트이므로 join하여 문자열로 만듭니다.
    text_str = "".join(st.text)
    print(f'Text: "{text_str}" (Length: {len(st.text)})')

    # 간선의 끝 위치(End 객체)를 해석하기 위한 전역 끝 값 결정
    global_edge_end_val = -1
    if hasattr(st, "leaf_end") and hasattr(st.leaf_end, "value"):
        global_edge_end_val = st.leaf_end.value
    else:
        # st.leaf_end.value를 찾을 수 없는 경우의 폴백 (예: 트리가 다른 방식으로 빌드된 경우)
        print("주의: st.leaf_end.value를 찾을 수 없습니다. 전역 끝 값으로 len(st.text) - 1 을 사용합니다.")
        global_edge_end_val = len(st.text) - 1

    if global_edge_end_val == -1 and len(st.text) > 0:  # 아직도 -1이면 len(st.text) -1 시도
        global_edge_end_val = len(st.text) - 1

    print(f"Root Node Index: {st.root}, Global End for Edges: {global_edge_end_val}")

    # 루트 노드에서부터 재귀적 시각화 시작
    _visualize_node_recursive(
        st,
        st.root,
        current_indent_str="",  # 초기 들여쓰기는 없음
        is_last_child_of_parent=True,  # 루트는 개념상 마지막 자식 (다음 레벨 들여쓰기 위함)
        global_edge_end_val=global_edge_end_val,
    )


################################################## !


# -------------------------------------------------------------
def _collect_sm_leaves_dfs(gst: SuffixTree) -> List[int]:
    """루트→좌→우 DFS 순서로 ‘SM 리프’ 노드 번호만 반환."""
    stack = [gst.root]
    order = []
    while stack:
        v = stack.pop()
        node = gst.nodes[v]
        # children을 문자순으로 넣으면 일관된 DFS 순서
        for c in sorted(node.children.values(), reverse=True):
            stack.append(c)
        if node.is_sm_leaf:
            order.append(v)
    return order


# -------------------------------------------------------------
def _hui_color_set_sizes(gst: SuffixTree, strings, k: int) -> Tuple[int, int]:
    # 1) DFS 순서로 리프를 수집
    dfs_leaves = _collect_sm_leaves_dfs(gst)
    n_strings = len(strings)
    print(f"DFS leaves: {dfs_leaves}")
    leaves_by_color: List[List[int]] = [[] for _ in range(n_strings)]
    for leaf in dfs_leaves:
        sid = gst.nodes[leaf].leaf_string
        leaves_by_color[sid].append(leaf)

    # 2) 연속 쌍 (leaf_i, leaf_{i+1}) → LCA 쿼리 작성
    queries: List[Tuple[int, int]] = []
    for L in leaves_by_color:
        for a, b in zip(L, L[1:]):
            queries.append((a, b))

    # 3) Tarjan offline LCA로 중복 색을 세는 노드
    lcas = _offline_lca_tarjan(gst, queries)
    dup_local = [0] * len(gst.nodes)
    for v in lcas:
        dup_local[v] += 1  # 해당 노드에서 색이 1회 중복

    # 4) 후위 순회로 (leaf_cnt, dup_cnt) 집계
    best_len = 0
    best_node = gst.root
    leaf_cnt = [0] * len(gst.nodes)
    dup_cnt = [0] * len(gst.nodes)

    for v in gst.postorder():
        node = gst.nodes[v]
        if not node.marked:  # Step4에서 살려둔 노드만!
            continue

        if node.is_sm_leaf:
            leaf_cnt[v] = 1
        for c in node.children.values():
            leaf_cnt[v] += leaf_cnt[c]
            dup_cnt[v] += dup_cnt[c]

        dup_cnt[v] += dup_local[v]
        distinct = leaf_cnt[v] - dup_cnt[v]  # 색 집합 크기

        if distinct >= k and node.depth > best_len:
            best_len, best_node = node.depth, v

    return best_len, best_node


# -------------------------------------------------------------


def longest_common_repeat(strings: List[str], k: int = 2) -> str:
    if not strings or k > len(strings):
        return ""

    global USED_SENTINEL_IDX
    USED_SENTINEL_IDX = 0  # reset sentinel index for each run

    sentinel_list: List[str] = []
    # ! step 1: Create a new string Ti′ for each 1 i to consider reversed and reverse-complemented repeats.
    new_strings: List[str] = []
    for i in range(len(strings)):
        new_strings.append(expand_string(strings[i]))
    strings = new_strings

    # ------------------------------------------------------------------
    # 2.1  Stitch all expanded strings into one text with unique sentinels
    # ------------------------------------------------------------------
    text: List[str] = []
    str_idx_at_pos: List[int] = []  # map suffix-start → Tᵢ index
    gst_offsets = []
    gst_depth_offsets = []
    for idx, s in enumerate(strings):
        gst_offsets.append(len(text))
        # gst_offsets.append(0)  # ! one sentinel for all strings
        for ch in s:
            text.append(ch)
            str_idx_at_pos.append(idx)
        # add a sentinel to mark the end of the last string
        _sen, _sen_idx = _safe_chr()
        # _sen, _sen_idx = "$", 0  # ! one sentinel for all strings
        text.append(_sen)  # unique sentinel for each string
        sentinel_list.append(_sen)
        str_idx_at_pos.append(idx)
    # global_sen, global_sen_idx = _safe_chr()
    # text.append(global_sen)  # global terminal
    # str_idx_at_pos.append(-1)

    print("strings:", strings)
    print("text:", text)
    for i, o in enumerate(gst_offsets):
        gst_depth_offsets.append(len(text) - o - len(strings[i]) - 1)

    # ! Step 2: Build STs and GST
    suffix_trees = [SuffixTree(list(s + sentinel_list[i])) for i, s in enumerate(strings)]
    # TODO: debug: check LCA property for each ST
    # for i, st in enumerate(suffix_trees):
    #     print(f"Suffix tree for string ({i}) {st.text}:")
    #     for v_i in range(len(st.nodes)):
    #         print(f"  {v_i}: {st.nodes[v_i]}")
    #     print()

    #     visualize_suffix_tree(st, title="")

    # assert check_lca_property(st), "LCA property check failed for one of the suffix trees."

    # ** gst
    gst = SuffixTree(text)
    # annotate leaf nodes with their string-id
    for n in gst.nodes:
        if not n.children:
            s_pos = n.start
            n.leaf_string = str_idx_at_pos[s_pos] if 0 <= s_pos < len(str_idx_at_pos) else -1

    # TODO: debug
    # print(f"GST {gst.text}:")
    # for v_i in range(len(gst.nodes)):
    #     print(f"  {v_i}: {gst.nodes[v_i]}")
    # print()

    # visualize_suffix_tree(gst, title="Generalized Suffix Tree (GST)")

    # ! Step 3: Find supermaximal repeats
    M_lists = []  # internal and supermaximal nodes
    for i, st in enumerate(suffix_trees):
        supermax_nodes_list = supermax_nodes(st)
        M_lists.append(supermax_nodes_list)

    print(f"M_lists: {M_lists}")

    V_lists = get_Vs_with_Ms(gst, suffix_trees, M_lists, gst_depth_offsets)
    print(f"V_lists: {V_lists}")

    # ! Step 4
    N_lists = _insert_supermax_leaves(gst, strings, V_lists, sentinel_list, gst_offsets)
    # print(f"N_lists: {N_lists}")
    _mark_supermax_paths(gst, N_lists)

    # visualize_suffix_tree(gst, title="Generalized Suffix Tree (GST)")

    # ! Step 5
    best_len, best_node = _hui_color_set_sizes(gst, strings, k)

    # ------------------------------------------------------------------
    # 2.3  Post-order DFS: propagate `once` / `twice` bitmasks upward
    # ------------------------------------------------------------------
    # best_len = 0
    # best_node = gst.root
    # for n_idx in gst.postorder():
    #     n = gst.nodes[n_idx]
    #     if not n.children:  # leaf
    #         if n.leaf_string >= 0:
    #             n.once = 1 << n.leaf_string
    #     else:
    #         once, twice = 0, 0
    #         for c_idx in n.children.values():
    #             c = gst.nodes[c_idx]
    #             twice |= c.twice
    #             overlap = once & c.once
    #             twice |= overlap
    #             once = (once | c.once) & ~twice
    #         n.once, n.twice = once, twice

    #     # update best
    #     if n.twice.bit_count() >= k and n.depth > best_len:
    #         best_len, best_node = n.depth, n_idx

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
    ans = "".join(ch for ch in candidate)
    return ans


###############################################################################
# 3.  Simple CLI test                                                          #
###############################################################################
if __name__ == "__main__":
    # dna = ["AACTGAACGA", "GCAACTAACT"]
    # print(f"l: {len(dna)}", f"k: {len(dna)}", dna)
    # print("=", longest_common_repeat(dna, k=len(dna)))

    # dna = ["AACTGAACGA", "GCAACTAACT", "GAATAACAAG"]
    # print(f"l: {len(dna)}", f"k: {len(dna)}", dna)
    # print("=", longest_common_repeat(dna, k=len(dna)))

    # dna = ["CAACGAAGAAG"]
    # dna = ["CAACGAAGAAG", "AAA", "A", "AAAA"]
    # dna = ["CAACGAAGAAG", "CAACGAAGAAG"]
    # dna = ["GCCCA", "CTGCA"]
    dna = ["AACTG", "ACTGCTG"]
    print(f"l: {len(dna)}", f"k: {len(dna)}", dna)
    print("=", longest_common_repeat(dna, k=len(dna)))
