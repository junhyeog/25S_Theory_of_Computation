from collections import defaultdict


def rev(s: str) -> str:
    return s[::-1]


def rev_comp(s: str) -> str:
    table = str.maketrans("ACGTUacgtu", "TGCAAtgcaa")
    return s.translate(table)[::-1]


def count_overlaps(text: str, pat: str) -> int:
    n = m = 0
    while True:
        i = text.find(pat, n)
        if i == -1:
            return m
        m += 1
        n = i + 1


def repeats_in_string(t: str, p: str, normal_only=False) -> bool:
    if not p:
        return True
    if len(p) > len(t):
        return False

    c0 = count_overlaps(t, p)
    c1 = count_overlaps(t, rev(p))
    c2 = count_overlaps(t, rev_comp(p))

    if normal_only:
        return c0 >= 2

    # return (c0 >= 2) or (c0 >= 1 and (c1 >= 1 or c2 >= 1)) # paper
    return (c0 + c1 + c2) >= 2  # pattern의 normal, reverse, reverse complement 모두 고려


def canonical(s: str) -> str:
    return min(s, rev(s), rev_comp(s))


# k개 이상 문자열에서 반복인가?
def ckeck_repeats(strings: list[str], k: int, candidate: str, verbose=True) -> tuple[bool, int]:
    flag_repeats = [repeats_in_string(t, candidate) for t in strings]
    cnt = sum(flag_repeats)
    if verbose:
        print(f"[-] {flag_repeats}")

    if cnt < k:
        if verbose:
            print(f"[!] Wrong answer: {candidate} appears in {cnt} strings, expected at least {k}.")
        return False, cnt

    if verbose:
        print(f"[+] {candidate} appears in {cnt} strings, which is enough.")
    return True, cnt


# 더 긴 공통 반복이 존재하는가? (완전 탐색)
def check_longer_repeats(strings: list[str], k: int, candidate: str, verbose=False) -> bool:
    base_len = len(candidate)
    max_len_in_data = max((len(s) for s in strings), default=0)

    for L in range(base_len + 1, max_len_in_data + 1):
        counter = defaultdict(int)
        for t in strings:
            if len(t) < L:
                continue
            frag_set = set()
            for i in range(len(t) - L + 1):
                frag = t[i : i + L]
                rep = canonical(frag)
                frag_set.add(rep)

            for rep in frag_set:
                if repeats_in_string(t, rep):
                    counter[rep] += 1

        for rep, cnt in counter.items():
            if cnt >= k:
                if verbose:
                    print(f"[!] Found longer repeat than {candidate}: {rep} appears in {cnt} strings.")
                return False
    if verbose:
        print(f"[+] No longer repeats than {candidate} found.")
    return True


def check_answer(strings: list[str], k: int, candidate: str, verbose=False) -> bool:
    if verbose:
        print(f"[new]")
        print(f"[-] strings: {strings}")
        print(f"[-] candidate: {candidate}")

    # 1) k개 이상 문자열에서 반복인가?
    if verbose:
        print(f"[1/2] Check if candidate appears in at least {k} strings...")
    flag_repeats, cnt = ckeck_repeats(strings, k, candidate, verbose)

    if not flag_repeats:
        return False

    # 2) 더 긴 공통 반복이 존재하는가?
    if verbose:
        print(f"[2/2] Check if there are longer repeats than {candidate}")
    if not check_longer_repeats(strings, k, candidate, verbose):
        return False

    return True


if __name__ == "__main__":
    T = ["AAGTCGAAGTC", "TCGAAGTCTCGA"]
    print(check_answer(T, k=2, candidate="TCGA"))
    print(check_answer(T, k=2, candidate="AAGT"))
