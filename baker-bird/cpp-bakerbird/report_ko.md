# Theory of Computation: hw1

## Writer
- Name: 윤준혁
- Student ID: 2023-23475
- Date: 2025-04-07


## 실행 환경
- OS: Ubuntu 22.04.3 LTS
- Compiler: g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
- 컴파일 옵션: -std=c++17

## Baker-Bird 알고리즘 구현

본 절에서는 Baker-Bird 2차원 패턴 매칭 알고리즘의 구현에서 사용된 주요 함수들의 구조와 동작 방식, 그리고 시간 및 공간 복잡도에 대해 상세히 기술한다. 구현은 Aho–Corasick 알고리즘과 Knuth-Morris-Pratt(KMP) 알고리즘을 결합하여 이루어졌으며, 문제 명세에 명시된 공간 복잡도 제약 \(O(|\Sigma|m^2 + n)\)을 만족하도록 최적화되었다.


### `char_to_index(char c)`

#### 기능
주어진 문자 `c`를 정수 인덱스로 변환한다. 고정된 알파벳 집합 Σ = {a–z, A–Z, 0–9}를 기반으로 하며, 총 62개의 문자를 0부터 61까지의 정수 인덱스로 매핑한다. 이를 통해 Trie 구조에서 상수 시간에 branching을 수행할 수 있도록 한다.

#### 변환 규칙
- `'a'` ~ `'z'` → 0 ~ 25
- `'A'` ~ `'Z'` → 26 ~ 51
- `'0'` ~ `'9'` → 52 ~ 61


### `insert_pat_row(const string &s, int r)`

#### 기능
패턴 행 문자열 `s`를 Aho–Corasick Trie에 삽입하고, 해당 문자열이 끝나는 노드에 고유 라벨 `r`을 부여한다. 이때 r은 Baker-Bird 알고리즘에서 패턴의 각 distinct 행에 부여된 인덱스를 의미한다.

#### 주요 동작
- 루트 노드부터 문자 하나씩 순차적으로 탐색
- 경로가 존재하지 않는 경우 새 노드를 할당
- 최종 노드에 `out[now] = r`을 저장하여 terminal node로 설정


### 2.3 `build_fail()`

#### 기능
Aho–Corasick Trie의 각 노드에 대해 실패 함수(failure function)를 구성한다.
이는 매칭 실패 시 fallback 동작을 정의하는 핵심 구조로, KMP 알고리즘의 실패 함수와 유사한 역할을 수행한다.
BFS를 사용하여 모든 노드에 대해 실패 링크를 구성하고, 실패 링크는 부모의 실패 링크를 따라 fallback 지점을 탐색하여 설정된다.



### `baker_bird(istream &in)`

#### 기능
전체 Baker-Bird 알고리즘을 수행하는 메인 함수로, 패턴과 텍스트를 입력받아 모든 패턴의 출현 위치를 탐색하고 반환한다.

#### 2.4.1 패턴 행 전처리

입력된 \( m \times m \) 패턴의 각 행을 문자열로 읽어 들이고, 중복된 행은 동일한 정수 (r)로 인코딩한다.
패턴의 각 행에 할당된 정수들로 이루어진 `pat_col` 배열을 생성한다.
이는 열 방향의 패턴을 의미한다. 즉, R 행렬에서 열 방향으로 pat_col을 검색하는 데 사용된다.

**시간 복잡도**: \( O(m^2) \)  
**공간 복잡도**: \( O(m^2) \)


#### Aho–Corasick 전처리

중복 없는 행 문자열을 `insert_pat_row()`를 통해 Trie에 삽입한다.
이후 `build_fail()`을 호출하여 failure function을 계산한다.

**시간 복잡도**: \( O(|\Sigma| \cdot m^2) \)  
**공간 복잡도**: \( O(|\Sigma| \cdot m^2) \)


#### KMP 전처리

수직 방향 탐색을 위한 패턴 시퀀스 `pat_col`에 대해 prefix table(`pre_pat_col`)을 생성한다.

**시간 복잡도**: \( O(m) \)  
**공간 복잡도**: \( O(m) \)


#### 텍스트 탐색 및 매칭

텍스트는 한 줄씩 읽으며, 전체 \( n \times n \) 텍스트를 저장하지 않고, 한번에 한 줄씩 저장 및 처리한다. 따라서 공간 복잡도를 줄일 수 있다.
각 문자에 대해 Aho–Corasick 탐색을 수행하여 해당 위치의 R값을 계산한다.
R 값에 대해 열 단위로 KMP 상태를 유지하며 수직 패턴 매칭을 수행한다.
매칭 성공 시 좌표 \((i - m + 1, j - m + 1)\)를 결과로 저장한다.

**시간 복잡도**: \( O(n^2) \)  
**공간 복잡도**: \( O(n) \)


### `main(int argc, char *argv[])`

#### 기능
명령행 인자에서 입력 파일 경로를 받아 파일을 열고, `baker_bird()` 함수를 실행하여 매칭 결과를 표준 출력으로 출력한다.

**시간 복잡도**: \( O(|\Sigma|m^2 + n^2) \)  
**공간 복잡도**: \( O(|\Sigma|m^2 + n) \)



## Checker 프로그램 구현

Checker 프로그램은 제출된 Baker-Bird 알고리즘의 출력이 정확한지를 검증하기 위해 작성된 보조 프로그램이다. 입력 텍스트와 패턴을 기반으로 정답을 독립적으로 재계산한 후, 제출자의 출력과 정확히 일치하는지 여부를 판단한다 입력의 제약 조건 \( m \leq n \leq 100 \) 내에서 완전탐색 기반의 방법을 사용한다.


### 알고리즘

Checker 프로그램은 입력 파일로부터 패턴과 텍스트를 읽은 후, 나이브 슬라이딩 윈도우 방식으로 텍스트 내의 모든 \( m \times m \) 영역을 탐색하여 패턴과 동일한 서브매트릭스를 찾는다.

####  정답 생성

1. 텍스트의 가능한 모든 \( (i, j) \) 좌표에 대해 시작점으로 설정:
   \[
   0 \leq i \leq n - m, \quad 0 \leq j \leq n - m
   \]
2. 각 시작 좌표에서 텍스트의 \( m \times m \) 부분 행렬을 추출하고, 패턴과 각 위치의 문자 비교:
   \[
   \text{text}[i + x][j + y] \overset{?}{=} \text{pattern}[x][y] \quad \forall 0 \leq x, y < m
   \]
3. 모든 문자가 일치할 경우, 해당 시작 좌표 \((i, j)\)를 정답 좌표로 기록


#### 정답과 출력 파일 비교
Checker는 제출자의 출력 파일로부터 다음 정보를 읽어들인다:

- 첫 번째 줄: 매칭된 좌표의 개수 \( k \)
- 다음 \( k \)줄: 매칭된 좌표들 \((a_1, b_1), (a_2, b_2), \ldots, (a_k, b_k)\)

이후 내부적으로 생성한 정답 좌표 리스트와 제출자의 출력 좌표 리스트를 다음 기준으로 비교한다:

1. **개수 비교**: 정답 좌표 수와 출력 좌표 수가 일치해야 한다.
2. **값 비교**: 각 좌표 \((i, j)\) 쌍이 동일한 순서로 일치해야 한다.

모든 좌표가 동일할 경우 `yes`, 그렇지 않으면 `no`를 출력한다.


### 시간 복잡도 분석

Checker는 입력 텍스트에서 가능한 모든 \( m \times m \) 영역에 대해 패턴과 완전 비교를 수행하므로, 주요 연산은 다음과 같이 구성된다:

- 가능한 시작 위치의 개수:  
  \[
  (n - m + 1)^2 = O(n^2)
  \]
- 각 시작 위치에서 비교하는 문자 수:  
  \[
  m \times m = O(m^2)
  \]

따라서 전체 시간 복잡도는 다음과 같다:

\[
O(n^2 \cdot m^2)
\]



### 공간 복잡도 분석

Checker는 다음과 같은 데이터를 메모리에 저장한다:

- 패턴: \( O(m^2) \)
- 텍스트: \( O(n^2) \)
- 정답 좌표 벡터: \( O(k) \)
- 제출자 출력 좌표 벡터: \( O(k) \)

이 때 k는 최대 \( n^2 \)이므로, 공간 복잡도는 다음과 같이 계산된다:
\[
O(m^2 + n^2)
\]



## 예제1

### 입력
```
3 8
dyy
vyz
tUG
85NtDdyy
tt8iAvyz
uTuivtUG
GB5sZdyy
vGOqgvyz
pyPy1tUG
qZcZJwtu
IIjzwloU
```
### Baker-Bird 알고리즘 출력
```
2
0 5
3 5
```

### checker 프로그램 출력
```
yes
```

## 예제2
### 입력
```
3 6
bzB
wv2
App
HMNbzB
na5wv2
bzBApp
wv2l7z
AppTdV
hTgnxH
```

### Baker-Bird 알고리즘 출력
```
2
0 3
2 0
```

### checker 프로그램 출력
```
yes
```

## 예제3
### 입력
```
1 7
T
XWbBDiT
gwSOj9J
T9cCfTT
T3T5TjC
TTTTTTT
YDTIUT7
T7LTcTn
```

### Baker-Bird 알고리즘 출력
```
19
0 6
2 0
2 5
2 6
3 0
3 2
3 4
4 0
4 1
4 2
4 3
4 4
4 5
4 6
5 2
5 5
6 0
6 3
6 5
```

### checker 프로그램 출력
```
yes
```

## 예제4
### 입력
```
2 10
FK
Mv
5HElIyFKQY
FKFKFKMv1Y
MvMvMv3nFK
RvWyWRs7Mv
FKzBFKVidF
MvFKMvIFKY
2GMvTDXMvw
FKXI0yHxFK
MvFKFKCTMv
3MMvMvkLTZ
```

### Baker-Bird 알고리즘 출력
```
13
0 6
1 0
1 2
1 4
2 8
4 0
4 4
5 2
5 7
7 0
7 8
8 2
8 4
```

### checker 프로그램 출력
```
yes
```
