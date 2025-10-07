#include <bits/stdc++.h>
using namespace std;

#define LF '\n'
#define SP ' '

const int ALPHABET_SIZE = 62 + 9;		  // [a-z] + [A-Z] + [0-9] + ['?', '!', ' ', '.', ',', ':', ';', '\n', '\r']
const int CHAR_BITS = 7;				  // 알파벳 71개 → 7비트
const uint32_t SENTINEL = ALPHABET_SIZE;  // 0‥70 밖 7비트 값(EOF 토큰용)

// Node structure for Trie
struct Node {
	int child[ALPHABET_SIZE + 1]{};	 // ALPHABET_SIZE + SENTINEL
	int phrase_idx = -1;			 // index of the phrase (-1 if not a phrase, '' ==> 0)
};

// Trie
vector<Node> trie;

/* 문자, 인덱스 매핑
 *   0‥25  : 'a'‥'z'
 *   26‥51 : 'A'‥'Z'
 *   52‥61 : '0'‥'9'
 *   62    : '?'
 *   63    : '!'
 *   64    : ' '   (space)
 *   65    : '.'
 *   66    : ','
 *   67    : ':'
 *   68    : ';'
 *   69    : '\n'
 *   70    : '\r'
 */
inline int code(char c) noexcept {
	if ('a' <= c && c <= 'z') return c - 'a';
	if ('A' <= c && c <= 'Z') return 26 + (c - 'A');
	if ('0' <= c && c <= '9') return 52 + (c - '0');
	switch (c) {
		case '?':
			return 62;
		case '!':
			return 63;
		case ' ':
			return 64;
		case '.':
			return 65;
		case ',':
			return 66;
		case ':':
			return 67;
		case ';':
			return 68;
		case '\n':
			return 69;
		case '\r':
			return 70;
		default:
			return -1;	// 예외 처리용
	}
}

// Functions for Trie
int make_new_node() {
	trie.push_back(Node{});
	return static_cast<int>(trie.size()) - 1;
}

int init_trie() {
	trie.clear();
	int root = make_new_node();	 // root = 0
	trie[root].phrase_idx = 0;	 // ""  → 0
	return root;
}

// class for bit output
class BitWriter {
	uint64_t buf = 0;  // 비트 버퍼 (LSB 먼저)
	int bits = 0;	   // buf 안에 현재 채워진 비트 수
	ostream& out;

   public:
	explicit BitWriter(ostream& o) : out(o) {}

	// n 비트의 정수 value 를 buf 에 저장
	void put(uint32_t value, int n) {
		buf |= (uint64_t)value << bits;	 // value LSB 가 먼저 나가도록
		bits += n;
		while (bits >= 8) {	 // 8비트씩 output
			out.put(char(buf & 0xFF));
			buf >>= 8;
			bits -= 8;
		}
	}
	void flush() {
		if (bits) {	 // 마지막 남은 비트 0‑padding
			out.put(char(buf & 0xFF));
			buf = bits = 0;
		}
	}
};

// LZ78 Encoder
void encode(istream& fin, ostream& fout) {
	// init dictionary
	int root = init_trie();	 // root = 0, phrase_idx = 0
	BitWriter bw(fout);

	int k = 1;						   // 현재 인덱스 비트폭 (1비트면 0,1 표현)
	unsigned int next_phrase_idx = 1;  // 새 phrase 에 부여할 인덱스
	int v = root;					   // 현재 매칭 노드

	char ch;
	while (true) {
		fin.get(ch);
		int cidx = code(ch);  // 0‥70
		if (fin.eof()) {
			cidx = SENTINEL;  // EOF
		}
		// // ! debug
		// cout << "[+] input: " << ch << SP << cidx << LF;

		int nxt = trie[v].child[cidx];

		// 자식 노드가 존재하면 매칭 계속
		if (nxt) {
			v = nxt;
			continue;
		}

		// longest‑match 종료 → 토큰 ⟨index, char⟩ 출력
		bw.put(static_cast<uint32_t>(trie[v].phrase_idx), k);  // 인덱스 k비트
		bw.put(static_cast<uint32_t>(cidx), CHAR_BITS);		   // 문자 7비트

		// // ! debug
		// cout << "[+] output: " << trie[v].phrase_idx << SP << ch << LF;

		// (v + ch) 노드를 새 phrase 로 등록
		nxt = make_new_node();
		trie[v].child[cidx] = nxt;
		trie[nxt].phrase_idx = next_phrase_idx++;  // 색인 부여

		// // ! debug
		// cout << "[+] new phrase: " << LF;
		// cout << " -  prev: " << trie[v].phrase_idx << LF;
		// cout << " -  next: " << trie[nxt].phrase_idx << LF;
		// cout << " -  char: " << ch << "(" << cidx << ")" << LF;

		// phrase 수가 2^k 에 도달 → 비트폭 한 단계 증가
		if (next_phrase_idx == (1u << k)) ++k;

		v = root;  // 매칭 재시작

		// EOF 처리
		if (fin.eof()) break;
	}

	bw.flush();	 // 버퍼 잔여 비트 처리
}

int main(int argc, char* argv[]) {
	chrono::system_clock::time_point start = chrono::system_clock::now();

	if (argc != 3) {
		cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << LF;
		return 1;
	}
	ifstream fin(argv[1], ios::binary);
	if (!fin) {
		cerr << "Error opening input file: " << argv[1] << LF;
		return 1;
	}
	ofstream fout(argv[2], ios::binary);
	if (!fout) {
		cerr << "Error opening output file: " << argv[2] << LF;
		return 1;
	}

	encode(fin, fout);

	fout.flush();

	fin.close();
	fout.close();

	cout << "Output written to " << argv[2] << LF;

	// running time 출력
	chrono::system_clock::time_point end = chrono::system_clock::now();
	chrono::duration<double, milli> msec = end - start;
	cout << "[+] encoding time (msec): " << msec.count() << " msec" << LF;

	// compession ratio 출력
	uintmax_t in_sz = filesystem::file_size(argv[1]);
	uintmax_t out_sz = filesystem::file_size(argv[2]);
	double ratio = 100.0 * (1.0 - (double)out_sz / in_sz);
	cout << "[+] input size (byte): " << in_sz << LF;
	cout << "[+] output size (byte): " << out_sz << LF;
	cout << "[+] compression ratio (%): " << fixed << setprecision(2) << ratio << "%" << LF;

	return 0;
}
