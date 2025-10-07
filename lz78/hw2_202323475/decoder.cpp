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
	int parent = -1;				 // parent node index
	char ch = -1;					 // char from parent to this node
};

// Trie
vector<Node> trie;

// 문자, 인덱스 매핑
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
			return -1;
	}
}

inline char decode_char(int idx) {
	if (idx < 26) return 'a' + idx;
	if (idx < 52) return 'A' + idx - 26;
	if (idx < 62) return '0' + idx - 52;
	static const char tbl[9] = {'?', '!', ' ', '.', ',', ':', ';', '\n', '\r'};
	return tbl[idx - 62];  // 62‥70
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

// class for bit input
class BitReader {
	uint64_t buf = 0;  // 비트 버퍼 (LSB 먼저)
	int bits = 0;	   // buf 안에 현재 채워진 비트 수
	istream& in;

   public:
	explicit BitReader(istream& s) : in(s) {}

	// n 비트의 정수 value 를 buf 에 저장
	uint32_t get(int n) {
		while (bits < n) {
			int byte = in.get();
			buf |= uint64_t((unsigned char)byte) << bits;
			bits += 8;
		}
		uint32_t val = buf & ((1u << n) - 1);
		buf >>= n;
		bits -= n;
		return val;
	}
};

void write_phrase(int idx, ostream& out) {
	vector<char> buf;
	buf.clear();
	while (idx) {  // root(0)까지 거꾸로 모은 뒤
		buf.push_back(trie[idx].ch);
		idx = trie[idx].parent;
	}
	for (auto it = buf.rbegin(); it != buf.rend(); ++it)
		out.put(*it);  // 역순으로 출력

	// // ! debug
	// cout << "[+] output phrase: ";
	// for (auto it = buf.rbegin(); it != buf.rend(); ++it)
	// 	cout << *it;
	// cout << LF;
}

// LZ78 Decoder
void decode(istream& fin, ostream& fout) {
	// init dictionary
	int root = init_trie();	 // root = 0, phrase_idx = 0
	BitReader br(fin);
	int k = 1;						   // 현재 인덱스 비트폭 (1비트면 0,1 표현)
	unsigned int next_phrase_idx = 1;  // 새 phrase 에 부여할 인덱스

	while (true) {
		// input phrase index
		uint32_t idx = br.get(k);  // prefix phrase index (k bit)
		// input next char
		uint32_t cidx = br.get(CHAR_BITS);	// 문자 코드 7 bit

		// // ! debug
		// cout << "[+] input (phrase_idx, next_char_idx): " << idx << SP << cidx << LF;

		if (idx >= trie.size())
			throw runtime_error("Bad index in bitstream");

		// write phrase
		write_phrase(static_cast<int>(idx), fout);

		// EOF 처리
		if (cidx == SENTINEL) break;

		// write next char
		char ch = decode_char(static_cast<int>(cidx));
		fout.put(ch);

		// // ! debug
		// cout << "[+] output char: " << ch << LF;

		// insert new phrase into trie
		int new_node = make_new_node();
		int parent_node = static_cast<int>(idx);
		trie[parent_node].child[cidx] = new_node;
		trie[new_node].parent = parent_node;
		trie[new_node].ch = ch;
		trie[new_node].phrase_idx = next_phrase_idx++;

		// // ! debug
		// cout << "[+] new phrase: " << LF;
		// cout << " -  prev: " << trie[parent_node].phrase_idx << LF;
		// cout << " -  next: " << trie[new_node].phrase_idx << LF;
		// cout << " -  char: " << ch << "(" << cidx << ")" << LF;

		// phrase 수가 2^k 에 도달 → 비트폭 한 단계 증가
		if (next_phrase_idx == (1u << k) && k) ++k;
	}
}

int main(int argc, char* argv[]) {
	chrono::system_clock::time_point start = chrono::system_clock::now();

	if (argc != 3) {
		cerr << "Usage: " << argv[0] << " <input_bin> <output_txt>" << LF;
		return 1;
	}
	ifstream fin(argv[1], ios::binary);
	if (!fin) {
		cerr << "Cannot open " << argv[1] << LF;
		return 1;
	}

	ofstream fout(argv[2], ios::binary);
	if (!fout) {
		cerr << "Cannot open " << argv[2] << LF;
		return 1;
	}

	decode(fin, fout);

	fout.flush();

	fin.close();
	fout.close();

	cout << "Decoded output written to " << argv[2] << LF;

	// running time 출력
	chrono::system_clock::time_point end = chrono::system_clock::now();
	chrono::duration<double, milli> msec = end - start;
	cout << "[+] decoding time (msec): " << msec.count() << " msec" << LF;

	return 0;
}
