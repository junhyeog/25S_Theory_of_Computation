#include <bits/stdc++.h>
using namespace std;

const int ALPHABET_SIZE = 62;  // [a-z] + [A-Z] + [0-9]
const int MAX_NODES = 11111;   // Enough for m (<=100) distinct rows each of length m
#define LF '\n'
#define SP ' '

// Global arrays for Aho–Corasick trie
int tri[MAX_NODES][ALPHABET_SIZE];	// Trie transitions
int fail[MAX_NODES];				// Failure function
int out[MAX_NODES];					// Terminal output (stores distinct row number)
int cnt = 0;						// Number of nodes used in the trie

// Convert a character to index: 'a'-'z' -> [0,25], 'A'-'Z' -> [26,51], '0'-'9' -> [52,61]
int char_to_index(char c) {
	if (c >= 'a' && c <= 'z') return c - 'a';
	if (c >= 'A' && c <= 'Z') return 26 + (c - 'A');
	if (c >= '0' && c <= '9') return 52 + (c - '0');
	return -1;
}

// Insert pattern s with distinct row number (r) into the trie
void insert_pat_row(const string &s, int r) {
	int now = 0;
	for (char c : s) {
		int idx = char_to_index(c);
		if (!tri[now][idx]) tri[now][idx] = ++cnt;
		now = tri[now][idx];
	}
	out[now] = r;
}

// Build the failure function for the Aho–Corasick algorithm
void build_fail() {
	queue<int> q;
	for (int i = 0; i < ALPHABET_SIZE; i++) {
		if (tri[0][i]) q.push(tri[0][i]);
	}
	while (!q.empty()) {
		int now = q.front();
		q.pop();
		for (int i = 0; i < ALPHABET_SIZE; i++) {
			int nxt = tri[now][i];
			if (!nxt) continue;
			q.push(nxt);
			// find f
			int f = fail[now];
			while (f and !tri[f][i]) f = fail[f];
			f = tri[f][i];
			fail[nxt] = f;
		}
	}
}

vector<pair<int, int>> baker_bird(istream &in) {
	// Read dimensions: m = pattern size, n = text size.
	int m, n;
	in >> m >> n;

	// Read pattern: an m×m array
	vector<string> pat(m);
	for (int i = 0; i < m; i++) {
		in >> pat[i];
	}

	// Map each pattern row (of length m) to a distinct number.
	// Duplicate rows get the same number.
	// pat_col is the sequence of distinct numbers corresponding to the pattern rows.
	unordered_map<string, int> func_r;
	int nextNum = 0;
	vector<int> pat_col(m);
	for (int i = 0; i < m; i++) {
		if (func_r.find(pat[i]) == func_r.end()) func_r[pat[i]] = ++nextNum;  // if not found, assign a new number
		pat_col[i] = func_r[pat[i]];
	}

	// Build the Aho–Corasick automaton using distinct pattern rows.
	for (auto &entry : func_r) {
		insert_pat_row(entry.first, entry.second);
	}
	build_fail();

	// Build the KMP failure function for pat_col.
	vector<int> pre_pat_col(m, 0);
	for (int i = 1, j = 0; i < m; i++) {
		while (j and pat_col[i] != pat_col[j]) j = pre_pat_col[j - 1];
		if (pat_col[i] == pat_col[j]) pre_pat_col[i] = ++j;
	}

	// Maintain a KMP index for each column (size n).
	vector<int> KMP_col(n, 0);
	vector<pair<int, int>> matches;

	// Process text row by row. This avoids storing the entire text.
	string row;
	int now, R_val;
	for (int i = 0; i < n; i++) {
		in >> row;	  // Read one row of the text (length n)
		int now = 0;  // Tri state for the current row
		for (int j = 0; j < n; j++) {
			int idx = char_to_index(row[j]);
			while (now and !tri[now][idx]) now = fail[now];
			now = tri[now][idx];
			// Compute the R value on the fly.
			R_val = out[now];

			// Update KMP index for column j using the R_val.
			int &k = KMP_col[j];
			while (k and R_val != pat_col[k]) k = pre_pat_col[k - 1];
			if (R_val == pat_col[k]) {
				if (k == m - 1)
					matches.push_back({i - m + 1, j - m + 1}), k = pre_pat_col[k];
				else
					k++;
			}
		}
	}

	return matches;
}

int main(int argc, char *argv[]) {
	if (argc < 2) {
		cerr << "Usage: " << argv[0] << " <input_file>" << endl;
		return 1;
	}
	ifstream fin(argv[1]);
	if (!fin) {
		cerr << "Error opening input file: " << argv[1] << endl;
		return 1;
	}
	auto matches = baker_bird(fin);
	fin.close();

	// Print
	cout << matches.size() << LF;
	for (auto &p : matches) cout << p.first << SP << p.second << LF;

	return 0;
}
