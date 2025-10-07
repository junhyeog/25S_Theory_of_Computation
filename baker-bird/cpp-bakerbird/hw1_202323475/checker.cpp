#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
	// The checker program expects exactly 2 arguments:
	// argv[1]: input file path, argv[2]: output file path.
	if (argc < 3) {
		cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
		return 1;
	}

	// Open input and output files.
	ifstream fin(argv[1]);
	ifstream fout(argv[2]);
	if (!fin) {
		cerr << "Error opening input file.\n";
		return 1;
	}
	if (!fout) {
		cerr << "Error opening output file.\n";
		return 1;
	}

	int m, n;
	fin >> m >> n;

	// Read the pattern matrix (m x m)
	vector<string> pattern(m);
	for (int i = 0; i < m; i++) {
		fin >> pattern[i];
	}

	// Read the text matrix (n x n)
	vector<string> text(n);
	for (int i = 0; i < n; i++) {
		fin >> text[i];
	}

	// Compute the expected output using a naive sliding-window search.
	// For every possible top-left starting index (i,j) in text, check if the
	// m×m submatrix matches the pattern exactly.
	vector<pair<int, int>> expected_pairs;
	for (int i = 0; i <= n - m; i++) {
		for (int j = 0; j <= n - m; j++) {
			bool match = true;
			for (int x = 0; x < m && match; x++) {
				for (int y = 0; y < m; y++) {
					if (text[i + x][j + y] != pattern[x][y]) {
						match = false;
						break;
					}
				}
			}
			if (match) {
				expected_pairs.push_back({i, j});
			}
		}
	}

	// Read the program's output.
	vector<pair<int, int>> output_pairs;
	int count;
	if (!(fout >> count)) {
		cout << "no\n";
		return 0;
	}
	for (int i = 0; i < count; i++) {
		int a, b;
		if (!(fout >> a >> b)) {
			cout << "no\n";
			return 0;
		}
		output_pairs.push_back({a, b});
	}
	// Ensure there is no extra data in the output file.
	int extra;
	if (fout >> extra) {
		cout << "no\n";
		return 0;
	}

	// Compare the entire output with expected pairs.
	if (output_pairs.size() != expected_pairs.size()) {
		cout << "no\n";
		return 0;
	}
	for (size_t i = 0; i < expected_pairs.size(); i++) {
		if (output_pairs[i] != expected_pairs[i]) {
			cout << "no\n";
			return 0;
		}
	}

	// If everything matches, output "yes".
	cout << "yes\n";
	return 0;
}
