#include <bits/stdc++.h>
using namespace std;

struct Pos {
	int i, j;
};

int main(int argc, char* argv[]) {
	if (argc < 2) {
		cerr << "Usage: " << argv[0] << " <output_file>\n";
		return 1;
	}
	ofstream fout(argv[1]);
	if (!fout) {
		cerr << "Error opening output file: " << argv[1] << "\n";
		return 1;
	}

	// Allowed characters: a-z, A-Z, 0-9 (62 characters)
	string allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
	int allowedSize = allowed.size();

	// Maximum allowed n (m <= n <= 100)
	int max_val = 10;

	// Initialize random number generator.
	random_device rd;
	mt19937 gen(rd());

	// Randomly select m between 1 and max_val.
	uniform_int_distribution<> dis_m(1, max_val);
	int m = dis_m(gen);

	// Randomly select n between m and max_val.
	uniform_int_distribution<> dis_n(m, max_val);
	int n = dis_n(gen);

	// Write m and n to the file.
	fout << m << " " << n << "\n";

	// Create a uniform distribution for character selection.
	uniform_int_distribution<> dis_char(0, allowedSize - 1);

	// Generate the pattern matrix (m x m).
	vector<string> pattern(m);
	for (int i = 0; i < m; i++) {
		string pat;
		for (int j = 0; j < m; j++) {
			pat.push_back(allowed[dis_char(gen)]);
		}
		pattern[i] = pat;
	}

	// Output the pattern.
	for (int i = 0; i < m; i++) {
		fout << pattern[i] << "\n";
	}

	// Generate the text matrix (n x n) randomly.
	vector<string> text(n);
	for (int i = 0; i < n; i++) {
		string line;
		for (int j = 0; j < n; j++) {
			line.push_back(allowed[dis_char(gen)]);
		}
		text[i] = line;
	}

	// Determine the maximum number of non-overlapping placements.
	// A simple (but not optimal) bound is to partition the text into blocks of size m×m.
	int maxNonOverlap = (n / m) * (n / m);
	if (maxNonOverlap < 1)
		maxNonOverlap = 1;

	// Randomly select k (number of pattern insertions) between 1 and maxNonOverlap.
	uniform_int_distribution<> dis_k(1, maxNonOverlap);
	int k = dis_k(gen);

	// Collect all valid top-left positions for an m×m submatrix.
	vector<Pos> validPositions;
	for (int i = 0; i <= n - m; i++) {
		for (int j = 0; j <= n - m; j++) {
			validPositions.push_back({i, j});
		}
	}
	// Shuffle the valid positions.
	shuffle(validPositions.begin(), validPositions.end(), gen);

	// Select k non-overlapping positions.
	vector<Pos> selected;
	for (auto pos : validPositions) {
		bool overlap = false;
		for (auto sel : selected) {
			// Check if pos and sel overlap.
			// They do not overlap if either:
			// pos.i >= sel.i + m, or sel.i >= pos.i + m, or pos.j >= sel.j + m, or sel.j >= pos.j + m.
			if (!(pos.i >= sel.i + m || sel.i >= pos.i + m || pos.j >= sel.j + m || sel.j >= pos.j + m)) {
				overlap = true;
				break;
			}
		}
		if (!overlap) {
			selected.push_back(pos);
			if ((int)selected.size() == k)
				break;
		}
	}
	// Adjust k if fewer than the chosen k positions are available.
	k = selected.size();

	// Embed the pattern into the text at each selected position.
	for (auto pos : selected) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				text[pos.i + i][pos.j + j] = pattern[i][j];
			}
		}
	}

	// Output the modified text.
	for (int i = 0; i < n; i++) {
		fout << text[i] << "\n";
	}

	fout.close();
	return 0;
}
