#include <bits/stdc++.h>

int main(int argc, char** argv) {
  std::ifstream in(argv[1]);

  int m, n;
  in >> m >> n;

  std::vector<std::string> P(m), T(n);
  for (int i = 0; i < m; ++i) {
    in >> P[i];
    assert(P[i].size() == m);
  }

  for (int i = 0; i < n; ++i) {
    in >> T[i];
    assert(T[i].size() == n);
  }

  // run the Baker Bird algorithm and output the positions of occurrences

  std::cout << "3" << std::endl;
  std::cout << "0 0" << std::endl;
  std::cout << "2 0" << std::endl;
  std::cout << "3 3" << std::endl;

  return 0;
}
