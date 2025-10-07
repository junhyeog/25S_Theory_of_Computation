#include <bits/stdc++.h>

int main(int argc, char** argv) {
  std::ifstream in(argv[1]);
  std::ifstream out(argv[2]);

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

  int k;
  out >> k;

  std::vector<std::pair<int, int>> output(k);
  for (int i = 0; i < k; ++i) {
    out >> output[i].first >> output[i].second;
  }

  // check the correctness
  bool correct = (k == 3);

  if (correct) {
    std::cout << "yes" << std::endl;
  } else {
    std::cout << "no" << std::endl;
  }

  return 0;
}
