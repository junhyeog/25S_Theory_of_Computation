#include <algorithm>
#include <iostream>
#include <map>
#include <memory>  // For smart pointers
#include <set>
#include <string>
#include <vector>

// Using directives and typedefs
using namespace std;
typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<string> vs;

// --- Configuration Constants ---
const char SPECIAL_CHAR_1 = '$';   // 논문에서는 %, # 사용. 구분을 위해 하나만 사용 가정.
								   // 실제 구현시 T, T^R, T^RC 구분을 위해 서로 다른 3개 필요.
								   // 혹은 문자열 ID와 함께 위치 정보로 구분.
const char UNIQUE_END_CHAR = '#';  // 각 문자열의 끝을 명확히 하기 위한 문자 (GST 구축 시)

// --- Forward Declarations ---
struct GSTNode;				  // 일반화된 접미사 트리 노드
class GeneralizedSuffixTree;  // 일반화된 접미사 트리
class SuffixTree;			  // 개별 접미사 트리 (GST의 일부로 통합되거나 유사한 구조 사용)

// --- Helper Functions (문자열 변형 등) ---
string reverse_string(const string& s) {
	string rs = s;
	reverse(rs.begin(), rs.end());
	return rs;
}

char complement_char(char c) {	// DNA 기준 예시
	switch (c) {
		case 'A':
			return 'T';
		case 'T':
			return 'A';
		case 'C':
			return 'G';
		case 'G':
			return 'C';
		default:
			return c;  // 오류 처리 또는 그대로 반환
	}
}

string reverse_complement_string(const string& s) {
	string rs = reverse_string(s);
	for (char& c : rs) c = complement_char(c);
	return rs;
}

// --- GST Node Structure ---
struct GSTNode {
	map<char, GSTNode*> children;
	GSTNode* suffix_link;
	int start_index;	   // concatenated_text에서의 시작 인덱스 (엣지 레이블)
	int* end_index_ptr;	   // 엣지 레이블의 끝 인덱스 (Ukkonen 트릭을 위해 포인터 사용)
	int string_id;		   // 이 노드가 어떤 원본 문자열 T_i'에 속하는지를 나타내는 ID (리프 노드에서 중요)
	int suffix_start_pos;  // 리프 노드의 경우, T_i' 내에서의 접미사 시작 위치

	// Step 3: ST_i의 노드 v에 해당하는 GST 노드 v' 식별을 위함
	// 이 노드가 어떤 T_i'의 초최대 반복에 해당하는지 표시 (V_i 집합 멤버)
	vector<int> is_supermaximal_for_T_prime_idx;

	// Step 4.1: 초최대 반복의 접미사로 인해 생성된 리프인지, 그리고 그 ID
	bool is_smr_derived_leaf;
	int smr_derived_leaf_original_string_id;

	// Step 4.2: GST_SMR에 포함될 노드인지 마킹
	bool marked_for_gst_smr;

	// LCA 계산을 위한 정보 (깊이, 부모 등은 실제 LCA 알고리즘 구현 시 필요)
	int depth_from_root_str_len;  // 루트로부터의 문자열 길이 (깊이)

	GSTNode(int start = -1, int* end_ptr = nullptr, GSTNode* sl = nullptr, int str_id = -1, int suffix_pos = -1)
		: suffix_link(sl), start_index(start), end_index_ptr(end_ptr), string_id(str_id), suffix_start_pos(suffix_pos), is_smr_derived_leaf(false), smr_derived_leaf_original_string_id(-1), marked_for_gst_smr(false), depth_from_root_str_len(0) {}

	bool is_leaf() const {
		return children.empty();
	}

	int get_edge_length(int current_global_end) const {
		if (!end_index_ptr) return 0;
		return (*end_index_ptr == -1 ? current_global_end : *end_index_ptr) - start_index + 1;
	}
};

// --- (Generalized) Suffix Tree Class ---
// 논문에서는 ST(T_i')와 GST(T_1'...T_l')를 구분하지만,
// 효율적인 구현에서는 GST 구축 알고리즘 하나로 모든 T_i'를 처리하고,
// 각 T_i'에 대한 연산(초최대 반복 찾기 등)은 GST 내에서 해당 문자열의 서픽스들만 고려하여 수행 가능.
// 여기서는 GST 클래스 중심으로 설명.
class GeneralizedSuffixTree {
   public:
	GSTNode* root;
	string concatenated_text_T_prime_all;  // 모든 T_i' 문자열을 특수문자로 구분하여 합친 것
	vector<int> T_prime_start_indices;	   // T_i'가 concatenated_text_T_prime_all에서 시작하는 인덱스
	vector<int> T_prime_lengths;
	int num_original_strings;
	int current_global_end;	 // Ukkonen 알고리즘에서 사용되는 전역 끝 인덱스

	// 선형 시간 GST 구축 (예: Ukkonen)은 매우 복잡하므로, 여기서는 인터페이스만 가정.
	// 실제로는 이 생성자 내에서 모든 T_prime 문자열들을 사용하여 GST를 구축해야 함.
	GeneralizedSuffixTree(const vector<string>& all_T_prime) {
		root = new GSTNode();
		current_global_end = -1;
		num_original_strings = all_T_prime.size();

		// concatenated_text_T_prime_all 생성 및 T_prime_start_indices, T_prime_lengths 채우기
		for (int i = 0; i < num_original_strings; ++i) {
			T_prime_start_indices.push_back(concatenated_text_T_prime_all.length());
			concatenated_text_T_prime_all += all_T_prime[i];
			T_prime_lengths.push_back(all_T_prime[i].length());
			concatenated_text_T_prime_all += UNIQUE_END_CHAR;  // 각 T_i' 끝에 고유 문자 추가
		}

		// TODO: 여기에 Ukkonen의 알고리즘 등을 사용하여 GST를 선형 시간에 구축하는 로직.
		// build_linear_time_gst(all_T_prime);
		cout << "GST construction (linear time, e.g., Ukkonen's) needs to be implemented here." << endl;
	}

	~GeneralizedSuffixTree() {
		// TODO: 모든 노드 메모리 해제
	}

	// Step 3: T_i'의 초최대 반복 찾기 (GST 내에서 수행)
	// 각 T_i'에 대해, 이 GST 내에서 초최대 반복에 해당하는 노드들을 찾아 V_i에 추가.
	// 반환값: map<int (T_prime_idx), vector<GSTNode*>> (V_sets)
	map<int, vector<GSTNode*>> find_all_supermaximal_repeat_nodes_in_gst() {
		map<int, vector<GSTNode*>> V_sets;
		cout << "Finding supermaximal repeats for each T_i' within GST (Step 3)..." << endl;
		for (int i = 0; i < num_original_strings; ++i) {
			// TODO: Gusfield의 초최대 반복 정의를 사용하여 T_i'에 대한 SMR 노드 찾기.
			// 1. T_i'에만 속하는 서픽스들을 고려하여 GST 내부 노드 v 순회.
			// 2. v의 모든 자식들이 T_i'에 대한 리프 노드여야 함.
			// 3. 각 리프에 해당하는 서픽스의 왼쪽 문자가 T_i' 내에서 서로 달라야 함.
			// 이는 각 노드가 어떤 문자열의 서픽스를 나타내는지, 그리고 그 왼쪽 문자를 알아야 함.
			// GSTNode에 추가 정보 저장 또는 탐색 중 계산 필요.
			// 예시: V_sets[i].push_back(some_node_representing_SMR_of_T_prime_i);
			// some_node_representing_SMR_of_T_prime_i->is_supermaximal_for_T_prime_idx.push_back(i);
		}
		return V_sets;
	}

	// Step 3: ST_i의 노드 v에 저장된 (j,j') 정보를 바탕으로 GST에서 v' 찾기
	// (이 함수는 find_all_supermaximal_repeat_nodes_in_gst 내부 로직의 일부가 될 수 있음)
	// 또는, SMR 문자열을 직접 GST에서 찾는 방식도 고려 가능.
	// 논문은 ST에서 (j,j')를 얻고 GST에서 두 리프의 LCA로 v'를 찾는다고 기술.
	// 이는 별도의 ST를 구축하거나, GST 내에서 각 T_i'를 "가상 ST"처럼 다뤄야 함.
	// GSTNode* find_corresponding_node_in_gst(int T_prime_idx, pii st_lca_leaf_suffix_starts) {
	//    // 1. T_prime_idx 문자열에서 st_lca_leaf_suffix_starts.first 와 .second로 시작하는 접미사 찾기
	//    // 2. 이 두 접미사에 해당하는 GST의 리프 노드 찾기
	//    // 3. 두 리프 노드의 LCA 찾기 (선형 시간 LCA 알고리즘 필요)
	//    // return lca_node;
	//    return nullptr; // Placeholder
	//}

	// Step 4.1: 초최대 반복 S의 접미사들을 GST에 삽입 (S$ 형태)
	// V_node는 SMR S를 나타내는 GST 상의 노드, T_prime_original_idx는 S가 어떤 T_k'에서 왔는지의 k
	void insert_smr_suffixes_with_dollar(GSTNode* V_node, int T_prime_original_idx, set<GSTNode*>& N_k_set) {
		cout << "  Inserting suffixes of SMR (represented by V_node from T'_" << T_prime_original_idx << ") with '$'..." << endl;
		// 논문 : "V_i의 각 원소에서 루트 노드로 가는 suffix link를 따라간다."
		// "이전에 방문하지 않은 노드 u를 만나면, ID i를 가진 새 리프 노드를 u의 자식으로 '$' 레이블과 함께 생성."
		GSTNode* current_traversal_node = V_node;
		set<GSTNode*> visited_in_this_traversal;  // 이 특정 SMR의 suffix link traversal 중 방문한 노드

		while (current_traversal_node != nullptr && visited_in_this_traversal.find(current_traversal_node) == visited_in_this_traversal.end()) {
			visited_in_this_traversal.insert(current_traversal_node);

			// 새 리프 노드 ('$' 엣지) 생성 및 N_k_set에 추가
			// 실제 '$' 문자를 사용하거나, 특수 플래그로 리프임을 표시.
			// GSTNode* new_dollar_leaf = new GSTNode(...); // '$' 엣지 정보
			// new_dollar_leaf->is_smr_derived_leaf = true;
			// new_dollar_leaf->smr_derived_leaf_original_string_id = T_prime_original_idx;
			// current_traversal_node->children[SOME_UNIQUE_DOLLAR_CHAR_OR_FLAG] = new_dollar_leaf;
			// N_k_set.insert(new_dollar_leaf);

			if (current_traversal_node == root) break;
			current_traversal_node = current_traversal_node->suffix_link;
		}
	}

	// Step 4.2: GST_SMR에 포함될 노드 마킹
	// N_all_sets는 모든 N_k_set들의 합집합 (또는 각 N_k 리프에서 시작)
	void mark_nodes_for_gst_smr(const set<GSTNode*>& N_all_dollar_leaves) {
		cout << "  Marking nodes for GST_SMR (paths from N_all_dollar_leaves to root)..." << endl;
		// 각 N_all_dollar_leaves의 리프에서 루트까지 부모 포인터를 따라 올라가며 마킹.
		// 부모 포인터가 없다면 DFS/BFS 등으로 경로상 노드를 찾아야 함.
		// 이미 마킹된 노드를 만나면 중단.
		// for (GSTNode* leaf : N_all_dollar_leaves) {
		//    GSTNode* curr = leaf;
		//    while (curr != nullptr && !curr->marked_for_gst_smr) {
		//        curr->marked_for_gst_smr = true;
		//        // curr = curr->parent; // 부모 포인터 필요
		//    }
		// }
	}

	// Step 4.3: 불필요한 노드/엣지 제거 (또는 새 GST_SMR 구축)
	unique_ptr<GeneralizedSuffixTree> build_final_gst_smar() {
		cout << "  Building final GST_SMR by pruning/copying marked nodes (Step 4.3)..." << endl;
		// 1. 원본 GST(this)를 순회.
		// 2. marked_for_gst_smr == false 인 노드/엣지는 제거 (논문 ).
		//    - 원래 GST_U'의 리프에서 시작하여 unmarked 노드/엣지 제거.
		//    - 이후, 자식이 하나뿐인 내부 노드(marked 일지라도) 제거 및 엣지 병합.
		// 이 작업은 매우 복잡하며, 새 트리를 구축하는 것이 더 깔끔할 수 있음.
		// unique_ptr<GeneralizedSuffixTree> gst_smr_final = make_unique<GeneralizedSuffixTree>(...);
		// // ... 로직 ...
		// return gst_smr_final;
		return nullptr;	 // Placeholder
	}

	// Step 5: 최종 GST_SMR에서 최장 공통 반복 찾기
	// k_target은 몇 개의 원본 문자열에 공통으로 나타나야 하는지 (Problem 1의 k)
	string find_lcr_in_final_gst(int k_target) {
		cout << "Finding LCR in final GST_SMR (Step 5)..." << endl;
		// Hui의 알고리즘 적용:
		// 1. GST_SMR의 각 내부 노드 v에 대해, v의 서브트리에 있는 smr_derived_leaf들의
		//    smr_derived_leaf_original_string_id가 몇 종류(색깔)인지 계산.
		// 2. 이 종류가 k_target 이상인 노드 v 중에서, 루트로부터의 문자열 길이(깊이)가 가장 긴 것을 찾음.
		//    (색깔 수 계산은 LCA를 활용하여 효율적으로 수행)

		// 이 함수는 실제 최종 GST_SMR 객체에 대해 호출되어야 함.
		// string longest_str = "";
		// int max_len = -1;
		// function<set<int>(GSTNode*, int)> dfs_lcr =
		//     [&](GSTNode* u, int current_str_len) -> set<int> {
		//     set<int> colors_in_subtree;
		//     if (u->is_smr_derived_leaf) {
		//         colors_in_subtree.insert(u->smr_derived_leaf_original_string_id);
		//     }
		//     for (auto const& [edge_char, child_node] : u->children) {
		//         // GST_SMR의 엣지만 고려해야 함
		//         set<int> child_colors = dfs_lcr(child_node, current_str_len + child_node->get_edge_length(current_global_end));
		//         colors_in_subtree.insert(child_colors.begin(), child_colors.end());
		//     }
		//     if (colors_in_subtree.size() >= k_target) {
		//         if (current_str_len > max_len) {
		//             max_len = current_str_len;
		//             // longest_str = reconstruct_string_from_root_to_node(u);
		//         }
		//     }
		//     return colors_in_subtree;
		// };
		// dfs_lcr(this->root /* GST_SMR의 루트 */, 0);
		// return longest_str;
		return "placeholder_lcr_result";
	}

	// LCA 함수 (선형 시간 전처리 후 상수 시간 질의 필요)
	// GSTNode* get_lca(GSTNode* node1, GSTNode* node2) {
	//    // TODO: Implement Harel & Tarjan or Bender & Farach-Colton LCA.
	//    return root; // Placeholder
	// }
};

// --- Main Solver Class ---
class LongestCommonRepeatSolver {
   public:
	vector<string> original_T_strings;
	vector<string> T_prime_strings;					// 변형된 문자열 T_i' = T_i % T_i^R # T_i^RC
	unique_ptr<GeneralizedSuffixTree> gst_U_prime;	// 모든 T_prime_strings에 대한 GST

	LongestCommonRepeatSolver(const vector<string>& initial_strings)
		: original_T_strings(initial_strings) {}

	string solve(int k_target_commonality) {
		// Step 1: 문자열 변형 (T_i -> T_i')
		cout << "Step 1: Transforming strings..." << endl;
		for (const string& t : original_T_strings) {
			string t_r = reverse_string(t);
			string t_rc = reverse_complement_string(t);
			// 논문의 T_i' = T_i % T_i^R # T_i^RC 형식. 구분자 주의.
			// 여기서는 간단히 하나의 특수문자만 사용한다고 가정. 실제로는 여러개 필요.
			T_prime_strings.push_back(t + SPECIAL_CHAR_1 + t_r + SPECIAL_CHAR_1 + t_rc);
		}

		// Step 2: GST(T_1'...T_l') 구축
		// (개별 ST(T_i') 구축은 GST 구축 과정에 통합되거나 GST를 통해 정보 추출 가능)
		cout << "Step 2: Building GST for all T_prime strings..." << endl;
		gst_U_prime = make_unique<GeneralizedSuffixTree>(T_prime_strings);

		// Step 3: 각 T_i'의 초최대 반복 찾고, GST_U_prime의 해당 노드(V_i 집합) 식별
		cout << "Step 3: Finding supermaximal repeats and identifying V_i sets in GST..." << endl;
		map<int, vector<GSTNode*>> V_sets = gst_U_prime->find_all_supermaximal_repeat_nodes_in_gst();

		// Step 4: 초최대 반복들의 GST (GST_SMR) 구축
		cout << "Step 4: Building GST of Supermaximal Repeats (GST_SMR)..." << endl;
		// 4.1. 각 SMR의 접미사들을 '$'와 함께 GST_U_prime에 삽입, 새 리프 N_k 생성
		map<int, set<GSTNode*>> N_k_sets;	// 각 T_k'에서 유래한 SMR들의 '$'리프 집합
		set<GSTNode*> N_all_dollar_leaves;	// 모든 '$'리프들의 합집합
		for (int k = 0; k < T_prime_strings.size(); ++k) {
			if (V_sets.count(k)) {
				for (GSTNode* v_node_representing_smr : V_sets[k]) {
					gst_U_prime->insert_smr_suffixes_with_dollar(v_node_representing_smr, k, N_k_sets[k]);
				}
				N_all_dollar_leaves.insert(N_k_sets[k].begin(), N_k_sets[k].end());
			}
		}

		// 4.2. N_k 리프에서 루트까지 경로상 노드 마킹
		gst_U_prime->mark_nodes_for_gst_smr(N_all_dollar_leaves);

		// 4.3. GST_U_prime을 정리하여 최종 GST_SMR 얻기 (또는 새로 구축)
		unique_ptr<GeneralizedSuffixTree> gst_SMR_final = gst_U_prime->build_final_gst_smar();
		// 이후 모든 연산은 gst_SMR_final에 대해 수행되어야 함.
		// 지금은 gst_SMR_final이 완성되었다고 가정하고, gst_U_prime을 (마킹된 상태로) 그대로 사용.

		// Step 5: GST_SMR에서 최장 공통 반복 찾기
		cout << "Step 5: Finding LCR in the final GST_SMR..." << endl;
		if (gst_SMR_final) {  // 만약 build_final_gst_smar가 실제 객체를 반환했다면
			return gst_SMR_final->find_lcr_in_final_gst(k_target_commonality);
		} else {  // 아니라면, 마킹된 gst_U_prime에 대해 (개념적으로) 수행
			// 이 경우, find_lcr_in_final_gst는 마킹된 노드만 고려하도록 수정 필요
			return gst_U_prime->find_lcr_in_final_gst(k_target_commonality);
		}
	}
};

int main() {
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);

	// 사용자 입력 예시
	int num_strings_l;
	cout << "Enter number of original strings (l): ";
	cin >> num_strings_l;
	vector<string> input_strings(num_strings_l);
	cin.ignore();  // Consume newline
	for (int i = 0; i < num_strings_l; ++i) {
		cout << "Enter string T_" << i + 1 << ": ";
		getline(cin, input_strings[i]);
	}

	int k_common;
	cout << "Enter number of strings for common repeat (k, where 1 <= k <= " << num_strings_l << "): ";
	cin >> k_common;

	if (k_common <= 0 || k_common > num_strings_l) {
		cerr << "Invalid value for k." << endl;
		return 1;
	}

	LongestCommonRepeatSolver solver(input_strings);
	string result = solver.solve(k_common);

	cout << "\n------------------------------------------" << endl;
	cout << "Longest Common (" << k_common << ", " << num_strings_l << ") Repeat: " << (result.empty() ? "Not found or placeholder" : result) << endl;
	cout << "------------------------------------------" << endl;

	return 0;
}
