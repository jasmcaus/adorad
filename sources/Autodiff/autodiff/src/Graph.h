#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

template <typename T> static void _getIndegrees(const T& root, std::unordered_map<T, int>& indegree, std::unordered_set<T>& visited) {
    visited.insert(root);
    for (const auto& child : root->children) {
        indegree[child] += 1;
        if (!visited.count(child)) {
            _getIndegrees(child, indegree, visited);
        }
    }
}

template <typename T> static std::unordered_map<T, int> getIndegrees(const T& root) {
    std::unordered_map<T, int> indegrees;
    std::unordered_set<T> visited;
    _getIndegrees(root, indegrees, visited);
    return indegrees;
}

template <typename T> std::vector<T> topologicalSort(const T& root) {
    std::vector<T> ret;
    std::unordered_map<T, int> indegrees = getIndegrees(root);
    std::queue<T> queue;
    queue.push(root);
    while (!queue.empty()) {
        const T& x = queue.front();
        queue.pop();
        ret.push_back(x);
        for (const T& child : x->children) {
            indegrees[child] -= 1;
            if (indegrees[child] == 0) {
                queue.push(child);
            }
        }
    }
    return ret;
}

template <typename T> std::vector<T> getParentsOfX(const T& root, const T& x) {
    std::vector<T> ret;
    std::queue<T> queue;
    queue.push(root);
    while (!queue.empty()) {
        const T& node = queue.front();
        queue.pop();
        for (const T& child : node->children) {
            if (x == child) {
                ret.push_back(node);
            } else {
                queue.push(child);
            }
        }
    }
    return ret;
}
