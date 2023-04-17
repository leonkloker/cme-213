#include <vector>

std::vector<uint> serialSum(const std::vector<uint> &v) {
    std::vector<uint> sums(2);
    for (uint i = 0; i < v.size(); i++) {
        if (v[i] % 2 == 0) sums[0] += v[i];
        else sums[1] += v[i];
    }
    return sums;
}

std::vector<uint> parallelSum(const std::vector<uint> &v) {
    std::vector<uint> sums(2);
    uint even_sum = 0;
    uint odd_sum = 0;
    #pragma omp parallel for reduction (+:even_sum,odd_sum)
    for (uint i = 0; i < v.size(); i++) {
        if (v[i] % 2 == 0) even_sum += v[i];
        else odd_sum += v[i];
    }
    sums[0] = even_sum;
    sums[1] = odd_sum;
    return sums;
}