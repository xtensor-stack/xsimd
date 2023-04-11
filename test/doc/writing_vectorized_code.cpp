#include <cstddef>
#include <vector>

void mean(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& res)
{
    std::size_t size = res.size();
    for (std::size_t i = 0; i < size; ++i)
    {
        res[i] = (a[i] + b[i]) / 2;
    }
}
