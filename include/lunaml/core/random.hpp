#pragma once

#include "types.hpp"

namespace nf {

class Random {
public:
    static void manual_seed(unsigned int seed);
    static std::mt19937& generator();
    static float uniform(float low = 0.0f, float high = 1.0f);
    static float normal(float mean = 0.0f, float stddev = 1.0f);

private:
    static std::mt19937 gen_;
};

} // namespace nf
