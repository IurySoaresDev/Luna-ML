#include "lunaml/core/random.hpp"

namespace nf {

std::mt19937 Random::gen_(std::random_device{}());

void Random::manual_seed(unsigned int seed) {
    gen_.seed(seed);
}

std::mt19937& Random::generator() {
    return gen_;
}

float Random::uniform(float low, float high) {
    std::uniform_real_distribution<float> dist(low, high);
    return dist(gen_);
}

float Random::normal(float mean, float stddev) {
    std::normal_distribution<float> dist(mean, stddev);
    return dist(gen_);
}

} // namespace nf
