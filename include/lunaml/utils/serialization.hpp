#pragma once
#include "lunaml/nn/module.hpp"
#include <fstream>
#include <string>

namespace nf { namespace utils {

class Serialization {
public:
    static void save(const std::vector<Tensor*>& params, const std::string& filepath);
    static void load(std::vector<Tensor*>& params, const std::string& filepath);
};

}} // namespace nf::utils
