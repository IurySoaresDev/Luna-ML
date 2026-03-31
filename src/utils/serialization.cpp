#include "lunaml/utils/serialization.hpp"

namespace nf { namespace utils {

void Serialization::save(const std::vector<Tensor*>& params, const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file for saving: " + filepath);

    // Write number of parameters
    size_t num_params = params.size();
    file.write(reinterpret_cast<const char*>(&num_params), sizeof(size_t));

    for (auto* p : params) {
        // Write ndim
        size_t ndim = p->ndim();
        file.write(reinterpret_cast<const char*>(&ndim), sizeof(size_t));
        // Write shape
        for (size_t d = 0; d < ndim; ++d) {
            size_t dim = p->shape(d);
            file.write(reinterpret_cast<const char*>(&dim), sizeof(size_t));
        }
        // Write data
        file.write(reinterpret_cast<const char*>(p->data()), p->size() * sizeof(float));
    }
}

void Serialization::load(std::vector<Tensor*>& params, const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file for loading: " + filepath);

    size_t num_params;
    file.read(reinterpret_cast<char*>(&num_params), sizeof(size_t));

    if (num_params != params.size()) {
        throw std::runtime_error("Parameter count mismatch");
    }

    for (auto* p : params) {
        size_t ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(size_t));
        Shape shape(ndim);
        for (size_t d = 0; d < ndim; ++d) {
            file.read(reinterpret_cast<char*>(&shape[d]), sizeof(size_t));
        }
        if (shape != p->shape()) {
            throw std::runtime_error("Shape mismatch during load");
        }
        file.read(reinterpret_cast<char*>(p->data()), p->size() * sizeof(float));
    }
}

}} // namespace nf::utils
