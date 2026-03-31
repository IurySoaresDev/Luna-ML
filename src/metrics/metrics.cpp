#include "lunaml/metrics/metrics.hpp"

namespace nf { namespace metrics {

float accuracy(const Tensor& predictions, const Tensor& targets) {
    size_t correct = 0;
    if (predictions.ndim() == 2 && predictions.shape(1) > 1) {
        // Multi-class: compare argmax
        auto pred_classes = predictions.argmax(1);
        for (size_t i = 0; i < targets.size(); ++i) {
            if (static_cast<int>(pred_classes.data()[i]) == static_cast<int>(targets.data()[i]))
                correct++;
        }
        return static_cast<float>(correct) / static_cast<float>(targets.size());
    }
    // Binary: threshold at 0.5
    for (size_t i = 0; i < predictions.size(); ++i) {
        int pred = predictions.data()[i] >= 0.5f ? 1 : 0;
        int tgt = static_cast<int>(targets.data()[i]);
        if (pred == tgt) correct++;
    }
    return static_cast<float>(correct) / static_cast<float>(predictions.size());
}

float precision(const Tensor& predictions, const Tensor& targets, float threshold) {
    size_t tp = 0, fp = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        int pred = predictions.data()[i] >= threshold ? 1 : 0;
        int tgt = static_cast<int>(targets.data()[i]);
        if (pred == 1 && tgt == 1) tp++;
        if (pred == 1 && tgt == 0) fp++;
    }
    return (tp + fp) > 0 ? static_cast<float>(tp) / (tp + fp) : 0.0f;
}

float recall(const Tensor& predictions, const Tensor& targets, float threshold) {
    size_t tp = 0, fn = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        int pred = predictions.data()[i] >= threshold ? 1 : 0;
        int tgt = static_cast<int>(targets.data()[i]);
        if (pred == 1 && tgt == 1) tp++;
        if (pred == 0 && tgt == 1) fn++;
    }
    return (tp + fn) > 0 ? static_cast<float>(tp) / (tp + fn) : 0.0f;
}

float f1_score(const Tensor& predictions, const Tensor& targets, float threshold) {
    float p = precision(predictions, targets, threshold);
    float r = recall(predictions, targets, threshold);
    return (p + r) > 0 ? 2.0f * p * r / (p + r) : 0.0f;
}

}} // namespace nf::metrics
