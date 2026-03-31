#pragma once
#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

namespace nf { namespace utils {

class Logger {
public:
    Logger(int total_epochs) : total_epochs_(total_epochs) {}

    void log_epoch(int epoch, float loss, float accuracy = -1.0f) {
        auto now = std::chrono::steady_clock::now();
        float elapsed = 0;
        if (epoch > 0) {
            elapsed = std::chrono::duration<float>(now - last_time_).count();
        }
        last_time_ = now;

        // Progress bar
        int bar_width = 30;
        float progress = static_cast<float>(epoch + 1) / total_epochs_;
        int filled = static_cast<int>(bar_width * progress);

        std::cout << "\r[";
        for (int i = 0; i < bar_width; ++i) {
            if (i < filled) std::cout << "█";
            else std::cout << "░";
        }
        std::cout << "] ";

        std::cout << std::fixed << std::setprecision(1);
        std::cout << (progress * 100.0f) << "% | ";
        std::cout << "Epoch " << std::setw(4) << (epoch + 1) << "/" << total_epochs_;
        std::cout << std::setprecision(6);
        std::cout << " | Loss: " << loss;

        if (accuracy >= 0) {
            std::cout << std::setprecision(2) << " | Acc: " << (accuracy * 100.0f) << "%";
        }

        if (epoch > 0 && elapsed > 0) {
            std::cout << std::setprecision(1) << " | " << elapsed << "s/epoch";
        }

        std::cout << std::flush;

        if (epoch + 1 == total_epochs_) std::cout << "\n";

        losses_.push_back(loss);
    }

    const std::vector<float>& losses() const { return losses_; }

private:
    int total_epochs_;
    std::chrono::steady_clock::time_point last_time_;
    std::vector<float> losses_;
};

}} // namespace nf::utils
