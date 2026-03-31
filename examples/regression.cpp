// ══════════════════════════════════════════════
// LunaML — Linear Regression Example
// ══════════════════════════════════════════════
#include <lunaml/lunaml.hpp>

using namespace nf;

int main() {
    Random::manual_seed(42);

    // Generate synthetic data: y = 3x + 2 + noise
    size_t N = 100;
    auto X = Tensor::rand({N, 1}) * 10.0f;  // [0, 10]
    auto noise = Tensor::randn({N, 1}) * 0.5f;
    auto Y = X * 3.0f + 2.0f + noise;

    // Simple linear model
    nn::Sequential model;
    model.add(std::make_shared<nn::Linear>(1, 1));

    auto optimizer = optim::SGD(model.parameters(), 0.001f);
    auto criterion = loss::MSE();

    std::cout << "═══════════════════════════════════════\n";
    std::cout << "  LunaML — Linear Regression\n";
    std::cout << "  Target: y = 3x + 2\n";
    std::cout << "═══════════════════════════════════════\n\n";

    utils::Logger logger(500);

    for (int epoch = 0; epoch < 500; ++epoch) {
        auto pred = model.forward(X);
        auto l = criterion(pred, Y);

        optimizer.zero_grad();
        l.backward();
        optimizer.step();

        if (epoch % 50 == 0 || epoch == 499) {
            logger.log_epoch(epoch, l.item());
        }
    }

    // Print learned parameters
    auto params = model.parameters();
    std::cout << "\n\n─── Learned Parameters ───\n";
    std::cout << "  Weight (expected ~3.0): " << params[0]->data()[0] << "\n";
    std::cout << "  Bias   (expected ~2.0): " << params[1]->data()[0] << "\n";

    // Test prediction
    auto test_x = Tensor::from_data({5.0f}, {1, 1});
    auto test_pred = model.forward(test_x);
    std::cout << "\n  f(5.0) = " << std::fixed << std::setprecision(4)
              << test_pred.item() << " (expected ~17.0)\n";

    std::cout << "\n✓ Training Complete!\n";
    return 0;
}
