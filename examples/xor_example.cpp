// ══════════════════════════════════════════════
// LunaML — XOR Example
// Trains a neural network to solve XOR
// ══════════════════════════════════════════════
#include <lunaml/lunaml.hpp>

using namespace nf;

int main() {
    Random::manual_seed(42);

    // XOR dataset
    auto X = Tensor::from_data({0,0, 0,1, 1,0, 1,1}, {4, 2});
    auto Y = Tensor::from_data({0, 1, 1, 0}, {4, 1});

    // Model: 2 -> 16 -> 1
    nn::Sequential model;
    model.add(std::make_shared<nn::Linear>(2, 16));
    model.add(std::make_shared<nn::ReLU>());
    model.add(std::make_shared<nn::Linear>(16, 1));
    model.add(std::make_shared<nn::Sigmoid>());

    auto optimizer = optim::Adam(model.parameters(), 0.01f);
    auto criterion = loss::BCE();

    utils::Logger logger(1000);

    std::cout << "═══════════════════════════════════════\n";
    std::cout << "  LunaML — XOR Network Training\n";
    std::cout << "═══════════════════════════════════════\n\n";

    for (int epoch = 0; epoch < 1000; ++epoch) {
        auto pred = model.forward(X);
        auto l = criterion(pred, Y);

        optimizer.zero_grad();
        l.backward();
        optimizer.step();

        if (epoch % 100 == 0 || epoch == 999) {
            float acc = metrics::accuracy(pred, Y);
            logger.log_epoch(epoch, l.item(), acc);
        }
    }

    std::cout << "\n─── Predictions ───\n";
    auto final_pred = model.forward(X);
    for (size_t i = 0; i < 4; ++i) {
        std::cout << "  Input: [" << X.data()[i*2] << ", " << X.data()[i*2+1]
                  << "] => Predicted: " << std::fixed << std::setprecision(4)
                  << final_pred.data()[i]
                  << " (Expected: " << Y.data()[i] << ")\n";
    }

    std::cout << "\n✓ Training Complete!\n";
    return 0;
}
