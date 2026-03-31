#include <gtest/gtest.h>
#include <lunaml/lunaml.hpp>
using namespace nf;

TEST(IntegrationTest, XORConvergence) {
    Random::manual_seed(42);

    auto X = Tensor::from_data({0,0, 0,1, 1,0, 1,1}, {4, 2});
    auto Y = Tensor::from_data({0, 1, 1, 0}, {4, 1});

    nn::Sequential model;
    model.add(std::make_shared<nn::Linear>(2, 16));
    model.add(std::make_shared<nn::ReLU>());
    model.add(std::make_shared<nn::Linear>(16, 1));
    model.add(std::make_shared<nn::Sigmoid>());

    optim::Adam optimizer(model.parameters(), 0.01f);
    loss::BCE criterion;

    float final_loss = 0;
    for (int epoch = 0; epoch < 2000; ++epoch) {
        auto pred = model.forward(X);
        auto l = criterion(pred, Y);
        optimizer.zero_grad();
        l.backward();
        optimizer.step();
        final_loss = l.item();
    }

    EXPECT_LT(final_loss, 0.1f);

    auto pred = model.forward(X);
    float acc = metrics::accuracy(pred, Y);
    EXPECT_GE(acc, 0.75f);
}

TEST(IntegrationTest, LinearRegression) {
    Random::manual_seed(123);

    // y = 2x + 1
    size_t N = 50;
    auto X = Tensor::rand({N, 1}) * 5.0f;
    auto Y = X * 2.0f + 1.0f;

    nn::Sequential model;
    model.add(std::make_shared<nn::Linear>(1, 1));

    optim::SGD optimizer(model.parameters(), 0.01f);
    loss::MSE criterion;

    for (int epoch = 0; epoch < 500; ++epoch) {
        auto pred = model.forward(X);
        auto l = criterion(pred, Y);
        optimizer.zero_grad();
        l.backward();
        optimizer.step();
    }

    auto params = model.parameters();
    float w = params[0]->data()[0];
    float b = params[1]->data()[0];

    EXPECT_NEAR(w, 2.0f, 0.5f);
    EXPECT_NEAR(b, 1.0f, 1.0f);
}

TEST(IntegrationTest, SaveLoadModel) {
    Random::manual_seed(42);
    nn::Linear linear(3, 2);
    auto params = linear.parameters();

    float w0_before = params[0]->data()[0];

    utils::Serialization::save(params, "/tmp/nf_test_model.nf");

    // Modify
    params[0]->data()[0] = 999.0f;
    EXPECT_FLOAT_EQ(params[0]->data()[0], 999.0f);

    // Load
    utils::Serialization::load(params, "/tmp/nf_test_model.nf");
    EXPECT_FLOAT_EQ(params[0]->data()[0], w0_before);
}
