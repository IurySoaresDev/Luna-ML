#include <gtest/gtest.h>
#include <lunaml/lunaml.hpp>
using namespace nf;

TEST(LayersTest, LinearForward) {
    Random::manual_seed(42);
    nn::Linear linear(3, 2);
    auto input = Tensor::ones({4, 3});
    auto output = linear.forward(input);
    EXPECT_EQ(output.shape(0), 4);
    EXPECT_EQ(output.shape(1), 2);
}

TEST(LayersTest, LinearParameters) {
    nn::Linear linear(3, 2);
    auto params = linear.parameters();
    EXPECT_EQ(params.size(), 2); // weight + bias
    EXPECT_EQ(params[0]->shape(0), 2); // out_features
    EXPECT_EQ(params[0]->shape(1), 3); // in_features
}

TEST(LayersTest, ReLU) {
    auto input = Tensor::from_data({-2, -1, 0, 1, 2}, {5});
    nn::ReLU relu;
    auto output = relu.forward(input);
    EXPECT_FLOAT_EQ(output.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(output.data()[1], 0.0f);
    EXPECT_FLOAT_EQ(output.data()[2], 0.0f);
    EXPECT_FLOAT_EQ(output.data()[3], 1.0f);
    EXPECT_FLOAT_EQ(output.data()[4], 2.0f);
}

TEST(LayersTest, Sigmoid) {
    auto input = Tensor::from_data({0.0f}, {1});
    nn::Sigmoid sig;
    auto output = sig.forward(input);
    EXPECT_NEAR(output.data()[0], 0.5f, 1e-5);
}

TEST(LayersTest, Sequential) {
    Random::manual_seed(42);
    nn::Sequential model;
    model.add(std::make_shared<nn::Linear>(2, 4));
    model.add(std::make_shared<nn::ReLU>());
    model.add(std::make_shared<nn::Linear>(4, 1));

    auto input = Tensor::ones({1, 2});
    auto output = model.forward(input);
    EXPECT_EQ(output.shape(0), 1);
    EXPECT_EQ(output.shape(1), 1);
}

TEST(LayersTest, SequentialParameters) {
    nn::Sequential model;
    model.add(std::make_shared<nn::Linear>(2, 4));
    model.add(std::make_shared<nn::ReLU>());
    model.add(std::make_shared<nn::Linear>(4, 1));
    auto params = model.parameters();
    EXPECT_EQ(params.size(), 4); // 2 layers * (weight + bias)
}

TEST(LayersTest, DropoutTrain) {
    Random::manual_seed(42);
    nn::Dropout dropout(0.5f);
    dropout.train();
    auto input = Tensor::ones({100});
    auto output = dropout.forward(input);
    // Some should be 0, some should be scaled up
    int zeros = 0;
    for (size_t i = 0; i < output.size(); ++i)
        if (output.data()[i] == 0.0f) zeros++;
    EXPECT_GT(zeros, 10);
    EXPECT_LT(zeros, 90);
}

TEST(LayersTest, DropoutEval) {
    nn::Dropout dropout(0.5f);
    dropout.eval();
    auto input = Tensor::ones({100});
    auto output = dropout.forward(input);
    for (size_t i = 0; i < output.size(); ++i)
        EXPECT_FLOAT_EQ(output.data()[i], 1.0f);
}
