#include "layers/Dense.cuh"
#include <stdexcept>
#include <cmath>

__host__ Dense::Dense(size_t inputSize, size_t outputSize)
{
    float stddev = std::sqrt(2.0f / static_cast<float>(inputSize));

    this->weights = Tensor({inputSize, outputSize});
    this->weights.randomizeNormal(0.0f, stddev);

    this->bias = Tensor({1, outputSize});
    this->bias.fill(0.0f);

    this->weightGradients = Tensor({inputSize, outputSize});
    this->biasGradients = Tensor({1, outputSize});
}

__host__ Tensor Dense::forward(const Tensor &input, bool isTraining)
{
    if (isTraining)
        this->inputTensor = input; // Save input for backward

    size_t inputRank = input.dims();
    if (inputRank == 3)
    {
        size_t batchSize = input.dim(0);
        size_t numTokens = input.dim(1);
        size_t featuresIn = input.dim(2);
        size_t newShape1[2] = {batchSize * numTokens, featuresIn};
        Tensor input2D = input.reshape(newShape1, 2);
        Tensor output2D = matrixMultiply(input2D, this->weights);
        output2D.addBroadcast(this->bias);
        size_t newShape2[3] = {batchSize, numTokens, this->bias.dim(1)};
        return output2D.reshape(newShape2, 3);
    }
    if (inputRank == 2)
    {
        Tensor output = matrixMultiply(input, this->weights);
        output.addBroadcast(this->bias);
        return output;
    }
    throw std::runtime_error("Dense::forward solo soporta entradas 2D o 3D.");
}

__host__ Tensor Dense::backward(const Tensor &outputGradient)
{
    size_t inputRank = this->inputTensor.dims();

    Tensor grad = outputGradient;
    Tensor input = this->inputTensor;

    if (inputRank == 3)
    {
        size_t batchSize = input.dim(0);
        size_t numTokens = input.dim(1);
        size_t featuresIn = input.dim(2);
        size_t featuresOut = grad.dim(2);

        if (!grad.isContiguous())
            grad = grad.contiguous();
        if (!input.isContiguous())
            input = input.contiguous();

        size_t reshapedInputShape[2] = {batchSize * numTokens, featuresIn};
        size_t reshapedGradShape[2] = {batchSize * numTokens, featuresOut};

        input = input.reshape(reshapedInputShape, 2);
        grad = grad.reshape(reshapedGradShape, 2);
    }

    Tensor inputT = input.transpose(0, 1);
    this->weightGradients = matrixMultiply(inputT, grad);
    this->biasGradients = grad.sum(0);

    Tensor weightsT = this->weights.transpose(0, 1);
    Tensor inputGrad = matrixMultiply(grad, weightsT);

    if (inputRank == 3)
    {
        size_t originalShape[3] = {
            this->inputTensor.dim(0),
            this->inputTensor.dim(1),
            this->inputTensor.dim(2)};
        return inputGrad.reshape(originalShape, 3);
    }
    return inputGrad;
}

__host__ std::vector<Tensor *> Dense::getParameters()
{
    return {&this->weights, &this->bias};
}

__host__ std::vector<Tensor *> Dense::getGradients()
{
    return {&this->weightGradients, &this->biasGradients};
}
