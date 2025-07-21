#include "layers/Dense.cuh"
#include <stdexcept>
#include <cmath>

Dense::Dense(size_t inputSize, size_t outputSize)
{
    float stddev = std::sqrt(2.0f / static_cast<float>(inputSize));

    this->weights = Tensor({inputSize, outputSize});
    this->weights.randomizeNormal(0.0f, stddev);

    this->bias = Tensor({1, outputSize});
    this->bias.fill(0.0f);

    this->weightGradients = Tensor({inputSize, outputSize});
    this->biasGradients = Tensor({1, outputSize});
}

Tensor Dense::forward(const Tensor &input, bool isTraining)
{
    if (isTraining)
        this->inputTensor = input; // Save input for backward

    size_t inputRank = input.dims();
    if (inputRank == 3)
    {
        size_t batch = input.dim(0), tokens = input.dim(1), inF = input.dim(2);
        size_t reshaped[2] = {batch * tokens, inF};

        Tensor flat = input.reshape(reshaped, 2);

        Tensor out = matrixMultiply(flat, this->weights);

        out.addBroadcast(this->bias);

        size_t reshapedBack[3] = {batch, tokens, this->bias.dim(1)};
        out = out.reshape(reshapedBack, 3);
        return out;
    }
    if (inputRank == 2)
    {
        Tensor output = matrixMultiply(input, this->weights);
        output.addBroadcast(this->bias);
        return output;
    }
    throw std::runtime_error("Dense::forward solo soporta entradas 2D o 3D.");
}

Tensor Dense::backward(const Tensor &outputGradient)
{
    size_t rank = this->inputTensor.dims();
    Tensor input = this->inputTensor;
    Tensor dout = outputGradient;

    if (rank == 3)
    {
        size_t batch = input.dim(0), tokens = input.dim(1), inF = input.dim(2), outF = outputGradient.dim(2);
        size_t flatShape[2] = {batch * tokens, inF};
        size_t gradShape[2] = {batch * tokens, outF};

        if (!input.isContiguous())
            input = input.contiguous();
        if (!dout.isContiguous())
            dout = dout.contiguous();

        input = input.reshape(flatShape, 2);
        dout = dout.reshape(gradShape, 2);
    }

    Tensor inputT = input.transpose(0, 1);
    this->weightGradients = matrixMultiply(inputT, dout);
    this->biasGradients = dout.sum(0);

    Tensor weightsT = this->weights.transpose(0, 1);
    Tensor dx = matrixMultiply(dout, weightsT);

    if (rank == 3)
    {
        size_t originalShape[3] = {
            this->inputTensor.dim(0),
            this->inputTensor.dim(1),
            this->inputTensor.dim(2)};
        return dx.reshape(originalShape, 3);
    }
    return dx;
}

std::vector<Tensor *> Dense::getParameters()
{
    return {&this->weights, &this->bias};
}

std::vector<Tensor *> Dense::getGradients()
{
    return {&this->weightGradients, &this->biasGradients};
}
