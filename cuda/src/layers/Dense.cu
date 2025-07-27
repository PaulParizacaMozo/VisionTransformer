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
    // input.printContents("Dense::forward - Input");
    if (inputRank == 3)
    {
        size_t batch = input.dim(0), tokens = input.dim(1), inF = input.dim(2);
        size_t reshaped[2] = {batch * tokens, inF};

        Tensor flat = input.reshape(reshaped, 2);
        // flat.printContents("Dense::forward - After reshape to 2D");
        // this->weights.printContents("Dense::forward - Weights");

        Tensor out = matrixMultiply(flat, this->weights);
        // out.printContents("Dense::forward - After matrixMultiply");
        // this->bias.printContents("Dense::forward - Bias");

        out.addBroadcast(this->bias);
        // out.printContents("Dense::forward - After addBroadcast(bias)");

        size_t reshapedBack[3] = {batch, tokens, this->bias.dim(1)};
        out = out.reshape(reshapedBack, 3);
        // out.printContents("Dense::forward - After reshape back to 3D");
        return out;
    }
    if (inputRank == 2)
    {
        // input.printContents("Dense::forward - Input 2D");

        // this->weights.printContents("Dense::forward - Weights");

        Tensor output = matrixMultiply(input, this->weights);
        // output.printContents("Dense::forward - After matrixMultiply");
        // this->bias.printContents("Dense::forward - Bias");
        output.addBroadcast(this->bias);
        // output.printContents("Dense::forward - After addBroadcast(bias)");
        return output;
    }
    throw std::runtime_error("Dense::forward solo soporta entradas 2D o 3D.");
}

Tensor Dense::backward(const Tensor &outputGradient)
{
    outputGradient.printFirstElements("Dense::backward - Output Gradient");
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
        // input.printContents("Dense::backward - Input 3D");
        // dout.printContents("Dense::backward - Output Gradient 3D");

        input = input.reshape(flatShape, 2);
        dout = dout.reshape(gradShape, 2);
        // input.printContents("Dense::backward - Reshaped Input 2D");
        // dout.printContents("Dense::backward - Reshaped Output Gradient 2D");
    }

    Tensor inputT = input.transpose(0, 1);
    // inputT.printContents("Dense::backward - Transposed Input");
    this->weightGradients = matrixMultiply(inputT, dout);
    // this->weightGradients.printContents("Dense::backward - Weight Gradients");
    this->biasGradients = dout.sum(0);
    // this->biasGradients.printContents("Dense::backward - Bias Gradients");

    Tensor copy_weights(weights);
    Tensor weightsT = copy_weights.transpose(0, 1);
    // weightsT.printContents("Dense::backward - Transposed Weights");
    Tensor dx = matrixMultiply(dout, weightsT);
    // dx.printContents("Dense::backward - Gradient w.r.t Input");

    if (rank == 3)
    {
        size_t originalShape[3] = {
            this->inputTensor.dim(0),
            this->inputTensor.dim(1),
            this->inputTensor.dim(2)};
        Tensor dx_copy(dx); // Copy to avoid modifying the original dx
        dx = dx_copy.reshape(originalShape, 3);
        // dx.printContents("Dense::backward - Reshaped Gradient w.r.t Input 3D");
        return dx;
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
