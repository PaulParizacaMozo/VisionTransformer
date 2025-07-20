#include "core/Tensor.cuh"
#include "layers/Dense.cuh"
#include "activations/GELU.cuh"
#include "activations/RELU.cuh"
#include "optimizers/Adam.cuh"
#include "losses/CrossEntropy.cuh"
#include <iostream>
#include <vector>
#include <assert.h>

void TestTensor()
{
    std::cout << "=== Test Tensor CUDA ===\n";

    // --- Test creación y fill ---
    Tensor A({2, 3});
    // A.printDebugInfo("A (unfill 2)");
    A.fill(2.0f);
    A.printDebugInfo("A (fill 2)");

    // --- Test randomize uniforme ---
    Tensor B({2, 3});
    B.randomize(0.0f, 1.0f);
    B.printDebugInfo("B (random uniform)");

    // --- Test randomize normal ---
    Tensor C({2, 3});
    C.randomizeNormal(0.0f, 0.5f);
    C.printDebugInfo("C (random normal)");

    // --- Test suma A + B ---
    Tensor D = A + B;
    D.printDebugInfo("D = A + B");

    // --- Test square ---
    Tensor Dsq = D.square();
    Dsq.printDebugInfo("D^2");

    // --- Test sum(axis=0) ---
    Tensor sum0 = D.sum(0);
    sum0.printDebugInfo("sum(D, axis=0)");
    D.printContents("D contents");
    sum0.printContents("sum0 contents");
    // --- Test sum(axis=1) ---
    Tensor sum1 = D.sum(1);
    sum1.printDebugInfo("sum(D, axis=1)");
    sum1.printContents("sum1 contents");

    // --- Test slice(axis=0) ---
    Tensor slice0 = D.slice(0, 0, 1);
    slice0.printDebugInfo("slice(D, axis=0, start=0, count=1)");

    // --- Test reshape ---
    size_t shapeReshape[] = {3, 2};
    Tensor D_reshaped = D.reshape(shapeReshape, 2);
    D_reshaped.printDebugInfo("D reshaped (3,2)");

    // --- Test transpose ---
    Tensor D_trans = D.transpose(0, 1);
    D_trans.printDebugInfo("D transposed");

    // --- Test contiguous (de un slice o transposed) ---
    Tensor D_contig = D_trans.contiguous();
    D_contig.printDebugInfo("D_trans contiguous");

    // --- Test addBroadcast (bias) ---
    Tensor bias({1, 3});
    bias.fill(0.5f);
    D.addBroadcast(bias);
    D.printDebugInfo("D + broadcast(bias)");

    // --- Test matrixMultiply ---
    Tensor MM1({4, 5});
    Tensor MM2({5, 6});
    MM1.randomize();
    MM2.randomize();
    Tensor MM = matrixMultiply(MM1, MM2);
    MM.printDebugInfo("matrixMultiply(MM1, MM2)");
    MM1.printContents("MM1 contents");
    MM2.printContents("MM2 contents");
    MM.printContents("MM contents");
    // --- Test concatenate ---
    Tensor T1({2, 1, 4});
    Tensor T2({2, 2, 4});
    T1.fill(1.0f);
    T2.fill(2.0f);
    Tensor ts[] = {T1, T2};
    Tensor cat = concatenate(ts, 2, 1);
    cat.printDebugInfo("Concatenate T1, T2 (axis=1)");
    T1.printContents("T1 contents");
    T2.printContents("T2 contents");
    cat.printContents("Concatenated contents");

    // --- Test expand ---
    Tensor V({3, 5});
    V.randomize();
    Tensor Vexp = expand(V, 0, 2); // (3,5) -> (2,3,5)
    Vexp.printDebugInfo("Expand V(3,5) along dim=0 to 2");
    V.printContents("V contents");
    Vexp.printContents("V expanded contents");
    std::cout << "=== Test Tensor completado ===\n";
}

void TestDense()
{
    std::cout << "=== Test Dense Layer ===" << std::endl;

    // Parámetros de prueba
    size_t batchSize = 2;
    size_t inputSize = 4;
    size_t outputSize = 3;

    // Crear entrada y capa Dense
    Tensor input({batchSize, inputSize});
    input.randomize();

    Dense dense(inputSize, outputSize);

    // FORWARD
    Tensor output = dense.forward(input, true);
    output.printDebugInfo("Dense::forward output");
    input.printContents("Input contents");
    output.printContents("Output contents");

    // Verifica que la salida tenga shape [batchSize, outputSize]
    assert(output.dims() == 2);
    assert(output.dim(0) == batchSize);
    assert(output.dim(1) == outputSize);

    // BACKWARD
    Tensor outputGrad({batchSize, outputSize});
    outputGrad.randomize(); // Gradiente simulado desde la siguiente capa

    Tensor inputGrad = dense.backward(outputGrad);
    inputGrad.printDebugInfo("Dense::backward input gradient");
    outputGrad.printContents("Output gradient contents");
    inputGrad.printContents("Input gradient contents");

    // Verifica que el gradiente de entrada tenga shape [batchSize, inputSize]
    assert(inputGrad.dims() == 2);
    assert(inputGrad.dim(0) == batchSize);
    assert(inputGrad.dim(1) == inputSize);

    // Parámetros y gradientes
    auto params = dense.getParameters();
    auto grads = dense.getGradients();

    assert(params.size() == 2);
    assert(grads.size() == 2);

    params[0]->printDebugInfo("Weights");
    params[1]->printDebugInfo("Bias");
    grads[0]->printDebugInfo("Weight Gradients");
    grads[1]->printDebugInfo("Bias Gradients");

    std::cout << "=== Test Dense completado ===" << std::endl;
}

void TestGELU()
{
    std::cout << "=== Test GELU Layer ===" << std::endl;

    // Parámetros de prueba
    size_t batchSize = 2;
    size_t features = 5;

    // Crear entrada aleatoria
    Tensor input({batchSize, features});
    input.randomize();

    // Crear capa GELU
    GELU gelu;

    // FORWARD
    input.printContents("Input contents");
    Tensor output = gelu.forward(input, true);
    output.printDebugInfo("GELU::forward output");
    output.printContents("Output contents");

    // Verifica que la salida tenga la misma shape que la entrada
    assert(output.dims() == 2);
    assert(output.dim(0) == batchSize);
    assert(output.dim(1) == features);

    // BACKWARD
    Tensor outputGrad({batchSize, features});
    outputGrad.randomize(); // Gradiente simulado desde la siguiente capa

    Tensor inputGrad = gelu.backward(outputGrad);
    inputGrad.printDebugInfo("GELU::backward input gradient");
    outputGrad.printContents("Output gradient contents");
    inputGrad.printContents("Input gradient contents");

    // Verifica que el gradiente tenga la misma shape que la entrada
    assert(inputGrad.dims() == 2);
    assert(inputGrad.dim(0) == batchSize);
    assert(inputGrad.dim(1) == features);

    std::cout << "=== Test GELU completado ===" << std::endl;
}

void TestReLU()
{
    std::cout << "=== Test ReLU Layer ===" << std::endl;

    // Parámetros de prueba
    size_t batchSize = 2;
    size_t features = 5;

    // Crear entrada aleatoria
    Tensor input({batchSize, features});
    input.randomize();

    // Crear capa ReLU
    ReLU relu;

    // FORWARD
    input.printContents("Input contents");
    Tensor output = relu.forward(input, true);
    output.printDebugInfo("ReLU::forward output");
    output.printContents("Output contents");

    // Verifica que la salida tenga la misma shape que la entrada
    assert(output.dims() == 2);
    assert(output.dim(0) == batchSize);
    assert(output.dim(1) == features);

    // BACKWARD
    Tensor outputGrad({batchSize, features});
    outputGrad.randomize(); // Gradiente simulado desde la siguiente capa

    Tensor inputGrad = relu.backward(outputGrad);
    inputGrad.printDebugInfo("ReLU::backward input gradient");
    outputGrad.printContents("Output gradient contents");
    inputGrad.printContents("Input gradient contents");

    // Verifica que el gradiente tenga la misma shape que la entrada
    assert(inputGrad.dims() == 2);
    assert(inputGrad.dim(0) == batchSize);
    assert(inputGrad.dim(1) == features);

    std::cout << "=== Test ReLU completado ===" << std::endl;
}

void TestAdam()
{
    std::cout << "=== Test Adam Optimizer ===" << std::endl;

    // Parámetros simulados (por ejemplo, pesos de una capa)
    size_t batchSize = 2;
    size_t features = 5;
    Tensor param({batchSize, features});
    param.randomize();

    // Gradientes simulados
    Tensor grad({batchSize, features});
    grad.randomize();

    // Almacenar copia original para comparación
    Tensor param_before = param;

    // Crear instancia del optimizador Adam
    Adam adam(0.01f); // learning rate pequeño para observar cambios graduales

    // Wrap en vector como espera Adam
    std::vector<Tensor *> parameters = {&param};
    std::vector<Tensor *> gradients = {&grad};

    std::cout << "--- Before update ---" << std::endl;
    param.printContents("param");

    // Ejecutar varias actualizaciones
    for (int i = 0; i < 5; ++i)
    {
        adam.update(parameters, gradients);
        std::cout << "--- After step " << i + 1 << " ---" << std::endl;
        param.printContents("param");
    }

    // Verificaciones
    assert(param.dims() == param_before.dims());
    assert(param.dim(0) == param_before.dim(0));
    assert(param.dim(1) == param_before.dim(1));

    std::cout << "=== Test Adam completado ===" << std::endl;
}

void TestCrossEntropyAndSoftmax()
{
    std::cout << "=== Test CrossEntropy + Softmax ===" << std::endl;

    const size_t batchSize = 2;
    const size_t numClasses = 3;

    // Logits simulados
    Tensor logits({batchSize, numClasses});
    logits.randomize(-1.0f, 1.0f); // Para tener valores diversos

    // Etiquetas one-hot simuladas (ej: clase 0 para primera fila, clase 2 para segunda)
    Tensor yTrue({batchSize, numClasses});

    // Manual one-hot
    std::vector<int> classes = {0, 2}; // O generar con rand()

    // Crear temporal en host
    std::vector<float> host_yTrue(batchSize * numClasses, 0.0f);
    for (size_t i = 0; i < batchSize; ++i)
        host_yTrue[i * numClasses + classes[i]] = 1.0f;
    yTrue.copyFromHost(host_yTrue.data());

    logits.printContents("logits");
    yTrue.printContents("yTrue (one-hot)");

    // 1. Softmax
    Tensor probs = softmax(logits);
    probs.printContents("softmax(probs)");

    // 2. Calcular pérdida
    CrossEntropy ceLoss;
    float loss = ceLoss.calculate(logits, yTrue);
    std::cout << "CrossEntropy Loss: " << loss << std::endl;

    // 3. Backward
    Tensor grad = ceLoss.backward(logits, yTrue);
    grad.printContents("gradient (softmax - yTrue)");

    // 4. Verificaciones básicas
    assert(probs.dim(0) == batchSize);
    assert(probs.dim(1) == numClasses);
    assert(grad.dim(0) == batchSize);
    assert(grad.dim(1) == numClasses);

    std::cout << "=== Test CrossEntropy + Softmax completado ===" << std::endl;
}

int main()
{
    cudaDeviceSynchronize();
    // TestTensor();
    // TestDense();
    // TestGELU();
    // TestReLU();
    // TestAdam();
    TestCrossEntropyAndSoftmax();
    return 0;
}
