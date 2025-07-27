#include "core/Tensor.cuh"
#include "layers/Dense.cuh"
#include "activations/GELU.cuh"
#include "activations/RELU.cuh"
#include "optimizers/Adam.cuh"
#include "layers/PatchEmbedding.cuh"
#include "layers/Embeddings.cuh"
#include "layers/FeedForward.cuh"
#include "layers/LayerNorm.cuh"
#include "layers/MultiHeadAttention.cuh"
#include "losses/CrossEntropy.cuh"
#include "model/TransformerEncoderBlock.cuh"
#include "model/VisionTransformer.cuh"
#include "model/Trainer.cuh"
#include "utils/DataReader.cuh"
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
    Tensor T1({2, 4, 16});
    Tensor T2({2, 1, 16});
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

    // --- Test softmax ---
    Tensor S({2, 3, 4}); // B=2, N=3, D=4
    S.randomize();
    Tensor S_sm = softmax(S, 2); // Softmax sobre última dimensión

    S_sm.printDebugInfo("Softmax(S, axis=2)");
    S.printContents("Input S contents");
    S_sm.printContents("Softmax output contents");

    // --- Test softmax_backward ---
    Tensor Sb({2, 3, 4}); // B=2, N=3, D=4
    Sb.randomize();
    Tensor S_smb = softmax(Sb, 2); // Forward

    Tensor grad_output({2, 3, 4});
    grad_output.randomize(); // Simula el gradiente dL/dS

    Tensor grad_input = softmax_backward(grad_output, S_smb); // dL/dZ

    // Prints
    grad_output.printDebugInfo("Grad output dL/dS");
    grad_output.printContents("Grad output contents");

    S_smb.printDebugInfo("Softmax output");
    S_smb.printContents("Softmax output contents");

    grad_input.printDebugInfo("Grad input dL/dZ");
    grad_input.printContents("Grad input contents");

    // --- Test batchMatrixMultiply ---
    Tensor BMM1({2, 4, 5}); // B = 2, M = 4, K = 5
    Tensor BMM2({2, 5, 6}); // B = 2, K = 5, N = 6
    BMM1.randomize();
    BMM2.randomize();

    Tensor BMM = batchMatrixMultiply(BMM1, BMM2);

    BMM.printDebugInfo("batchMatrixMultiply(BMM1, BMM2)");
    BMM1.printContents("BMM1 contents");
    BMM2.printContents("BMM2 contents");
    BMM.printContents("BMM contents");

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

    outputGrad.printContents("Output gradient contents");
    Tensor inputGrad = dense.backward(outputGrad);
    inputGrad.printDebugInfo("Dense::backward input gradient");
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

void TestPatchEmbedding()
{
    std::cout << "=== Test PatchEmbedding Layer ===" << std::endl;

    // Parámetros de prueba
    size_t batch_size = 2;
    size_t image_height = 8;
    size_t image_width = 8;
    size_t patch_size = 4;
    size_t in_channels = 3;
    size_t embedding_dim = 16;

    // Calcular dimensiones del tensor de entrada
    Tensor input({batch_size, in_channels, image_height, image_width});
    input.randomize(); // Rellenar con valores aleatorios

    // Crear capa PatchEmbedding
    PatchEmbedding patchEmbedding(image_height, image_width, patch_size, in_channels, embedding_dim);

    // FORWARD
    input.printContents("Input contents");
    Tensor outputptahc = patchEmbedding.forward(input, true);
    outputptahc.printContents("Output contents");
    outputptahc.printDebugInfo("PatchEmbedding::forward output");

    // Verificar dimensiones de salida
    size_t num_patches = (image_height / patch_size) * (image_width / patch_size);
    assert(outputptahc.dims() == 3);
    assert(outputptahc.dim(0) == batch_size);
    assert(outputptahc.dim(1) == num_patches);
    assert(outputptahc.dim(2) == embedding_dim);

    // BACKWARD
    Tensor outputGrad({batch_size, num_patches, embedding_dim});
    outputGrad.randomize(); // Gradiente simulado desde la siguiente capa

    outputGrad.printContents("Output gradient contents");
    Tensor inputGrad = patchEmbedding.backward(outputGrad);
    inputGrad.printDebugInfo("PatchEmbedding::backward input gradient");
    inputGrad.printContents("Input gradient contents");

    // Verificar que el gradiente de entrada tenga la misma shape que el input
    assert(inputGrad.dims() == 4);
    assert(inputGrad.dim(0) == batch_size);
    assert(inputGrad.dim(1) == in_channels);
    assert(inputGrad.dim(2) == image_height);
    assert(inputGrad.dim(3) == image_width);

    // Parámetros y gradientes
    auto params = patchEmbedding.getParameters();
    auto grads = patchEmbedding.getGradients();

    assert(params.size() == 2);
    assert(grads.size() == 2);

    params[0]->printDebugInfo("Weights");
    params[1]->printDebugInfo("Bias");
    grads[0]->printDebugInfo("Weight Gradients");
    grads[1]->printDebugInfo("Bias Gradients");

    std::cout << "=== Test PatchEmbedding completado ===" << std::endl;
}

void TestEmbeddings()
{
    std::cout << "=== Test Embeddings Layer ===" << std::endl;

    // --- Parámetros de prueba ---
    size_t batch_size = 2;
    size_t image_height = 8;
    size_t image_width = 8;
    size_t patch_size = 4;
    size_t in_channels = 3;
    size_t embedding_dim = 16;

    // --- Input simulado ---
    Tensor input({batch_size, in_channels, image_height, image_width});
    input.randomize();

    // --- Crear Embeddings ---
    Embeddings embeddings(image_height, image_width, patch_size, in_channels, embedding_dim);

    // --- Forward ---
    input.printContents("Input Image Tensor");
    Tensor output = embeddings.forward(input, true);
    output.printDebugInfo("Embeddings::forward output");
    output.printContents("Embeddings Output");

    size_t num_patches = (image_height / patch_size) * (image_width / patch_size);
    assert(output.dims() == 3); // [B, N+1, D]
    assert(output.dim(0) == batch_size);
    assert(output.dim(1) == num_patches + 1);
    assert(output.dim(2) == embedding_dim);

    // --- Backward ---
    Tensor outGrad({batch_size, num_patches + 1, embedding_dim});
    outGrad.randomize();
    outGrad.printContents("Output Gradient (Simulated)");

    Tensor inputGrad = embeddings.backward(outGrad);
    inputGrad.printDebugInfo("Embeddings::backward output");
    inputGrad.printContents("Input Gradient");

    // Verifica que el gradiente tenga misma forma que el input
    assert(inputGrad.dims() == 4);
    assert(inputGrad.dim(0) == batch_size);
    assert(inputGrad.dim(1) == in_channels);
    assert(inputGrad.dim(2) == image_height);
    assert(inputGrad.dim(3) == image_width);

    // --- Parámetros entrenables ---
    auto params = embeddings.getParameters();
    auto grads = embeddings.getGradients();

    // clsToken + positionalEncoding + parámetros del patcher
    assert(params.size() >= 2);
    assert(grads.size() >= 2);

    params[0]->printDebugInfo("clsToken Weights");
    params[1]->printDebugInfo("PositionalEncoding Weights");

    grads[0]->printDebugInfo("clsToken Gradient");
    grads[1]->printDebugInfo("PositionalEncoding Gradient");

    std::cout << "=== Test Embeddings completado ===" << std::endl;
}

void TestFeedForward()
{
    std::cout << "=== Test FeedForward Layer ===" << std::endl;

    // --- Parámetros de prueba ---
    size_t batch_size = 2;
    size_t num_patches = 4;
    size_t embedding_dim = 16;
    size_t hidden_dim = 64;

    // --- Input simulado ---
    Tensor input({batch_size, num_patches, embedding_dim});
    input.randomize();
    input.printContents("Input Tensor");

    // --- Crear capa FeedForward ---
    FeedForward ff(embedding_dim, hidden_dim);

    // --- Forward ---
    Tensor output = ff.forward(input, true);
    output.printDebugInfo("FeedForward::forward output");
    output.printContents("FeedForward Output");

    // Verifica forma de salida: misma que input excepto último dim => D
    assert(output.dims() == 3);
    assert(output.dim(0) == batch_size);
    assert(output.dim(1) == num_patches);
    assert(output.dim(2) == embedding_dim); // ya que regresa al mismo embedding_dim

    // --- Backward ---
    Tensor outGrad({batch_size, num_patches, embedding_dim});
    outGrad.randomize();
    outGrad.printContents("Output Gradient (Simulated)");

    Tensor inputGrad = ff.backward(outGrad);
    inputGrad.printDebugInfo("FeedForward::backward output");
    inputGrad.printContents("Input Gradient");

    // Verifica que el gradiente tenga misma forma que el input
    assert(inputGrad.dims() == 3);
    assert(inputGrad.dim(0) == batch_size);
    assert(inputGrad.dim(1) == num_patches);
    assert(inputGrad.dim(2) == embedding_dim);

    // --- Parámetros entrenables ---
    auto params = ff.getParameters();
    auto grads = ff.getGradients();

    assert(params.size() == 4); // dense1: W,b | dense2: W,b
    assert(grads.size() == 4);

    params[0]->printDebugInfo("Dense1 Weights");
    params[1]->printDebugInfo("Dense1 Bias");
    params[2]->printDebugInfo("Dense2 Weights");
    params[3]->printDebugInfo("Dense2 Bias");

    grads[0]->printDebugInfo("Dense1 Weight Grad");
    grads[1]->printDebugInfo("Dense1 Bias Grad");
    grads[2]->printDebugInfo("Dense2 Weight Grad");
    grads[3]->printDebugInfo("Dense2 Bias Grad");

    std::cout << "=== Test FeedForward completado ===" << std::endl;
}

void TestLayerNorm()
{
    std::cout << "=== Test LayerNorm Layer ===" << std::endl;

    // --- Parámetros de prueba ---
    size_t batch_size = 2;
    size_t num_patches = 4;
    size_t embedding_dim = 16;

    // --- Input simulado ---
    Tensor input({batch_size, num_patches, embedding_dim});
    input.randomize();
    input.printContents("Input Tensor");

    // --- Crear capa LayerNorm ---
    LayerNorm layerNorm(embedding_dim); // Normaliza sobre última dimensión

    // --- Forward ---
    Tensor output = layerNorm.forward(input, true);
    output.printDebugInfo("LayerNorm::forward output");
    output.printContents("LayerNorm Output");

    // Verifica forma de salida: igual que input
    assert(output.dims() == 3);
    assert(output.dim(0) == batch_size);
    assert(output.dim(1) == num_patches);
    assert(output.dim(2) == embedding_dim);

    // --- Backward ---
    Tensor outGrad({batch_size, num_patches, embedding_dim});
    outGrad.randomize();
    outGrad.printContents("Output Gradient (Simulated)");

    Tensor inputGrad = layerNorm.backward(outGrad);
    inputGrad.printDebugInfo("LayerNorm::backward output");
    inputGrad.printContents("Input Gradient");

    // Verifica que el gradiente tenga misma forma que el input
    assert(inputGrad.dims() == 3);
    assert(inputGrad.dim(0) == batch_size);
    assert(inputGrad.dim(1) == num_patches);
    assert(inputGrad.dim(2) == embedding_dim);

    // --- Parámetros entrenables ---
    auto params = layerNorm.getParameters();
    auto grads = layerNorm.getGradients();

    assert(params.size() == 2); // gamma y beta
    assert(grads.size() == 2);

    params[0]->printDebugInfo("Gamma (Escala)");
    params[1]->printDebugInfo("Beta (Desplazamiento)");

    grads[0]->printDebugInfo("Gamma Gradient");
    grads[1]->printDebugInfo("Beta Gradient");

    std::cout << "=== Test LayerNorm completado ===" << std::endl;
}

void TestMultiHeadAttention()
{
    std::cout << "=== Test MultiHeadAttention Layer ===" << std::endl;

    // --- Parámetros de prueba ---
    size_t batch_size = 2;
    size_t num_patches = 4;
    size_t embedding_dim = 16;
    size_t num_heads = 2;
    // size_t batch_size = 1;
    // size_t num_patches = 2;
    // size_t embedding_dim = 4;
    // size_t num_heads = 2;

    // --- Input simulado ---
    Tensor input({batch_size, num_patches, embedding_dim});
    input.randomize();
    input.printContents("Input Tensor");

    // // --- Crear capa MultiHeadAttention ---
    MultiHeadAttention mha(embedding_dim, num_heads);

    // --- Forward ---
    Tensor output = mha.forward(input, true);
    output.printDebugInfo("MultiHeadAttention::forward output");
    output.printContents("Attention Output");

    // Verifica forma de salida: igual a input (batch, seq_len, embedding_dim)
    assert(output.dims() == 3);
    assert(output.dim(0) == batch_size);
    assert(output.dim(1) == num_patches);
    assert(output.dim(2) == embedding_dim);

    auto params = mha.getParameters();
    auto grads = mha.getGradients();

    assert(params.size() == 8);
    assert(grads.size() == 8);

    params[0]->printContents("Q Projection Weights");
    params[1]->printContents("Q Projection Bias");
    params[2]->printContents("K Projection Weights");
    params[3]->printContents("K Projection Bias");
    params[4]->printContents("V Projection Weights");
    params[5]->printContents("V Projection Bias");
    params[6]->printContents("Output Projection Weights");
    params[7]->printContents("Output Projection Bias");

    grads[0]->printContents("Q Projection Grad Weights");
    grads[1]->printContents("Q Projection Grad Bias");
    grads[2]->printContents("K Projection Grad Weights");
    grads[3]->printContents("K Projection Grad Bias");
    grads[4]->printContents("V Projection Grad Weights");
    grads[5]->printContents("V Projection Grad Bias");
    grads[6]->printContents("Output Projection Grad Weights");
    grads[7]->printContents("Output Projection Grad Bias");

    // --- Backward ---
    Tensor outGrad({batch_size, num_patches, embedding_dim});
    outGrad.randomize();
    outGrad.printContents("Output Gradient (Simulated)");

    Tensor inputGrad = mha.backward(outGrad);
    inputGrad.printDebugInfo("MultiHeadAttention::backward output");
    inputGrad.printContents("Input Gradient");

    // Verifica que el gradiente tenga misma forma que el input
    assert(inputGrad.dims() == 3);
    assert(inputGrad.dim(0) == batch_size);
    assert(inputGrad.dim(1) == num_patches);
    assert(inputGrad.dim(2) == embedding_dim);

    // --- Parámetros entrenables ---
    params = mha.getParameters();
    grads = mha.getGradients();

    assert(params.size() == 8);
    assert(grads.size() == 8);

    params[0]->printContents("Q Projection Weights");
    params[1]->printContents("Q Projection Bias");
    params[2]->printContents("K Projection Weights");
    params[3]->printContents("K Projection Bias");
    params[4]->printContents("V Projection Weights");
    params[5]->printContents("V Projection Bias");
    params[6]->printContents("Output Projection Weights");
    params[7]->printContents("Output Projection Bias");

    grads[0]->printContents("Q Projection Grad Weights");
    grads[1]->printContents("Q Projection Grad Bias");
    grads[2]->printContents("K Projection Grad Weights");
    grads[3]->printContents("K Projection Grad Bias");
    grads[4]->printContents("V Projection Grad Weights");
    grads[5]->printContents("V Projection Grad Bias");
    grads[6]->printContents("Output Projection Grad Weights");
    grads[7]->printContents("Output Projection Grad Bias");

    std::cout << "=== Test MultiHeadAttention completado ===" << std::endl;
}

void TestTransformerEncoderBlock()
{
    std::cout << "=== Test TransformerEncoderBlock ===" << std::endl;

    // --- Configuración ---
    size_t batch_size = 2;
    size_t num_tokens = 4;
    size_t embedding_dim = 16;
    size_t num_heads = 4;
    size_t mlp_hidden_dim = 64;

    // --- Entrada simulada ---
    Tensor input({batch_size, num_tokens, embedding_dim});
    input.randomize();
    input.printContents("Input Tensor");

    // --- Crear bloque ---
    TransformerEncoderBlock encoder(embedding_dim, num_heads, mlp_hidden_dim);

    // --- Forward ---
    Tensor output = encoder.forward(input, true);
    output.printDebugInfo("TransformerEncoderBlock::forward output");
    output.printContents("Encoder Output");

    assert(output.dims() == 3);
    assert(output.dim(0) == batch_size);
    assert(output.dim(1) == num_tokens);
    assert(output.dim(2) == embedding_dim);

    // --- Parámetros y gradientes (antes del backward) ---
    std::vector<std::string> paramNames = {
        // norm1
        "LayerNorm1::gamma", "LayerNorm1::beta",
        // Attention: Q, K, V, Out
        "Q Projection::weights", "Q Projection::bias",
        "K Projection::weights", "K Projection::bias",
        "V Projection::weights", "V Projection::bias",
        "Out Projection::weights", "Out Projection::bias",
        // norm2
        "LayerNorm2::gamma", "LayerNorm2::beta",
        // FeedForward: Dense1 & Dense2
        "FFN::Dense1::weights", "FFN::Dense1::bias",
        "FFN::Dense2::weights", "FFN::Dense2::bias"};

    auto params = encoder.getParameters();
    auto grads = encoder.getGradients();

    std::cout << "[Antes del backward] Total Parámetros: " << params.size() << std::endl;
    for (size_t i = 0; i < params.size(); ++i)
    {
        params[i]->printContents("Param [" + std::to_string(i) + "]: " + paramNames[i]);
        grads[i]->printContents("Grad  [" + std::to_string(i) + "]: " + paramNames[i]);
    }

    // --- Backward ---
    Tensor outGrad({batch_size, num_tokens, embedding_dim});
    outGrad.randomize();
    outGrad.printContents("Output Gradient (Simulated)");

    Tensor inputGrad = encoder.backward(outGrad);
    inputGrad.printDebugInfo("TransformerEncoderBlock::backward output");
    inputGrad.printContents("Input Gradient");

    assert(inputGrad.dims() == 3);
    assert(inputGrad.dim(0) == batch_size);
    assert(inputGrad.dim(1) == num_tokens);
    assert(inputGrad.dim(2) == embedding_dim);

    auto params2 = encoder.getParameters();
    auto grads2 = encoder.getGradients();

    std::cout << "[Después del backward] Total Parámetros: " << params.size() << std::endl;
    for (size_t i = 0; i < params.size(); ++i)
    {
        params2[i]->printContents("Post-Param [" + std::to_string(i) + "]: " + paramNames[i]);
        grads2[i]->printContents("Post-Grad  [" + std::to_string(i) + "]: " + paramNames[i]);
    }

    std::cout << "=== Test TransformerEncoderBlock completado ===" << std::endl;
}

void TestVisionTransformer()
{
    std::cout << "=== Test VisionTransformer ===" << std::endl;

    // --- Configuración ---
    ViTConfig config;
    config.image_size = 28;
    config.patch_size = 7;
    config.in_channels = 1;
    config.embedding_dim = 128;
    config.num_heads = 8;
    config.num_layers = 4;
    config.mlp_hidden_dim = 512;
    config.num_classes = 10;

    size_t batch_size = 10;
    Tensor input({batch_size, config.in_channels, config.image_size, config.image_size});
    input.randomizeNormal(0.0f, 1.0f);
    input.printDebugInfo("Input Tensor");
    input.printContents("Input Image Tensor");

    // --- Crear modelo ---
    VisionTransformer vit(config);

    // --- Forward ---
    Tensor output = vit.forward(input, true);
    output.printDebugInfo("VisionTransformer::forward output");
    output.printContents("Output Logits");

    assert(output.dims() == 2);
    assert(output.dim(0) == batch_size);
    assert(output.dim(1) == config.num_classes);

    // --- Parámetros y gradientes (antes del backward) ---
    auto params = vit.getParameters();
    auto grads = vit.getGradients();

    // std::cout << "[Antes del backward] Total Parámetros: " << params.size() << std::endl;
    // for (size_t i = 0; i < params.size(); ++i)
    // {
    //     params[i]->printContents("Param [" + std::to_string(i) + "]");
    //     grads[i]->printContents("Grad  [" + std::to_string(i) + "]");
    // }

    // --- Simulación de gradiente de salida ---
    Tensor outGrad({batch_size, config.num_classes});
    outGrad.randomizeNormal(0.0f, 1.0f);
    outGrad.printContents("Output Gradient (Simulated)");

    // --- Backward ---
    Tensor inputGrad = vit.backward(outGrad);
    inputGrad.printDebugInfo("VisionTransformer::backward output");
    inputGrad.printContents("Input Gradient");

    assert(inputGrad.dims() == 4);
    assert(inputGrad.dim(0) == batch_size);
    assert(inputGrad.dim(1) == config.in_channels);
    assert(inputGrad.dim(2) == config.image_size);
    assert(inputGrad.dim(3) == config.image_size);

    auto params2 = vit.getParameters();
    auto grads2 = vit.getGradients();

    // std::cout << "[Después del backward] Total Parámetros: " << params2.size() << std::endl;
    // for (size_t i = 0; i < params2.size(); ++i)
    // {
    //     params2[i]->printContents("Post-Param [" + std::to_string(i) + "]");
    //     grads2[i]->printContents("Post-Grad  [" + std::to_string(i) + "]");
    // }

    std::cout << "=== Test VisionTransformer completado ===" << std::endl;
}

void TestLoadCSVData()
{
    std::cout << "=== Test load_csv_data ===" << std::endl;

    const std::string filePath = "../data/mnist_train.csv";
    float sample_fraction = 0.01f; // usar una pequeña fracción para pruebas rápidas

    auto [features, labels] = load_csv_data(filePath, sample_fraction);

    // --- Verificar contenido básico ---
    features.printDebugInfo("Features tensor");
    labels.printDebugInfo("Labels tensor");

    // --- Aserciones básicas ---
    assert(features.dim(0) == labels.dim(0));               // mismo número de muestras
    assert(features.dims() == 4);                           // [N, C, H, W]
    assert(features.dim(1) == 1);                           // canal único (MNIST)
    assert(features.dim(2) == 28 && features.dim(3) == 28); // tamaño de imagen
    assert(labels.dims() == 2);                             // one-hot [N, num_classes]

    size_t num_samples = features.dim(0);
    size_t num_channels = features.dim(1);
    size_t height = features.dim(2);
    size_t width = features.dim(3);
    size_t num_classes = labels.dim(1);

    std::cout << "Loaded " << num_samples << " samples with shape ["
              << num_channels << ", " << height << ", " << width << "] and "
              << num_classes << " classes.\n";

    // --- Imprimir primeras muestras ---
    std::cout << "--- Primeras 3 muestras ---" << std::endl;
    for (size_t i = 0; i < std::min(num_samples, size_t(3)); ++i)
    {
        std::cout << "Sample " << i << ":\n";
        features.printImageAtIndex("features", i);
        labels.printLabelAtIndex("labels", i);
    }

    std::cout << "=== Test load_csv_data completado ===" << std::endl;
}

void TestTrainer()
{
    std::cout << "=== Test Trainer ===" << std::endl;
    // --- 1. Definir Configuraciones ---
    ViTConfig model_config;
    model_config.embedding_dim = 128;
    model_config.num_layers = 1;
    model_config.num_heads = 2;
    model_config.patch_size = 7;
    model_config.mlp_hidden_dim = model_config.embedding_dim * 4;

    TrainerConfig train_config;
    train_config.epochs = 10;
    train_config.batch_size = 256;
    train_config.learning_rate = 0.0001f;

    // --- 2. Cargar Datos ---
    std::cout << "--- Cargando Datos de Fashion MNIST ---" << std::endl;
    auto train_data = load_csv_data("../data/mnist_train.csv", 0.5f);
    auto test_data = load_csv_data("../data/mnist_test.csv", 1.0f);

    // --- 3. Crear Modelo y Entrenador ---
    VisionTransformer model(model_config);
    Trainer trainer(model, train_config);

    // --- 4. Ejecutar el Entrenamiento y la Evaluación ---
    trainer.train(train_data, test_data);

    std::cout << "\n¡Entrenamiento completado!" << std::endl;

    // --- 5. Guardar el Modelo ---
    const std::string weights_path = "vit_fashion_mnist.weights.test";
    std::cout << "\nGuardando pesos del modelo entrenado en: " << weights_path << std::endl;
    // ModelUtils::save_weights(model, weights_path);
    std::cout << "=== Test Trainer completado ===" << std::endl;
}

int main()
{
    cudaDeviceSynchronize();
    //  TestTensor();
    //  TestDense();
    //  TestGELU();
    //  TestReLU();
    //  TestAdam();
    //  TestCrossEntropyAndSoftmax();
    //  TestPatchEmbedding();
    //  TestEmbeddings();
    //  TestFeedForward();
    //  TestLayerNorm();
    //  TestMultiHeadAttention();
    //  TestTransformerEncoderBlock();
    //  TestVisionTransformer();
    // TestLoadCSVData();
    TestTrainer();
    return 0;
}
