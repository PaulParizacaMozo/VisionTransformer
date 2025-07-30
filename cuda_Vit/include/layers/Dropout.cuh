#pragma once

#include "layers/Layer.cuh"

/**
 * @class Dropout
 * @brief Implementa la capa de regularización Dropout.
 *
 * Durante el entrenamiento, pone a cero aleatoriamente una fracción de los elementos
 * de entrada con probabilidad 'rate'. Para compensar, los elementos restantes se
 * escalan por un factor de 1.0 / (1.0 - rate). Esta técnica se conoce como
 * "inverted dropout".
 *
 * Durante la inferencia (isTraining = false), la capa no realiza ninguna operación
 * y simplemente devuelve la entrada sin modificar.
 */
class Dropout : public Layer
{
public:
    /**
     * @brief Constructor de la capa Dropout.
     * @param rate La probabilidad de que un elemento sea puesto a cero (ej. 0.1 para 10%).
     *             Debe estar en el rango [0, 1).
     */
    explicit Dropout(float rate = 0.5f);

    /**
     * @brief Realiza el paso hacia adelante.
     * @param input El tensor de entrada.
     * @param isTraining Si es true, aplica la máscara de dropout. Si es false, no hace nada.
     * @return El tensor de salida, posiblemente con algunos elementos a cero.
     * @override
     */
    Tensor forward(const Tensor &input, bool isTraining) override;

    /**
     * @brief Realiza el paso hacia atrás.
     * @details Aplica la máscara guardada del forward pass al gradiente de salida.
     * @param outputGradient El gradiente que fluye desde la capa siguiente.
     * @return El gradiente con respecto a la entrada de esta capa.
     * @override
     */
    Tensor backward(const Tensor &outputGradient) override;

    /**
     * @brief Devuelve el nombre de la capa.
     * @override
     */
    std::string getName() const override { return "Dropout"; }

    // Dropout es una capa sin parámetros entrenables, por lo que getParameters()
    // y getGradients() no necesitan ser sobreescritos (devuelven vacío por defecto).

private:
    float rate;  // Probabilidad de apagar una neurona
    float scale; // Factor de escala para las neuronas restantes: 1 / (1 - rate)

    // Máscara binaria (en realidad, de floats 0.0 o 'scale') que se genera en el
    // forward pass y se reutiliza en el backward pass.
    Tensor mask;
};
