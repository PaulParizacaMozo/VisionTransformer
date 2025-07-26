// include/Visualizer.hpp
#ifndef VISUALIZADOR_H
#define VISUALIZADOR_H

#include <algorithm>               // std::min
#include <opencv2/opencv.hpp>
#include "core/Tensor.hpp"
#include "model/VisionTransformer.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "utils/ModelUtils.hpp"

class Visualizador {
public:
    Visualizador(int ancho = 640,
                 int alto  = 480,
                 const ViTConfig &cfg = ViTConfig());
    ~Visualizador();

    bool inicializar();
    void ejecutar();

    /// Carga pesos en el modelo
    void cargarPesos(const std::string &ruta) {
        ModelUtils::load_weights(modelo, ruta);
    }

private:
    GLFWwindow* ventana;
    int ancho_ventana;
    int alto_ventana;
    cv::VideoCapture camara;
    GLuint textura_id;

    VisionTransformer modelo;
    std::string      ultima_prediccion;

    void cargarTextura(const cv::Mat &imagen);
    Tensor capturarImagen();
    Tensor predecir(const Tensor &entrada);

    static void callbackTeclado(GLFWwindow* ventana,
                                int tecla,
                                int scancode,
                                int accion,
                                int mods);
};

#endif // VISUALIZADOR_H
