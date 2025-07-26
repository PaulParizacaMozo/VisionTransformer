#ifndef VISUALIZADOR_H
#define VISUALIZADOR_H

#include <opencv2/opencv.hpp>
#include "core/Tensor.hpp"
#include "model/VisionTransformer.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "utils/ModelUtils.hpp"


class Visualizador {
public:
    Visualizador(int ancho=640, int alto=480, const ViTConfig &cfg = ViTConfig());
    ~Visualizador();
    
    bool inicializar();
    void ejecutar();
    Tensor capturarImagen();
    Tensor predecir(const Tensor& entrada);

    VisionTransformer modelo;

    void cargarPesos(const std::string &ruta) {
      ModelUtils::load_weights(modelo, ruta);
    }

private:
    GLFWwindow* ventana;
    int ancho_ventana;
    int alto_ventana;
    cv::VideoCapture camara;
    GLuint textura_id;
    
    void cargarTextura(const cv::Mat& imagen);
    static void callbackTeclado(GLFWwindow* ventana, int tecla, int scancode, int accion, int mods);
    
    std::string ultima_prediccion; // Para almacenar el resultado
};

#endif