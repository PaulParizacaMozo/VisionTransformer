#ifndef VISUALIZADOR_H
#define VISUALIZADOR_H

#include <opencv2/opencv.hpp>
#include "core/Tensor.hpp"
#include "model/VisionTransformer.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>

class Visualizador {
public:
    Visualizador(int ancho=640, int alto=480);
    ~Visualizador();
    
    bool inicializar();
    void ejecutar();
    Tensor capturarImagen() const;
    Tensor predecir(const Tensor& entrada) const;

private:
    GLFWwindow* ventana;
    int ancho_ventana;
    int alto_ventana;
    cv::VideoCapture camara;
    GLuint textura_id;
    VisionTransformer modelo;
    
    void cargarTextura(const cv::Mat& imagen);
    static void callbackTeclado(GLFWwindow* ventana, int tecla, int scancode, int accion, int mods);
};

#endif