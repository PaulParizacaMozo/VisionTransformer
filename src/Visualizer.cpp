#include "Visualizer.hpp"
#include "model/VisionTransformer.hpp"

Visualizador::Visualizador(int ancho,int alto) 
{
    this->ancho_ventana=ancho;
    this->alto_ventana=alto;
    this->ventana=nullptr;
}

Visualizador::~Visualizador() 
{
    if(ventana) glfwDestroyWindow(ventana);
    glfwTerminate();
}

bool Visualizador::inicializarCamara() 
{
    if(!glfwInit()) return false;
    
    ventana = glfwCreateWindow(ancho_ventana,alto_ventana,"VISION TRANSFORMER PODEROSO",NULL,NULL);
    if(!ventana) return false;
    
    glfwMakeContextCurrent(ventana);
    glewInit();
    glfwSetKeyCallback(ventana,callbackTeclado);
    
    camara.open(0);
    return camara.isOpened();
}

Tensor Visualizador::capturarImagen() 
{
    cv::Mat frame;
    camara>>frame;
    
    // Convertir a escala de grises y redimensionar
    cv::cvtColor(frame,frame,cv::COLOR_BGR2GRAY);
    cv::resize(frame,frame,cv::Size(28,28));
    
    // Convertir a tensor (1,28,28,1)
    Tensor imagen({1,28,28,1});
    float* datos = imagen.obtenerDatos();
    
    for(int y = 0; y < 28; y++) {
        for(int x = 0; x < 28; x++) {
            datos[y*28 + x] = frame.at<uchar>(y,x) / 255.0f;
        }
    }
    return imagen;
}

Tensor Visualizador::predecir(const Tensor& entrada) {
    // Instanciar y usar el modelo (asumiendo que existe)
    VisionTransformer modelo;
    // Cargar pesos preentrenados aquí
    return modelo.forward(entrada,false);  // false = modo inferencia
}

void Visualizador::ejecutar() 
{
    while(!glfwWindowShouldClose(ventana)) {
        glClear(GL_COLOR_BUFFER_BIT);
        
        //mostrar camara? Puede ser
        glfwSwapBuffers(ventana);
        glfwPollEvents();
    }
}

void Visualizador::callbackTeclado(GLFWwindow* ventana,int tecla,int scancode,int accion,int mods) 
{
    if(tecla==GLFW_KEY_ESCAPE && accion==GLFW_PRESS)
        glfwSetWindowShouldClose(ventana,GL_TRUE);
    if(tecla==GLFW_KEY_SPACE && accion==GLFW_PRESS) {
        Visualizador* vis = static_cast<Visualizador*>(glfwGetWindowUserPointer(ventana));
        Tensor imagen = vis->capturarImagen();
        Tensor prediccion = vis->predecir(imagen);
        //resultado poderoso, se mostaría
    }
}