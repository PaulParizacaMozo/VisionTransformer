#include "Visualizer.hpp"
#include <iostream>
#include <stdexcept>

// Configuración compatible con MNIST (28x28,1 canal)
constexpr int TAM_IMAGEN=28;
constexpr float MEDIA=0.1307f;
constexpr float DESV_EST=0.3081f;

using namespace std;

Visualizador::Visualizador(int ancho,int alto) 
    : ventana(nullptr),ancho_ventana(ancho),alto_ventana(alto),textura_id(0) 
    {}

Visualizador::~Visualizador() 
{
    if(ventana) glfwDestroyWindow(ventana);
    glfwTerminate();
    if(textura_id) glDeleteTextures(1,&textura_id);
}

bool Visualizador::inicializar() 
{
    // 1. Inicializar GLFW
    if(!glfwInit()) 
    {
        cout<<"Error malvado al empezar con GLFW"<<endl;
        return false;
    }
    
    // 2. Crear ventana
    ventana=glfwCreateWindow(ancho_ventana,alto_ventana,"Vision Transformer",NULL,NULL);
    if(!ventana) 
    {
        cout<<"Error malvado al crear ventana GLFW"<<endl;
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(ventana);
    glfwSetKeyCallback(ventana,callbackTeclado);
    
    // 3. Inicializar GLEW
    if(glewInit() != GLEW_OK) 
    {
        cout<<"Error malvado al inicializar GLEW"<<endl;
        return false;
    }
    
    // 4. Inicializar cámara
    camara.open(0);
    if(!camara.isOpened()) 
    {
        cout<<"Error malvado al abrir cámara"<<endl;
        return false;
    }
    
    // 5. Configurar OpenGL
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1,&textura_id);
    
    return true;
}

void Visualizador::cargarTextura(const cv::Mat& imagen) 
{
    glBindTexture(GL_TEXTURE_2D,textura_id);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,imagen.cols,imagen.rows,0,GL_BGR,GL_UNSIGNED_BYTE,imagen.data);
}

Tensor Visualizador::capturarImagen() const {
    cv::Mat frame;
    camara>>frame;
    
    // Preprocesamiento idéntico al dataset MNIST
    cv::Mat frame_gris;
    cv::cvtColor(frame,frame_gris,cv::COLOR_BGR2GRAY);
    cv::resize(frame_gris,frame_gris,cv::Size(TAM_IMAGEN,TAM_IMAGEN));
    
    // Convertir a tensor (1,1,28,28)
    Tensor imagen({1,1,TAM_IMAGEN,TAM_IMAGEN});
    float* datos=imagen.getData();
    
    for(int y=0; y < TAM_IMAGEN; y++) 
    {
        for(int x=0; x < TAM_IMAGEN; x++) 
        {
            float valor=frame_gris.at<uchar>(y,x) / 255.0f;
            datos[y * TAM_IMAGEN + x]=(valor - MEDIA) / DESV_EST;
        }
    }
    return imagen;
}

Tensor Visualizador::predecir(const Tensor& entrada) const {
    return modelo.predict(entrada);
}

void Visualizador::ejecutar() 
{
    while(!glfwWindowShouldClose(ventana)) 
    {
        // Capturar frame
        cv::Mat frame;
        camara>>frame;
        if(frame.empty()) 
        {
            continue;
        }
        
        // Voltear horizontalmente (efecto espejo)
        cv::flip(frame,frame,1);
        
        // Actualizar textura
        cargarTextura(frame);
        
        // Renderizar
        glClear(GL_COLOR_BUFFER_BIT);
        
        glBindTexture(GL_TEXTURE_2D,textura_id);
        glBegin(GL_QUADS);
            glTexCoord2f(0,1); glVertex2f(-1,-1);
            glTexCoord2f(1,1); glVertex2f(1,-1);
            glTexCoord2f(1,0); glVertex2f(1,1);
            glTexCoord2f(0,0); glVertex2f(-1,1);
        glEnd();
        
        glfwSwapBuffers(ventana);
        glfwPollEvents();
    }
}

void Visualizador::callbackTeclado(GLFWwindow* ventana,int tecla,int scancode,int accion,int mods) 
{
    if(tecla == GLFW_KEY_ESCAPE && accion == GLFW_PRESS) 
    {
        glfwSetWindowShouldClose(ventana,GL_TRUE);
    }
}