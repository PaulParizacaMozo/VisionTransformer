#include "Visualizer.hpp"
#include <deque>
#include <numeric>
#include <iostream>
#include <chrono>
#include <thread>

// Configuración ajustable
constexpr int   TAM_IMAGEN = 28;
constexpr float MEAN       = 0.5f;      // Fashion-MNIST usa 0.5
constexpr float STD_DEV    = 0.5f;      // Fashion-MNIST usa 0.5
constexpr int   SMOOTH_WINDOW = 5;      // Para promediar frames
constexpr int   CAM_INDEX = 0;          // Índice de cámara
constexpr int   CAM_WIDTH = 640;         // Resolución deseada
constexpr int   CAM_HEIGHT = 480;

using namespace std;
using namespace std::chrono;

Visualizador::Visualizador(int ancho, int alto, const ViTConfig &cfg)
    : ventana(nullptr),
      ancho_ventana(ancho),
      alto_ventana(alto),
      modelo(cfg),
      textura_id(0)
{
    // No inicializar cámara aquí, se hace en inicializar()
}

Visualizador::~Visualizador() {
    if (ventana) glfwDestroyWindow(ventana);
    if (camara.isOpened()) camara.release();
    glfwTerminate();
    if (textura_id) glDeleteTextures(1, &textura_id);
}

bool Visualizador::inicializar() {
    if (!glfwInit()) {
        cout << "Error al iniciar GLFW\n";
        return false;
    }
    
    ventana = glfwCreateWindow(ancho_ventana, alto_ventana, "Vision Transformer", nullptr, nullptr);
    if (!ventana) {
        cout << "Error al crear ventana GLFW\n";
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(ventana);
    glfwSetKeyCallback(ventana, callbackTeclado);
    glfwSetWindowUserPointer(ventana, this);

    if (glewInit() != GLEW_OK) {
        cout << "Error al inicializar GLEW\n";
        return false;
    }

    // Inicializar cámara con resolución específica
    camara.open(CAM_INDEX, cv::CAP_ANY);
    if (!camara.isOpened()) {
        cout << "Error al abrir cámara en índice " << CAM_INDEX << "\n";
        
        // Listar cámaras disponibles
        cout << "Probando cámaras disponibles...\n";
        for (int i = 0; i < 5; i++) {
            cv::VideoCapture test_cam(i);
            if (test_cam.isOpened()) {
                cout << "-> Cámara encontrada en índice " << i << "\n";
                test_cam.release();
            }
        }
        return false;
    }

    // Configurar resolución
    camara.set(cv::CAP_PROP_FRAME_WIDTH, CAM_WIDTH);
    camara.set(cv::CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT);
    
    // Verificar configuración real
    double actual_width = camara.get(cv::CAP_PROP_FRAME_WIDTH);
    double actual_height = camara.get(cv::CAP_PROP_FRAME_HEIGHT);
    cout << "Resolución cámara: " << actual_width << "x" << actual_height << "\n";

    // Configurar OpenGL
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &textura_id);
    
    return true;
}

void Visualizador::cargarTextura(const cv::Mat &imagen) {
    glBindTexture(GL_TEXTURE_2D, textura_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imagen.cols, imagen.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, imagen.data);
}

Tensor Visualizador::capturarImagen() {
    static deque<cv::Mat> buffer;
    cv::Mat frame;
    camara >> frame;
    if (frame.empty()) throw runtime_error("Cámara devolvió frame vacío");

    buffer.push_back(frame.clone());
    if (buffer.size() > SMOOTH_WINDOW) buffer.pop_front();

    cv::Mat avg = cv::Mat::zeros(frame.size(), frame.type());
    for (const auto& f : buffer) avg += f;
    avg.convertTo(avg, -1, 1.0/buffer.size());

    // 1. Convertir a escala de grises - IGUAL QUE testImage.cpp
    cv::Mat gris;
    cv::cvtColor(avg, gris, cv::COLOR_BGR2GRAY);

    cv::Mat equalized;
    cv::equalizeHist(gris, equalized);
    gris = equalized;

    // 2. Recorte central - MANTENEMOS PARA ENFOCAR EL DÍGITO
    int center_x = gris.cols / 2;
    int center_y = gris.rows / 2;
    int crop_size = std::min(gris.cols, gris.rows) * 0.8; // 80% del área
    cv::Rect roi(
        center_x - crop_size/2,
        center_y - crop_size/2,
        crop_size,
        crop_size
    );
    cv::Mat cropped = gris(roi);

    // 3. Redimensionar a 28x28 - EXACTAMENTE COMO testImage.cpp
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(28, 28), 0, 0, cv::INTER_AREA);

    // 4. Normalización MNIST - IGUAL QUE testImage.cpp
    Tensor tensor({1, 1, 28, 28});
    float* data = tensor.getData();
    
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            float pixel = resized.at<uchar>(y, x) / 255.0f;
            data[y * 28 + x] = (pixel - 0.1307f) / 0.3081f;
        }
    }
    
    return tensor;
}

Tensor Visualizador::procesarImagen(cv::Mat& img) {
    Tensor tensor({1, 1, 28, 28});
    float* data = tensor.getData();
    const float MEAN = 0.5f;    // Fashion-MNIST
    const float STD = 0.5f;
    
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            float pixel = img.at<uchar>(y, x) / 255.0f;
            data[y*28 + x] = (pixel - MEAN) / STD;
        }
    }
    return tensor;
}

Tensor Visualizador::predecir(const Tensor &entrada) {
    return modelo.forward(entrada, /*isTraining=*/false);
}

void Visualizador::ejecutar() {
    while (!glfwWindowShouldClose(ventana)) {
        cv::Mat frame;
        camara >> frame;
        if (frame.empty()) continue;
        cv::flip(frame, frame, 1);

        cargarTextura(frame);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindTexture(GL_TEXTURE_2D, textura_id);
        glBegin(GL_QUADS);
            glTexCoord2f(0,1); glVertex2f(-1,-1);
            glTexCoord2f(1,1); glVertex2f( 1,-1);
            glTexCoord2f(1,0); glVertex2f( 1, 1);
            glTexCoord2f(0,0); glVertex2f(-1, 1);
        glEnd();

        glfwSwapBuffers(ventana);
        glfwPollEvents();
    }
}

void Visualizador::callbackTeclado(GLFWwindow* win,
                                  int tecla,
                                  int scancode,
                                  int accion,
                                  int mods)
{
    auto* self = static_cast<Visualizador*>(glfwGetWindowUserPointer(win));

    if (tecla == GLFW_KEY_ESCAPE && accion == GLFW_PRESS) {
        glfwSetWindowShouldClose(win, GL_TRUE);
    }
    if (tecla == GLFW_KEY_SPACE && accion == GLFW_PRESS) {
        try {
            Tensor img    = self->capturarImagen();
            Tensor logits = self->predecir(img);
            const float* d = logits.getData();
            int clase     = 0;
            float maxv    = d[0];
            // Recorre 10 clases (0–9)
            for (int i = 1; i < 10; ++i) {
                if (d[i] > maxv) { maxv = d[i]; clase = i; }
            }
            string pred = "Prediccion: " + to_string(clase);
            if (pred != self->ultima_prediccion) {
                self->ultima_prediccion = pred;
                cout << ">>> " << pred << " <<<\n";
                glfwSetWindowTitle(win,
                    ("Vision Transformer - " + pred).c_str());
            }
        } catch (const exception &e) {
            cerr << "Error predicción: " << e.what() << endl;
        }
    }
}
