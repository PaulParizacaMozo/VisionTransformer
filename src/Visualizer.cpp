// src/Visualizer.cpp
#include "Visualizer.hpp"
#include <deque>
#include <iostream>

// Ajustes para Fashion‑MNIST
constexpr int   TAM_IMAGEN    = 28;
constexpr float MEAN_FASHION  = 0.5f;
constexpr float STD_FASHION   = 0.5f;
constexpr int   SMOOTH_WINDOW = 5;    // ventana temporal
constexpr int   CAM_INDEX     = 0;
constexpr int   CAM_WIDTH     = 640;
constexpr int   CAM_HEIGHT    = 480;

using namespace std;

Visualizador::Visualizador(int ancho,
                           int alto,
                           const ViTConfig &cfg)
    : ventana(nullptr),
      ancho_ventana(ancho),
      alto_ventana(alto),
      modelo(cfg),
      textura_id(0)
{}

Visualizador::~Visualizador() {
    if (ventana)         glfwDestroyWindow(ventana);
    if (camara.isOpened()) camara.release();
    glfwTerminate();
    if (textura_id)      glDeleteTextures(1, &textura_id);
}

bool Visualizador::inicializar() {
    if (!glfwInit()) {
        cerr << "Error al iniciar GLFW\n"; return false;
    }
    ventana = glfwCreateWindow(ancho_ventana, alto_ventana,
                               "Vision Transformer", nullptr, nullptr);
    if (!ventana) {
        cerr << "Error al crear ventana GLFW\n";
        glfwTerminate(); return false;
    }
    glfwMakeContextCurrent(ventana);
    glfwSetKeyCallback(ventana, callbackTeclado);
    glfwSetWindowUserPointer(ventana, this);
    if (glewInit() != GLEW_OK) {
        cerr << "Error al inicializar GLEW\n"; return false;
    }

    camara.open(CAM_INDEX);
    if (!camara.isOpened()) {
        cerr << "No se pudo abrir cámara #" << CAM_INDEX << "\n";
        return false;
    }
    camara.set(cv::CAP_PROP_FRAME_WIDTH,  CAM_WIDTH);
    camara.set(cv::CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT);
    cout << "Cámara iniciada: "
         << camara.get(cv::CAP_PROP_FRAME_WIDTH) << "×"
         << camara.get(cv::CAP_PROP_FRAME_HEIGHT) << "\n";

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &textura_id);
    return true;
}

void Visualizador::cargarTextura(const cv::Mat &imagen) {
    glBindTexture(GL_TEXTURE_2D, textura_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D,
                 0, GL_RGB,
                 imagen.cols, imagen.rows,
                 0, GL_BGR, GL_UNSIGNED_BYTE,
                 imagen.data);
}

Tensor Visualizador::capturarImagen() {
    static deque<cv::Mat> buffer;
    cv::Mat frame;
    camara >> frame;
    if (frame.empty()) throw runtime_error("Frame vacío");

    // Suavizado temporal
    buffer.push_back(frame.clone());
    if (buffer.size() > SMOOTH_WINDOW) buffer.pop_front();
    cv::Mat avg = cv::Mat::zeros(frame.size(), frame.type());
    for (auto &f : buffer) avg += f;
    avg /= static_cast<double>(buffer.size());

    // Escala de grises
    cv::Mat gris;
    cv::cvtColor(avg, gris, cv::COLOR_BGR2GRAY);

    // Crop central (80% del mínimo lado)
    int min_side = min(gris.cols, gris.rows);
    int crop_sz  = static_cast<int>(min_side * 0.8);
    int x0 = (gris.cols - crop_sz) / 2;
    int y0 = (gris.rows - crop_sz) / 2;
    cv::Mat roi = gris(cv::Rect(x0, y0, crop_sz, crop_sz));

    // Resize a 28×28
    cv::Mat resized;
    cv::resize(roi, resized, cv::Size(TAM_IMAGEN, TAM_IMAGEN),
               0, 0, cv::INTER_AREA);

    // A tensor normalizado Fashion‑MNIST
    Tensor tensor({1,1,TAM_IMAGEN,TAM_IMAGEN});
    float* data = tensor.getData();
    for (int y = 0; y < TAM_IMAGEN; ++y) {
        for (int x = 0; x < TAM_IMAGEN; ++x) {
            float v = resized.at<uchar>(y,x) / 255.0f;
            data[y*TAM_IMAGEN + x] = (v - MEAN_FASHION) / STD_FASHION;
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
            int clase = 0;
            float maxv = d[0];
            // Recorre solo 0–9
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
