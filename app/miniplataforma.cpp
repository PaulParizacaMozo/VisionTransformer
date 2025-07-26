#include <gtk/gtk.h>
#include "model/Trainer.hpp"
#include "utils/ModelUtils.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>  // Necesario para getOpenFileName
#include <iostream>
#include <string>
#include <stdexcept>


#define STB_IMAGE_IMPLEMENTATION
#include "utils/stb_image.h"  // Incluir STB


// Función para mostrar el diálogo de selección de archivos
std::string mostrar_selector_archivos() {
    gtk_init(NULL, NULL);
    
    GtkWidget *dialog = gtk_file_chooser_dialog_new(
        "Seleccionar imagen",
        NULL,
        GTK_FILE_CHOOSER_ACTION_OPEN,
        "Cancelar", GTK_RESPONSE_CANCEL,
        "Abrir", GTK_RESPONSE_ACCEPT,
        NULL
    );
    
    // Filtro para imágenes
    GtkFileFilter *filter = gtk_file_filter_new();
    gtk_file_filter_set_name(filter, "Imágenes");
    gtk_file_filter_add_pattern(filter, "*.png");
    gtk_file_filter_add_pattern(filter, "*.jpg");
    gtk_file_filter_add_pattern(filter, "*.jpeg");
    gtk_file_filter_add_pattern(filter, "*.bmp");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);
    
    std::string ruta_seleccionada;
    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char *filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        ruta_seleccionada = filename;
        g_free(filename);
    }
    
    gtk_widget_destroy(dialog);
    while (gtk_events_pending()) gtk_main_iteration();
    
    return ruta_seleccionada;
}

Tensor cargar_imagen(const std::string& ruta) {
    int ancho, alto, canales;
    unsigned char* datos = stbi_load(ruta.c_str(), &ancho, &alto, &canales, 1);
    if (!datos) {
        throw std::runtime_error("Error cargando imagen: " + ruta);
    }

    const int tam_objetivo = 28;
    Tensor entrada({1, 1, tam_objetivo, tam_objetivo});

    float escala_x = static_cast<float>(ancho) / tam_objetivo;
    float escala_y = static_cast<float>(alto) / tam_objetivo;

    for (int y = 0; y < tam_objetivo; ++y) {
        for (int x = 0; x < tam_objetivo; ++x) {
            int src_x = static_cast<int>(x * escala_x);
            int src_y = static_cast<int>(y * escala_y);
            int idx = src_y * ancho + src_x;

            float pixel = static_cast<float>(datos[idx]) / 255.0f;
            pixel = (pixel - 0.1307f) / 0.3081f;  // Normalización MNIST

            entrada(0, 0, y, x) = pixel;
        }
    }

    stbi_image_free(datos);
    return entrada;
}

int main() {
    // 1. Cargar modelo preentrenado
    ViTConfig config;
    config.embedding_dim = 196;
    config.num_layers = 2;
    config.num_heads = 4;
    config.patch_size = 7;
    config.mlp_hidden_dim = config.embedding_dim * 4;

    VisionTransformer modelo(config);
    ModelUtils::load_weights(modelo, "../vit_fashion_mnist.weights.30.ep");

    // 2. Crear ventana de OpenCV
    const std::string nombre_ventana = "Vision Transformer - Miniplataforma";
    cv::namedWindow(nombre_ventana, cv::WINDOW_NORMAL);
    cv::resizeWindow(nombre_ventana, 800, 600);

    // 3. Bucle principal
    bool ejecutando = true;
    while (ejecutando) {
        // Mostrar menú principal
        cv::Mat pantalla = cv::Mat::zeros(600, 800, CV_8UC3);
        cv::putText(pantalla, "MINIPLATAFORMA VISION TRANSFORMER", 
                   cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 1.2, 
                   cv::Scalar(0, 255, 0), 2);
        
        cv::putText(pantalla, "Opciones:", 
                   cv::Point(100, 180), cv::FONT_HERSHEY_SIMPLEX, 0.8, 
                   cv::Scalar(255, 255, 255), 1);
        
        cv::putText(pantalla, "1. Seleccionar imagen (S)", 
                   cv::Point(120, 230), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                   cv::Scalar(200, 200, 0), 1);
        
        cv::putText(pantalla, "2. Salir (ESC)", 
                   cv::Point(120, 270), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                   cv::Scalar(200, 200, 0), 1);
        
        cv::imshow(nombre_ventana, pantalla);
        
        // Esperar acción del usuario
        int tecla = cv::waitKey(0);
        
        if (tecla == 27) {  // Tecla ESC
            ejecutando = false;
        }
        else if (tecla == 's' || tecla == 'S') {
            // 4. Mostrar diálogo gráfico para seleccionar archivo
            std::string ruta_archivo = mostrar_selector_archivos();
            
            if (ruta_archivo.empty()) continue;
            
            try {
                // 5. Cargar y preprocesar imagen
                Tensor entrada = cargar_imagen(ruta_archivo);
                
                // 6. Realizar predicción
                Tensor logits = modelo.forward(entrada, false);
                Tensor probabilidades = softmax(logits);
                
                int clase_predicha = -1;
                float max_prob = -1.0f;
                for (int j = 0; j < 10; ++j) {
                    if (probabilidades(0, j) > max_prob) {
                        max_prob = probabilidades(0, j);
                        clase_predicha = j;
                    }
                }
                
                // 7. Mostrar resultados en ventana persistente
                cv::Mat imagen = cv::imread(ruta_archivo);
                if (!imagen.empty()) {
                    cv::resize(imagen, imagen, cv::Size(400, 400));
                    
                    // Crear ventana para resultados
                    const std::string ventana_resultado = "Resultado de Predicción";
                    cv::namedWindow(ventana_resultado, cv::WINDOW_NORMAL);
                    
                    // Mostrar predicción
                    std::string texto_prediccion = "Prediccion: " + std::to_string(clase_predicha);
                    std::string texto_confianza = "Confianza: " + std::to_string(max_prob * 100) + "%";
                    
                    cv::putText(imagen, texto_prediccion, 
                               cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                               cv::Scalar(0, 255, 0), 2);
                    
                    cv::putText(imagen, texto_confianza, 
                               cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                               cv::Scalar(0, 200, 255), 2);
                    
                    // Instrucciones para el usuario
                    cv::putText(imagen, "Presione cualquier tecla para continuar", 
                               cv::Point(20, 380), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                               cv::Scalar(255, 255, 255), 1);
                    
                    cv::imshow(ventana_resultado, imagen);
                    
                    // Esperar hasta que el usuario presione una tecla
                    cv::waitKey(0);
                    cv::destroyWindow(ventana_resultado);
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << std::endl;
            }
        }
    }
    
    cv::destroyAllWindows();
    return 0;
}