#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <cstdio>
#include <memory>
#include <array>
#include <string>

namespace fs = std::filesystem;

const std::string FRAMES_DIR = "frames";
const std::string MODEL_EXECUTABLE = "build/testLabel";
const int CAPTURE_INTERVAL_MS = 1000; // 5 segundos

std::string last_prediction = "Prediccion: - (0%)";

void ensure_frames_directory() {
    if (!fs::exists(FRAMES_DIR)) {
        fs::create_directory(FRAMES_DIR);
    } else if (!fs::is_directory(FRAMES_DIR)) {
        throw std::runtime_error(FRAMES_DIR + " existe pero no es un directorio");
    }
}

std::string generate_filename() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    ss << "_" << std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()).count() % 1000;
    
    return FRAMES_DIR + "/frame_" + ss.str() + ".png";
}

std::string execute_model(const std::string& image_path) {
    std::array<char, 128> buffer;
    std::string result;
    std::string command = MODEL_EXECUTABLE + " " + image_path;
    
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("Error al ejecutar el comando");
    }
    
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    
    return result;
}

void process_prediction_output(const std::string& prediction) {
    std::istringstream iss(prediction);
    std::string line;
    std::vector<std::string> relevant_lines;
    
    // Extraer solo las líneas relevantes (ignorar logs iniciales)
    while (std::getline(iss, line)) {
        if (!line.empty()) {
            // Solo considerar líneas que contienen números (predicción o confianza)
            if (!line.empty() && (isdigit(line[0]) || line[0] == '-')) {
                relevant_lines.push_back(line);
            }
        }
    }
    
    if (relevant_lines.size() >= 2) {
        std::string class_pred = relevant_lines[0];
        // Formatear confianza a 2 decimales
        float confidence = std::stof(relevant_lines[1]);
        std::stringstream confidence_ss;
        confidence_ss << std::fixed << std::setprecision(2) << confidence;
        
        last_prediction = "Prediccion: " + class_pred + " (" + confidence_ss.str() + "%)";
    }
}

void process_frame(cv::Mat frame) {
    // Convertir y guardar imagen
    cv::Mat gray, resized;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, resized, cv::Size(28, 28));
    std::string image_path = generate_filename();
    cv::imwrite(image_path, resized);
    
    // Ejecutar modelo y procesar salida
    std::string prediction = execute_model(image_path);
    process_prediction_output(prediction);
    
    // Log en consola
    std::cout << "\n--- Predicción ---\n";
    std::cout << last_prediction << "\n";
    std::cout << "Imagen: " << image_path << "\n";
}

void draw_prediction(cv::Mat& frame) {
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.8;
    int thickness = 2;
    cv::Scalar color(0, 255, 0); // Verde
    
    // Tamaño y posición del texto
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(last_prediction, font, font_scale, thickness, &baseline);
    cv::Point text_org(frame.cols - text_size.width - 20, 30);
    
    // Fondo semitransparente
    cv::rectangle(frame, 
                 cv::Point(frame.cols - text_size.width - 30, 10),
                 cv::Point(frame.cols - 10, 40),
                 cv::Scalar(0, 0, 0), 
                 cv::FILLED);
    
    // Dibujar texto
    cv::putText(frame, last_prediction, text_org, font, font_scale, color, thickness);
}

int main() {
    try {
        ensure_frames_directory();
        cv::VideoCapture cap(0);
        
        if (!cap.isOpened()) {
            throw std::runtime_error("No se pudo abrir la cámara");
        }

        cv::namedWindow("RealTime Capture", cv::WINDOW_AUTOSIZE);
        auto last_capture = std::chrono::steady_clock::now();
        
        std::cout << "Sistema iniciado. Presione ESC para salir...\n";

        while (true) {
            cv::Mat frame;
            cap >> frame;
            
            if (frame.empty()) {
                throw std::runtime_error("Frame vacío recibido");
            }

            // Procesar frame cada CAPTURE_INTERVAL_MS
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_capture).count() >= CAPTURE_INTERVAL_MS) {
                process_frame(frame.clone());
                last_capture = now;
            }

            draw_prediction(frame);
            cv::imshow("RealTime Capture", frame);

            if (cv::waitKey(30) == 27) break;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}