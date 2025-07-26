#include "Visualizer.hpp"
#include "utils/ModelUtils.hpp"
#include <iostream>

using namespace std;

int main() {
    try {
        // Configuración específica para los pesos que tenemos
        ViTConfig config;
        config.image_size = 28;
        config.patch_size = 7;
        config.in_channels = 1;
        config.num_classes = 10;
        config.embedding_dim = 196; // 128
        config.num_heads = 4;       // 4 cabezas
        config.num_layers = 2;      // 2 bloques encoder (no 4)
        config.mlp_hidden_dim = 784; // 128 * 4 = 512

        Visualizador vis(640, 480, config); // Pasar configuración
        if(!vis.inicializar()) {
            cout << "Error al inicializar visualizador" << endl;
            return 1;
        }
        
        // Cargar pesos preentrenados
        vis.cargarPesos("../vit_fashion_mnist.weights.30.ep");
        cout << "Pesos cargados correctamente!" << endl;
        vis.ejecutar();
    } 
    catch(const exception& e) {
        cout << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}