#include "Visualizer.hpp"
#include "utils/ModelUtils.hpp"
#include <iostream>

using namespace std;

int main() {
    try {
        Visualizador vis;
        if(!vis.inicializar()) 
        {
            cout<<"Error bendito al inicializar visualizador"<<endl;
            return 1;
        }
        
        // Cargar pesos preentrenados
        ModelUtils::load_weights(vis.modelo, "vit_mnist.weights.test");
        vis.ejecutar();
    } 
    catch(const exception& excepGOD) 
    {
        cout<<"Error: "<<excepGOD.what()<<endl;
        return 1;
    }
    return 0;
}