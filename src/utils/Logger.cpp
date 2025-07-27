#include "utils/Logger.hpp"

Logger::Logger(const std::string& filename, bool overwrite) {
    file.open(filename, overwrite ? (std::ios::out | std::ios::trunc) : std::ios::app);
    
    if (!file.is_open()) {
        throw std::runtime_error("Error al abrir: " + filename);
    }

    if (overwrite) {
        file << "epoch,epochs_total,train_loss,train_acc,test_loss,test_acc\n";
    }
}

Logger::~Logger() {
    if (file.is_open()) {
        file.close();
    }
}

void Logger::log_epoch(int epoch, int epochs_total, 
                      float train_loss, float train_acc,
                      float test_loss, float test_acc) {
    file << std::fixed << std::setprecision(4)
         << epoch + 1 << ","
         << epochs_total << ","
         << train_loss << ","
         << train_acc << ","
         << test_loss << ","
         << test_acc << "\n";
    
    file.flush();
}