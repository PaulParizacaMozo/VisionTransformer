#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <string>
#include <fstream>
#include <iomanip>

class Logger {
public:
    Logger(const std::string& filename, bool overwrite = true);
    ~Logger();

    void log_epoch(int epoch, int epochs_total, 
                   float train_loss, float train_acc,
                   float test_loss, float test_acc);

private:
    std::ofstream file;
    bool is_first_entry = true;
};

#endif // LOGGER_HPP