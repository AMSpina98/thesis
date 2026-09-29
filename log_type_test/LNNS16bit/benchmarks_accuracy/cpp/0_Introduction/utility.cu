#include <fstream>
#include <filesystem>
#include <string>
#include <iostream>

void saveResultsCSV(
    const std::string& filename,
    const std::string& format,
    int elements,
    float threshold,
    float maxAbsError,
    float avgAbsError,
    float RMSE,
    float maxRelError,
    float avgRelError,
    int count)
{
    std::ifstream checkFile(filename);
    bool writeHeader = !checkFile.good();
    checkFile.close();

    std::ofstream file(filename, std::ios::app);

    if (!file.is_open()) {
        std::cerr << "Error opening " << filename << "\n";
        return;
    }

    if (writeHeader) {
        file << "Format,"
             << "Elements,"
             << "Threshold,"
             << "MaxAbsError,"
             << "AvgAbsError,"
             << "RMSE,"
             << "MaxRelError,"
             << "AvgRelError,"
             << "Count\n";
    }

    file << format << ","
         << elements << ","
         << threshold << ","
         << maxAbsError << ","
         << avgAbsError << ","
         << RMSE << ","
         << maxRelError << ","
         << avgRelError << ","
         << count << "\n";

    file.close();
}


void saveTimingCSV(
    const std::string& filename,
    const std::string& format,
    int elements,
    float timeLNNS,
    float timeFloat)
{
    std::ifstream checkFile(filename);
    bool writeHeader = !checkFile.good();
    checkFile.close();

    std::ofstream file(filename, std::ios::app);

    if (!file.is_open()) {
        std::cerr << "Error opening " << filename << "\n";
        return;
    }

    if (writeHeader) {
        file << "Format,"
             << "Elements,"
             << "TimeLNNS_ms,"
             << "TimeFloat_ms,"
             << "Overhead_percent\n";
    }

    float overhead = (timeLNNS - timeFloat) / timeFloat * 100.0f;

    file << format << ","
         << elements << ","
         << timeLNNS << ","
         << timeFloat << ","
         << overhead << "\n";

    file.close();
}