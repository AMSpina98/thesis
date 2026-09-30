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


void saveTimingStatsCSV(
    const std::string& filename,
    const std::string& format,
    int elements,
    float meanLNNS, float medianLNNS, float stddevLNNS, float p95LNNS,
    float meanFloat, float medianFloat, float stddevFloat, float p95Float)
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
             << "MeanLNNS_ms,"
             << "MedianLNNS_ms,"
             << "StdDevLNNS_ms,"
             << "P95LNNS_ms,"
             << "MeanFloat_ms,"
             << "MedianFloat_ms,"
             << "StdDevFloat_ms,"
             << "P95Float_ms,"
             << "OverheadMean_percent,"
             << "OverheadMedian_percent\n";
    }

    float overheadMean   = (meanFloat   != 0.0f) ? (meanLNNS   - meanFloat)   / meanFloat   * 100.0f : 0.0f;
    float overheadMedian = (medianFloat != 0.0f) ? (medianLNNS - medianFloat) / medianFloat * 100.0f : 0.0f;

    file << format << ","
         << elements << ","
         << meanLNNS << ","
         << medianLNNS << ","
         << stddevLNNS << ","
         << p95LNNS << ","
         << meanFloat << ","
         << medianFloat << ","
         << stddevFloat << ","
         << p95Float << ","
         << overheadMean << ","
         << overheadMedian << "\n";

    file.close();
}