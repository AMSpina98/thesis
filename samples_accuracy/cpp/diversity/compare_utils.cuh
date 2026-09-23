#pragma once
#include <string>
#include <fstream>
#include <cmath>
using namespace std;

template <typename T>
struct ArrayErrorData {
    int mismatch_count = 0;
    int first_error_index = -1;
    T max_abs_diff = 0;
    int max_diff_index = -1;
    double max_rel_diff = 0.0;
    int max_rel_diff_index = -1;
    double avg_rel_diff = 0.0;
    double rel_RMSE = 0.0;
    double abs_RMSE = 0.0;
};

template <typename T>
struct ValueErrorData {
    T abs_diff = 0;
    double rel_diff = 0.0;
};

template <typename T>
void compare_array(T* array, T* o_array, int count, ArrayErrorData<T>& error_data){
    

    for (int i = 0; i < count; i++) {
        if (array[i] != o_array[i]) {
            error_data.mismatch_count++;
            if (error_data.first_error_index == -1) {
                error_data.first_error_index = i;
            }
            T abs_diff = std::abs(array[i] - o_array[i]);
            if (abs_diff > error_data.max_abs_diff) {
                error_data.max_abs_diff = abs_diff;
                error_data.max_diff_index = i;
            }
            double rel_diff = std::abs(static_cast<double>(array[i] - o_array[i]));
            error_data.abs_RMSE += rel_diff * rel_diff;
            double denominator = std::abs(static_cast<double>(array[i]));
            if(denominator > 0) rel_diff /= denominator;
            if (rel_diff > error_data.max_rel_diff) {
                error_data.max_rel_diff = rel_diff;
                error_data.max_rel_diff_index = i;
            }
            error_data.avg_rel_diff += rel_diff;
            error_data.rel_RMSE += rel_diff * rel_diff;
        }
    }
    error_data.avg_rel_diff /= count;
    error_data.rel_RMSE = std::sqrt(error_data.rel_RMSE / count);
    error_data.abs_RMSE = std::sqrt(error_data.abs_RMSE / count);
}
template <typename T>
void compare_value(T value, T o_value, int count, ValueErrorData<T>& error_data){
    error_data.abs_diff = std::abs(value - o_value);
    double denominator = std::abs(static_cast<double>(value));
    if(denominator > 0)
        error_data.rel_diff = error_data.abs_diff / denominator;
    else
        error_data.rel_diff = error_data.abs_diff;
}

template <typename T>
void write_array_to_file(const std::string& filename, const std::string& array_name,  T* array, T* o_array, int count){
    ArrayErrorData<T> error_data;
    compare_array(array, o_array, count, error_data);
    std::ifstream check(filename);
    bool exists = check.good();
    check.close();
    std::ofstream file(filename, std::ios::app);
    if (exists) {
        file << "\n\n";
    }
    file << array_name << "\n";
    file << "mismatch_count,first_error_index,max_abs_diff,max_diff_index,max_rel_diff,max_rel_diff_index\n";
    file << error_data.mismatch_count << ","
         << error_data.first_error_index << ","
         << error_data.max_abs_diff << ","
         << error_data.max_diff_index << ","
         << error_data.max_rel_diff << ","
         << error_data.max_rel_diff_index << "\n";
}

template <typename T>
void write_value_to_file(const std::string& filename, const std::string& value_name, T value, T o_value){
    ValueErrorData<T> error_data;
    compare_value(value, o_value, 1, error_data);
    std::ifstream check(filename);
    bool exists = check.good();
    check.close();
    std::ofstream file(filename, std::ios::app);
    if (exists) {
        file << "\n\n";
    }
    file << value_name << "\n";
    file << "abs_diff,rel_diff\n";
    file << error_data.abs_diff << ","
         << error_data.rel_diff << "\n";
}

void write_time_to_file(const std::string& filename, const std::string& kernel_name, float time, float o_time){
    std::ifstream check(filename);
    bool exists = check.good();
    check.close();
    std::ofstream file(filename, std::ios::app);
    if (exists) {
        file << "\n\n";
    }
    file << kernel_name << " time results\n";
    file << "time,o_time\n";
    file << time << "," << o_time << "\n";
}

template <typename T>
void frobenius_norms(T* matrix, int count, double& frobenius){
    frobenius = 0.0;
    for (int i = 0; i < count; i++) {
        frobenius += static_cast<double>(matrix[i]) * static_cast<double>(matrix[i]);
    }
    frobenius = std::sqrt(frobenius);
}

template <typename T>
void frobenius_error(T* matrix, T* o_matrix, int count, double& error){
    error = 0.0;
    for (int i = 0; i < count; i++) {
        double diff = static_cast<double>(matrix[i]) - static_cast<double>(o_matrix[i]);
        error += diff * diff;
    }
    error = std::sqrt(error);
}

template <typename T>
void write_matrix_to_file(const std::string& filename, const std::string& matrix_name,  T* matrix, T* o_matrix, int count){
    ArrayErrorData<T> error_data;
    compare_array(matrix, o_matrix, count, error_data);

    double frobenius = 0.0;
    frobenius_norms(matrix, count, frobenius);

    double o_frobenius = 0.0;
    frobenius_norms(o_matrix, count, o_frobenius);

    double frobenius_err = 0.0;
    frobenius_error(matrix, o_matrix, count, frobenius_err);


    std::ifstream check(filename);
    bool exists = check.good();
    check.close();
    std::ofstream file(filename, std::ios::app);
    if (exists) {
        file << "\n\n";
    }
    file << matrix_name << "\n";
    file << "mismatch_count,first_error_index,max_abs_diff,max_diff_index,max_rel_diff,max_rel_diff_index,frobenius_error,frobenius,o_frobenius\n";
    file << error_data.mismatch_count << ","
         << error_data.first_error_index << ","
         << error_data.max_abs_diff << ","
         << error_data.max_diff_index << ","
         << error_data.max_rel_diff << ","
         << error_data.max_rel_diff_index << ","
         << frobenius_err << ","
         << frobenius << ","
         << o_frobenius << "\n";
}