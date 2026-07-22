#include "./compare_utils.h"
#include <string>
#include <fstream>
#include <cmath>
using namespace std;

void compute_int_error(BTreeErrorData& error, long original, long redundant, int index){
    long diff = labs(original - redundant);

    // errore relativo: evitiamo divisione per zero quando original == 0
    double denom = (original != 0) ? fabs((double)original) : 1.0;
    double rel_diff = (double)diff / denom;

    error.avg_rel_diff += rel_diff;
    error.RMSE += rel_diff * rel_diff;

    if(rel_diff > error.max_rel_diff)
    {
        error.max_rel_diff = rel_diff;
        error.max_rel_diff_index = index;
    }

    if(diff != 0)
    {
        error.mismatch_count++;

        if(error.first_error_index == -1)
            error.first_error_index = index;

        if(diff > error.max_abs_diff)
        {
            error.max_abs_diff = diff;
            error.max_diff_index = index;
        }
    }
}

void finalize_int_error(BTreeErrorData& error, int count)
{
    if (count <= 0) return;
    error.avg_rel_diff /= count;
    error.RMSE = sqrt(error.RMSE / count);
}

void write_btree_error_row(std::ofstream& file, string name, const BTreeErrorData& e){
    file << name << ","
         << e.mismatch_count << ","
         << e.first_error_index << ","
         << e.max_abs_diff << ","
         << e.max_diff_index << ","
         << e.max_rel_diff << ","
         << e.max_rel_diff_index << ","
         << e.avg_rel_diff << ","
         << e.RMSE
         << "\n";
}

void save_btree_results(const BTreeResultData& result, string filename){
    std::ofstream file(filename);

    if(!file.is_open())
    {
        printf("ERROR: cannot open %s\n", filename.c_str());
        return;
    }
    file << "field,mismatch_count,"
            "first_error,max_diff,max_diff_index,"
            "max_rel_diff,max_rel_diff_index,avg_rel_diff,RMSE\n";
    write_btree_error_row(file,
                          "ans",
                          result.ans);
    write_btree_error_row(file,
                          "offset",
                          result.offset);
    write_btree_error_row(file,
                          "currKnode",
                          result.currKnode);
    file << "\nTiming\n";
    file << "original_kernel_ms,"
         << result.original_time_ms
         << "\n";

    file << "redundant_kernel_ms,"
         << result.redundant_time_ms
         << "\n";

    file << "overhead_percent,"
         << result.overhead_percent
         << "\n";
    file.close();
}
void compare_btree(
        long* offset,
        long* o_offset,
        long* currKnode,
        long* o_currKnode,
        record* ans,
        record* o_ans,
        int count,
        double original_time,
        double redundant_time,
        string filename)
{

    BTreeResultData result;
    for(int i=0;i<count;i++)
    {
        compute_int_error(
            result.offset,
            offset[i],
            o_offset[i],
            i);
        compute_int_error(
            result.currKnode,
            currKnode[i],
            o_currKnode[i],
            i);
        compute_int_error(
            result.ans,
            ans[i].value,
            o_ans[i].value,
            i);
    }

    finalize_int_error(result.offset, count);
    finalize_int_error(result.currKnode, count);
    finalize_int_error(result.ans, count);

    result.original_time_ms = original_time;
    result.redundant_time_ms = redundant_time;
    result.overhead_percent =
        ((redundant_time-original_time)
        /original_time)*100.0;
    save_btree_results(result, filename);
}
void save_btree_results_2(const BTreeResultData_2& result, string filename){
    std::ofstream file(filename);
    if(!file.is_open())
    {
        printf("ERROR: cannot open %s\n", filename.c_str());
        return;
    }
    file << "field,mismatch_count,"
            "first_error,max_diff,max_diff_index,"
            "max_rel_diff,max_rel_diff_index,avg_rel_diff,RMSE\n";
    write_btree_error_row(file,
                          "Recstart",
                          result.recstart);
    write_btree_error_row(file,
                          "Reclength",
                          result.reclength);
    write_btree_error_row(file,
                          "offset",
                          result.offset);
    write_btree_error_row(file,
                          "offset_2",
                          result.offset_2);
    write_btree_error_row(file,
                          "currKnode",
                          result.currKnode);
    write_btree_error_row(file,
                          "lastKnode",
                          result.lastKnode);
    file << "\nTiming\n";
    file << "original_kernel_ms,"
         << result.original_time_ms
         << "\n";
    file << "redundant_kernel_ms,"
         << result.redundant_time_ms
         << "\n";
    file << "overhead_percent,"
         << result.overhead_percent
         << "\n";
    file.close();
}
void compare_btree_2(
        long* offset,
        long* o_offset,
        long* offset_2,
        long* o_offset_2,
        long* currKnode,
        long* o_currKnode,
        long* lastKnode,
        long* o_lastKnode,
        int* recstart,
        int* o_recstart,
        int* reclength,
        int* o_reclength,
        int count,
        double original_time,
        double redundant_time,
        string filename)
{
    BTreeResultData_2 result;
    for(int i=0;i<count;i++)
    {
        compute_int_error(
            result.offset,
            offset[i],
            o_offset[i],
            i);
        compute_int_error(
            result.offset_2,
            offset_2[i],
            o_offset_2[i],
            i);
        compute_int_error(
            result.currKnode,
            currKnode[i],
            o_currKnode[i],
            i);
        compute_int_error(
            result.lastKnode,
            lastKnode[i],
            o_lastKnode[i],
            i);
        compute_int_error(
            result.recstart,
            recstart[i],
            o_recstart[i],
            i);
        compute_int_error(
            result.reclength,
            reclength[i],
            o_reclength[i],
            i);
    }

    finalize_int_error(result.offset, count);
    finalize_int_error(result.offset_2, count);
    finalize_int_error(result.currKnode, count);
    finalize_int_error(result.lastKnode, count);
    finalize_int_error(result.recstart, count);
    finalize_int_error(result.reclength, count);

    result.original_time_ms = original_time;
    result.redundant_time_ms = redundant_time;
    result.overhead_percent =
        ((redundant_time-original_time)
        /original_time)*100.0;
    save_btree_results_2(result, filename);
}
