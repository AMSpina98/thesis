#include <string>
#include <fstream>
#include "../common.h"	
using namespace std;

struct BTreeErrorData {
    int mismatch_count = 0;
    int first_error_index = -1;
    int max_abs_diff = 0;
    int max_diff_index = -1;
    double max_rel_diff = 0.0;
    int max_rel_diff_index = -1;
    double avg_rel_diff = 0.0;
    double RMSE = 0.0;
};

struct BTreeResultData {

    BTreeErrorData ans;
    BTreeErrorData offset;
    BTreeErrorData currKnode;

    double original_time_ms = 0.0;
    double redundant_time_ms = 0.0;
    double overhead_percent = 0.0;
};

struct BTreeResultData_2 {

    BTreeErrorData recstart;
    BTreeErrorData reclength;

    BTreeErrorData offset;
    BTreeErrorData offset_2;
    BTreeErrorData currKnode;
    BTreeErrorData lastKnode;

    double original_time_ms = 0.0;
    double redundant_time_ms = 0.0;
    double overhead_percent = 0.0;
};

void compute_int_error(BTreeErrorData& error, long original, long redundant, int index);

void finalize_int_error(BTreeErrorData& error, int count);

void write_btree_error_row(std::ofstream& file, string name, const BTreeErrorData& e);


void save_btree_results(const BTreeResultData& result, string filename);


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
        string filename);

        
void save_btree_results_2(const BTreeResultData_2& result, string filename);


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

        string filename);
