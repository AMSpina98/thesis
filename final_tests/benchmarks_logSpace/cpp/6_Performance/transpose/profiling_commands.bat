@echo off
if not exist "..\..\results\transpose" mkdir "..\..\results\transpose"
if not exist "..\..\results\transpose\reports" mkdir "..\..\results\transpose\reports"
if not exist "..\..\results\transpose\csv" mkdir "..\..\results\transpose\csv"

nvcc -I../../../Common -std=c++17 -Xcompiler "/Zc:preprocessor" transpose.cu
if errorlevel 1 exit /b 1

(
ncu -o ../../results/transpose/reports/transpose_float_naive --set full --kernel-name transposeNaive --launch-skip 1 --launch-count 1 a.exe -testNum=0
ncu -o ../../results/transpose/reports/transpose_float_opt --set full --kernel-name transposeNoBankConflicts --launch-skip 1 --launch-count 1 a.exe -testNum=0
ncu -o ../../results/transpose/reports/transpose_double_naive --set full --kernel-name transposeNaive --launch-skip 1 --launch-count 1 a.exe -testNum=1
ncu -o ../../results/transpose/reports/transpose_double_opt --set full --kernel-name transposeNoBankConflicts --launch-skip 1 --launch-count 1 a.exe -testNum=1
ncu -o ../../results/transpose/reports/transpose_encfp_naive --set full --kernel-name transposeNaive --launch-skip 1 --launch-count 1 a.exe -testNum=2
ncu -o ../../results/transpose/reports/transpose_encfp_opt --set full --kernel-name transposeNoBankConflicts --launch-skip 1 --launch-count 1 a.exe -testNum=2
ncu -o ../../results/transpose/reports/transpose_encdb_naive --set full --kernel-name transposeNaive --launch-skip 1 --launch-count 1 a.exe -testNum=3
ncu -o ../../results/transpose/reports/transpose_encdb_opt --set full --kernel-name transposeNoBankConflicts --launch-skip 1 --launch-count 1 a.exe -testNum=3
ncu -o ../../results/transpose/reports/transpose_s2_16_10_naive --set full --kernel-name transposeNaive --launch-skip 1 --launch-count 1 a.exe -testNum=4
ncu -o ../../results/transpose/reports/transpose_s2_16_10_opt --set full --kernel-name transposeNoBankConflicts --launch-skip 1 --launch-count 1 a.exe -testNum=4
ncu -o ../../results/transpose/reports/transpose_s2_16_11_naive --set full --kernel-name transposeNaive --launch-skip 1 --launch-count 1 a.exe -testNum=5
ncu -o ../../results/transpose/reports/transpose_s2_16_11_opt --set full --kernel-name transposeNoBankConflicts --launch-skip 1 --launch-count 1 a.exe -testNum=5
ncu -o ../../results/transpose/reports/transpose_s2_32_23_naive --set full --kernel-name transposeNaive --launch-skip 1 --launch-count 1 a.exe -testNum=6
ncu -o ../../results/transpose/reports/transpose_s2_32_23_opt --set full --kernel-name transposeNoBankConflicts --launch-skip 1 --launch-count 1 a.exe -testNum=6

ncu --import ../../results/transpose/reports/transpose_float_naive.ncu-rep --csv --page raw > ../../results/transpose/csv/transpose_float_naive.csv
ncu --import ../../results/transpose/reports/transpose_float_opt.ncu-rep --csv --page raw > ../../results/transpose/csv/transpose_float_opt.csv
ncu --import ../../results/transpose/reports/transpose_double_naive.ncu-rep --csv --page raw > ../../results/transpose/csv/transpose_double_naive.csv
ncu --import ../../results/transpose/reports/transpose_double_opt.ncu-rep --csv --page raw > ../../results/transpose/csv/transpose_double_opt.csv
ncu --import ../../results/transpose/reports/transpose_encfp_naive.ncu-rep --csv --page raw > ../../results/transpose/csv/transpose_encfp_naive.csv
ncu --import ../../results/transpose/reports/transpose_encfp_opt.ncu-rep --csv --page raw > ../../results/transpose/csv/transpose_encfp_opt.csv
ncu --import ../../results/transpose/reports/transpose_encdb_naive.ncu-rep --csv --page raw > ../../results/transpose/csv/transpose_encdb_naive.csv
ncu --import ../../results/transpose/reports/transpose_encdb_opt.ncu-rep --csv --page raw > ../../results/transpose/csv/transpose_encdb_opt.csv
ncu --import ../../results/transpose/reports/transpose_s2_16_10_naive.ncu-rep --csv --page raw > ../../results/transpose/csv/transpose_s2_16_10_naive.csv
ncu --import ../../results/transpose/reports/transpose_s2_16_10_opt.ncu-rep --csv --page raw > ../../results/transpose/csv/transpose_s2_16_10_opt.csv
ncu --import ../../results/transpose/reports/transpose_s2_16_11_naive.ncu-rep --csv --page raw > ../../results/transpose/csv/transpose_s2_16_11_naive.csv
ncu --import ../../results/transpose/reports/transpose_s2_16_11_opt.ncu-rep --csv --page raw > ../../results/transpose/csv/transpose_s2_16_11_opt.csv
ncu --import ../../results/transpose/reports/transpose_s2_32_23_naive.ncu-rep --csv --page raw > ../../results/transpose/csv/transpose_s2_32_23_naive.csv
ncu --import ../../results/transpose/reports/transpose_s2_32_23_opt.ncu-rep --csv --page raw > ../../results/transpose/csv/transpose_s2_32_23_opt.csv
)