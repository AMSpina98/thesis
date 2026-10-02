@echo off
if not exist "..\..\results\reduction" mkdir "..\..\results\reduction"
if not exist "..\..\results\reduction\reports" mkdir "..\..\results\reduction\reports"
if not exist "..\..\results\reduction\csv" mkdir "..\..\results\reduction\csv"

nvcc -I../../../Common -std=c++17 -Xcompiler "/Zc:preprocessor" reduction.cu reduction_kernel.cu
if errorlevel 1 exit /b 1

(
ncu -o ../../results/reduction/reports/reduction_float --set full --launch-skip 1 --launch-count 1 a.exe -type=float
ncu -o ../../results/reduction/reports/reduction_double --set full --launch-skip 1 --launch-count 1 a.exe -type=double
ncu -o ../../results/reduction/reports/reduction_encfp --set full --launch-skip 1 --launch-count 1 a.exe -type="Encapsulator<float>"
ncu -o ../../results/reduction/reports/reduction_encdb --set full --launch-skip 1 --launch-count 1 a.exe -type="Encapsulator<double>"
ncu -o ../../results/reduction/reports/reduction_s2_16_10 --set full --launch-skip 1 --launch-count 1 a.exe -type=S2_16_10
ncu -o ../../results/reduction/reports/reduction_s2_16_11 --set full --launch-skip 1 --launch-count 1 a.exe -type=S2_16_11
ncu -o ../../results/reduction/reports/reduction_s2_32_21 --set full --launch-skip 1 --launch-count 1 a.exe -type=S2_32_21

ncu --import ../../results/reduction/reports/reduction_float.ncu-rep --csv --page raw > ../../results/reduction/csv/reduction_float.csv
ncu --import ../../results/reduction/reports/reduction_double.ncu-rep --csv --page raw > ../../results/reduction/csv/reduction_double.csv
ncu --import ../../results/reduction/reports/reduction_encfp.ncu-rep --csv --page raw > ../../results/reduction/csv/reduction_encfp.csv
ncu --import ../../results/reduction/reports/reduction_encdb.ncu-rep --csv --page raw > ../../results/reduction/csv/reduction_encdb.csv
ncu --import ../../results/reduction/reports/reduction_s2_16_10.ncu-rep --csv --page raw > ../../results/reduction/csv/reduction_s2_16_10.csv
ncu --import ../../results/reduction/reports/reduction_s2_16_11.ncu-rep --csv --page raw > ../../results/reduction/csv/reduction_s2_16_11.csv
ncu --import ../../results/reduction/reports/reduction_s2_32_21.ncu-rep --csv --page raw > ../../results/reduction/csv/reduction_s2_32_21.csv
)