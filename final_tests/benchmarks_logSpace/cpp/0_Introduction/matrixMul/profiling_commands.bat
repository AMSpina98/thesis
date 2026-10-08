@echo off
if not exist "..\..\results\matrixMul" mkdir "..\..\results\matrixMul"
if not exist "..\..\results\matrixMul\reports" mkdir "..\..\results\matrixMul\reports"
if not exist "..\..\results\matrixMul\csv" mkdir "..\..\results\matrixMul\csv"

nvcc -I../../../Common matrixMul.cu
if errorlevel 1 exit /b 1

(
ncu -o ../../results/matrixMul/reports/mat_mul_float --set full a.exe -testNum=0
ncu -o ../../results/matrixMul/reports/mat_mul_double --set full a.exe -testNum=1
ncu -o ../../results/matrixMul/reports/mat_mul_encfp --set full a.exe -testNum=2
ncu -o ../../results/matrixMul/reports/mat_mul_encdb --set full a.exe -testNum=3
ncu -o ../../results/matrixMul/reports/mat_mul_s2_16_10 --set full a.exe -testNum=4
ncu -o ../../results/matrixMul/reports/mat_mul_s2_16_11 --set full a.exe -testNum=5
ncu -o ../../results/matrixMul/reports/mat_mul_s2_32_23 --set full a.exe -testNum=6
ncu -o ../../results/matrixMul/reports/mat_mul_s3_16_10 --set full a.exe -testNum=7
ncu -o ../../results/matrixMul/reports/mat_mul_s3_32_21 --set full a.exe -testNum=8

ncu --import ../../results/matrixMul/reports/mat_mul_float.ncu-rep --csv --page raw > ../../results/matrixMul/csv/mat_mul_float.csv
ncu --import ../../results/matrixMul/reports/mat_mul_double.ncu-rep --csv --page raw > ../../results/matrixMul/csv/mat_mul_double.csv
ncu --import ../../results/matrixMul/reports/mat_mul_encfp.ncu-rep --csv --page raw > ../../results/matrixMul/csv/mat_mul_encfp.csv
ncu --import ../../results/matrixMul/reports/mat_mul_encdb.ncu-rep --csv --page raw > ../../results/matrixMul/csv/mat_mul_encdb.csv
ncu --import ../../results/matrixMul/reports/mat_mul_s2_16_10.ncu-rep --csv --page raw > ../../results/matrixMul/csv/mat_mul_s2_16_10.csv
ncu --import ../../results/matrixMul/reports/mat_mul_s2_16_11.ncu-rep --csv --page raw > ../../results/matrixMul/csv/mat_mul_s2_16_11.csv
ncu --import ../../results/matrixMul/reports/mat_mul_s2_32_23.ncu-rep --csv --page raw > ../../results/matrixMul/csv/mat_mul_s2_32_23.csv
ncu --import ../../results/matrixMul/reports/mat_mul_s3_16_10.ncu-rep --csv --page raw > ../../results/matrixMul/csv/mat_mul_s3_16_10.csv
ncu --import ../../results/matrixMul/reports/mat_mul_s3_32_21.ncu-rep --csv --page raw > ../../results/matrixMul/csv/mat_mul_s3_32_21.csv
)