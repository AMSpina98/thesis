@echo off
if not exist "..\..\results\vectorAdd" mkdir "..\..\results\vectorAdd"
if not exist "..\..\results\vectorAdd\reports" mkdir "..\..\results\vectorAdd\reports"
if not exist "..\..\results\vectorAdd\csv" mkdir "..\..\results\vectorAdd\csv"

nvcc -I../../../Common vectorAdd.cu
if errorlevel 1 exit /b 1

(
ncu -o ../../results/vectorAdd/reports/vec_add_float --set full a.exe -testNum=0
ncu -o ../../results/vectorAdd/reports/vec_add_double --set full a.exe -testNum=1
ncu -o ../../results/vectorAdd/reports/vec_add_encfp --set full a.exe -testNum=2
ncu -o ../../results/vectorAdd/reports/vec_add_encdb --set full a.exe -testNum=3
ncu -o ../../results/vectorAdd/reports/vec_add_s2_16_10 --set full a.exe -testNum=4
ncu -o ../../results/vectorAdd/reports/vec_add_s2_16_11 --set full a.exe -testNum=5
ncu -o ../../results/vectorAdd/reports/vec_add_s2_32_23 --set full a.exe -testNum=6
ncu -o ../../results/vectorAdd/reports/vec_add_s3_16_10 --set full a.exe -testNum=7
ncu -o ../../results/vectorAdd/reports/vec_add_s3_32_21 --set full a.exe -testNum=8

ncu --import ../../results/vectorAdd/reports/vec_add_float.ncu-rep --csv --page raw > ../../results/vectorAdd/csv/vec_add_float.csv
ncu --import ../../results/vectorAdd/reports/vec_add_double.ncu-rep --csv --page raw > ../../results/vectorAdd/csv/vec_add_double.csv
ncu --import ../../results/vectorAdd/reports/vec_add_encfp.ncu-rep --csv --page raw > ../../results/vectorAdd/csv/vec_add_encfp.csv
ncu --import ../../results/vectorAdd/reports/vec_add_encdb.ncu-rep --csv --page raw > ../../results/vectorAdd/csv/vec_add_encdb.csv
ncu --import ../../results/vectorAdd/reports/vec_add_s2_16_10.ncu-rep --csv --page raw > ../../results/vectorAdd/csv/vec_add_s2_16_10.csv
ncu --import ../../results/vectorAdd/reports/vec_add_s2_16_11.ncu-rep --csv --page raw > ../../results/vectorAdd/csv/vec_add_s2_16_11.csv
ncu --import ../../results/vectorAdd/reports/vec_add_s2_32_23.ncu-rep --csv --page raw > ../../results/vectorAdd/csv/vec_add_s2_32_23.csv
ncu --import ../../results/vectorAdd/reports/vec_add_s3_16_10.ncu-rep --csv --page raw > ../../results/vectorAdd/csv/vec_add_s3_16_10.csv
ncu --import ../../results/vectorAdd/reports/vec_add_s3_32_21.ncu-rep --csv --page raw > ../../results/vectorAdd/csv/vec_add_s3_32_21.csv
)