@echo off
rem Accuracy run: compila e lancia tutte le varianti; i CSV vengono scritti (in append) nella cartella corrente.
nvcc -std=c++17 -I../../../Common -Xcompiler "/Zc:preprocessor" reduction.cu reduction_kernel.cu -o reduction.exe
if errorlevel 1 exit /b 1

reduction.exe "--type=float"
if errorlevel 1 echo ERRORE con type=float
reduction.exe "--type=double"
if errorlevel 1 echo ERRORE con type=double
reduction.exe "--type=Encapsulator<float>"
if errorlevel 1 echo ERRORE con type=Encapsulator^<float^>
reduction.exe "--type=Encapsulator<double>"
if errorlevel 1 echo ERRORE con type=Encapsulator^<double^>
reduction.exe "--type=S2_16_10"
if errorlevel 1 echo ERRORE con type=S2_16_10
reduction.exe "--type=S2_16_11"
if errorlevel 1 echo ERRORE con type=S2_16_11
reduction.exe "--type=S2_32_23"
if errorlevel 1 echo ERRORE con type=S2_32_23
reduction.exe "--type=S3_16_10"
if errorlevel 1 echo ERRORE con type=S3_16_10
reduction.exe "--type=S3_32_21"
if errorlevel 1 echo ERRORE con type=S3_32_21
