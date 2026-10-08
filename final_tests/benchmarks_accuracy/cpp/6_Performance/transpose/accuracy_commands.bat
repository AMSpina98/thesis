@echo off
rem Accuracy run: compila e lancia tutte le varianti; i CSV vengono scritti (in append) nella cartella corrente.
nvcc -std=c++17 -I../../../Common transpose.cu -o a.exe
if errorlevel 1 exit /b 1

a.exe -testNum=0
if errorlevel 1 echo ERRORE in testNum=0 ^(float^)
a.exe -testNum=1
if errorlevel 1 echo ERRORE in testNum=1 ^(double^)
a.exe -testNum=2
if errorlevel 1 echo ERRORE in testNum=2 ^(EncFp^)
a.exe -testNum=3
if errorlevel 1 echo ERRORE in testNum=3 ^(EncDb^)
a.exe -testNum=4
if errorlevel 1 echo ERRORE in testNum=4 ^(S2_16_10^)
a.exe -testNum=5
if errorlevel 1 echo ERRORE in testNum=5 ^(S2_16_11^)
a.exe -testNum=6
if errorlevel 1 echo ERRORE in testNum=6 ^(S2_32_23^)
a.exe -testNum=7
if errorlevel 1 echo ERRORE in testNum=7 ^(S3_16_10^)
a.exe -testNum=8
if errorlevel 1 echo ERRORE in testNum=8 ^(S3_32_21^)
