# NMAKE Makefile per backprop (Windows, CUDA Toolkit + MSVC)
# Lanciare da "x64 Native Tools Command Prompt for VS" cosi' cl.exe e nvcc
# risultano entrambi nel PATH.

CC = cl
CC_FLAGS = /O2 /nologo /EHsc

NVCC = nvcc
NVCC_FLAGS = -I..\diversity

# GTX 1650 Ti -> Turing -> sm_75
CUDA_FLAGS = -gencode arch=compute_75,code=sm_75

!IFDEF dbg
NVCC_FLAGS = $(NVCC_FLAGS) -g -G -O0
!ELSE
NVCC_FLAGS = $(NVCC_FLAGS) -O3 -std=c++17 --source-in-ptx -lineinfo
!ENDIF

all: backprop.exe

backprop.exe: backprop.obj facetrain.obj imagenet.obj backprop_cuda.obj
	$(NVCC) backprop.obj facetrain.obj imagenet.obj backprop_cuda.obj -o backprop.exe

facetrain.obj: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) /c facetrain.c /Fofacetrain.obj

backprop.obj: backprop.c backprop.h
	$(CC) $(CC_FLAGS) /c backprop.c /Fobackprop.obj

imagenet.obj: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) /c imagenet.c /Foimagenet.obj

backprop_cuda.obj: backprop_cuda.cu backprop.h
	$(NVCC) $(NVCC_FLAGS) $(CUDA_FLAGS) -c backprop_cuda.cu -o backprop_cuda.obj

clean:
	del /Q *.obj backprop.exe *.exp *.lib 2>NUL