NVCC = nvcc

# GTX 1650 Ti -> Turing -> sm_75
CUDA_FLAGS = -gencode arch=compute_75,code=sm_75

NVCC_FLAGS = -O3 -std=c++17 --source-in-ptx -lineinfo $(CUDA_FLAGS)


all: srad.exe

srad.exe: main.obj
	$(NVCC) main.obj -o srad.exe

main.obj: main.cu define.c graphics.c extract_kernel.cu prepare_kernel.cu reduce_kernel.cu srad_kernel.cu srad2_kernel.cu compress_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -c main.cu -o main.obj

clean:
	del /Q *.obj srad.exe 2>NUL