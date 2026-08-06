# NMAKE Makefile per B+Tree
# Windows + CUDA Toolkit + MSVC
#
# Da eseguire da:
# "x64 Native Tools Command Prompt for VS"
#
# così cl.exe e nvcc sono disponibili nel PATH.


# ============================================================
# COMPILERS
# ============================================================

C_C = cl
CUD_C = nvcc


# ============================================================
# CUDA
# ============================================================

# Directory contenente NL_type.cuh / diversity
CUDA_INC = -I..\..\cuda\diversity

# GTX 1650 Ti -> Turing -> compute capability 7.5
CUDA_FLAG = -gencode arch=compute_75,code=sm_75


# ============================================================
# OPTIMIZATION
# ============================================================

C_FLAGS = /O2
CUDA_FLAGS = -O3 -std=c++17 --source-in-ptx -lineinfo


# ============================================================
# TARGET
# ============================================================

all: btree.exe


# ============================================================
# EXECUTABLE
# ============================================================

btree.exe: main.obj kernel_gpu_cuda_wrapper.obj kernel_gpu_cuda_wrapper_2.obj timer.obj num.obj cuda.obj
	$(CUD_C) main.obj kernel_gpu_cuda_wrapper.obj kernel_gpu_cuda_wrapper_2.obj timer.obj num.obj cuda.obj -o btree.exe


# ============================================================
# MAIN
# ============================================================

main.obj: main.c common.h main.h
	$(C_C) /O2 /c main.c /Fomain.obj


# ============================================================
# KERNELS
# ============================================================

kernel_gpu_cuda_wrapper.obj: kernel\kernel_gpu_cuda_wrapper.cu common.h kernel\kernel_gpu_cuda_wrapper.h
	$(CUD_C) $(CUDA_FLAGS) $(CUDA_FLAG) $(CUDA_INC) -c kernel\kernel_gpu_cuda_wrapper.cu -o kernel_gpu_cuda_wrapper.obj


kernel_gpu_cuda_wrapper_2.obj: kernel\kernel_gpu_cuda_wrapper_2.cu common.h kernel\kernel_gpu_cuda_wrapper_2.h
	$(CUD_C) $(CUDA_FLAGS) $(CUDA_FLAG) $(CUDA_INC) -c kernel\kernel_gpu_cuda_wrapper_2.cu -o kernel_gpu_cuda_wrapper_2.obj



# ============================================================
# UTILITIES
# ============================================================

timer.obj: util\timer\timer.c common.h util\timer\timer.h
	$(C_C) /O2 /c util\timer\timer.c /Fotimer.obj


num.obj: util\num\num.c common.h util\num\num.h
	$(C_C) /O2 /c util\num\num.c /Fonum.obj


cuda.obj: util\cuda\cuda.cu common.h util\cuda\cuda.h
	$(CUD_C) $(CUDA_FLAGS) $(CUDA_FLAG) $(CUDA_INC) -c util\cuda\cuda.cu -o cuda.obj


# ============================================================
# CLEAN
# ============================================================

clean:
	del /Q *.obj btree.exe *.exp *.lib 2>NUL