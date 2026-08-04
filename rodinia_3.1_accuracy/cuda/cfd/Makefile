# NMAKE Makefile per B+Tree (Windows, CUDA Toolkit + MSVC)
# Lanciare da "x64 Native Tools Command Prompt for VS"
# in modo che cl.exe e nvcc siano nel PATH.

CUD_C = nvcc
C_C   = cl

# GTX 1650 Ti -> Turing -> sm_75
CUDA_FLAG = -gencode arch=compute_75,code=sm_75

# Include per le librerie/diversity personalizzate
CUDA_INC = -I..\..\cuda\diversity

# Ottimizzazione
C_OPT    = /O2
CUDA_OPT = -O3 -std=c++17


# ============================================================
# Executable
# ============================================================

all: b+tree.exe


b+tree.exe: main.obj kernel_gpu_cuda_wrapper.obj kernel_gpu_cuda_wrapper_2.obj compare_utils.obj timer.obj num.obj cuda.obj
	$(CUD_C) main.obj kernel_gpu_cuda_wrapper.obj kernel_gpu_cuda_wrapper_2.obj compare_utils.obj timer.obj num.obj cuda.obj -o b+tree.exe


# ============================================================
# Main
# ============================================================

main.obj: main.c common.h main.h
	$(C_C) $(C_OPT) /c main.c /Fo$@


# ============================================================
# CUDA kernels
# ============================================================

kernel_gpu_cuda_wrapper.obj: kernel\kernel_gpu_cuda_wrapper.cu common.h kernel\kernel_gpu_cuda_wrapper.h
	$(CUD_C) $(CUDA_OPT) $(CUDA_FLAG) $(CUDA_INC) -c kernel\kernel_gpu_cuda_wrapper.cu -o kernel_gpu_cuda_wrapper.obj


kernel_gpu_cuda_wrapper_2.obj: kernel\kernel_gpu_cuda_wrapper_2.cu common.h kernel\kernel_gpu_cuda_wrapper_2.h
	$(CUD_C) $(CUDA_OPT) $(CUDA_FLAG) $(CUDA_INC) -c kernel\kernel_gpu_cuda_wrapper_2.cu -o kernel_gpu_cuda_wrapper_2.obj


compare_utils.obj: kernel\compare_utils.cu common.h kernel\compare_utils.h
	$(CUD_C) $(CUDA_OPT) $(CUDA_FLAG) $(CUDA_INC) -c kernel\compare_utils.cu -o compare_utils.obj


# ============================================================
# Utilities
# ============================================================

timer.obj: util\timer\timer.c common.h util\timer\timer.h
	$(C_C) $(C_OPT) /c util\timer\timer.c /Fo$@


num.obj: util\num\num.c common.h util\num\num.h
	$(C_C) $(C_OPT) /c util\num\num.c /Fo$@


cuda.obj: util\cuda\cuda.cu common.h util\cuda\cuda.h
	$(CUD_C) $(CUDA_OPT) $(CUDA_FLAG) $(CUDA_INC) -c util\cuda\cuda.cu -o cuda.obj


# ============================================================
# Clean
# ============================================================

clean:
	del /Q *.obj b+tree.exe *.exp *.lib 2>NUL
	del /Q kernel\*.obj 2>NUL
	del /Q util\timer\*.obj 2>NUL
	del /Q util\num\*.obj 2>NUL
	del /Q util\cuda\*.obj 2>NUL
	del /Q kernel\btree_results*.csv 2>NUL
	del /Q output.txt 2>NUL