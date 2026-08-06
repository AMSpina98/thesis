# NMAKE Makefile per CFD (Windows, CUDA Toolkit + MSVC)
# Lanciare da "x64 Native Tools Command Prompt for VS" cosi' cl.exe e nvcc
# risultano entrambi nel PATH.

CUD_C = nvcc

# Optional includes
CUDA_INC = -I..\..\cuda\diversity

# GTX 1650 Ti -> Turing -> sm_75
CUDA_FLAG = -gencode arch=compute_75,code=sm_75

# Optimization + profiling support
OPT = -O3 -std=c++17 --source-in-ptx -lineinfo


all: euler3d.exe euler3d_double.exe pre_euler3d.exe pre_euler3d_double.exe


# ============================================================
# Executables
# ============================================================

euler3d.exe: euler3d.obj
	$(CUD_C) euler3d.obj -o euler3d.exe

euler3d_double.exe: euler3d_double.obj
	$(CUD_C) euler3d_double.obj -o euler3d_double.exe

pre_euler3d.exe: pre_euler3d.obj
	$(CUD_C) pre_euler3d.obj -o pre_euler3d.exe

pre_euler3d_double.exe: pre_euler3d_double.obj
	$(CUD_C) pre_euler3d_double.obj -o pre_euler3d_double.exe


# ============================================================
# Objects
# ============================================================

euler3d.obj: euler3d.cu
	$(CUD_C) $(OPT) -Xptxas -v $(CUDA_FLAG) $(CUDA_INC) -c euler3d.cu -o euler3d.obj

euler3d_double.obj: euler3d_double.cu
	$(CUD_C) $(OPT) -Xptxas -v $(CUDA_FLAG) $(CUDA_INC) -c euler3d_double.cu -o euler3d_double.obj

pre_euler3d.obj: pre_euler3d.cu
	$(CUD_C) $(OPT) -Xptxas -v $(CUDA_FLAG) $(CUDA_INC) -c pre_euler3d.cu -o pre_euler3d.obj

pre_euler3d_double.obj: pre_euler3d_double.cu
	$(CUD_C) $(OPT) -Xptxas -v $(CUDA_FLAG) $(CUDA_INC) -c pre_euler3d_double.cu -o pre_euler3d_double.obj


clean:
	del /Q *.obj euler3d.exe euler3d_double.exe pre_euler3d.exe pre_euler3d_double.exe *.exp *.lib 2>NUL