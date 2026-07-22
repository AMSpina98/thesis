# Compiler
NVCC = /usr/local/cuda/bin/nvcc

# Flags per Jetson Nano (Maxwell SM 5.3)
ARCH := -gencode arch=compute_53,code=sm_53 \
        -gencode arch=compute_53,code=compute_53


# Optimization flags
NVCC_FLAGS := -O3 -Xptxas -v $(ARCH)
NVCC_FLAGS += -I$/usr/local/cuda/include
NVCC_FLAGS += -I../../cuda/diversity
CXXFLAGS += -O3
NVCC_FLAGS += -lineinfo

# Targets
all: euler3d euler3d_double pre_euler3d pre_euler3d_double

euler3d: euler3d.cu
	$(NVCC) $(NVCC_FLAGS) euler3d.cu -o euler3d

euler3d_double: euler3d_double.cu
	$(NVCC) $(NVCC_FLAGS) euler3d_double.cu -o euler3d_double

pre_euler3d: pre_euler3d.cu
	$(NVCC) $(NVCC_FLAGS) pre_euler3d.cu -o pre_euler3d

pre_euler3d_double: pre_euler3d_double.cu
	$(NVCC) $(NVCC_FLAGS) pre_euler3d_double.cu -o pre_euler3d_double

clean:
	rm -f euler3d euler3d_double pre_euler3d pre_euler3d_double *.linkinfo
