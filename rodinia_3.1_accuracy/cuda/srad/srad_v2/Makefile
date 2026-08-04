NVCC = nvcc

# GTX 1650 Ti -> Turing -> sm_75
CUDA_FLAGS = -gencode arch=compute_75,code=sm_75

NVCC_FLAGS = -O3 -std=c++17 $(CUDA_FLAGS)

SRC = srad.cu
EXE = srad.exe


all: $(EXE)

$(EXE): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(EXE)

debug: $(SRC)
	$(NVCC) -g -G -std=c++17 $(CUDA_FLAGS) $(SRC) -o $(EXE)

clean:
	del /Q $(EXE) result.txt 2>NUL