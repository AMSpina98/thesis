all: euler3d euler3d_double  pre_euler3d  pre_euler3d_double 

GENCODE_FLAGS = -gencode arch=compute_70,code=sm_70 \
                -gencode arch=compute_80,code=sm_80 \
                -gencode arch=compute_90,code=sm_90

euler3d: euler3d.cu
	nvcc $(KERNEL_DIM) -O2 -Xptxas -v $(GENCODE_FLAGS) euler3d.cu -o euler3d

euler3d_double: euler3d_double.cu
	nvcc -Xptxas -v -O3 $(GENCODE_FLAGS) euler3d_double.cu -o euler3d_double

pre_euler3d: pre_euler3d.cu
	nvcc -Xptxas -v -O3 $(GENCODE_FLAGS) pre_euler3d.cu -o pre_euler3d

pre_euler3d_double: pre_euler3d_double.cu
	nvcc -Xptxas -v -O3 $(GENCODE_FLAGS) pre_euler3d_double.cu -o pre_euler3d_double

clean:
	rm -f euler3d euler3d_double pre_euler3d pre_euler3d_double *.linkinfo