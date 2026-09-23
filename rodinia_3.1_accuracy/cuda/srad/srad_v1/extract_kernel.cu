// statistical kernel
#include "../../diversity/diversity.cuh"
#include "../../diversity/converters.cuh"

__global__ void extract(	long d_Ne, fp *d_I){										// pointer to input image (DEVICE GLOBAL MEMORY)

	// indexes
	int bx = blockIdx.x;													// get current horizontal block index
	int tx = threadIdx.x;													// get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;						// unique thread id, more threads than actual elements !!!
	
	// copy input to output & log uncompress
	if(ei<d_Ne){															// do only for the number of elements, omit extra threads

		//d_I[ei] = exp(d_I[ei]/255);												// exponentiate input IMAGE and copy to output image
		d_I[ei] = exp(dDiv(d_I[ei], 255));												// exponentiate input IMAGE and copy to output image

	}

}

__global__ void o_extract(	long d_Ne,
											fp *d_I){										// pointer to input image (DEVICE GLOBAL MEMORY)

	// indexes
	int bx = blockIdx.x;													// get current horizontal block index
	int tx = threadIdx.x;													// get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;						// unique thread id, more threads than actual elements !!!

	// copy input to output & log uncompress
	if(ei<d_Ne){															// do only for the number of elements, omit extra threads

		d_I[ei] = exp(d_I[ei]/255);												// exponentiate input IMAGE and copy to output image

	}

}