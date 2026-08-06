// statistical kernel
#include "../../diversity/diversity.cuh"
#include "../../diversity/converters.cuh"

__global__ void compress(	long d_Ne,
											fp *d_I){									// pointer to output image (DEVICE GLOBAL MEMORY)

	// indexes
	int bx = blockIdx.x;													// get current horizontal block index
	int tx = threadIdx.x;													// get current horizontal thread index
	//int ei = (bx*NUMBER_THREADS)+tx;											// unique thread id, more threads than actual elements !!!
	int ei = dAdd(dMul(bx, NUMBER_THREADS), tx);											// unique thread id, more threads than actual elements !!!

	// copy input to output & log uncompress
	if(ei<d_Ne){															// do only for the number of elements, omit extra threads

		d_I[ei] = dMul(log(d_I[ei]), 255);												// exponentiate input IMAGE and copy to output image
		//d_I[ei] = log(d_I[ei])*255;												// exponentiate input IMAGE and copy to output image

	}

}
