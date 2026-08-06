// statistical kernel
#include "../../diversity/diversity.cuh"
#include "../../diversity/converters.cuh"
__global__ void reduce(	long d_Ne,											// number of elements in array
										int d_no,											// number of sums to reduce
										int d_mul,											// increment
										fp *d_sums,										// pointer to partial sums variable (DEVICE GLOBAL MEMORY)
										fp *d_sums2){

	// indexes
    int bx = blockIdx.x;													// get current horizontal block index
	int tx = threadIdx.x;													// get current horizontal thread index
	//int ei = (bx*NUMBER_THREADS)+tx;										// unique thread id, more threads than actual elements !!!
	int ei = dAdd(dMul(bx, NUMBER_THREADS), tx);										// unique thread id, more threads than actual elements !!!
	//int nf = NUMBER_THREADS-(gridDim.x*NUMBER_THREADS-d_no);				// number of elements assigned to last block
	int nf = dSub(NUMBER_THREADS, dSub(dMul(gridDim.x, NUMBER_THREADS), d_no));				// number of elements assigned to last block
	int df = 0;																// divisibility factor for the last block

	// statistical
	__shared__ fp d_psum[NUMBER_THREADS];								// data for block calculations allocated by every block in its shared memory
	__shared__ fp d_psum2[NUMBER_THREADS];

	// counters
	int i;

	// copy data to shared memory
	if(ei<d_no){															// do only for the number of elements, omit extra threads

		//d_psum[tx] = d_sums[ei*d_mul];
		d_psum[tx] = d_sums[dMul(ei, d_mul)];
		//d_psum2[tx] = d_sums2[ei*d_mul];
		d_psum2[tx] = d_sums2[dMul(ei, d_mul)];

	}

	// Lingjie Zhang modifited at Nov 1 / 2015
    __syncthreads();
    // end Lingjie Zhang's modification

	// reduction of sums if all blocks are full (rare case)	
	if(nf == NUMBER_THREADS){
		// sum of every 2, 4, ..., NUMBER_THREADS elements
		//for(i=2; i<=NUMBER_THREADS; i=2*i){
		for(i=2; i<=NUMBER_THREADS; i=dMul(2, i)){
			// sum of elements
			if(dAdd(tx, 1) % i == 0){						// every ith
				//d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];				// every ith
				d_psum[tx] = dAdd(d_psum[tx], d_psum[dSub(tx, dDiv(i, 2))]);
				//d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];						
				d_psum2[tx] = dAdd(d_psum2[tx], d_psum2[dSub(tx, dDiv(i, 2))]);
			}
			// synchronization
			__syncthreads();
		}
		// final sumation by last thread in every block
		//if(tx==(NUMBER_THREADS-1)){											// block result stored in global memory
		if(tx==dSub(NUMBER_THREADS, 1)){											// block result stored in global memory
			//d_sums[bx*d_mul*NUMBER_THREADS] = d_psum[tx];
			d_sums[dMul(bx, dMul(d_mul, NUMBER_THREADS))] = d_psum[tx];
			//d_sums2[bx*d_mul*NUMBER_THREADS] = d_psum2[tx];
			d_sums2[dMul(bx, dMul(d_mul, NUMBER_THREADS))] = d_psum2[tx];
		}
	}
	// reduction of sums if last block is not full (common case)
	else{ 
		// for full blocks (all except for last block)
		//if(bx != (gridDim.x - 1)){											//
		if(bx != dSub(gridDim.x, 1)){											//
			// sum of every 2, 4, ..., NUMBER_THREADS elements
			//for(i=2; i<=NUMBER_THREADS; i=2*i){								//
			for(i=2; i<=NUMBER_THREADS; i=dMul(2, i)){								//
				// sum of elements
				//if((tx+1) % i == 0){										// every ith
				if(dAdd(tx, 1) % i == 0){
					//d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];				// every ith
					d_psum[tx] = dAdd(d_psum[tx], d_psum[dSub(tx, dDiv(i, 2))]);
					//d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];						
					d_psum2[tx] = dAdd(d_psum2[tx], d_psum2[dSub(tx, dDiv(i, 2))]);
				}
				// synchronization
				__syncthreads();											//
			}
			// final sumation by last thread in every block
			//if(tx==(NUMBER_THREADS-1)){										// block result stored in global memory
			if(tx==dSub(NUMBER_THREADS, 1)){	
				//d_sums[bx*d_mul*NUMBER_THREADS] = d_psum[tx];						// block result stored in global memory
				d_sums[dMul(bx, dMul(d_mul, NUMBER_THREADS))] = d_psum[tx];
				//d_sums2[bx*d_mul*NUMBER_THREADS] = d_psum2[tx];			
				d_sums2[dMul(bx, dMul(d_mul, NUMBER_THREADS))] = d_psum2[tx];
			}
		}
		// for not full block (last block)
		else{																//
			// figure out divisibility
			//for(i=2; i<=NUMBER_THREADS; i=2*i){								//
			for(i=2; i<=NUMBER_THREADS; i=dMul(2, i)){								//
				if(nf >= i){
					df = i;
				}
			}
			// sum of every 2, 4, ..., NUMBER_THREADS elements
			//for(i=2; i<=df; i=2*i){											//
			for(i=2; i<=df; i=dMul(2, i)){											//
				// sum of elements (only busy threads)
				//if((tx+1) % i == 0 && tx<df){								// every ith
				if(dAdd(tx, 1) % i == 0 && tx<df){		
					//d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];// every ith
					d_psum[tx] = dAdd(d_psum[tx], d_psum[dSub(tx, dDiv(i, 2))]);
					//d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];						
					d_psum2[tx] = dAdd(d_psum2[tx], d_psum2[dSub(tx, dDiv(i, 2))]);
				}
				// synchronization (all threads)
				__syncthreads();											//
			}
			// remainder / final summation by last thread
			//if(tx==(df-1)){										//
			if(tx==dSub(df, 1)){										//
				// compute the remainder and final summation by last busy thread
				//for(i=(bx*NUMBER_THREADS)+df; i<(bx*NUMBER_THREADS)+nf; i++){						//
				for(i=dAdd(dMul(bx, NUMBER_THREADS), df); i<dAdd(dMul(bx, NUMBER_THREADS), nf); i = dAdd(i, 1)){						//
					//d_psum[tx] = d_psum[tx] + d_sums[i];
					d_psum[tx] = dAdd(d_psum[tx], d_sums[i]);
					//d_psum2[tx] = d_psum2[tx] + d_sums2[i];
					d_psum2[tx] = dAdd(d_psum2[tx], d_sums2[i]);
				}
				// final sumation by last thread in every block
				//d_sums[bx*d_mul*NUMBER_THREADS] = d_psum[tx];
				d_sums[dMul(bx, dMul(d_mul, NUMBER_THREADS))] = d_psum[tx];
				//d_sums2[bx*d_mul*NUMBER_THREADS] = d_psum2[tx];
				d_sums2[dMul(bx, dMul(d_mul, NUMBER_THREADS))] = d_psum2[tx];
			}
		}
	}

}
