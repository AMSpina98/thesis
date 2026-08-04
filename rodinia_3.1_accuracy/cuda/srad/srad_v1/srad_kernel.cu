// BUG IN SRAD APPLICATIONS SEEMS TO BE SOMEWHERE IN THIS CODE, WRONG MEMORY ACCESS
#include "../../diversity/diversity.cuh"
#include "../../diversity/converters.cuh"

// srad kernel
__global__ void srad(	fp d_lambda, 
									 int d_Nr, 
									 int d_Nc, 
									 long d_Ne, 
									 int *d_iN, 
									 int *d_iS, 
									 int *d_jE, 
									 int *d_jW, 
									 fp *d_dN, 
									 fp *d_dS, 
									 fp *d_dE, 
									 fp *d_dW, 
									 fp d_q0sqr, 
									 fp *d_c, 
									 fp *d_I){

	// indexes
    int bx = blockIdx.x;													// get current horizontal block index
	int tx = threadIdx.x;													// get current horizontal thread index
	//int ei = bx*NUMBER_THREADS+tx;											// more threads than actual elements !!!
	int ei = dAdd(dMul(bx, NUMBER_THREADS), tx);											// more threads than actual elements !!!
	int row;																// column, x position
	int col;																// row, y position

	// variables
	fp d_Jc;
	fp d_dN_loc, d_dS_loc, d_dW_loc, d_dE_loc;
	fp d_c_loc;
	fp d_G2,d_L,d_num,d_den,d_qsqr;
	
	// figure out row/col location in new matrix
	//row = (ei+1) % d_Nr - 1;													// (0-n) row
	row = dSub(dAdd(ei, 1) % d_Nr, 1);													// (0-n) row
	//col = (ei+1) / d_Nr + 1 - 1;												// to verify
	col = dSub(dAdd(dDiv(dAdd(ei, 1), d_Nr), 1), 1);												// (0-n) column
	//if((ei+1) % d_Nr == 0){
	if(dAdd(ei, 1) % d_Nr == 0){
		//row = d_Nr - 1;
		row = dSub(d_Nr, 1);
		//col = col - 1;
		col = dSub(col, 1);
	}
	
	if(ei<d_Ne){															// make sure that only threads matching jobs run
		
		// directional derivatives, ICOV, diffusion coefficent
		d_Jc = d_I[ei];														// get value of the current element
		
		// directional derivates (every element of IMAGE)(try to copy to shared memory or temp files)
		//d_dN_loc = d_I[d_iN[row] + d_Nr*col] - d_Jc;						// north direction derivative
		d_dN_loc = dSub(d_I[dAdd(d_iN[row], dMul(d_Nr, col))], d_Jc);						// north direction derivative
		//d_dS_loc = d_I[d_iS[row] + d_Nr*col] - d_Jc;						// south direction derivative
		d_dS_loc = dSub(d_I[dAdd(d_iS[row], dMul(d_Nr, col))], d_Jc);						// south direction derivative
		//d_dW_loc = d_I[row + d_Nr*d_jW[col]] - d_Jc;						// west direction derivative
		d_dW_loc = dSub(d_I[dAdd(row, dMul(d_Nr, d_jW[col]))], d_Jc);						// west direction derivative
		//d_dE_loc = d_I[row + d_Nr*d_jE[col]] - d_Jc;						// east direction derivative
		d_dE_loc = dSub(d_I[dAdd(row, dMul(d_Nr, d_jE[col]))], d_Jc);						// east direction derivative
	 
		// normalized discrete gradient mag squared (equ 52,53)
		//d_G2 = (d_dN_loc*d_dN_loc + d_dS_loc*d_dS_loc + d_dW_loc*d_dW_loc + d_dE_loc*d_dE_loc) / (d_Jc*d_Jc);	// gradient (based on derivatives)
		d_G2 = dDiv(dAdd(dAdd(dMul(d_dN_loc, d_dN_loc), dMul(d_dS_loc, d_dS_loc)), dAdd(dMul(d_dW_loc, d_dW_loc), dMul(d_dE_loc, d_dE_loc))), dMul(d_Jc, d_Jc));	// gradient (based on derivatives)
		
		// normalized discrete laplacian (equ 54)
		//d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) / d_Jc;			// laplacian (based on derivatives)
		d_L = dDiv(dAdd(dAdd(dAdd(d_dN_loc, d_dS_loc), d_dW_loc), d_dE_loc), d_Jc);			// laplacian (based on derivatives)

		// ICOV (equ 31/35)
		//d_num  = (0.5*d_G2) - ((1.0/16.0)*(d_L*d_L)) ;						// num (based on gradient and laplacian)
		d_num  = dSub(dMul(0.5, d_G2), dMul(dDiv(1.0, 16.0), dMul(d_L, d_L)));						// num (based on gradient and laplacian)
		//d_den  = 1 + (0.25*d_L);												// den (based on laplacian)
		d_den  = dAdd(1, dMul(0.25, d_L));												// den (based on laplacian)
		//d_qsqr = d_num/(d_den*d_den);										// qsqr (based on num and den)
		d_qsqr = dDiv(d_num, dMul(d_den, d_den));										// qsqr (based on num and den)
	 
		// diffusion coefficent (equ 33) (every element of IMAGE)
		//d_den = (d_qsqr-d_q0sqr) / (d_q0sqr * (1+d_q0sqr)) ;				// den (based on qsqr and q0sqr)
		d_den = dDiv(dSub(d_qsqr, d_q0sqr), dMul(d_q0sqr, dAdd(1, d_q0sqr)));				// den (based on qsqr and q0sqr)
		//d_c_loc = 1.0 / (1.0+d_den) ;										// diffusion coefficient (based on den)
		d_c_loc = dDiv(1.0, dAdd(1.0, d_den));										// diffusion coefficient (based on den)
	    
		// saturate diffusion coefficent to 0-1 range
		if (d_c_loc < 0){													// if diffusion coefficient < 0
			d_c_loc = 0;													// ... set to 0
		}
		else if (d_c_loc > 1){												// if diffusion coefficient > 1
			d_c_loc = 1;													// ... set to 1
		}

		// save data to global memory
		d_dN[ei] = d_dN_loc; 
		d_dS[ei] = d_dS_loc; 
		d_dW[ei] = d_dW_loc; 
		d_dE[ei] = d_dE_loc;
		d_c[ei] = d_c_loc;
			
	}
	
}



__global__ void o_srad(	fp d_lambda, 
									 int d_Nr, 
									 int d_Nc, 
									 long d_Ne, 
									 int *d_iN, 
									 int *d_iS, 
									 int *d_jE, 
									 int *d_jW, 
									 fp *d_dN, 
									 fp *d_dS, 
									 fp *d_dE, 
									 fp *d_dW, 
									 fp d_q0sqr, 
									 fp *d_c, 
									 fp *d_I){

	// indexes
    int bx = blockIdx.x;													// get current horizontal block index
	int tx = threadIdx.x;													// get current horizontal thread index
	int ei = bx*NUMBER_THREADS+tx;											// more threads than actual elements !!!
	int row;																// column, x position
	int col;																// row, y position

	// variables
	fp d_Jc;
	fp d_dN_loc, d_dS_loc, d_dW_loc, d_dE_loc;
	fp d_c_loc;
	fp d_G2,d_L,d_num,d_den,d_qsqr;
	
	// figure out row/col location in new matrix
	row = (ei+1) % d_Nr - 1;													// (0-n) row
	col = (ei+1) / d_Nr + 1 - 1;												// (0-n) column
	if((ei+1) % d_Nr == 0){
		row = d_Nr - 1;
		col = col - 1;
	}
	
	if(ei<d_Ne){															// make sure that only threads matching jobs run
		
		// directional derivatives, ICOV, diffusion coefficent
		d_Jc = d_I[ei];														// get value of the current element
		
		// directional derivates (every element of IMAGE)(try to copy to shared memory or temp files)
		d_dN_loc = d_I[d_iN[row] + d_Nr*col] - d_Jc;						// north direction derivative
		d_dS_loc = d_I[d_iS[row] + d_Nr*col] - d_Jc;						// south direction derivative
		d_dW_loc = d_I[row + d_Nr*d_jW[col]] - d_Jc;						// west direction derivative
		d_dE_loc = d_I[row + d_Nr*d_jE[col]] - d_Jc;						// east direction derivative
	         
		// normalized discrete gradient mag squared (equ 52,53)
		d_G2 = (d_dN_loc*d_dN_loc + d_dS_loc*d_dS_loc + d_dW_loc*d_dW_loc + d_dE_loc*d_dE_loc) / (d_Jc*d_Jc);	// gradient (based on derivatives)
		
		// normalized discrete laplacian (equ 54)
		d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) / d_Jc;			// laplacian (based on derivatives)

		// ICOV (equ 31/35)
		d_num  = (0.5*d_G2) - ((1.0/16.0)*(d_L*d_L)) ;						// num (based on gradient and laplacian)
		d_den  = 1 + (0.25*d_L);												// den (based on laplacian)
		d_qsqr = d_num/(d_den*d_den);										// qsqr (based on num and den)
	 
		// diffusion coefficent (equ 33) (every element of IMAGE)
		d_den = (d_qsqr-d_q0sqr) / (d_q0sqr * (1+d_q0sqr)) ;				// den (based on qsqr and q0sqr)
		d_c_loc = 1.0 / (1.0+d_den) ;										// diffusion coefficient (based on den)
	    
		// saturate diffusion coefficent to 0-1 range
		if (d_c_loc < 0){													// if diffusion coefficient < 0
			d_c_loc = 0;													// ... set to 0
		}
		else if (d_c_loc > 1){												// if diffusion coefficient > 1
			d_c_loc = 1;													// ... set to 1
		}

		// save data to global memory
		d_dN[ei] = d_dN_loc; 
		d_dS[ei] = d_dS_loc; 
		d_dW[ei] = d_dW_loc; 
		d_dE[ei] = d_dE_loc;
		d_c[ei] = d_c_loc;
			
	}
	
}
