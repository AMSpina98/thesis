#include "srad.h"
#include <stdio.h>

__global__ void
srad_cuda_1(
		  float *E_C, 
		  float *W_C, 
		  float *N_C, 
		  float *S_C,
		  float * J_cuda, 
		  float * C_cuda, 
		  int cols, 
		  int rows, 
		  float q0sqr
) 
{

  //block id
  int bx = blockIdx.x;
  int by = blockIdx.y;

  //thread id
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  //indices
  //int index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
  int index   = dAdd(dAdd(dMul(dMul(cols, BLOCK_SIZE), by), dMul(BLOCK_SIZE, bx)) + dAdd(dMul(cols, ty), tx));
  //int index_n = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + tx - cols;
  int index_n = dAdd(dMul(dMul(cols, BLOCK_SIZE), by), dSub(dAdd(dMul(BLOCK_SIZE, bx), tx), cols));
  //int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
  int index_s = dAdd(dMul(dMul(cols, BLOCK_SIZE), by), dAdd(dMul(BLOCK_SIZE, bx), dMul(cols, BLOCK_SIZE) + tx));
  //int index_w = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty - 1;
  int index_w = dAdd(dMul(dMul(cols, BLOCK_SIZE), by), dAdd(dMul(BLOCK_SIZE, bx), dSub(dMul(cols, ty), 1)));
  //int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;
  int index_e = dAdd(dMul(dMul(cols, BLOCK_SIZE), by), dAdd(dMul(BLOCK_SIZE, bx), dAdd(dMul(cols, ty), BLOCK_SIZE)));

  float n, w, e, s, jc, g2, l, num, den, qsqr, c;

  //shared memory allocation
  __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float temp_result[BLOCK_SIZE][BLOCK_SIZE];

  __shared__ float north[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float south[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float  east[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float  west[BLOCK_SIZE][BLOCK_SIZE];

  //load data to shared memory
  north[ty][tx] = J_cuda[index_n]; 
  south[ty][tx] = J_cuda[index_s];
  if ( by == 0 ){
  //north[ty][tx] = J_cuda[BLOCK_SIZE * bx + tx]; 
  north[ty][tx] = J_cuda[dAdd(dMul(BLOCK_SIZE, bx), tx)]; 
  }
  //else if ( by == gridDim.y - 1 ){
  else if ( by == dSub(gridDim.y, 1) ){
  //south[ty][tx] = J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx];
  south[ty][tx] = J_cuda[dAdd(dAdd(dMul(dMul(cols, BLOCK_SIZE), dSub(gridDim.y, 1)), dMul(BLOCK_SIZE, bx)), dAdd(dMul(cols, dSub(BLOCK_SIZE, 1)), tx))];
  }
   __syncthreads();
 
  west[ty][tx] = J_cuda[index_w];
  east[ty][tx] = J_cuda[index_e];

  if ( bx == 0 ){
  //west[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + cols * ty]; 
  west[ty][tx] = J_cuda[dAdd(dMul(cols, dMul(BLOCK_SIZE, by)), dMul(cols, ty))]; 
  }
  //else if ( bx == gridDim.x - 1 ){
  else if ( bx == dSub(gridDim.x, 1) ){
  //east[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * ( gridDim.x - 1) + cols * ty + BLOCK_SIZE-1];
  east[ty][tx] = J_cuda[dAdd(dAdd(dMul(cols, dMul(BLOCK_SIZE, by)), dMul(BLOCK_SIZE, dSub(gridDim.x, 1))), dAdd(dMul(cols, ty), dSub(BLOCK_SIZE, 1)))];
  }
 
  __syncthreads();
  
 

  temp[ty][tx]      = J_cuda[index];

  __syncthreads();

   jc = temp[ty][tx];

   if ( ty == 0 && tx == 0 ){ //nw
	//n  = north[ty][tx] - jc;
	n  = dSub(north[ty][tx], jc);
    //s  = temp[ty+1][tx] - jc;
    s  = dSub(temp[ty+1][tx], jc);
    //w  = west[ty][tx]  - jc; 
    w  = dSub(west[ty][tx], jc); 
    //e  = temp[ty][tx+1] - jc;
    e  = dSub(temp[ty][tx+1], jc);
   }	    
   //else if ( ty == 0 && tx == BLOCK_SIZE-1 ){ //ne
   else if ( ty == 0 && tx == dSub(BLOCK_SIZE, 1) ){ //ne
	//n  = north[ty][tx] - jc;
	n  = dSub(north[ty][tx], jc);
    //s  = temp[ty+1][tx] - jc;
    s  = dSub(temp[ty+1][tx], jc);
    //w  = temp[ty][tx-1] - jc; 
    w  = dSub(temp[ty][tx-1], jc); 
    //e  = east[ty][tx] - jc;
    e  = dSub(east[ty][tx], jc);
   }
   //else if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
   else if ( ty == dSub(BLOCK_SIZE, 1) && tx == dSub(BLOCK_SIZE, 1)){ //se
	//n  = temp[ty-1][tx] - jc;
	n  = dSub(temp[ty-1][tx], jc);
    //s  = south[ty][tx] - jc;
    s  = dSub(south[ty][tx], jc);
    //w  = temp[ty][tx-1] - jc; 
    w  = dSub(temp[ty][tx-1], jc); 
    //e  = east[ty][tx]  - jc;
    e  = dSub(east[ty][tx], jc);
   }
   //else if ( ty == BLOCK_SIZE -1 && tx == 0 ){//sw
   else if ( ty == dSub(BLOCK_SIZE, 1) && tx == 0 ){//sw
	//n  = temp[ty-1][tx] - jc;
	n  = dSub(temp[ty-1][tx], jc);
    //s  = south[ty][tx] - jc;
    s  = dSub(south[ty][tx], jc);
    //w  = west[ty][tx]  - jc; 
    w  = dSub(west[ty][tx], jc); 
    //e  = temp[ty][tx+1] - jc;
    e  = dSub(temp[ty][tx+1], jc);
   }

   else if ( ty == 0 ){ //n
	//n  = north[ty][tx] - jc;
	n  = dSub(north[ty][tx], jc);
    //s  = temp[ty+1][tx] - jc;
    s  = dSub(temp[ty+1][tx], jc);
    //w  = temp[ty][tx-1] - jc; 
    w  = dSub(temp[ty][tx-1], jc); 
    //e  = temp[ty][tx+1] - jc;
    e  = dSub(temp[ty][tx+1], jc);
   }
   //else if ( tx == BLOCK_SIZE -1 ){ //e
   else if ( tx == dSub(BLOCK_SIZE, 1) ){ //e
	//n  = temp[ty-1][tx] - jc;
	n  = dSub(temp[ty-1][tx], jc);
    //s  = temp[ty+1][tx] - jc;
    s  = dSub(temp[ty+1][tx], jc);
    //w  = temp[ty][tx-1] - jc; 
    w  = dSub(temp[ty][tx-1], jc); 
    //e  = east[ty][tx] - jc;
    e  = dSub(east[ty][tx], jc);
   }
   //else if ( ty == BLOCK_SIZE -1){ //s
   else if ( ty == dSub(BLOCK_SIZE, 1) ){ //s
	//n  = temp[ty-1][tx] - jc;
	n  = dSub(temp[ty-1][tx], jc);
    //s  = south[ty][tx] - jc;
    s  = dSub(south[ty][tx], jc);
    //w  = temp[ty][tx-1] - jc; 
    w  = dSub(temp[ty][tx-1], jc); 
    //e  = temp[ty][tx+1] - jc;
    e  = dSub(temp[ty][tx+1], jc);
   }
   else if ( tx == 0 ){ //w
	//n  = temp[ty-1][tx] - jc;
	n  = dSub(temp[ty-1][tx], jc);
    //s  = temp[ty+1][tx] - jc;
    s  = dSub(temp[ty+1][tx], jc);
    //w  = west[ty][tx] - jc; 
    w  = dSub(west[ty][tx], jc); 
    //e  = temp[ty][tx+1] - jc;
    e  = dSub(temp[ty][tx+1], jc);
   }
   else{  //the data elements which are not on the borders 
	//n  = temp[ty-1][tx] - jc;
	n  = dSub(temp[ty-1][tx], jc);
    //s  = temp[ty+1][tx] - jc;
    s  = dSub(temp[ty+1][tx], jc);
    //w  = temp[ty][tx-1] - jc; 
    w  = dSub(temp[ty][tx-1], jc); 
    //e  = temp[ty][tx+1] - jc;
    e  = dSub(temp[ty][tx+1], jc);
   }


    //g2 = ( n * n + s * s + w * w + e * e ) / (jc * jc);
    g2 = dDiv(dAdd(dAdd(dMul(n, n), dMul(s, s)), dAdd(dMul(w, w), dMul(e, e))), dMul(jc, jc));

    //l = ( n + s + w + e ) / jc;
    l = dDiv(dAdd(dAdd(n, s), dAdd(w, e)), jc);

	//num  = (0.5*g2) - ((1.0/16.0)*(l*l)) ;
	num  = dSub(dMul(0.5, g2), dMul(dDiv(1.0, 16.0), dMul(l, l)));
	//den  = 1 + (.25*l);
	den  = dAdd(1, dMul(0.25, l));
	//qsqr = num/(den*den);
	qsqr = dDiv(num, dMul(den, den));

	// diffusion coefficent (equ 33)
	//den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
	den = dDiv(dSub(qsqr, q0sqr), dMul(q0sqr, dAdd(1, q0sqr)));
	//c = 1.0 / (1.0+den);
	c = dDiv(1.0, dAdd(1.0, den));

    // saturate diffusion coefficent
	if (c < 0){temp_result[ty][tx] = 0;}
	else if (c > 1) {temp_result[ty][tx] = 1;}
	else {temp_result[ty][tx] = c;}

    __syncthreads();

    C_cuda[index] = temp_result[ty][tx];
	E_C[index] = e;
	W_C[index] = w;
	S_C[index] = s;
	N_C[index] = n;

}

__global__ void
srad_cuda_2(
		  float *E_C, 
		  float *W_C, 
		  float *N_C, 
		  float *S_C,	
		  float * J_cuda, 
		  float * C_cuda, 
		  int cols, 
		  int rows, 
		  float lambda,
		  float q0sqr
) 
{
	//block id
	int bx = blockIdx.x;
    int by = blockIdx.y;

	//thread id
    int tx = threadIdx.x;
    int ty = threadIdx.y;

	//indices
    //int index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
    int index   = dAdd(dAdd(dMul(cols, dMul(BLOCK_SIZE, by)), dMul(BLOCK_SIZE, bx)), dAdd(dMul(cols, ty), tx));
	//int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
	int index_s = dAdd(dAdd(dMul(cols, dMul(BLOCK_SIZE, by)), dMul(BLOCK_SIZE, bx)), dAdd(dMul(cols, BLOCK_SIZE), tx));
    //int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;
    int index_e = dAdd(dAdd(dMul(cols, dMul(BLOCK_SIZE, by)), dMul(BLOCK_SIZE, bx)), dAdd(dMul(cols, ty), BLOCK_SIZE));
	float cc, cn, cs, ce, cw, d_sum;

	//shared memory allocation
	__shared__ float south_c[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float  east_c[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float c_cuda_result[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];

    //load data to shared memory
	temp[ty][tx]      = J_cuda[index];

    __syncthreads();
	 
	south_c[ty][tx] = C_cuda[index_s];

	//if ( by == gridDim.y - 1 ){
	if ( by == dSub(gridDim.y, 1) ){
	//south_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx];
	south_c[ty][tx] = C_cuda[dAdd(dMul(cols, dMul(BLOCK_SIZE, dSub(gridDim.y, 1))), dAdd(dMul(BLOCK_SIZE, bx), dAdd(dMul(cols, dSub(BLOCK_SIZE, 1)), tx)))];
	}
	__syncthreads();
	 
	 
	east_c[ty][tx] = C_cuda[index_e];
	
	//if ( bx == gridDim.x - 1 ){
	if ( bx == dSub(gridDim.x, 1) ){
	//east_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * ( gridDim.x - 1) + cols * ty + BLOCK_SIZE-1];
	east_c[ty][tx] = C_cuda[dAdd(dMul(cols, dMul(BLOCK_SIZE, by)), dAdd(dMul(BLOCK_SIZE, dSub(gridDim.x, 1)), dAdd(dMul(cols, ty), dSub(BLOCK_SIZE, 1))))];
	}
	 
    __syncthreads();
  
    c_cuda_temp[ty][tx]      = C_cuda[index];

    __syncthreads();

	cc = c_cuda_temp[ty][tx];

   //if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
   if ( ty == dSub(BLOCK_SIZE, 1) && tx == dSub(BLOCK_SIZE, 1)){ //se
	cn  = cc;
    cs  = south_c[ty][tx];
    cw  = cc; 
    ce  = east_c[ty][tx];
   } 
   //else if ( tx == BLOCK_SIZE -1 ){ //e
   else if ( tx == dSub(BLOCK_SIZE, 1) ){ //e
	cn  = cc;
    //cs  = c_cuda_temp[ty+1][tx];
    cs  = c_cuda_temp[dAdd(ty, 1)][tx];
    cw  = cc; 
    ce  = east_c[ty][tx];
   }
   //else if ( ty == BLOCK_SIZE -1){ //s
   else if ( ty == dSub(BLOCK_SIZE, 1) ){ //s
	cn  = cc;
    cs  = south_c[ty][tx];
    cw  = cc; 
    //ce  = c_cuda_temp[ty][tx+1];
    ce  = c_cuda_temp[ty][dAdd(tx, 1)];
   }
   else{ //the data elements which are not on the borders 
	cn  = cc;
    //cs  = c_cuda_temp[ty+1][tx];
    cs  = c_cuda_temp[dAdd(ty, 1)][tx];
    cw  = cc; 
    //ce  = c_cuda_temp[ty][tx+1];
    ce  = c_cuda_temp[ty][dAdd(tx, 1)];
   }

   // divergence (equ 58)
   //d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];
   d_sum = dAdd(dMul(cn, N_C[index]), dAdd(dMul(cs, S_C[index]), dAdd(dMul(cw, W_C[index]), dMul(ce, E_C[index]))));

   // image update (equ 61)
   //c_cuda_result[ty][tx] = temp[ty][tx] + 0.25 * lambda * d_sum;
   c_cuda_result[ty][tx] = dAdd(temp[ty][tx], dMul(0.25, dMul(lambda, d_sum)));

   __syncthreads();
              
   J_cuda[index] = c_cuda_result[ty][tx];
    
}


__global__ void
o_srad_cuda_1(
		  float *E_C, 
		  float *W_C, 
		  float *N_C, 
		  float *S_C,
		  float * J_cuda, 
		  float * C_cuda, 
		  int cols, 
		  int rows, 
		  float q0sqr
) 
{

  //block id
  int bx = blockIdx.x;
  int by = blockIdx.y;

  //thread id
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  //indices
  int index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
  int index_n = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + tx - cols;
  int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
  int index_w = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty - 1;
  int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;

  float n, w, e, s, jc, g2, l, num, den, qsqr, c;

  //shared memory allocation
  __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float temp_result[BLOCK_SIZE][BLOCK_SIZE];

  __shared__ float north[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float south[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float  east[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float  west[BLOCK_SIZE][BLOCK_SIZE];

  //load data to shared memory
  north[ty][tx] = J_cuda[index_n]; 
  south[ty][tx] = J_cuda[index_s];
  if ( by == 0 ){
  north[ty][tx] = J_cuda[BLOCK_SIZE * bx + tx]; 
  }
  else if ( by == gridDim.y - 1 ){
  south[ty][tx] = J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx];
  }
   __syncthreads();
 
  west[ty][tx] = J_cuda[index_w];
  east[ty][tx] = J_cuda[index_e];

  if ( bx == 0 ){
  west[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + cols * ty]; 
  }
  else if ( bx == gridDim.x - 1 ){
  east[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * ( gridDim.x - 1) + cols * ty + BLOCK_SIZE-1];
  }
 
  __syncthreads();
  
 

  temp[ty][tx]      = J_cuda[index];

  __syncthreads();

   jc = temp[ty][tx];

   if ( ty == 0 && tx == 0 ){ //nw
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = west[ty][tx]  - jc; 
    e  = temp[ty][tx+1] - jc;
   }	    
   else if ( ty == 0 && tx == BLOCK_SIZE-1 ){ //ne
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx] - jc;
   }
   else if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx]  - jc;
   }
   else if ( ty == BLOCK_SIZE -1 && tx == 0 ){//sw
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = west[ty][tx]  - jc; 
    e  = temp[ty][tx+1] - jc;
   }

   else if ( ty == 0 ){ //n
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
   }
   else if ( tx == BLOCK_SIZE -1 ){ //e
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx] - jc;
   }
   else if ( ty == BLOCK_SIZE -1){ //s
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
   }
   else if ( tx == 0 ){ //w
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = west[ty][tx] - jc; 
    e  = temp[ty][tx+1] - jc;
   }
   else{  //the data elements which are not on the borders 
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
   }


    g2 = ( n * n + s * s + w * w + e * e ) / (jc * jc);

    l = ( n + s + w + e ) / jc;

	num  = (0.5*g2) - ((1.0/16.0)*(l*l)) ;
	den  = 1 + (.25*l);
	qsqr = num/(den*den);

	// diffusion coefficent (equ 33)
	den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
	c = 1.0 / (1.0+den) ;

    // saturate diffusion coefficent
	if (c < 0){temp_result[ty][tx] = 0;}
	else if (c > 1) {temp_result[ty][tx] = 1;}
	else {temp_result[ty][tx] = c;}

    __syncthreads();

    C_cuda[index] = temp_result[ty][tx];
	E_C[index] = e;
	W_C[index] = w;
	S_C[index] = s;
	N_C[index] = n;

}

__global__ void
o_srad_cuda_2(
		  float *E_C, 
		  float *W_C, 
		  float *N_C, 
		  float *S_C,	
		  float * J_cuda, 
		  float * C_cuda, 
		  int cols, 
		  int rows, 
		  float lambda,
		  float q0sqr
) 
{
	//block id
	int bx = blockIdx.x;
    int by = blockIdx.y;

	//thread id
    int tx = threadIdx.x;
    int ty = threadIdx.y;

	//indices
    int index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
	int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
    int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;
	float cc, cn, cs, ce, cw, d_sum;

	//shared memory allocation
	__shared__ float south_c[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float  east_c[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float c_cuda_result[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];

    //load data to shared memory
	temp[ty][tx]      = J_cuda[index];

    __syncthreads();
	 
	south_c[ty][tx] = C_cuda[index_s];

	if ( by == gridDim.y - 1 ){
	south_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx];
	}
	__syncthreads();
	 
	 
	east_c[ty][tx] = C_cuda[index_e];
	
	if ( bx == gridDim.x - 1 ){
	east_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * ( gridDim.x - 1) + cols * ty + BLOCK_SIZE-1];
	}
	 
    __syncthreads();
  
    c_cuda_temp[ty][tx]      = C_cuda[index];

    __syncthreads();

	cc = c_cuda_temp[ty][tx];

   if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
	cn  = cc;
    cs  = south_c[ty][tx];
    cw  = cc; 
    ce  = east_c[ty][tx];
   } 
   else if ( tx == BLOCK_SIZE -1 ){ //e
	cn  = cc;
    cs  = c_cuda_temp[ty+1][tx];
    cw  = cc; 
    ce  = east_c[ty][tx];
   }
   else if ( ty == BLOCK_SIZE -1){ //s
	cn  = cc;
    cs  = south_c[ty][tx];
    cw  = cc; 
    ce  = c_cuda_temp[ty][tx+1];
   }
   else{ //the data elements which are not on the borders 
	cn  = cc;
    cs  = c_cuda_temp[ty+1][tx];
    cw  = cc; 
    ce  = c_cuda_temp[ty][tx+1];
   }

   // divergence (equ 58)
   d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];

   // image update (equ 61)
   c_cuda_result[ty][tx] = temp[ty][tx] + 0.25 * lambda * d_sum;

   __syncthreads();
              
   J_cuda[index] = c_cuda_result[ty][tx];
    
}
