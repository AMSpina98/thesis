// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "srad.h"
#include "./../compare_utils.cuh"

// includes, project
#include <cuda.h>

// includes, kernels
#include "srad_kernel.cu"

void random_matrix(float *I, int rows, int cols);
void runTest( int argc, char** argv);
void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <lamda> <no. of iter>\n", argv[0]);
	fprintf(stderr, "\t<rows>   - number of rows\n");
	fprintf(stderr, "\t<cols>    - number of cols\n");
	fprintf(stderr, "\t<y1> 	 - y1 value of the speckle\n");
	fprintf(stderr, "\t<y2>      - y2 value of the speckle\n");
	fprintf(stderr, "\t<x1>       - x1 value of the speckle\n");
	fprintf(stderr, "\t<x2>       - x2 value of the speckle\n");
	fprintf(stderr, "\t<lamda>   - lambda (0,1)\n");
	fprintf(stderr, "\t<no. of iter>   - number of iterations\n");
	
	exit(1);
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
    runTest( argc, argv);

    return EXIT_SUCCESS;
}


void
runTest( int argc, char** argv) 
{
    int rows, cols, size_I, size_R, niter = 10, iter;
    float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI,varROI ;

	

	cudaEvent_t startE, stopE;
	cudaEventCreate(&startE);
	cudaEventCreate(&stopE);
	cudaEventRecord(startE);



#ifdef CPU
	float Jc, G2, L, num, den, qsqr;
	int *iN,*iS,*jE,*jW, k;
	float *dN,*dS,*dW,*dE;
	float cN,cS,cW,cE,D;
#endif

#ifdef GPU
	
	float *J_cuda, *o_J_cuda;
    float *C_cuda, *o_C_cuda;
	float *E_C, *W_C, *N_C, *S_C, *o_E_C, *o_W_C, *o_N_C, *o_S_C;

#endif

	unsigned int r1, r2, c1, c2;
	float *c;
    
	
 
	if (argc == 9)
	{
		rows = atoi(argv[1]);  //number of rows in the domain
		cols = atoi(argv[2]);  //number of cols in the domain
		if ((rows%16!=0) || (cols%16!=0)){
		fprintf(stderr, "rows and cols must be multiples of 16\n");
		exit(1);
		}
		r1   = atoi(argv[3]);  //y1 position of the speckle
		r2   = atoi(argv[4]);  //y2 position of the speckle
		c1   = atoi(argv[5]);  //x1 position of the speckle
		c2   = atoi(argv[6]);  //x2 position of the speckle
		lambda = atof(argv[7]); //Lambda value
		niter = atoi(argv[8]); //number of iterations
		
	}
    else{
	usage(argc, argv);
    }



	size_I = cols * rows;
    size_R = (r2-r1+1)*(c2-c1+1);   

	I = (float *)malloc( size_I * sizeof(float) );
    J = (float *)malloc( size_I * sizeof(float) );
	c  = (float *)malloc(sizeof(float)* size_I) ;

	
	float *E_C_host = (float *)malloc( size_I * sizeof(float) );
	float *W_C_host = (float *)malloc( size_I * sizeof(float) );
	float *N_C_host = (float *)malloc( size_I * sizeof(float) );
	float *S_C_host = (float *)malloc( size_I * sizeof(float) );
	float *C_host = (float *)malloc( size_I * sizeof(float) );
	float *o_E_C_host = (float *)malloc( size_I * sizeof(float) );
	float *o_W_C_host = (float *)malloc( size_I * sizeof(float) );
	float *o_N_C_host = (float *)malloc( size_I * sizeof(float) );
	float *o_S_C_host = (float *)malloc( size_I * sizeof(float) );
	float *o_C_host = (float *)malloc( size_I * sizeof(float) );
	float *o_J_host = (float *)malloc( size_I * sizeof(float) );
	float *J_host = (float *)malloc( size_I * sizeof(float) );


#ifdef CPU

    iN = (int *)malloc(sizeof(unsigned int*) * rows) ;
    iS = (int *)malloc(sizeof(unsigned int*) * rows) ;
    jW = (int *)malloc(sizeof(unsigned int*) * cols) ;
    jE = (int *)malloc(sizeof(unsigned int*) * cols) ;    


	dN = (float *)malloc(sizeof(float)* size_I) ;
    dS = (float *)malloc(sizeof(float)* size_I) ;
    dW = (float *)malloc(sizeof(float)* size_I) ;
    dE = (float *)malloc(sizeof(float)* size_I) ;    
    

    for (int i=0; i< rows; i++) {
        iN[i] = i-1;
        iS[i] = i+1;
    }    
    for (int j=0; j< cols; j++) {
        jW[j] = j-1;
        jE[j] = j+1;
    }
    iN[0]    = 0;
    iS[rows-1] = rows-1;
    jW[0]    = 0;
    jE[cols-1] = cols-1;

#endif

#ifdef GPU

	//Allocate device memory
    cudaMalloc((void**)& J_cuda, sizeof(float)* size_I);
    cudaMalloc((void**)& C_cuda, sizeof(float)* size_I);
	cudaMalloc((void**)& E_C, sizeof(float)* size_I);
	cudaMalloc((void**)& W_C, sizeof(float)* size_I);
	cudaMalloc((void**)& S_C, sizeof(float)* size_I);
	cudaMalloc((void**)& N_C, sizeof(float)* size_I);
	cudaMalloc((void**)& o_C_cuda, sizeof(float)* size_I);
	cudaMalloc((void**)& o_E_C, sizeof(float)* size_I);
	cudaMalloc((void**)& o_W_C, sizeof(float)* size_I);
	cudaMalloc((void**)& o_N_C, sizeof(float)* size_I);
	cudaMalloc((void**)& o_S_C, sizeof(float)* size_I);
	cudaMalloc((void**)& o_J_cuda, sizeof(float)* size_I);

#endif 

	printf("Randomizing the input matrix\n");
	//Generate a random matrix
	random_matrix(I, rows, cols);

    for (int k = 0;  k < size_I; k++ ) {
     	J[k] = (float)exp(I[k]) ;
    }
	printf("Start the SRAD main loop\n");
 for (iter=0; iter< niter; iter++){     
		sum=0; sum2=0;
        for (int i=r1; i<=r2; i++) {
            for (int j=c1; j<=c2; j++) {
                tmp   = J[i * cols + j];
                sum  += tmp ;
                sum2 += tmp*tmp;
            }
        }
        meanROI = sum / size_R;
        varROI  = (sum2 / size_R) - meanROI*meanROI;
        q0sqr   = varROI / (meanROI*meanROI);

#ifdef CPU
        
		for (int i = 0 ; i < rows ; i++) {
            for (int j = 0; j < cols; j++) { 
		
				k = i * cols + j;
				Jc = J[k];
 
				// directional derivates
                dN[k] = J[iN[i] * cols + j] - Jc;
                dS[k] = J[iS[i] * cols + j] - Jc;
                dW[k] = J[i * cols + jW[j]] - Jc;
                dE[k] = J[i * cols + jE[j]] - Jc;
			
                G2 = (dN[k]*dN[k] + dS[k]*dS[k] 
                    + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);

   		        L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

				num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;
                den  = 1 + (.25*L);
                qsqr = num/(den*den);
 
                // diffusion coefficent (equ 33)
                den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
                c[k] = 1.0 / (1.0+den) ;
                
                // saturate diffusion coefficent
                if (c[k] < 0) {c[k] = 0;}
                else if (c[k] > 1) {c[k] = 1;}
		}
	}
         for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {        

                // current index
                k = i * cols + j;
                
                // diffusion coefficent
					cN = c[k];
					cS = c[iS[i] * cols + j];
					cW = c[k];
					cE = c[i * cols + jE[j]];

                // divergence (equ 58)
                D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
                
                // image update (equ 61)
                J[k] = J[k] + 0.25*lambda*D;
            }
	}

#endif // CPU


#ifdef GPU

	//Currently the input size must be divided by 16 - the block size
	int block_x = cols/BLOCK_SIZE ;
    int block_y = rows/BLOCK_SIZE ;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(block_x , block_y);
    

	//Copy data from main memory to device memory
	cudaMemcpy(J_cuda, J, sizeof(float) * size_I, cudaMemcpyHostToDevice);
	cudaMemcpy(o_J_cuda, J, sizeof(float) * size_I, cudaMemcpyHostToDevice);

	//Run kernels



	srad_cuda_1<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, q0sqr); 

	cudaEventRecord(stopE);
	cudaEventSynchronize(stopE);
	float elapsed;
	cudaEventElapsedTime(&elapsed, startE, stopE);

	cudaDeviceSynchronize();

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("srad_cuda_1 error: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	
	cudaEventRecord(startE);

	o_srad_cuda_1<<<dimGrid, dimBlock>>>(o_E_C, o_W_C, o_N_C, o_S_C, o_J_cuda, o_C_cuda, cols, rows, q0sqr); 

	cudaEventRecord(stopE);
	cudaEventSynchronize(stopE);
	float o_elapsed_1;
	cudaEventElapsedTime(&o_elapsed_1, startE, stopE);

	cudaDeviceSynchronize();

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("o_srad_cuda_1 error: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	cudaEventRecord(startE);

	srad_cuda_2<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, lambda, q0sqr); 

	cudaEventRecord(stopE);
	cudaEventSynchronize(stopE);
	float elapsed_2;
	cudaEventElapsedTime(&elapsed_2, startE, stopE);

	cudaDeviceSynchronize();

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("srad_cuda_2 error: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	
	cudaEventRecord(startE);
	
	o_srad_cuda_2<<<dimGrid, dimBlock>>>(o_E_C, o_W_C, o_N_C, o_S_C, o_J_cuda, o_C_cuda, cols, rows, lambda, q0sqr); 

	cudaEventRecord(stopE);
	cudaEventSynchronize(stopE);
	float o_elapsed_2;
	cudaEventElapsedTime(&o_elapsed_2, startE, stopE);

	cudaDeviceSynchronize();

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("o_srad_cuda_2 error: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}


	cudaMemcpy(E_C_host, E_C, size_I * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(W_C_host, W_C, size_I * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(N_C_host, N_C, size_I * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(S_C_host, S_C, size_I * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(C_host, C_cuda, size_I * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(o_E_C_host, o_E_C, size_I * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(o_W_C_host, o_W_C, size_I * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(o_N_C_host, o_N_C, size_I * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(o_S_C_host, o_S_C, size_I * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(o_C_host, o_C_cuda, size_I * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(o_J_host, o_J_cuda, size_I * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(J_host, J_cuda, size_I * sizeof(float), cudaMemcpyDeviceToHost);

	write_array_to_file("srad_v2_results.csv", "E_C", E_C_host, o_E_C_host, size_I);
	write_array_to_file("srad_v2_results.csv", "W_C", W_C_host, o_W_C_host, size_I);
	write_array_to_file("srad_v2_results.csv", "N_C", N_C_host, o_N_C_host, size_I);
	write_array_to_file("srad_v2_results.csv", "S_C", S_C_host, o_S_C_host, size_I);
	write_array_to_file("srad_v2_results.csv", "C", C_host, o_C_host, size_I);
	write_array_to_file("srad_v2_results.csv", "J", J_host, o_J_host, size_I);
	write_time_to_file("srad_v2_results.csv", "srad_cuda_1", elapsed, o_elapsed_1);
	write_time_to_file("srad_v2_results.csv", "srad_cuda_2", elapsed_2, o_elapsed_2);

	//Copy data from device memory to main memory
    cudaMemcpy(J, J_cuda, sizeof(float) * size_I, cudaMemcpyDeviceToHost);

#endif   
}

    cudaDeviceSynchronize();

#ifdef OUTPUT
    //Printing output	
		printf("Printing Output:\n"); 
    for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
         printf("%.5f ", J[i * cols + j]); 
		}	
     printf("\n"); 
   }
#endif 

	printf("Computation Done\n");
	cudaEventDestroy(startE);
	cudaEventDestroy(stopE);
	free(E_C_host);
	free(W_C_host);
	free(N_C_host);
	free(S_C_host);
	free(C_host);
	free(o_E_C_host);
	free(o_W_C_host);
	free(o_N_C_host);
	free(o_S_C_host);
	free(o_C_host);
	free(o_J_host);
	free(J_host);

  
	free(I);
	free(J);
#ifdef CPU
	free(iN); free(iS); free(jW); free(jE);
    free(dN); free(dS); free(dW); free(dE);
#endif
#ifdef GPU
    cudaFree(C_cuda);
	cudaFree(J_cuda);
	cudaFree(E_C);
	cudaFree(W_C);
	cudaFree(N_C);
	cudaFree(S_C);
    cudaFree(o_C_cuda);
	cudaFree(o_J_cuda);
	cudaFree(o_E_C);
	cudaFree(o_W_C);
	cudaFree(o_N_C);
	cudaFree(o_S_C);
#endif 
	free(c);
  
}


void random_matrix(float *I, int rows, int cols){
    
	srand(7);
	
	for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
		 I[i * cols + j] = rand()/(float)RAND_MAX ;
		}
	}

}

