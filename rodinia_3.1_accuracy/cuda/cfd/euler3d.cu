// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

//#include <cutil.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "diversity.cuh"
#include "converters.cuh"
#include "cuda_compat.h"
#include <string>
#include <string_view>

using namespace std;

 
 
/*
 * Options 
 * 
 */ 
#define GAMMA 1.4f
#define iterations 2000
// #ifndef block_length
// 	#define block_length 192
// #endif



#define NDIM 3
#define NNB 4

#define RK 3	// 3rd order RK
#define ff_mach 1.2f
#define deg_angle_of_attack 0.0f

/*
 * not options
 */

#ifdef RD_WG_SIZE_0_0
	#define BLOCK_SIZE_0 RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
	#define BLOCK_SIZE_0 RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
	#define BLOCK_SIZE_0 RD_WG_SIZE
#else
	#define BLOCK_SIZE_0 192
#endif

#ifdef RD_WG_SIZE_1_0
	#define BLOCK_SIZE_1 RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
	#define BLOCK_SIZE_1 RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
	#define BLOCK_SIZE_1 RD_WG_SIZE
#else
	#define BLOCK_SIZE_1 192
#endif

#ifdef RD_WG_SIZE_2_0
	#define BLOCK_SIZE_2 RD_WG_SIZE_2_0
#elif defined(RD_WG_SIZE_1)
	#define BLOCK_SIZE_2 RD_WG_SIZE_2
#elif defined(RD_WG_SIZE)
	#define BLOCK_SIZE_2 RD_WG_SIZE
#else
	#define BLOCK_SIZE_2 192
#endif

#ifdef RD_WG_SIZE_3_0
	#define BLOCK_SIZE_3 RD_WG_SIZE_3_0
#elif defined(RD_WG_SIZE_3)
	#define BLOCK_SIZE_3 RD_WG_SIZE_3
#elif defined(RD_WG_SIZE)
	#define BLOCK_SIZE_3 RD_WG_SIZE
#else
	#define BLOCK_SIZE_3 192
#endif

#ifdef RD_WG_SIZE_4_0
	#define BLOCK_SIZE_4 RD_WG_SIZE_4_0
#elif defined(RD_WG_SIZE_4)
	#define BLOCK_SIZE_4 RD_WG_SIZE_4
#elif defined(RD_WG_SIZE)
	#define BLOCK_SIZE_4 RD_WG_SIZE
#else
	#define BLOCK_SIZE_4 192
#endif



// #if block_length > 128
// #warning "the kernels may fail too launch on some systems if the block length is too large"
// #endif


#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)

#define CHECKPOINT (iterations/20)


/*
 * Generic functions
 */
template <typename T>
T* alloc(int N)
{
	T* t;
	checkCudaErrors(cudaMalloc((void**)&t, sizeof(T)*N));
	return t;
}

template <typename T>
void dealloc(T* array)
{
	checkCudaErrors(cudaFree((void*)array));
}

template <typename T>
void copy(T* dst, T* src, int N)
{
	checkCudaErrors(cudaMemcpy((void*)dst, (void*)src, N*sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
void upload(T* dst, T* src, int N)
{
	checkCudaErrors(cudaMemcpy((void*)dst, (void*)src, N*sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void download(T* dst, T* src, int N)
{
	checkCudaErrors(cudaMemcpy((void*)dst, (void*)src, N*sizeof(T), cudaMemcpyDeviceToHost));
}

void dump(float* variables, int nel, int nelr)
{
	float* h_variables = new float[nelr*NVAR];
	download(h_variables, variables, nelr*NVAR);

	{
		std::ofstream file("density");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY*nelr] << std::endl;
	}


	{
		std::ofstream file("momentum");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++)
		{
			for(int j = 0; j != NDIM; j++)
				file << h_variables[i + (VAR_MOMENTUM+j)*nelr] << " ";
			file << std::endl;
		}
	}
	
	{
		std::ofstream file("density_energy");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY_ENERGY*nelr] << std::endl;
	}
	delete[] h_variables;
}

/*
 * Element-based Cell-centered FVM solver functions
 */
__constant__ float ff_variable[NVAR];
__constant__ float3 ff_flux_contribution_momentum_x[1];
__constant__ float3 ff_flux_contribution_momentum_y[1];
__constant__ float3 ff_flux_contribution_momentum_z[1];
__constant__ float3 ff_flux_contribution_density_energy[1];


__constant__ float original_ff_variable[NVAR];
__constant__ float3 original_ff_flux_contribution_momentum_x[1];
__constant__ float3 original_ff_flux_contribution_momentum_y[1];
__constant__ float3 original_ff_flux_contribution_momentum_z[1];
__constant__ float3 original_ff_flux_contribution_density_energy[1];

__global__ void cuda_initialize_variables(int nelr, float* variables)
{
	const int i = dAdd(dMul(blockDim.x, blockIdx.x), threadIdx.x);
	for(int j = 0; j < NVAR; j = dAdd(j, 1)) {
		float v = ff_variable[j];
		variables[dAdd(i, dMul(j, nelr))] = v;
	}
}
void initialize_variables(int nelr, float* variables)
{
	dim3 Dg(nelr / BLOCK_SIZE_1), Db(BLOCK_SIZE_1);
	cuda_initialize_variables<<<Dg, Db>>>(nelr, variables);
	getLastCudaError("initialize_variables failed");
}

__host__ __device__ inline void compute_flux_contribution(float& density, float3& momentum, float& density_energy, float& pressure, float3& velocity, float3& fc_momentum_x, float3& fc_momentum_y, float3& fc_momentum_z, float3& fc_density_energy)
{
#ifdef __CUDA_ARCH__

		fc_momentum_x.x = dAdd(dMul(velocity.x, momentum.x), pressure);
		fc_momentum_x.y = dMul(velocity.x, momentum.y);
		fc_momentum_x.z = dMul(velocity.x, momentum.z);
		
		
		fc_momentum_y.x = fc_momentum_x.y;
		fc_momentum_y.y = dAdd(dMul(velocity.y, momentum.y), pressure);
		fc_momentum_y.z = dMul(velocity.y, momentum.z);

		fc_momentum_z.x = fc_momentum_x.z;
		fc_momentum_z.y = fc_momentum_y.z;
		fc_momentum_z.z = dAdd(dMul(velocity.z, momentum.z), pressure);

		float de_p = dAdd(density_energy, pressure);
		fc_density_energy.x = dMul(velocity.x, de_p);
		fc_density_energy.y = dMul(velocity.y, de_p);
		fc_density_energy.z = dMul(velocity.z, de_p);

#else
    
		fc_momentum_x.x = velocity.x*momentum.x + pressure;
		fc_momentum_x.y = velocity.x*momentum.y;
		fc_momentum_x.z = velocity.x*momentum.z;
		
		fc_momentum_y.x = fc_momentum_x.y;
		fc_momentum_y.y = velocity.y*momentum.y + pressure;
		fc_momentum_y.z = velocity.y*momentum.z;

		fc_momentum_z.x = fc_momentum_x.z;
		fc_momentum_z.y = fc_momentum_y.z;
		fc_momentum_z.z = velocity.z*momentum.z + pressure;

		float de_p = density_energy+pressure;
		fc_density_energy.x = velocity.x*de_p;
		fc_density_energy.y = velocity.y*de_p;
		fc_density_energy.z = velocity.z*de_p;
	
#endif
}

__device__ inline void compute_velocity(float& density, float3& momentum, float3& velocity)
{
	//velocity.x = momentum.x / density;
	//velocity.y = momentum.y / density;
	//velocity.z = momentum.z / density;
	velocity.x = dDiv(momentum.x, density);
	velocity.y = dDiv(momentum.y, density);
	velocity.z = dDiv(momentum.z, density);
}
	
__device__ inline float compute_speed_sqd(float3& velocity)
{
	//return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
	return dAdd(dAdd(dMul(velocity.x, velocity.x), dMul(velocity.y, velocity.y)), dMul(velocity.z, velocity.z));
}

__device__ inline float compute_pressure(float& density, float& density_energy, float& speed_sqd)
{
	//return (float(GAMMA)-float(1.0f))*(density_energy - float(0.5f)*density*speed_sqd);
	return dMul(dSub(float(GAMMA), float(1.0f)), dSub(density_energy, dMul(float(0.5f), dMul(density, speed_sqd))));
	
}

__device__ inline float compute_speed_of_sound(float& density, float& pressure)
{
	//return sqrtf(float(GAMMA)*pressure/density);
	return sqrtf(dDiv(dMul(float(GAMMA), pressure), density));
	
}

__global__ void cuda_compute_step_factor(int nelr, float* variables, float* areas, float* step_factors)
{
	//const int i = (blockDim.x*blockIdx.x + threadIdx.x);
	const int i = dAdd(dMul(blockDim.x, blockIdx.x), threadIdx.x);

	//float density = variables[i + VAR_DENSITY*nelr];
	float density = variables[dAdd(i, dMul(VAR_DENSITY, nelr))];
	float3 momentum;
	//momentum.x = variables[i + (VAR_MOMENTUM+0)*nelr];
	//momentum.y = variables[i + (VAR_MOMENTUM+1)*nelr];
	//momentum.z = variables[i + (VAR_MOMENTUM+2)*nelr];
	momentum.x = variables[dAdd(i, dMul(dAdd(VAR_MOMENTUM, 0), nelr))];
	momentum.y = variables[dAdd(i, dMul(dAdd(VAR_MOMENTUM, 1), nelr))];
	momentum.z = variables[dAdd(i, dMul(dAdd(VAR_MOMENTUM, 2), nelr))];
	
	//float density_energy = variables[i + VAR_DENSITY_ENERGY*nelr];
	float density_energy = variables[dAdd(i, dMul(VAR_DENSITY_ENERGY, nelr))];
	
	float3 velocity;       compute_velocity(density, momentum, velocity);
	float speed_sqd      = compute_speed_sqd(velocity);
	float pressure       = compute_pressure(density, density_energy, speed_sqd);
	float speed_of_sound = compute_speed_of_sound(density, pressure);

	// dt = float(0.5f) * sqrtf(areas[i]) /  (||v|| + c).... but when we do time stepping, this later would need to be divided by the area, so we just do it all at once
	//step_factors[i] = float(0.5f) / (sqrtf(areas[i]) * (sqrtf(speed_sqd) + speed_of_sound));
	step_factors[i] = dDiv(float(0.5f), dMul(sqrtf(areas[i]), dAdd(sqrtf(speed_sqd), speed_of_sound)));
}
void compute_step_factor(int nelr, float* variables, float* areas, float* step_factors)
{
	dim3 Dg(nelr / BLOCK_SIZE_2), Db(BLOCK_SIZE_2);
	cuda_compute_step_factor<<<Dg, Db>>>(nelr, variables, areas, step_factors);		
	getLastCudaError("compute_step_factor failed");
}

/*
 *
 *
*/
__global__ void cuda_compute_flux(int nelr, int* elements_surrounding_elements, float* normals, float* variables, float* fluxes)
{
	const float smoothing_coefficient = float(0.2f);
	//const int i = (blockDim.x*blockIdx.x + threadIdx.x);
	const int i = dAdd(dMul(blockDim.x, blockIdx.x), threadIdx.x);
	
	int j, nb;
	float3 normal; float normal_len;
	float factor;
	
	//float density_i = variables[i + VAR_DENSITY*nelr];
	float density_i = variables[dAdd(i, dMul(VAR_DENSITY, nelr))];
	float3 momentum_i;
	//momentum_i.x = variables[i + (VAR_MOMENTUM+0)*nelr];
	//momentum_i.y = variables[i + (VAR_MOMENTUM+1)*nelr];
	//momentum_i.z = variables[i + (VAR_MOMENTUM+2)*nelr];
	momentum_i.x = variables[dAdd(i, dMul(dAdd(VAR_MOMENTUM, 0), nelr))];
	momentum_i.y = variables[dAdd(i, dMul(dAdd(VAR_MOMENTUM, 1), nelr))];
	momentum_i.z = variables[dAdd(i, dMul(dAdd(VAR_MOMENTUM, 2), nelr))];

	//float density_energy_i = variables[i + VAR_DENSITY_ENERGY*nelr];
	float density_energy_i = variables[dAdd(i, dMul(VAR_DENSITY_ENERGY, nelr))];

	float3 velocity_i;             				compute_velocity(density_i, momentum_i, velocity_i);
	float speed_sqd_i                          = compute_speed_sqd(velocity_i);
	float speed_i                              = sqrtf(speed_sqd_i);


	float pressure_i                           = compute_pressure(density_i, density_energy_i, speed_sqd_i);
	float speed_of_sound_i                     = compute_speed_of_sound(density_i, pressure_i);
	float3 flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z;
	float3 flux_contribution_i_density_energy;	
	compute_flux_contribution(density_i, momentum_i, density_energy_i, pressure_i, velocity_i, flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z, flux_contribution_i_density_energy);
	
	float flux_i_density = float(0.0f);
	float3 flux_i_momentum;
	flux_i_momentum.x = float(0.0f);
	flux_i_momentum.y = float(0.0f);
	flux_i_momentum.z = float(0.0f);
	float flux_i_density_energy = float(0.0f);
		
	float3 velocity_nb;
	float density_nb, density_energy_nb;
	float3 momentum_nb;
	float3 flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z;
	float3 flux_contribution_nb_density_energy;	
	float speed_sqd_nb, speed_of_sound_nb, pressure_nb;
	
	#pragma unroll
	for(j = 0; j < NNB; j = dAdd(j, 1))
	{
		//nb = elements_surrounding_elements[i + j*nelr];
		//normal.x = normals[i + (j + 0*NNB)*nelr];
		//normal.y = normals[i + (j + 1*NNB)*nelr];
		//normal.z = normals[i + (j + 2*NNB)*nelr];
		nb = elements_surrounding_elements[dAdd(i, dMul(j, nelr))];
		normal.x = normals[dAdd(i, dMul(dAdd(j, dMul(0, NNB)), nelr))];
		normal.y = normals[dAdd(i, dMul(dAdd(j, dMul(1, NNB)), nelr))];
		normal.z = normals[dAdd(i, dMul(dAdd(j, dMul(2, NNB)), nelr))];
		//normal_len = sqrtf(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
		normal_len = sqrtf(dAdd(dAdd(dMul(normal.x, normal.x), dMul(normal.y, normal.y)), dMul(normal.z, normal.z)));
		
		if(nb >= 0) 	// a legitimate neighbor
		{
			//density_nb = variables[nb + VAR_DENSITY*nelr];
			//momentum_nb.x = variables[nb + (VAR_MOMENTUM+0)*nelr];
			//momentum_nb.y = variables[nb + (VAR_MOMENTUM+1)*nelr];
			//momentum_nb.z = variables[nb + (VAR_MOMENTUM+2)*nelr];
			//density_energy_nb = variables[nb + VAR_DENSITY_ENERGY*nelr];
			density_nb = variables[dAdd(nb, dMul(VAR_DENSITY, nelr))];
			momentum_nb.x = variables[dAdd(nb, dMul(dAdd(VAR_MOMENTUM, 0), nelr))];
			momentum_nb.y = variables[dAdd(nb, dMul(dAdd(VAR_MOMENTUM, 1), nelr))];
			momentum_nb.z = variables[dAdd(nb, dMul(dAdd(VAR_MOMENTUM, 2), nelr))];
			density_energy_nb = variables[dAdd(nb, dMul(VAR_DENSITY_ENERGY, nelr))];
												compute_velocity(density_nb, momentum_nb, velocity_nb);
			speed_sqd_nb                      = compute_speed_sqd(velocity_nb);
			pressure_nb                       = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
			speed_of_sound_nb                 = compute_speed_of_sound(density_nb, pressure_nb);
			                                    compute_flux_contribution(density_nb, momentum_nb, density_energy_nb, pressure_nb, velocity_nb, flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z, flux_contribution_nb_density_energy);
			
			// artificial viscosity
			//factor = -normal_len*smoothing_coefficient*float(0.5f)*(speed_i + sqrtf(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
			factor = dSub(0, dMul(normal_len, dMul(smoothing_coefficient, dMul(float(0.5f), dAdd(dAdd(speed_i, sqrtf(speed_sqd_nb)), dAdd(speed_of_sound_i, speed_of_sound_nb))))));
			//flux_i_density += factor*(density_i-density_nb);
			flux_i_density = dAdd(flux_i_density, dMul(factor, dSub(density_i, density_nb)));
			//flux_i_density_energy += factor*(density_energy_i-density_energy_nb);
			flux_i_density_energy = dAdd(flux_i_density_energy, dMul(factor, dSub(density_energy_i, density_energy_nb)));
			//flux_i_momentum.x += factor*(momentum_i.x-momentum_nb.x);
			flux_i_momentum.x = dAdd(flux_i_momentum.x, dMul(factor, dSub(momentum_i.x, momentum_nb.x)));
			//flux_i_momentum.y += factor*(momentum_i.y-momentum_nb.y);
			flux_i_momentum.y = dAdd(flux_i_momentum.y, dMul(factor, dSub(momentum_i.y, momentum_nb.y)));
			//flux_i_momentum.z += factor*(momentum_i.z-momentum_nb.z);
			flux_i_momentum.z = dAdd(flux_i_momentum.z, dMul(factor, dSub(momentum_i.z, momentum_nb.z)));

			// accumulate cell-centered fluxes
			//factor = float(0.5f)*normal.x;
			//flux_i_density += factor*(momentum_nb.x+momentum_i.x);
			//flux_i_density_energy += factor*(flux_contribution_nb_density_energy.x+flux_contribution_i_density_energy.x);
			//flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.x+flux_contribution_i_momentum_x.x);
			//flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.x+flux_contribution_i_momentum_y.x);
			//flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.x+flux_contribution_i_momentum_z.x);
			factor = dMul(float(0.5f), normal.x);
			flux_i_density = dAdd(flux_i_density, dMul(factor, dAdd(momentum_nb.x, momentum_i.x)));
			flux_i_density_energy = dAdd(flux_i_density_energy, dMul(factor, dAdd(flux_contribution_nb_density_energy.x, flux_contribution_i_density_energy.x)));
			flux_i_momentum.x = dAdd(flux_i_momentum.x, dMul(factor, dAdd(flux_contribution_nb_momentum_x.x, flux_contribution_i_momentum_x.x)));
			flux_i_momentum.y = dAdd(flux_i_momentum.y, dMul(factor, dAdd(flux_contribution_nb_momentum_y.x, flux_contribution_i_momentum_y.x)));
			flux_i_momentum.z = dAdd(flux_i_momentum.z, dMul(factor, dAdd(flux_contribution_nb_momentum_z.x, flux_contribution_i_momentum_z.x)));
			//factor = float(0.5f)*normal.y;
			//flux_i_density += factor*(momentum_nb.y+momentum_i.y);
			//flux_i_density_energy += factor*(flux_contribution_nb_density_energy.y+flux_contribution_i_density_energy.y);
			//flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.y+flux_contribution_i_momentum_x.y);
			//flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.y+flux_contribution_i_momentum_y.y);
			//flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.y+flux_contribution_i_momentum_z.y);
			factor = dMul(float(0.5f), normal.y);
			flux_i_density = dAdd(flux_i_density, dMul(factor, dAdd(momentum_nb.y, momentum_i.y)));
			flux_i_density_energy = dAdd(flux_i_density_energy, dMul(factor, dAdd(flux_contribution_nb_density_energy.y, flux_contribution_i_density_energy.y)));
			flux_i_momentum.x = dAdd(flux_i_momentum.x, dMul(factor, dAdd(flux_contribution_nb_momentum_x.y, flux_contribution_i_momentum_x.y)));
			flux_i_momentum.y = dAdd(flux_i_momentum.y, dMul(factor, dAdd(flux_contribution_nb_momentum_y.y, flux_contribution_i_momentum_y.y)));
			flux_i_momentum.z = dAdd(flux_i_momentum.z, dMul(factor, dAdd(flux_contribution_nb_momentum_z.y, flux_contribution_i_momentum_z.y)));
			
			//factor = float(0.5f)*normal.z;
			//flux_i_density += factor*(momentum_nb.z+momentum_i.z);
			//flux_i_density_energy += factor*(flux_contribution_nb_density_energy.z+flux_contribution_i_density_energy.z);
			//flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.z+flux_contribution_i_momentum_x.z);
			//flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.z+flux_contribution_i_momentum_y.z);
			//flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.z+flux_contribution_i_momentum_z.z);
			factor = dMul(float(0.5f), normal.z);
			flux_i_density = dAdd(flux_i_density, dMul(factor, dAdd(momentum_nb.z, momentum_i.z)));
			flux_i_density_energy = dAdd(flux_i_density_energy, dMul(factor, dAdd(flux_contribution_nb_density_energy.z, flux_contribution_i_density_energy.z)));
			flux_i_momentum.x = dAdd(flux_i_momentum.x, dMul(factor, dAdd(flux_contribution_nb_momentum_x.z, flux_contribution_i_momentum_x.z)));
			flux_i_momentum.y = dAdd(flux_i_momentum.y, dMul(factor, dAdd(flux_contribution_nb_momentum_y.z, flux_contribution_i_momentum_y.z)));
			flux_i_momentum.z = dAdd(flux_i_momentum.z, dMul(factor, dAdd(flux_contribution_nb_momentum_z.z, flux_contribution_i_momentum_z.z)));
		}
		else if(nb == -1)	// a wing boundary
		{
			//flux_i_momentum.x += normal.x*pressure_i;
			//flux_i_momentum.y += normal.y*pressure_i;
			//flux_i_momentum.z += normal.z*pressure_i;
			flux_i_momentum.x = dAdd(flux_i_momentum.x, dMul(normal.x, pressure_i));
			flux_i_momentum.y = dAdd(flux_i_momentum.y, dMul(normal.y, pressure_i));
			flux_i_momentum.z = dAdd(flux_i_momentum.z, dMul(normal.z, pressure_i));
		}
		else if(nb == -2) // a far field boundary
		{
			//factor = float(0.5f)*normal.x;
			//flux_i_density += factor*(ff_variable[VAR_MOMENTUM+0]+momentum_i.x);
			//flux_i_density_energy += factor*(ff_flux_contribution_density_energy[0].x+flux_contribution_i_density_energy.x);
			//flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x[0].x + flux_contribution_i_momentum_x.x);
			//flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y[0].x + flux_contribution_i_momentum_y.x);
			//flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z[0].x + flux_contribution_i_momentum_z.x);
			factor = dMul(float(0.5f), normal.x);
			flux_i_density = dAdd(flux_i_density, dMul(factor, dAdd(ff_variable[dAdd(VAR_MOMENTUM, 0)], momentum_i.x)));
			flux_i_density_energy = dAdd(flux_i_density_energy, dMul(factor, dAdd(ff_flux_contribution_density_energy[0].x, flux_contribution_i_density_energy.x)));
			flux_i_momentum.x = dAdd(flux_i_momentum.x, dMul(factor, dAdd(ff_flux_contribution_momentum_x[0].x, flux_contribution_i_momentum_x.x)));
			flux_i_momentum.y = dAdd(flux_i_momentum.y, dMul(factor, dAdd(ff_flux_contribution_momentum_y[0].x, flux_contribution_i_momentum_y.x)));
			flux_i_momentum.z = dAdd(flux_i_momentum.z, dMul(factor, dAdd(ff_flux_contribution_momentum_z[0].x, flux_contribution_i_momentum_z.x)));
			
			//factor = float(0.5f)*normal.y;
			//flux_i_density += factor*(ff_variable[VAR_MOMENTUM+1]+momentum_i.y);
			//flux_i_density_energy += factor*(ff_flux_contribution_density_energy[0].y+flux_contribution_i_density_energy.y);
			//flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x[0].y + flux_contribution_i_momentum_x.y);
			//flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y[0].y + flux_contribution_i_momentum_y.y);
			//flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z[0].y + flux_contribution_i_momentum_z.y);
			factor = dMul(float(0.5f), normal.y);
			flux_i_density = dAdd(flux_i_density, dMul(factor, dAdd(ff_variable[dAdd(VAR_MOMENTUM, 1)], momentum_i.y)));
			flux_i_density_energy = dAdd(flux_i_density_energy, dMul(factor, dAdd(ff_flux_contribution_density_energy[0].y, flux_contribution_i_density_energy.y)));
			flux_i_momentum.x = dAdd(flux_i_momentum.x, dMul(factor, dAdd(ff_flux_contribution_momentum_x[0].y, flux_contribution_i_momentum_x.y)));
			flux_i_momentum.y = dAdd(flux_i_momentum.y, dMul(factor, dAdd(ff_flux_contribution_momentum_y[0].y, flux_contribution_i_momentum_y.y)));
			flux_i_momentum.z = dAdd(flux_i_momentum.z, dMul(factor, dAdd(ff_flux_contribution_momentum_z[0].y, flux_contribution_i_momentum_z.y)));

			//factor = float(0.5f)*normal.z;
			//flux_i_density += factor*(ff_variable[VAR_MOMENTUM+2]+momentum_i.z);
			//flux_i_density_energy += factor*(ff_flux_contribution_density_energy[0].z+flux_contribution_i_density_energy.z);
			//flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x[0].z + flux_contribution_i_momentum_x.z);
			//flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y[0].z + flux_contribution_i_momentum_y.z);
			//flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z[0].z + flux_contribution_i_momentum_z.z);
			factor = dMul(float(0.5f), normal.z);
			flux_i_density = dAdd(flux_i_density, dMul(factor, dAdd(ff_variable[dAdd(VAR_MOMENTUM, 2)], momentum_i.z)));
			flux_i_density_energy = dAdd(flux_i_density_energy, dMul(factor, dAdd(ff_flux_contribution_density_energy[0].z, flux_contribution_i_density_energy.z)));
			flux_i_momentum.x = dAdd(flux_i_momentum.x, dMul(factor, dAdd(ff_flux_contribution_momentum_x[0].z, flux_contribution_i_momentum_x.z)));
			flux_i_momentum.y = dAdd(flux_i_momentum.y, dMul(factor, dAdd(ff_flux_contribution_momentum_y[0].z, flux_contribution_i_momentum_y.z)));
			flux_i_momentum.z = dAdd(flux_i_momentum.z, dMul(factor, dAdd(ff_flux_contribution_momentum_z[0].z, flux_contribution_i_momentum_z.z)));

		}
	}

	//fluxes[i + VAR_DENSITY*nelr] = flux_i_density;
	//fluxes[i + (VAR_MOMENTUM+0)*nelr] = flux_i_momentum.x;
	//fluxes[i + (VAR_MOMENTUM+1)*nelr] = flux_i_momentum.y;
	//fluxes[i + (VAR_MOMENTUM+2)*nelr] = flux_i_momentum.z;
	//fluxes[i + VAR_DENSITY_ENERGY*nelr] = flux_i_density_energy;
	fluxes[dAdd(i, dMul(VAR_DENSITY, nelr))] = flux_i_density;
	fluxes[dAdd(i, dMul(dAdd(VAR_MOMENTUM, 0), nelr))] = flux_i_momentum.x;
	fluxes[dAdd(i, dMul(dAdd(VAR_MOMENTUM, 1), nelr))] = flux_i_momentum.y;
	fluxes[dAdd(i, dMul(dAdd(VAR_MOMENTUM, 2), nelr))] = flux_i_momentum.z;
	fluxes[dAdd(i, dMul(VAR_DENSITY_ENERGY, nelr))] = flux_i_density_energy;
}
void compute_flux(int nelr, int* elements_surrounding_elements, float* normals, float* variables, float* fluxes)
{
	dim3 Dg(nelr / BLOCK_SIZE_3), Db(BLOCK_SIZE_3);
	cuda_compute_flux<<<Dg,Db>>>(nelr, elements_surrounding_elements, normals, variables, fluxes);
	getLastCudaError("compute_flux failed");
}

__global__ void cuda_time_step(int j, int nelr, float* old_variables, float* variables, float* step_factors, float* fluxes)
{
	//const int i = (blockDim.x*blockIdx.x + threadIdx.x);
	const int i = dAdd(dMul(blockDim.x, blockIdx.x), threadIdx.x);

	//float factor = step_factors[i]/float(RK+1-j);
	float factor = dDiv(step_factors[i], float(dSub(dAdd(RK,1), j)));

	//variables[i + VAR_DENSITY*nelr] = old_variables[i + VAR_DENSITY*nelr] + factor*fluxes[i + VAR_DENSITY*nelr];
	//variables[i + VAR_DENSITY_ENERGY*nelr] = old_variables[i + VAR_DENSITY_ENERGY*nelr] + factor*fluxes[i + VAR_DENSITY_ENERGY*nelr];
	//variables[i + (VAR_MOMENTUM+0)*nelr] = old_variables[i + (VAR_MOMENTUM+0)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+0)*nelr];
	//variables[i + (VAR_MOMENTUM+1)*nelr] = old_variables[i + (VAR_MOMENTUM+1)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+1)*nelr];	
	//variables[i + (VAR_MOMENTUM+2)*nelr] = old_variables[i + (VAR_MOMENTUM+2)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+2)*nelr];	
	variables[dAdd(i, dMul(VAR_DENSITY, nelr))] = dAdd(old_variables[dAdd(i, dMul(VAR_DENSITY, nelr))], dMul(factor, fluxes[dAdd(i, dMul(VAR_DENSITY, nelr))]));
	variables[dAdd(i, dMul(VAR_DENSITY_ENERGY, nelr))] = dAdd(old_variables[dAdd(i, dMul(VAR_DENSITY_ENERGY, nelr))], dMul(factor, fluxes[dAdd(i, dMul(VAR_DENSITY_ENERGY, nelr))]));
	variables[dAdd(i, dMul(dAdd(VAR_MOMENTUM, 0), nelr))] = dAdd(old_variables[dAdd(i, dMul(dAdd(VAR_MOMENTUM, 0), nelr))], dMul(factor, fluxes[dAdd(i, dMul(dAdd(VAR_MOMENTUM, 0), nelr))]));
	variables[dAdd(i, dMul(dAdd(VAR_MOMENTUM, 1), nelr))] = dAdd(old_variables[dAdd(i, dMul(dAdd(VAR_MOMENTUM, 1), nelr))], dMul(factor, fluxes[dAdd(i, dMul(dAdd(VAR_MOMENTUM, 1), nelr))]));	
	variables[dAdd(i, dMul(dAdd(VAR_MOMENTUM, 2), nelr))] = dAdd(old_variables[dAdd(i, dMul(dAdd(VAR_MOMENTUM, 2), nelr))], dMul(factor, fluxes[dAdd(i, dMul(dAdd(VAR_MOMENTUM, 2), nelr))]));	
}
void time_step(int j, int nelr, float* old_variables, float* variables, float* step_factors, float* fluxes)
{
	dim3 Dg(nelr / BLOCK_SIZE_4), Db(BLOCK_SIZE_4);
	cuda_time_step<<<Dg,Db>>>(j, nelr, old_variables, variables, step_factors, fluxes);
	getLastCudaError("update failed");
}

/************************************************************
*************************ORIGINAL CODE***********************
************************************************************/

__global__ void original_cuda_initialize_variables(int nelr, float* variables)
{
	const int i = (blockDim.x*blockIdx.x + threadIdx.x);
	for(int j = 0; j < NVAR; j++)
		variables[i + j*nelr] = original_ff_variable[j];
}
void original_initialize_variables(int nelr, float* variables)
{
	dim3 Dg(nelr / BLOCK_SIZE_1), Db(BLOCK_SIZE_1);
	original_cuda_initialize_variables<<<Dg, Db>>>(nelr, variables);
	getLastCudaError("initialize_variables failed");
}

__device__ __host__ inline void original_compute_flux_contribution(float& density, float3& momentum, float& density_energy, float& pressure, float3& velocity, float3& fc_momentum_x, float3& fc_momentum_y, float3& fc_momentum_z, float3& fc_density_energy)
{
	fc_momentum_x.x = velocity.x*momentum.x + pressure;
	fc_momentum_x.y = velocity.x*momentum.y;
	fc_momentum_x.z = velocity.x*momentum.z;
	
	
	fc_momentum_y.x = fc_momentum_x.y;
	fc_momentum_y.y = velocity.y*momentum.y + pressure;
	fc_momentum_y.z = velocity.y*momentum.z;

	fc_momentum_z.x = fc_momentum_x.z;
	fc_momentum_z.y = fc_momentum_y.z;
	fc_momentum_z.z = velocity.z*momentum.z + pressure;

	float de_p = density_energy+pressure;
	fc_density_energy.x = velocity.x*de_p;
	fc_density_energy.y = velocity.y*de_p;
	fc_density_energy.z = velocity.z*de_p;
}

__device__ inline void original_compute_velocity(float& density, float3& momentum, float3& velocity)
{
	velocity.x = momentum.x / density;
	velocity.y = momentum.y / density;
	velocity.z = momentum.z / density;
}
	
__device__ inline float original_compute_speed_sqd(float3& velocity)
{
	return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}

__device__ inline float original_compute_pressure(float& density, float& density_energy, float& speed_sqd)
{
	return (float(GAMMA)-float(1.0f))*(density_energy - float(0.5f)*density*speed_sqd);
}

__device__ inline float original_compute_speed_of_sound(float& density, float& pressure)
{
	return sqrtf(float(GAMMA)*pressure/density);
}

__global__ void original_cuda_compute_step_factor(int nelr, float* variables, float* areas, float* step_factors)
{
	const int i = (blockDim.x*blockIdx.x + threadIdx.x);

	float density = variables[i + VAR_DENSITY*nelr];
	float3 momentum;
	momentum.x = variables[i + (VAR_MOMENTUM+0)*nelr];
	momentum.y = variables[i + (VAR_MOMENTUM+1)*nelr];
	momentum.z = variables[i + (VAR_MOMENTUM+2)*nelr];
	
	float density_energy = variables[i + VAR_DENSITY_ENERGY*nelr];
	
	float3 velocity;       original_compute_velocity(density, momentum, velocity);
	float speed_sqd      = original_compute_speed_sqd(velocity);
	float pressure       = original_compute_pressure(density, density_energy, speed_sqd);
	float speed_of_sound = original_compute_speed_of_sound(density, pressure);

	// dt = float(0.5f) * sqrtf(areas[i]) /  (||v|| + c).... but when we do time stepping, this later would need to be divided by the area, so we just do it all at once
	step_factors[i] = float(0.5f) / (sqrtf(areas[i]) * (sqrtf(speed_sqd) + speed_of_sound));
}
void original_compute_step_factor(int nelr, float* variables, float* areas, float* step_factors)
{
	dim3 Dg(nelr / BLOCK_SIZE_2), Db(BLOCK_SIZE_2);
	cuda_compute_step_factor<<<Dg, Db>>>(nelr, variables, areas, step_factors);		
	getLastCudaError("compute_step_factor failed");
}

/*
 *
 *
*/
__global__ void original_cuda_compute_flux(int nelr, int* elements_surrounding_elements, float* normals, float* variables, float* fluxes)
{
	const float smoothing_coefficient = float(0.2f);
	const int i = (blockDim.x*blockIdx.x + threadIdx.x);
	
	int j, nb;
	float3 normal; float normal_len;
	float factor;
	
	float density_i = variables[i + VAR_DENSITY*nelr];
	float3 momentum_i;
	momentum_i.x = variables[i + (VAR_MOMENTUM+0)*nelr];
	momentum_i.y = variables[i + (VAR_MOMENTUM+1)*nelr];
	momentum_i.z = variables[i + (VAR_MOMENTUM+2)*nelr];

	float density_energy_i = variables[i + VAR_DENSITY_ENERGY*nelr];

	float3 velocity_i;             				original_compute_velocity(density_i, momentum_i, velocity_i);
	float speed_sqd_i                          = original_compute_speed_sqd(velocity_i);
	float speed_i                              = sqrtf(speed_sqd_i);
	float pressure_i                           = original_compute_pressure(density_i, density_energy_i, speed_sqd_i);
	float speed_of_sound_i                     = original_compute_speed_of_sound(density_i, pressure_i);
	float3 flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z;
	float3 flux_contribution_i_density_energy;	
	original_compute_flux_contribution(density_i, momentum_i, density_energy_i, pressure_i, velocity_i, flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z, flux_contribution_i_density_energy);
	
	float flux_i_density = float(0.0f);
	float3 flux_i_momentum;
	flux_i_momentum.x = float(0.0f);
	flux_i_momentum.y = float(0.0f);
	flux_i_momentum.z = float(0.0f);
	float flux_i_density_energy = float(0.0f);
		
	float3 velocity_nb;
	float density_nb, density_energy_nb;
	float3 momentum_nb;
	float3 flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z;
	float3 flux_contribution_nb_density_energy;	
	float speed_sqd_nb, speed_of_sound_nb, pressure_nb;
	
	#pragma unroll
	for(j = 0; j < NNB; j++)
	{
		nb = elements_surrounding_elements[i + j*nelr];
		normal.x = normals[i + (j + 0*NNB)*nelr];
		normal.y = normals[i + (j + 1*NNB)*nelr];
		normal.z = normals[i + (j + 2*NNB)*nelr];
		normal_len = sqrtf(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
		
		if(nb >= 0) 	// a legitimate neighbor
		{
			density_nb = variables[nb + VAR_DENSITY*nelr];
			momentum_nb.x = variables[nb + (VAR_MOMENTUM+0)*nelr];
			momentum_nb.y = variables[nb + (VAR_MOMENTUM+1)*nelr];
			momentum_nb.z = variables[nb + (VAR_MOMENTUM+2)*nelr];
			density_energy_nb = variables[nb + VAR_DENSITY_ENERGY*nelr];
												original_compute_velocity(density_nb, momentum_nb, velocity_nb);
			speed_sqd_nb                      = original_compute_speed_sqd(velocity_nb);
			pressure_nb                       = original_compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
			speed_of_sound_nb                 = original_compute_speed_of_sound(density_nb, pressure_nb);
			                                    original_compute_flux_contribution(density_nb, momentum_nb, density_energy_nb, pressure_nb, velocity_nb, flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z, flux_contribution_nb_density_energy);
			
			// artificial viscosity
			factor = -normal_len*smoothing_coefficient*float(0.5f)*(speed_i + sqrtf(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
			flux_i_density += factor*(density_i-density_nb);
			flux_i_density_energy += factor*(density_energy_i-density_energy_nb);
			flux_i_momentum.x += factor*(momentum_i.x-momentum_nb.x);
			flux_i_momentum.y += factor*(momentum_i.y-momentum_nb.y);
			flux_i_momentum.z += factor*(momentum_i.z-momentum_nb.z);

			// accumulate cell-centered fluxes
			factor = float(0.5f)*normal.x;
			flux_i_density += factor*(momentum_nb.x+momentum_i.x);
			flux_i_density_energy += factor*(flux_contribution_nb_density_energy.x+flux_contribution_i_density_energy.x);
			flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.x+flux_contribution_i_momentum_x.x);
			flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.x+flux_contribution_i_momentum_y.x);
			flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.x+flux_contribution_i_momentum_z.x);
			
			factor = float(0.5f)*normal.y;
			flux_i_density += factor*(momentum_nb.y+momentum_i.y);
			flux_i_density_energy += factor*(flux_contribution_nb_density_energy.y+flux_contribution_i_density_energy.y);
			flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.y+flux_contribution_i_momentum_x.y);
			flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.y+flux_contribution_i_momentum_y.y);
			flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.y+flux_contribution_i_momentum_z.y);
			
			factor = float(0.5f)*normal.z;
			flux_i_density += factor*(momentum_nb.z+momentum_i.z);
			flux_i_density_energy += factor*(flux_contribution_nb_density_energy.z+flux_contribution_i_density_energy.z);
			flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.z+flux_contribution_i_momentum_x.z);
			flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.z+flux_contribution_i_momentum_y.z);
			flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.z+flux_contribution_i_momentum_z.z);
		}
		else if(nb == -1)	// a wing boundary
		{
			flux_i_momentum.x += normal.x*pressure_i;
			flux_i_momentum.y += normal.y*pressure_i;
			flux_i_momentum.z += normal.z*pressure_i;
		}
		else if(nb == -2) // a far field boundary
		{
			factor = float(0.5f)*normal.x;
			flux_i_density += factor*(original_ff_variable[VAR_MOMENTUM+0]+momentum_i.x);
			flux_i_density_energy += factor*(original_ff_flux_contribution_density_energy[0].x+flux_contribution_i_density_energy.x);
			flux_i_momentum.x += factor*(original_ff_flux_contribution_momentum_x[0].x + flux_contribution_i_momentum_x.x);
			flux_i_momentum.y += factor*(original_ff_flux_contribution_momentum_y[0].x + flux_contribution_i_momentum_y.x);
			flux_i_momentum.z += factor*(original_ff_flux_contribution_momentum_z[0].x + flux_contribution_i_momentum_z.x);
			
			factor = float(0.5f)*normal.y;
			flux_i_density += factor*(original_ff_variable[VAR_MOMENTUM+1]+momentum_i.y);
			flux_i_density_energy += factor*(original_ff_flux_contribution_density_energy[0].y+flux_contribution_i_density_energy.y);
			flux_i_momentum.x += factor*(original_ff_flux_contribution_momentum_x[0].y + flux_contribution_i_momentum_x.y);
			flux_i_momentum.y += factor*(original_ff_flux_contribution_momentum_y[0].y + flux_contribution_i_momentum_y.y);
			flux_i_momentum.z += factor*(original_ff_flux_contribution_momentum_z[0].y + flux_contribution_i_momentum_z.y);

			factor = float(0.5f)*normal.z;
			flux_i_density += factor*(original_ff_variable[VAR_MOMENTUM+2]+momentum_i.z);
			flux_i_density_energy += factor*(original_ff_flux_contribution_density_energy[0].z+flux_contribution_i_density_energy.z);
			flux_i_momentum.x += factor*(original_ff_flux_contribution_momentum_x[0].z + flux_contribution_i_momentum_x.z);
			flux_i_momentum.y += factor*(original_ff_flux_contribution_momentum_y[0].z + flux_contribution_i_momentum_y.z);
			flux_i_momentum.z += factor*(original_ff_flux_contribution_momentum_z[0].z + flux_contribution_i_momentum_z.z);

		}
	}

	fluxes[i + VAR_DENSITY*nelr] = flux_i_density;
	fluxes[i + (VAR_MOMENTUM+0)*nelr] = flux_i_momentum.x;
	fluxes[i + (VAR_MOMENTUM+1)*nelr] = flux_i_momentum.y;
	fluxes[i + (VAR_MOMENTUM+2)*nelr] = flux_i_momentum.z;
	fluxes[i + VAR_DENSITY_ENERGY*nelr] = flux_i_density_energy;
}
void original_compute_flux(int nelr, int* elements_surrounding_elements, float* normals, float* variables, float* fluxes)
{
	dim3 Dg(nelr / BLOCK_SIZE_3), Db(BLOCK_SIZE_3);
	original_cuda_compute_flux<<<Dg,Db>>>(nelr, elements_surrounding_elements, normals, variables, fluxes);
	getLastCudaError("compute_flux failed");
}

__global__ void original_cuda_time_step(int j, int nelr, float* old_variables, float* variables, float* step_factors, float* fluxes)
{
	const int i = (blockDim.x*blockIdx.x + threadIdx.x);

	float factor = step_factors[i]/float(RK+1-j);

	variables[i + VAR_DENSITY*nelr] = old_variables[i + VAR_DENSITY*nelr] + factor*fluxes[i + VAR_DENSITY*nelr];
	variables[i + VAR_DENSITY_ENERGY*nelr] = old_variables[i + VAR_DENSITY_ENERGY*nelr] + factor*fluxes[i + VAR_DENSITY_ENERGY*nelr];
	variables[i + (VAR_MOMENTUM+0)*nelr] = old_variables[i + (VAR_MOMENTUM+0)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+0)*nelr];
	variables[i + (VAR_MOMENTUM+1)*nelr] = old_variables[i + (VAR_MOMENTUM+1)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+1)*nelr];	
	variables[i + (VAR_MOMENTUM+2)*nelr] = old_variables[i + (VAR_MOMENTUM+2)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+2)*nelr];	
}
void original_time_step(int j, int nelr, float* old_variables, float* variables, float* step_factors, float* fluxes)
{
	dim3 Dg(nelr / BLOCK_SIZE_4), Db(BLOCK_SIZE_4);
	original_cuda_time_step<<<Dg,Db>>>(j, nelr, old_variables, variables, step_factors, fluxes);
	getLastCudaError("update failed");
}


/* 
 * compare function
 */


struct Error_data{
	double max_abs = 0.0;
	double max_rel = 0.0;
	int max_index = -1;
	int max_rel_index = -1;
	int error_count  = 0;
	double avg_err = 0.0;
	double RMSE = 0.0;
};

struct Result_data{
	Error_data density;
	Error_data mom_x;
	Error_data mom_y;
	Error_data mom_z;
	Error_data density_energy;
	Error_data flux_density;
	Error_data flux_mom_x;
	Error_data flux_mom_y;
	Error_data flux_mom_z;
	Error_data flux_density_energy;
	Error_data step_factor;
};

void compute_error(Error_data& error, std::vector<float> &v, std::vector<float> &ov, int index, int nelr, int ofset)
{
	double o_value = ov[index + ofset*nelr];
	double err = fabs(v[index + ofset*nelr] - o_value);
	double rel_err = err / std::max(fabs(o_value), 1e-20);
	if(err > error.max_abs)
	{
		error.max_abs = err;
		error.max_index = index;
	}
	if(rel_err > error.max_rel)
	{
		error.max_rel = rel_err;
		error.max_rel_index = index;
	}
	if(rel_err > 0.01f)
	{
		error.error_count++;
	}
	error.avg_err += rel_err;
	error.RMSE += rel_err*rel_err;
}

void finalize_error(Error_data& error, int nelr)
{
	error.avg_err /= nelr;
	error.RMSE /= nelr;
	error.RMSE = sqrt(error.RMSE);
}

void write_error_row(std::ofstream &file, std::string_view field_name, const Error_data &e)
{
    file << field_name << ","
         << e.max_abs << ","
         << e.max_rel << ","
         << e.max_index << ","
         << e.max_rel_index << ","
         << e.error_count << ","
         << e.avg_err << ","
         << e.RMSE << "\n";
}

void save_results(const Result_data &result, string &filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        printf("ERROR: could not open %s for writing\n", filename.c_str());
        return;
    }

    file << "field,max_abs,max_rel,max_index,max_rel_index,error_count,avg_err,RMSE\n";

    write_error_row(file, "density",           result.density);
    write_error_row(file, "momentum_x",        result.mom_x);
    write_error_row(file, "momentum_y",        result.mom_y);
    write_error_row(file, "momentum_z",        result.mom_z);
    write_error_row(file, "density_energy",    result.density_energy);
    write_error_row(file, "flux_density",         result.flux_density);
    write_error_row(file, "flux_momentum_x",      result.flux_mom_x);
    write_error_row(file, "flux_momentum_y",      result.flux_mom_y);
    write_error_row(file, "flux_momentum_z",      result.flux_mom_z);
    write_error_row(file, "flux_density_energy",  result.flux_density_energy);
    write_error_row(file, "step_factor",          result.step_factor);

    file.close();
}

void compare(float* variables, float* original_variables, float* fluxes, float* original_fluxes, float* step_factors, float* original_step_factors, int nelr, string &output_file){
	cudaDeviceSynchronize();
	std::vector<float> v(nelr * NVAR);
    std::vector<float> ov(nelr * NVAR);
    std::vector<float> f(nelr * NVAR);
    std::vector<float> of(nelr * NVAR);
    std::vector<float> sf(nelr);
    std::vector<float> osf(nelr);

    download(v.data(), variables, nelr * NVAR);
    download(ov.data(), original_variables, nelr * NVAR);
    download(f.data(), fluxes, nelr * NVAR);
    download(of.data(), original_fluxes, nelr * NVAR);
    download(sf.data(), step_factors, nelr);
    download(osf.data(), original_step_factors, nelr);
	Result_data result;


	for(int i = 0; i < nelr; ++i)
	{
		compute_error(result.density, v, ov, i, nelr, VAR_DENSITY);
		compute_error(result.mom_x, v, ov, i, nelr, VAR_MOMENTUM+0);
		compute_error(result.mom_y, v, ov, i, nelr, VAR_MOMENTUM+1);
		compute_error(result.mom_z, v, ov, i, nelr, VAR_MOMENTUM+2);
		compute_error(result.density_energy, v, ov, i, nelr, VAR_DENSITY_ENERGY);
		compute_error(result.flux_density, f, of, i, nelr, VAR_DENSITY);
		compute_error(result.flux_mom_x, f, of, i, nelr, VAR_MOMENTUM+0);
		compute_error(result.flux_mom_y, f, of, i, nelr, VAR_MOMENTUM+1);
		compute_error(result.flux_mom_z, f, of, i, nelr, VAR_MOMENTUM+2);
		compute_error(result.flux_density_energy, f, of, i, nelr, VAR_DENSITY_ENERGY);
		compute_error(result.step_factor, sf, osf, i, nelr, 0);
	}
	finalize_error(result.density, nelr);
	finalize_error(result.mom_x, nelr);
	finalize_error(result.mom_y, nelr);
	finalize_error(result.mom_z, nelr);
	finalize_error(result.density_energy, nelr);
	finalize_error(result.flux_density, nelr);
	finalize_error(result.flux_mom_x, nelr);
	finalize_error(result.flux_mom_y, nelr);
	finalize_error(result.flux_mom_z, nelr);
	finalize_error(result.flux_density_energy, nelr);
	finalize_error(result.step_factor, nelr);
	save_results(result, output_file);
}


void save_times(float step_factor_time, float o_step_factor_time, float flux_time, float o_flux_time, float time_step_time, float o_time_step_time, string &output_file){
	std::ofstream file(output_file);
    if (!file.is_open())
    {
        printf("ERROR: could not open %s for writing\n", output_file.c_str());
        return;
    }
	float over_head_sf = ((step_factor_time - o_step_factor_time) / step_factor_time) * 100;
	float over_head_f = ((flux_time - o_flux_time) / o_flux_time) * 100;
	float over_head_ts = ((time_step_time - o_time_step_time) / o_time_step_time) * 100;

	file << "time k1: " << step_factor_time << "\noriginal time k1: " << o_step_factor_time << "\noverhead k1: " << over_head_sf
		<< "\ntime k2: " << flux_time << "\noriginal time k2: " << o_flux_time << "\noverhead k2: " << over_head_f
		<< "\ntime k3: " << time_step_time << "\noriginal time k3: " << o_time_step_time << "\noverhead k3: " << over_head_ts << "\n";
}

/*
 * Main function
 */
int main(int argc, char** argv)
{
  printf("WG size of kernel:initialize = %d, WG size of kernel:compute_step_factor = %d, WG size of kernel:compute_flux = %d, WG size of kernel:time_step = %d\n", BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCK_SIZE_3, BLOCK_SIZE_4);

	if (argc < 2)
	{
		std::cout << "specify data file name" << std::endl;
		return 0;
	}
	const char* data_file_name = argv[1];
	
	cudaDeviceProp prop;
	int dev;
	
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaGetDevice(&dev));
	checkCudaErrors(cudaGetDeviceProperties(&prop, dev));
	
	printf("Name:                     %s\n", prop.name);

	// set far field conditions and load them into constant memory on the gpu
	{
		float h_ff_variable[NVAR];
		const float angle_of_attack = float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);
		
		h_ff_variable[VAR_DENSITY] = float(1.4);
		
		float ff_pressure = float(1.0f);
		float ff_speed_of_sound = sqrt(GAMMA*ff_pressure / h_ff_variable[VAR_DENSITY]);
		float ff_speed = float(ff_mach)*ff_speed_of_sound;
		
		float3 ff_velocity;
		ff_velocity.x = ff_speed*float(cos((float)angle_of_attack));
		ff_velocity.y = ff_speed*float(sin((float)angle_of_attack));
		ff_velocity.z = 0.0f;
		
		h_ff_variable[VAR_MOMENTUM+0] = h_ff_variable[VAR_DENSITY] * ff_velocity.x;
		h_ff_variable[VAR_MOMENTUM+1] = h_ff_variable[VAR_DENSITY] * ff_velocity.y;
		h_ff_variable[VAR_MOMENTUM+2] = h_ff_variable[VAR_DENSITY] * ff_velocity.z;
				
		h_ff_variable[VAR_DENSITY_ENERGY] = h_ff_variable[VAR_DENSITY]*(float(0.5f)*(ff_speed*ff_speed)) + (ff_pressure / float(GAMMA-1.0f));

		float3 h_ff_momentum;
		h_ff_momentum.x = *(h_ff_variable+VAR_MOMENTUM+0);
		h_ff_momentum.y = *(h_ff_variable+VAR_MOMENTUM+1);
		h_ff_momentum.z = *(h_ff_variable+VAR_MOMENTUM+2);
		float3 h_ff_flux_contribution_momentum_x;
		float3 h_ff_flux_contribution_momentum_y;
		float3 h_ff_flux_contribution_momentum_z;
		float3 h_ff_flux_contribution_density_energy;
		compute_flux_contribution(h_ff_variable[VAR_DENSITY], h_ff_momentum, h_ff_variable[VAR_DENSITY_ENERGY], ff_pressure, ff_velocity, h_ff_flux_contribution_momentum_x, h_ff_flux_contribution_momentum_y, h_ff_flux_contribution_momentum_z, h_ff_flux_contribution_density_energy);

		// copy far field conditions to the gpu
		checkCudaErrors( cudaMemcpyToSymbol(ff_variable,          h_ff_variable,          NVAR*sizeof(float)) );
		checkCudaErrors( cudaMemcpyToSymbol(ff_flux_contribution_momentum_x, &h_ff_flux_contribution_momentum_x, sizeof(float3)) );
		checkCudaErrors( cudaMemcpyToSymbol(ff_flux_contribution_momentum_y, &h_ff_flux_contribution_momentum_y, sizeof(float3)) );
		checkCudaErrors( cudaMemcpyToSymbol(ff_flux_contribution_momentum_z, &h_ff_flux_contribution_momentum_z, sizeof(float3)) );
		checkCudaErrors( cudaMemcpyToSymbol(ff_flux_contribution_density_energy, &h_ff_flux_contribution_density_energy, sizeof(float3)) );	

		checkCudaErrors(cudaMemcpyToSymbol(original_ff_variable, h_ff_variable, NVAR * sizeof(float)));
		checkCudaErrors(cudaMemcpyToSymbol(original_ff_flux_contribution_momentum_x, &h_ff_flux_contribution_momentum_x, sizeof(float3)));
		checkCudaErrors(cudaMemcpyToSymbol(original_ff_flux_contribution_momentum_y, &h_ff_flux_contribution_momentum_y, sizeof(float3)));
		checkCudaErrors(cudaMemcpyToSymbol(original_ff_flux_contribution_momentum_z, &h_ff_flux_contribution_momentum_z, sizeof(float3)));
		checkCudaErrors(cudaMemcpyToSymbol(original_ff_flux_contribution_density_energy, &h_ff_flux_contribution_density_energy, sizeof(float3)));

	}
	
	int nel;
	int nelr;
	
	// read in domain geometry
	float* areas;
	float* original_areas;
	int* elements_surrounding_elements;
	int* original_elements_surrounding_elements;
	float* normals;
	float* original_normals;
	{
		std::ifstream file(data_file_name);
	
		file >> nel;
		nelr = BLOCK_SIZE_0*((nel / BLOCK_SIZE_0 )+ std::min(1, nel % BLOCK_SIZE_0));

		float* h_areas = new float[nelr];
		int* h_elements_surrounding_elements = new int[nelr*NNB];
		float* h_normals = new float[nelr*NDIM*NNB];
				
		// read in data
		for(int i = 0; i < nel; i++)
		{
			file >> h_areas[i];
			for(int j = 0; j < NNB; j++)
			{
				file >> h_elements_surrounding_elements[i + j*nelr];
				if(h_elements_surrounding_elements[i+j*nelr] < 0) h_elements_surrounding_elements[i+j*nelr] = -1;
				h_elements_surrounding_elements[i + j*nelr]--; //it's coming in with Fortran numbering				
				
				for(int k = 0; k < NDIM; k++)
				{
					file >> h_normals[i + (j + k*NNB)*nelr];
					h_normals[i + (j + k*NNB)*nelr] = -h_normals[i + (j + k*NNB)*nelr];
				}
			}
		}
		
		// fill in remaining data
		int last = nel-1;
		for(int i = nel; i < nelr; i++)
		{
			h_areas[i] = h_areas[last];
			for(int j = 0; j < NNB; j++)
			{
				// duplicate the last element
				h_elements_surrounding_elements[i + j*nelr] = h_elements_surrounding_elements[last + j*nelr];	
				for(int k = 0; k < NDIM; k++) h_normals[last + (j + k*NNB)*nelr] = h_normals[last + (j + k*NNB)*nelr];
			}
		}
		
		areas = alloc<float>(nelr);
		original_areas = alloc<float>(nelr);
		upload<float>(areas, h_areas, nelr);
		upload<float>(original_areas, h_areas, nelr);

		elements_surrounding_elements = alloc<int>(nelr*NNB);
		original_elements_surrounding_elements = alloc<int>(nelr*NNB);
		upload<int>(elements_surrounding_elements, h_elements_surrounding_elements, nelr*NNB);
		upload<int>(original_elements_surrounding_elements, h_elements_surrounding_elements, nelr*NNB);

		normals = alloc<float>(nelr*NDIM*NNB);
		original_normals = alloc<float>(nelr*NDIM*NNB);
		upload<float>(normals, h_normals, nelr*NDIM*NNB);
		upload<float>(original_normals, h_normals, nelr*NDIM*NNB);
				
		delete[] h_areas;
		delete[] h_elements_surrounding_elements;
		delete[] h_normals;
	}

	// Create arrays and set initial conditions
	float* variables = alloc<float>(nelr*NVAR);
	float* original_variables = alloc<float>(nelr*NVAR);
	initialize_variables(nelr, variables);
	original_initialize_variables(nelr, original_variables);

	float* old_variables = alloc<float>(nelr*NVAR);   	
	float* fluxes = alloc<float>(nelr*NVAR);
	float* step_factors = alloc<float>(nelr); 
	float* original_old_variables = alloc<float>(nelr*NVAR);   	
	float* original_fluxes = alloc<float>(nelr*NVAR);
	float* original_step_factors = alloc<float>(nelr); 

	cudaEvent_t startE, stopE;
    cudaEventCreate(&startE);
    cudaEventCreate(&stopE);

	float elapsed, elapsed_step_factor=0, o_elapsed_step_factor=0, elapsed_flux=0, o_elapsed_flux=0, elapsed_time_step=0, o_elapsed_time_step=0;
	float avg_elapsed_step_factor, avg_o_elapsed_step_factor, avg_elapsed_flux, avg_o_elapsed_flux, avg_elapsed_time_step, avg_o_elapsed_time_step;

	// make sure all memory is floatly allocated before we start timing
	initialize_variables(nelr, old_variables);
	initialize_variables(nelr, fluxes);
	original_initialize_variables(nelr, original_old_variables);
	original_initialize_variables(nelr, original_fluxes);
	cudaMemset( (void*) step_factors, 0, sizeof(float)*nelr );
	cudaMemset( (void*) original_step_factors, 0, sizeof(float)*nelr );
	// make sure CUDA isn't still doing something before we start timing
	cudaDeviceSynchronize();

	// these need to be computed the first time in order to compute time step
	std::cout << "Starting..." << std::endl;

	StopWatchInterface *timer = 0;
	  //	unsigned int timer = 0;

	// CUT_SAFE_CALL( cutCreateTimer( &timer));
	// CUT_SAFE_CALL( cutStartTimer( timer));
	sdkCreateTimer(&timer); 
	sdkStartTimer(timer); 
	// Begin iterations
	//Here we do the time stepping


	
	for(int i = 0; i < iterations; i++)
	{
		copy<float>(old_variables, variables, nelr*NVAR);
		
		// for the first iteration we compute the time step
		cudaEventRecord(startE);
		compute_step_factor(nelr, variables, areas, step_factors);
		cudaEventRecord(stopE);
		cudaEventSynchronize(stopE);
		cudaEventElapsedTime(&elapsed, startE, stopE);
		elapsed_step_factor += elapsed;

		getLastCudaError("compute_step_factor failed");

		
		for(int j = 0; j < RK; j++)
		{
            cudaEventRecord(startE);
			compute_flux(nelr, elements_surrounding_elements, normals, variables, fluxes);
			cudaEventRecord(stopE);
			cudaEventSynchronize(stopE);
			cudaEventElapsedTime(&elapsed, startE, stopE);
			elapsed_flux +=elapsed;

			getLastCudaError("compute_flux failed");	
			
            cudaEventRecord(startE);
			time_step(j, nelr, old_variables, variables, step_factors, fluxes);
			cudaEventRecord(stopE);
			cudaEventSynchronize(stopE);
			cudaEventElapsedTime(&elapsed, startE, stopE);
			elapsed_time_step += elapsed;
		
			getLastCudaError("time_step failed");	
			
		}


		copy<float>(original_old_variables, original_variables, nelr*NVAR);
		
		cudaEventRecord(startE);
		original_compute_step_factor(nelr, original_variables, original_areas, original_step_factors);
		cudaEventRecord(stopE);
		cudaEventSynchronize(stopE);
		cudaEventElapsedTime(&elapsed, startE, stopE);
		o_elapsed_step_factor += elapsed;

		getLastCudaError("original_compute_step_factor failed");

		
		for(int j = 0; j < RK; j++)
		{
            cudaEventRecord(startE);
			original_compute_flux(nelr, original_elements_surrounding_elements, original_normals, original_variables, original_fluxes);
			cudaEventRecord(stopE);
			cudaEventSynchronize(stopE);
			cudaEventElapsedTime(&elapsed, startE, stopE);
			o_elapsed_flux += elapsed;

			getLastCudaError("original_compute_flux failed");	
			
            cudaEventRecord(startE);
			original_time_step(j, nelr, original_old_variables, original_variables, original_step_factors, original_fluxes);
			cudaEventRecord(stopE);
			cudaEventSynchronize(stopE);
			cudaEventElapsedTime(&elapsed, startE, stopE);
			o_elapsed_time_step += elapsed;
		
			getLastCudaError("original_time_step failed");	
		}


		if((i + 1) % CHECKPOINT == 0){
			std::cout << "Iteration: " << i + 1 << std::endl;
			string filename = "results_iter_" + std::to_string((i + 1) / CHECKPOINT) + ".csv";
			string filename_time = "times_iter_" + std::to_string((i + 1) / CHECKPOINT) + ".csv";
			compare(variables, original_variables, fluxes, original_fluxes, step_factors, original_step_factors, nelr, filename);

			avg_elapsed_step_factor = elapsed_step_factor/(i+1);
			avg_elapsed_flux = elapsed_flux/(i+1);
			avg_elapsed_time_step = elapsed_time_step/(i+1);
			avg_o_elapsed_step_factor = o_elapsed_step_factor/(i+1);
			avg_o_elapsed_flux = o_elapsed_flux/(i+1);
			avg_o_elapsed_time_step = o_elapsed_time_step/(i+1);
			save_times(avg_elapsed_step_factor, avg_o_elapsed_step_factor, avg_elapsed_flux, avg_o_elapsed_flux, avg_elapsed_time_step, avg_o_elapsed_time_step, filename_time);

		}

	}

	cudaDeviceSynchronize();
	//	CUT_SAFE_CALL( cutStopTimer(timer) );  
	sdkStopTimer(timer); 

	std::cout  << (sdkGetAverageTimerValue(timer)/1000.0)  / iterations << " seconds per iteration" << std::endl;


	std::cout << "Saving solution..." << std::endl;
	dump(variables, nel, nelr);
	std::cout << "Saved solution..." << std::endl;

	
	std::cout << "Cleaning up..." << std::endl;
	
	cudaEventDestroy(startE);
	cudaEventDestroy(stopE);
	dealloc<float>(areas);
	dealloc<int>(elements_surrounding_elements);
	dealloc<float>(normals);
	dealloc<float>(original_areas);
	dealloc<int>(original_elements_surrounding_elements);
	dealloc<float>(original_normals);
	
	dealloc<float>(variables);
	dealloc<float>(old_variables);
	dealloc<float>(fluxes);
	dealloc<float>(step_factors);
	dealloc<float>(original_variables);
	dealloc<float>(original_old_variables);
	dealloc<float>(original_fluxes);
	dealloc<float>(original_step_factors);

	std::cout << "Done..." << std::endl;

	return 0;
}
