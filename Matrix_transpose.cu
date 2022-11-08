#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <cstdlib> 
#include <iostream>
#include <ctime>


#define TILE_WIDTH 32
#define BLOCK_m 8


using namespace std;

void checkResult(float *hostRef, float *gpuRef, int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;
	for (int i = 0; i < N; i++) {
		if (abs(hostRef[i] - gpuRef[i] > epsilon)) {
			match = 0;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at current %u\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}
	if (match) printf("Arrays match! \n");
}


float* create_random_matrix(int row, int col)
{
	float* data;
	size_t dsize = row*col*sizeof(float);
	data = (float*)malloc(dsize);
		// generate different seed for random number
		time_t t;
		srand((unsigned)time(&t));

		for (int i = 0; i < row*col; i++)
		{
			data[i] = (float)(rand() & 0xFF) / 10.0f;
		}
		return data;
	
}



void cpuTransposeMat(float *m1, float *mres, int n, int m, int nreps) {
	for (int r = 0; r < nreps; r++) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (i*(n)+j < m*n && j*(m)+i < m*n) {
					mres[j*(m)+i] = m1[i*(n)+j];
				}
			}
		}
	}
}


__global__ void copy(float *idata, float *odata, int n, int m, int nreps) 
{
	int ix = blockIdx.x * TILE_WIDTH + threadIdx.x;
	int iy = blockIdx.y * TILE_WIDTH + threadIdx.y;

	int index = ix + n * iy;
	for (int r = 0; r < nreps; r++) {
#pragma unroll
		for (int i = 0; i < TILE_WIDTH; i += BLOCK_m) {
			if (index + i*n < n*m) {
				odata[index + i*n] = idata[index + i*n];
			}
		}
	}
}


__global__ void transposeNaive(float *idata, float *odata, int n, int m, int nreps) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	for (int r = 0; r < nreps; r++) {
#pragma unroll
		for (int i = 0; i < TILE_WIDTH; i += BLOCK_m) {
			if (ix*m + (iy + i) < n*m && (iy + i)*n < n*m) {
				int outIndex = ix*m + (iy)+i;
				int inIndex = (iy)*n + ix + i*n;
				if (outIndex < n*m && inIndex < n*m) {
					odata[outIndex] = idata[inIndex];
				}
			}
		}
	}
}


__global__ void transposeSharedMem(float *idata, float *odata, int n, int m, int nreps) {
	__shared__ float tile[TILE_WIDTH][TILE_WIDTH];
	int ix_orig = blockIdx.x * TILE_WIDTH + threadIdx.x;
	int iy_orig = blockIdx.y * TILE_WIDTH + threadIdx.y;

	int inIndex = (iy_orig)*n + ix_orig;

	int ix_shift = blockIdx.y * TILE_WIDTH + threadIdx.x;
	int iy_shift = blockIdx.x * TILE_WIDTH + threadIdx.y;

	int outIndex = iy_shift*m + (ix_shift);

	for (int r = 0; r < nreps; r++)
	{
#pragma unroll
		for (int i = 0; i < TILE_WIDTH; i += BLOCK_m) {
			if (ix_orig < n && iy_orig < m) {
				if (inIndex + (i*n) < n*m && threadIdx.y + i < TILE_WIDTH) {
					tile[threadIdx.y + i][threadIdx.x] = idata[inIndex + (i*n)];
				}
			}
		}
		__syncthreads();
#pragma unroll
		for (int i = 0; i < TILE_WIDTH; i += BLOCK_m) {
			if (ix_shift < m && iy_shift < n) {
				if (outIndex + i*m < n*m && threadIdx.y + i < TILE_WIDTH) {
					odata[outIndex + i*m] = tile[threadIdx.x][threadIdx.y + i];
				}
			}
		}
	}
}


double bandwidth(size_t x, size_t y, size_t type, double time, int reps) {
	return(((x*y * 2 * reps) / (1024.0*1024.0*1024.0)) / time);
}


int main(int argc, char *argv[])
{
	int m = 100;
	int n = 500;
	int reps = 10000;
	double bw;
	double duration;

	printf("Matrix: %d x %d, Reps: %d\n", m, n, reps);
	//CPU
	float *m1 = NULL;
	float *m_cpu = NULL;

	m1 = (float *)malloc((m*n)*sizeof(float));
	m_cpu = (float *)malloc((m*n)*sizeof(float));
	m1 = create_random_matrix(m, n);
	std::clock_t start;
	start = std::clock();
	cpuTransposeMat(m1, m_cpu, n, m, reps);
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "Serial Transpose: " << duration * 1000 << "ms" << '\n';
	bw = bandwidth(m, n, sizeof(float), duration, reps);
	printf("Effective Bandwidth GB/s: %f \n\n", bw);
	printf("\n");


	dim3 dimBlock(TILE_WIDTH, BLOCK_m);
	dim3 dimGrid(ceil((n + dimBlock.x - 1) / dimBlock.x), ceil((m + dimBlock.y - 1) / dimBlock.y));


	//COPY
	float *m_copy = NULL, *m_copy_gpu = NULL;

	cudaMalloc((void **)&m_copy, (m*n)*sizeof(float));
	cudaMalloc((void **)&m_copy_gpu, (m*n)*sizeof(float));

	cudaMemcpy(m_copy, m1, (m*n)*sizeof(float), cudaMemcpyHostToDevice);
	start = std::clock();
	copy << <dimGrid, dimBlock >> >(m_copy, m_copy_gpu, n, m, reps);
	cudaDeviceSynchronize();
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "Paralel Transpose" << duration * 1000 << "ms" << '\n';
	bw = bandwidth(m, n, sizeof(float), duration, reps);
	printf("Effective Bandwidth GB/s: %f \n\n", bw);
	printf("\n");


	float *result = NULL;
	result = (float *)malloc((m*n)*sizeof(float));
	cudaMemcpy(result, m_copy_gpu, (m*n)*sizeof(float), cudaMemcpyDeviceToHost);
	checkResult(m1, result, m*n);

	// NAIVE
	float *m_naive_gpu = NULL;
	if (cudaMalloc((void **)&m_naive_gpu, (m*n)*sizeof(float))) exit(1);

	start = std::clock();
	transposeNaive << <dimGrid, dimBlock >> >(m_copy, m_naive_gpu, n, m, reps);
	cudaDeviceSynchronize();
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "Naive Transpose" << duration * 1000 << "ms" << '\n';
	bw = bandwidth(m, n, sizeof(float), duration, reps);
	printf("Effective Bandwidth GB/s: %f \n\n", bw);
	printf("\n");

	cudaMemcpy(result, m_naive_gpu, (m*n)*sizeof(float), cudaMemcpyDeviceToHost);
	checkResult(m_cpu, result, m*n);




	//SHARED
	float *m_shared_gpu = NULL;
	cudaMalloc((void **)&m_shared_gpu, (m*n)*sizeof(float));

	start = std::clock();
	transposeSharedMem << <dimGrid, dimBlock >> >(m_copy, m_shared_gpu, n, m, reps);
	cudaDeviceSynchronize();
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "Shared Mem Transpose" << duration * 1000 << "ms" << '\n';
	bw = bandwidth(m, n, sizeof(float), duration, reps);
	printf("Effective Bandwidth GB/s: %f \n\n", bw);
	printf("\n");

	cudaMemcpy(result, m_shared_gpu, (m*n)*sizeof(float), cudaMemcpyDeviceToHost);
	checkResult(m_cpu, result, m*n);




	cudaFree(m_shared_gpu);
	cudaFree(m_naive_gpu);
	cudaFree(m_copy);
	cudaFree(m_copy_gpu);


	free(m1);
	free(m_cpu);
	free(result);

	cudaDeviceReset();
	return 0;
}

