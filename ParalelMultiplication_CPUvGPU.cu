
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>


#include <time.h>
#include <stdio.h>
#include <cstdlib> 
#include <iostream>
#include <ctime>
#define TILE_WIDTH 32

const int row1 = 300;
const int col1 = 1000;
const int row2 = 1000;
const int col2 = 500;



using namespace std;

float* create_random_matrix(int row, int col);
void CPUmultiply(float *matrix1, float *matrix2, float *sum);
void print(float* matrix, int row, int col);
void cuda_helper_func(float* matrix1, float* matrix2, float* sum);
void cuda_helper_func2(float* matrix1, float* matrix2, float* sum);
void checkResult(float *hostRef, float *gpuRef);
__global__ void nonSharedKernel(float *d_A, float *d_B, float *d_C)
{
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	if ((i < row1) && (j < col2))
	{
		float tempSum = 0.0;
		for (int k = 0; k<col1; k++)
		{
			tempSum += d_A[i*row1 + k] * d_B[k*col2 + j];
		}
		d_C[i*col1 + j] = tempSum;
	}
}

__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width)
{

	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	// Identify the row and column of the Pd element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;
	// Loop over the Md and Nd tiles required to compute the Pd element
	for (int m = 0; m < Width / TILE_WIDTH; ++m)
	{
		// Coolaborative loading of Md and Nd tiles into shared memory
		Mds[ty][tx] = Md[Row*Width + (m*TILE_WIDTH + tx)];
		Nds[ty][tx] = Nd[Col + (m*TILE_WIDTH + ty)*Width];

		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += Mds[ty][k] * Nds[k][tx];

		__syncthreads();

	}
	Pd[Row*Width + Col] = Pvalue;


}
int main(){

	float* M = create_random_matrix(row1, col1);
	float* N = create_random_matrix(row2, col2);
	float* sum1 = create_random_matrix(row1, col2);
	float* sum2 = create_random_matrix(row1, col2);
	float* sum3 = create_random_matrix(row1, col2);

	float tempSum = 0.0;
	clock_t start, end;
	start = clock();
	CPUmultiply(M, N, sum1);
	end = clock();
	cout << "Time Elapsed for CPU multiplication of 300x1000 with 1000x500 Matrix is: " << (double)(end - start) / CLOCKS_PER_SEC * 1000 << " milliSeconds" << endl;
	
	cuda_helper_func(M, N, sum2);

	cuda_helper_func2(M, N, sum3);
	checkResult(sum1, sum2);
	checkResult(sum1,sum3);


	return 0;

}

void print(float* matrix, int row, int col)
{
	for (int i = 0; i < col; i++)
	{

		for (int j = 0; j < row; j++)
		{

			cout << matrix[i*row + j] << "  ";
		}
		cout << endl;
	}

}

float* create_random_matrix(int row, int col)
{

	float* matrix;
	size_t dsize = row*col*sizeof(float);
	matrix = (float*)malloc(dsize);
	for (int i = 0; i < col; i++)
	{
		for (int j = 0; j < row; j++)
		{
			matrix[i*row + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		}
	}
	return matrix;
}

void checkResult(float *hostRef, float *gpuRef)
{

	double epsilon = 0.5;
	bool match = 1;

	for (int i = 0; i<row1; i++)
	for (int j = 0; j<col1; j++)
	{
		
		if (abs(hostRef[i*col2 + j] - gpuRef[i*col2 + j]) > epsilon )
		{
			

			match = 0;
			
			
			break;
		}
	}

	if (match)
		printf("Arrays match!\n\n");

	return;
}

void CPUmultiply(float *matrix1, float *matrix2, float *sum)
{
	float tempSum = 0;
	for (int i = 0; i < col2; i++)
	{
		for (int j = 0; j < row1; j++)
		{
			tempSum = 0.0f;

			for (int k = 0; k<col1; k++)
			{
				tempSum += matrix1[i*row1 + k] * matrix2[k*col2 + j];
			}


			sum[i*row1 + j] = tempSum;

		}
	}

}

void cuda_helper_func(float* matrix1, float* matrix2, float* sum)
{
	cudaError_t cudaStatus;
	float time;

	float *d_m;
	cudaStatus = cudaMalloc((float**)&d_m, row1*col1*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 1: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaMemcpy(d_m, matrix1, row1*col1*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 2: %s\n", cudaGetErrorString(cudaStatus));
	}

	float *d_n;
	cudaStatus = cudaMalloc((float**)&d_n, row2*col2*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 3: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaMemcpy(d_n, matrix2, row2*col2*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 4: %s\n", cudaGetErrorString(cudaStatus));
	}
	float *d_sum;
	cudaStatus = cudaMalloc((float**)&d_sum, row1*col2*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 3: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaMemcpy(d_sum, sum, row1*col2*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 4: %s\n", cudaGetErrorString(cudaStatus));
	}
	//------------------------------------------------------------------------------

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	dim3 dimBlock(16, 16);
	dim3 dimGrid(ceil((col2 + 16 - 1) / 16), ceil((row1 + 16 - 1) / 16));

	nonSharedKernel << <dimGrid, dimBlock >> >(d_m, d_n, d_sum);

	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "failed 11: %s\n", cudaGetErrorString(cudaStatus));

	cudaDeviceSynchronize();
	(cudaEventRecord(stop, 0));
	(cudaEventSynchronize(stop));
	(cudaEventElapsedTime(&time, start, stop));

	cout << "Time Elapsed for pararlel multiplication without Shared Memory // 300x1000  1000x500 Matrix: " << time << "ms" << '\n';


	cudaStatus = cudaMemcpy(sum, d_sum, row1*col2*sizeof(float), cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 12: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaFree(d_m);
	cudaFree(d_n);
	cudaFree(d_sum);
	cudaDeviceReset();

}

void cuda_helper_func2(float* matrix1, float* matrix2, float* sum)
{
	cudaError_t cudaStatus;
	float time;

	float *d_m;
	cudaStatus = cudaMalloc((float**)&d_m, row1*col1*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 1: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaMemcpy(d_m, matrix1, row1*col1*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 2: %s\n", cudaGetErrorString(cudaStatus));
	}

	float *d_n;
	cudaStatus = cudaMalloc((float**)&d_n, row2*col2*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 3: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaMemcpy(d_n, matrix2, row2*col2*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 4: %s\n", cudaGetErrorString(cudaStatus));
	}
	float *d_sum;
	cudaStatus = cudaMalloc((float**)&d_sum, row1*col2*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 3: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaMemcpy(d_sum, sum, row1*col2*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 4: %s\n", cudaGetErrorString(cudaStatus));
	}
	//------------------------------------------------------------------------------

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	dim3 dimBlock1(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid1(ceil(col1 / TILE_WIDTH), ceil(col1 / TILE_WIDTH));

	MatrixMulKernel << <dimGrid1, dimBlock1 >> >(d_m, d_n, d_sum, col1);

	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "failed 11: %s\n", cudaGetErrorString(cudaStatus));

	cudaDeviceSynchronize();
	(cudaEventRecord(stop, 0));
	(cudaEventSynchronize(stop));
	(cudaEventElapsedTime(&time, start, stop));

	cout << "Time Elapsed for pararlel multiplication with Shared Memory // 300x1000  1000x500 Matrix: " << time << "ms" << '\n';


	cudaStatus = cudaMemcpy(sum, d_sum, row1*col2*sizeof(float), cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 12: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaFree(d_m);
	cudaFree(d_n);
	cudaFree(d_sum);
	cudaDeviceReset();

}