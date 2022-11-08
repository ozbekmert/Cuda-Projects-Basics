
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <cstdlib> 
#include <iostream>
#include <ctime>


const int row1 = 10;
const int col1 = 10;

const int row2 = 100;
const int col2 = 100;

const int row3 = 1000;
const int col3 = 1000;

const int row4 = 500;
const int col4 = 2000;

const int row5 = 100;
const int col5 = 10000;

using namespace std;

float* create_random_matrix(int row, int col);
float* create_zero_matrix(int row, int col);
float* matrix_add(int row, int col, float* matrix);
void cuda_helper_func(int config1, int config2, int row, int col, float* matrix,float* sum);

__global__ void cuda_add(float *matrix, float *sum, int row) {

	int rowCUDA = blockIdx.x * blockDim.x + threadIdx.x;
	int colCUDA = blockIdx.y * blockDim.y + threadIdx.y;

	sum[(rowCUDA)*row + colCUDA] = matrix[(rowCUDA)*row + colCUDA] + matrix[(rowCUDA)*row + colCUDA];
}

int main(){

	float* matrix1 = create_random_matrix(row1, col1);
	float* matrix2 = create_random_matrix(row2, col2);
	float* matrix3 = create_random_matrix(row3, col3);
	float* matrix4 = create_random_matrix(row4, col4);
	float* matrix5 = create_random_matrix(row5, col5);
	float* sum1;
	float* sum2;
	float* sum3;
	float* sum4;
	float* sum5;

	std::clock_t start;
	double duration;

	start = std::clock();
	sum1 = matrix_add(row1, col1, matrix1);
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "Time Elapsed for serial addition of 10x10 Matrix is: " << duration * 1000 << "ms" << '\n';

	start = std::clock();
	sum2 = matrix_add(row2, col2, matrix2);
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "Time Elapsed for serial addition of 100x100 Matrix is: " << duration * 1000 << "ms" << '\n';

	start = std::clock();
	sum3 = matrix_add(row3, col3, matrix3);
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "Time Elapsed for serial addition of 1000x1000 Matrix is: " << duration * 1000 << "ms" << '\n';

	start = std::clock();
	sum4 = matrix_add(row4, col4, matrix4);
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "Time Elapsed for serial addition of 500x2000 Matrix is: " << duration * 1000 << "ms" << '\n';

	start = std::clock();
	sum5 = matrix_add(row5, col5, matrix5);
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "Time Elapsed for serial addition of 100x10000 Matrix is: " << duration * 1000 << "ms" << '\n';

	int config[6] = {16,16,16,32,32,16};


	for (int k = 0; k < 6; k = k + 2)
		cuda_helper_func(config[k], config[k + 1], row5, col5, matrix5, sum5);

	return 0;

}

float* matrix_add(int row, int col, float* matrix)
{
	float* sum;
	size_t dsize = row*col*sizeof(float);
	sum = (float*)malloc(dsize);
	for (int i = 0; i < row; i++)
	for (int j = 0; j < col; j++)
	{
		sum[i*row + j] = 0;

	}
	for (int i = 0; i < row; i++)
	for (int j = 0; j <col; j++)
	{
		sum[i*row + j] = matrix[i*row + j] + matrix[i*row + j];
	}

	return sum;


}

float* create_random_matrix(int row, int col)
{

	float* matrix;
	size_t dsize = row*col*sizeof(float);
	matrix = (float*)malloc(dsize);

	for (int i = 0; i < row; i++)
	for (int j = 0; j < col; j++) 
	{
		matrix[i*row + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		
	}
	return matrix;
}


void cuda_helper_func(int config1, int config2, int row, int col,float* matrix, float* sum)
{
	cudaError_t cudaStatus;
	float time;

	float *d_a;
	cudaStatus = cudaMalloc((float**)&d_a, row*col*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 1: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaMemcpy(d_a, matrix, row*col*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 2: %s\n", cudaGetErrorString(cudaStatus));
	}

	float *d_a1;
	cudaStatus = cudaMalloc((float**)&d_a1, row*col*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 3: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaMemcpy(d_a1, sum, row*col*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 4: %s\n", cudaGetErrorString(cudaStatus));
	}
	//------------------------------------------------------------------------------

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cuda_add<<<config1, config2 >> >(d_a, d_a1,row);

	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "failed 11: %s\n", cudaGetErrorString(cudaStatus));

	cudaDeviceSynchronize();
	(cudaEventRecord(stop, 0));
	(cudaEventSynchronize(stop));
	(cudaEventElapsedTime(&time, start, stop));

	cout << "Time Elapsed for pararlel addition of 100x10000 Matrix with " << config1 << "-" << config2 << " configuration is: " << time << "ms" << '\n';


	cudaStatus = cudaMemcpy(sum, d_a1, row*col*sizeof(float), cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 12: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaFree(d_a);
	cudaFree(d_a1);
	cudaDeviceReset();

}





