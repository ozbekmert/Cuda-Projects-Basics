#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream> 

using namespace std;
void vectorPrint(int *arr, int Size);
int vectorSum(int *arr, int Size);
void vectorPrint2(int *arr, int row, int col);
void lunchCuda_withoutReduction(int *full_Imag, int *partial_Imag, int row, int col, int row2, int col2);
void lunchCuda_withReduction(int *full_Imag, int *partial_Imag, int row, int col, int row2, int col2);

void printDevProp(cudaDeviceProp devProp);

__global__ void imagDif(int* bigImage, int* partial_Imag, int* res, int* counter, int hShift, int vShift, int row, int col, int row2, int col2, int count)
{
	counter[count] = 0;
	__syncthreads();

	int idxM1 = blockIdx.x * blockDim.x + threadIdx.x + hShift;
	int idyM1 = blockIdx.y * blockDim.y + threadIdx.y + vShift;

	int idxM2 = blockIdx.x * blockDim.x + threadIdx.x;
	int idyM2 = blockIdx.y * blockDim.y + threadIdx.y;



	if (bigImage[(idxM1)*row + idyM1] == partial_Imag[(idxM2)*row2 + idyM2])
	{
		atomicAdd(&counter[count], 1);
	}

	if (counter[count] == row2*col2)
		res[count] = 1;
	else
		res[count] = 0;

}

__global__ void reducted0_imagDif(int* bigImage, int* partial_Imag, int* res, int* hShift, int* vShift, int row, int col, int row2, int col2)
{
	__shared__ int res_data[4];

	int idxM1 = blockIdx.x * blockDim.x + threadIdx.x - hShift[blockIdx.x];
	int idyM1 = blockIdx.y * blockDim.y + threadIdx.y - vShift[blockIdx.y];

	int idxM2 = blockIdx.x * blockDim.x + threadIdx.x - (blockIdx.x * blockDim.x)*hShift[blockIdx.x];;
	int idyM2 = blockIdx.y * blockDim.y + threadIdx.y - (blockIdx.y * blockDim.y)*vShift[blockIdx.y];;
	int global_index = threadIdx.x + blockDim.x * threadIdx.y;

	res_data[global_index] = 0;

	if (bigImage[(idxM1)*row + idyM1] == partial_Imag[(idxM2)*row2 + idyM2])
	{

		res_data[global_index] = 1;

	}

	for (unsigned int s = 1; s < row2*col2; s *= 2)
	{
		if (global_index % (2 * s) == 0)
		{
			res_data[global_index] += res_data[global_index + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (res_data[0] == 4)
	{
		res[blockIdx.x*row2 + blockIdx.y] = 1;

	}
}



int main()
{

	/*
	// Number of CUDA devices
	int devCount;
	cudaGetDeviceCount(&devCount);
	printf("CUDA Device Query...\n");
	printf("There are %d CUDA devices.\n", devCount);

	// Iterate through devices
	for (int i = 0; i < devCount; ++i)
	{
	// Get device properties
	printf("\nCUDA Device #%d\n", i);
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, i);
	printDevProp(devProp);
	}
	*/

	int row = 40;
	int col = 40;
	int row2 = 2;
	int col2 = 2;
	int M1[1600];
	int M2[4];

	for (int index = 0; index < row*col; index++)
		M1[index] = (int)rand() % 100;




	for (int r = 0; r < row2; r++)
	{
		for (int c = 0; c < col2; c++)
		{
			M2[r*row2 + c] = M1[(r + 1)*row + c + 1];

		}
	}


	vectorPrint2(M1, row, col);
	vectorPrint2(M2, row2, col2);


	int horizontalshift = 0; // initial Shift is zero
	int verticalShift = 0; // initial Shift is zero
	double count = 0.0;

	// beginning of the serial process to find targetted face in a picture
	// Measuring Time begins here
	float time;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);


	while (verticalShift <= col - col2)
	{
		while (horizontalshift <= row - row2)
		{
			for (int rowX = horizontalshift; rowX < row2 + horizontalshift; rowX++)
			{
				for (int colX = verticalShift; colX < col2 + verticalShift; colX++)
				{

					if (M1[rowX*row + colX] == M2[(rowX - horizontalshift)*row2 + colX - verticalShift])
					{
						count++;
						if (count == row2*col2)
						{

							cout << "First Index of Match: " << horizontalshift << "-" << verticalShift << endl;
							cout << "Last Index of Match : " << rowX << "-" << colX << endl;
							double similatity = 100 * (count / (row2*col2));
							cout << "similarity: %" << similatity << endl;

						}
					}


				}
			}
			if (count < row2*col2)
			{
				count = 0;
			}
			horizontalshift++;
		}
		horizontalshift = 0;
		verticalShift++;
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	printf("Time to generate:  %3.1f ms \n", time);
	cout << "-------------------------------------------------------------------------------" << endl;
	cout << "-------------------------------------------------------------------------------" << endl;
	cout << "-------------------------------------------------------------------------------" << endl;


	lunchCuda_withoutReduction(M1, M2, row, col, row2, col2);
	cout << "-------------------------------------------------------------------------------" << endl;
	cout << "-------------------------------------------------------------------------------" << endl;
	cout << "-------------------------------------------------------------------------------" << endl;

	lunchCuda_withReduction(M1, M2, row, col, row2, col2);
	cout << "finished" << endl;

	return 0;
}



void vectorPrint(int *arr, int Size)
{

	for (int n = 0; n<Size; n++)
	{
		cout << arr[n] << " ";

	}
	cout << endl;

}
void vectorPrint2(int *arr, int row, int col)
{
	cout << endl;

	for (int j = 0; j<row; j++)
	{
		for (int i = 0; i < col; i++)
			cout << arr[j*row + i] << "  ";
		cout << endl;

	}
	cout << endl;

}
void findMatch(int *arr, int row, int col)
{


	for (int j = 0; j<row; j++)
	{
		for (int i = 0; i < col; i++)
		if (arr[j*row + i] == 1)
			cout << "found on: " << " First Index Row:" << j << " First Index Col:" << i << endl;



	}


}

int vectorSum(int *arr, int Size)
{
	int sum = 0;
	for (int n = 0; n<Size; n++)
	{
		sum += arr[n];

	}
	return sum;

}
void lunchCuda_withoutReduction(int *full_Imag, int *partial_Imag, int row, int col, int row2, int col2)
{

	cudaError_t cudaStatus;

	int hShift = 0; // initial Shift is zero
	int vShift = 0; // initial Shift is zero

	//--------------------- Copy Full Image to Memory-------------------------------
	//------------------------------------------------------------------------------
	int *d_a;
	cudaStatus = cudaMalloc((int**)&d_a, row*col*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 1: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaMemcpy(d_a, full_Imag, row*col*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 2: %s\n", cudaGetErrorString(cudaStatus));
	}
	//------------------------------------------------------------------------------

	//--------------------- Copy Partial Image to Memory-------------------------------
	//------------------------------------------------------------------------------
	int *d_a1;
	cudaStatus = cudaMalloc((int**)&d_a1, row2*col2*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 3: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaMemcpy(d_a1, partial_Imag, row2*col2*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 4: %s\n", cudaGetErrorString(cudaStatus));
	}
	//------------------------------------------------------------------------------


	//--------------------- Copy Result Array to Memory-------------------------------
	//------------------------------------------------------------------------------


	int *res;
	int *d_a2;
	res = (int*)malloc(row*col*sizeof(int));
	cudaStatus = cudaMalloc((int**)&d_a2, row*col*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 5: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaMemcpy(d_a2, res, row*col*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 6: %s\n", cudaGetErrorString(cudaStatus));
	}

	//------------------------------------------------------------------------------

	//--------------------- Copy Counter Array to Memory-------------------------------
	//------------------------------------------------------------------------------


	int* counter;
	int *d_a3;
	counter = (int*)malloc((row - row2 + 1)*(col - col2 + 1)*sizeof(int));
	cudaStatus = cudaMalloc((int**)&d_a3, (row - row2 + 1)*(col - col2 + 1)*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 9: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaMemcpy(d_a3, counter, (row - row2 + 1)*(col - col2 + 1)*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 10: %s\n", cudaGetErrorString(cudaStatus));
	}

	//------------------------------------------------------------------------------

	dim3 threadsPerBlock(row2, col2);
	dim3 numBlocks(row2 / threadsPerBlock.x, col2 / threadsPerBlock.y);


	int count = 0;
	float time;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	while (vShift <= col - col2)
	{
		while (hShift <= row - row2)
		{
			imagDif << <numBlocks, threadsPerBlock >> >(d_a, d_a1, d_a2, d_a3, hShift, vShift, row, col, row2, col2, count);

			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "failed 7: %s\n", cudaGetErrorString(cudaStatus));
			}

			hShift++;
			count++;


		}
		hShift = 0;
		vShift++;
	}
	cudaDeviceSynchronize();

	(cudaEventRecord(stop, 0));
	(cudaEventSynchronize(stop));
	(cudaEventElapsedTime(&time, start, stop));

	printf("Time to generate:  %3.1f ms \n", time);



	cudaStatus = cudaMemcpy(res, d_a2, row*col*sizeof(int), cudaMemcpyDeviceToHost);



	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 8: %s\n", cudaGetErrorString(cudaStatus));
	}


	findMatch(res, row - row2 + 1, col - col2 + 1);
	//vectorPrint2(res, row - row2 + 1, col - col2 + 1);

	cudaFree(d_a);
	cudaFree(d_a1);
	cudaFree(d_a2);
	cudaDeviceReset();


}



void lunchCuda_withReduction(int *full_Imag, int *partial_Imag, int row, int col, int row2, int col2)
{

	cudaError_t cudaStatus;

	//--------------------- Copy Full Image to Memory-------------------------------
	//------------------------------------------------------------------------------
	int *d_a;
	cudaStatus = cudaMalloc((int**)&d_a, row*col*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 1: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaMemcpy(d_a, full_Imag, row*col*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 2: %s\n", cudaGetErrorString(cudaStatus));
	}
	//------------------------------------------------------------------------------

	//--------------------- Copy Partial Image to Memory-------------------------------
	//------------------------------------------------------------------------------
	int *d_a1;
	cudaStatus = cudaMalloc((int**)&d_a1, row2*col2*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 3: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaMemcpy(d_a1, partial_Imag, row2*col2*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 4: %s\n", cudaGetErrorString(cudaStatus));
	}
	//------------------------------------------------------------------------------


	//--------------------- Copy Result Array to Memory-------------------------------
	//--------------------------------------------------------------------------------
	int *res;
	int *d_a2;
	res = (int*)malloc((row - row2 + 1)*(col - col2 + 1)*sizeof(int));
	cudaStatus = cudaMalloc((int**)&d_a2, (row - row2 + 1)*(col - col2 + 1)*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 5: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaMemcpy(d_a2, res, (row - row2 + 1)*(col - col2 + 1)*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 6: %s\n", cudaGetErrorString(cudaStatus));
	}

	//------------------------------------------------------------------------------
	//--------------------- Copy Counter H_V Shift Counter to Memory-------------------------------
	//------------------------------------------------------------------------------

	int* hShift = new int[row - row2 + 1];
	for (int i = 0; i < (row - row2+1); i++)
		hShift[i] = i;
	int *d_a4;
	cudaStatus = cudaMalloc((int**)&d_a4, (row - row2 + 1)*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 7: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaMemcpy(d_a4, hShift, (row - row2 + 1)*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 8: %s\n", cudaGetErrorString(cudaStatus));
	}

	int* vShift = new int[col - col2 + 1];
	for (int i = 0; i < (col - col2+1); i++)
		vShift[i] = i;

	int *d_a5;
	cudaStatus = cudaMalloc((int**)&d_a5, (col - col2 + 1)*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 9: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaMemcpy(d_a5, vShift, (col - col2 + 1)*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 10: %s\n", cudaGetErrorString(cudaStatus));
	}


	dim3 threadsPerBlock(row2, col2);
	dim3 numBlocks(row - row2 + 1, col - col2 + 1);
	float time;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	reducted0_imagDif << <numBlocks, threadsPerBlock >> >(d_a, d_a1, d_a2, d_a4, d_a5, row, col, row2, col2);

	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "failed 11: %s\n", cudaGetErrorString(cudaStatus));

	cudaDeviceSynchronize();

	(cudaEventRecord(stop, 0));
	(cudaEventSynchronize(stop));
	(cudaEventElapsedTime(&time, start, stop));

	printf("Time to generate:  %3.1f ms \n", time);



	cudaStatus = cudaMemcpy(res, d_a2, (row - row2 + 1)*(col - col2 + 1)*sizeof(int), cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed 12: %s\n", cudaGetErrorString(cudaStatus));
	}


	findMatch(res, row - row2 + 1, col - col2 + 1);
	//vectorPrint2(res, row - row2 + 1, col - col2 + 1);

	cudaFree(d_a);
	cudaFree(d_a1);
	cudaFree(d_a2);
	cudaDeviceReset();


}




void printDevProp(cudaDeviceProp devProp)
{
	printf("Major revision number:         %d\n", devProp.major);
	printf("Minor revision number:         %d\n", devProp.minor);
	printf("Name:                          %s\n", devProp.name);
	printf("Total global memory:           %u\n", devProp.totalGlobalMem);
	printf("Total shared memory per block: %u\n", devProp.sharedMemPerBlock);
	printf("Total registers per block:     %d\n", devProp.regsPerBlock);
	printf("Warp size:                     %d\n", devProp.warpSize);
	printf("Maximum memory pitch:          %u\n", devProp.memPitch);
	printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
	printf("Clock rate:                    %d\n", devProp.clockRate);
	printf("Total constant memory:         %u\n", devProp.totalConstMem);
	printf("Texture alignment:             %u\n", devProp.textureAlignment);
	printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
	printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
	printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
	return;
}