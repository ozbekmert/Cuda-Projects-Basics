#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include <iostream>
#include <time.h>
#include <ctime>

#define TILE_DIM 32
#define COL 1920
#define ROW 1080
#define FILT2X2 2
#define FILT4X4 4
#define FILT8X8 8

void CPUconvolution(float *data, float *filter, float *output, int filterSize);
void checkResult(float *hostRef, float *gpuRef);


__constant__ float filter1[FILT2X2*FILT2X2];
__constant__ float filter2[FILT4X4*FILT4X4];
__constant__ float filter3[FILT8X8*FILT8X8];

texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

typedef float mtx1[COL];
using namespace std;

__global__ void globalToShared2x2(float *out, float *data, float *filter, int filterSize)
{
	__shared__ float sharedMem[4];

	int mm, nn, ii, jj;
	sharedMem[0] = filter[0];
	sharedMem[1] = filter[1];
	sharedMem[2] = filter[2];
	sharedMem[3] = filter[3];
	int x = blockIdx.x * 32 + threadIdx.x;
	int y = blockIdx.y * 32 + threadIdx.y;
	int f_dimX = filterSize / 2;
	int f_dimY = filterSize / 2;


	for (int m = 0; m < filterSize; ++m)
	{
		mm = filterSize - 1 - m;

		for (int n = 0; n < filterSize; ++n)
		{
			nn = filterSize - 1 - n;
			ii = x + (m - f_dimY);
			jj = y + (n - f_dimX);
			// ignore input samples which are out of bound
			if (ii >= 0 && ii < ROW && jj >= 0 && jj < COL)
				out[x*COL + y] += data[ii*COL + jj] * sharedMem[mm*COL + nn];
		}
	}


}
__global__ void globalToShared4x4(float *out, float *data, float *filter, int filterSize)
{
	__shared__ float sharedMem[16];
	int mm, nn, ii, jj;
	sharedMem[0] = filter[0];
	sharedMem[1] = filter[1];
	sharedMem[2] = filter[2];
	sharedMem[3] = filter[3];
	sharedMem[4] = filter[4];
	sharedMem[5] = filter[5];
	sharedMem[6] = filter[6];
	sharedMem[7] = filter[7];
	sharedMem[8] = filter[8];
	sharedMem[9] = filter[9];
	sharedMem[10] = filter[10];
	sharedMem[11] = filter[11];
	sharedMem[12] = filter[12];
	sharedMem[13] = filter[13];
	sharedMem[14] = filter[14];
	sharedMem[15] = filter[15];
	int x = blockIdx.x * 32 + threadIdx.x;
	int y = blockIdx.y * 32 + threadIdx.y;
	int f_dimX = filterSize / 2;
	int f_dimY = filterSize / 2;


	for (int m = 0; m < filterSize; ++m)
	{
		mm = filterSize - 1 - m;

		for (int n = 0; n < filterSize; ++n)
		{
			nn = filterSize - 1 - n;
			ii = x + (m - f_dimY);
			jj = y + (n - f_dimX);
			// ignore input samples which are out of bound
			if (ii >= 0 && ii < ROW && jj >= 0 && jj < COL)
				out[x*COL + y] += data[ii*COL + jj] * sharedMem[mm*COL + nn];
		}
	}


}
__global__ void globalToShared8x8(float *out, float *data, float *filter, int filterSize)
{
	__shared__ float sharedMem[64];
	int mm, nn, ii, jj;
	sharedMem[0] = filter[0];
	sharedMem[1] = filter[1];
	sharedMem[2] = filter[2];
	sharedMem[3] = filter[3];
	sharedMem[4] = filter[4];
	sharedMem[5] = filter[5];
	sharedMem[6] = filter[6];
	sharedMem[7] = filter[7];
	sharedMem[8] = filter[8];
	sharedMem[9] = filter[9];
	sharedMem[10] = filter[10];
	sharedMem[11] = filter[11];
	sharedMem[12] = filter[12];
	sharedMem[13] = filter[13];
	sharedMem[14] = filter[14];
	sharedMem[15] = filter[15];
	sharedMem[16] = filter[16];
	sharedMem[17] = filter[17];
	sharedMem[18] = filter[18];
	sharedMem[19] = filter[19];
	sharedMem[20] = filter[20];
	sharedMem[21] = filter[21];
	sharedMem[22] = filter[22];
	sharedMem[23] = filter[23];
	sharedMem[24] = filter[24];
	sharedMem[25] = filter[25];
	sharedMem[26] = filter[26];
	sharedMem[27] = filter[27];
	sharedMem[28] = filter[28];
	sharedMem[29] = filter[29];
	sharedMem[30] = filter[30];
	sharedMem[31] = filter[31];
	sharedMem[32] = filter[32];
	sharedMem[33] = filter[33];
	sharedMem[34] = filter[34];
	sharedMem[35] = filter[35];
	sharedMem[36] = filter[36];
	sharedMem[37] = filter[37];
	sharedMem[38] = filter[38];
	sharedMem[39] = filter[39];
	sharedMem[40] = filter[40];
	sharedMem[41] = filter[41];
	sharedMem[42] = filter[42];
	sharedMem[43] = filter[43];
	sharedMem[44] = filter[44];
	sharedMem[45] = filter[45];
	sharedMem[46] = filter[46];
	sharedMem[47] = filter[47];
	sharedMem[48] = filter[48];
	sharedMem[49] = filter[49];
	sharedMem[50] = filter[50];
	sharedMem[51] = filter[51];
	sharedMem[52] = filter[52];
	sharedMem[53] = filter[53];
	sharedMem[54] = filter[54];
	sharedMem[55] = filter[55];
	sharedMem[56] = filter[56];
	sharedMem[57] = filter[57];
	sharedMem[58] = filter[58];
	sharedMem[59] = filter[59];
	sharedMem[60] = filter[60];
	sharedMem[61] = filter[61];
	sharedMem[62] = filter[62];
	sharedMem[63] = filter[63];
	int x = blockIdx.x * 32 + threadIdx.x;
	int y = blockIdx.y * 32 + threadIdx.y;
	int f_dimX = filterSize / 2;
	int f_dimY = filterSize / 2;


	for (int m = 0; m < filterSize; ++m)
	{
		mm = filterSize - 1 - m;

		for (int n = 0; n < filterSize; ++n)
		{
			nn = filterSize - 1 - n;
			ii = x + (m - f_dimY);
			jj = y + (n - f_dimX);
			// ignore input samples which are out of bound
			if (ii >= 0 && ii < ROW && jj >= 0 && jj < COL)
				out[x*COL + y] += data[ii*COL + jj] * sharedMem[mm*COL + nn];
		}
	}


}
__global__ void textureToRegister8x8(float *out, float *data, int filterSize)
{

	int x;
	int y;
	register float registers[64];
	registers[0] = tex2D(texRef, 0, 0);
	registers[1] = tex2D(texRef, 0, 1);
	registers[2] = tex2D(texRef, 0, 2);
	registers[3] = tex2D(texRef, 0, 3);
	registers[4] = tex2D(texRef, 0, 4);
	registers[5] = tex2D(texRef, 0, 5);
	registers[6] = tex2D(texRef, 0, 6);
	registers[7] = tex2D(texRef, 0, 7);
	registers[8] = tex2D(texRef, 1, 0);
	registers[9] = tex2D(texRef, 1, 1);
	registers[10] = tex2D(texRef, 1, 2);
	registers[11] = tex2D(texRef, 1, 3);
	registers[12] = tex2D(texRef, 1, 4);
	registers[13] = tex2D(texRef, 1, 5);
	registers[14] = tex2D(texRef, 1, 6);
	registers[15] = tex2D(texRef, 1, 7);
	registers[16] = tex2D(texRef, 2, 0);
	registers[17] = tex2D(texRef, 2, 1);
	registers[18] = tex2D(texRef, 2, 2);
	registers[19] = tex2D(texRef, 2, 3);
	registers[20] = tex2D(texRef, 2, 4);
	registers[21] = tex2D(texRef, 2, 5);
	registers[22] = tex2D(texRef, 2, 6);
	registers[23] = tex2D(texRef, 2, 7);
	registers[24] = tex2D(texRef, 3, 0);
	registers[25] = tex2D(texRef, 3, 1);
	registers[26] = tex2D(texRef, 3, 2);
	registers[27] = tex2D(texRef, 3, 3);
	registers[28] = tex2D(texRef, 3, 4);
	registers[29] = tex2D(texRef, 3, 5);
	registers[30] = tex2D(texRef, 3, 6);
	registers[31] = tex2D(texRef, 3, 7);
	registers[32] = tex2D(texRef, 4, 0);
	registers[33] = tex2D(texRef, 4, 1);
	registers[34] = tex2D(texRef, 4, 2);
	registers[35] = tex2D(texRef, 4, 3);
	registers[36] = tex2D(texRef, 4, 4);
	registers[37] = tex2D(texRef, 4, 5);
	registers[38] = tex2D(texRef, 4, 6);
	registers[39] = tex2D(texRef, 4, 7);
	registers[40] = tex2D(texRef, 5, 0);
	registers[41] = tex2D(texRef, 5, 1);
	registers[42] = tex2D(texRef, 5, 2);
	registers[43] = tex2D(texRef, 5, 3);
	registers[44] = tex2D(texRef, 5, 4);
	registers[45] = tex2D(texRef, 5, 5);
	registers[46] = tex2D(texRef, 5, 6);
	registers[47] = tex2D(texRef, 5, 7);
	registers[48] = tex2D(texRef, 6, 0);
	registers[49] = tex2D(texRef, 6, 1);
	registers[50] = tex2D(texRef, 6, 2);
	registers[51] = tex2D(texRef, 6, 3);
	registers[52] = tex2D(texRef, 6, 4);
	registers[53] = tex2D(texRef, 6, 5);
	registers[54] = tex2D(texRef, 6, 6);
	registers[55] = tex2D(texRef, 6, 7);
	registers[56] = tex2D(texRef, 7, 0);
	registers[57] = tex2D(texRef, 7, 1);
	registers[58] = tex2D(texRef, 7, 2);
	registers[59] = tex2D(texRef, 7, 3);
	registers[60] = tex2D(texRef, 7, 4);
	registers[61] = tex2D(texRef, 7, 5);
	registers[62] = tex2D(texRef, 7, 6);
	registers[63] = tex2D(texRef, 7, 7);

	x = blockIdx.x*blockDim.x + threadIdx.x;
	y = blockIdx.y*blockDim.y + threadIdx.y;
	int mm, nn, ii, jj;
	int f_dimX = filterSize / 2;
	int f_dimY = filterSize / 2;


	for (int m = 0; m < filterSize; ++m)
	{
		mm = filterSize - 1 - m;

		for (int n = 0; n < filterSize; ++n)
		{
			nn = filterSize - 1 - n;
			ii = x + (m - f_dimY);
			jj = y + (n - f_dimX);
			// ignore input samples which are out of bound
			if (ii >= 0 && ii < ROW && jj >= 0 && jj < COL)
				out[x*COL + y] += data[ii*COL + jj] * registers[mm*COL + nn];
		}
	}


}
__global__ void textureToRegister4x4(float *out, float *data, int filterSize)
{

	int x;
	int y;
	register float registers[16];
	registers[0] = tex2D(texRef, 0, 0);
	registers[1] = tex2D(texRef, 0, 1);
	registers[2] = tex2D(texRef, 0, 2);
	registers[3] = tex2D(texRef, 0, 3);
	registers[4] = tex2D(texRef, 1, 0);
	registers[5] = tex2D(texRef, 1, 1);
	registers[6] = tex2D(texRef, 1, 2);
	registers[7] = tex2D(texRef, 1, 3);
	registers[8] = tex2D(texRef, 2, 0);
	registers[9] = tex2D(texRef, 2, 1);
	registers[10] = tex2D(texRef, 2, 2);
	registers[11] = tex2D(texRef, 2, 3);
	registers[12] = tex2D(texRef, 3, 0);
	registers[13] = tex2D(texRef, 3, 1);
	registers[14] = tex2D(texRef, 3, 2);
	registers[15] = tex2D(texRef, 3, 3);

	x = blockIdx.x*blockDim.x + threadIdx.x;
	y = blockIdx.y*blockDim.y + threadIdx.y;
	int mm, nn, ii, jj;
	int f_dimX = filterSize / 2;
	int f_dimY = filterSize / 2;


	for (int m = 0; m < filterSize; ++m)
	{
		mm = filterSize - 1 - m;

		for (int n = 0; n < filterSize; ++n)
		{
			nn = filterSize - 1 - n;
			ii = x + (m - f_dimY);
			jj = y + (n - f_dimX);
			// ignore input samples which are out of bound
			if (ii >= 0 && ii < ROW && jj >= 0 && jj < COL)
				out[x*COL + y] += data[ii*COL + jj] * registers[mm*COL + nn];
		}
	}


}
__global__ void textureToRegister2x2(float *out, float *data, int filterSize)
{

	int x;
	int y;
	register float registers[4];
	registers[0] = tex2D(texRef, 0, 0);
	registers[1] = tex2D(texRef, 0, 1);
	registers[2] = tex2D(texRef, 1, 0);
	registers[3] = tex2D(texRef, 1, 1);

	x = blockIdx.x*blockDim.x + threadIdx.x;
	y = blockIdx.y*blockDim.y + threadIdx.y;
	int mm, nn, ii, jj;
	int f_dimX = filterSize / 2;
	int f_dimY = filterSize / 2;


	for (int m = 0; m < filterSize; ++m)
	{
		mm = filterSize - 1 - m;

		for (int n = 0; n < filterSize; ++n)
		{
			nn = filterSize - 1 - n;
			ii = x + (m - f_dimY);
			jj = y + (n - f_dimX);
			// ignore input samples which are out of bound
			if (ii >= 0 && ii < ROW && jj >= 0 && jj < COL)
				out[x*COL + y] += data[ii*COL + jj] * registers[mm*COL + nn];
		}
	}


}
__global__ void globalToRegister2x2(float *out, float *data,float *filter, int filterSize)
{

	register float registers[4];
	int mm, nn, ii, jj;
	registers[0] = filter[0];
	registers[1] = filter[1];
	registers[2] = filter[2];
	registers[3] = filter[3];
	int x = blockIdx.x * 32 + threadIdx.x;
	int y = blockIdx.y * 32 + threadIdx.y;
	int f_dimX = filterSize / 2;
	int f_dimY = filterSize / 2;


			for (int m = 0; m < filterSize; ++m)
			{
				mm = filterSize - 1 - m;

				for (int n = 0; n < filterSize; ++n)
				{
					nn = filterSize - 1 - n;
					ii = x + (m - f_dimY);
					jj = y + (n - f_dimX);
					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ROW && jj >= 0 && jj < COL)
						out[x*COL + y] += data[ii*COL + jj] * registers[mm*COL + nn];
				}
			}


}
__global__ void globalToRegister4x4(float *out, float *data, float *filter, int filterSize)
{

	register float registers[16];
	int mm, nn, ii, jj;
    registers[0] = filter[0];
	registers[1] = filter[1];
	registers[2] = filter[2];
	registers[3] = filter[3];
	registers[4] = filter[4];
	registers[5] = filter[5];
	registers[6] = filter[6];
	registers[7] = filter[7];
	registers[8] = filter[8];
	registers[9] = filter[9];
	registers[10] = filter[10];
	registers[11] = filter[11];
	registers[12] = filter[12];
	registers[13] = filter[13];
	registers[14] = filter[14];
	registers[15] = filter[15];
	int x = blockIdx.x * 32 + threadIdx.x;
	int y = blockIdx.y * 32 + threadIdx.y;
	int f_dimX = filterSize / 2;
	int f_dimY = filterSize / 2;


	for (int m = 0; m < filterSize; ++m)
	{
		mm = filterSize - 1 - m;

		for (int n = 0; n < filterSize; ++n)
		{
			nn = filterSize - 1 - n;
			ii = x + (m - f_dimY);
			jj = y + (n - f_dimX);
			// ignore input samples which are out of bound
			if (ii >= 0 && ii < ROW && jj >= 0 && jj < COL)
				out[x*COL + y] += data[ii*COL + jj] * registers[mm*COL + nn];
		}
	}


}
__global__ void globalToRegister8x8(float *out, float *data, float *filter, int filterSize)
{

	register float registers[64];
	int mm, nn, ii, jj;
	registers[0] = filter[0];
	registers[1] = filter[1];
	registers[2] = filter[2];
	registers[3] = filter[3];
	registers[4] = filter[4];
	registers[5] = filter[5];
	registers[6] = filter[6];
	registers[7] = filter[7];
	registers[8] = filter[8];
	registers[9] = filter[9];
	registers[10] = filter[10];
	registers[11] = filter[11];
	registers[12] = filter[12];
	registers[13] = filter[13];
	registers[14] = filter[14];
	registers[15] = filter[15];
	registers[16] = filter[16];
	registers[17] = filter[17];
	registers[18] = filter[18];
	registers[19] = filter[19];
	registers[20] = filter[20];
	registers[21] = filter[21];
	registers[22] = filter[22];
	registers[23] = filter[23];
	registers[24] = filter[24];
	registers[25] = filter[25];
	registers[26] = filter[26];
	registers[27] = filter[27];
	registers[28] = filter[28];
	registers[29] = filter[29];
	registers[30] = filter[30];
	registers[31] = filter[31];
	registers[32] = filter[32];
	registers[33] = filter[33];
	registers[34] = filter[34];
	registers[35] = filter[35];
	registers[36] = filter[36];
	registers[37] = filter[37];
	registers[38] = filter[38];
	registers[39] = filter[39];
	registers[40] = filter[40];
	registers[41] = filter[41];
	registers[42] = filter[42];
	registers[43] = filter[43];
	registers[44] = filter[44];
	registers[45] = filter[45];
	registers[46] = filter[46];
	registers[47] = filter[47];
	registers[48] = filter[48];
	registers[49] = filter[49];
	registers[50] = filter[50];
	registers[51] = filter[51];
	registers[52] = filter[52];
	registers[53] = filter[53];
	registers[54] = filter[54];
	registers[55] = filter[55];
	registers[56] = filter[56];
	registers[57] = filter[57];
	registers[58] = filter[58];
	registers[59] = filter[59];
	registers[60] = filter[60];
	registers[61] = filter[61];
	registers[62] = filter[62];
	registers[63] = filter[63];
	int x = blockIdx.x * 32 + threadIdx.x;
	int y = blockIdx.y * 32 + threadIdx.y;
	int f_dimX = filterSize / 2;
	int f_dimY = filterSize / 2;


	for (int m = 0; m < filterSize; ++m)
	{
		mm = filterSize - 1 - m;

		for (int n = 0; n < filterSize; ++n)
		{
			nn = filterSize - 1 - n;
			ii = x + (m - f_dimY);
			jj = y + (n - f_dimX);
			// ignore input samples which are out of bound
			if (ii >= 0 && ii < ROW && jj >= 0 && jj < COL)
				out[x*COL + y] += data[ii*COL + jj] * registers[mm*COL + nn];
		}
	}


}
__global__ void kernel(float *out, float *data, const int ilp)
{
	int i;
	int col = blockIdx.x * TILE_DIM + threadIdx.x;
	int row = blockIdx.y * TILE_DIM + threadIdx.y;
	int index = row * COL + col;
	int add = TILE_DIM / ilp;

#pragma unroll
	for (i = 0; i < TILE_DIM; i += add)
	if (col < COL && row < ROW)
		out[index + i * COL] = data[index + i * COL];
}

int main()
{
	float *hostRef2x2, *hostRef4x4, *hostRef8x8;



	// Creating 1920 x 1080 Matrix , Convolved Output , 2x2, 4x4, 8x8 filters with non-zero weights
	float *data;
	data = (float*)malloc(ROW*COL*sizeof(float));
	float *out;
	out = (float*)malloc(ROW*COL*sizeof(float));
	float *filter2x2;
	filter2x2 = (float*)malloc(FILT2X2*FILT2X2*sizeof(float));
	float *filter4x4;
	filter4x4 = (float*)malloc(FILT4X4*FILT4X4*sizeof(float));
	float *filter8x8;
	filter8x8 = (float*)malloc(FILT8X8*FILT8X8*sizeof(float));
	for (int i = 0; i<ROW; i++)
	for (int j = 0; j<COL; j++)
	{
		data[i*COL + j] = ((float)rand() / (float)(RAND_MAX)) * 10;
		out[i*COL + j] = 0;
	}	
	for (int i = 0; i<FILT2X2; i++)
	for (int j = 0; j<FILT2X2; j++)
	{
		filter2x2[i*COL + j] = ((float)rand() / (float)(RAND_MAX)) * 10;
	}
	for (int i = 0; i<FILT4X4; i++)
	for (int j = 0; j<FILT4X4; j++)
	{
		filter4x4[i*COL + j] = ((float)rand() / (float)(RAND_MAX)) * 10;
	}
	for (int i = 0; i<FILT8X8; i++)
	for (int j = 0; j<FILT8X8; j++)
	{
		filter8x8[i*COL + j] = ((float)rand() / (float)(RAND_MAX)) * 10;
	}
	// End of creating Matrices


	// Convolution on CPU 
	CPUconvolution(data, filter2x2, out, FILT2X2);
	hostRef2x2=out; 
	CPUconvolution(data, filter4x4, out, FILT4X4);
	hostRef4x4 = out;
	CPUconvolution(data, filter8x8, out, FILT8X8);
	hostRef8x8 = out;

	//Convolution on GPU 
	for (int ilp = 0; ilp < 9; ilp++)
	{

		if ((ilp == 1) || (ilp == 4) || (ilp == 8))
		{

			float *d_A, *d_B, *d_C;
			dim3 dim_grid, dim_block;
			dim3 dimBlock(16, 16);
			dim3 dimGrid(ceil((COL + 16 - 1) / 16), ceil((ROW + 16 - 1) / 16));
			dim_block.x = TILE_DIM;
			dim_block.y = TILE_DIM / ilp;
			dim_grid.x = (COL + TILE_DIM - 1) / TILE_DIM;
			dim_grid.y = (ROW + TILE_DIM - 1) / TILE_DIM;
			cudaMalloc((void**)&d_A, ROW*COL * sizeof(float));
			cudaMalloc((void**)&d_B, ROW*COL * sizeof(float));
			cudaMalloc((void**)&d_C, FILT2X2*FILT2X2 * sizeof(float));
			cudaMemcpy(d_A, data, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(d_C, filter2x2, FILT2X2 * FILT2X2 * sizeof(float));
			float time_elapsed = 0;
			cudaEvent_t start, end;
			cudaEventCreate(&start);
			cudaEventCreate(&end);
			cudaEventRecord(start, 0);
			kernel << <dim_grid, dim_block >> >(d_B, d_A,ilp);
			cudaThreadSynchronize();
			cudaEventRecord(end, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&time_elapsed, start, end);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyDeviceToHost);
			checkResult(hostRef2x2, out);
			cout << "Time Elapsed for Convolution on GPU with filter size " << FILT2X2 << "x" << FILT2X2 << "& ILP: "<< ilp << " is: " << time_elapsed << "milliSeconds" << endl;
			cudaFree(d_A);
			cudaFree(d_B);
			cudaFree(d_C);
			cudaEventDestroy(start);
			cudaEventDestroy(end);
			cudaDeviceReset();
		}
	}

	for (int ilp = 0; ilp < 9; ilp++)
	{

		if ((ilp == 1) || (ilp == 4) || (ilp == 8))
		{

			float *d_A, *d_B, *d_C;
			dim3 dim_grid, dim_block;
			dim3 dimBlock(16, 16);
			dim3 dimGrid(ceil((COL + 16 - 1) / 16), ceil((ROW + 16 - 1) / 16));
			dim_block.x = TILE_DIM;
			dim_block.y = TILE_DIM / ilp;
			dim_grid.x = (COL + TILE_DIM - 1) / TILE_DIM;
			dim_grid.y = (ROW + TILE_DIM - 1) / TILE_DIM;
			cudaMalloc((void**)&d_A, ROW*COL * sizeof(float));
			cudaMalloc((void**)&d_B, ROW*COL * sizeof(float));
			cudaMalloc((void**)&d_C, FILT4X4*FILT4X4 * sizeof(float));
			cudaMemcpy(d_A, data, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(d_C, filter4x4, FILT4X4 * FILT4X4 * sizeof(float));
			float time_elapsed = 0;
			cudaEvent_t start, end;
			cudaEventCreate(&start);
			cudaEventCreate(&end);
			cudaEventRecord(start, 0);
			kernel << <dim_grid, dim_block >> >(d_B, d_A, ilp);
			cudaThreadSynchronize();
			cudaEventRecord(end, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&time_elapsed, start, end);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyDeviceToHost);
			checkResult(hostRef4x4, out);
			cout << "Time Elapsed for Convolution on GPU with filter size " << FILT4X4 << "x" << FILT4X4 << "& ILP: "<< ilp << " is: " << time_elapsed << "milliSeconds" << endl;
			cudaFree(d_A);
			cudaFree(d_B);
			cudaFree(d_C);
			cudaEventDestroy(start);
			cudaEventDestroy(end);
			cudaDeviceReset();
		}
	}
	for (int ilp = 0; ilp < 9; ilp++)
	{

		if ((ilp == 1) || (ilp == 4) || (ilp == 8))
		{

			float *d_A, *d_B, *d_C;
			dim3 dim_grid, dim_block;
			dim3 dimBlock(16, 16);
			dim3 dimGrid(ceil((COL + 16 - 1) / 16), ceil((ROW + 16 - 1) / 16));
			dim_block.x = TILE_DIM;
			dim_block.y = TILE_DIM / ilp;
			dim_grid.x = (COL + TILE_DIM - 1) / TILE_DIM;
			dim_grid.y = (ROW + TILE_DIM - 1) / TILE_DIM;
			cudaMalloc((void**)&d_A, ROW*COL * sizeof(float));
			cudaMalloc((void**)&d_B, ROW*COL * sizeof(float));
			cudaMalloc((void**)&d_C, FILT8X8*FILT8X8 * sizeof(float));
			cudaMemcpy(d_A, data, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(d_C, filter8x8, FILT8X8 * FILT8X8 * sizeof(float));
			float time_elapsed = 0;
			cudaEvent_t start, end;
			cudaEventCreate(&start);
			cudaEventCreate(&end);
			cudaEventRecord(start, 0);
			kernel << <dim_grid, dim_block >> >(d_B, d_A, ilp);
			cudaThreadSynchronize();
			cudaEventRecord(end, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&time_elapsed, start, end);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyDeviceToHost);
			checkResult(hostRef8x8, out);
			cout << "Time Elapsed for Convolution on GPU with filter size " << FILT8X8 << "x" << FILT8X8 << "& ILP: " << ilp << " is: " << time_elapsed << "milliSeconds" << endl;
			cudaFree(d_A);
			cudaFree(d_B);
			cudaFree(d_C);
			cudaEventDestroy(start);
			cudaEventDestroy(end);
			cudaDeviceReset();
		}
	}


	for (int loop = 2; loop < 9; loop++)
	{
		if (loop == 2)
		{


			//Convolution on GPU  with Global to Register Copy
			float *d_A, *d_B, *d_C;
			dim3 dim_grid, dim_block;
			dim3 dimBlock(16, 16);
			dim3 dimGrid(ceil((COL + 16 - 1) / 16), ceil((ROW + 16 - 1) / 16));
			dim_block.x = TILE_DIM;
			dim_block.y = TILE_DIM;
			dim_grid.x = (COL + TILE_DIM - 1) / TILE_DIM;
			dim_grid.y = (ROW + TILE_DIM - 1) / TILE_DIM;
			cudaMalloc((void**)&d_A, ROW*COL * sizeof(float));
			cudaMalloc((void**)&d_B, ROW*COL * sizeof(float));
			cudaMalloc((void**)&d_C, FILT2X2*FILT2X2 * sizeof(float));
			cudaMemcpy(d_A, data, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(d_C, filter2x2, FILT2X2 * FILT2X2 * sizeof(float));
			float time_elapsed = 0;
			cudaEvent_t start, end;
			cudaEventCreate(&start);
			cudaEventCreate(&end);
			cudaEventRecord(start, 0);
			globalToRegister2x2 << <dim_grid, dim_block >> >(d_B, d_A, filter2x2, 2);
			cudaThreadSynchronize();
			cudaEventRecord(end, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&time_elapsed, start, end);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyDeviceToHost);
			checkResult(hostRef2x2, out);
			cout << "Time Elapsed for Convolution(Global to Register Copy) on GPU with filter size " << FILT2X2 << "x" << FILT2X2 << " is: " << time_elapsed << "milliSeconds" << endl;
			cudaFree(d_A);
			cudaFree(d_B);
			cudaFree(d_C);
			cudaEventDestroy(start);
			cudaEventDestroy(end);
			cudaDeviceReset();
		}
		else if (loop == 4)
		{


			//Convolution on GPU  with Global to Register Copy
			float *d_A, *d_B, *d_C;
			dim3 dim_grid, dim_block;
			dim3 dimBlock(16, 16);
			dim3 dimGrid(ceil((COL + 16 - 1) / 16), ceil((ROW + 16 - 1) / 16));
			dim_block.x = TILE_DIM;
			dim_block.y = TILE_DIM;
			dim_grid.x = (COL + TILE_DIM - 1) / TILE_DIM;
			dim_grid.y = (ROW + TILE_DIM - 1) / TILE_DIM;
			cudaMalloc((void**)&d_A, ROW*COL * sizeof(float));
			cudaMalloc((void**)&d_B, ROW*COL * sizeof(float));
			cudaMalloc((void**)&d_C, FILT4X4*FILT4X4 * sizeof(float));
			cudaMemcpy(d_A, data, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(d_C, filter4x4, FILT4X4 * FILT4X4 * sizeof(float));
			float time_elapsed = 0;
			cudaEvent_t start, end;
			cudaEventCreate(&start);
			cudaEventCreate(&end);
			cudaEventRecord(start, 0);
			globalToRegister4x4 << <dim_grid, dim_block >> >(d_B, d_A, filter4x4, 16);
			cudaThreadSynchronize();
			cudaEventRecord(end, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&time_elapsed, start, end);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyDeviceToHost);
			checkResult(hostRef4x4, out);
			cout << "Time Elapsed for Convolution(Global to Register Copy) on GPU with filter size " << FILT4X4 << "x" << FILT4X4 << " is: " << time_elapsed << "milliSeconds" << endl;
			cudaFree(d_A);
			cudaFree(d_B);
			cudaFree(d_C);
			cudaEventDestroy(start);
			cudaEventDestroy(end);
			cudaDeviceReset();
		}
		else if (loop == 8){

			//Convolution on GPU  with Global to Register Copy
			float *d_A, *d_B, *d_C;
			dim3 dim_grid, dim_block;
			dim3 dimBlock(16, 16);
			dim3 dimGrid(ceil((COL + 16 - 1) / 16), ceil((ROW + 16 - 1) / 16));
			dim_block.x = TILE_DIM;
			dim_block.y = TILE_DIM;
			dim_grid.x = (COL + TILE_DIM - 1) / TILE_DIM;
			dim_grid.y = (ROW + TILE_DIM - 1) / TILE_DIM;
			cudaMalloc((void**)&d_A, ROW*COL * sizeof(float));
			cudaMalloc((void**)&d_B, ROW*COL * sizeof(float));
			cudaMalloc((void**)&d_C, FILT8X8*FILT8X8 * sizeof(float));
			cudaMemcpy(d_A, data, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(d_C, filter8x8, FILT8X8 * FILT8X8 * sizeof(float));
			float time_elapsed = 0;
			cudaEvent_t start, end;
			cudaEventCreate(&start);
			cudaEventCreate(&end);
			cudaEventRecord(start, 0);
			globalToRegister8x8 << <dim_grid, dim_block >> >(d_B, d_A, filter8x8, 64);
			cudaThreadSynchronize();
			cudaEventRecord(end, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&time_elapsed, start, end);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyDeviceToHost);
			checkResult(hostRef8x8, out);
			cout << "Time Elapsed for Convolution(Global to Register Copy) on GPU with filter size " << FILT8X8 << "x" << FILT8X8 << " is: " << time_elapsed << "milliSeconds" << endl;
			cudaFree(d_A);
			cudaFree(d_B);
			cudaFree(d_C);
			cudaEventDestroy(start);
			cudaEventDestroy(end);
			cudaDeviceReset();

		}
	}

	//Convolution on GPU  with Texture to Register Copy
	for (int loop = 2; loop < 9; loop++)
	{
		if (loop == 2)
		{

	
				int size = 2;
				float *d_A, *d_B;
				cudaMalloc((void**)&d_A, ROW*COL * sizeof(float));
				cudaMalloc((void**)&d_B, ROW*COL * sizeof(float));
				cudaMemcpy(d_A, data, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
				dim3 blocknum;
				dim3 blocksize;
				cudaArray* carray;
				cudaChannelFormatDesc channel;
				channel = cudaCreateChannelDesc<float>();
				cudaMallocArray(&carray, &channel, size, size);
				//copy matrix from host to device memory
				float bytes = sizeof(float)*size*size;
				cudaMemcpyToArray(carray, 0, 0, filter2x2, bytes, cudaMemcpyHostToDevice);
				//set texture filter mode property
				//use cudaFilterModePoint or cudaFilterModeLinear
				texRef.filterMode = cudaFilterModePoint;
				//set texture address mode property
				//use cudaAddressModeClamp or cudaAddressModeWrap
				texRef.addressMode[0] = cudaAddressModeWrap;
				texRef.addressMode[1] = cudaAddressModeClamp;
				//bind texture reference with cuda array
				cudaBindTextureToArray(texRef, carray);
				blocksize.x = 16;
				blocksize.y = 16;
				blocknum.x = (int)ceil((float)size / 16);
				blocknum.y = (int)ceil((float)size / 16);
				//execute device kernel
				//(float *out, float *data, float *filter, int filterSize)
				float time_elapsed = 0;
				cudaEvent_t start, end;
				cudaEventCreate(&start);
				cudaEventCreate(&end);
				cudaEventRecord(start, 0);
				kernel << <blocknum, blocksize >> >(d_B, d_A , 4);
				cudaThreadSynchronize();
				cudaEventRecord(end, 0);
				cudaEventSynchronize(start);
				cudaEventSynchronize(end);
				cudaEventElapsedTime(&time_elapsed, start, end);
				cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyDeviceToHost);
				checkResult(hostRef2x2, out);
				cout << "Time Elapsed for Convolution(Texture to Register Copy) on GPU with filter size " << FILT2X2 << "x" << FILT2X2 << " is: " << time_elapsed << "milliSeconds" << endl;
				//unbind texture reference to free resource
				cudaUnbindTexture(texRef);
				//copy result matrix from device to host memory
				cudaMemcpy(d_B, out, bytes, cudaMemcpyDeviceToHost);
				//free host and device memory
				cudaFree(d_A);
				cudaFree(d_B);
				cudaEventDestroy(start);
				cudaEventDestroy(end);
				cudaDeviceReset();
		}
		else if (loop == 4)
		{
			int size = 4;
			float *d_A, *d_B;
			cudaMalloc((void**)&d_A, ROW*COL * sizeof(float));
			cudaMalloc((void**)&d_B, ROW*COL * sizeof(float));
			cudaMemcpy(d_A, data, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			dim3 blocknum;
			dim3 blocksize;
			cudaArray* carray;
			cudaChannelFormatDesc channel;
			channel = cudaCreateChannelDesc<float>();
			cudaMallocArray(&carray, &channel, size, size);
			//copy matrix from host to device memory
			float bytes = sizeof(float)*size*size;
			cudaMemcpyToArray(carray, 0, 0, filter4x4, bytes, cudaMemcpyHostToDevice);
			//set texture filter mode property
			//use cudaFilterModePoint or cudaFilterModeLinear
			texRef.filterMode = cudaFilterModePoint;
			//set texture address mode property
			//use cudaAddressModeClamp or cudaAddressModeWrap
			texRef.addressMode[0] = cudaAddressModeWrap;
			texRef.addressMode[1] = cudaAddressModeClamp;
			//bind texture reference with cuda array
			cudaBindTextureToArray(texRef, carray);
			blocksize.x = 16;
			blocksize.y = 16;
			blocknum.x = (int)ceil((float)size / 16);
			blocknum.y = (int)ceil((float)size / 16);
			//execute device kernel
			//(float *out, float *data, float *filter, int filterSize)
			float time_elapsed = 0;
			cudaEvent_t start, end;
			cudaEventCreate(&start);
			cudaEventCreate(&end);
			cudaEventRecord(start, 0);
			kernel << <blocknum, blocksize >> >(d_B, d_A, 4);
			cudaThreadSynchronize();
			cudaEventRecord(end, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&time_elapsed, start, end);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyDeviceToHost);
			checkResult(hostRef4x4, out);
			cout << "Time Elapsed for Convolution(Texture to Register Copy) on GPU with filter size " << FILT4X4 << "x" << FILT4X4 << " is: " << time_elapsed << "milliSeconds" << endl;
			//unbind texture reference to free resource
			cudaUnbindTexture(texRef);
			//copy result matrix from device to host memory
			cudaMemcpy(d_B, out, bytes, cudaMemcpyDeviceToHost);
			//free host and device memory
			cudaFree(d_A);
			cudaFree(d_B);
			cudaEventDestroy(start);
			cudaEventDestroy(end);
			cudaDeviceReset();

		}
		else if (loop == 8)
		{
			int size = 8;
			float *d_A, *d_B;
			cudaMalloc((void**)&d_A, ROW*COL * sizeof(float));
			cudaMalloc((void**)&d_B, ROW*COL * sizeof(float));
			cudaMemcpy(d_A, data, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			dim3 blocknum;
			dim3 blocksize;
			cudaArray* carray;
			cudaChannelFormatDesc channel;
			channel = cudaCreateChannelDesc<float>();
			cudaMallocArray(&carray, &channel, size, size);
			//copy matrix from host to device memory
			float bytes = sizeof(float)*size*size;
			cudaMemcpyToArray(carray, 0, 0, filter8x8, bytes, cudaMemcpyHostToDevice);
			//set texture filter mode property
			//use cudaFilterModePoint or cudaFilterModeLinear
			texRef.filterMode = cudaFilterModePoint;
			//set texture address mode property
			//use cudaAddressModeClamp or cudaAddressModeWrap
			texRef.addressMode[0] = cudaAddressModeWrap;
			texRef.addressMode[1] = cudaAddressModeClamp;
			//bind texture reference with cuda array
			cudaBindTextureToArray(texRef, carray);
			blocksize.x = 16;
			blocksize.y = 16;
			blocknum.x = (int)ceil((float)size / 16);
			blocknum.y = (int)ceil((float)size / 16);
			//execute device kernel
			//(float *out, float *data, float *filter, int filterSize)
			float time_elapsed = 0;
			cudaEvent_t start, end;
			cudaEventCreate(&start);
			cudaEventCreate(&end);
			cudaEventRecord(start, 0);
			kernel << <blocknum, blocksize >> >(d_B, d_A, 4);
			cudaThreadSynchronize();
			cudaEventRecord(end, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&time_elapsed, start, end);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyDeviceToHost);
			checkResult(hostRef8x8, out);
			cout << "Time Elapsed for Convolution(Texture to Register Copy) on GPU with filter size " << FILT8X8 << "x" << FILT8X8 << " is: " << time_elapsed << "milliSeconds" << endl;
			//unbind texture reference to free resource
			cudaUnbindTexture(texRef);
			//copy result matrix from device to host memory
			cudaMemcpy(d_B, out, bytes, cudaMemcpyDeviceToHost);
			//free host and device memory
			cudaFree(d_A);
			cudaFree(d_B);
			cudaEventDestroy(start);
			cudaEventDestroy(end);
			cudaDeviceReset();
		}
	}


	for (int loop = 2; loop < 9; loop++)
	{
		if (loop == 2)
		{


			//Convolution on GPU  with Global to Register Copy
			float *d_A, *d_B, *d_C;
			dim3 dim_grid, dim_block;
			dim3 dimBlock(16, 16);
			dim3 dimGrid(ceil((COL + 16 - 1) / 16), ceil((ROW + 16 - 1) / 16));
			dim_block.x = TILE_DIM;
			dim_block.y = TILE_DIM;
			dim_grid.x = (COL + TILE_DIM - 1) / TILE_DIM;
			dim_grid.y = (ROW + TILE_DIM - 1) / TILE_DIM;
			cudaMalloc((void**)&d_A, ROW*COL * sizeof(float));
			cudaMalloc((void**)&d_B, ROW*COL * sizeof(float));
			cudaMalloc((void**)&d_C, FILT2X2*FILT2X2 * sizeof(float));
			cudaMemcpy(d_A, data, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(d_C, filter2x2, FILT2X2 * FILT2X2 * sizeof(float));
			float time_elapsed = 0;
			cudaEvent_t start, end;
			cudaEventCreate(&start);
			cudaEventCreate(&end);
			cudaEventRecord(start, 0);
			globalToShared2x2<< <dim_grid, dim_block >> >(d_B, d_A, filter2x2, 2);
			cudaThreadSynchronize();
			cudaEventRecord(end, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&time_elapsed, start, end);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyDeviceToHost);
			checkResult(hostRef2x2, out);
			cout << "Time Elapsed for Convolution(Global to Shared Mem Copy) on GPU with filter size " << FILT2X2 << "x" << FILT2X2 << " is: " << time_elapsed << "milliSeconds" << endl;
			cudaFree(d_A);
			cudaFree(d_B);
			cudaFree(d_C);
			cudaEventDestroy(start);
			cudaEventDestroy(end);
			cudaDeviceReset();
		}
		else if (loop == 4)
		{


			//Convolution on GPU  with Global to Register Copy
			float *d_A, *d_B, *d_C;
			dim3 dim_grid, dim_block;
			dim3 dimBlock(16, 16);
			dim3 dimGrid(ceil((COL + 16 - 1) / 16), ceil((ROW + 16 - 1) / 16));
			dim_block.x = TILE_DIM;
			dim_block.y = TILE_DIM;
			dim_grid.x = (COL + TILE_DIM - 1) / TILE_DIM;
			dim_grid.y = (ROW + TILE_DIM - 1) / TILE_DIM;
			cudaMalloc((void**)&d_A, ROW*COL * sizeof(float));
			cudaMalloc((void**)&d_B, ROW*COL * sizeof(float));
			cudaMalloc((void**)&d_C, FILT4X4*FILT4X4 * sizeof(float));
			cudaMemcpy(d_A, data, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(d_C, filter4x4, FILT4X4 * FILT4X4 * sizeof(float));
			float time_elapsed = 0;
			cudaEvent_t start, end;
			cudaEventCreate(&start);
			cudaEventCreate(&end);
			cudaEventRecord(start, 0);
			globalToShared4x4 << <dim_grid, dim_block >> >(d_B, d_A, filter4x4, 16);
			cudaThreadSynchronize();
			cudaEventRecord(end, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&time_elapsed, start, end);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyDeviceToHost);
			checkResult(hostRef4x4, out);
			cout << "Time Elapsed for Convolution(Global to Shared Mem Copy) on GPU with filter size " << FILT4X4 << "x" << FILT4X4 << " is: " << time_elapsed << "milliSeconds" << endl;
			cudaFree(d_A);
			cudaFree(d_B);
			cudaFree(d_C);
			cudaEventDestroy(start);
			cudaEventDestroy(end);
			cudaDeviceReset();
		}
		else if (loop == 8){

			//Convolution on GPU  with Global to Register Copy
			float *d_A, *d_B, *d_C;
			dim3 dim_grid, dim_block;
			dim3 dimBlock(16, 16);
			dim3 dimGrid(ceil((COL + 16 - 1) / 16), ceil((ROW + 16 - 1) / 16));
			dim_block.x = TILE_DIM;
			dim_block.y = TILE_DIM;
			dim_grid.x = (COL + TILE_DIM - 1) / TILE_DIM;
			dim_grid.y = (ROW + TILE_DIM - 1) / TILE_DIM;
			cudaMalloc((void**)&d_A, ROW*COL * sizeof(float));
			cudaMalloc((void**)&d_B, ROW*COL * sizeof(float));
			cudaMalloc((void**)&d_C, FILT8X8*FILT8X8 * sizeof(float));
			cudaMemcpy(d_A, data, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(d_C, filter8x8, FILT8X8 * FILT8X8 * sizeof(float));
			float time_elapsed = 0;
			cudaEvent_t start, end;
			cudaEventCreate(&start);
			cudaEventCreate(&end);
			cudaEventRecord(start, 0);
			globalToShared8x8 << <dim_grid, dim_block >> >(d_B, d_A, filter8x8, 64);
			cudaThreadSynchronize();
			cudaEventRecord(end, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&time_elapsed, start, end);
			cudaMemcpy(d_B, out, ROW*COL * sizeof(float), cudaMemcpyDeviceToHost);
			checkResult(hostRef8x8, out);
			cout << "Time Elapsed for Convolution(Global to Shared Mem Copy) on GPU with filter size " << FILT8X8 << "x" << FILT8X8 << " is: " << time_elapsed << "milliSeconds" << endl;
			cudaFree(d_A);
			cudaFree(d_B);
			cudaFree(d_C);
			cudaEventDestroy(start);
			cudaEventDestroy(end);
			cudaDeviceReset();

		}
	}

	return 0;
}


void checkResult(float *hostRef, float *gpuRef)
{
	double epsilon = 0.00000001;
	bool match = 1;

	for (int i = 0; i<ROW; i++)
	for (int j = 0; j<COL; j++)
	{
		if (abs(hostRef[i*COL + j] - gpuRef[i*COL + j]) > epsilon)
		{
			match = 0;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d %d\n", hostRef[i*COL + j], hostRef[i*COL + j], i, j);
			break;
		}
	}

	if (match)
		printf("Arrays match!\n\n");

	return;
}


void CPUconvolution(float *data, float *filter, float *output, int filterSize)
{

	int mm, nn, ii, jj = 0;
	clock_t start, end;
	start = clock();
	//######################### 2D Convolution on CPU ##########################################
	//##########################################################################################
	int x = filterSize / 2;
	int y = filterSize / 2;

	for (int i = 0; i <ROW; ++i)
	{
		for (int j = 0; j < COL; ++j)
		{
			for (int m = 0; m < filterSize; ++m)
			{
				mm = filterSize - 1 - m;
				for (int n = 0; n < filterSize; ++n) 
				{
					nn = filterSize - 1 - n; 
					ii = i + (m - y);
					jj = j + (n - x);
					// ignore input samples which are out of bound
						if (ii >= 0 && ii < ROW && jj >= 0 && jj < COL)
							output[i*COL + j] += data[ii*COL + jj] * filter[mm*COL+nn] ;
				}
			}
		}
	}
	
	end = clock();
	cout << "Time Elapsed for Convolution on CPU with filter size "<< filterSize << "x"<< filterSize <<" is: "<< (end-start) << "milliSeconds" << endl;
	
	
}





