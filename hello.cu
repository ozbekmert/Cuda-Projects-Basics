
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<stdio.h>


__global__ void helloFromGPU(void)
{
	printf("Hello World from GPU! \n");
}

int main()
{
	helloFromGPU <<<1, 10 >> >();
	cudaDeviceReset();
	return 0;
}
