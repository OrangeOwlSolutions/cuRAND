// --- Generate random numbers with cuRAND's Mersenne Twister

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>
#include <curand_kernel.h>
/* include MTGP host helper functions */
#include <curand_mtgp32_host.h>

#define BLOCKSIZE	256
#define GRIDSIZE	64

/*******************/
/* GPU ERROR CHECK */
/*******************/
#define gpuErrchk(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

/*******************/
/* iDivUp FUNCTION */
/*******************/
__host__ __device__ int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/*********************/
/* GENERATION KERNEL */
/*********************/
__global__ void generate_kernel(curandStateMtgp32 * __restrict__ state, float * __restrict__ result, const int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	for (int k = tid; k < N; k += blockDim.x * gridDim.x)
		result[k] = curand_uniform(&state[blockIdx.x]);
}

/********/
/* MAIN */
/********/
int main()
{
	const int N = 217 * 123;

	// --- Allocate space for results on host
	float *hostResults = (float *)malloc(N * sizeof(float));

	// --- Allocate and initialize space for results on device 
	float *devResults; gpuErrchk(cudaMalloc(&devResults, N * sizeof(float)));
	gpuErrchk(cudaMemset(devResults, 0, N * sizeof(float)));

	// --- Setup the pseudorandom number generator
	curandStateMtgp32 *devMTGPStates; gpuErrchk(cudaMalloc(&devMTGPStates, GRIDSIZE * sizeof(curandStateMtgp32)));
	mtgp32_kernel_params *devKernelParams; gpuErrchk(cudaMalloc(&devKernelParams, sizeof(mtgp32_kernel_params)));
	CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams));
	//CURAND_CALL(curandMakeMTGP32KernelState(devMTGPStates, mtgp32dc_params_fast_11213, devKernelParams, GRIDSIZE, 1234));
	CURAND_CALL(curandMakeMTGP32KernelState(devMTGPStates, mtgp32dc_params_fast_11213, devKernelParams, GRIDSIZE, time(NULL)));

	// --- Generate pseudo-random sequence and copy to the host
	generate_kernel << <GRIDSIZE, BLOCKSIZE >> >(devMTGPStates, devResults, N);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(hostResults, devResults, N * sizeof(float), cudaMemcpyDeviceToHost));

	// --- Print results
	//for (int i = 0; i < N; i++) {
	for (int i = 0; i < 10; i++) {
		printf("%f\n", hostResults[i]);
	}

	// --- Cleanup
	gpuErrchk(cudaFree(devMTGPStates));
	gpuErrchk(cudaFree(devResults));
	free(hostResults);

	return 0;
}
