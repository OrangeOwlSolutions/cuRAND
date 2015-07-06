#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Utilities.cuh"

#define BLOCKSIZE 256

/******************************/
/* SEED INITIALIZATION KERNEL */
/******************************/
__global__ void cuRAND_setup_kernel(unsigned long seed, curandState ∗ __restrict__ state, const int N) {

	int tid = threadIdx.x + blockIdx.x ∗ blockDim.x;

	// --- Each thread gets the same seed (seed), a different sequence number (tid), no offset (0)
    if (tid < N) curand_init(seed, tid, 0, &state[tid]);
}

/*********************/
/* GENERATION KERNEL */
/*********************/
__global__ void cuRAND_generation_kernel(curandState ∗ __restrict__ state,  int ∗ __restrict__ d_random_numbers, const int N) {

	int tid = threadIdx.x + blockIdx.x ∗ blockDim.x;

	if (tid < N) {
		curandState localState  = state[tid];
		d_random_numbers[tid]	= curand(&localState) % 64;
		state[tid] = localState;
	}
}

/********/
/* MAIN */
/********/
int main() {
	const int N = 10;

    int *h_random_numbers  = (int*)malloc(N * sizeof(int));
    int *d_random_numbers;		gpuErrchk(cudaMalloc((void**)&d_random_numbers, N * sizeof(int)));

	curandState ∗devStates;		gpuErrchk(cudaMalloc((void**)&devStates, N * sizeof(devStates)));

	cuRAND_setup_kernel<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(1234, devStates, N);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	cuRAND_generation_kernel<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(devStates, d_random_numbers, N);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_random_numbers, d_random_numbers, N * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i=0; i<N; i++) printf("%i %i\n", i, h_random_numbers[i]);

}
