#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define DSIZE 8192*16
#define nTPB 256

/***********************/
/* CUDA ERROR CHECKING */
/***********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line); 
        if (abort) exit(code);
    }
}

/*************************/
/* CURAND INITIALIZATION */
/*************************/
__global__ void initCurand(curandState *state, unsigned long seed){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void testrand1(curandState *state, float *a){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    a[idx] = curand_uniform(&state[idx]);
}

__global__ void testrand2(unsigned long seed, float *a){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(seed, idx, 0, &state);
    a[idx] = curand_uniform(&state);
}

/********/
/* MAIN */
/********/
int main() {

    int n_iter = 20;

    curandState *devState;  gpuErrchk(cudaMalloc((void**)&devState, DSIZE*sizeof(curandState)));

    float *d_a;             gpuErrchk(cudaMalloc((void**)&d_a, DSIZE*sizeof(float)));

    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i=0; i<n_iter; i++) {

        initCurand<<<(DSIZE+nTPB-1)/nTPB,nTPB>>>(devState, 1);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        testrand1<<<(DSIZE+nTPB-1)/nTPB,nTPB>>>(devState, d_a);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Elapsed time for separate kernels:  %3.1f ms \n", time);

    cudaEventRecord(start, 0);

    for (int i=0; i<n_iter; i++) {

        testrand2<<<(DSIZE+nTPB-1)/nTPB,nTPB>>>(1, d_a);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Elapsed time for single kernels:  %3.1f ms \n", time);

    cudaEventRecord(start, 0);

    for (int i=0; i<n_iter; i++) {

        initCurand<<<(DSIZE+nTPB-1)/nTPB,nTPB>>>(devState, 1);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        testrand1<<<(DSIZE+nTPB-1)/nTPB,nTPB>>>(devState, d_a);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        testrand1<<<(DSIZE+nTPB-1)/nTPB,nTPB>>>(devState, d_a);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        testrand1<<<(DSIZE+nTPB-1)/nTPB,nTPB>>>(devState, d_a);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Elapsed time for separate kernels with multiple random number generation:  %3.1f ms \n", time);

    cudaEventRecord(start, 0);

    for (int i=0; i<n_iter; i++) {

        testrand2<<<(DSIZE+nTPB-1)/nTPB,nTPB>>>(1, d_a);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        testrand2<<<(DSIZE+nTPB-1)/nTPB,nTPB>>>(1, d_a);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        testrand2<<<(DSIZE+nTPB-1)/nTPB,nTPB>>>(1, d_a);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Elapsed time for single kernels for multiple random number generation:  %3.1f ms \n", time);

    getchar();
}
