#include <cstdio>
#include <omp.h>

__global__ void block(float *a, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=N) return;
  a[i] = i;
}

int main(void) {
  const int N = 2000;
  const int M = 1024;
  int num_gpus;
  cudaGetDeviceCount(&num_gpus);
  float *a;
#pragma omp parallel num_threads(num_gpus)
  {
    int device = omp_get_thread_num();
    printf("%d\n",device);
    cudaSetDevice(device);
    cudaMallocManaged(&a, N*sizeof(float));
    block<<<(N+M-1)/M,M>>>(a,N);
    cudaDeviceSynchronize();
  }
  for (int i=0; i<N; i++)
    printf("%d %g\n",i,a[i]);
  cudaFree(a);
}
