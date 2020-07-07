#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
using namespace std;

__global__ void matmul(float *A, float *B, float *C, int N) {
  int i = blockIdx.x;
  int j = threadIdx.x;
  float sum = 0;
  for (int k=0; k<N; k++) {
    sum += A[N*i+k] * B[N*k+j];
  }
  C[N*i+j] = sum;
}

int main(int argc, char **argv) {
  int N = 1024;
  int size = N * N * sizeof(float);
  float *A, *B, *C;
  cudaMallocManaged(&A, size);
  cudaMallocManaged(&B, size);
  cudaMallocManaged(&C, size);
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
      C[N*i+j] = 0;
    }
  }
  auto tic = chrono::steady_clock::now();
  matmul<<<N,N>>>(A, B, C, N);
  cudaDeviceSynchronize();
  auto toc = chrono::steady_clock::now();
  double time = chrono::duration<double>(toc - tic).count();
  printf("N=%d: %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9);
  #pragma omp parallel for
  for (int i=0; i<N; i++)
    for (int k=0; k<N; k++)
      for (int j=0; j<N; j++)
        C[N*i+j] -= A[N*i+k] * B[N*k+j];
  double err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(C[N*i+j]);
  printf("error: %lf\n",err/N/N);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}
