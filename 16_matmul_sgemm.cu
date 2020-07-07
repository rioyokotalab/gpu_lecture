#include <cmath>
#include <cublas_v2.h>
#include <cstdlib>
#include <cstdio>
#include <chrono>
using namespace std;

#define M 1024

int main(int argc, char **argv) {
  int N = 2048;
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
  float alpha = 1.0;
  float beta = 0.0;
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);
  stat = cublasSetMatrix(N, N, sizeof(*A), A, N, A, N);
  stat = cublasSetMatrix(N, N, sizeof(*B), B, N, B, N);
  stat = cublasSetMatrix(N, N, sizeof(*C), C, N, C, N);
  auto tic = chrono::steady_clock::now();
  stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                     &alpha, B, N, A, N, &beta, C, N);
  cudaDeviceSynchronize();
  auto toc = chrono::steady_clock::now();
  stat = cublasGetMatrix(N, N, sizeof(*C), C, N, C, N);
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
  cublasDestroy(handle);
}
