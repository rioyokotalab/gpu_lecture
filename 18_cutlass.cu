#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <chrono>
using namespace std;

__global__ void kernel(int dim_m, int dim_n, int dim_k,
		       float *d_a, float *d_b, float *d_c) {
  const int ItemsPerVector = 4;
  const int VectorsPerThread = 2;
  const int ItemsPerThread = VectorsPerThread * ItemsPerVector; // 8

  const int ThreadsPerWarpY = 4;
  const int ThreadsPerWarpX = 8;
  const int ThreadsPerWarp = ThreadsPerWarpX * ThreadsPerWarpY; // 32
  const int WarpsPerBlockX = 1;
  const int ThreadsPerBlock = 64;

  const int ItemsPerWarpY = ThreadsPerWarpY * ItemsPerThread; // 32
  const int ItemsPerWarpX = ThreadsPerWarpX * ItemsPerThread; // 64
  const int ItemsPerBlockX = WarpsPerBlockX * ItemsPerWarpX; // 64

  const int Ktile = 8;
  const int VectorsPerMtile = ThreadsPerWarpX * VectorsPerThread; // 16 A #rows
  const int ThreadsPerKtile = ThreadsPerBlock / VectorsPerMtile; // 4 A #cols
  const int VectorsPerKtile = Ktile / ItemsPerVector; // 2 B #rows
  const int ThreadsPerNtile = ThreadsPerBlock / VectorsPerKtile; // 32 B #cols

  struct __align__(16) vec_t { float d[ItemsPerVector]; };
  vec_t *tile_a;
  vec_t *tile_b;
  vec_t __align__(16) thread_a[VectorsPerThread];
  vec_t __align__(16) thread_b[VectorsPerThread];
  __shared__ float __align__(16) block_a[Ktile][ItemsPerBlockX];
  __shared__ float __align__(16) block_b[Ktile][ItemsPerBlockX];
  float __align__(16) fragment_a[ItemsPerThread];
  float __align__(16) fragment_b[ItemsPerThread];
  float __align__(16) fragment_c[ItemsPerThread][ItemsPerThread];

  int offset_a_m = ItemsPerBlockX * blockIdx.x / ItemsPerVector;
  int offset_b_n = ItemsPerBlockX * blockIdx.y;
  int lda = dim_m / ItemsPerVector;
  int ldb = dim_k / ItemsPerVector;
  int a_m = threadIdx.x % VectorsPerMtile; // 16
  int a_k = threadIdx.x / VectorsPerMtile; // 4
  int b_k = threadIdx.x % VectorsPerKtile; // 2
  int b_n = threadIdx.x / VectorsPerKtile; // 32
  tile_a = reinterpret_cast<vec_t*>(&d_a[(a_k * lda + (a_m + offset_a_m)) * ItemsPerVector]);
  tile_b = reinterpret_cast<vec_t*>(&d_b[((b_n + offset_b_n) * ldb + b_k) * ItemsPerVector]);
  for (int m = 0; m < ItemsPerThread; ++m)
    for (int n = 0; n < ItemsPerThread; ++n)
      fragment_c[m][n] = 0;

  int warp_id = threadIdx.x / ThreadsPerWarp; // 2
  int warp_x = warp_id % WarpsPerBlockX; // 2
  int warp_y = warp_id / WarpsPerBlockX; // 1
  int lane_id = threadIdx.x % ThreadsPerWarp; // 32
  int lane_x = lane_id / ThreadsPerWarpY; // 8
  int lane_y = lane_id % ThreadsPerWarpY; // 4
  int offset_y = lane_y * ItemsPerVector + warp_y * ItemsPerWarpY; // 1*4*vec
  int offset_x = lane_x * ItemsPerVector + warp_x * ItemsPerWarpX; // 2*8*vec
  int stride_a = 0;
  int stride_b = 0;
  for (int kk = 0; kk < dim_k; kk += Ktile) {
    for (int i = 0; i < VectorsPerThread; ++i) {
      thread_a[i] = tile_a[stride_a + i * ThreadsPerKtile * lda]; // 2*a_k*vec
      thread_b[i] = tile_b[stride_b + i * ThreadsPerNtile * ldb]; // 2*b_n*vec
    }
    __syncthreads();
    for (int i = 0; i < VectorsPerThread; ++i) {
      for (int j = 0; j < ItemsPerVector; ++j) {                                         //    4       16
	block_a[a_k + i * ThreadsPerKtile][a_m * ItemsPerVector + j] = thread_a[i].d[j]; // (2*a_k) * (a_m*vec)
	block_b[b_k * ItemsPerVector + j][b_n + i * ThreadsPerNtile] = thread_b[i].d[j]; // (2*b_n) * (b_k*vec) ^T
      }                                                                                  //    32      2
    }
    __syncthreads();
    stride_a += lda * Ktile;
    stride_b += Ktile / ItemsPerVector;
#pragma unroll
    for (int k = 0; k < Ktile; k++) {
      for (int i = 0; i < VectorsPerThread; ++i) {
	for (int j = 0; j < ItemsPerVector; ++j) {
	  fragment_a[i * ItemsPerVector + j] = block_a[k][offset_y + (i * ThreadsPerWarpY * ItemsPerVector) + j]; // 4
	  fragment_b[i * ItemsPerVector + j] = block_b[k][offset_x + (i * ThreadsPerWarpX * ItemsPerVector) + j]; // 8
	}
      }
      for (int m = 0; m < ItemsPerThread; ++m) {
	for (int n = 0; n < ItemsPerThread; ++n) {
	  fragment_c[m][n] += fragment_a[m] * fragment_b[n];
	}
      }
    }
  }
  for (int ix = 0; ix < ItemsPerThread; ++ix) {
    for (int iy = 0; iy < ItemsPerThread; iy += ItemsPerVector) {
      int vx = ix / ItemsPerVector;
      int vy = iy / ItemsPerVector;
      int tx = offset_x + (vx * ThreadsPerWarpX * ItemsPerVector) + (ix % ItemsPerVector);
      int ty = offset_y + (vy * ThreadsPerWarpY * ItemsPerVector) + (iy % ItemsPerVector);
      int bx = ItemsPerBlockX * blockIdx.y + tx;
      int by = ItemsPerBlockX * blockIdx.x + ty;
      for (int i = 0; i < ItemsPerVector; ++i) {
	if (bx < dim_n && (by + i) < dim_m) {
	  d_c[bx * dim_m + by + i] = fragment_c[iy + i][ix];
	}
      }
    }
  }
}

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;
  float *A, *B, *C, *C2;
  cudaMallocManaged(&A, m * k * sizeof(float));
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      C[m*i+j] = C2[m*i+j] = 0;
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    cublasSgemm(cublas_handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		m,
		n,
		k,
		&alpha,
		A,
		m,
		B,
		k,
		&beta,
		C,
		m);
    cudaDeviceSynchronize();
  }
  auto toc = chrono::steady_clock::now();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;
  int tile = 64;
  dim3 block = dim3(tile);
  dim3 grid = dim3((m+tile-1)/tile, (n+tile-1)/tile);
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<< grid, block >>>(m,
			      n,
			      k,
			      A,
			      B,
			      C2);
    cudaDeviceSynchronize();
  }
  toc = chrono::steady_clock::now();
  double tcutlass = chrono::duration<double>(toc - tic).count() / Nt;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C[m*i+j] - C2[m*i+j]);
    }
  }
  printf("error: %lf\n", err/n/m);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C2);
  cublasDestroy(cublas_handle);
}
