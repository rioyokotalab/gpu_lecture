00_hello: 00_hello.cu
	nvcc 00_hello.cu
01_thread: 01_thread.cu
	nvcc 01_thread.cu
02_block: 02_block.cu
	nvcc 02_block.cu
03_if: 03_if.cu
	nvcc 03_if.cu
04_atomic: 04_atomic.cu
	nvcc 04_atomic.cu
05_reduction: 05_reduction.cu
	nvcc 05_reduction.cu
06_shared: 06_shared.cu
	nvcc 06_shared.cu
07_warp: 07_warp.cu
	nvcc 07_warp.cu
08_scan: 08_scan.cu
	nvcc 08_scan.cu
09_cooperative: 09_cooperative.cu
	nvcc 09_cooperative.cu -arch=sm_60 -rdc=true
10_matmul: 10_matmul.cu
	nvcc 10_matmul.cu -Xcompiler "-O3 -fopenmp" -lcublas
11_matmul_shared: 11_matmul_shared.cu
	nvcc 11_matmul_shared.cu -Xcompiler "-O3 -fopenmp" -lcublas
12_block_8x8: 12_block_8x8.cu
	nvcc 12_block_8x8.cu -Xcompiler "-O3 -fopenmp" -lcublas
13_reg_load: 13_reg_load.cu
	nvcc 13_reg_load.cu -Xcompiler "-O3 -fopenmp" -lcublas
14_align: 14_align.cu
	nvcc 14_align.cu -Xcompiler "-O3 -fopenmp" -lcublas
15_warp: 15_warp.cu
	nvcc 15_warp.cu -Xcompiler "-O3 -fopenmp" -lcublas
16_vector_4x2: 16_vector_4x2.cu
	nvcc 16_vector_4x2.cu -Xcompiler "-O3 -fopenmp" -lcublas
17_check: 17_check.cu
	nvcc 17_check.cu
