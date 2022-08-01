# Library

> Reference
>
> 1. Professional CUDA C Programming chapter 8



### Overview

* Libraries

| library  | domain                                       |
| -------- | -------------------------------------------- |
| cuFFT    | FFT                                          |
| cuBLAS   | BLAS 1,2,3                                   |
| CULA     | Linear Algebra                               |
| cuSPARSE | Sparse Linear Algebra                        |
| CUSP     | Sparse Linear Algebra and Graph Computations |
| cuRAND   | Random Number Generation                     |
| Thrust   | Parallel Algo and Data Structure             |
|          |                                              |



* 常见调用CUDA library过程

1. create library handle
   1. contains contextual library information such as the format of data structures used, the devices used for computation, and other environmental data.
   2. you must allocate and initialize the handle before making any library calls.
2. allocate device memory for input
3. convert input to library-support format
4. copy host data to device memory for library
   1. analogous to cudaMemcpy, though in many cases a library-specific function is used. 
   2. when transferring a vector from the host to the device in a cuBLAS-based application, cublasSetVector should be used. 底层使用stride call to cudaMemcoy
5. config library
   1. config through parameter
   2. config thorugh function handle
6. executing
7. retrieving result from device memory
8. convert result back to original format
9. release CUDA resoruce (handler)
   1. there is some overhead in allocating and releasing resources, so it is bet- ter to reuse resources across multiple invocations of CUDA library calls when possible.





### cuSPARSE

cuSPARSE includes a range of general-purpose sparse linear algebra routines.

1. Level 1 functions operate exclusively on dense and sparse vectors.
2. Level 2 functions operate on sparse matrices and dense vectors.  
3. Level 3 functions operate on sparse matrices and dense matrices.



* function call

<img src="Note.assets/Screen Shot 2022-07-31 at 7.05.31 PM.png" alt="Screen Shot 2022-07-31 at 7.05.31 PM" style="zoom:50%;" />



* data format

<img src="Note.assets/Screen Shot 2022-07-31 at 7.06.09 PM.png" alt="Screen Shot 2022-07-31 at 7.06.09 PM" style="zoom:50%;" />



* data format conversion

<img src="Note.assets/Screen Shot 2022-07-31 at 7.06.21 PM.png" alt="Screen Shot 2022-07-31 at 7.06.21 PM" style="zoom:50%;" />



* 常见注意

1. ensuring proper matrix and vector formatting.
   1.  错误的format会导致segfault/validation error
2. check conversion 是否成功
   1. Automated full dataset verification might be possible by performing the inverse format conversion back to the native data format, and verifying that the twice-converted values are equivalent to the original values. 通过正反两次conversion来验证
3. scalar parameter是以reference pass in的



### cuBLAS

cuBLAS includes CUDA ports of all functions in the standard Basic Linear Algebra Subprograms (BLAS) library for Levels 1, 2, and 3.

For compatibility reasons, the cuBLAS library also chooses to use column-major storage.

1. cuBLAS Level 1 contains vector-only operations like vector addition. 
2. cuBLAS Level 2 contains matrix-vector operations like matrix-vector multiplication. 
3. cuBLAS Level 3 contains matrix-matrix operations like matrix-multiplication. 



两种API，legacy cuBLAS API is deprecated, 使用current cuBLAS API



* data transform 

use custom cuBLAS routines such as cublasSetVector/cublasGetVector and cublasSetMatrix/cublasGetMatrix to transfer data between the host and device. . Although you can think of these specialized functions as wrappers around cudaMemcpy, they are well-optimized to transfer both strided and unstrided data. 使用cuBLAS特定的data transfer routine, 这些routine是被optimized的给传输数据



* 注意

1. If you commonly use row-major programming languages, development with cuBLAS can require extra attention to detail.



### cuFFT

cuFFT includes methods for performing fast Fourier transforms (FFTs) and their inverse.

An FFT is a transformation in signal processing that converts a signal from the time domain to the frequency domain. An inverse FFT does the opposite. 

两个部分

1.  the core, high-performance cuFFT library
2.  the portability library, cuFFTW

cuFFTW is designed to maximize portability from existing code that uses FFTW. A wide range of the functions in the FFTW library are identi- cally supported in cuFFTW. In addition, the cuFFTW library assumes all inputs passed are in host memory and handles all of the allocation (cudaMalloc) and transfers (cudaMemcpy) for the user. Although this might lead to suboptimal performance, it greatly accelerates the porting process. cuFFTW是为了portablity，与FFTW的API和使用方法一致，但是有suboptimal performence

cuFFT的handler叫做plans



input output data type

1. Complex to comple 
2. real to complex 
3. complex to real 



### cuRAND

cuRAND includes methods for rapid random number generation using the GPU.

支持两种random value generation

1. pseudo-random RNGs (PRNG)
   1. A pseudo-random RNG (PRNG) uses an RNG algorithm to produce a sequence of random numbers where each value has an equal probability of being anywhere along the range of valid values for the storage type that RNG uses.
   2. When true randomness is required, a PRNG is a better choice
2. quasi-random RNGs (QRNG)
   1. A QRNG makes an effort to fill the range of the output type evenly. Hence, if the last value sampled by a QRNG was 2, then P(2) for the next value has actually decreased. The samplings of a QRNG’s sequence are not statis- tically independent.
   2. QRNGs are useful in exploring spaces that are largely not understood. They guarantee a more even sampling of a multi-dimensional space than PRNGs but might also find fea- tures that a regular sampling interval will miss.



