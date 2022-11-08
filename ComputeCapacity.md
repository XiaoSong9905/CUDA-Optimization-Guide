# Computation Capacity

> Reference
>
> 1. CUDA C++ Programming Guide chapter K [link](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
> 2. Wiki CUDA [link](https://en.wikipedia.org/wiki/CUDA)
> 3. Professional CUDA C Programming chapter 3 (for cc 2.x, 3.x)



> TODO 还需要补充不同cc下，在访问global memory的时候，对于cache的使用。尤其是5.x 开始的都没有重新总结



## 1.x Tesla



## 2.x Fermi

* each SM contain

1. 32 single-precision CUDA cores
   1. fully pipelined ALU and FPU per CUDA core: one float / int per clock cycle
   2. 在Fermi中并没有专门int或者专门float的arch，而是CUDA Core承担float/int的计算
2. two warp scheduler and two dispatch unit
   1. dynamic scheduling 
   2. two scheduler select two warps 
   3. issue one instruction from each warp to a group of 16 CUDA cores, 16 load/store units, or 4 SFU
3. 4 SFU: one instruction per thread per clock cycle
4. 64 KB on-chip configureable shared memory / L1 cache 
5. 16 load/store unit (Figure 3.1)
6. 32 KB register file



* across SM

1. 16 SM total
2. GigaThread engine : distribute thread blocks to SM warp schedulers
3. 768 KB L2 cache



* concurrency

1. 16-way concurrency. 16 kernels to be run on device at same time



<img src="Note.assets/Screen Shot 2022-07-31 at 12.06.06 PM.png" alt="Screen Shot 2022-07-31 at 12.06.06 PM" style="zoom:50%;" />

<img src="Note.assets/Screen Shot 2022-07-31 at 12.07.20 PM.png" alt="Screen Shot 2022-07-31 at 12.07.20 PM" style="zoom:50%;" />



## 3.x Kepler

* Each SM contain

1. 192 single-precision CUDA core
2. 64 double-precision unit (DP Unit)
3. 32 special function unit for single precision float (SFU)
4. 32 load/store unit (LD/ST)
5. 4 warp scheduler and 8 dispatch
   1. dynamic scheduling
   2. four warp scheduler select four warp to execute
   3. at every instruction issue time, each scheduler issues two (因为2 dispatch per scheduler) independent instructions for one of its assigned warps that is ready to execute

6. 64 KB on-chip configureable shared memory / L1 cache 
   1. L1 cache for load local memory nd register spill over

   2. cc 3.5 3.7 可以opt-in to cache global memory access on both L1 & L2 通过compiler `-Xptxas -dlcm=ca`，但是默认global memory访问不经过L1 

7. 64 KB register file
8. per SM read-only data cache 
   1. 用于read from constant memory
9. per SM read-only texture cache 48kb
   1. cc 3.5 3.7 可以用于read global memory
   2. 也可以被texture memory使用
   3. 与L1 cache不是unified的



* across SM contain

1. 1.5MB L2 cache 用于 local memory & global memory



* feature

1. 32 hardware work queue for hyper-q
2. dynamic parallel



<img src="Note.assets/Screen Shot 2022-07-31 at 12.17.39 PM.png" alt="Screen Shot 2022-07-31 at 12.17.39 PM" style="zoom:50%;" />

<img src="Note.assets/Screen Shot 2022-07-31 at 12.25.40 PM.png" alt="Screen Shot 2022-07-31 at 12.25.40 PM" style="zoom:50%;" />



## 5.x Maxwell

* each SM contain

1. 128 CUDA core for arithmetic op
2. 32 special function unit for single precision float
3. 4 warp scheduler
   1. dynamic scheduling
   2. each warp scheduler issue 1 instruction per clock cycle
4. L1 cache/texture cache of 24 KB
   1. 在某些条件下可以通过config来用于访问global memory
   2. default not enable L1 cache for global memory access
   3. L1 cache 与 read only texture cache 是unified的，这点与3.x是不一样的。

5. shared memory 64KB/96KB
   1. 这里shared memory与L1 cache不再是同一个chip了

6. read-only constant cache
   1. 用于constant memory access




* across SM

1. L2 cache 用于 local or global memory



## 6.x Pascal

* Each SM core

1. 64 (cc 6.0) / 128 (cc 6.1 & 6.2) CUDA core for arithemetic
2. 16 (cc 6.0) / 32 (cc 6.1 & 6.2) special function unit for single precision float 
3. 2 (cc 6.0) / 4 (cc 6.1 & 6.2) warp scheduler 
   1. dynamic assign warp to warp scheduler. When an SM is given warps to execute, it first distributes them among its schedulers
   2. each scheduler issues one instruction for one of its assigned warps that is ready to execute

4. read-only constant cache
   1. read from constant memory space

5. L1/texture cache of size 24 KB (6.0 and 6.2) or 48 KB (6.1),
   1. 是unified的
   2. 用于read global memory

6. shared memory of size 64 KB (6.0 and 6.2) or 96 KB (6.1).



* across SM

1. L2 cache 用于 local or global memory



## 7.x Volta & Turing

* each SM

1. 4 processing block, each contain 
   1. 16 FP32, 16 INT32, 8 FP64, 2 tensor core, L0 instruction cache, one warp scheduler, one dispatch unit, and 64KB Register file
   2. static distribute warp among schedulers. each scheduler issue one instruction for one of its assigned warp per clock cycle. 支持independent thread scheduling

2. read only constant cache 用于 constant memory space
3. unified L1 & shared memory of size 128 KB (volta) / 96 KB (Turing)
   1. can be configued between l1 & shared memory. 
   2. driver automatically configures the shared memory capacity for each kernel to avoid shared memory occupancy bottlenecks while also allowing concurrent execution with already launched kernels where possible. In most cases, the driver's default behavior should provide optimal performance. 自动config shared memory的大小，大多数情况是optimal的
   3. default enable L1 cache for global memory access
   4. smem bank same as Fermi.



* Across SM

L2 cache



## 8.x Ampere

### Resource

* each SM have 

1. 64 FP32 cores for single-precision arithmetic operations in devices of compute capability 8.0 and 128 FP32 cores in devices of compute capability 8.6,
2. 32 FP64 cores for double-precision arithmetic operations in devices of compute capability 8.0 and 2 FP64 cores in devices of compute capability 8.6
3. 64 INT32 cores for integer math,
4. 4 mixed-precision Third Generation Tensor Cores supporting half-precision (fp16), __nv_bfloat16, tf32, sub-byte and double precision (fp64) matrix arithmetic (see Warp matrix functions for details),
5. 16 special function units for single-precision floating-point transcendental functions,
6. 4 warp schedulers.
7. read only constant cache
8. unified L1 & shared memory
   1. can be configed 
   2. default enable L1 cache for global memory access



### schedule

static distribute warp to scheduler

each scheduler issue 1 instruction each clock cycle



### global and shared memory

global same as 5.x

shared memory bank same as 5.x

shared memory configuration same as 7.x



## 9.x Hopper & Lovelace