# Memory Model

## Memory Hierarchy

> Reference
>
> 1. CUDA C++ Best Practice Guide chapter 9.2
> 1. Programming Massively Parallel Processors 3rd edition 3rd chapter 5

<img src="Note.assets/Screen Shot 2022-06-21 at 9.48.33 PM.png" alt="Screen Shot 2022-06-21 at 9.48.33 PM" style="zoom:50%;" />



pointer arre used to point to data objects in global memory



* latency 

| type                       | clock cycle                             |
| -------------------------- | --------------------------------------- |
| register                   | 1                                       |
| shared memory              | 5                                       |
| local memory               | 500 (without cache)                     |
| global memory              | 500                                     |
| constant memory with cache | 1(same as register)-5(same as L1 cache) |
| L1 cache                   | 5                                       |



* 为什么重视内存访问

对于GPU programming来说，one must have a clear understanding of the desirable (e.g., gather in CUDA) and undesirable (e.g., scatter in CUDA) memory access behaviors to make a wise decision.



## Global Memory

### Bandwidth

> Reference
>
> 1. CUDA C++ Best Practices Guide chapter 8



* device memory 可以分为两类

1. linear memory
2. CUDA arrays。常用于texture



#### Timing

##### CPU Timer

相比起GPU event来说，比较粗糙

```cpp
// sync all kernel on device before timer
cudaDeviceSynchronize();

// start CPU timer

// do some work

// sync all kernel on device before timer
cudaDeviceSynchronize();

// end CPU timer

// compute time
```



##### GPU Event 

使用GPU clock，所以是OS independent的

```cpp
cudaEvent_t start, stop;
float time;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// start event
cudaEventRecord( start, 0 );

// do some work
kernel<<<grid,threads>>> ( d_odata, d_idata, size_x, size_y, NUM_REPS);

// end event
cudaEventRecord( stop, 0 );

// wait for end event
cudaEventSynchronize( stop );

// compute time
cudaEventElapsedTime( &time, start, stop );
cudaEventDestroy( start );
cudaEventDestroy( stop );
```



#### Bandwith calculation

##### Theoratical

* HBM2 example

NVIDIA Tesla V100 uses HBM2 (double data rate) RAM with a memory clock rate of 877 MHz and a 4096-bit-wide memory interface.


$$
0.877 * 10^9 * 4096 / 8 * 2 / 10^9 = 898GB/s
$$


* GDDR

enable ECC的GDDR内存因为有ECC overhead会导致theoretical bandwidth降低

HBM2因为有专门给ECC的部分，所以没有ECC overhead



##### Effective

$$
((B_r + B_w) / 10^9 ) / time
$$



#### Visual profiler 

* requested global load/store throughput 

程序使用的内存的bandwidth，对应effective bandwidth

不考虑memory line/cache line的大小



* global load / store throuhput

考虑到memory line要一起传送的带宽。

相当于物理理论上的带宽



希望requested throughput靠近global throughput, 这样才没有浪费bandwidth



### Device Memory DRAM

> Reference
>
> 1. UIUC ECE Lecture 4
> 2. Berkeley CS 267 Lecture 7 on memory colesing



##### Bit Line & Select Line

<img src="Note.assets/Screen Shot 2022-05-30 at 11.52.45 PM.png" alt="Screen Shot 2022-05-30 at 11.52.45 PM" style="zoom:50%;" />



* 原理

一个capacitor储存bit

一个select选择读取哪个capacitor

一个bit line read / write数据。每个bit line只读取一个bit的数据，也就是多个select里面只select一个

需要constantly check value / recharge value (where the dynamic name come from)



* 特点

bit line的capacitance很大，导致速度很慢。

bit的capacitance很小，需要使用sense amplifier来放大信号



<img src="Note.assets/Screen Shot 2022-06-27 at 9.26.00 PM.png" alt="Screen Shot 2022-06-27 at 9.26.00 PM" style="zoom:50%;" />




##### Core Array & Burst

<img src="Note.assets/Screen Shot 2022-05-31 at 12.02.50 AM.png" alt="Screen Shot 2022-05-31 at 12.02.50 AM" style="zoom:50%;" />


多个bit line组成core array

数据传输分为两个部分。core array -> column latches / buffer -> mux pin interface 

core array -> buffer 的耗时比较久

buffer -> mux pin interface 的耗时相对较小

**burst** 当访问一个内存位置的时候，多个bit line的数据都会从core array传输到column latches （全部红色的line），然后再使用mux来选择传送给bus interace哪个数据 / one burst of memory access to get data that used by multiple attemps to read.

**burst size** 读取一次memory address，会有多少个数据从core array被放到buffer中。

常见的GPU burst size是 1024 bits / 128 bytes (from Fermi). 这里的burst size经常被叫做**line size**

当L1 cache disabled at compile time (default enable), burst size是32 bytes. 




##### Multiple Banks

<img src="Note.assets/Screen Shot 2022-05-31 at 12.07.59 AM.png" alt="Screen Shot 2022-05-31 at 12.07.59 AM" style="zoom:50%;" />



<img src="Note.assets/Screen Shot 2022-05-31 at 12.08.29 AM.png" alt="Screen Shot 2022-05-31 at 12.08.29 AM" style="zoom:50%;" />


只用burst并不能实现processor所需要的DRAM bandwidth。

因为bank访问core array cell的时间很长（上图蓝色部分）而实际使用bus interface传输数据时间很短（上图红色部分），通常比例是20:1， 如果只使用一个bank，interface bus会idle。所以需要在一个bus 上使用多个bank，来充分利用bus bandwidth。如果使用多个bank，大家交替使用interface bus，保证bus不会idle

通过多个bank链接到interface bus，从而让interface bus充分的使用，也就保证了每个时间都有数据从interface bus传送过来。



* 一个bus需要多少个bank？

如果访问core array与使用bus传输数据的时间比例是20:1，那么一个bus至少需要21个bank才能充分使用bus bandwidth。



一般bus有更多的bank，不仅仅是ratio+1，原因是

1. 使用更多的bank，更能让data spread out across bank。如果一块data只在一个bank上的话，需要多个burst才能完全访问（时间很久）。如果一块data在多个bank的话，可以overlap core array access time （总时间变短）
2. 每个bank可以存储的诗句有限，否则访问一个bank的latency会很大。



##### Multiple Channels

<img src="Note.assets/Screen Shot 2022-05-31 at 2.47.07 PM.png" alt="Screen Shot 2022-05-31 at 2.47.07 PM" style="zoom:50%;" />


modern Double data rate （DDR） bus可以传输two word of data in each clock cycle. 

假设bus clock speed是1GHz， 每秒钟只能传送 8 bytes / words * 2 words per clock * 1 GHz = 16 GB/sec. 但是一般GPU processor要求128GB/s的数据

单独一个channel/一个bus interface不足以达到processor要求DRAM bandwidth，所以需要使用多个channel。



##### Interleaved data distribution 

<img src="Note.assets/Screen Shot 2022-05-31 at 2.53.55 PM.png" alt="Screen Shot 2022-05-31 at 2.53.55 PM" style="zoom:50%;" />

是什么：把array spread across banks and channel in the memory system. 这样允许core array acccess time overlap, 减少总access time. 

为了实现max bandwidth, 对于数据的访问需要利用interleaved data distribution. 要让 memory accesses must be evenly distributed to the channels and banks. 



### Memory Coarlesed & Aligned

> Reference
>
> 1. UIUC ECE Lecture 4
> 2. Berkeley CS 267 Lecture 7 on memory colesing
> 3. NVIDIA Tech Blog Coalesced Transaction Size [link](https://forums.developer.nvidia.com/t/coalesced-transaction-size/24602)
> 4. Blog CUDA基础 4.3 内存访问模式 [link](https://face2ai.com/CUDA-F-4-3-内存访问模式/)
> 5. CUDA C++ Best Practices Guide chapter 9.2.1
> 6. Professional CUDA C Programming chapter 4
> 7. CUDA C++ Programing Guide chapter K.3
> 8. CUDA C++ Programming Guide chapter 3.2.2
> 9. CUDA C++ Programming Guide chapter 5.3.2
> 10. NVIDIA Tech Blog Cache behavior when loading global data to shared memory in Fermi [link]
> 11. NVIDIA Tech Blog Coalesed Transaction Size [link](https://forums.developer.nvidia.com/t/coalesced-transaction-size/24602)



#### What is memory Coarlesed

Memory operations are also issued per warp. When executing a memory instruction, each thread in a warp provides a memory address it is loading or storing. Cooperatively, the 32 threads in a warp present a single memory access request comprised of the requested addresses, which is serviced by one or more device memory transactions. 对于内存的request是以warp为单位进行issue的而不是thread为单位进行的。warp内的多个thread访问内存地址首先会以warp为单位合并为一个warp memory request，这个warp memory request由一个或者多个memory transaction满足。具体使用几个memory transaction取决于warp memory request访问的数据范围以及每个memory transaction的大小。一个memory transaction可以理解为一个ISA层面的memory 访问

global memory request一定会经过L2，是否经过L1取决于cc和config，是否经过read only texture cache取决于cc和code。(Figure 4.6)

<img src="Note.assets/Screen Shot 2022-08-01 at 1.32.01 PM.png" alt="Screen Shot 2022-08-01 at 1.32.01 PM" style="zoom:50%;" />



* CPU充分利用memory bandwidth

CPU有很大的cache，CPU thread访问连续的内存会被cache在per CPU Core的cache中。不同的CPU thread由于有不同的core，读取的数据会被不同的core的cache保存，所以不相互影响。

对于CPU来说，充分利用内存的方法是每个core负责一段连续的内存。e.g. thread 1 : array0-99; thread 2 : array 100-199; thread 3 : array 200-299.



* GPU 充分利用memory bandwidth

GPU的cache小，一个SM内的多个thread会共享L1 cache。thread0读取数据产生的cache会对thread1读取数据产生的cache产生影响。而且GPU是以warp为单位来issue memory request的。

when many warps execute on the same multiprocessor simultaneously, as is generally the case, the cache line may easily be evicted from the cache between iterations i and i+1. CUDA中充分利用bandwidth需要warp内的threads在某一个iteration/timestep内花费全部transaction data segment / cache line， 因为有很多warp同时在sm上运行，等下一个iteration的时候 cache line/DRAM buffer已经被清空了。



* 常用优化方法

1. aligned and coarlesed memory access 从而确保充分利用bandwidth
2. sufficent concurrent memory operation 从而确保可以hide latency
   1. loop unroll 从而增加independent memory access per warp, 减少hide latency所需要的active warp per sm
   2. modify execution configuration 从而确保每个SM都有足够的active warp。



#### What is memory Aligned

Aligned memory accesses occur when the first address of a device memory transaction is an even multiple of the cache granularity being used to service the transaction (either 32 bytes for L2 cache or 128 bytes for L1 cache). Performing a misaligned load will cause wasted bandwidth. 

Warp memory request的起始位置是cache line的偶数倍。如果使用L1 128bytes cache line的话则需要起始位置是128 bytes的偶数倍。如果使用L2 32 bytes cache line的话则需要起始位置是32 bytes的偶数倍



* image library

当读取image 文件的时候，library经常会padded width = multiply of burst size. 

如果没有padded的话，raw 1的起始位置会是misaligned from DRAM burst，导致读取的时候多读几个burst/memory segment，让速度变慢

padded info叫做 `pitch` 

<img src="Note.assets/IMG_463A2479525D-1.jpeg" alt="IMG_463A2479525D-1" style="zoom:50%;" />



* CUDA API

使用CUDA API分配数据是会align 256 bytes的

```cpp
// 1d, aligned to 256 bytes
cudaMalloc();
cudaMemcpy();
cudaFree();

// 2d 分配, aligned to 256 bytes
cudaMallocPitch();
cudaMemcpy2D();

// 3d, aligned to 256 bytes
cudaMalloc3D();
cudaMemcpy3D();
```



* Align on struct

Global memory instructions support reading or writing words of size equal to 1, 2, 4, 8, or 16 bytes. If this size and alignment requirement is not fulfilled, the access compiles to multiple instructions with interleaved access patterns that prevent these instructions from fully coalescing. CUDA支持的数据大小是1,2,4,8,16. 如果自定义的struct不是这些大小的话，则会导致产生多个non coarlesed transaction。

如果一个struct是7 bytes，那么padding成8 bytes会用coarlesed access。但是如果不paddig的话则会是多个transaction。

下面的marcro可以align struct从而确保coarlesed access

```cpp
struct __align__(16) {
  float x;
  float y;
  float z; 
};
```



#### Global Memory Read

注意： GPU L1 cache is designed for spatial but not temporal locality. Frequent access to a cached L1 memory location does not increase the probability that the data will stay in cache. L1 cache是用于spatial（连续读取array）而不是temporal（读取同一个位置的），因为cache line很容易被其余的thread evict。



* global memory load efficency

<img src="Note.assets/Screen Shot 2022-08-01 at 2.40.23 PM.png" alt="Screen Shot 2022-08-01 at 2.40.23 PM" style="zoom:50%;" />

nvprof 里面 gld_efficency metrics 就衡量了和这个



* Simple model

在128 bytes/32 bytes的模式下，会产生128 bytes/ 32 bytes / 64 bytes的memory transaction （32 bytes当four segment的时候也会是128 bytes）。如果不考虑的那么仔细，那么可以粗略的认为Global memory resides in device memory and device memory is accessed via 32-, 64-, or 128- byte memory transactions.



##### Read-only texture cache

CC 3.5+ 可以使用read only texture cache

The granularity of loads through the read-only cache is 32 bytes. 



##### CC 2.x Fermi

2.x default 使用 L1 + L2 cache

2.x 可以通过config disable L1 cache

```shell
// disable L1 cache
-Xptxas -dlcm=cg

// enable L1 cache
-Xptxas -dlcm=ca
```



* 当使用L1 + L2 128 bytes transaction的时候

If the size of the words accessed by each thread is more than 4 bytes, a memory request by a warp is first split into separate 128-byte memory requests that are issued independently. 如果每个thread请求的数据大于4 bytes（32 * 4 = 128)，则会被切分为多个128 bytes memory request来进行。

如果每个thread请求8 bytes，two 128-bytes memory request, one for each half-warp. 这样保证了每个传送的128 bytes数据都被充分利用(16 threads * 8 bytes each)

如果每个thread请求16 bytes，four 128-bytes memory requesy, one for each quarter-warp. 这样保证了传送的128 bytes数据被充分利用



每一个memory request会进一步被broken down to cache line request 然后issue independently



The addresses requested by all threads in a warp fall within one cache line of 128 bytes. Only a single 128-byte transaction is required to complete the memory load operation. 

<img src="Note.assets/Screen Shot 2022-08-01 at 2.24.59 PM.png" alt="Screen Shot 2022-08-01 at 2.24.59 PM" style="zoom:50%;" />



access is aligned and the referenced addresses are not consecutive by thread ID, but rather randomized within a 128-byte range. Because the addresses requested by the threads in a warp still fall within one cache line, only one 128-byte transaction is needed to fulfill this memory load operation. 只要warp memory request是在128 bytes transaction内，只会进行一个memory transaction。

<img src="Note.assets/Screen Shot 2022-08-01 at 2.26.20 PM.png" alt="Screen Shot 2022-08-01 at 2.26.20 PM" style="zoom:50%;" />



warp requests 32 consecutive four-byte data elements that are not aligned. The addresses requested by the threads in the warp fall across two 128-byte seg- ments in global memory. Because the physical load operations performed by an SM must be aligned at 128-byte boundaries when the L1 cache is enabled, two 128-byte transactions are required to ful- fill this memory load operation. 由于misalign导致产生两个128 bytes transaction

<img src="Note.assets/Screen Shot 2022-08-01 at 2.27.40 PM.png" alt="Screen Shot 2022-08-01 at 2.27.40 PM" style="zoom:50%;" />



 all threads in the warp request the same address

<img src="Note.assets/Screen Shot 2022-08-01 at 2.32.37 PM.png" alt="Screen Shot 2022-08-01 at 2.32.37 PM" style="zoom:50%;" />



threads in a warp request 32 four-byte addresses scattered across global memory.

<img src="Note.assets/Screen Shot 2022-08-01 at 2.28.42 PM.png" alt="Screen Shot 2022-08-01 at 2.28.42 PM" style="zoom:50%;" />



* 当使用L2 only 32 bytes transaction的时候

performend at granularity of 32 bytes memory segments

Memory transactions can be one, two, or four segments at a time. 注意这里是说一次memory transaction是one segment long / two segment long / four segment long. 尽管是four segment long 但是依旧是one memory transaction. 



The addresses for the 128 bytes requested fall within four segments, and bus utilization is 100 percent.

<img src="Note.assets/Screen Shot 2022-08-01 at 2.31.39 PM.png" alt="Screen Shot 2022-08-01 at 2.31.39 PM" style="zoom:50%;" />



memory access is aligned and thread accesses are not sequential, but randomized within a 128-byte range.

<img src="Note.assets/Screen Shot 2022-08-01 at 2.32.05 PM.png" alt="Screen Shot 2022-08-01 at 2.32.05 PM" style="zoom:50%;" />



all threads in the warp request the same data

<img src="Note.assets/Screen Shot 2022-08-01 at 2.32.59 PM.png" alt="Screen Shot 2022-08-01 at 2.32.59 PM" style="zoom:50%;" />



warp requests 32 4-byte words scattered across global memory. 

<img src="Note.assets/Screen Shot 2022-08-01 at 2.33.23 PM.png" alt="Screen Shot 2022-08-01 at 2.33.23 PM" style="zoom:50%;" />



##### CC 3.x Kepler

3.x default 使用 L2 cache，不使用L1 cache

3.5 / 3.7 可以使用read only texture cache

3.5 / 3.7 可以config使用L1 cache

L1 cache line size 128 bytes

L2 cache line size 32 bytes

当使用L2 cache only的时候，memory transaction是32 bytesEach memory transaction may be conducted by one, two, or four 32 bytes segments。可以减少over-fecth

当使用L1 + L2 cache的时候，memory transaction是128 bytes. Memory request 首先会去L1，如果L1 miss会去L2，如果L2 miss会去DRAM。



memory transaction在使用L1+L2 / L2 only的时候，与 Fermi 一样



##### CC 5.x Maxwell

5.x default使用L2 cache，行为与3.x使用L2 cache only一样，是32 bytes transaction

5.x 可以使用read only texture cache，是32 bytes transaction

5.x 可以config使用L1 cache（default不使用）



> TODO 不确定使用L1 cache的情况下的memory transaction



##### CC 6.x Pascal



> TODO 不确定使用L1 cache的情况下的memory transaction
>
> 不确定是否default enable L1 cache



#### Global Memory Write

The L1 cache is not used for store operations on either Fermi or Kepler GPUs, store operations are only cached in the L2 cache before being sent to device memory. 只用L2会被write使用，L1不被write使用。

Stores are performed at a 32-byte segment granularity. Memory transactions can be one, two, or four segments at a time.

If a non-atomic instruction executed by a warp writes to the same location in global memory for more than one of the threads of the warp, only one thread performs a write and which thread does it is undefined. 如果多个thread non-atomic写入同一个global memory address，只有一个thread写入会被进行（不会replay），但是具体是哪个thread是不确定的



* efficency 

memory store efficency 与 memory load efficency的定义相似

nvprof 里面 gst_efficency metrics 就衡量了和这个



* transaction & segment 

If two addresses fall within the same 128-byte region but not within an aligned 64-byte region, one four-segment transaction will be issued (that is, issuing a single four-segment transaction performs better than issuing two one-segment transactions). 当传送4 segment的时候，依旧是one memory transaction。1 four segment memory transaction的速度是大于 2 two segment memory transaction的速度的.

when a 128-byte write request is issued from a warp, the request will be serviced by one four-segment transaction and one one-segment transaction. Therefore, 128 bytes were requested and 160 bytes were loaded, resulting in 80 percent efficiency. 在write的时候如果128 bytes misaligned，则会产生1 four segment transaction和1 one segment transaction.



* Example

memory access is aligned and all threads in a warp access a consecutive 128-byte range. store request is serviced by one four-segment transaction.

<img src="Note.assets/Screen Shot 2022-08-01 at 2.45.31 PM.png" alt="Screen Shot 2022-08-01 at 2.45.31 PM" style="zoom:50%;" />



Memory access is aligned, but the addresses are scat- tered along a 192-byte range. This store request is serviced by three one-segment transactions.

<img src="Note.assets/Screen Shot 2022-08-01 at 2.46.31 PM.png" alt="Screen Shot 2022-08-01 at 2.46.31 PM" style="zoom:50%;" />



memory access is aligned and the addresses accessed are in a consecutive 64-byte range. This store request is serviced with one two-segment transaction.

<img src="Note.assets/Screen Shot 2022-08-01 at 2.46.47 PM.png" alt="Screen Shot 2022-08-01 at 2.46.47 PM" style="zoom:50%;" />



#### From hardware

第一次访问，全部4个数据都放到buffer里

第一次使用前2个数据

<img src="Note.assets/Screen Shot 2022-05-31 at 12.05.15 AM.png" alt="Screen Shot 2022-05-31 at 12.05.15 AM" style="zoom:50%;" />

第二次访问使用后面两个数据（连续内存访问），直接从buffer里读取数据，不用再去core array

<img src="Note.assets/Screen Shot 2022-05-31 at 12.05.50 AM.png" alt="Screen Shot 2022-05-31 at 12.05.50 AM" style="zoom:50%;" />



**bursting** 每一次读取burst of data，读取的数据应该被充分使用，因为读取burst里面的两个数据的时间远远小于读取两个random address/两个burst。

蓝色的部分是core array到buffer的时间。红色的部分是buffer到pin的时间

<img src="Note.assets/Screen Shot 2022-05-31 at 12.06.52 AM.png" alt="Screen Shot 2022-05-31 at 12.06.52 AM" style="zoom:50%;" />



### Minimize host to device transfer

> Reference
>
> 1. CUDA C++ Best Practices Guide chapter 9.1



#### Batch Small Transfer

host memory -> device global memory 的拷贝是有overhead的。

希望避免多个small memory copy, 希望是one large memory copy

所以要pack多个small memory copy to large memory copy



#### Fully utilize transfer

一旦数据从host传送到device上，就尽量多的使用，不再传送数据

有些时候就算计算放在gpu上计算不划算，但是考虑到减少传送一次数据，使用gpu计算




## Shared memory

### Basic

* 是什么

1. on chip (For volta use same physical resources SRAM)
2. SRAM support random access, don't have constrain of burst like DRAM(global memory)
3. shared memory latency is roughly 20 to 30 times lower than global memory, and bandwidth is nearly 10 times higher.



* 什么时候使用

1. 数据有复用的话，考虑使用shared memory
1. share memory只存在memory bank conflict的问题，没有non-sequential / unaligned access by warp 的问题



* 使用时需要注意

1. 使用shared memory一定要注意不要忘记synchronize的使用
2. 从Volta开始，warp内部不是lock step的，所以warp内部使用shared memory有时候也需要memory fence
3. shared memory时有限的resource，需要考虑使用shared memory以后一个sm能有多少个thread和block



* load from global memory to shared memory 过程

内存拷贝与CPU相似，需要经过register (CUDA11有async，不经过register的方法，see below section)

global memory -> cache (optional L1)L2 -> per thread register -> shared memory

不存在直接从global memory到shared memory的硬件



* 测试shared memory对occupancy的影响

By simply increasing this parameter (without modifying the kernel), it is possible to effectively reduce the occupancy of the kernel and measure its effect on performance.



#### API

* dynamic use

只可以分配为1D

```cpp
extern __shared__ int tile[];

MyKernel<<<blocksPerGrid, threadsPerBlock, isize*sizeof(int)>>>(...);
```



* static use

可以分配为1/2/3D

```cpp
__shared__ float a[size_x][size_y];
```



* config L1 cache & shared memory

> Reference
>
> 1. Professional CUDA C Programming Guide chapter 5

L1 + Shared 一共有64 kb memory

shared memory使用32 bank访问。L1 cache使用cache line来访问。

如果kernel使用很多shared memory，prefer larger shared memory

如果kernel使用很多register，prefer larger L1 cache。因为register会spilling to L1 cache



config for whole device

```cpp
cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig);
```

<img src="Note.assets/Screen Shot 2022-07-30 at 11.28.41 AM.png" alt="Screen Shot 2022-07-30 at 11.28.41 AM" style="zoom:50%;" />



config for kernel

Launching a kernel with a different preference than the most recent preference setting might result in implicit device synchronization. 如果当前kernel的setting与前一个kernel的不一样，可能会导致implicit sync with device

```cpp
cudaError_t cudaFuncSetCacheConfig(const void* func, enum cudaFuncCacheca cheConfig);
```





### Memory Bank

> Reference
>
> 1. CUDA C++ Best Practice Guide chapter 9.2.3
> 2. CUDA C++ Programming Guide chapter K.3
> 3. Caltech CS179 Lecture 5
> 4. Professional CUDA C Programming chapter 5



#### Shared memory warp level transaction

Shared memory accesses are issued per warp. 对于shared memory的访问是以warp为单位进行访问的，而不是warp内每个thread分别访问shared memory。warp内的多个therad首先会合并threads之间的访问为一个或多个transaction，然后去访问shared memory。这样做的好处是（比起每个thread自己访问shared memory）增加数据的利用率，减少对shared memory的总access次数。

Ideally, each request to access shared memory by a warp is serviced in one transaction. In the worst case, each request to shared memory is executed sequentially in 32 unique transactions. 最好的情况一次warp对shared memory的访问只会引发一次transaction。最坏的情况一次warp对shared memory的访问会引发32次transaction。

If multiple threads access the same word in shared memory, one thread fetches the word, and sends it to the other threads via multicast/broadcast. 如果一个warp内的threads都concurrently访问同一个shared memory location，则只有一个thread访问这个location/只产生一次memory transaction，然后这个memory被broadcast到每一个threads。同样的broadcast在使用constant cache，warp level shuffle的时候都有。

If a shared memory load or store operation issued by a warp does not access more than one memory location per bank, the operation can be serviced by one memory transaction. 一个warp对shared memory的load/store在每个bank上只有一个location（并且在每个bank上请求的数据是小于bank bandwidth的），则只产生一个合并的memory transaction。最好的使用shared memory的办法就是确保请求的数据分布在每个bank中，每个bank充分的利用bank自己的bandwidth



#### Memory bank & bank conflict

* 特点

1. shared memory is divided into equally sized memory modules (banks) that can be accessed simultaneously. Therefore, any memory load or store of n addresses that spans n distinct memory banks can be serviced simultaneously, yielding an effective bandwidth that is n times as high as the bandwidth of a single bank。shared memory底层被切分为多个memory bank来使用。同时访问多个bank可以被同时serve，具有一个bank的n倍的bandwidth。
2. When multiple addresses in a shared memory request fall into the same memory bank, a bank conflict occurs, causing the request to be replayed. The hardware splits a request with a bank conflict into as many separate conflict-free transactions as necessary, decreasing the effective bandwidth by a factor equal to the number of separate memory transactions required. 对于shared memory的多个访问如果都落到一个bank中（并且不是same word，无法触发broadcast），则这些request会被replay。Hardware会split conflict request into serialized conflict free request。 (Figure 21 middle image)。花费的时间会是num replay * one bank free time
3. all threads in a warp read the same address within a single bank. One memory transaction is executed, and the accessed word is broadcast to all requesting threads. 如果多个memory request对应到any sub-word of one aligned 32 bit word (尽管在一个bank内)，访问不会serialize。如果是read则会broadcast，如果是write则只要一个thread写，但是哪个thread写是undefined的 (Figure 22 right two image). 这种情况下，bandwidth的使用依旧很低，因为num banks * bank bandwidth这么多的数据只用于传送一个word的数据。



Three typical situations occur when a request to shared memory is issued by a warp: 三种warp访问shared memory的类型

1. parallle access：no bank conflict
2. serial access : bank conflict cause serialized access to shared memory bank
3. broadcast access: single address read in single bank and broadcast to all threads in warp



<img src="Note.assets/Screen Shot 2022-06-27 at 10.42.50 PM.png" alt="Screen Shot 2022-06-27 at 10.42.50 PM" style="zoom:50%;" />

<img src="Note.assets/Screen Shot 2022-06-27 at 10.43.04 PM.png" alt="Screen Shot 2022-06-27 at 10.43.04 PM" style="zoom:50%;" />





#### Access Mode 32/64-bit

shared memory bank width: defines which shared memory addresses are in which shared memory banks. 

4 bytes (32-bits) for devices of compute capability except 3.x

8 bytes (64-bits) for devices of compute capability 3.x



* Fermi 2.x (and all cc except 3,x)

For a Fermi (2.x) device, the bank width is 32-bits and there are 32 banks. Each bank has a bandwidth of 32 bits per two clock cycles. Successive 32-bit words map to successive banks. (also apply for none 3.x device, only differ in number of clock cycle per transaction)

<img src="Note.assets/Screen Shot 2022-07-30 at 10.39.06 AM.png" alt="Screen Shot 2022-07-30 at 10.39.06 AM" style="zoom:50%;" />

A bank conflict does not occur when two threads from the same warp access the same address. In that case, for read accesses, the word is broadcast to the requesting threads, and for write accesses, the word is written by only one of the threads — which thread performs the write is undefined. 当访问同一个bank内的32-bit word (4 bytes)的时候没有bank conflict。如果是read则broadcast。如果是write则有一个thread成功，具体是哪个是undefined的。

Figure 5-5 上面是bytes address对应word indx。下面是word index对应bank index

<img src="Note.assets/Screen Shot 2022-07-30 at 10.39.21 AM.png" alt="Screen Shot 2022-07-30 at 10.39.21 AM" style="zoom:50%;" />



* Kepler 3.x

For Kepler devices, shared memory has 32 banks with 64-bit mode and 32-bit mode.

In 64-bit mode, successive 64-bit words map to successive banks. Each bank has a bandwidth of 64 bits per clock cycle.

<img src="Note.assets/Screen Shot 2022-07-30 at 10.42.38 AM.png" alt="Screen Shot 2022-07-30 at 10.42.38 AM" style="zoom:50%;" />

A shared memory request from a warp does not generate a bank conflict if two threads access any sub-word within the same 64-bit word because only a single 64-bit read is necessary to satisfy both requests. As a result, 64-bit mode always causes the same or fewer bank conflicts for the same access pattern on Kepler devices relative to Fermi. 当访问同一个bank内的64-bit word(8 bytes)的时候没有bank conflict。

1. read access：64 bits word会被broadcast到全部的threads
2. write access：warp内只有一个thread会发生write，具体是哪个thread是undefined的

In 32-bit mode, successive 32-bit words map to successive banks. However, because Kepler has a band- width of 64 bits per clock cycle, accessing two 32-bit words in the same bank does not always imply
a retry. It may be possible to read 64-bits in a single clock cycle and pass only the 32 bits requested to each thread. 数据到bank的映射是按照32-bit为单位的。但是bank width依旧是64 bit的，也就意味着一个clock cycle可以传送64 bit的数据，也就意味着在32-bit mode下，访问同一个bank的2个32-bit word不一定产生bank conflict，因为bank width是64，可以把2个word都传送出去。bank conflict的本质是bank width小，所以无法传送过多的数据

1. read access：32 bit word会被broadcast
2. write access：warp内只有一个thread会发生write，具体是哪个thread是undefined的

Figure 5-6是32 bit mode情况下的数据分布

<img src="Note.assets/Screen Shot 2022-07-30 at 10.47.06 AM.png" alt="Screen Shot 2022-07-30 at 10.47.06 AM" style="zoom:50%;" />

A large bank size may yield higher bandwidth for shared memory access, but may result in more bank conflicts depending on the application’s shared memory access patterns. 大的bank会带来更大的bandwidth，但是会有更多的conflict



* Configure Kepler

Changing the shared memory configuration between kernel launches might require an implicit device synchronization point. 改变Kepler下shared memory bank可能会导致implicit sync with device

```cpp
// query access mode 
cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig *pConfig);

cudaSharedMemBankSizeFourByte
cudaSharedMemBankSizeEightByte

// setting access mode
cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config);
cudaSharedMemBankSizeDefault 
cudaSharedMemBankSizeFourByte 
cudaSharedMemBankSizeEightByte
```



#### Stride access

stride access对于global memory是waste bandwith/not coarlesed access, 对于shared memory是bank conflict.

stride 指的是warp内的连续的thread，访问内存的间隔。如果t0访问float array idx0，t1访问float array idx1，则stride of one 32 bit word



* 不同的stride，对应的conflict类型 (假设2.x 32-bit mode)

stride of one 32 bit word : conflict free (见上Fig 21 above left)

stride of two 32 bit word : 16 x 2-way (2-way表示会被serialize为两个access) bank conflict (见上Fig 21 middle)

stride of three 32 bit word : conflict free (见上Fig 21 right)

...

stride of 32 32 bit word : 1 x 32-way (32-way表示会被serialize为32个access) bank conflict

也就是奇数的stride是conflict free的，偶数的stride是有conflict的



#### Avoid bank conflict

* stride of 32 32 bits word (32-bit mode)

stride of 32 32 bits word 产生 1 x 32-way bank conflict 经常发生在使用shared memory处理2D array of 32 x 32，每个thread负责一个row。这样每个thread对应的row开始都会是在同一个bank中。

解决方法是pad 2d array to sizd 32 x 33, 这样每个thread负责的一个row的开始都不是在一个bank中 (stride of 33 33 bit word是conflict free的)

对于padding 64-bit与32-bit mode的方法是不一样的。有些在32-bit上是conflict free的，在64-bit上就有conflict了



* stride of 1 32 bits words

are both coarlesed global memory access & shared memory conflict free



* 常见pattern

In the “load from global, store into shared, do quadratic computation on shared data” pattern, you sometimes have to choose between noncoalesced loads or bank conflicts on stores. Generally bank conflicts on stores will be faster, but it’ s worth benchmarking. The important thing is that the shared memory loads in the “quadratic computation” part of the code are conflict-free (because there are more of these loads than either other operation).



* performence compared with global memory

> Reference
>
> 1. CUDA Developer Form About the different memories [link](https://forums.developer.nvidia.com/t/about-the-different-memories/1861/6)



shared memory is fast even if there are bank conflicts. Even with 16-way bank conflicts, shared memory is dozens of times faster than gobal memory. shared memory就算是有bank conflict也比global memory要快很多

many people get too worried about bank conflicts. Optimize for bank conflicts last, especially if they are only 2- or 4-way conflicts, which may take more instructions to optimize away than they cost anyway.  有些时候为了避免shared memory bank conflict从而做了很多优化，但是由于额外的增加intrinsic，导致perf反而变差



### Corner-Turning

> Reference
>
> 1. UIUC ECE Lecture 4



#### 是什么

Use of a transposed thread order to allow memory loads to coalesce when loading global to shared.  如果直接访问global memory是不coarlesed的，通过使用transpose thread的方法，coarlesed load global memory到shared memory上。在使用shared memory的时候因为是SRAM所以不存在coarlesed的问题



#### Example GEMM data access

当使用tilnig+每个thread读取一个M N到shared memory的时候，读取M也是burst的。这是因为比起上面的simple code使用iteration读取，这里使用多个thread读取，一次burst的数据会被临近的thread使用(M00 M01分别被2个thread读取，每个thread只读取一个M elem)，而不是下一个iteration被清空。

这里对于M没有使用显性的memory transpose，但是因为使用多个thread读取数据，依旧保证了burst，这与CPU代码需要使用transpose是不一样的。

同时shared memory使用的是SRAM，不像DRAM有burst的问题，所以读取M的shared memory的时候尽管不是连续读取也没有问题。shared memories are implemented as intrinsically high-speed on-chip memory that does not require coalescing to achieve high data access rate.

<img src="Note.assets/Screen Shot 2022-05-31 at 12.10.56 AM.png" alt="Screen Shot 2022-05-31 at 12.10.56 AM" style="zoom:70%;" />





### Async global memory to shared memory

> Reference
>
> 1. CUDA C++ best practice guide chapter 9.2.3.4

CUDA 11.0 允许async copy from global memory to shared memory



* 优点

1. Overlap copying data with computation
2. avoid using intermediary register file, reduce register pressue & reduce instruction pipeline pressure.
   1. thus further increase kernel occupancy
3. hardware accelerated on A100 (higher bandwidth, lower latency)
   1. 相比起sync的拷贝来说，async的latency (avg clock cycle)是更小的

<img src="Note.assets/Screen Shot 2022-06-28 at 11.19.35 PM.png" alt="Screen Shot 2022-06-28 at 11.19.35 PM" style="zoom:50%;" />




* 与Cache关系

可以optionally 使用L1 cache. 

如果每个thread拷贝16 bytes数据，L1 cache会被bypassed (enabled)

因为shared memory是per SM的，所以不涉及使用L2 cache(across SM)



* optimization

对于sync拷贝，num data是multiply of 4是最快的。估计compiler是使用了内部的vector

对于async拷贝，data size是8/16是最快的。

<img src="Note.assets/Screen Shot 2022-06-28 at 11.53.30 AM.png" alt="Screen Shot 2022-06-28 at 11.53.30 AM" style="zoom:50%;" />



#### Example

```cpp
template <typename T>
__global__ void pipeline_kernel_sync(T *global, uint64_t *clock, size_t copy_count) {
  extern __shared__ char s[];
  T *shared = reinterpret_cast<T *>(s);

  uint64_t clock_start = clock64();

  for (size_t i = 0; i < copy_count; ++i) {
    shared[blockDim.x * i + threadIdx.x] = global[blockDim.x * i + threadIdx.x];
  }

  uint64_t clock_end = clock64();

  atomicAdd(reinterpret_cast<unsigned long long *>(clock),
            clock_end - clock_start);
}

template <typename T>
__global__ void pipeline_kernel_async(T *global, uint64_t *clock, size_t copy_count) {
  extern __shared__ char s[];
  T *shared = reinterpret_cast<T *>(s);

  uint64_t clock_start = clock64();

  //pipeline pipe;
  for (size_t i = 0; i < copy_count; ++i) {
    __pipeline_memcpy_async(&shared[blockDim.x * i + threadIdx.x],
                            &global[blockDim.x * i + threadIdx.x], sizeof(T));
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);
  
  uint64_t clock_end = clock64();

  atomicAdd(reinterpret_cast<unsigned long long *>(clock),
            clock_end - clock_start);
}

```



#### API

* __pipeline_memcpy_async()

instructions to load from global memory and store directly into shared memory are issued as soon as this function is called



* __pipeline_wait_prior(0)

wait until all instruction in pipe object have been executed



## Constant cache & Read-Only Cache

### Difference

> Reference
>
> 1. Professional CUDA C Programming chapter 5
> 1. CUDA Developer Form Do 7.x device have readonly constant cache [link](https://forums.developer.nvidia.com/t/do-7-x-devices-have-a-readonly-constant-cache/220844)
> 1. CUDA Developer Form const __restrict__ read faster than __constant__ ? [link](https://forums.developer.nvidia.com/t/const-restrict-read-faster-than-constant/31982)



对于不同compute capacity的硬件，constant cache，read-only texture cache, L1 cache的关系是不太一样的。



GPU 一共有4中类型的cache

1. L1 Cache
2. L2 Cache
3. read-only constant cache (through constant memory)
4. read-only texture cache (thorugh texture memory / ldg load global memory)



(For Kepler) The read-only cache is separate and distinct from the constant cache. Data loaded through the constant cache must be relatively small and must be accessed uniformly for good performance (all threads of a warp should access the same location at any given time), whereas data loaded through the read-only cache can be much larger and can be accessed in a non-uniform pattern. 

Read-only cache 与 constant cache 是两个东西

constant cache适用于small + all warp threads read same location (access uniform)

read-only cache适用于much larger + non-uniform pattern / stream through array

两种cache分别有自己的使用方法。



使用constant cache对于uniform access(all warp threads read same location) 的performance更好，是因为constant memory对于broadcast access pattern的优化比起read-only cache更好。(不确定对于最新的GPU arch是否还有在perf上的区别)





### Constant Memory & Constant Cache

> Reference
>
> 1. Professional CUDA C Programming chapter 5
> 1. CUDA C++ Best Practice Guide 9.2.6
> 1. Caltech CS179 lecture 5
> 1. UIUC ECE 408 Lecture 7



Constant memory is a special-purpose memory used for data that is read-only and accessed uniformly by threads in a warp. Constant Memory用于在device上的uniform read. 物理上与global memory都在off chip device memory上

64 kb constant memory for user, 64 kb for compiler. kernel arguments are passed through constnat memory 

Constant memory is as fast as register

higher throughput than L1 cache. Same 5 cycle latency as L1 cache.

Constant memory variables can be visible across multiple source files when using the CUDA separate compilation capability. constant memory不仅仅可以被相同file的全部grid可见，还是visibale across soruce file的

常用于储存formula的coefficent。warp threads会一起访问某一个coefficent，这样是最适合constant memory的。之所以不用register储存coefficent是因为有太大的register pressure，导致num block/SM下降



#### Broadcast

The constant cache has a single port that broadcasts data to each thread in a warp. 

each time when a constant is access from cache, it can be broadcast to all threads in a warp. this makes constant memory almost as efficent as registers. 当warp thread访问相同的constant memory location的时候，会进行broadcast



#### Serialization

Accesses to different addresses by threads within a warp are serialized (by split one large request into seprate request). Thus, the cost of a constant memory read scales linearly with the number of unique addresses read by threads within a warp. warp threads访问不同的constant memory location，会导致访问被serialize

warp内对于constant cache不同地址的访问是serialized的。Accesses to different addresses by threads within a warp are serialized, thus the cost scales linearly with the number of unique addresses read by all threads within a warp.

如果t0访问constant cache addr 0， t1访问constant cache addr 1，这两个对constant cache的访问会serialized。

对于使用constant cache，最好的访问方法是all threads within warp only access a few (serialization not too much)  / same memory address (use broadcast) of constant cache. 



#### API

```cpp
// copy host to constant memory on host
cudaError_t cudaMemcpyToSymbol(const void *symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind)
```



API with example

```cpp
__constant__ float coef[RADIUS + 1];


__global__ void stencil_1d(float *in, float *out) { 
  // shared memory
  __shared__ float smem[BDIM + 2*RADIUS];
  // index to global memory
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // index to shared memory for stencil calculatioin 
  int sidx = threadIdx.x + RADIUS;
  // Read data from global memory into shared memory 
  smem[sidx] = in[idx];
  // read halo part to shared memory 
  if (threadIdx.x < RADIUS) {
    smem[sidx - RADIUS] = in[idx - RADIUS];
    smem[sidx + BDIM] = in[idx + BDIM]; 
  }
  // Synchronize (ensure all the data is available) 
  __syncthreads();
  // Apply the stencil
  float tmp = 0.0f;
  
  #pragma unroll
  for (int i = 1; i <= RADIUS; i++) {
  	tmp += coef[i] * (smem[sidx+i] - smem[sidx-i]); 
  }
  // Store the result
  out[idx] = tmp; 
}
```



### Read-Only Texture Cache

> Reference
>
> 1. Professional CUDA C Programming chapter 5
> 2. CUDA C++ Programming Guide chapter B.10
> 3. Memory Statistics - Caches [link](https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/memorystatisticscaches.htm)
> 4. Memory Statistics - Global [link](https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/memorystatisticsglobal.htm)
> 5. Blog CUDA-F-4-3-内存访问模式 [link](https://face2ai.com/CUDA-F-4-3-内存访问模式/)
> 6. Stackoverflow What is the difference between __ldg() intrinsic and a normal execution? [link](https://stackoverflow.com/questions/26603188/what-is-the-difference-between-ldg-intrinsic-and-a-normal-execution)



GK110 adds the ability for read-only data in global memory to be loaded through the same cache used by the texture pipeline via a standard pointer without the need to bind a texture beforehand and without the sizing limitations of standard textures. Since this is a separate cache with a separate memory pipe and with relaxed memory coalescing rules, use of this feature can benefit the performance of bandwidth-limited kernels. Kepler开始GPU支持对global memory使用per SM read-only cache。底层使用GPU texture pipeline as read-only cache for data stored in global memory

Global memory accesses are routed either through L1 and L2, or only L2, depending on the architecture and the type of instructions used. Global read-only memory accesses are routed through the texture and L2 caches. Texture memory is read-only device memory, and is routed through the texture cache and the L2 cache.

通过read-only texture cache (也会通过L2 Cache) 读取global memory比起normal global memory read (会通过L1+L2 cache)有更大的bandwidth

The granularity of loads through the read-only cache is 32 bytes. read only cache是32 bytes granularity的

相比起L1，对于scatter read使用read-only cache更有效。



#### API

下面的两种使用方法都是indicate to the compiler that data is read-only for the duration of a kernel. 也就是代表一个部分的内存不会在一个kernel内一会是read-only，一会是write



* intrinsic

对于computation capacity > 3.5 的设备，可以使用intrinsic来强制得到对应data type T的数据。

```cpp
__global__ void kernel(float* output, float* input) 
{ 
  ...
	output[idx] += __ldg(&input[idx]);
	... 
}
```



* compiler hint

对于compiler的hint，让compiler生成read-only cache读取

对于复杂的kernel，有些时候compiler hint可能不管用，还是推荐ldg读取

```cpp
void kernel(float* output, const float* __restrict__ input) 
{ 
  ...
	output[idx] += input[idx]; 
}
```



## L1 & L2 Cache

Some part of cache related topic are included in global memory



#### Cache VS Shared Memory

* same

1. both on chip. For volta use same physical resources SRAM



* different

1. programmer control shared memory 
2. micro-arch determine content of cache



#### L2 cache persisting

> Reference
>
> 1. CUDA C++ Best Practices Guide chapter 9.2.2
> 2. CUDA C++ Programming Guide chapter 3.2.3



* 是什么

CUDA 11.0 + compute capacity 8.0 可以config persistence of data in L2 cache

可以设定L2中有多少数据是用于persistance的。



* 设定L2 persistance

Persisting accesses have prioritized use of this set-aside portion of L2 cache, whereas normal or streaming, accesses to global memory can only utilize this portion of L2 when it is unused by persisting accesses.

```cpp
cudaGetDeviceProperties(&prop, device_id);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize); 
/* Set aside max possible size of L2 cache for persisting accesses */
```



* 设定user data access policy window

可以通过L2 access policy window设定user data中有多少数据是persistance的，windows的大小

`hitRatio` 的意义是 if the hitRatio value is 0.6, 60% of the memory accesses in the global memory region [ptr..ptr+num_bytes) have the persisting property and 40% of the memory accesses have the streaming property

需要确保num_bytes * hitRatio部分的数据小于使用L2 persistance的大小（剩余部分的num_bytes将会使用L2 streaming来访问，帮助减少cache thrashing）。如果超过了L2 persistance的大小，CUDA runtime依旧会尝试把数据放到L2 persistance的部分，导致thrashing of L2 cache line. 



使用stream （也可以使用graphkernelnode，见C++ programming guide）

```cpp
cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persisting accesses.
                                                                              // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
stream_attribute.accessPolicyWindow.hitRatio  = 1.0;                          // Hint for L2 cache hit ratio for persisting accesses in the num_bytes region
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

//Set the attributes to a CUDA stream of type cudaStream_t
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   
```



* 例子

如果L2 set-aside cache是16 kb，num_bytes in accessPolicyWindow 是32kb

With a hitRatio of 0.5, the hardware will select, at random, 16KB of the 32KB window to be designated as persisting and cached in the set-aside L2 cache area.

With a hitRatio of 1.0, the hardware will attempt to cache the whole 32KB window in the set-aside L2 cache area. Since the set-aside area is smaller than the window, cache lines will be evicted to keep the most recently used 16KB of the 32KB data in the set-aside portion of the L2 cache. 产生cache liner eviction，反而导致perf降低



* hitProp

1. cudaAccessPropertyStreaming ：  less likely to persist in the L2 cache because these accesses are preferentially evicted.
2. cudaAccessPropertyPersisting： more likely to persist in the L2 cache because these accesses are preferentially retained in the set-aside portion of L2 cache.
3. cudaAccessPropertyNormal： resets previously applied persisting access property to a normal status. 
   1. Memory accesses with the persisting property from previous CUDA kernels may be retained in L2 cache long after their intended use. This persistence-after-use reduces the amount of L2 cache available to subsequent kernels that do not use the persisting property.  前一个kernel的L2 cache可能会在前一个kernel运行结束后依旧在L2中，使用这个property来清空前一个kernel所使用的L2 cache



* reset all presisting l2 cache

hitProp=cudaAccessPropertyNormal会reset a previous persisting memory region 

cudaCtxResetPersistingL2Cache会reset all persisting L2 cache line

L2 cache也会automatic reset在一定时间后，但是不推荐依赖于automiatic reset，因为重甲的时间会很大



* concurrent kernel utilization L2 set-aside cache

L2 set-aside cache portion is shared among all these concurrent CUDA kernels. As a result, the net utilization of this set- aside cache portion is the sum of all the concurrent kernels' individual use. 如果有多个kernel concurrent的话，则使用的总persistance=每一个kernel的sum，可能会存在oversubscrabe resource的情况导致cache thrashing， 反而导致perf不好。

如果多个concurrent kernel同时运行的话，需要让sum of persistance使用< L2 set aside



## Local memory

#### Basic

* local memory (software concept) 对应的hardware

local memory与global memory都是放在off-chip device memory上



* 什么样的automatic variable会放在local memory上

> Reference
>
> 1. CUDA Developer Form const restrict read faster than constant [link](https://forums.developer.nvidia.com/t/const-restrict-read-faster-than-constant/31982/9)



I would claim perspective is important: The default storage for a local array is local memory, where “local” means “thread-local”.  array默认都是放在thread private的local memory上

The compiler may, as an optimization, promote the local array to register storage. array如果fix size + small in size有可能会被compiler放在register上。本质上是被compiler optimize从local memory放到了register上

否则会被放在local memory上，因为compiler不知道这个array会有多长，无法把array拆分后放到regsiter中。

struct如果占用空间很大的话，也有可能被放在local memory上



* 如何确定var是在local memory上

通过PTX只可以确定第一轮编译以后是否在local memory上。

但是第一轮不再local memory上不代表后面不会放到local memory上

可以通过 `--ptxas-options=-v` 查看总local memory使用用量。



#### Coarlesed

> Reference
>
> 1. CUDA C++ Programming guide 5.2.3



##### local memory access 特点

因为和global memory一样都在device memory的硬件上。所以与global memory一样有high latency & low bandwidth

对于local memory的访问也需要和global memory一样需要coarlesed



##### automatic coarlesed layout

consequtive 32 bits / 4 bytes words are accessed by consequtive thread. 

也就是如果local memory array是4 bytes的话，只要每个thread同时访问的idx是一样的，对device memory的访问就是coarlesed。



local memory在device上的layout：t0 idx0, t1 idx0, t2 idx0, ... t31 idx0, t0 idx1, t1 idx 1



##### cache behaviour 

compute capability 3.x local memory accesses are always cached in L1 and L2 in the same way as global memory accesses (see Compute Capability 3.x).

compute capability 5.x and 6.x, local memory accesses are always cached in L2 in the same way as global memory accesses (see Compute Capability 5.x and Compute Capability 6.x).



## Register

> Reference
>
> 1. CUDA Form Saving registers with smaller data types? [link](https://forums.developer.nvidia.com/t/saving-registers-with-smaller-data-types/7376)



Registers 是 32 bit / 4 bytes 大小的 (same size as int / single precision float)。如果数据类型是double的话，则使用2个register。

可以通过pack small data into a register (e.g. 2 short) and use bitmask + shift 来读取。从而减少register usage per thread



* Bank conflict 

> Reference
>
> 1. CUDA C++ Best practice 9.2.7

Register 也会有bank conflict，只不过这是完全由compiler处理的，programmer对于解决register bank conflict没有任何控制。

并不需要特意把数据pack成vector type从而来避免bank conflict



* 控制per thread max register

可以通过compiler option来控制max register pre thread

```shell
-maxrregcount=N
```



## Atomic

CUDA provides atomic functions that perform read-modify-write atomic operations on 32-bits or 64-bits of global memory or shared memory. 用于32 bit和64bit的atomic操作

三种主要atomic类型：arithmetic functions, bitwise functions, and swap functions. 



<img src="Note.assets/Screen Shot 2022-07-28 at 10.25.43 PM.png" alt="Screen Shot 2022-07-28 at 10.25.43 PM" style="zoom:50%;" />



#### CAS

> Referece
>
> 1. Professional CUDA C Programming Guide chapter 7



CAS compare and swap 是一切atomic operation的基础，全部的atomic 操作都可以使用CAS实现。虽然实际上CUDA atomic都是naively supported的

CAS takes as input three items: A memory location, an expected value at that memory location, and the value you would like to store at that memory location.

CAS进行下面的三个步骤

<img src="Note.assets/Screen Shot 2022-07-28 at 10.20.24 PM.png" alt="Screen Shot 2022-07-28 at 10.20.24 PM" style="zoom:50%;" />



#### Implement your own atomic

* build float atomic add with float CAS

之所以使用while loop，是因为在add value to dst的时候，可能存在数据被改变，所以使用while loop来确定expected确实等于我设定的值

这里的while loop和thread写入产生conflict的replay很像（see below)

这里有个非常重要的店：it's safe to read memory location that's being atomically modify by other thread

```cpp
__device__ int myAtomicAdd(int *address, int incr) 
{
	// Create an initial guess for the value stored at *address. 
  int expected = *address;
  int oldValue = atomicCAS(address, expected, expected + incr);
	// Loop while expected is incorrect. 
  while (oldValue != expected) {
		expected = oldValue;
		oldValue = atomicCAS(address, expected, expected + incr); 
  }
	return oldValue; 
}
```



* build non-existing atomic with existing atomic

通过使用type conversion intrinsic，从而使用已经存在的atomic function实现不存在的data type 的atomic function

```cpp
__device__ float myAtomicAdd(float *address, float incr) 
{
	// Convert address to point to a supported type of the same size 
  unsigned int *typedAddress = (unsigned int *)address;
	// Stored the expected and desired float values as an unsigned int 
  float currentVal = *address;
	unsigned int expected = __float2uint_rn(currentVal);
	unsigned int desired = __float2uint_rn(currentVale + incr);
	int oldIntValue = atomicCAS(typedAddress, expected, desired); 
  
  while (oldIntValue != expected) 
  {
		expected = oldIntValue;
    /*
    * Convert the value read from typedAddress to a float, increment, 
    * and then convert back 	to an unsigned int
    */
    desired = __float2uint_rn(__uint2float_rn(oldIntValue) + incr);
    oldIntValue = atomicCAS(typedAddress, expected, desired); 
	}
	
  return __uint2float_rn(oldIntValue); 
}
```




#### Non-atomic write behavior

> Reference
>
> 1. CUDA C++ Programming guide chapter k.3



如果warp内的多个thread non-atomic write to same memory location, 则只要一个thread会进行write，但是是哪个thread是undefined的



#### Latency & Throughput 

> Reference
>
> 1. UIUC 508 Lecture 2
> 2. Programming Massively Parallel Processors 3rd edition Chapter 9



* latency

atomic 操作的latency = dram load latency + internal routing + dram store latency

因为需要先读取gloabl memory，把数据传送给SM（这个时候其余的thread/SM不能r/w这块内存)，再把数据传送给global memory

对于global memory，latency是few hunderdes cycle

对于last level cache, latency是few tens cycle

对于shared memocy, latency是few cycle

<img src="Note.assets/Screen Shot 2022-06-18 at 4.49.28 PM.png" alt="Screen Shot 2022-06-18 at 4.49.28 PM" style="zoom:50%;" />



modern GPU支持在last level cache上进行atomic操作(应该是只有shared memory atomic)，也就把atomic的latency从few hunderdes cycle变为了few tens cycle. 这个优化不需要任何programmer的更改，是通过使用更先进的hardware来实现的。




* Throughput

GPU通过很多thread来hide latency。也就要求many DRAM access happen simutaniously来充分利用hardware bandwidth.

使用atomic操作以后，对于某一个memory location的访问是serialize的，导致实际bandwidth降低。

atomic throughput与latency是成反比的。



* 例子

假设64 bit double data rate DRAM with 8 channel 1GHz clock frequency. DRAM latency is 200 cycles

peak throughput是 8 (64 bit per transfer) * 2 (two transfer per clock pe channel) * 8 (channels) * 1G (clock per second) = 128GB/second



atomic的latency如果是400 clock cycle

throughput就是 1/400 atomic / clock * 1GHz clock/second = 2.5 M atomic / second



如果uniform的atomic 26 alphabet，则throughput是 2.5M * 26 / second



#### Evolving
> Reference
>
> 1. UIUC 408 Lecture 18



GPU atomic随着GPU Arch也在改进

atomic on shared memory (latency & bandwidth) >> atomic on global memory 



* GT200

atomic is on global memory, no L2 cache 



* Fermi to Kelpler

both atomic on L2 cache

Improve atomic by add more l2 cache buffer 



* kepler to maxwell

improve shared memory atomic through using hardware. 

Kepler use software for shared memory atomic



* after maxwell

atomic is rouphly the same

the flexibility of atomic is changed. now have atomic within warp / block.



* computation capacity 1.1

32-bit atomic in global memory



* computation capacity 1.2

32-bit atomic in shared memory

64 bit atomic in global memory 



* computation capacity 2.0

64 bit atomic in shared memory 



#### Replay

> Reference
>
> 1. Professional CUDA C Programming Guide chapter 7

Conflicting atomic accesses to a shared location might require one or more retries by conflicting threads, analogous to running more than one iteration of myAtomicAdd’s loop. 如果多个thread 对于同一个memory location进行atomic操作，在同一时间只会有一个thread成功，其余的thread会被replay

If multiple threads in a warp issue an atomic operation on the same location in memory, warp execution is serialized. Because only a single thread’s atomic operation can succeed, all others must retry. If a single atomic instruction requires n cycles, and t threads in the same warp execute that atomic instruction on the same memory location, then the elapsed time will be t×n, as only one thread is successful on each successive retry. rest of the threads in the warp are also stalled waiting for all atomic operations to complete. 在warp内如何threads atomic concurrent的写入同一个memory location，则会产生retry。当某一个thread retry的时候，其余的thread会像是branch divergence一样stall



#### Warp-aggregated

> Reference
>
> 1. CUDA Tech Blog CUDA Pro Tip: Optimized Filtering with Warp-Aggregated Atomics [link](https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/)



NVCC compiler (from CUDA 9) now performs warp aggregation for atomics automatically in many cases, so you can get higher performance with no extra effort. In fact, the code generated by the compiler is actually faster than the manually-written warp aggregation code. NVCC现在支持自动编译代码使用warp aggregation而且速度比手写的warp aggregation要更快，下面的例子只是用于展示warp aggregation这个方法。



是什么：In warp aggregation, the threads of a warp first compute a total increment among themselves, and then elect a single thread to atomically add the increment to a global counter. This aggregation reduces the number of atomics performed by up to the number of threads in a warp (up to 32x on current GPUs), and can dramatically improve performance. 以warp为单位进行atomic，首先在warp内部计算出来要atomic的值，然后选出一个thread执行atomic。这样减少了atomic操作的执行次数（atomic操作导致serial execution以及更低的bandwidth，更多的atomic操作带来更低的bandwidth）。

atoimc次数与bandwidth是log的反向相关。下图中的横轴可以理解为number of atomic operation.

<img src="Note.assets/image2.png" alt="Figure 1. Performance of filtering with global atomics on Kepler K80 GPU (CUDA 8.0.61)." style="zoom:60%;" />



## Zero-Copy Memory

> Reference
>
> 1. Professional CUDA C Programming chapter 4



> Note: 并不是很常用



GPU threads can directly access zero-copy memory (on host).

Zero-copy memory is pinned (non-pageable) memory that is mapped into the device address space. 本质上是pinned host memory映射到device address space

When using zero-copy memory to share data between the host and device, you must synchronize memory accesses across the host and device. Modifying data in zero-copy memory from both the host and device at the same time will result in undefined behavior. 如果在host/device修改了内存，需要synchronize来保证consistency。



优点

1. Leveraging host memory when there is insufficient device memory
2. Avoiding explicit data transfer between the host and device



缺点

1. For discrete systems with devices connected to the host via PCIe bus, zero-copy memory is advantageous only in special cases. 对于使用PCIe链接的GPU与CPU，zero-copy 速度比global memory/device memory要慢。就算是加上了memory transfer time也还是慢。
2. In integrated architectures, CPUs and GPUs are fused onto a single die and physi- cally share main memory. In this architecture, zero-copy memory is more likely to benefit both performance and programmability because no copies over the PCIe bus are necessary. 当GPU CPU使用同一个memory，使用zero-copy对于performence和programability都有帮助
3. 很多时候用zero copy memory只是因为
   1. device memory不够用
   2. 简化程序



#### API

```cpp
cudaError_t cudaHostAlloc(void **pHost, size_t count, unsigned int flags);
```

cudaHostAllocDefault makes the behavior of cudaHostAlloc identical to cudaMallocHost

cudaHostAllocPortable returns pinned memory that can be used by all CUDA contexts, not just the one that performed the allocation. 

cudaHostAllocWriteCombined returns write-combined memory, which can be transferred across the PCI Express bus more quickly on some system configurations but cannot be read efficiently by most hosts.

cudaHostAllocMapped, which returns host memory that is mapped into the device address space.



## Unified Virtual Address

> Reference
>
> 1. Professional CUDA C Programming Chapter 4



> Note: 并不是很常用



Devices with compute capability 2.0 and later support a special addressing mode called Unified Virtual Addressing (UVA). UVA, introduced in CUDA 4.0, is supported on 64-bit Linux systems. With UVA, host memory and device memory share a single virtual address space. UVa让CPU GPU的virtual memory是共享的

<img src="Note.assets/Screen Shot 2022-07-30 at 10.05.28 PM.png" alt="Screen Shot 2022-07-30 at 10.05.28 PM" style="zoom:50%;" />



With UVA, there is no need to acquire the device pointer or manage two pointers to what is physically the same data. 

```cpp
// allocate zero-copy memory at the host side 
cudaHostAlloc((void **)&h_A, nBytes, cudaHostAllocMapped); 
cudaHostAlloc((void **)&h_B, nBytes, cudaHostAllocMapped);
// initialize data at the host side 
initialData(h_A, nElem); 
initialData(h_B, nElem);
// invoke the kernel with zero-copy memory 
sumArraysZeroCopy<<<grid, block>>>(h_A, h_B, d_C, nElem);
```



## Unified Memory

> Reference
>
> 1. Professional CUDA C Programming Chapter 4



> Note: 并不是很常用



从CUDA6中引入

Unified Memory creates a pool of managed memory, where each allocation from this memory pool is accessible on both the CPU and GPU with the same memory address (that is, pointer). The underlying system automatically migrates data in the unified memory space between the host and device. 一个memory pool可以同时在device host上使用，unified memory底层对memory transaction进行维护。

Unified Memory depends on Unified Virtual Addressing (UVA) support, but they are entirely differ- ent technologies. UVA provides a single virtual memory address space for all processors in the system. However, UVA does not automatically migrate data from one physical location to another; that is a capability unique to Unified Memory. UVA是相同virtual memory space，但是依旧需要对memory进行host to device的拷贝。Unified Memory是对用户来说只有一个memory，由unified memory来负责底层数据的拷贝。



#### API

* static

```cpp
__device__ __managed__ int y;
```



* dynamic

```cpp
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags=0);
```



## Weakly-Ordered Memory Model

> Reference
>
> 1. CUDA Toolkits document [link](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)
> 2. stackoverflow [link](https://stackoverflow.com/questions/5232689/cuda-threadfence)
> 3. Professional CUDA C Programming guide chapter 5



#### 是什么

Modern memory architectures have a relaxed memory model. This means that the memory accesses are not necessarily executed in the order in which they appear in the program. CUDA adopts a weakly-ordered memory model to enable more aggressive compiler optimizations. The order in which a GPU thread writes data to different memories, such as shared memory, global memory, page-locked host memory, or the memory of a peer device, is not necessarily the same order of those accesses in the source code. The order in which a thread’s writes become visible to other threads may not match the actual order in which those writes were performed. CUDA使用weakly-ordered memory model。程序对memory的访问与实际上硬件对memory的访问是不一样的。一个thread写入shared memory, global memory, paged lock memory的顺序与另一个thread观察到的顺序是不一样的。如果两个thread一个read，一个write，没有sync的话，则行为是undefined的

可以通过使用memory fence或者barriers来保证不同thread读取到的数据是expected的



#### Explicit Barriers

__syncthreads ensures that all global and shared memory accesses made by these threads prior to the barrier point are visible to all threads in the same block. 应用范围是all caller thread within threads block的。all threads within threads block对内存的操作barrier以后对all threads within threads block可见

```cpp
void __syncthreads();
```



deadlock:

syncthread不能与branch一起使用。

all threads in threads block must call the same syncthreads function call. 可以理解为每个syncthreads function call有自己的unique id。当某个unique id syncthreads 运行以后，需要all threads in threads block都运行这个特定的synchthread方程才可以继续

```cpp
if (threadID % 2 == 0) {
  __syncthreads();
} else { 
  __syncthreads();
}
```



#### Memory Fence

Memory fence functions ensure that any memory write before the fence is visible to other threads after the fence （取决于不同的API，应用范文也是不一样的）. 通过使用memory fence，保证 (1) all write before fence对于程序(不同的scope)来说发生在all write after fence之前. (2) all read before fence对于程序(不同的scope)来说发生在all read after fence之前。



* `void __threadfence_block();`

ensures that all writes to shared memory and global memory made by a calling thread before the fence are visible to other threads in the same block after the fence. 应用范围是单一calling thread对global and shared memory的操作是visible to all other threads in same threads block。

这个API不用被all threads in threads block调用。用于只想visible某个threads对内存的操作。



使用within block fence 可能会存在的问题

<img src="Note.assets/Screen Shot 2022-07-30 at 11.54.08 AM.png" alt="Screen Shot 2022-07-30 at 11.54.08 AM" style="zoom:50%;" />



* `void __threadfence();`

stalls the calling thread until all of its writes to global memory are visible to all threads in the same grid. 应用范围是单一calling threads对于global memory的操作是visible to all other threads in grid。

之所以没有shared memory，是因为visibel to all threads in grid，是跨越block的



* 例子 1

下面这个例子中，不可能得到A=1,B=20。因为X=10一定发生在Y=20之前，如果observe了Y=20的话，则X=10一定运行完了

```cpp
__device__ int X = 1, Y = 2;

__device__ void writeXY()
{
    X = 10;
    __threadfence();
    Y = 20;
}

__device__ void readXY()
{
    int B = Y;
    __threadfence();
    int A = X;
}
```



* 例子 2

Imagine, that one block produces some data, and then uses atomic operation to mark a flag that the data is there. But it is possible that the other block, after seeing the flag, still reads incorrect or incomplete data.

一个block写入global memory数据以及用atomic写入flag，另一个block通过flag判断是否可以读取global memory的数据。

 If no fence is placed between storing the partial sum and incrementing the counter, the counter might increment before the partial sum is stored 

如果没有memory fence的话，可能flag会首先被atomic设置了，然后才设置global memory的数据。这样另一个block在读取到flag以后就开始读取global memmory的值可能就是不对的。

通过使用memory fence，确保在fence后面读取memory的数据确实是fence之前写入的数据



* `void __threadfence_system();`

stalls the calling thread to ensure all its writes to global memory, page- locked host memory, and the memory of other devices are visible to all threads in all devices and host threads. 应用范围是单一calling thread对于global, page lock host, memory of other device的操作是visible to all threads in all device and host threads



#### Volatile

> Reference
>
> 1. CUDA Toolkits Document I.4.3.3 [link](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#volatile-qualifier)



* 是什么

compiler可以对global memory/shared memory的read write进行优化，例如cache在L1 cache或者register上，只要符合memory fence的要求就可以进行优化。

Declaring a variable in global or shared memory using the volatile qualifier prevents compiler optimization which might temporally cache data in registers or local memory. With the volatile qualifier, the compiler assumes that the variable’s value can be changed or used at any time by any other thread.  声明volatile以后，compiler假设某个thread对内存的操作会any time被其余的thread使用，所以不适用cache进行优化，全部的写入会写入到gloabl memory/shared memory上。这样另一个thread可以读取对应的内存并且得到正确的数值。




## Hardware Implementation

### PCIe

GPU与CPU通过PCIe链接

PCIe：多个link，每个link包含多个lanes

lane：Each lane is 1-bit wide (4 wires, each 2-wire pair can transmit 8Gb/s in one direction) 支持双向数据传播

<img src="Note.assets/Screen Shot 2022-07-14 at 5.44.07 PM.png" alt="Screen Shot 2022-07-14 at 5.44.07 PM" style="zoom:50%;" />



北桥南桥都是用PCIe来链接

<img src="Note.assets/Screen Shot 2022-07-14 at 5.45.09 PM.png" alt="Screen Shot 2022-07-14 at 5.45.09 PM" style="zoom:50%;" />

### DMA

Direct Memory Access：充分利用bandwidth和IO bus。DMA使用physical address for source and destination, 使用pinnned memory传输数据。

作用：在使用pinned memory做数据拷贝以后，系统使用DMA，可以更充分的利用PCIe的带宽。如果不使用DMA拷贝，则系统无法充分利用PCIe的带宽




## Others



#### restrict

> Reference
>
> 1. CUDA C++ Programming Guide chapter B.2.5



* 作用

不使用restrict，compiler会假设input ptr存在底层数据overlap，也就会存在write to one pointer would overwrite content of another register. 所以每次读取ptr data的时候都会从array中读取，不进行任何的reuse。

这也就导致不适用restrict会导致compiler的一些优化不能进行。这包含

1. reduce number of memory access
2. reduce number of computaiton 

需要all ptr都使用restrict才能起到作用。



* 缺点

由于对数据进行了复用，要求kernel使用更多的register，也就增加了kernel register pressure。

对于一些register是压力的kernel来说，使用restrict增加了register 使用反而是个不好的选择



* 例子

```cpp
// 不使用resirtci
void foo(const float* a,
         const float* b, 
         float* c) {
    c[0] = a[0] * b[0];
    c[1] = a[0] * b[0];
    c[2] = a[0] * b[0] * a[1];
    c[3] = a[0] * a[1];
    c[4] = a[0] * b[0];
    c[5] = b[0];
    ...
}

// 使用restrict后compilier优化的对应代码
void foo(const float* __restrict__ a,
         const float* __restrict__ b,
         float* __restrict__ c)
{
    float t0 = a[0];
    float t1 = b[0];
    float t2 = t0 * t1;
    float t3 = a[1];
    c[0] = t2;
    c[1] = t2;
    c[4] = t2;
    c[2] = t2 * t3;
    c[3] = t0 * t3;
    c[5] = t1;
    .
}
```

