## Common Optimization Techniques

> Reference
>
> 1. Algorithm and Data Optimization Techniques for Scaling to Massively Threaded Systems

<img src="Note.assets/Screen Shot 2022-06-04 at 5.09.40 PM.png" alt="Screen Shot 2022-06-04 at 5.09.40 PM" style="zoom:50%;" />



### Data Layout Transformation

> Reference
>
> DL A Data Layout Transformation System for Heterogeneous Computing

GPU充分利用burst memory是很重要的优化方法

如果burst内的数据没有立刻被使用的话（DRAM的buffer中存放burst），则会被下一个burst代替，需要重新传输。

对于CPU来说，data layout对程序的影响没有那么显著，因为CPU有large cache per thread，可以cache部分数据，没有那么依赖于DRAM的burst data。下面的array of struct结构中，thread0的cache会储存整个struct的内容。

对于GPU来说，data layout对程序的影响很显著，因为GPU的cache比较小。GPU的cache主要适用于memory coalesing，而不是locality

SoA或者DA(discrete array)的结构对GPU有用，因为充分利用burst 的结果

<img src="Note.assets/Screen Shot 2022-06-04 at 5.00.49 PM.png" alt="Screen Shot 2022-06-04 at 5.00.49 PM" style="zoom:50%;" />



ASTA array of structures of tiled arrays 是一种 SoA的变体。相当于AoS of mini-SoA(of size coarsening factor)

1. 解决OpenCL需要对不同hw有不同数据结构的kernel的问题
2. 解决`partition camping`，也就是数据集中在某一个bank/channel上，没有充分利用DRAM aggregate bandwidth

通常`coarsening factor` (下面eg是4) at least the number of thread partitioning in memory access (num thread in block)

<img src="Note.assets/Screen Shot 2022-06-04 at 5.07.21 PM.png" alt="Screen Shot 2022-06-04 at 5.07.21 PM" style="zoom:50%;" />



在NVIDIA的arch下，DA与ASTA的perf相似

<img src="Note.assets/Screen Shot 2022-06-04 at 5.09.01 PM.png" alt="Screen Shot 2022-06-04 at 5.09.01 PM" style="zoom:50%;" />



### Scatter to Gather

> UIUC ECE 508 Lecture 2, 5

scatter对于编程来说更加直接

GPU应该避免使用scatter，应该使用gather的方法

在GPU上的程序改变scatter为gather可以提升性能



* 是什么

scatter : parallel over input, writing value to non-contigious memory location

gather : parallel over output, reading values from non-contigious memory location。也叫做owner compoutes rules



* scatter 缺点

1. contentious write (write conflict) 需要被hardware serialize。（下图红色的arrow）。当thread多的时候会有很多conflict，write到某一个位置会被serialized
2. random write无法充分利用memory burst
3. atomic的arch直到最近才被支持



* gather 优点

1. write的时候充分利用burst
2. 没有write conflict，不需要serialize write
3. input会有重复的，可以利用好cache



* 一般程序特点 （为什么大家习惯写scatter code）

1. input一般是irregular的，output一般是regular的。
   1. 从irregular data映射到regular data是简答的，这也是为什么很多程序是scatter的
   2. input是particle coordinate(x,y,z), output是3d spatial grid

2. 有些时候each input只影响有限个output，所以conflict write的影响没有那么大



* gather缺点

1. 存在overlapping read，但是可以被hardware使用cache/shared memory来缓解

<img src="Note.assets/Screen Shot 2022-06-04 at 9.52.02 PM.png" alt="Screen Shot 2022-06-04 at 9.52.02 PM" style="zoom:50%;" />



#### Example

1. DCS



### Tiling

是什么：buffer input into on-chip storage, to be read multiple times. 

效果：reduce global memory bandwidth pressure



<img src="Note.assets/Screen Shot 2022-06-04 at 9.55.10 PM.png" alt="Screen Shot 2022-06-04 at 9.55.10 PM" style="zoom:50%;" />



* 为什么有shared memory/scratch pad

on chip storage越大（shared memory越大，tile size越大），越能减少bandwidth的压力

如果on chip storage只可被单独thread可见，则on chip storage会比较小（无法给几千个thread分配大的on chip storage）。解决方法是share on chip storage across thread通过scratchpad/shared memory



* 效果

取决于不适用tiling，只使用cache的效果怎么样。

在modern GPU上，cache相对更加复杂+大，所以使用tiling的效果就没有那么多。UIUC 408 Lecture 14里面的例子里，使用tilning只提升了40%左右的速度，原因是因为绝大多数access to global memory都是通过L1 cache的，cache hit rate有98%。



#### Example
1. GEMM
2. Conv




### Joint Register and Shared Memory Tiling

> Reference
>
> UIUC ECE 508 Lecture 4
>
> Benchmarking GPUs to Tune Dense Linear Algebra



* register 特点

1. low latency 
2. high throughput : per thread per clock cycle可以进行多个register访问与计算
3. load data是serial的
4. private to each thread
5. 进行register tiling需要thread coarsening



* shared memory 特点

1. comparable latency
2. lower throughput compared with register 
3. can be loaded cooperatively by multiple thread



* 为什么joint

1. hardware path是不同的，可以combine tiling for register and shared memory 来增加throughput



#### Source of reuse

在做shared memory tiling的时候，reuse来自于shared memory的数据被多个thread访问，而不是来自于一个thread内部访问一个value多次。

Tile size是T * T的话，每一个thread load一个M，一个N到shared memory，sync（确保数据都在shared memory中），然后遍历一小行M和一小列N来计算一个P，sync（确保shared memory被使用完），然后处理下一个tile

对于每一个M的值，被T（结果P中tile的num col）个thread使用。

对于每一个N的值，被T（结果P中tile的num row）个thread使用。

<img src="Note.assets/Screen Shot 2022-06-06 at 6.00.42 PM.png" alt="Screen Shot 2022-06-06 at 6.00.42 PM" style="zoom:50%;" />

从上面的分析中知道S的大小是independent of reuse factor, 所以S并不一定要等于T。

同时因为计算P的会有两个sync（load to shared memory, wait for all comp on shared memory finish)， 所以S的大小也不能太小，否则sync会占用主要的时间

同时tile size不一定是square的。

every M value reused U time

every N value reused T time

<img src="Note.assets/Screen Shot 2022-06-06 at 6.16.45 PM.png" alt="Screen Shot 2022-06-06 at 6.16.45 PM" style="zoom:50%;" />

#### Example
1. GEMM : joint register and shared memory tiling



### Grid-stride loop / thread granularity / thread coarsening 

> Ref
>
> 1. Berkeley CS 267 Lecture 7
> 2. Programming Massively Parallel Processors 3rd edition Chapter 5
> 3. UIUC 508 Lecture 3
> 4. NVIDIA Tech BLOG CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops  [link](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)



* 是什么

1. 原来需要多个thread完成的工作，现在使用一个thread完成，从而减少redundant work （parallel经常会有redundant computation在不同的thread上）
2. 两种类型：一个是让一个thread在1个iteration中完成原来多个thread的工作。另一个是让一个thread block在完成当前thread block的工作后（与原来的工作相同），再处理下一个thread block。
3. thread 0会处理 elem 0, elem 0 + num thread in grid, elem 0 * 2 * num thread in grid. 每一次的step是grid，也是为什么叫做grid stride loop的原因
   1. 对比起来，每个thread处理一个元素的loop叫做`monolithic kernel`

<img src="Note.assets/Screen Shot 2022-06-05 at 7.58.14 PM.png" alt="Screen Shot 2022-06-05 at 7.58.14 PM" style="zoom:50%;" />



* 优点

1. Eliminating redundant work can ease the pressure on the instruction processing bandwidth and improve the overall execution speed of the kernel. 对重复工作结果进行复用，从而减少instruction processing stream的贷款限制
   1. 可以理解为一些会重复的computation，现在shared through register。本来register 是local to each thread, 无法shared across thread的
   2. 访问register的throughput很大，per thread per cycle可以访问多个register file
   3. 访问register的latency很小，只有1 clock cycle

2. scalability，可以支持program size > total num thread on hardware. 

3. 可以tune code with num block = multiply of SM, 然后使用grid stride loop来支持不同大小的问题。

   1. ```cpp
      int numSMs;
      cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
      // Perform SAXPY on 1M elements
      saxpy<<<32*numSMs, 256>>>(1 << 20, 2.0, x, y);
      ```

4. Thread reuse amortizes thread creation and destruction cost along with any other processing the kernel might do before or after the loop (such as thread-private or shared data initialization).

5. 更容易debugging，可以把launch 1 thread 1 block的kernel来debug，不需要改变kernel内容

   1. ```cpp
      saxpy<<<1,1>>>(1<<20, 2.0, x, y);
      ```

6. readability as sequential code 。与sequential code一致都有for loop的存在，更好理解代码

7. 在unroll loop的时候，loop body变得更大了，可以更好的使用ILP



* 缺点

1. （在情况1下）每个thread使用更多的register，可能导致一个sm内总的thread数量减少（因为register constrain）。导致insufficent amount of parallelism。
   1. not enough block per sm to keep sm busy
   2. not enough block to balance across sm (thread合并了以后，总的thread数量减小，总的block数量也就减少了，而且每个block的时间久了，容易导致imbalance)
   3. not enough thread to hide latency。通过warp间swap来hide latency，但是当总thread减少，总warp减少

2. larger computation tiles. 产生more padding and wasted computation，一般通过reduce number of thread per block 解决
   1. 如果一个thread在coarsening以后干了k*k个thread的工作，把原来的block size分别变为width/k和height/k来避免more padding and waste computation




（one output per thread的idle)

<img src="Note.assets/Screen Shot 2022-06-05 at 8.46.53 PM.png" alt="Screen Shot 2022-06-05 at 8.46.53 PM" style="zoom:50%;" />

(two output per thread的idle，更多idle)

<img src="Note.assets/Screen Shot 2022-06-05 at 8.47.17 PM.png" alt="Screen Shot 2022-06-05 at 8.47.17 PM" style="zoom:50%;" />





* 为什么使用

increase efficency outweight reduce of parallelism就可以



* 例子1

```cpp
int numSMs;
cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
add<<<32 * numSMs, blockSize>>>(N, x, y);

// GPU function to add two vectors
__global__
void add(int n, float *x, float *y) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  // 这里通过for loop对thread进行复用
  for (int i = index; i < n; i+=stride)
    y[i] = x[i] + y[i];
}
```



#### Example
1. DCS : thread granularity
2. 7 point stencil 
3. GEMM: thread granularity



### Privatization

是什么：buffer output into on-chip storage (or even register), to be write multiple times. 

每个thread/group of thread有自己的local copy of output，首先写在local copy里，然后再合并到final output中。下图中显示了privitization at multiple level

效果：避免多个thread通过使用atomic同时写入一个内存地址，使用atomic会drastically decrease memory throughput。

<img src="Note.assets/Screen Shot 2022-06-04 at 9.58.57 PM.png" alt="Screen Shot 2022-06-04 at 9.58.57 PM" style="zoom:50%;" />



* GPU上使用的缺点

CPU上由于thread的数量较小，private copy of output不会是问题

GPU上由于thread的总数量很多，使用privitization需要注意

1. data foorprint of the copy。使用shared memory或者是register是否会导致 thread  per sm 减少
2. overhead of combining private copy会比较大，因为这里依旧需要atomic



解决方法是one copy for a group/block of thread in scratchpad / shared memory, 这样可以同时兼顾latency (5 cycle)与bandwidth（在shared memory上使用atomic的bandwidth依旧可以）



#### Example

1. Histogram




### Algorithm Cascading

> Reference
>
> 1. UIUC 408 Lecture 15 on reduction, algorithm cascading
> 2. Programming Massively Parallel Processors 3rd edition Chapter 8.5 three phase



* 是什么

混合sequential 与 parallel 算法，从而让每个thread有足够的工作(sequential)来避免parallel的overhead，而且允许thread之间通过parallle来进行计算



#### Example

1. Prefix-sum three phase



### Cutoff Binning

> Reference
>
> 1. UIUC ECE 508 Lecture 5 & 6



* 是什么

只考虑在一个cutoff threshold内的元素之间的关系（基于physica），不考虑cutoff外的元素之间的关系或者cutoff外用简单的计算来近似



* 为什么用

为了解决data scalability的问题，使用近似算法从而得到 linear complexity的结果

cutoff binning允许O(n)复杂度算法



* 特点

cutoff binning对于cpu来说容易adopt，因为CPU可以使用scatter的方法

cutoff binning对于gpu来说难adopt，因为gpu使用gather的方法



#### Example

1. biomolecules



### Binning

> Reference
>
> 1. UIUC ECE 508 Lecture 5 & 6



* 为什么使用binning

gpu prefer gather computation over scatter computation. 也就需要有一个output element到input element的映射关系，这样才可以对于每一个output element用一个thread，通过映射关系找到全部对应的input element，计算结果。

但是parallel over output有scalability issue

一般input是irregular的，output是regular的，很难从regular到irregular找到映射，从irregular到regular的映射更简单一些(e.g.  atom 3d location is irregular, grid position is regular, use 3d location & divide to get grid location)



* 是什么 (input binning)

把irregular input放到某种regular的bin中，这样从regular output到irregular input的映射的时候，就可以到对应的regular bin中去找，加快速度。

每个bin包含某种property（e.g. spatial location), 相当于把data按照bin的property给coarlesed。

可以理解为data coarlesing，当访问bin的时候，访问all data inside bin。



#### Data Scalability

* 是什么

complexity 与 input size 不是linear的情况



* 什么时候产生

给output (regular)，判断哪些input(irregular)是相关的（或者哪些input是不相关的）是很难的。

如果通过遍历全部的input来找到每一个output的相关，会产生data scalability issue



* 为什么是问题

如果代码为了改变为parallel over gather而改变complexity为n log n / log n的话，对于large data来说是效果不好的，因为越是数据大，log的效果越明显

sequential O(N) algorithm can easily outperform O(n log n) parallel algorithm

但是使用gpu的情况又是在数据量很大的情况。

<img src="Note.assets/Screen Shot 2022-07-13 at 6.22.24 PM.png" alt="Screen Shot 2022-07-13 at 6.22.24 PM" style="zoom:50%;" />



* 对于HPC

first thing when have parallel algorithm, how to change it to O(n) so that it's data scalable



* poor scalability例子 ： DCS

DCS算法需要对每一个grid point计算每一个atom的contribution，complexity是 O(V^2)的



#### Example

1. CDS的简化算法



### Compaction

是什么：压缩数据中的hole，从而减少memory overhead（global memory, shared memory, memory transfer bandwidth)

<img src="Note.assets/Screen Shot 2022-06-04 at 5.12.26 PM.png" alt="Screen Shot 2022-06-04 at 5.12.26 PM" style="zoom:50%;" />



#### Example

1. SpMV



### Regularization

是什么：解决thread之间的load imbalance问题



* load imbalance的问题

1. 导致thread divergence
2. 一个block内如果有load imbalance，会导致resource在整个block运行结束之前（也就是imbalance里最多的work）不会释放，导致block占用有限的resource更多的时间（尽管在imbalance的时候，block不需要这么多的resource），导致num thread per SM降低



