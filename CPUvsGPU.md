

> Reference
>
> 1. Berkeley CS 267 Lecture 7
> 2. UIUC 408 L1
> 3. Programming Massively Parallel Processors 3rd chapter 1



## CPU vs GPU



* 什么是GPU

GPU是heterogeneous chip. 有负责不同功能的计算模块

<img src="Note.assets/Screen Shot 2022-02-10 at 11.38.25 AM.png" alt="Screen Shot 2022-02-10 at 11.38.25 AM" style="zoom:50%;" />



SMs: streaming multiprocessors

SPs: streaming processors : each SM have multiple SP that share control logic and instruction cache



* 为了设么设计

GPU design for high throughput, don't care about throughput so much

CPU design for low latency

<img src="Note.assets/Screen Shot 2022-05-21 at 11.15.12 AM.png" alt="Screen Shot 2022-05-21 at 11.15.12 AM" style="zoom:50%;" />



* CPU GPU

CPU : multicore system : latency oriented 

GPU : manycore / many-thread system : throughput oriented



## Idea to design throuput oriented GPU

* Idea 1 ： 去除CPU中让CPU serialize code运行更快的

CPU中包含out of order execution, branch predictor, memory prefetch等机制让CPU运行serialize code fast，但是这些部分占用很大的memory和chip。

GPU去除这些部分。

<img src="Note.assets/Screen Shot 2022-02-10 at 11.45.50 AM.png" alt="Screen Shot 2022-02-10 at 11.45.50 AM" style="zoom:50%;" />



* Idea 2 ：larger number of smaller simpler core

相比起使用small number of complex core, GPU的工作经常simple core就可以处理。

但也带来了挑战，需要programmer expose large parallel从而充分利用全部的core



* idea 3：让simple core共享instruction stream，减少负责Fetch Decode的芯片面积

因为很多工作都是parallel的，所以多个small simple core共享instruction stream就可以，减少了chip上负责instruction stream的部分。

SIMT single instruction multiple threads. 

SIMT 与 SIMD 有一些不一样。SIMT可以平行thread，而SIMD只可以平行instruction



* idea 4：使用mask来解决branching

在CPU中使用branch prediction

在GPU中，使用mask来解决branching

<img src="Note.assets/Screen Shot 2022-02-10 at 11.50.08 AM.png" alt="Screen Shot 2022-02-10 at 11.50.08 AM" style="zoom:50%;" />



* idea 5：hide latency instead of reduce latency

CPU通过fancy cache + prefetch logic来avoid stall

GPU通过lots of thread来hide latency。这依赖于fast switch to other threads, 也就需要keep lots of threads alive.

<img src="Note.assets/Screen Shot 2022-02-10 at 11.51.41 AM.png" alt="Screen Shot 2022-02-10 at 11.51.41 AM" style="zoom:50%;" />



* GPU Register 特点

GPU的register通常很大，在V100里与half L1 cahce+shared memory一样大

经常也被叫做inverted memory hierchy

