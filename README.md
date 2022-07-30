# CUDA Optimization Guide

## Acknowlegement

This repo was originally part of my [HPC Note](https://github.com/XiaoSong9905/HPC-Notes). With more content related to CUDA added to the original note, I decided to open a seprate repo dedicated to CUDA optimization. 

Correction on mistakes is highly welcomed. Please post a issue if you found one.

To open markdown file with better format, [typora](https://typora.io) is recommended (its beta version is free).




## Disclaimer
I do not contain the copyright of some image files included in this note. The copyright belongs to the original author. 

Any content inside this repo is **OPEN FOR EDUCATION PURPOSE** but **NOT ALLOWED FOR COMMERCIAL USE**.




## File Structure
```shell
# Difference in architecture difference behind CPU and GPU
CPUvsGPU.md

# Memory model of CUDA and memory related optimization techniques (including synchronization)
MemoryModel.md

# Program model of CUDA and program related optimization techniques (including stream)
ProgramModel.md

# Arithmetic related topic. accuracy, speed, etc
Arithmetic.md

# Measure Performence (including nsight, etc)
MeasurePerformence.md

# Other common use optimization techniques that not included as part of programmodel / memory model
CommonOptimizationTechniques.md

# Cases that refer to the above optimization techniques and show how those optimization techniques can be applied to real applications.
Cases.md
```



## Major Refrence

> Note: I also refer to other papers / blogs that's not listed below.

* Courses
  * UIUC ECE 408
  * UIUC ECE 508
  * UC Berkeley CS 267
  * CMU 16.418
  
* Book (CUDA)
  * Programing Massively Parallel Processors 3rd edition
  * CUDA C++ Best Practices Guide
  * CUDA C++ Programing Guide
  * Professional CUDA C Progaming
  
* Book (Arch)
  
  * General-Purpose Graphics Processor Architecture
  * Processor Microarchitecture: An Implementation Perspective
  
* Papers
  * Algorithm and Data Optimization Techniques for Scaling to Massively Threaded Systems

* Blogs & code

  * CUTLASS: Fast Linear Algebra in CUDA C++ [link](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)

  * Cutlass [github](https://github.com/NVIDIA/cutlass)