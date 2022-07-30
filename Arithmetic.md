# Arithmetic



## Data representation

> Reference
>
> 1. Programming Massively Parallel Processors 3rd edition Chapter 6



### IEEE Standard

IEEE-753 Floating Point Standard 是硬件与软件使用的标准

`S` sign bit

`E` exponent bit

`M` mantissa bit



single precision float : 1 bit S, 8 bit E, 23 bit M

double precision float : 1 bit S, 11 bit E, 52 bit M



### Normalized representation



<img src="Note.assets/Screen Shot 2022-06-01 at 2.21.54 PM.png" alt="Screen Shot 2022-06-01 at 2.21.54 PM" style="zoom:50%;" />



下面的图片展示了normalized float point的representable number

<img src="Note.assets/Screen Shot 2022-06-01 at 2.25.08 PM.png" alt="Screen Shot 2022-06-01 at 2.25.08 PM" style="zoom:50%;" />



#### M Bit

在normalized representation下，要求M的形式是 `1.M`, 这样保证每一个float都有unique mantissa bit

number of mantissa bit 用于 floating point **precision**

有n个m bit，每个major interval就会有2^n个representable number。如果value无法被representable number表示，就会被rounded。

因为normalizd representation的限制，靠近0的部部分(0-0.5) 有representation gap，这个部分的数据无法被表达，要不被round到0，要不被round到0.5

`0` 无法用上面的normalized representation公式表达，这是一个很大的问题。

M bit每多一位，major interval中可以表达的number就多一倍，accuracy就会多一倍



#### E Bit

使用 2's complement `excess represerntation`, 这样可以通过比较bit来直接比较floating point的大小，让hw上实现起来更快

<img src="Note.assets/Screen Shot 2022-06-01 at 2.24.29 PM.png" alt="Screen Shot 2022-06-01 at 2.24.29 PM" style="zoom:50%;" />

number of exponent bit 用于 floating point **range**

有n个e bit，就会有n+1个interval。Figure 6.5中有三个major interval，分别对应

<img src="Note.assets/Screen Shot 2022-06-01 at 2.42.41 PM.png" alt="Screen Shot 2022-06-01 at 2.42.41 PM" style="zoom:50%;" />

major interval的大小距离0越近，interval大小越近。因为每个interval有固定数量个representable number，所以靠近0的interval的precision越大。对于很多division和converge的问题，这是大的问题



### Denormalized representation

当E=0的时候，mantissa的constrain没有，assume `0.M` 的形式而不是 `1.M` 的形式。

从理解的角度上来说，就是把靠近0的major interval spread out在0到interval end之间。原来靠近0的interval是(0.5, 1)有4个representable number（4是因为假设的Mbit个数），现在把这4个representable number分散在(0, 1)之间

<img src="Note.assets/Screen Shot 2022-06-01 at 2.47.17 PM.png" alt="Screen Shot 2022-06-01 at 2.47.17 PM" style="zoom:50%;" />



<img src="Note.assets/Screen Shot 2022-07-28 at 9.46.34 PM.png" alt="Screen Shot 2022-07-28 at 9.46.34 PM" style="zoom:50%;" />



#### GPU Hardware

支持denormalized representaiton的硬件比较复杂，因为需要判断e是否是0，从而决定对应的M的format (1.M, 0.M)

computaiton capacity 1.3+ 支持 denormalized double floating point

computation capacity 2.0+ 支持 denormalized single floating point



#### Special Bit

<img src="Note.assets/Screen Shot 2022-06-01 at 2.52.02 PM.png" alt="Screen Shot 2022-06-01 at 2.52.02 PM" style="zoom:50%;" />



## Arithmetic accuracy

> Reference
>
> 1. Programming Massively Parallel Processors 3rd edition Chapter 6



### Accuracy from Hardware

计算中的accuracy问题是因为计算的结果（或者临时结果）无法被准确表达，被rounding，所以导致accuracy问题。

plus minus，硬件accuracy控制在 0.5D ULP (units in last place)

division transcendental通过polynomial approximate实现，硬件accuracy一般比加减的accuracy要大



### Accuracy from Algorithm / Software

#### 大数吃小数

##### 原因

越靠近0（小数）的precision越大，越靠近inf（大数）的precision越小。

大数+小数=大数，结果的大数会因为rounding的问题无法表达被加上来的小数。



例子：sequential相加，从大数加到小数，结果中小数被忽略

<img src="Note.assets/Screen Shot 2022-06-01 at 2.56.28 PM.png" alt="Screen Shot 2022-06-01 at 2.56.28 PM" style="zoom:50%;" />



##### 解决方法

sort input data，然后sequential相加，这样很多小数彼此相加就能得到足够大的数，再与大数相加的时候不会被吃掉

<img src="Note.assets/Screen Shot 2022-06-01 at 2.56.58 PM.png" alt="Screen Shot 2022-06-01 at 2.56.58 PM" style="zoom:50%;" />



#### Numerical stable / unstable

If an algorithm fails to follow a desired order of operations for an input, it may fail to find a solution even though the solution exists.

`numerically stable`: Algorithms that can always find an appropriate operation order, thus finding a solution to the problem as long as it exists for any given input values, are called . 

` numerically unstable`: Algorithms that fall short are referred to as



例子：gaussian elimination需要使用pivioting的方法来解决numerical unstable的问题



## Optimization

> Reference
>
> 1. CUDA C++ Programming Guide chapter 5.4.1
> 2. CUDA C++ Best practice chapter 11.1
> 3. Professional CUDA C Programming Guide chapter 7



### Intrinsic and Standard Function

Standard functions are used to support operations that are accessible from, and standardized across, the host and device.  such as sqrt, exp, and sin. 是跨平台的方程。`func`

Intrinsic functions can only be accessed from device code. compiler has special knowledge about its behavior, which enables more aggressive optimization and specialized instruction generation. 只可以在device上调用，compiler知道更多信息可以更aggresive的优化, 很多intrinsic funciton对应hardware unit。 `__func`

Intrinsic functions are faster than their equivalent standard functions but less numerically precise.



* powf PTX as example

`__powf` intrinsic 在用ptx翻译以后，只有7 line of instruction

`powf` standard function 再用ptx翻译以后，有344 line of instruction

从perf的角度上对比，intrinsic function 比起 standard function 有明显的perf boost



* floating point intrinsic function

对于普通的floating point操作` __fadd, __fsub, __fmul`，可以通过intrinsic来控制rounding方法

`__fmul_rn` 就是用rn控制了rounding to nearest

<img src="Note.assets/Screen Shot 2022-07-28 at 10.11.17 PM.png" alt="Screen Shot 2022-07-28 at 10.11.17 PM" style="zoom:50%;" />



fast reciprocal square root & square root:

compiler可以优化一些 `1.0 / sqrtf()` 为 `rsqrtf()` ，但并不总发生

应该手动使用rsqrtf()



fast intrinsic division:

速度会更快，但是有accuracy 降低

```cpp
__fdividef(x, y)
```



#### limite use of FMAD

当使用MAD的时候，accuracy会下降，所以有些程序在避免使用MAD

> 不是很确定这点，在berkeley cs 267， james说用fma的话accuracy会增加，因为只有一次rounding。不知道为什么在GPU上使用MAD会导致accuracy下降



* global compiler flag

--fmad option to nvcc globally enables or disables the FMAD optimization for an entire compilation unit

--fmad=false prevents the compiler from fusing any multiplies with additions, hurting performance but likely improving the numerical accuracy of your application.



* part of program not use mad

` __fmul and __dmul` 可以用来替换`*`， 这样compiler就不会声称MAD的代码了。

这样就可以通过 `--fmad`控制全局，通过`__fmul`控制部分



### Single VS Double

single double float的perf差异源于

1. communication，需要传输两倍数据
   1. I/O between host and device
   2. I/O between global memory to device register
2. computation，占用两个register来计算
3. 由于double占用了更多的register，reduce resource available to each thread in thread block, 也就减少了block on sm



large numerical differences between single- and double- precision results that can accumulate in iterative applications as imprecise outputs from one iteration are used as inputs to the next iteration. 当进行iteration计算的时候，更建议使用double precision，因为error会在每个iteration accumulate，最终导致很大的error



### 计算替换

#### Exponential

在可能的时候，使用square root, cube roots, inverse等等来计算exp，可以显著的（1）加速 （2）更精确

<img src="Note.assets/Screen Shot 2022-06-28 at 2.33.44 PM.png" alt="Screen Shot 2022-06-28 at 2.33.44 PM" style="zoom:50%;" />

For exponentiation using base 2 or 10, use the functions exp2() or expf2() and exp10() or expf10() rather than the functions pow() or powf() 对于base 2 10的exp，有专门对应的方程，它们的速度更快

small integer power $x^2, x^3$, use explicit multiplication 总是比general purpose power要更快更准确的



#### Loop conter

the compiler can optimize more aggressively with signed arithmetic than it can with unsigned arithmetic.

所以要使用signed loop counter



#### 16 bits float

使用2个half precision的数据结构(half2 datatype is used for half precision and `__nv_bfloat162` be used for `__nv_bfloat16` precision) + vector intrinsic (`__hadd2, __hsub2, __hmul2, __hfma2`)从而每个instruction可以一次处理两个16 bit数据。



The intrinsic `__halves2half2` is provided to convert two half precision values to the half2 datatype.
The intrinsic `__halves2bfloat162` is provided to convert two `__nv_bfloat` precision values to the `__nv_bfloat162` datatype.



本质原因是register size是32 bits的，所以一次使用到register的计算可以是32 bits



#### Int Division & Modulo

int的division和modulo会花费20 instruction。

compiler会进行一些conversion，但只有在n是数值(i.e. 一个数字，而不是变量)的时候。避免直接使用division/modulp是推荐的（当知道一个varaible是power of 2的时候）

一个principle是，如果知道var会是某个数值，则直接在代码中用数值替换。如果知道var是power of 2，则用下面的公式替换。

 

如果n是power of 2（也就是编程的时候自己知道，则用下面的公式代替）

i/n = i >> log2(n)

i % n = (i & (n-1))



#### Type conversion

assign literal to single float var的时候要使用 `f` suffix，否则float literal默认是double的，在assignment的时候compiler会增加一个type conversion instruction



#### Sin & Cos

single float: sinf(x), cosf(x), tanf(x), sincosf(x) 

double float: sincos(x)



* 特点

1. 方程会根据input magnitude选择运行slow path/ fast path



* slow path

argument sufficent large in magnitude

使用lenghty computation实现

slow path会使用local memory来储存intermediate variable来避免lengthy computation的过程中使用过多的register，这导致slow path的throuput与latency都比起fast path要小很多

使用taylor expansion来实现。



* fast path

argument sufficent small in magnitude

使用few multiply-add实现



* more specific math

Replace `sin(π*<expr>) with sinpi(<expr>)`, `cos(π*<expr>) with cospi(<expr>)`, and `sincos(π*<expr>) with sincospi(<expr>)`



1. 更快
2. 更accurate



### Compiler Flag

<img src="Note.assets/Screen Shot 2022-07-28 at 10.15.40 PM.png" alt="Screen Shot 2022-07-28 at 10.15.40 PM" style="zoom:50%;" />

* denormalized numbers are flushed to zero

```shell
// enable higher throughput
-ftz=true
```



* less precision division

```shell
-prec-div=false
```



* less precision square root

```shell
-prec-sqrt=false
```



* use intrinsic instead of standard function

会自动把全部的`func` 变为`__func` 的实现

注意，只对single precision起租用

```cpp
-use_fast_math=true
```



* use mad

```cpp
--fmad=true
```
