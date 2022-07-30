

# Measure Performence

> Reference
>
> 1. UIUC ECE 408 Lecture 27
> 2. UC Berkeley CS267 Project 1



#### Speedup

time ( sequential ) / time ( parallel )

在measure sequential的时候，一般measure最好的，找到的baseline，而不是自己写的baseline



#### Efficency

speedup on P processor / P 



常用于衡量多cpu core的程序

希望是接近1，大多数情况都是小于1的



superlinear speedup：由于使用了extra resoruce (例如cache等等) 所以efficency大于1



对于GPU来说，efficency衡量的就不是很有效了。

对于GPU，一般衡量efficency是 compare resource used with GPU's peak value



#### Scalability

对于多少个processor，efficency是接近1。对于多少个processor efficency是减小的

speed up <-> num of processor的图

good scalability : speed up curve not fall of for max measureable value of P



下图的这个例子里，就不是scalable code

<img src="Note.assets/Screen Shot 2022-07-14 at 6.02.25 PM.png" alt="Screen Shot 2022-07-14 at 6.02.25 PM" style="zoom:50%;" />



#### Strong Scaling

In strong scaling we keep the problem size constant but increase the number of processors



#### Weak Scaling

In weak scaling we increase the problem size proportionally to the number of processors so the work/processor stays the same
