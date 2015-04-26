# cuda_knapsack_01
Cuda Implementation of Knapsack 0/1 Dynamic Programming Problem

In order to solve the dependency of the bottom up table, the kernel function is called in a sequential order, incrementing the target ID within a while loop. This could be optimized further through the use of Dynamic Parallelism.

Further benchmarking is required.

For n = 10;
CUDA = 0.000059424 seconds
CPU = 0.03348 seconds

For n = 100;
... <Further benchmarking needs to be done>
