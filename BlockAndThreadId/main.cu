#include <stdio.h>

#define NUM_BLOCKS 16
#define BLOCK_WIDTH 16

__global__ void hello()
{
    printf("Hello world! blcokid: %d\nthreadid:%d\n", blockIdx.x, threadIdx.x);
}


int main(int argc,char **argv)
{
    // launch the kernel
    hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();

    // force the printf()s to flush
    cudaDeviceSynchronize();

    printf("That's all!\n");

    return 0;
}