#include <stdio.h>

__global__ void checkPrime(int * d_in)
{
  //get thread id
  int id = blockIdx.x + threadIdx.x;  
  	
	//num is now in local memory, much quicker
  int num = d_in[id];
	
  //couple of corner cases
  if(num == 0 || num == 1)
  {
    d_in[id] = 0;
    return;
  }
  
	//assume prime until proven otherwise
	d_in[id] = 1;		
	
	//quicker to check 2 on its own
	//then we can count up in 2s (only need to check odd numbers) starting from 3	
	if(num % 2 == 0)
	{
		d_in[id] = 0;
	}
	else
	{
    //only need to check upto ceil of sqrt(num)
    //better to start from 3 and count up rather than down    
    //do sqrt here not in loop to stop it being evaluated each time round
		int sqrtNum = (int)sqrt((float)num);
    
    for(int i = 3; i < sqrtNum + 1; i += 2)
		{
		  if(num % i == 0)
		  {
        d_in[id] = 0;
        return;
		  }		  
		}
	}
}

int main(int argc, char ** argv)
{
  //anything over 1000000 crashes it, not sure why
  //possibly to do with my vram
	const int ARRAY_SIZE = 100000;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

	//generate the input array on the host
	int h_in[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++) 
	{
		h_in[i] = i;
	}	 

	//declare GPU memory pointers
	int * d_in;	
 
	//allocate GPU memory
	cudaMalloc((void**) &d_in, ARRAY_BYTES);	
 
  //transfer the array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	//launch the kernel
  //not sure what the best ratio of blocks to threads is
	checkPrime<<<ARRAY_SIZE/100, 100>>>(d_in);

	//copy back the result array to the CPU
	cudaMemcpy(h_in, d_in, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	//print out the resulting array of primes
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
    if(h_in[i])
    {
      printf("%d\n", i);
    }		
	}

	cudaFree(d_in);
  
	return 0;
}