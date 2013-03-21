#include <stdio.h>

__global__ void checkPrime(int * d_out, int * d_in)
{
 int id = threadIdx.x;  
  
  if(id == 0 || id == 1)
  {
    d_out[id] = 0;
    return;
  }
	
	//num is now in local memory, much quicker
  int num = d_in[id];
	
	//assume prime until proven otherwise
	d_out[id] = 1;
		
	
	//quicker to check 2 on its own
	//then we can count up in 2s (only need to check odd numbers) starting from 3	
	if(num % 2 == 0)
	{
		d_out[id] = 0;
	}
	else
	{
    //only need to check upto sqrt(num)
    //better to start from 3 and count up rather than down    
    //do sqrt here not in loop to stop it being evaluated each time round
		int sqrtNum = (int)sqrt((float)num);
    for(int i = 3; i < sqrtNum + 1; i += 2)
		{
		  if(num % i == 0)
		  {
        d_out[id] = 0;
        break;
		  }		  
		}
	}
}

int main(int argc, char ** argv)
{
	const int ARRAY_SIZE = 64;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

	// generate the input array on the host
	int h_in[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++) 
	{
		h_in[i] = i;
	}	
  int h_out[ARRAY_SIZE];

	// declare GPU memory pointers
	int * d_in;	
  int * d_out;

	// allocate GPU memory
	cudaMalloc((void**) &d_in, ARRAY_BYTES);	
  cudaMalloc((void**) &d_out, ARRAY_BYTES);
	
  // transfer the array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// launch the kernel
	checkPrime<<<1, ARRAY_SIZE>>>(d_out, d_in);

	// copy back the result array to the CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// print out the resulting array
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
    if(h_out[i])
    {
      printf("%d\n", i);
    }		
	}

	cudaFree(d_in);
  cudaFree(d_out);
	
	return 0;
}