#include <stdio.h>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include "measure.cu"
#include "utilities.cu"

#define device_t float*
#define host_t float
#define v_size sizeof(host_t)

__global__ void kernelVectorAdd(device_t a, device_t b, device_t c) {
  c[blockIdx.x + 32 * threadIdx.x] = a[blockIdx.x + 32 * threadIdx.x] +
  b[blockIdx.x + 32 * threadIdx.x];
}

__global__ void kernelMatrixTransform(device_t source, device_t target, unsigned int nVectors, unsigned int nValues, unsigned int threads) {
  /* linear index in the buffer */
  unsigned int i = blockIdx.x * threads + threadIdx.x;

  unsigned int sy = i / nValues;
  unsigned int sx = ((i * 4) % nValues);

  unsigned int ty = sy / 2; 
  unsigned int tx = sx; 

  target[ty * nValues / 2 + tx    ]       = source[sy * nValues + sx   ];
  target[ty * nValues / 2 + tx + 1]       = source[sy * nValues + sx + 1];
  target[(ty + 1) * nValues / 2 + tx    ] = source[sy * nValues + sx + 2];
  target[(ty + 1) * nValues / 2 + tx + 1] = source[sy * nValues + sx + 3];


  // target[((ty) * (nValues / 2)) + tx] = 
  //target[i] = 
  //tx;
  //source[sy * nValues + sx];
  //target[i] = sx;
  //target[ty * (nValues / 2) + 1 + tx * 2] = source[i + 1];
 // target[(ty + 1) * (nValues / 2) + tx * 2] = source[i + 2];
 // target[(ty + 1) * (nValues / 2) + 1 + tx * 2] = source[i + 3];
//
  //target[ty * (nValues / 2) + tx + 1] =
  //ty;
  //source[sy * nValues + sx + 1];
  
  //target[(ty + 1) * (nValues / 2) + tx] =     
  //ty;
  //source[sy * nValues + sx + 2];
  
  //target[(ty + 1) * (nValues / 2) + tx + 1] =    
  //ty;
  //source[sy * nValues + sx + 3];
}

int main(void) {
  /* usefull constants */
  const unsigned int nVectors = 8;
  const unsigned int nValues = 16;

  const unsigned int nBytes = nVectors * nValues * v_size;
  const unsigned int nElements = nVectors * nValues;

  /* allocate host memory */
  host_t *a = (host_t*) malloc(nBytes);
  host_t *b = (host_t*) malloc(nBytes);
  host_t *c = (host_t*) malloc(nBytes);

  printf("sizeof(host_t): %ld bytes\n", sizeof(host_t));
  printf("host: %p %p %p\n", a, b, c);
  printf("allocated %u bytes on the host\n", nBytes * 3);

  /* initialize host memory */
  for (unsigned int i = 0; i < nElements; i++) {
    a[i] = b[i] = i; 
    c[i] = 0; 
  }

  /* allocate device memory */
  device_t deviceA = NULL;
  device_t deviceB = NULL;

  cudaMalloc(&deviceA, nBytes);
  cudaMalloc(&deviceB, nBytes);

  printf("device: %p %p\n", deviceA, deviceB);
  printf("allocated %u bytes on the device\n", nBytes * 2);

  unsigned int threads = 8;

  const unsigned int blockSize = nElements / threads;
  printf("%u threads per %u elements = %u block size\n", threads, nElements, blockSize); 

  cudaMemcpy(deviceA, a, nBytes, cudaMemcpyHostToDevice);
  kernelMatrixTransform<<<blockSize, threads>>>(deviceA, deviceB, nVectors, nValues, threads);
  cudaMemcpy(b, deviceB, nBytes, cudaMemcpyDeviceToHost);

  prettyPrint(a, nVectors, nValues);
  prettyPrint(b, nVectors * 2, nValues / 2);

  /* free all pointers we got */
  cudaFree(deviceA);
  cudaFree(deviceB);
  free(a);
  free(b);
  free(c);
}
