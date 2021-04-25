#include <stdio.h>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <chrono>

#define device_t float*
#define host_t float
#define v_size sizeof(host_t)

struct Measure {
  std::chrono::time_point<std::chrono::high_resolution_clock> mStart;
  std::chrono::time_point<std::chrono::high_resolution_clock> mEnd;

  Measure() {} 

  void fromNow() {
    mStart = std::chrono::high_resolution_clock::now();
  }

  void untilNow() {
    mEnd = std::chrono::high_resolution_clock::now();
  }

  float millis() {
    return nanos() / 1000000.0; 
  }

  float micros() {
    return nanos() / 1000.0; 
  }

  float nanos() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(mEnd - mStart).count() * 1.0;
  }
};

struct CudaMeasure {
  cudaEvent_t mStart;
  cudaEvent_t mEnd;
  float elapsedMs;

  CudaMeasure(): elapsedMs(0) {
    cudaEventCreate(&mStart);
    cudaEventCreate(&mEnd);
  }

  ~CudaMeasure() {
    cudaEventDestroy(mStart);
    cudaEventDestroy(mEnd);
  }

  void fromNow() {
    cudaEventRecord(mStart, 0);
  }

  void untilNow() {
    cudaEventRecord(mEnd, 0);
    cudaEventSynchronize(mEnd);

    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, mStart, mEnd);

    elapsedMs = elapsedTime;
  }

  float millis() {
    return elapsedMs;
  }

  float micros() {
    return  (elapsedMs * 1000.0);
  }

  float nanos() {
    return  (elapsedMs * 1000000.00);
  }

  float value() {
    return elapsedMs;
  }

};

__global__ void kernelVectorAdd(device_t a, device_t b, device_t c) {
  c[blockIdx.x + 32 * threadIdx.x] = a[blockIdx.x + 32 * threadIdx.x] +
  b[blockIdx.x + 32 * threadIdx.x];
}

int main(void) {
  host_t *a = (host_t*) malloc(1024 * v_size);
  host_t *b = (host_t*) malloc(1024 * v_size);
  host_t *c = (host_t*) malloc(1024 * v_size);
  host_t *d = (host_t*) malloc(1024 * v_size);

  for (unsigned long i = 0; i < 1024; i++) {
    a[i] = b[i] = i;
    c[i] = 0; 
  }

  device_t deviceA = NULL;
  device_t deviceB = NULL;
  device_t deviceC = NULL;

  cudaMalloc(&deviceA, 1024 * v_size);
  cudaMalloc(&deviceB, 1024 * v_size);
  cudaMalloc(&deviceC, 1024 * v_size);

  printf("sizeof(host_t): %ld bytes\n", sizeof(host_t));
  printf("device: %p %p %p\n", deviceA, deviceB, deviceC);

  cudaMemcpy(deviceA, a, 1024 * v_size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, b, 1024 * v_size, cudaMemcpyHostToDevice);

  printf("calling a kernel on the device...\n");

  CudaMeasure device;
  device.fromNow();

  kernelVectorAdd<<<32, 32>>>(deviceA, deviceB, deviceC);

  device.untilNow();

  printf("kernel is done\n");
  printf("device: %f micros\n", device.micros()); 

  cudaMemcpy(c, deviceC, 1024 * v_size, cudaMemcpyDeviceToHost);

  Measure cpu;
  cpu.fromNow();
  for (int i = 0; i < 1024; i++)
    d[i] = a[i] + b[i];
  cpu.untilNow();
  printf("cpu: %f micros\n", cpu.micros()); 

  for (int i = 0; i < 1024; i++)
    if (c[i] != d[i]) printf("missmatch on %d: %f != %f\n", i, c[i], d[i]);

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  free(a);
  free(b);
  free(c);
  free(d);

  return 0;
}
