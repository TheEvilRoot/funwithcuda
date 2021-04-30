#ifndef MEASURE_CU
#define MEASURE_CU

#include <chrono>

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

#endif
