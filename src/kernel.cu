#include <stdio.h>
#include <curand_kernel.h>
#include <chrono>

#define host_t float
#define device_t float*
#define t_size sizeof(host_t)

#define size_t unsigned long
#define time_point_t std::chrono::time_point<std::chrono::high_resolution_clock> 

template<typename A, typename B>
struct pair_t { A first; B second; };

host_t fZero(size_t x, size_t y, size_t w) { return 0; }
host_t fIndex(size_t x, size_t y, size_t w) { return (y * w + x); }

void printMatrix(FILE* file, host_t* matrix, size_t width, size_t height) {
  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++)
      fprintf(file, "%03.0f ", matrix[y * width + x]);
    fprintf(file, "\n");
  }
  fprintf(file, "\n");
}

void fillMatrix(host_t* matrix, size_t width, size_t height, host_t(*function)(size_t, size_t, size_t)) {
  for (size_t y = 0; y < height; y++) 
    for (size_t x = 0; x < width; x++)
      matrix[y * width + x] = function(x, y, width); 
}

pair_t<size_t, size_t> compare(host_t* a, host_t* b, size_t width, size_t height) {
  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      if (a[y * width + x] != b[y * width + x]) { 
        printf("missmatch %f %f\n", a[y * width + x], b[y * width + x]);
        return pair_t<size_t, size_t>{x, y};
      }
    }
  }
  return pair_t<size_t, size_t>{width, height};
}
  
struct shared_t {
  device_t device_ptr;
  host_t *host_ptr;

  size_t count;
  size_t bytes;

  shared_t(size_t element_count, bool cuda): 
  host_ptr{(host_t*) malloc(element_count * t_size)},
  device_ptr{nullptr},
  count{element_count},
  bytes{element_count * t_size} {
    if (cuda) {
      cudaMalloc(&device_ptr, bytes); 
    }
    printf("[shared_t] host:%p device:%p count:%lu bytes:%lu\n",
           host_ptr, device_ptr, element_count, bytes);
  }

  ~shared_t() {
    printf("[shared_t] dispose %p %p\n", host_ptr, device_ptr);
    if (host_ptr != nullptr) 
      free(host_ptr);
    if (device_ptr != nullptr)
      cudaFree(device_ptr);
  }

  int sync(cudaMemcpyKind kind) {
    if (host_ptr == nullptr) {
      printf("[shared_t] host_ptr is nullptr (host:%p, device:%p)\n", host_ptr, device_ptr);
      return 1;
    }

    if (device_ptr == nullptr) {
      printf("[shared_t] device_ptr is nullptr (host:%p, device:%p)\n", host_ptr, device_ptr);
      return 2;
    }
    if (kind == cudaMemcpyDeviceToHost)
      return cudaMemcpy(host_ptr, device_ptr, bytes, kind); 
    else 
      return cudaMemcpy(device_ptr, host_ptr, bytes, kind);
  }

  int upload() {
    auto result = sync(cudaMemcpyHostToDevice);
    if (result == cudaSuccess)
      return 0;

    printf("[shared_t] upload of (%p, %p, %lu) failed %d\n", 
           device_ptr, host_ptr, bytes, result);
    return result;
  }

  int download() {
    auto result = sync(cudaMemcpyDeviceToHost);
    if (result == cudaSuccess)
      return 0;

    printf("[shared_t] download of (%p, %p, %lu) failed %d\n", 
           device_ptr, host_ptr, bytes, result);
    return result;
  }

  int randomize() {
    if (device_ptr == nullptr) {
      printf("[shared_t] device_ptr is nullptr for randomize (host:%p, device:%p)\n",
             host_ptr, device_ptr);
      return 1;
    }

    curandGenerator_t generator;
    auto r_gen_create = curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    if (r_gen_create != CURAND_STATUS_SUCCESS) {
      printf("[shared_t] create generator failed %d\n", r_gen_create);
      return r_gen_create;
    }

    auto r_gen = curandGenerate(generator, (unsigned int*) device_ptr, count); 
    if (r_gen != CURAND_STATUS_SUCCESS) {
      printf("[shared_t] generate failed %d\n", r_gen);
    }

    auto r_destroy = curandDestroyGenerator(generator);
    if (r_destroy != CURAND_STATUS_SUCCESS) {
      printf("[shared_t] destroy generator failed %d\n", r_destroy);
      return r_destroy;
    }

    return 0;
  }

  device_t device() const {
    if (device_ptr == nullptr)
      printf("[shared_t] device() has to return nullptr (host:%p, device:%p)\n", 
             host_ptr, device_ptr);
    return device_ptr;
  }

  host_t* host() const {
    if (host_ptr == nullptr)
      printf("[shared_t] host() has to return nullptr (host:%p, device:%p)\n",
             host_ptr, device_ptr);
    return host_ptr;
  }
};

struct measure_t {
  bool use_cuda;
  time_point_t tp_start;
  time_point_t tp_end;

  cudaEvent_t e_start;
  cudaEvent_t e_end;

  float mcs_value;
  float cuda_mcs_value;

  measure_t(bool cuda): 
  use_cuda{cuda} { 
    if (use_cuda) {
      auto r_start = cudaEventCreate(&e_start);
      auto r_end = cudaEventCreate(&e_end);
      printf("[measure_t] create cuda events start:%d end:%d\n", r_start, r_end);
      if (r_start != cudaSuccess || r_end != cudaSuccess) {
        if (r_start != cudaSuccess && r_end == cudaSuccess)
          cudaEventDestroy(e_end);
        if (r_start == cudaSuccess && r_end != cudaSuccess)
          cudaEventDestroy(e_start);
        printf("[measure_t] disabling cuda events\n");
        use_cuda = false;
      }
    }
  }
  ~measure_t() {
    if (use_cuda) {
        auto r_end = cudaEventDestroy(e_end);
        auto r_start = cudaEventDestroy(e_start);
        printf("[measure_t] dispose cuda events start:%d end:%d\n", r_start, r_end);
    }
  }

  void start() {
    tp_start = std::chrono::high_resolution_clock::now();
    if (use_cuda) 
      cudaEventRecord(e_start, 0);
  }

  void end() {
    tp_end = std::chrono::high_resolution_clock::now();
    mcs_value = std::chrono::duration_cast<std::chrono::microseconds>(tp_end - tp_start).count() * 1.0; 
    if (use_cuda) {
      auto r_record = cudaEventRecord(e_end, 0);
      auto r_sync = cudaEventSynchronize(e_end);

      float elapsed = 0;
      auto r_time = cudaEventElapsedTime(&elapsed, e_start, e_end);

      if (r_record != cudaSuccess || r_sync != cudaSuccess || r_time != cudaSuccess) {
        printf("[measure_t] cuda stop event is failed record:%d sync:%d time:%d\n", 
               r_record, r_sync, r_time);
        return;
      }
     
      cuda_mcs_value = elapsed * 1000.0;
    }
  }

  float mcs() const { return mcs_value; }
  float cuda_mcs() const { return cuda_mcs_value; }
  float ms() const { return mcs_value / 1000.0; }
  float cuda_ms() { return cuda_mcs_value / 1000.0; }
};

__global__ void transformKernel(const device_t input, device_t output, size_t width, size_t threads) {
  const size_t i = blockIdx.x * threads + threadIdx.x; 

  /* *block* vector index is equal to linear index */
  /* but *block* index is 4 linear indecices */
  const size_t y = i * 4 / width;

  /* every *block* index is 4 linear indices */
  const size_t x = 4 * (i % (width / 4));
  
  /* every input vector is 2 output vectors */
  const size_t oy = y * 2;

  /* every *block* index is 2 linear indices */
  const size_t ox = i * 2 % (width / 2);

  /* aligned view into *block* */
  output[(1 + oy) * width / 2 + ox + 0] = input[y * width + x + 0];
  output[(1 + oy) * width / 2 + ox + 1] = input[y * width + x + 1];
  output[(0 + oy) * width / 2 + ox + 0] = input[y * width + x + 2];
  output[(0 + oy) * width / 2 + ox + 1] = input[y * width + x + 3];
}

__host__ void transformCpu(const host_t* input, host_t* output, size_t width, size_t height) {
  const size_t iter = width * height / 4;
  for (size_t i = 0; i < iter; i++) {
    const size_t y = i * 4 / width;
    const size_t x = 4 * (i % (width / 4));
    const size_t oy = y * 2;
    const size_t ox = i * 2 % (width / 2);
    output[(1 + oy) * width / 2 + ox + 0] = input[y * width + x + 0];
    output[(1 + oy) * width / 2 + ox + 1] = input[y * width + x + 1];
    output[(0 + oy) * width / 2 + ox + 0] = input[y * width + x + 2];
    output[(0 + oy) * width / 2 + ox + 1] = input[y * width + x + 3];
  }
}

int main() {
  FILE* matrix_file =fopen("/dev/null", "w"); 
  const size_t height = 1024;
  const size_t width = 1024;
  const size_t count = height * width;
  
  const size_t o_width = width / 2;
  const size_t o_height = height * 2;

  shared_t input{count, true};
  shared_t output{count, true};
  shared_t check{count, false};
  measure_t measure{true};
  measure_t cpu_measure{false};

  /* generate random numbers in device memory */
  if (input.randomize() != 0)
    return 4;

  /* copy it to host memory */
  if (input.download() != 0)
    return 3;

  /* fIndex to fill item with it's index */
  /* fRamdom to fill item with random value */
  fillMatrix(output.host(), width, height, fZero);
  fillMatrix(check.host(), width, height, fZero);

  /* copy data into the device */
  if (input.upload() != 0)
    return 1;

  /* output copy isn't necessary */
  if (output.upload() != 0)
    return 1;

  /* calculate kernel start params */
  const size_t iter_count = count / 4;  
  const size_t thread_count = 256;

  /* execute kernel on the device */
  measure.start();
  transformKernel<<<iter_count / thread_count, thread_count>>>(
    input.device(),
    output.device(),
    width,
    thread_count
  );
  measure.end();

  cpu_measure.start();
  transformCpu(input.host(), check.host(), width, height); 
  cpu_measure.end();

  /* copy data from the device */
  if (output.download() != 0)
    return 2;

  /* input copy isn't necessary */
  if (input.download() != 0)
    return 2;

  fprintf(matrix_file, "input: \n");
  printMatrix(matrix_file, input.host(), width, height);
  fprintf(matrix_file, "output: \n");
  printMatrix(matrix_file, output.host(), o_width, o_height);
  fprintf(matrix_file, "check: \n");
  printMatrix(matrix_file, check.host(), o_width, o_height);
  
  printf("measure cuda: %f mcs\n", measure.cuda_mcs());
  printf("measure cpu: %f mcs\n", cpu_measure.mcs());

  auto result = compare(output.host(), check.host(), o_width, o_height);
  if (result.first < o_width || result.second < o_height) {
    printf("compare missmatch on %lu %lu\n", result.first, result.second);
  } else {
    printf("compare check passed!\n");
  }

  /* all shared_t objects is allocated on the stack */
  /* so, destructor will automatically free all memory */
  return 0;
}
