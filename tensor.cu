#include <cstring>
#include <stdlib.h>

#include "tensor.h"
#include "util.h"
using namespace std;

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

Tensor::Tensor(const vector<int> &shape_) {
  reshape(shape_);
  // buf = (float *)malloc(n * sizeof(float));
  CHECK_CUDA(cudaMalloc(&gbuf, n * sizeof(float)));
}

Tensor::Tensor(float *data, const vector<int> &shape_) {
  reshape(shape_);
  // buf = (float *)malloc(n * sizeof(float));
  CHECK_CUDA(cudaMalloc(&gbuf, n * sizeof(float)));
  
  // memcpy(buf, data, get_elem() * sizeof(float));
  cudaMemcpy(gbuf, data, get_elem() * sizeof(float), cudaMemcpyHostToDevice);
}

Tensor::~Tensor() {
  //free(buf); 
  cudaFree(gbuf);
}

void Tensor::load(const char *filename) {
  size_t m;
  float *tmp;
  tmp = (float *)read_binary(filename, &m);
  CHECK_CUDA(cudaMemcpy(gbuf, tmp, sizeof(tmp), cudaMemcpyHostToDevice));
  n = m;
  reshape({n});
}
void Tensor::save(const char *filename) { 
    float *tmp;
    CHECK_CUDA(cudaMemcpy(tmp, gbuf, get_elem() * sizeof(float), cudaMemcpyDeviceToHost));
    write_binary(tmp, filename, n);
  }

int Tensor::get_elem() { return n; }

void Tensor::reshape(const vector<int> &shape_) {
  n = 1;
  ndim = shape_.size(); // ndim<=4
  for (int i = 0; i < ndim; i++) {
    shape[i] = shape_[i];
    n *= shape[i];
  }
}
