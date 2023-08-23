#include <cmath>
#include <cstring>

#include "model.h"
#include "util.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

extern int N;

// class BrainTumorModel(nn.Module):
//
//  def __init__(self):
//      super().__init__()
//      self.conv0 = nn.Sequential(
//          nn.Conv2d(1,128,kernel_size=3),
//          nn.InstanceNorm2d(128),
//          nn.MaxPool2d(2,2),
//          nn.ReLU()
//      )
//
//      self.conv1 = nn.Sequential(
//          nn.Conv2d(128,256,kernel_size=3),
//          nn.InstanceNorm2d(256),
//          nn.MaxPool2d(2,2),
//          nn.ReLU()
//      )
//
//      self.linear1 = nn.Linear(62,128)
//      self.linear2 = nn.Linear(128,64)
//      self.flat = nn.Flatten(1)
//      self.linear3 = nn.Linear(1015808,2)
//
//  def forward(self,x):
//      x = self.conv0(x)
//      x = self.conv1(x)
//      x = F.relu(self.linear1(x))
//      x = self.linear2(x)
//      x = self.flat(x)
//      x = self.linear3(x)
//
//      return x

static Tensor *conv0_weight, *conv0_bias, *conv1_weight, *conv1_bias,
    *linear1_weight, *linear1_bias, *linear2_weight, *linear2_bias,
    *linear3_weight, *linear3_bias, *instanceNorm2d0_weight,
    *instanceNorm2d0_bias, *instanceNorm2d1_weight, *instanceNorm2d1_bias;

static Tensor *input, *output, *c1, *i1, *m1, *c2, *i2, *m2, *l1, *l2;

void initialize_model(const char *parameter_fname) {
  size_t m; // 2345922
  float *buf = (float *)read_binary(parameter_fname, &m);
  conv0_weight = new Tensor(buf, {128, 1, 3, 3});
  buf += 1152;
  conv0_bias = new Tensor(buf, {128});
  buf += 128;
  instanceNorm2d0_weight = new Tensor(buf, {128});
  buf += 128;
  instanceNorm2d0_bias = new Tensor(buf, {128});
  buf += 128;
  conv1_weight = new Tensor(buf, {256, 128, 3, 3});
  buf += 294912;
  conv1_bias = new Tensor(buf, {256});
  buf += 256;
  instanceNorm2d1_weight = new Tensor(buf, {256});
  buf += 256;
  instanceNorm2d1_bias = new Tensor(buf, {256});
  buf += 256;
  linear1_weight = new Tensor(buf, {62, 128});
  buf += 7936;
  linear1_bias = new Tensor(buf, {128});
  buf += 128;
  linear2_weight = new Tensor(buf, {128, 64});
  buf += 8192;
  linear2_bias = new Tensor(buf, {64});
  buf += 64;
  linear3_weight = new Tensor(buf, {1015808, 2});
  buf += 2031616;
  linear3_bias = new Tensor(buf, {2});
  buf += 2;

  input = new Tensor({N, 1, 256, 256});
  output = new Tensor({N, 2});
  c1 = new Tensor({N, 128, 254, 254});
  i1 = new Tensor({N, 128, 254, 254});
  m1 = new Tensor({N, 128, 127, 127});
  c2 = new Tensor({N, 256, 125, 125});
  i2 = new Tensor({N, 256, 125, 125});
  m2 = new Tensor({N, 256, 62, 62});
  l1 = new Tensor({N, 256, 62, 128});
  l2 = new Tensor({N, 256, 62, 64});
  CHECK_CUDA(cudaDeviceSynchronize());
}
// Conv2D
// https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
// Size of in  = N * C_IN * H_IN * W_IN
// Size of out = N * C_OUT * (H_IN-K+1) * (W_IN-K+1)
// Weight : C_OUT * C_IN * K * K
// Bias : C_OUT

static void conv2d(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                   Tensor *bias_t);

// MaxPool2d
// https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
// size of in  = N * H_IN * W_IN
// size of out = N * (H / kH) * (W / kW)
static void maxpool2d(Tensor *in_t, Tensor *out_t, int kH, int kW);

// InstanceNorm2D
// https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html
// size of in  = N * C * H * W
// size of out = N * C * H * W
// weight : C
// bias : C
static void instancenorm2d(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                           Tensor *bias_t);

// Linear
// https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
// size of in  = N * H_IN
// size of out = N * H_OUT
// weight : H_OUT * H_IN
// bias : H_OUT
static void linear(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                   Tensor *bias_t);

// ReLU (inplace)
// https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
// size of in & out = N
static void relu(Tensor *inout_t);

void model_forward(float *inputN, float *outputN) {

  CHECK_CUDA(cudaMemcpy(input->gbuf, inputN, N*256*256*sizeof(float), cudaMemcpyHostToDevice));

  int total_threads;
  int block_size = 1024;
  dim3 blockDim(block_size);
  
  //0th conv2d
  total_threads = N * 128 * 254 * 254;
  dim3 gridDim((total_threads + block_size -1)/block_size);
  conv2d_kernel<<<gridDim, blockDim>>>(input->gbuf, c1->gbuf, conv0_weight->gbuf, conv0_bias->gbuf, N, conv0_weight->shape[2], 
                                      conv0_weight->shape[1], conv0_weight->shape[0], conv0_bias->shape[1], conv0_bias->shape[2]); //TODO
  CHECK_CUDA(cudaDeviceSynchronize());

  // 0th InstanceNorm
  total_threads = N * 128 * 254 * 254;
  dim3 gridDim((total_threads + block_size -1)/block_size);
  instancenorm2d_kernel<<<gridDim, blockDim>>>(c1->gbuf, i1->gbuf, instanceNorm2d0_weight->gbuf, instanceNorm2d0_bias->gbuf,
                                              N, c1->shape[0], c1->shape[1], c1->shape[2]); //TODO
  CHECK_CUDA(cudaDeviceSynchronize());

  //0th maxpool2d
  total_threads = N * 128 * 127 * 127;
  dim3 gridDim((total_threads + block_size -1)/block_size);
  maxpool2d_kernel<<<gridDim, blockDim>>>(i1->gbuf, m1->gbuf, N, i1->shape[1], i1->shape[2]);
  CHECK_CUDA(cudaDeviceSynchronize());

  //0th relu
  total_threads = N * 128 * 127 * 127;
  dim3 gridDim((total_threads + block_size -1)/block_size);
  relu_kernel<<<gridDim, blockDim>>>(m1->gbuf, m1->get_elem());
  CHECK_CUDA(cudaDeviceSynchronize());

  //1th conv2d
  total_threads = N * 128 * 127 * 127;
  dim3 gridDim((total_threads + block_size -1)/block_size);
  conv2d_kernel<<<gridDim, blockDim>>>(m1->gbuf, c2->gbuf, conv1_weight->gbuf, conv1_bias->gbuf, N, conv1_weight->shape[2], 
                                      conv1_weight->shape[1], conv1_weight->shape[0], conv1_bias->shape[1], conv1_bias->shape[2]); //TODO
  CHECK_CUDA(cudaDeviceSynchronize());

  //1th InstanceNorm
  total_threads = N * 256 * 125 * 125;
  dim3 gridDim((total_threads + block_size -1)/block_size);
  instancenorm2d_kernel<<<gridDim, blockDim>>>(c2->gbuf, i2->gbuf, instanceNorm2d1_weight->gbuf, instanceNorm2d1_bias->gbuf,
                                              N, c2->shape[0], c2->shape[1], c2->shape[2]); //TODO
  CHECK_CUDA(cudaDeviceSynchronize());

  //1th maxpool2d
  total_threads = N * 256 * 62 * 62;
  dim3 gridDim((total_threads + block_size -1)/block_size);
  maxpool2d_kernel<<<gridDim, blockDim>>>(i2->gbuf, m2->gbuf, N, i2->shape[1], i2->shape[2]);
  CHECK_CUDA(cudaDeviceSynchronize());

  //1th relu
  total_threads = N * 256 * 62 * 62;
  dim3 gridDim((total_threads + block_size -1)/block_size);
  relu_kernel<<<gridDim, blockDim>>>(m2->gbuf, m2->get_elem());
  CHECK_CUDA(cudaDeviceSynchronize());

  //0th linear
  total_threads = N * 256 * 62 * 128;
  dim3 gridDim((total_threads + block_size -1)/block_size);
  linear_kernel<<<gridDim, blockDim>>>(m2->gbuf, l1->gbuf, linear1_weight->gbuf, linear1_bias->gbuf, N, 256, 62, 62, 128);
  CHECK_CUDA(cudaDeviceSynchronize());

  //linear-relu
  total_threads = N * 256 * 62 * 128;
  dim3 gridDim((total_threads + block_size -1)/block_size);
  relu_kernel<<<gridDim, blockDim>>>(l1->gbuf, l1->get_elem());
  CHECK_CUDA(cudaDeviceSynchronize());

  //1st linear
  total_threads = N * 256 * 62 * 64;
  dim3 gridDim((total_threads + block_size -1)/block_size);
  linear_kernel<<<gridDim, blockDim>>>(l1->gbuf, l2->gbuf, linear2_weight->gbuf, linear2_bias->gbuf, N, 256, 62, 128, 64);
  CHECK_CUDA(cudaDeviceSynchronize());
  
  //linear-reshape
  l2->reshape({N, 1, 1015808});

  //2nd linear
  total_threads = N * 2;
  dim3 gridDim((total_threads + block_size -1)/block_size);
  linear_kernel<<<gridDim, blockDim>>>(l2->gbuf, output->gbuf, linear3_weight->gbuf, linear3_bias->gbuf, N, 1, 1, 1015808, 2);
  CHECK_CUDA(cudaDeviceSynchronize());

  for (int idx = 0; idx < N; idx++) {
    memcpy(input->buf, inputN + 256 * 256 * idx, 256 * 256 * sizeof(float));

    conv2d(input, c1, conv0_weight, conv0_bias);
    instancenorm2d(c1, i1, instanceNorm2d0_weight, instanceNorm2d0_bias);
    maxpool2d(i1, m1, 2, 2);
    relu(m1);
    conv2d(m1, c2, conv1_weight, conv1_bias);
    instancenorm2d(c2, i2, instanceNorm2d1_weight, instanceNorm2d1_bias);
    maxpool2d(i2, m2, 2, 2);
    relu(m2);
    linear(m2, l1, linear1_weight, linear1_bias);
    relu(l1);
    linear(l1, l2, linear2_weight, linear2_bias);
    l2->reshape({1, 1015808});
    linear(l2, output, linear3_weight, linear3_bias);

    memcpy(outputN + 2 * idx, output->buf, 2 * sizeof(float));
  }
}

__global__ void conv2d_kernel(float *in_buf, float *out_buf, float *weight_buf, float *bias_buf,
                              int N, int K, int C_IN, int C_OUT, int H_IN, int W_OUT) {
  

}

static void conv2d(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                   Tensor *bias_t) {
  float *in = in_t->buf;
  float *out = out_t->buf;
  float *weight = weight_t->buf;
  float *bias = bias_t->buf;

  int K = weight_t->shape[2]; //=weight_t->shape[3];

  int C_IN = weight_t->shape[1];  //=in_t->shape[0];
  int C_OUT = weight_t->shape[0]; //=out_t->shape[0];

  int H_IN = in_t->shape[1];
  int W_IN = in_t->shape[2];
  int H_OUT = H_IN - K + 1; //=out_t->shape[1];
  int W_OUT = W_IN - K + 1; //=out_t->shape[2];

  for (int c_out = 0; c_out < C_OUT; c_out++) {
    for (int h_out = 0; h_out < H_OUT; h_out++) {
      for (int w_out = 0; w_out < W_OUT; w_out++) {
        out[c_out * H_OUT * W_OUT + h_out * W_OUT + w_out] = bias[c_out];
        for (int c_in = 0; c_in < C_IN; c_in++) {
          for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
              out[c_out * H_OUT * W_OUT + h_out * W_OUT + w_out] +=
                  in[c_in * H_IN * W_IN + (h_out + kh) * W_IN + (w_out + kw)] *
                  weight[c_out * C_IN * K * K + c_in * K * K + kh * K + kw];
            }
          }
        }
      }
    }
  }
}

__global__ void instancenorm2d_kernel(float *in_buf, float *out_buf, float *weight_buf, float *bias_buf,
                                      int N, int C, int H, int W) {
  
}

static void instancenorm2d(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                           Tensor *bias_t) {
  float *in = in_t->buf;
  float *out = out_t->buf;
  float *weight = weight_t->buf;
  float *bias = bias_t->buf;

  int C = in_t->shape[0]; //=out_t->shape[0];
  int H = in_t->shape[1]; //=out_t->shape[1];
  int W = in_t->shape[2]; //=out_t->shape[2];

  for (int c = 0; c < C; c++) {
    float e = 0, v = 0;

    // Caculate mean
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        e += in[c * H * W + h * W + w];
      }
    }
    e /= H * W;

    // Caculate Variance
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        v += (in[c * H * W + h * W + w] - e) * (in[c * H * W + h * W + w] - e);
      }
    }
    v /= H * W;

    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        out[c * H * W + h * W + w] =
            (in[c * H * W + h * W + w] - e) / sqrt(v + 1e-5) * weight[c] +
            bias[c];
      }
    }
  }
}

__global__ void linear_kernel(float *in_buf, float *out_buf, float *weight_buf, float *bias_buf,
                              int N, int C_IN, int W_IN, int H_IN, int H_OUT) {
  
}

static void linear(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                   Tensor *bias_t) {
  float *in = in_t->buf;
  float *out = out_t->buf;
  float *weight = weight_t->buf;
  float *bias = bias_t->buf;

  int H_IN = weight_t->shape[0];  // in_t의 마지막 차원
  int H_OUT = weight_t->shape[1]; // out_t의 마지막 차원

  int N = in_t->get_elem() / H_IN; //=out_t->get_elem()/H_OUT

  for (int n = 0; n < N; n++) {
    for (int h_out = 0; h_out < H_OUT; h_out++) {
      out[n * H_OUT + h_out] = bias[h_out];
      for (int h_in = 0; h_in < H_IN; h_in++) {
        out[n * H_OUT + h_out] +=
            in[n * H_IN + h_in] * weight[h_out * H_IN + h_in];
      }
    }
  }
}

__global__ void maxpool2d_kernel(float *in_buf, float *out_buf, int N, int H_IN, int W_IN) {

}

static void maxpool2d(Tensor *in_t, Tensor *out_t, int kH, int kW) {
  float *in = in_t->buf;
  float *out = out_t->buf;

  int H_IN = in_t->shape[1];
  int W_IN = in_t->shape[2];
  int H_OUT = H_IN / kH; // =out_t->shape[1];
  int W_OUT = W_IN / kW; // =out_t->shape[2];

  int N = in_t->shape[0];

  for (int n = 0; n < N; n++) {
    for (int h_out = 0; h_out < H_OUT; h_out++) {
      for (int w_out = 0; w_out < W_OUT; w_out++) {
        out[n * H_OUT * W_OUT + h_out * W_OUT + w_out] =
            in[n * H_IN * W_IN + (h_out * kH) * H_IN + (w_out * kW)];
        for (int kh = 0; kh < kH; kh++)
          for (int kw = 0; kw < kW; kw++)
            out[n * H_OUT * W_OUT + h_out * W_OUT + w_out] =
                fmaxf(out[n * H_OUT * W_OUT + h_out * W_OUT + w_out],
                      in[n * H_IN * W_IN + (h_out * kH + kh) * H_IN +
                         (w_out * kW + kw)]);
      }
    }
  }
}

__global__ void relu_kernel(float *inout_buf, int total) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(tidx >= total) return;
  inout_buf[tidx] = (inout_buf[tidx] > 0) ? inout_buf[tidx] : 0;
}

static void relu(Tensor *inout_t) {
  float *inout = inout_t->buf;
  int N = inout_t->get_elem();
  for (int n = 0; n < N; n++) {
    inout[n] = fmaxf(inout[n], 0);
  }
}

void finalize_model() {
  delete (conv0_weight);
  delete (conv0_bias);
  delete (conv1_weight);
  delete (conv1_bias);
  delete (linear1_weight);
  delete (linear1_bias);
  delete (linear2_weight);
  delete (linear2_bias);
  delete (linear3_weight);
  delete (linear3_bias);
  delete (instanceNorm2d0_weight);
  delete (instanceNorm2d0_bias);
  delete (instanceNorm2d1_weight);
  delete (instanceNorm2d1_bias);
  delete (input);
  delete (output);
  delete (c1);
  delete (i1);
  delete (m1);
  delete (c2);
  delete (i2);
  delete (m2);
  delete (l1);
  delete (l2);
}
