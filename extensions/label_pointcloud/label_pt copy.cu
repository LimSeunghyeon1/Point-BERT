#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

// Kernel for float precision
__global__ void label_pointcloud_kernel(const float* data, const int* label, float* output, int* input_index, int* count, int B, int N, int num_labels) {
    int batch = blockIdx.x;
    int point = threadIdx.x;
    if (batch < B && point < N) {
        int lbl = label[batch * N + point];
        int index = atomicAdd(&count[batch * num_labels + lbl], 1);
        int offset = (batch * num_labels + lbl) * N * 3 + index * 3;
        output[offset] = data[batch * N * 3 + point * 3];
        output[offset + 1] = data[batch * N * 3 + point * 3 + 1];
        output[offset + 2] = data[batch * N * 3 + point * 3 + 2];
        input_index[batch * N + point] = (batch * num_labels + lbl) * N + index;
    }
}

// Kernel for double precision
__global__ void label_pointcloud_kernel_double(const double* data, const int* label, double* output, int* input_index, int* count, int B, int N, int num_labels) {
    int batch = blockIdx.x;
    int point = threadIdx.x;
    if (batch < B && point < N) {
        int lbl = label[batch * N + point];
        int index = atomicAdd(&count[batch * num_labels + lbl], 1);
        int offset = (batch * num_labels + lbl) * N * 3 + index * 3;
        output[offset] = data[batch * N * 3 + point * 3];
        output[offset + 1] = data[batch * N * 3 + point * 3 + 1];
        output[offset + 2] = data[batch * N * 3 + point * 3 + 2];
        input_index[batch * N + point] = (batch * num_labels + lbl) * N + index;
    }
}

// Kernel for float precision backward
__global__ void label_pointcloud_backward_kernel(const float* grad_output, const int* label, float* grad_data, int* input_index, int B, int N, int num_labels) {
    int batch = blockIdx.x;
    int point = threadIdx.x;
    if (batch < B && point < N) {
        int lbl = label[batch * N + point];
        int offset = input_index[batch * N + point] * 3;
        grad_data[(batch * N + point) * 3] = grad_output[offset];
        grad_data[(batch * N + point) * 3 + 1] = grad_output[offset + 1];
        grad_data[(batch * N + point) * 3 + 2] = grad_output[offset + 2];
    }
}

// Kernel for double precision backward
__global__ void label_pointcloud_backward_kernel_double(const double* grad_output, const int* label, double* grad_data, int* input_index, int B, int N, int num_labels) {
    int batch = blockIdx.x;
    int point = threadIdx.x;
    if (batch < B && point < N) {
        int lbl = label[batch * N + point];
        int offset = input_index[batch * N + point] * 3;
        grad_data[(batch * N + point) * 3] = grad_output[offset];
        grad_data[(batch * N + point) * 3 + 1] = grad_output[offset + 1];
        grad_data[(batch * N + point) * 3 + 2] = grad_output[offset + 2];
    }
}

extern "C" void launch_label_pointcloud_kernel(const float* data, const int* label, float* output, int* input_index, int* count, int B, int N, int num_labels) {
    label_pointcloud_kernel<<<B, N>>>(data, label, output, input_index, count, B, N, num_labels);
}

extern "C" void launch_label_pointcloud_kernel_double(const double* data, const int* label, double* output, int* input_index, int* count, int B, int N, int num_labels) {
    label_pointcloud_kernel_double<<<B, N>>>(data, label, output, input_index, count, B, N, num_labels);
}

extern "C" void launch_label_pointcloud_backward_kernel(const float* grad_output, const int* label, float* grad_data, int* input_index, int B, int N, int num_labels) {
    label_pointcloud_backward_kernel<<<B, N>>>(grad_output, label, grad_data, input_index, B, N, num_labels);
}

extern "C" void launch_label_pointcloud_backward_kernel_double(const double* grad_output, const int* label, double* grad_data, int* input_index, int B, int N, int num_labels) {
    label_pointcloud_backward_kernel_double<<<B, N>>>(grad_output, label, grad_data, input_index, B, N, num_labels);
}
