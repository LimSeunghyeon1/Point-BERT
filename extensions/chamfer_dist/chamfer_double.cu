/*
 * @Author: Haozhe Xie
 * @Date:   2019-08-07 20:54:24
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2020-06-17 14:58:55
 * @Email:  cshzxie@gmail.com
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

__global__ void chamfer_dist_kernel(int batch_size,
                                    int n,
                                    const double* xyz1,
                                    int m,
                                    const double* xyz2,
                                    double* dist,
                                    int* indexes) {
  const int batch = 512;
  __shared__ double buf[batch * 3];
  // 블록 인덱스는 batch 하나하나를 의미하고 하나의 그리드에 여러 블록이 있어서 여러개의 배치가 한번에 처리된다.
  // gridDim은 하나의 그리드가 얼마나 많은 배치를 담고 있는가를 의미
  for (int i = blockIdx.x; i < batch_size; i += gridDim.x) {
    for (int k2 = 0; k2 < m; k2 += batch) {
      int end_k = min(m, k2 + batch) - k2;
      // 버퍼는 xyz2에서 i번째 배치에서 k2번째 point부터 k2+batch번째까지의 포인트를 저장한다.
      // 스레드는 0번부터 batch개 까지의 point의 x, y, z 하나하나를 의미한다.
      for (int j = threadIdx.x; j < end_k * 3; j += blockDim.x) {
        buf[j] = xyz2[(i * m + k2) * 3 + j];
      }
      __syncthreads();
      // blockdimx는 스레드의 개수 == 한번에 처리하는 point의 개수, gridDim.y만큼
      /*
      thread 하나 하나는 shared 메모리에 있는 end
      */ 
      for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n;
           j += blockDim.x * gridDim.y) {
        double x1            = xyz1[(i * n + j) * 3 + 0];
        double y1            = xyz1[(i * n + j) * 3 + 1];
        double z1            = xyz1[(i * n + j) * 3 + 2];
        double best_dist     = 0;
        int best_dist_index = 0;
        int end_ka          = end_k - (end_k & 3);
        if (end_ka == batch) {
          for (int k = 0; k < batch; k += 4) {
            {
              double x2   = buf[k * 3 + 0] - x1;
              double y2   = buf[k * 3 + 1] - y1;
              double z2   = buf[k * 3 + 2] - z1;
              double dist = x2 * x2 + y2 * y2 + z2 * z2;

              if (k == 0 || dist < best_dist) {
                best_dist       = dist;
                best_dist_index = k + k2;
              }
            }
            {
              double x2   = buf[k * 3 + 3] - x1;
              double y2   = buf[k * 3 + 4] - y1;
              double z2   = buf[k * 3 + 5] - z1;
              double dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (dist < best_dist) {
                best_dist       = dist;
                best_dist_index = k + k2 + 1;
              }
            }
            {
              double x2   = buf[k * 3 + 6] - x1;
              double y2   = buf[k * 3 + 7] - y1;
              double z2   = buf[k * 3 + 8] - z1;
              double dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (dist < best_dist) {
                best_dist       = dist;
                best_dist_index = k + k2 + 2;
              }
            }
            {
              double x2   = buf[k * 3 + 9] - x1;
              double y2   = buf[k * 3 + 10] - y1;
              double z2   = buf[k * 3 + 11] - z1;
              double dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (dist < best_dist) {
                best_dist       = dist;
                best_dist_index = k + k2 + 3;
              }
            }
          }
        } else {
          for (int k = 0; k < end_ka; k += 4) {
            {
              double x2   = buf[k * 3 + 0] - x1;
              double y2   = buf[k * 3 + 1] - y1;
              double z2   = buf[k * 3 + 2] - z1;
              double dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (k == 0 || dist < best_dist) {
                best_dist       = dist;
                best_dist_index = k + k2;
              }
            }
            {
              double x2   = buf[k * 3 + 3] - x1;
              double y2   = buf[k * 3 + 4] - y1;
              double z2   = buf[k * 3 + 5] - z1;
              double dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (dist < best_dist) {
                best_dist       = dist;
                best_dist_index = k + k2 + 1;
              }
            }
            {
              double x2   = buf[k * 3 + 6] - x1;
              double y2   = buf[k * 3 + 7] - y1;
              double z2   = buf[k * 3 + 8] - z1;
              double dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (dist < best_dist) {
                best_dist       = dist;
                best_dist_index = k + k2 + 2;
              }
            }
            {
              double x2   = buf[k * 3 + 9] - x1;
              double y2   = buf[k * 3 + 10] - y1;
              double z2   = buf[k * 3 + 11] - z1;
              double dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (dist < best_dist) {
                best_dist       = dist;
                best_dist_index = k + k2 + 3;
              }
            }
          }
        }
        for (int k = end_ka; k < end_k; k++) {
          double x2   = buf[k * 3 + 0] - x1;
          double y2   = buf[k * 3 + 1] - y1;
          double z2   = buf[k * 3 + 2] - z1;
          double dist = x2 * x2 + y2 * y2 + z2 * z2;
          if (k == 0 || dist < best_dist) {
            best_dist       = dist;
            best_dist_index = k + k2;
          }
        }
        if (k2 == 0 || dist[(i * n + j)] > best_dist) {
          dist[(i * n + j)]    = best_dist;
          indexes[(i * n + j)] = best_dist_index;
        }
      }
      __syncthreads();
    }
  }
}

__global__ void chamfer_dist_kernel_matrix(int batch_size,
                                    int num_label,
                                    int num_label2,
                                    int num_points,
                                    int num_points2,
                                    const double* xyz1_matrix,
                                    const double* xyz2_matrix,
                                    double* dist_matrix,
                                    int* indexes_matrix) {
  const int batch = 512;
  __shared__ double buf[batch * 3];

  // Calculate the global index for batch and labels
  int label1_idx = blockIdx.y;
  int label2_idx = blockIdx.z;

  if (label1_idx >= num_label || label2_idx >= num_label2) return;

  for (int batch_idx = blockIdx.x; batch_idx < batch_size; batch_idx += gridDim.x) {
    
    // Load points from xyz2_matrix into shared memory
    for (int k2 = 0; k2 < num_points2; k2 += batch) {
      int end_k = min(num_points2, k2 + batch) - k2;
      
      for (int j = threadIdx.x; j < end_k * 3; j += blockDim.x) {
        buf[j] = xyz2_matrix[(batch_idx * num_label2 * num_points2 * 3) + (label2_idx * num_points2 * 3) + (k2 * 3) + j];
      }
      
      __syncthreads();
      
      // Compute Chamfer distance
      for (int j = threadIdx.x; j < num_points; j += blockDim.x) {
        double x1 = xyz1_matrix[(batch_idx * num_label * num_points * 3) + (label1_idx * num_points * 3) + (j * 3) + 0];
        double y1 = xyz1_matrix[(batch_idx * num_label * num_points * 3) + (label1_idx * num_points * 3) + (j * 3) + 1];
        double z1 = xyz1_matrix[(batch_idx * num_label * num_points * 3) + (label1_idx * num_points * 3) + (j * 3) + 2];
        
        double best_dist = 0;
        int best_dist_index = 0;
        
        int end_ka = end_k - (end_k & 3);
        if (end_ka == batch) {
          for (int k = 0; k < batch; k += 4) {
            {
              double x2 = buf[k * 3 + 0] - x1;
              double y2 = buf[k * 3 + 1] - y1;
              double z2 = buf[k * 3 + 2] - z1;
              double dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (k == 0 || dist < best_dist) {
                best_dist = dist;
                best_dist_index = k + k2;
              }
            }
            {
              double x2 = buf[k * 3 + 3] - x1;
              double y2 = buf[k * 3 + 4] - y1;
              double z2 = buf[k * 3 + 5] - z1;
              double dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (dist < best_dist) {
                best_dist = dist;
                best_dist_index = k + k2 + 1;
              }
            }
            {
              double x2 = buf[k * 3 + 6] - x1;
              double y2 = buf[k * 3 + 7] - y1;
              double z2 = buf[k * 3 + 8] - z1;
              double dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (dist < best_dist) {
                best_dist = dist;
                best_dist_index = k + k2 + 2;
              }
            }
            {
              double x2 = buf[k * 3 + 9] - x1;
              double y2 = buf[k * 3 + 10] - y1;
              double z2 = buf[k * 3 + 11] - z1;
              double dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (dist < best_dist) {
                best_dist = dist;
                best_dist_index = k + k2 + 3;
              }
            }
          }
        } else {
          for (int k = 0; k < end_ka; k += 4) {
            {
              double x2 = buf[k * 3 + 0] - x1;
              double y2 = buf[k * 3 + 1] - y1;
              double z2 = buf[k * 3 + 2] - z1;
              double dist = x2 * x2 + y2 * y2 + z2 * z2;
              
              if (k == 0 ||dist < best_dist) {
                best_dist = dist;
                best_dist_index = k + k2;
              }
            }
            {
              double x2 = buf[k * 3 + 3] - x1;
              double y2 = buf[k * 3 + 4] - y1;
              double z2 = buf[k * 3 + 5] - z1;
              double dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (dist < best_dist) {
                best_dist = dist;
                best_dist_index = k + k2 + 1;
              }
            }
            {
              double x2 = buf[k * 3 + 6] - x1;
              double y2 = buf[k * 3 + 7] - y1;
              double z2 = buf[k * 3 + 8] - z1;
              double dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (dist < best_dist) {
                best_dist = dist;
                best_dist_index = k + k2 + 2;
              }
            }
            {
              double x2 = buf[k * 3 + 9] - x1;
              double y2 = buf[k * 3 + 10] - y1;
              double z2 = buf[k * 3 + 11] - z1;
              double dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (dist < best_dist) {
                best_dist = dist;
                best_dist_index = k + k2 + 3;
              }
            }
          }
        }
      
        for (int k = end_ka; k < end_k; k++) {
          double x2 = buf[k * 3 + 0] - x1;
          double y2 = buf[k * 3 + 1] - y1;
          double z2 = buf[k * 3 + 2] - z1;
          double dist = x2 * x2 + y2 * y2 + z2 * z2;
          if (k == 0 || dist < best_dist) {
            best_dist = dist;
            best_dist_index = k + k2;
          }
        }
        int cur_idx = batch_idx * num_label * num_label2 * num_points + label1_idx * num_label2 * num_points + label2_idx * num_points + j;
        if (k2 == 0 || dist_matrix[cur_idx] > best_dist){
          dist_matrix[cur_idx] = best_dist;
          indexes_matrix[cur_idx] = best_dist_index;
        }
      }
      __syncthreads();
    }
  }
}


std::vector<torch::Tensor> chamfer_cuda_forward(torch::Tensor xyz1,
                                                torch::Tensor xyz2) {
  const int batch_size = xyz1.size(0);
  const int n          = xyz1.size(1);  // num_points point cloud A
  const int m          = xyz2.size(1);  // num_points point cloud B
  torch::Tensor dist1 =
    torch::zeros({batch_size, n}, torch::CUDA(torch::kDouble));
  torch::Tensor dist2 =
    torch::zeros({batch_size, m}, torch::CUDA(torch::kDouble));
  torch::Tensor idx1 = torch::zeros({batch_size, n}, torch::CUDA(torch::kInt));
  torch::Tensor idx2 = torch::zeros({batch_size, m}, torch::CUDA(torch::kInt));

  chamfer_dist_kernel<<<dim3(32, 16, 1), 512>>>(
    batch_size, n, xyz1.data_ptr<double>(), m, xyz2.data_ptr<double>(),
    dist1.data_ptr<double>(), idx1.data_ptr<int>());
  chamfer_dist_kernel<<<dim3(32, 16, 1), 512>>>(
    batch_size, m, xyz2.data_ptr<double>(), n, xyz1.data_ptr<double>(),
    dist2.data_ptr<double>(), idx2.data_ptr<int>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in chamfer_cuda_forward: %s\n", cudaGetErrorString(err));
  }
  return {dist1, dist2, idx1, idx2};
}

std::vector<torch::Tensor> chamfer_cuda_forward_matrix(torch::Tensor xyz1,
                                                torch::Tensor xyz2) {

  
  const int batch_size = xyz1.size(0);
  const int num_label = xyz1.size(1);
  const int num_label2 = xyz2.size(1);
  const int num_points = xyz1.size(2);
  const int num_points2 = xyz2.size(2);
  

  torch::Tensor dist1 = torch::zeros({batch_size, num_label, num_label2, num_points}, torch::CUDA(torch::kDouble));
  torch::Tensor dist2 = torch::zeros({batch_size, num_label2, num_label, num_points2}, torch::CUDA(torch::kDouble));
  torch::Tensor idx1 = torch::zeros({batch_size, num_label, num_label2, num_points}, torch::CUDA(torch::kInt));
  torch::Tensor idx2 = torch::zeros({batch_size, num_label2, num_label, num_points2}, torch::CUDA(torch::kInt));

  int threads_per_block = 512;
  // int blocks_per_batch = max(32, (num_points + threads_per_block - 1) / threads_per_block);
  // int blocks_per_batch2 = max(32, (num_points2 + threads_per_block - 1) / threads_per_block);
  int blocks_per_batch = 512;
  int blocks_per_batch2 = 512;
  dim3 grid(blocks_per_batch, num_label, num_label2);
  dim3 grid2(blocks_per_batch2, num_label2, num_label);
  dim3 block(threads_per_block);
  /*
  (int batch_size,
    int num_label,
    int num_points,
    const double* xyz1_matrix,
    const double* xyz2_matrix,
    double* dist_matrix,
    int* indexes_matrix
  */

  chamfer_dist_kernel_matrix<<<grid, block>>>(
    batch_size, num_label, num_label2, num_points, num_points2,
    xyz1.data_ptr<double>(), xyz2.data_ptr<double>(),
    dist1.data_ptr<double>(), idx1.data_ptr<int>());

  chamfer_dist_kernel_matrix<<<grid2, block>>>(
    batch_size, num_label2, num_label, num_points2, num_points,
    xyz2.data_ptr<double>(), xyz1.data_ptr<double>(),
    dist2.data_ptr<double>(), idx2.data_ptr<int>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in chamfer_cuda_forward: %s\n", cudaGetErrorString(err));
  }

  return {dist1, dist2, idx1, idx2};
}



__global__ void chamfer_dist_grad_kernel(int b,
                                         int n,
                                         const double* xyz1,
                                         int m,
                                         const double* xyz2,
                                         const double* grad_dist1,
                                         const int* idx1,
                                         double* grad_xyz1,
                                         double* grad_xyz2) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n;
         j += blockDim.x * gridDim.y) {
      double x1 = xyz1[(i * n + j) * 3 + 0];
      double y1 = xyz1[(i * n + j) * 3 + 1];
      double z1 = xyz1[(i * n + j) * 3 + 2];
      int j2   = idx1[i * n + j];
      double x2 = xyz2[(i * m + j2) * 3 + 0];
      double y2 = xyz2[(i * m + j2) * 3 + 1];
      double z2 = xyz2[(i * m + j2) * 3 + 2];
      double g  = grad_dist1[i * n + j] * 2;
      atomicAdd(&(grad_xyz1[(i * n + j) * 3 + 0]), g * (x1 - x2));
      atomicAdd(&(grad_xyz1[(i * n + j) * 3 + 1]), g * (y1 - y2));
      atomicAdd(&(grad_xyz1[(i * n + j) * 3 + 2]), g * (z1 - z2));
      atomicAdd(&(grad_xyz2[(i * m + j2) * 3 + 0]), -(g * (x1 - x2)));
      atomicAdd(&(grad_xyz2[(i * m + j2) * 3 + 1]), -(g * (y1 - y2)));
      atomicAdd(&(grad_xyz2[(i * m + j2) * 3 + 2]), -(g * (z1 - z2)));
    }
  }
}

__global__ void chamfer_dist_grad_kernel_matrix(int batch_size,
                                                  int num_label,
                                                  int num_label2,
                                                  int num_points,
                                                  int num_points2,
                                                  const double* xyz1,
                                                  const double* xyz2,
                                                  const double* grad_dist1,
                                                  const int* idx1,
                                                  double* grad_xyz1,
                                                  double* grad_xyz2) {
  for (int b = blockIdx.x; b < batch_size; b += gridDim.x) {
    for (int l1 = blockIdx.y; l1 < num_label; l1 += gridDim.y) {
      for (int l2 = blockIdx.z; l2 < num_label2; l2 += gridDim.z){
        for (int j = threadIdx.x ; j < num_points; j += blockDim.x) {
          double x1 = xyz1[((b * num_label + l1) * num_points + j) * 3 + 0];
          double y1 = xyz1[((b * num_label + l1) * num_points + j) * 3 + 1];
          double z1 = xyz1[((b * num_label + l1) * num_points + j) * 3 + 2];
          int j2 = idx1[((b * num_label * num_label2 + l1 * num_label2 + l2) * num_points + j)];

          double x2 = xyz2[((b * num_label2 + l2) * num_points2 + j2) * 3 + 0];
          double y2 = xyz2[((b * num_label2 + l2) * num_points2 + j2) * 3 + 1];
          double z2 = xyz2[((b * num_label2 + l2) * num_points2 + j2) * 3 + 2];
          double g = grad_dist1[((b * num_label * num_label2 + l1 * num_label2 + l2) * num_points + j)] * 2;

          atomicAdd(&(grad_xyz1[((b * num_label + l1) * num_points + j) * 3 + 0]), g * (x1 - x2));
          atomicAdd(&(grad_xyz1[((b * num_label + l1) * num_points + j) * 3 + 1]), g * (y1 - y2));
          atomicAdd(&(grad_xyz1[((b * num_label + l1) * num_points + j) * 3 + 2]), g * (z1 - z2));
          atomicAdd(&(grad_xyz2[((b * num_label2 + l2) * num_points2 + j2) * 3 + 0]), -g * (x1 - x2));
          atomicAdd(&(grad_xyz2[((b * num_label2 + l2) * num_points2 + j2) * 3 + 1]), -g * (y1 - y2));
          atomicAdd(&(grad_xyz2[((b * num_label2 + l2) * num_points2 + j2) * 3 + 2]), -g * (z1 - z2));
        }

      }
      
    }
  }
}


std::vector<torch::Tensor> chamfer_cuda_backward(torch::Tensor xyz1,
                                                 torch::Tensor xyz2,
                                                 torch::Tensor idx1,
                                                 torch::Tensor idx2,
                                                 torch::Tensor grad_dist1,
                                                 torch::Tensor grad_dist2) {
  const int batch_size    = xyz1.size(0);
  const int n             = xyz1.size(1);  // num_points point cloud A
  const int m             = xyz2.size(1);  // num_points point cloud B
  torch::Tensor grad_xyz1 = torch::zeros_like(xyz1, torch::CUDA(torch::kDouble));
  torch::Tensor grad_xyz2 = torch::zeros_like(xyz2, torch::CUDA(torch::kDouble));

  chamfer_dist_grad_kernel<<<dim3(1, 16, 1), 256>>>(
    batch_size, n, xyz1.data_ptr<double>(), m, xyz2.data_ptr<double>(),
    grad_dist1.data_ptr<double>(), idx1.data_ptr<int>(),
    grad_xyz1.data_ptr<double>(), grad_xyz2.data_ptr<double>());
  chamfer_dist_grad_kernel<<<dim3(1, 16, 1), 256>>>(
    batch_size, m, xyz2.data_ptr<double>(), n, xyz1.data_ptr<double>(),
    grad_dist2.data_ptr<double>(), idx2.data_ptr<int>(),
    grad_xyz2.data_ptr<double>(), grad_xyz1.data_ptr<double>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in chamfer_cuda_backward: %s\n", cudaGetErrorString(err));
  }
  return {grad_xyz1, grad_xyz2};
}

std::vector<torch::Tensor> chamfer_cuda_backward_matrix(torch::Tensor xyz1,
                                                 torch::Tensor xyz2,
                                                 torch::Tensor idx1,
                                                 torch::Tensor idx2,
                                                 torch::Tensor grad_dist1,
                                                 torch::Tensor grad_dist2) {
  const int batch_size = xyz1.size(0);
  const int num_label = xyz1.size(1);
  const int num_label2 = xyz2.size(1);

  const int num_points = xyz1.size(2);
  const int num_points2 = xyz2.size(2);

  int threads_per_block = 256;
  int blocks_per_batch = 512;
  int blocks_per_batch2 = 512;
  dim3 grid(blocks_per_batch, num_label, num_label2);
  
  dim3 grid2(blocks_per_batch2, num_label2, num_label);
  
  dim3 block(threads_per_block);


  torch::Tensor grad_xyz1 = torch::zeros_like(xyz1, torch::CUDA(torch::kDouble));
  torch::Tensor grad_xyz2 = torch::zeros_like(xyz2, torch::CUDA(torch::kDouble));

  chamfer_dist_grad_kernel_matrix<<<grid, block>>>(
    batch_size, num_label, num_label2, num_points, num_points2,
    xyz1.data_ptr<double>(), xyz2.data_ptr<double>(),
    grad_dist1.data_ptr<double>(), idx1.data_ptr<int>(),
    grad_xyz1.data_ptr<double>(), grad_xyz2.data_ptr<double>());
  
  chamfer_dist_grad_kernel_matrix<<<grid2, block>>>(
    batch_size, num_label2, num_label, num_points2, num_points,
    xyz2.data_ptr<double>(), xyz1.data_ptr<double>(),
    grad_dist2.data_ptr<double>(), idx2.data_ptr<int>(),
    grad_xyz2.data_ptr<double>(), grad_xyz1.data_ptr<double>());


  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in chamfer_cuda_forward: %s\n", cudaGetErrorString(err));
  }
  
  return {grad_xyz1, grad_xyz2};
}