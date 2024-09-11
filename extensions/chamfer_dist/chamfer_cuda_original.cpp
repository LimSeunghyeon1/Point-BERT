/*
 * @Author: Haozhe Xie
 * @Date:   2019-08-07 20:54:24
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2019-12-10 10:33:50
 * @Email:  cshzxie@gmail.com
 */

#include <torch/extension.h>
#include <vector>
#include <iostream>
using namespace std;

// 특정 포인트를 제거하는 함수


std::vector<torch::Tensor> chamfer_cuda_forward(torch::Tensor xyz1,
                                                torch::Tensor xyz2);

std::vector<torch::Tensor> chamfer_cuda_backward(torch::Tensor xyz1,
                                                 torch::Tensor xyz2,
                                                 torch::Tensor idx1,
                                                 torch::Tensor idx2,
                                                 torch::Tensor grad_dist1,
                                                 torch::Tensor grad_dist2);


std::vector<torch::Tensor> chamfer_cuda_forward_double(torch::Tensor xyz1,
                                                torch::Tensor xyz2);

std::vector<torch::Tensor> chamfer_cuda_backward_double(torch::Tensor xyz1,
                                                 torch::Tensor xyz2,
                                                 torch::Tensor idx1,
                                                 torch::Tensor idx2,
                                                 torch::Tensor grad_dist1,
                                                 torch::Tensor grad_dist2);
                                                 



std::vector<torch::Tensor> chamfer_forward(torch::Tensor xyz1,
                                           torch::Tensor xyz2) {
  if (xyz1.scalar_type() == torch::kFloat && xyz2.scalar_type() == torch::kFloat) {
    return chamfer_cuda_forward(xyz1, xyz2);
  } else if (xyz1.scalar_type() == torch::kDouble && xyz2.scalar_type() == torch::kDouble) {
    return chamfer_cuda_forward_double(xyz1, xyz2);
  } else {
    throw std::invalid_argument("Input tensors must have the same scalar type, either float or double.");
  }
}

std::vector<torch::Tensor> chamfer_backward(torch::Tensor xyz1,
                                            torch::Tensor xyz2,
                                            torch::Tensor idx1,
                                            torch::Tensor idx2,
                                            torch::Tensor grad_dist1,
                                            torch::Tensor grad_dist2) {
  if (xyz1.scalar_type() == torch::kFloat && xyz2.scalar_type() == torch::kFloat) {
    return chamfer_cuda_backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2);
  } else if (xyz1.scalar_type() == torch::kDouble && xyz2.scalar_type() == torch::kDouble) {
    return chamfer_cuda_backward_double(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2);
  } else {
    throw std::invalid_argument("Input tensors must have the same scalar type, either float or double.");
  }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &chamfer_forward, "Chamfer forward (CUDA)");
  m.def("backward", &chamfer_backward, "Chamfer backward (CUDA)");
}
