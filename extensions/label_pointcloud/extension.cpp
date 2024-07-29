#include <torch/extension.h>
#include <tuple>
#include <iostream>

// CUDA functions declarations for float
extern "C" void launch_label_pointcloud_kernel(const float* data, const int* label, float* output,  int B, int N, int num_labels);
extern "C" void launch_label_pointcloud_backward_kernel(const float* grad_output, const int* label, float* grad_data, int B, int N, int num_labels);

// CUDA functions declarations for double
extern "C" void launch_label_pointcloud_kernel_double(const double* data, const int* label, double* output, int B, int N, int num_labels);
extern "C" void launch_label_pointcloud_backward_kernel_double(const double* grad_output, const int* label, double* grad_data, int B, int N, int num_labels);

torch::Tensor label_pointcloud(torch::Tensor data, torch::Tensor label, int num_labels) {
    if (!data.is_cuda() || !label.is_cuda()) {
        throw std::runtime_error("Input tensors must be CUDA tensors");
    }

    int B = data.size(0);
    int N = data.size(1);

    auto options = torch::TensorOptions().dtype(data.dtype()).device(data.device());
    auto output = torch::zeros({B, num_labels, N, 3}, options);
    // auto input_index = torch::zeros({B, N}, options.dtype(torch::kInt32));

    if (data.dtype() == torch::kFloat) {
        launch_label_pointcloud_kernel(data.data_ptr<float>(), label.data_ptr<int>(), output.data_ptr<float>(),  B, N, num_labels);
    } else if (data.dtype() == torch::kDouble) {
        launch_label_pointcloud_kernel_double(data.data_ptr<double>(), label.data_ptr<int>(), output.data_ptr<double>(),  B, N, num_labels);
    } else {
        throw std::runtime_error("Unsupported data type");
    }

    return output;
}

torch::Tensor label_pointcloud_backward(torch::Tensor grad_output, torch::Tensor label, int B, int N, int num_labels) {
    if (!grad_output.is_cuda() || !label.is_cuda() ) {
        throw std::runtime_error("grad tensors must be CUDA tensors");
    }
    
    auto options = torch::TensorOptions().dtype(grad_output.dtype()).device(grad_output.device());
    auto grad_data = torch::zeros({B, N, 3}, options);

    if (grad_output.dtype() == torch::kFloat) {
        launch_label_pointcloud_backward_kernel(grad_output.data_ptr<float>(), label.data_ptr<int>(), grad_data.data_ptr<float>(), B, N, num_labels);
    } else if (grad_output.dtype() == torch::kDouble) {
        launch_label_pointcloud_backward_kernel_double(grad_output.data_ptr<double>(), label.data_ptr<int>(), grad_data.data_ptr<double>(), B, N, num_labels);
    } else {
        throw std::runtime_error("Unsupported data type");
    }

    return grad_data;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("label_pointcloud", &label_pointcloud, "Label Point Cloud");
    m.def("label_pointcloud_backward", &label_pointcloud_backward, "Label Point Cloud Backward");
}
