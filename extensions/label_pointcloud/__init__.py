import torch
from torch.autograd import Function
import label_pointcloud_extension

class LabelPointCloudFunction(Function):
    @staticmethod
    def forward(ctx, data, label, num_labels):
        B, N, _ = data.size()
        label = label.int()
        data = data.contiguous()
        label = label.contiguous()
        assert N >= num_labels, "not supported"
        output = label_pointcloud_extension.label_pointcloud(data, label, num_labels)
        
        ctx.save_for_backward(data, label)
        ctx.num_labels = num_labels
        ctx.B = B
        ctx.N = N
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        data, label = ctx.saved_tensors
        B = ctx.B
        N = ctx.N
        num_labels = ctx.num_labels
        grad_output = grad_output.contiguous()
        label = label.contiguous()
        grad_data = label_pointcloud_extension.label_pointcloud_backward(grad_output, label, B, N, num_labels)
        return grad_data, None, None

def label_pointcloud(data, label, num_labels):
    return LabelPointCloudFunction.apply(data, label, num_labels)

