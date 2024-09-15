# -*- coding: utf-8 -*-
# @Author: Thibault GROUEIX
# @Date:   2019-08-07 20:54:24
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-18 15:06:25
# @Email:  cshzxie@gmail.com

import torch

import chamfer


class ChamferFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
        
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        
        grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2



class ChamferMatrixFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        
        dist1, dist2, idx1, idx2 = chamfer.forward_matrix(xyz1, xyz2)
        
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        grad_xyz1, grad_xyz2 = chamfer.backward_matrix(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2


'''
어떤 텐서 B num_label N 3
num_label과 N개 사이의 거리
'''
class ChamferDistanceMatrixL2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyz1_matrix, xyz2_matrix):
        dist1, dist2 = ChamferMatrixFunction.apply(xyz1_matrix, xyz2_matrix)
        dist2 = dist2.permute(0, 2, 1, 3)
        return torch.mean(dist1, dim=-1) + torch.mean(dist2, dim=-1)
    
    
        

class ChamferDistanceL2(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2, contiguous=True):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)

            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)
            
        
        
            

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        
        return torch.mean(dist1, dim=-1) + torch.mean(dist2, dim=-1)

class ChamferDistancePadL2(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunctionPad.apply(xyz1, xyz2)
        return torch.mean(dist1) + torch.mean(dist2)
    

class ChamferDistanceL2_split(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1), torch.mean(dist2)

class ChamferDistanceL1(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        dist1 = torch.sqrt(dist1)
        dist2 = torch.sqrt(dist2)
        return (torch.mean(dist1) + torch.mean(dist2))/2

def get_mean_pointcloud(xyz1_matrix):
    # 예시 입력 데이터 (B, num_label, N, 3) 형상
    B, num_label, N, _ = xyz1_matrix.shape

    # (0,0,0) 좌표를 제외한 평균값 계산
    # 조건: xyz1_matrix의 각 요소에서 (0,0,0)을 제외하고 평균값을 계산
    # (0,0,0) 좌표가 포함된 마스크 생성
    
    mask = (xyz1_matrix != torch.tensor([0, 0, 0]).cuda()).any(dim=-1)

    # (0,0,0)을 제외한 점들의 평균값 계산
    # 각 배치, 라벨에 대해서 유효한 포인트들만 선택
    sum_xyz1_matrix = torch.zeros((B, num_label, 3)).cuda()
    count = torch.zeros((B, num_label)).cuda()
    
    for i in range(B):
        for j in range(num_label):
            
            valid_points = xyz1_matrix[i, j, mask[i, j]]
            
            if valid_points.size(0) > 0:  # 유효한 포인트가 있는 경우만
                sum_xyz1_matrix[i, j] = valid_points.sum(dim=0)
                count[i, j] = valid_points.size(0)
    
    # 평균값 계산
    mean_xyz1_matrix = sum_xyz1_matrix / (count.unsqueeze(-1) + 1e-6)
    
    for i in range(B):
        for j in range(num_label):
            xyz1_matrix[i, j, ~mask[i, j]] = mean_xyz1_matrix[i, j]
    
    return xyz1_matrix

def chamfer_distance_matrix(xyz1_matrix, xyz2_matrix, contiguous=True):
    B, num_labels, N, _ = xyz1_matrix.shape
    B, num_groups, group_size, _ = xyz2_matrix.shape
    xyz1_matrix = get_mean_pointcloud(xyz1_matrix)
    
    
    cdist = ChamferDistanceL2(ignore_zeros=False)
    dist_matrix = torch.zeros(B, num_labels, num_groups).cuda()
    for i in range(num_labels):
        for j in range(num_groups):
            xyz1 = xyz1_matrix[:, i, ...]
            xyz2 = xyz2_matrix[:, j, ...]
            dist_matrix[:, i, j] = cdist(xyz1, xyz2, contiguous=contiguous)

    return dist_matrix

def chamfer_distance_matrix_batch(xyz1_matrix, xyz2_matrix):
    B, num_labels, N, _ = xyz1_matrix.shape
    B2, num_groups, group_size, _ = xyz2_matrix.shape
    assert B == B2, "batch must be same"
    assert B <= 8, "Please use lower batch size ..., unexpected behavior detected at higher batch" 
    xyz1_matrix = get_mean_pointcloud(xyz1_matrix)
    
    cdist = ChamferDistanceMatrixL2()
    return cdist(xyz1_matrix, xyz2_matrix)

def shuffle_tensor(tensor):
    # 텐서의 배치 차원을 유지하면서 각 배치의 포인트를 랜덤하게 섞습니다.
    batch_size, num_labels, num_points, num_features = tensor.shape
    shuffled_tensor = torch.empty_like(tensor)
    
    for b in range(batch_size):
        for l in range(num_labels):
            # 현재 배치의 포인트를 랜덤하게 섞습니다.
            indices = torch.randperm(num_points)
            shuffled_tensor[b][l] = tensor[b][l][indices]
    
    return shuffled_tensor

if __name__ == '__main__':
    from torch.autograd import gradcheck

    torch.manual_seed(0)
    import numpy as np
    test_num = 100
    import random
    import time
    # 가설: batch=1일때는 contiguous 조건 안붙여도 괜찮다.
    
    # for i in range(test_num):
        
    #     batch = random.randint(1, 2)
    #     # batch = 1
    #     x_points = random.randint(100, 512)
    #     y_points = random.randint(100, 512)
    #     x_label = random.randint(2, 10)
    #     y_label = random.randint(2, 10)
    #     print("TEST: i", i, "BATCH", batch, "x label", x_label, "y_label", y_label, "xpoints", x_points, "y points", y_points)        
    #     x = torch.randn(batch, x_label, x_points, 3).cuda().double()
    #     y = torch.randn(batch, y_label, y_points, 3).cuda().double()
    #     x.requires_grad = True
    #     y.requires_grad = True
    #     tt = time.time()
        # result2 = chamfer_distance_matrix_batch(x.clone(), y.clone())
        # gradcheck(ChamferMatrixFunction.apply, (x, y))
        # print("result2 time", time.time()-tt)
        # assert torch.abs(result1 - result2).max() < 1e-4, torch.abs(result1 - result2).max()    
    # print("CLEAR")
    
    # for i in range(test_num):
        
    #     # batch = random.randint(1, 32)
    #     batch = 2
    #     x_points = random.randint(100, 512)
    #     y_points = random.randint(100, 512)
    #     print("TEST: i", i, "BATCH", batch, "xpoints", x_points, "y points", y_points)        
    #     x = torch.randn(batch, x_points, 3).cuda().double()
    #     y = torch.randn(batch, y_points, 3).cuda().double()
    #     x.requires_grad = True
    #     y.requires_grad = True
    #     tt = time.time()
    #     # result2 = chamfer_distance_matrix_batch(x.clone(), y.clone())
    #     gradcheck(ChamferFunction.apply, (x, y))
    #     print("result2 time", time.time()-tt)
    #     # assert torch.abs(result1 - result2).max() < 1e-4, torch.abs(result1 - result2).max()    
    # print("CLEAR")
    
    
    for i in range(test_num):
        
        batch = random.randint(1, 8)
        # batch = 1
        x_points = random.randint(100, 8192)
        y_points = random.randint(100, 8192)
        x_label = random.randint(2, 128)
        y_label = random.randint(2, 128)
        print("TEST: i", i, "BATCH", batch, "x label", x_label, "y_label", y_label, "xpoints", x_points, "y points", y_points)
        
        x = torch.randn(batch, x_label, x_points, 3).cuda()
        y = torch.randn(batch, y_label, y_points, 3).cuda()
        tt = time.time()
        result1 = chamfer_distance_matrix(x.clone(), y.clone(),True)
        print("result1 time", time.time()-tt)
        tt = time.time()
        result2 = chamfer_distance_matrix_batch(x.clone(), y.clone())
        print("result2 time", time.time()-tt)
        assert torch.abs(result1 - result2).max() < 1e-4, torch.abs(result1 - result2).max()    
    print("CLEAR")
    
    
    