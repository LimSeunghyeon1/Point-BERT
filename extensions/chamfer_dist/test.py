# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-10 10:38:01
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-26 14:21:36
# @Email:  cshzxie@gmail.com
#
# Note:
# - Replace float -> double, kFloat -> kDouble in chamfer.cu

import os
import sys
import torch
import unittest


from torch.autograd import gradcheck

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from extensions.chamfer_dist import ChamferFunction, ChamferFunctionPad, ChamferDistancePadL2,ChamferDistanceL2

def shuffle_tensor(tensor):
    # 텐서의 배치 차원을 유지하면서 각 배치의 포인트를 랜덤하게 섞습니다.
    batch_size, num_points, num_features = tensor.shape
    shuffled_tensor = torch.empty_like(tensor)
    
    for b in range(batch_size):
        # 현재 배치의 포인트를 랜덤하게 섞습니다.
        indices = torch.randperm(num_points)
        shuffled_tensor[b] = tensor[b][indices]
    
    return shuffled_tensor

class ChamferDistanceTestCase(unittest.TestCase):
    def test_chamfer_dist(self):
        x = torch.rand(4, 128, 3).double()
        '''
        # 랜덤 제로 포인트를 생성
        zeros = torch.zeros(4, 128, 3)

        # 새로운 텐서 생성: 원본 텐서와 제로 포인트를 수직으로 합칩니다.
        new_x = torch.cat((x, zeros), dim=1)
        shuffled_new_x = shuffle_tensor(new_x)
        shuffled_new_x.requires_grad=True
        '''
        
        y = x.clone()
        x.requires_grad = True
        y.requires_grad = True
    
        
        # print("IS TWO ARE SAME?")
        print("1:", ChamferDistanceL2(ignore_zeros=True)(x.cuda(), y.cuda()))
        # print("gt:", ChamferDistanceL2(ignore_zeros=True)(x.cuda(), y.cuda()))
        # print("2:", ChamferDistanceL2(ignore_zeros=True)(shuffled_new_x.cuda(), y.cuda()))
        
        print(gradcheck(ChamferFunction.apply, [x.cuda(), y.cuda()]))
        # print(gradcheck(ChamferFunctionPad.apply, [x.cuda(), y.cuda()], eps=.1, atol=.1))
        # print(gradcheck(ChamferFunction.apply, [x.cuda(), y.cuda()], eps=.1, atol=.1))


if __name__ == '__main__':
    ChamferDistanceTestCase().test_chamfer_dist()
    # unittest.main()
    # import pdb
    # x = torch.rand(32,128,3)
    # y = torch.rand(32,128,3)
    # pdb.set_trace()

    
