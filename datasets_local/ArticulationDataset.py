import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import h5py
import json
from torch.utils.data import Dataset
import os
import cv2
from PIL import Image
# from Small_Network.models.pointnet2_utils import farthest_point_sample
from pointnet2_ops.pointnet2_utils import furthest_point_sample
import open3d as o3d
import matplotlib.pyplot as plt
import time
import pickle
import random
from collections import OrderedDict
from plyfile import PlyData
from angle_emb import AnglE


def random_rotation_matrix():
    """
    Create a random 3D rotation matrix.
    """
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    z = np.random.uniform(0, 2 * np.pi)

    r_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

    r_y = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])

    r_z = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])

    rotation_matrix = r_z @ r_y @ r_x
    return rotation_matrix

class PartDataset(Dataset):
    def __init__(self, split, points_num, dirpath, **kwargs):
        self.split  = split
        self.dirpath = dirpath
        if 'start_idx' in kwargs.keys():
            self.start_idx = kwargs['start_idx']
        else:
            self.start_idx = 0
        self.valid_data = self._load_data()
        self.points_num = points_num
        if 'mpn' in kwargs.keys():
            self.mpn = kwargs['mpn']
            self.num_nodes = kwargs['num_nodes']

        else:
            self.mpn = False
        
    def __getitem__(self, index):
        tic = time.time()
        cloud_path = self.valid_data[index]
        instance_path = '/'.join(cloud_path.split('/')[:-2])
        instance_pose_path = '/'.join(cloud_path.split('/')[:-1])
        with open(os.path.join(instance_path, 'link_cfg.json'), 'r') as f:
            instance_pose_json = json.load(f)
        
        model_id = instance_path.split('/')[-1]
        assert model_id.isdigit(), model_id
        model_id = int(model_id)
        # faucet인지 아닌지 체크용

        for instance_pose_dict in instance_pose_json.values():
            if instance_pose_dict['index'] == 0:
                taxomony_id = instance_pose_dict['name']
                
                
        
        # Ply 파일 읽기
        ply_data = PlyData.read(cloud_path)
        
        # Vertex 데이터 추출
        vertex_data = ply_data['vertex']

        # 필요한 속성 추출
        x = vertex_data['x']
        y = vertex_data['y']
        z = vertex_data['z']
        sdf = vertex_data['sdf']
        label = vertex_data['label'] - 1
        assert label.min() == 0 # 0은 없었다고 가정
        assert label.max() < self.num_nodes, instance_pose_path
        if self.mpn:
                
            angle = AnglE.from_pretrained('SeanLee97/angle-bert-base-uncased-nli-en-v1', pooling_strategy='cls_avg').cuda()
            
            instance2langemb = torch.zeros(self.num_nodes, 768).cuda() # HARDCODED
            for instance_pose_dict in instance_pose_json.values():
                
                if instance_pose_dict['index'] != 0:
                    idx = instance_pose_dict['index'] - 1 #HARDCODED 0은 없었다.
                    instance2langemb[idx] = angle.encode([instance_pose_dict['name']], to_numpy=False)[0] #N 768
                    norm = torch.norm(instance2langemb[idx], p=2, dim=-1, keepdim=True)
                    # 벡터를 노름으로 나누어 단위 벡터를 만듭니다.
                    instance2langemb[idx] = instance2langemb[idx] / (norm + 1e-6)
                    
            joint_info_dict = dict()
            joint_info_dict['confidence'] = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float32).cuda()
            joint_info_dict['type'] = torch.zeros(self.num_nodes, self.num_nodes, 2, dtype=torch.float32).cuda()
            joint_info_dict['qpos'] = torch.full((self.num_nodes, self.num_nodes),-1, dtype=torch.float32).cuda()
            joint_info_dict['qpos_min'] = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float32).cuda()
            joint_info_dict['qpos_max'] = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float32).cuda()
            # joint information도 추가
            with open(os.path.join(instance_pose_path, 'joint_cfg.json'), 'r') as f:
                joint_dict = json.load(f)
            
            for joint_info in joint_dict.values():
                joint_info_dict['confidence'][joint_info['parent_link']['index']-1][joint_info['child_link']['index']-1] = 1
                if joint_info['type'] == 'prismatic':
                    joint_info_dict['type'][joint_info['parent_link']['index']-1][joint_info['child_link']['index']-1][0] = 1
                elif joint_info['type'] == 'revolute_unwrapped': 
                    #limit이 있는 revolute joint만 학습한다.
                    joint_info_dict['type'][joint_info['parent_link']['index']-1][joint_info['child_link']['index']-1][1] = 1
                elif joint_info['type'] == 'revolute':
                    pass
                else:
                    print(joint_info['type'])
                    raise NotImplementedError
                qpos_range = joint_info['qpos_limit'][1] - joint_info['qpos_limit'][0]
                # assert qpos_range > 0, qpos_range
                if np.isfinite(qpos_range) and qpos_range != 0:
                    joint_info_dict['qpos_min'][joint_info['parent_link']['index']-1][joint_info['child_link']['index']-1] = joint_info['qpos_limit'][0]
                    joint_info_dict['qpos_max'][joint_info['parent_link']['index']-1][joint_info['child_link']['index']-1] = joint_info['qpos_limit'][1]
                    assert joint_info['type'] == 'revolute_unwrapped' or joint_info['type'] == 'prismatic', instance_path
                    joint_info_dict['qpos'][joint_info['parent_link']['index']-1][joint_info['child_link']['index']-1] = (joint_info['qpos'] - joint_info['qpos_limit'][0]) / qpos_range
                    assert joint_info_dict['qpos'][joint_info['parent_link']['index']-1][joint_info['child_link']['index']-1] >= 0 and joint_info_dict['qpos'][joint_info['parent_link']['index']-1][joint_info['child_link']['index']-1] <= 1, joint_info_dict['qpos']
                else:
                    # assert joint_info['type'] == 'revolute' or joint_info['type'] == 'prismatic'
                    pass
        # Numpy array로 변환
        vertex_array = np.vstack((x, y, z, sdf, label)).T
        # remain only negative sdf
        vertex_array = vertex_array[vertex_array[...,-2] < 0]
        
        if self.split == 'trn':
            #unorganized로 바꿈
            shuf = list(range(len(vertex_array)))
            random.shuffle(shuf)
            vertex_array = vertex_array[shuf]
        
        
        
        pc = vertex_array[:, :3]
        lbl = vertex_array[:, -1]
        
        if len(pc) < self.points_num:
            pc_pad = np.zeros((self.points_num, 3))
            lbl_pad = np.zeros((self.points_num))
            pc_pad[:len(pc), :] = pc
            lbl_pad[:len(pc)] = lbl
            pc = torch.from_numpy(pc_pad).float().cuda()
            lbl = torch.from_numpy(lbl_pad).type(torch.int64).cuda()
        else:
            pc = torch.from_numpy(pc).unsqueeze(0).float().cuda()
            lbl = torch.from_numpy(lbl).type(torch.int64).cuda()
            pc = pc.contiguous()
            input_pcid = furthest_point_sample(pc, self.points_num).long().reshape(-1)
            pc = pc[:, input_pcid, :].squeeze()
            assert len(lbl.shape) == 1
            lbl = lbl[input_pcid]
            # if self.split == 'trn':
                # Add Gaussian noise
                # noise = 0.01 * np.random.randn(len(pc_lbl), 3)
                # pc_lbl += torch.from_numpy(noise).float().cuda()
                
                # Random 3D rotation
                # rotation_matrix = random_rotation_matrix()
                # pc_lbl = torch.from_numpy((rotation_matrix @ pc_lbl.cpu().numpy().T).T).float().cuda()        
        if self.mpn:
            return taxomony_id, model_id, pc.squeeze(), lbl, instance2langemb, joint_info_dict, cloud_path
        else:
            return taxomony_id, model_id, pc.squeeze(), lbl
        
    def __len__(self):
        return len(self.valid_data) 
        
    def _load_data(self):
        total_valid_paths = []
        dir = self.dirpath

        for dirpath, dirname, filenames in os.walk(dir):
            data_label = dirpath.split('/')[-1]
            for filename in filenames:
                # if filename == 'points_with_sdf_label.ply':
                if filename == 'full_point_cloud.ply':
                    obj_idx = dirpath.split('/')[-2]
                    assert obj_idx.isdigit(), obj_idx
                    if int(obj_idx) >= self.start_idx:
                        total_valid_paths.append(os.path.join(dirpath, filename))
            # if data_label.split('_')[0].isdigit():
            #     total_valid_paths.append(dirpath)
        return total_valid_paths    
    
    
    def _get_label_embedding(label_path):
        with open(label_path, 'r') as json_file:
            link_dict = json.load(json_file)
        idx2name = dict()    
        for link_item in link_dict.values():
            idx2name[link_item['index']] = link_item['name']
        return idx2name 
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    
    
'''
이전 데이터셋 나중에 쓸 수도 있으니 백업용
'''
class PartDatasetWithLang(Dataset):
    def __init__(self, split, points_num, dirpath, idx2name, mode=None):
        self.split  = split
        self.dirpath = dirpath

        self.valid_data = self._load_data()
        self.points_num = points_num
        self.mode = mode
        self.idx2name = idx2name
        label_text = list(self.idx2name.values())
        self.label_text = label_text
        
        #Bring LLM(Angle-BERT)
        if 'lang' in self.split:
            
            angle = AnglE.from_pretrained('SeanLee97/angle-bert-base-uncased-nli-en-v1', pooling_strategy='cls_avg').cuda()
            label_embed_list = angle.encode(label_text)
            self.name2embed = dict()
            for i, text in enumerate(label_text):
                self.name2embed[text] = label_embed_list[i]
        
    def __getitem__(self, index):
        cloud_path = self.valid_data[index]
        instance_path = '/'.join(cloud_path.split('/')[:-2])
        instance_pose_path = '/'.join(cloud_path.split('/')[:-1])
        with open(os.path.join(instance_path, 'link_cfg.json'), 'r') as f:
            instance_pose_json = json.load(f)
        label_names = list(self.idx2name.values())
        instancelbl2name = dict()
        print("instance pose", instance_pose_json.values())
        for instance_pose_dict in instance_pose_json.values():
            name = instance_pose_dict['name']
            instancelbl = instance_pose_dict['index']
            if instancelbl == 0:
                assert instance_pose_dict['name'] == 'Faucet' , instance_pose_dict #HARDCODED
            if name in label_names:
                instancelbl2name[instancelbl] = name
                
        # Ply 파일 읽기
        ply_data = PlyData.read(cloud_path)

        
        # Vertex 데이터 추출
        vertex_data = ply_data['vertex']

        # 필요한 속성 추출
        x = vertex_data['x']
        y = vertex_data['y']
        z = vertex_data['z']
        red = vertex_data['red']
        green = vertex_data['green']
        blue = vertex_data['blue']
        alpha = vertex_data['alpha']
        label = vertex_data['label']
            
        # Numpy array로 변환
        vertex_array = np.vstack((x, y, z, red, green, blue, alpha, label)).T
        if self.split == 'trn':
            #unorganized로 바꿈
            shuf = list(range(len(vertex_array)))
            random.shuffle(shuf)
            vertex_array = vertex_array[shuf]
        
        pc = vertex_array[:, :3]
        color = vertex_array[:, 3:6]
        lbl = vertex_array[:, -1]
        lbl_unique = np.unique(lbl)
        
        data = list()
        print("instancelbl2name", instancelbl2name)
        for lu in lbl_unique:
            if lu not in instancelbl2name.keys():
                print("lu", lu)
                continue
            
            pc_lbl = pc[lbl == lu, :]
            lbl_name = instancelbl2name[lu]
            # Normalization
            print("b4", pc_lbl)
            pc_lbl = (pc_lbl - pc_lbl.mean(axis=0)) / (pc_lbl.std(axis=0) + 1e-10)
            if len(pc_lbl) < self.points_num:
                pc_lbl_pad = np.zeros((self.points_num, 3))
                pc_lbl_pad[:len(pc_lbl), :] = pc_lbl
                pc_lbl = torch.from_numpy(pc_lbl_pad).float().cuda()
            else:
                pc_lbl = torch.from_numpy(pc_lbl).unsqueeze(0).float().cuda()
                pc_lbl = pc_lbl.contiguous()
                input_pcid = furthest_point_sample(pc_lbl, self.points_num).long().reshape(-1)
                pc_lbl = pc_lbl[:, input_pcid, :].squeeze()

            
            

            # if self.split == 'trn':
                # Add Gaussian noise
                # noise = 0.01 * np.random.randn(len(pc_lbl), 3)
                # pc_lbl += torch.from_numpy(noise).float().cuda()
                
                # Random 3D rotation
                # rotation_matrix = random_rotation_matrix()
                # pc_lbl = torch.from_numpy((rotation_matrix @ pc_lbl.cpu().numpy().T).T).float().cuda()

            if 'lang' in self.split:
                data.append({'pc': pc_lbl, "lang_emb": self.name2embed[lbl_name]})
            else:
                data.append({'pc': pc_lbl})
        
        
        return data 
        
    def __len__(self):
        return len(self.valid_data) 
        
    def _load_data(self):
        total_valid_paths = []
        dir = self.dirpath

        for dirpath, dirname, filenames in os.walk(dir):
            data_label = dirpath.split('/')[-1]
            for filename in filenames:
                if filename == 'points_with_sdf_label.ply':
                    total_valid_paths.append(os.path.join(dirpath, filename))
            # if data_label.split('_')[0].isdigit():
            #     total_valid_paths.append(dirpath)
        return total_valid_paths    
    
    
    def _get_label_embedding(label_path):
        with open(label_path, 'r') as json_file:
            link_dict = json.load(json_file)
        idx2name = dict()    
        for link_item in link_dict.values():
            idx2name[link_item['index']] = link_item['name']
        return idx2name 
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc