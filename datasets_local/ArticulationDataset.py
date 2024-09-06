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
import sys
from plyfile import PlyData
from angle_emb import AnglE

sys.path.append('A-SDF/')
from tools.switch_label import to_switch_label


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
    def __init__(self, split, points_num, dirpath, data_split_file, **kwargs):
        self.split  = split
        self.dirpath = dirpath
        if 'token_dims' in kwargs.keys():
            self.token_dims = kwargs['token_dims']
        else:
            self.token_dims = 768
        if 'start_idx' in kwargs.keys():
            self.start_idx = kwargs['start_idx']
        else:
            self.start_idx = 0
        self.data_split_file = data_split_file
        
        self.valid_data = self._load_data()
        self.points_num = points_num
        if 'mpn' in kwargs.keys():
            self.mpn = kwargs['mpn']
            self.num_nodes = kwargs['num_nodes']
        else:
            self.mpn = False
      
        
        
        self.language_embed_dict = np.load(kwargs['language_embed_file'], allow_pickle=True)
            
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
        label = vertex_data['label'] - 1
        
        if 'sdf' in vertex_data:
            sdf = vertex_data['sdf'] 
            # Numpy array로 변환
            vertex_array = np.vstack((x, y, z, sdf, label)).T
            # remain only negative sdf
            vertex_array = vertex_array[vertex_array[...,-2] < 0]
        else:
            vertex_array = np.vstack((x, y, z, label)).T
            
        assert vertex_array[...,-1].min() == 0 # 0은 없었다고 가정
        
        assert label.max() < self.num_nodes, instance_pose_path
        if self.mpn:
                
            # angle = AnglE.from_pretrained('SeanLee97/angle-bert-base-uncased-nli-en-v1', pooling_strategy='cls_avg').cuda()
            
            instance2langemb = torch.zeros(self.num_nodes, self.token_dims).cuda() # HARDCODED
            for instance_pose_dict in instance_pose_json.values():
                
                if instance_pose_dict['index'] != 0:
                    idx = instance_pose_dict['index'] - 1 #HARDCODED 0은 없었다.
                    # encoded_feat = angle.encode([instance_pose_dict['name']], to_numpy=False)[0] #N 768
                    instance2langemb[idx] = torch.tensor(self.language_embed_dict[instance_pose_dict['name']]).cuda()
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
       
        if self.split == 'trn':
            #unorganized로 바꿈
            shuf = list(range(len(vertex_array)))
            random.shuffle(shuf)
            vertex_array = vertex_array[shuf]
        
        
        
        pc = vertex_array[:, :3]
        pc = self.pc_norm(pc)
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
        
        #validity check
        check_data = np.load(self.data_split_file, allow_pickle=True)
        print("checking...")
        for dirpath, dirname, filenames in os.walk(dir):
            data_label = dirpath.split('/')[-1]
            #validity check
            if dirpath.split('/')[-1].split('_')[0] == 'pose':
                assert os.path.isfile(os.path.join(dirpath, 'points_with_sdf_label_binary.ply')) or os.path.isfile(os.path.join(dirpath, 'points_with_labels_binary.ply'))
                spt, cat, inst = dirpath.split('/')[-4:-1]
                assert inst.isdigit(), inst
                inst = int(inst)
                assert check_data[inst] == [cat, spt], f"{inst}, {cat}, {spt}, answer: {check_data[inst]}"
                # obj_idx = dirpath.split('/')[-2]
                # assert obj_idx.isdigit(), obj_idx
                if os.path.isfile(os.path.join(dirpath, 'points_with_sdf_label_binary.ply')):
                    total_valid_paths.append(os.path.join(dirpath, 'points_with_sdf_label_binary.ply'))
                elif os.path.isfile(os.path.join(dirpath, 'points_with_labels_binary.ply')):
                    total_valid_paths.append(os.path.join(dirpath, 'points_with_labels_binary.ply'))
                else:
                    raise NotImplementedError
            # for filename in filenames:
            #     # if filename == 'points_with_sdf_label.ply':
            #     if filename == 'points_with_sdf_label_binary.ply' or filename == 'points_with_labels_binary.ply':
            #         obj_idx = dirpath.split('/')[-2]
            #         assert obj_idx.isdigit(), obj_idx
            #         if int(obj_idx) >= self.start_idx:
            #             total_valid_paths.append(os.path.join(dirpath, filename))
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
    
  
  
class PartDatasetTableOne(Dataset):
    def __init__(self, split, points_num, dirpath, dir_pkl:str, **kwargs):
        self.split  = split
        # 여기서 dirpath
        self.dirpath = dirpath
        if 'token_dims' in kwargs.keys():
            self.token_dims = kwargs['token_dims']
        else:
            self.token_dims = 768
        if 'start_idx' in kwargs.keys():
            self.start_idx = kwargs['start_idx']
        else:
            self.start_idx = 0
        
        # table1용 세팅
        self.dir_pkl = dir_pkl
        
        self.mode = self.dir_pkl.split('/')[-1][:-4]
        assert self.mode in {"double_revolute", "double_prismatic", "one_revolute", "one_prismatic"}, self.mode
        if 'double' in self.mode:
            self.num_atc = 2
        else:
            self.num_atc = 1
        
        self.valid_data = self._load_data()
        self.points_num = points_num
        if 'mpn' in kwargs.keys():
            self.mpn = kwargs['mpn']
            self.num_nodes = kwargs['num_nodes']
        else:
            self.mpn = False
        
        if 'normalize' in kwargs['normalize']:
            self.normalize = kwargs['normalize']
        else:
            self.normalize = True
            
          
        
            
        self.language_embed_dict = np.load(kwargs['language_embed_file'], allow_pickle=True)
            
    def __getitem__(self, index):
        '''
        part label이 0부터 num_atc인거 까지만 노드로 불러온다. 
        '''        
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
        label = vertex_data['label'] 
        
        
        
        label = label - 1
        assert 'sdf' in vertex_data
        label = label[sdf < 0]
        # 라벨을 바꾸어줌: -1 안해줘도 그냥 그대로 가도록 해주는 코드
        new_label = np.full_like(label, -100)
        if model_id in to_switch_label:
            unique_label = np.unique(label)
            for ul in unique_label:
                # ul은 0,1,2, .... 일 것임
                to_change = to_switch_label[model_id][ul]
                if to_switch_label[model_id][ul] != -100:
                    new_label[label == ul] = to_change
                    print("instance", model_id, "change", ul, "to", to_change)
        else:
            # <= num_atc_parts인것들 저장,    0 ~ num_atc_parts 
            for i in range(self.num_atc+1):
                new_label[label == i] = i
        
        #체크 num_atc_parts+1보다 큰라벨은 없어야
        assert new_label.max() == self.num_atc+1, f"num_atc_parts: {self.num_atc}, new label: {new_label.max()}"        
        
        label = new_label
        
        
        sdf = vertex_data['sdf'] 
        # Numpy array로 변환
        vertex_array = np.vstack((x, y, z, sdf)).T
        # remain only negative sdf
        vertex_array = vertex_array[vertex_array[...,-1] < 0]
        
        #concat with label
        vertex_array = np.concatenate((vertex_array, label[..., np.newaxis]), axis=-1)
        assert vertex_array.shape[-1] == 5
            
        assert vertex_array[...,-1].min() == 0 # 0은 없었다고 가정
        
        
        
        if self.mpn:
                
            # angle = AnglE.from_pretrained('SeanLee97/angle-bert-base-uncased-nli-en-v1', pooling_strategy='cls_avg').cuda()
            
            instance2langemb = torch.zeros(self.num_nodes, self.token_dims).cuda() # HARDCODED
            for instance_pose_dict in instance_pose_json.values():
                
                if instance_pose_dict['index'] != 0:
                    idx = instance_pose_dict['index'] - 1 #HARDCODED 0은 없었다.
                    # encoded_feat = angle.encode([instance_pose_dict['name']], to_numpy=False)[0] #N 768
                    instance2langemb[idx] = torch.tensor(self.language_embed_dict[instance_pose_dict['name']]).cuda()
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
                '''
                1부터 시작하는 링크 인덱스 
                '''
                # parent link와 child link 탐색하고, num_atc parts가=1이면 1,2 num_atc_parts=2이면 1,2,3만본다.
                p_idx = joint_info['parent_link']['index']
                c_idx = joint_info['child_link']['index']
                
                # 라벨 스위치
                if model_id in to_switch_label:
                    to_p_change = to_switch_label[model_id][p_idx-1]
                    print("pidx with zero start", p_idx-1, "change to", to_p_change)
                    if to_switch_label[model_id][p_idx-1] == -100:
                        continue
                    p_idx = to_p_change
                    
                    
                    to_c_change = to_switch_label[model_id][c_idx-1]
                    print("pidx with zero start", c_idx-1, "change to", to_c_change)
                    if to_switch_label[model_id][c_idx-1] == -100:
                        continue
                    c_idx = to_c_change
                    
                '''
                이 실험에서는 형평성을 위해 집중할 label과 joint에만 유효하다. 
                '''
                if self.num_atc == 1:
                    if (p_idx == 0 and c_idx) == 1 or (p_idx == 1 and c_idx == 0):
                        
                        joint_info_dict['confidence'][p_idx][c_idx] = 1
                        if joint_info['type'] == 'prismatic':
                            joint_info_dict['type'][p_idx][c_idx][0] = 1
                        elif joint_info['type'] == 'revolute_unwrapped': 
                            #limit이 있는 revolute joint만 학습한다.
                            joint_info_dict['type'][p_idx][c_idx][1] = 1
                        
                        elif joint_info['type'] == 'revolute':
                            print("THIS CANNOT BE HAPPENED")
                            raise  NotImplementedError
                        else:
                            print(joint_info['type'])
                            raise NotImplementedError
                        qpos_range = joint_info['qpos_limit'][1] - joint_info['qpos_limit'][0]
                        # assert qpos_range > 0, qpos_range
                        if np.isfinite(qpos_range) and qpos_range != 0:
                            joint_info_dict['qpos_min'][p_idx][c_idx] = joint_info['qpos_limit'][0]
                            joint_info_dict['qpos_max'][p_idx][c_idx] = joint_info['qpos_limit'][1]
                            assert joint_info['type'] == 'revolute_unwrapped' or joint_info['type'] == 'prismatic', instance_path
                            joint_info_dict['qpos'][p_idx][c_idx] = (joint_info['qpos'] - joint_info['qpos_limit'][0]) / qpos_range
                            assert joint_info_dict['qpos'][p_idx][c_idx] >= 0 and joint_info_dict['qpos'][p_idx][c_idx] <= 1, joint_info_dict['qpos']
                        else:
                            # assert joint_info['type'] == 'revolute' or joint_info['type'] == 'prismatic'
                            print("THIS CANNOT BE HAPPENED")
                            raise NotImplementedError
                        
                elif self.num_atc == 2:
                    if (p_idx == 0 and c_idx) == 1 or (p_idx == 1 and c_idx == 0) or (p_idx == 0 and c_idx == 2) or (p_idx == 2 and c_idx == 0):
                        '''
                        NOTE: 위에거 복붙이어서 neat하지 않음
                        '''
                        joint_info_dict['confidence'][p_idx][c_idx] = 1
                        if joint_info['type'] == 'prismatic':
                            joint_info_dict['type'][p_idx][c_idx][0] = 1
                        elif joint_info['type'] == 'revolute_unwrapped': 
                            #limit이 있는 revolute joint만 학습한다.
                            joint_info_dict['type'][p_idx][c_idx][1] = 1
                        
                        elif joint_info['type'] == 'revolute':
                            print("THIS CANNOT BE HAPPENED")
                            raise  NotImplementedError
                        else:
                            print(joint_info['type'])
                            raise NotImplementedError
                        qpos_range = joint_info['qpos_limit'][1] - joint_info['qpos_limit'][0]
                        # assert qpos_range > 0, qpos_range
                        if np.isfinite(qpos_range) and qpos_range != 0:
                            joint_info_dict['qpos_min'][p_idx][c_idx] = joint_info['qpos_limit'][0]
                            joint_info_dict['qpos_max'][p_idx][c_idx] = joint_info['qpos_limit'][1]
                            assert joint_info['type'] == 'revolute_unwrapped' or joint_info['type'] == 'prismatic', instance_path
                            joint_info_dict['qpos'][p_idx][c_idx] = (joint_info['qpos'] - joint_info['qpos_limit'][0]) / qpos_range
                            assert joint_info_dict['qpos'][p_idx][c_idx] >= 0 and joint_info_dict['qpos'][p_idx][c_idx] <= 1, joint_info_dict['qpos']
                        else:
                            # assert joint_info['type'] == 'revolute' or joint_info['type'] == 'prismatic'
                            print("THIS CANNOT BE HAPPENED")
                            raise NotImplementedError
                else:
                    raise NotImplementedError
            
            
            # 체크용
            if self.num_atc == 1:
                assert (joint_info_dict['qpos'][0][1] != 0) or (joint_info_dict['qpos'][1][0] != 0), joint_info_dict['qpos']
            elif self.num_atc == 2:
                assert ((joint_info_dict['qpos'][0][1] != 0) or (joint_info_dict['qpos'][1][0] != 0)) and ((joint_info_dict['qpos'][0][2] != 0) or (joint_info_dict['qpos'][2][0] != 0)), joint_info_dict['qpos']
       
        if self.split == 'trn':
            #unorganized로 바꿈
            shuf = list(range(len(vertex_array)))
            random.shuffle(shuf)
            vertex_array = vertex_array[shuf]
        
        
        
        pc = vertex_array[:, :3]
        if self.normalize:
            # 테이블 1 실험은 normalize해서 limit range 주어진것과 아닌 것을 비교할 것이므로 설정해둔다.
            pc = self.pc_norm(pc)
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
        #여기서 dirpath는 pose_data까지여야 
        dir = self.dirpath
        
        print("INIT TABLE 1 MODE.")
        #실험용으로 사용할 데이터
        pkl_data = np.load(self.pkl_dir, allow_pickle=True)
        
        for cat in pkl_data.keys():
            for spt in pkl_data[cat].keys():
                instances = pkl_data[cat][spt]
                for instance in instances:
                    for i in range(100):
                        filename = os.path.join(dir, cat, spt, str(instance), f"pose_{i}", 'points_with_sdf_label_binary.ply')
                        assert os.path.isfile(filename)
                        total_valid_paths.append(filename)
                    
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
    