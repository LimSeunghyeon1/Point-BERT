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
# from angle_emb import AnglE

sys.path.append('./A-SDF/')
from switch_label import to_switch_label


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
        
        new_label = vertex_array[...,-1]
        
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
        
        points_num = self.points_num
        if len(pc) < points_num:
            # Padding logic for smaller point clouds
            pc_pad = np.zeros((points_num, 3))
            lbl_pad = np.zeros((points_num))
            pc_pad[:len(pc), :] = pc
            lbl_pad[:len(pc)] = lbl
            pc = torch.from_numpy(pc_pad).float().cuda()
            lbl = torch.from_numpy(lbl_pad).type(torch.int64).cuda()
        else:
            pc = torch.from_numpy(pc).unsqueeze(0).float().cuda()
            lbl = torch.from_numpy(lbl).type(torch.int64).cuda()
            pc = pc.contiguous()

            # 샘플링에 앞서 각 라벨에서 최소 하나의 포인트를 먼저 선택
            unique_labels = torch.unique(lbl)
            selected_indices = []

            for label_ in unique_labels:
                label_indices = torch.where(lbl == label_)[0]
                if len(label_indices) > 0:
                    selected_indices.append(label_indices[torch.randint(len(label_indices), (1,))])

            selected_indices = torch.cat(selected_indices)

            # 나머지 포인트는 기존 방식으로 샘플링
            if len(selected_indices) < points_num:
                remaining_points = points_num - len(selected_indices)
                input_pcid = furthest_point_sample(pc, remaining_points).long().reshape(-1)

                # 기존 샘플링 포인트에 포함되지 않도록 필터링
                input_pcid = input_pcid[~torch.isin(input_pcid, selected_indices)]
                selected_indices = torch.cat((selected_indices, input_pcid))

            # 선택된 포인트로 pc와 lbl 업데이트
            pc = pc[:, selected_indices, :].squeeze()
            lbl = lbl[selected_indices]
            
        assert set(new_label.flatten().tolist()) == set(lbl.flatten().tolist()), f"{set(new_label.flatten().tolist())} , {set(lbl.flatten().tolist())}"   
        if self.mpn:
            return taxomony_id, model_id, pc.squeeze(), lbl, instance2langemb, joint_info_dict, cloud_path
        else:
            return taxomony_id, model_id, pc.squeeze(), lbl
        
    def __len__(self):
        return len(self.valid_data) 
        
    def _load_data(self):
        total_valid_paths = []
        dir = self.dirpath
        check_data = np.load(self.data_split_file, allow_pickle=True)
        print("checking...")

        if self.split == 'all':
            # mpn loader에 담을 모든 데이터 (train , val, test split을 담음)
            # NOTE: 여기서 dir은 '../pose_data/'... all 아니면 ../pose_data/train/Table/ 이렇게 됨  
            #validity check
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
        else:
            '''
            split == all 이외에는 deprecated될 예정
            '''
            #validity check
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
        
        if 'normalize' in kwargs.keys():
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
        sdf = vertex_data['sdf'] 
        
        
        
        label = label - 1
        assert 'sdf' in vertex_data
        label = label[sdf < 0]
        new_label = np.full_like(label, -100)
        print("new label,", np.unique(new_label), "oh label", np.unique(label))
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
        assert new_label.max() == self.num_atc, f"num_atc_parts: {self.num_atc}, new label: {new_label.max()}"        
        
        label = new_label
        
        print("label unique right after new", np.unique(label))
        # Numpy array로 변환
        vertex_array = np.vstack((x, y, z, sdf)).T
        # remain only negative sdf
        vertex_array = vertex_array[vertex_array[...,-1] < 0]
        
        #concat with label
        vertex_array = np.concatenate((vertex_array, label[..., np.newaxis]), axis=-1)
        assert vertex_array.shape[-1] == 5
            
        # assert vertex_array[...,-1].min() == 0, f"{np.unique(vertex_array[...,-1])}" # 0은 없었다고 가정
        
        
        
        if self.mpn:
                
            # angle = AnglE.from_pretrained('SeanLee97/angle-bert-base-uncased-nli-en-v1', pooling_strategy='cls_avg').cuda()
            
            instance2langemb = torch.zeros(self.num_nodes, self.token_dims).cuda() # HARDCODED
            used_indices = set()
            
            for instance_pose_dict in instance_pose_json.values():
                
                if instance_pose_dict['index'] != 0:
                    idx = instance_pose_dict['index'] - 1 #HARDCODED 0은 없었다.
                    if model_id in to_switch_label:
                        to_change = to_switch_label[model_id][idx]
                        if to_switch_label[model_id][idx] != -100:
                            print("switch", idx, "to", to_change, "in object", model_id)        
                            used_indices.add(to_change)
                            # encoded_feat = angle.encode([instance_pose_dict['name']], to_numpy=False)[0] #N 768
                            instance2langemb[to_change] = torch.tensor(self.language_embed_dict[instance_pose_dict['name']]).cuda()
                            norm = torch.norm(instance2langemb[to_change], p=2, dim=-1, keepdim=True)
                            # 벡터를 노름으로 나누어 단위 벡터를 만듭니다.
                            instance2langemb[to_change] = instance2langemb[to_change] / (norm + 1e-6)
                    else:
                        #그냥 그대로 가도 상관 없음
                        if idx <= self.num_atc:
                            instance2langemb[idx] = torch.tensor(self.language_embed_dict[instance_pose_dict['name']]).cuda()
                            norm = torch.norm(instance2langemb[idx], p=2, dim=-1, keepdim=True)
                            # 벡터를 노름으로 나누어 단위 벡터를 만듭니다.
                            instance2langemb[idx] = instance2langemb[idx] / (norm + 1e-6)
                            used_indices.add(idx)
                
            
            #TODO: used_indices로 무결성 체크
            assert used_indices == {i for i in range(self.num_atc+1)}, used_indices
            
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
                            if self.normalize:
                                joint_info_dict['qpos'][p_idx][c_idx] = (joint_info['qpos'] - joint_info['qpos_limit'][0]) / qpos_range
                                assert joint_info_dict['qpos'][p_idx][c_idx] >= 0 and joint_info_dict['qpos'][p_idx][c_idx] <= 1, joint_info_dict['qpos']
                            else:
                                joint_info_dict['qpos'][p_idx][c_idx] = joint_info['qpos'] - joint_info['qpos_limit'][0]
                                
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
                            if self.normalize:
                                joint_info_dict['qpos'][p_idx][c_idx] = (joint_info['qpos'] - joint_info['qpos_limit'][0]) / qpos_range
                                assert joint_info_dict['qpos'][p_idx][c_idx] >= 0 and joint_info_dict['qpos'][p_idx][c_idx] <= 1, joint_info_dict['qpos']
                            else:
                                #normalize안하면, qpos_min으로 부터 0으로 친다. 
                                joint_info_dict['qpos'][p_idx][c_idx] = joint_info['qpos'] - joint_info['qpos_limit'][0]
                                
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
        points_num = self.points_num
        
        if len(pc) < points_num:
            # Padding logic for smaller point clouds
            pc_pad = np.zeros((points_num, 3))
            lbl_pad = np.zeros((points_num))
            pc_pad[:len(pc), :] = pc
            lbl_pad[:len(pc)] = lbl
            pc = torch.from_numpy(pc_pad).float().cuda()
            lbl = torch.from_numpy(lbl_pad).type(torch.int64).cuda()
        else:
            pc = torch.from_numpy(pc).unsqueeze(0).float().cuda()
            lbl = torch.from_numpy(lbl).type(torch.int64).cuda()
            pc = pc.contiguous()

            # 샘플링에 앞서 각 라벨에서 최소 하나의 포인트를 먼저 선택
            unique_labels = torch.unique(lbl)
            selected_indices = []

            for label_ in unique_labels:
                label_indices = torch.where(lbl == label_)[0]
                if len(label_indices) > 0:
                    selected_indices.append(label_indices[torch.randint(len(label_indices), (1,))])

            selected_indices = torch.cat(selected_indices)

            # 나머지 포인트는 기존 방식으로 샘플링
            if len(selected_indices) < points_num:
                remaining_points = points_num - len(selected_indices)
                input_pcid = furthest_point_sample(pc, remaining_points).long().reshape(-1)

                # 기존 샘플링 포인트에 포함되지 않도록 필터링
                input_pcid = input_pcid[~torch.isin(input_pcid, selected_indices)]
                selected_indices = torch.cat((selected_indices, input_pcid))

            # 선택된 포인트로 pc와 lbl 업데이트
            pc = pc[:, selected_indices, :].squeeze()
            lbl = lbl[selected_indices]
            
        assert set(label.flatten().tolist()) == set(lbl.flatten().tolist()), f"{set(label.flatten().tolist())} , {set(lbl.flatten().tolist())}"
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
        pkl_data = np.load(self.dir_pkl, allow_pickle=True)
        
        for cat in pkl_data.keys():
            for spt in pkl_data[cat].keys():
                instances = pkl_data[cat][spt]
                for instance in instances:
                    for i in range(100):
                        filename = os.path.join(dir, spt, cat, str(instance), f"pose_{i}", 'points_with_sdf_label_binary.ply')
                        assert os.path.isfile(filename), filename
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


def test_func(cloud_path, num_atc, num_nodes, token_dims, normalize, points_num):
    '''
    part label이 0부터 num_atc인거 까지만 노드로 불러온다. 
    '''        
        
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
    sdf = vertex_data['sdf'] 
    
    
    
    label = label - 1
    assert 'sdf' in vertex_data
    label = label[sdf < 0]
    new_label = np.full_like(label, -100)
    print("new label,", np.unique(new_label), "oh label", np.unique(label))
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
        for i in range(num_atc+1):
            new_label[label == i] = i
    
    #체크 num_atc_parts+1보다 큰라벨은 없어야
    assert new_label.max() == num_atc, f"num_atc_parts: {num_atc}, new label: {new_label.max()}"        
    
    label = new_label
    
    print("label unique right after new", np.unique(label))
    # Numpy array로 변환
    vertex_array = np.vstack((x, y, z, sdf)).T
    # remain only negative sdf
    vertex_array = vertex_array[vertex_array[...,-1] < 0]
    
    #concat with label
    vertex_array = np.concatenate((vertex_array, label[..., np.newaxis]), axis=-1)
    assert vertex_array.shape[-1] == 5
        
    # assert vertex_array[...,-1].min() == 0, f"{np.unique(vertex_array[...,-1])}" # 0은 없었다고 가정
    
    
            
    # angle = AnglE.from_pretrained('SeanLee97/angle-bert-base-uncased-nli-en-v1', pooling_strategy='cls_avg').cuda()
    
    instance2langemb = torch.zeros(num_nodes, token_dims).cuda() # HARDCODED
    used_indices = set()
    
    # for instance_pose_dict in instance_pose_json.values():
        
    #     if instance_pose_dict['index'] != 0:
    #         idx = instance_pose_dict['index'] - 1 #HARDCODED 0은 없었다.
    #         if model_id in to_switch_label:
    #             to_change = to_switch_label[model_id][idx]
    #             if to_switch_label[model_id][idx] != -100:
    #                 print("switch", idx, "to", to_change, "in object", model_id)        
    #                 used_indices.add(to_change)
    #                 # encoded_feat = angle.encode([instance_pose_dict['name']], to_numpy=False)[0] #N 768
    #                 instance2langemb[to_change] = torch.tensor(self.language_embed_dict[instance_pose_dict['name']]).cuda()
    #                 norm = torch.norm(instance2langemb[to_change], p=2, dim=-1, keepdim=True)
    #                 # 벡터를 노름으로 나누어 단위 벡터를 만듭니다.
    #                 instance2langemb[to_change] = instance2langemb[to_change] / (norm + 1e-6)
    #         else:
    #             #그냥 그대로 가도 상관 없음
    #             if idx <= self.num_atc:
    #                 instance2langemb[idx] = torch.tensor(self.language_embed_dict[instance_pose_dict['name']]).cuda()
    #                 norm = torch.norm(instance2langemb[idx], p=2, dim=-1, keepdim=True)
    #                 # 벡터를 노름으로 나누어 단위 벡터를 만듭니다.
    #                 instance2langemb[idx] = instance2langemb[idx] / (norm + 1e-6)
    #                 used_indices.add(idx)
        
        
    #TODO: used_indices로 무결성 체크
    # assert used_indices == {i for i in range(self.num_atc+1)}, used_indices
    
    joint_info_dict = dict()
    joint_info_dict['confidence'] = torch.zeros(num_nodes, num_nodes, dtype=torch.float32).cuda()
    joint_info_dict['type'] = torch.zeros(num_nodes, num_nodes, 2, dtype=torch.float32).cuda()
    joint_info_dict['qpos'] = torch.full((num_nodes, num_nodes),-1, dtype=torch.float32).cuda()
    joint_info_dict['qpos_min'] = torch.zeros(num_nodes, num_nodes, dtype=torch.float32).cuda()
    joint_info_dict['qpos_max'] = torch.zeros(num_nodes, num_nodes, dtype=torch.float32).cuda()
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
        if num_atc == 1:
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
                    if normalize:
                        joint_info_dict['qpos'][p_idx][c_idx] = (joint_info['qpos'] - joint_info['qpos_limit'][0]) / qpos_range
                        assert joint_info_dict['qpos'][p_idx][c_idx] >= 0 and joint_info_dict['qpos'][p_idx][c_idx] <= 1, joint_info_dict['qpos']
                    else:
                        joint_info_dict['qpos'][p_idx][c_idx] = joint_info['qpos'] - joint_info['qpos_limit'][0]
                        
                else:
                    # assert joint_info['type'] == 'revolute' or joint_info['type'] == 'prismatic'
                    print("THIS CANNOT BE HAPPENED")
                    raise NotImplementedError
                
        elif num_atc == 2:
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
                    if normalize:
                        joint_info_dict['qpos'][p_idx][c_idx] = (joint_info['qpos'] - joint_info['qpos_limit'][0]) / qpos_range
                        assert joint_info_dict['qpos'][p_idx][c_idx] >= 0 and joint_info_dict['qpos'][p_idx][c_idx] <= 1, joint_info_dict['qpos']
                    else:
                        #normalize안하면, qpos_min으로 부터 0으로 친다. 
                        joint_info_dict['qpos'][p_idx][c_idx] = joint_info['qpos'] - joint_info['qpos_limit'][0]
                        
                else:
                    # assert joint_info['type'] == 'revolute' or joint_info['type'] == 'prismatic'
                    print("THIS CANNOT BE HAPPENED")
                    raise NotImplementedError
        else:
            raise NotImplementedError
    
        
        # 체크용
        if num_atc == 1:
            assert (joint_info_dict['qpos'][0][1] != 0) or (joint_info_dict['qpos'][1][0] != 0), joint_info_dict['qpos']
        elif num_atc == 2:
            assert ((joint_info_dict['qpos'][0][1] != 0) or (joint_info_dict['qpos'][1][0] != 0)) and ((joint_info_dict['qpos'][0][2] != 0) or (joint_info_dict['qpos'][2][0] != 0)), joint_info_dict['qpos']
    
    
    pc = vertex_array[:, :3]
 
    lbl = vertex_array[:, -1]
    
    if len(pc) < points_num:
        # Padding logic for smaller point clouds
        pc_pad = np.zeros((points_num, 3))
        lbl_pad = np.zeros((points_num))
        pc_pad[:len(pc), :] = pc
        lbl_pad[:len(pc)] = lbl
        pc = torch.from_numpy(pc_pad).float().cuda()
        lbl = torch.from_numpy(lbl_pad).type(torch.int64).cuda()
    else:
        pc = torch.from_numpy(pc).unsqueeze(0).float().cuda()
        lbl = torch.from_numpy(lbl).type(torch.int64).cuda()
        pc = pc.contiguous()

        # 샘플링에 앞서 각 라벨에서 최소 하나의 포인트를 먼저 선택
        unique_labels = torch.unique(lbl)
        selected_indices = []

        for label_ in unique_labels:
            label_indices = torch.where(lbl == label_)[0]
            if len(label_indices) > 0:
                selected_indices.append(label_indices[torch.randint(len(label_indices), (1,))])

        selected_indices = torch.cat(selected_indices)

        # 나머지 포인트는 기존 방식으로 샘플링
        if len(selected_indices) < points_num:
            remaining_points = points_num - len(selected_indices)
            input_pcid = furthest_point_sample(pc, remaining_points).long().reshape(-1)

            # 기존 샘플링 포인트에 포함되지 않도록 필터링
            input_pcid = input_pcid[~torch.isin(input_pcid, selected_indices)]
            selected_indices = torch.cat((selected_indices, input_pcid))

        # 선택된 포인트로 pc와 lbl 업데이트
        pc = pc[:, selected_indices, :].squeeze()
        lbl = lbl[selected_indices]
        print("final lbl", torch.unique(lbl))      
        print("AFTER==========================")
        print("one", len(pc[lbl==1]), "zero", len(pc[lbl==0]), "-100", len(pc[lbl==-100]))
        
    assert set(label.flatten().tolist()) == set(lbl.flatten().tolist()), f"{set(label.flatten().tolist())} , {set(lbl.flatten().tolist())}"

    # assert set(label.flatten().tolist()) == set(lbl.flatten().tolist()), f"{set(label.flatten().tolist())} , {set(lbl.flatten().tolist())}""
    return taxomony_id, model_id, pc.squeeze(), lbl, instance2langemb, joint_info_dict, cloud_path

if __name__ == '__main__':
    for i in range(100):
        print("CNT", i , "==========================")
        t = time.time()
        
        test_func(f'/home/ubuntu/data/projects/pose_data/val/Toaster/103475/pose_{i}/points_with_sdf_label_binary.ply', 1, 10, 256, True, 1024)
        print("time", time.time() - t)