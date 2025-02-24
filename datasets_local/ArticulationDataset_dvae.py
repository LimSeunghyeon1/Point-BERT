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
    def __init__(self, split, points_num, dirpath, data_split_file, normalize, real_world):
        self.split  = split
        self.dirpath = dirpath
        self.data_split_file = data_split_file
        self.real_world = real_world
        
        self.valid_data = self._load_data()
        self.points_num = points_num
        self.normalize = normalize
        
    def __getitem__(self, index):
        cloud_path = self.valid_data[index]
        instance_path = '/'.join(cloud_path.split('/')[:-2])
        # print("cloud_path", cloud_path)
        instance_pose_path = '/'.join(cloud_path.split('/')[:-1])
        with open(os.path.join(instance_path, 'link_cfg.json'), 'r') as f:
            instance_pose_json = json.load(f)
        model_id = instance_path.split('/')[-1]
        assert model_id.isdigit(), model_id
        model_id = int(model_id)
        # faucet인지 아닌지 체크용
        for instance_pose_dict in instance_pose_json.values():
            if instance_pose_dict['index'] == 0:
                # assert instance_pose_dict['name'] == 'Faucet' , instance_pose_dict #HARDCODED
                taxomony_id = instance_pose_dict['name']
        
        if self.real_world:
            data = np.load(cloud_path, allow_pickle=True) #"pc", "rgb", "label"
            pc = data["pc"]
            rgb = data["rgb"]
            label = data["label"] - 1
            points = np.concatenate((pc, rgb, np.expand_dims(label, axis=-1)), axis=-1)
            points = points[~np.any(np.isnan(points), axis=-1)]
            
            
            
            points = points.reshape(-1, 7)
            
            
            points = points[points[:, -1] >= 0]
            
            # pc, rgb, label 분리
            pc = points[:, :3]
            rgb = points[:, 3:6]
            lbl = points[:, -1]

            # 고유한 라벨 추출
            unique_labels = np.unique(lbl)

            # 최종 마스크 초기화 (모든 포인트를 제외 상태로 시작)
            final_mask = np.zeros(points.shape[0], dtype=bool)

            # 각 라벨별로 필터링 수행
            for label in unique_labels:
                # 현재 라벨에 해당하는 포인트 인덱스
                label_mask = (lbl == label)
                
                # 해당 라벨의 pc 데이터 추출
                pc_label = pc[label_mask]
                
                # 라벨별 평균 및 표준편차 계산
                pc_mean = np.mean(pc_label, axis=0)
                pc_std = np.std(pc_label, axis=0)
                
                # 표준편차 1.5배를 초과하지 않는 포인트 마스크
                within_std_mask = np.all(np.abs(pc_label - pc_mean) <= 1.5 * pc_std, axis=1)
                
                # 최종 마스크에 반영
                final_mask[label_mask] = within_std_mask

            # 마스크를 적용하여 필터링된 포인트 얻기
            filtered_points = points[final_mask]

            # 필터링된 포인트의 최소 label이 0인지 확인
            if not self.real_world:
                assert filtered_points[:, -1].min() == 0, "최소 라벨 값이 0이 아닙니다."

            if self.split == 'trn':
                #unorganized로 바꿈
                shuf = list(range(len(filtered_points)))
                random.shuffle(shuf)
                filtered_points = filtered_points[shuf]
            
            # pc, rgb, lbl 재분리
            pc = filtered_points[:, :3]
            rgb = filtered_points[:, 3:6]
            lbl = filtered_points[:, -1]

            
            if self.normalize:
                pc = self.pc_norm(pc)
            assert np.unique(lbl).min() >= 0
            
            
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
                
        
        else:
            # Ply 파일 읽기
            ply_data = PlyData.read(cloud_path)
            # Vertex 데이터 추출
            vertex_data = ply_data['vertex']

            # real world에서는 sdf 없음 아직까지는
            x = vertex_data['x']
            y = vertex_data['y']
            z = vertex_data['z']
            label = vertex_data['label'] - 1
            
            
            # 필요한 속성 추출
            sdf = vertex_data['sdf']
            # Numpy array로 변환
            vertex_array = np.vstack((x, y, z, sdf, label)).T
            # remain only negative sdf
            vertex_array = vertex_array[vertex_array[...,-2] < 0]
            

            assert vertex_array[...,-1].min() == 0 # 0은 없었다고 가정
            
            if self.split == 'trn':
                #unorganized로 바꿈
                shuf = list(range(len(vertex_array)))
                random.shuffle(shuf)
                vertex_array = vertex_array[shuf]
            
            pc = vertex_array[:, :3]
            if self.normalize:
                pc = self.pc_norm(pc)
            lbl = vertex_array[:, -1]
            assert np.unique(lbl).min() >= 0
            
            
            
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
        return taxomony_id, model_id, pc.squeeze(), lbl
        
    def __len__(self):
        return len(self.valid_data) 
        
    def _load_data(self):
        total_valid_paths = []
        dir = self.dirpath
        
        if not self.real_world:
            #validity check
            check_data = np.load(self.data_split_file, allow_pickle=True)

        for dirpath, dirname, filenames in os.walk(dir):
            data_label = dirpath.split('/')[-1]
            for filename in filenames:
                if self.real_world:
                    if filename == 'traj.pkl':
                        spt, cat, inst = dirpath.split('/')[-4:-1]
                        inst = int(inst)
                        total_valid_paths.append(os.path.join(dirpath, filename))
                else:
                    if filename == 'points_with_sdf_label_binary.ply':
                        spt, cat, inst = dirpath.split('/')[-4:-1]
                        inst = int(inst)
                        if not self.real_world:
                            assert check_data[inst] == [cat, spt], f"{inst}, {cat}, {spt}, answer: {check_data[inst]}"
                        total_valid_paths.append(os.path.join(dirpath, filename))
            # if data_label.split('_')[0].isdigit():
            #     total_valid_paths.append(dirpath)
        print("TOTAL VALID PATH", len(total_valid_paths))
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
            if dirpath.split('/')[-1].split('_')[0] == 'pose' and dirpath.split('/')[-1].split('_')[1].isdigit():
                assert os.path.isfile(os.path.hjoin(dirpath), 'points_with_sdf_label.ply'), dirpath
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