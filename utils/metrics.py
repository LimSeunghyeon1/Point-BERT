# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-08 14:31:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-05-25 09:13:32
# @Email:  cshzxie@gmail.com

import logging
import open3d

from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import torch.nn.functional as F
import torch

'''
NOTICE articulation project 관련 metric 계산은 모두 src/articulation_metrics.py 참고
여기에 있는 관련 items는 모두 deprecated이므로 사용하지 마시오.

'''

class Metrics(object):
    ITEMS = [{
        'name': 'F-Score',
        'enabled': True,
        'eval_func': 'cls._get_f_score',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'CDL1',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distancel1',
        'eval_object': ChamferDistanceL1(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'CDL2',
        'enabled': True, # mpn 버전에서는 다 disabled
        'eval_func': 'cls._get_chamfer_distancel2',
        'eval_object': ChamferDistanceL2(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'Language_ACC',
        'enabled': False, 
        'eval_func': 'cls._get_language_acc',
        'is_greater_better': True,
        'init_value': 0
    }, {
       'name': 'Edge_Conf_ACC',
        'enabled': False, 
        'eval_func': 'cls._get_edge_conf_acc',
        'is_greater_better': True,
        'init_value': 0
        
    }, {
        'name': 'Edge_Type_ACC',
        'enabled': False, 
        'eval_func': 'cls._get_edge_type_acc',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'Qpos_Angle_Diff', 
        'enabled': False, 
        'eval_func': 'cls._get_angle_diff',
        'is_greater_better': False,
        'init_value': 0
    }, {
        'name': 'Qpos_Trans_Diff', #ground truth와 10cm 이하로 차이 나는 joint 비율 (translation)
        'enabled': False, # mpn 버전에서는 다 disabled
        'eval_func': 'cls._get_trans_diff',
        'is_greater_better': False,
        'init_value': 0
    },
    
        {
        'name': 'Qpos_Angle_Acc_10', 
        'enabled': False, 
        'eval_func': 'cls._get_angle_acc_10',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'Qpos_Trans_Acc_0.1', #ground truth와 10cm 이하로 차이 나는 joint 비율 (translation)
        'enabled': False, 
        'eval_func': 'cls._get_trans_acc_01',
        'is_greater_better': True,
        'init_value': 0
    }]

    @classmethod
    def get(cls, pred, gt):
        _items = cls.items()
        _values = [0] * len(_items)
        for i, item in enumerate(_items):
            eval_func = eval(item['eval_func'])
            _values[i] = eval_func(pred, gt)

        return _values

    # bi articulation project version
    @classmethod
    def get_articulation(cls, metrics_dict):
        _items = cls.items()
        _values = [0] * len(_items)
        for i, item in enumerate(_items):
            eval_func = eval(item['eval_func'])
            arg = metrics_dict[item['name']]
            _values[i] = eval_func(arg[0], arg[1], arg[2])

        return _values
    

    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i['enabled']]

    @classmethod
    def names(cls):
        _items = cls.items()
        return [i['name'] for i in _items]

    @classmethod
    def _get_language_acc(cls, pred_out, target_out, mask, atol=1e-6):
        assert len(target_out.shape) == 3, target_out.shape
        # target_out B, num_nodes, 768
        if len(pred_out.shape) == 2:
            pred_out = pred_out.view_as(target_out)
        
        # 벡터 정규화 (크기 1로)
        pred_norm = F.normalize(pred_out, p=2, dim=2)  # a의 각 행 벡터를 정규화
        target_norm = F.normalize(target_out, p=2, dim=2)  # b의 각 행 벡터를 정규화

        # 코사인 유사도 계산 (내적)
        cosine_similarity = torch.matmul(pred_norm, target_norm.transpose(1, 2))  # shape: (batch_size, num_nodes, num_nodes)
        cs_argmax = torch.argmax(cosine_similarity, dim=-1)
        # 정확도 계산
        batch_size, num_nodes, _ = target_out.shape

        # 마스크 계산
        # mask = (target_out != 0).any(dim=-1, keepdim=True).float()

        # 각 배치의 각 행 벡터에 대한 위치를 추적
        target_flat = target_out.view(batch_size, num_nodes, -1)
        pred_flat = torch.gather(target_flat, 1, cs_argmax.unsqueeze(-1).expand(-1, -1, target_flat.size(-1)))

        # 허용 오차 내에서 요소별 비교
        l2_distances = torch.sqrt(torch.sum((pred_flat - target_flat) ** 2, dim=-1) + 1e-8)
        correct = l2_distances <= atol
        
        # 마스크 적용
        masked_correct = correct * mask.squeeze(-1)
        accuracy = masked_correct.sum().item() / (mask.sum().item() + 1e-8)
        # accuracy = correct.type(torch.float32).sum().item() / correct.numel()

        return accuracy
        
    
    @classmethod
    def _get_edge_conf_acc(cls, pred_out, target_out, mask=None):
        # pred_out: B num_nodes num_nodes 
        if len(target_out.shape) == 4:
            assert target_out.shape[1] == 1
            target_out = target_out.squeeze(1)
        
        # Ensure pred_out and target_out are of the same dtype
        pred_out = pred_out.to(target_out.dtype)

        correct = (pred_out == target_out)
        
        
        accuracy = correct.sum().item() / correct.numel()
        return accuracy
        
    
    @classmethod
    def _get_edge_type_acc(cls, pred_out, target_out, mask):
        #pred_out: B num_nodes num_nodes 2
        if len(target_out.shape) == 5:
            assert target_out.shape[1] == 1
            target_out = target_out.squeeze(1)
        
        # Ensure pred_out and target_out are of the same dtype
        pred_out = pred_out.to(target_out.dtype)
        
        pred_out = torch.argmax(pred_out, dim=-1)
        assert mask.shape == pred_out.shape, f"mask shape: {mask.shape}, pred_out shape: {pred_out.shape}"
        
        target_out = torch.argmax(target_out, dim=-1)
        
        correct = (pred_out == target_out)
        masked_correct = correct * mask

        accuracy = masked_correct.sum().item() / (mask.sum().item() + 1e-8)
        return accuracy
    
    @classmethod
    def _get_angle_diff(cls):
        pass
    
    @classmethod
    def _get_trans_diff(cls):
        pass
        
    
    @classmethod
    def _get_angle_acc_10(cls):
        pass
    
    @classmethod
    def _get_trans_acc_01(cls):
        pass

    @classmethod
    def _get_f_score(cls, pred, gt, th=0.01):

        """References: https://github.com/lmb-freiburg/what3d/blob/master/util.py"""
        b = pred.size(0)
        assert pred.size(0) == gt.size(0)
        if b != 1:
            f_score_list = []
            for idx in range(b):
                f_score_list.append(cls._get_f_score(pred[idx:idx+1], gt[idx:idx+1]))
            return sum(f_score_list)/len(f_score_list)
        else:
            pred = cls._get_open3d_ptcloud(pred)
            gt = cls._get_open3d_ptcloud(gt)

            dist1 = pred.compute_point_cloud_distance(gt)
            dist2 = gt.compute_point_cloud_distance(pred)

            recall = float(sum(d < th for d in dist2)) / float(len(dist2))
            precision = float(sum(d < th for d in dist1)) / float(len(dist1))
            return 2 * recall * precision / (recall + precision) if recall + precision else 0

    @classmethod
    def _get_open3d_ptcloud(cls, tensor):
        """pred and gt bs is 1"""
        tensor = tensor.squeeze().cpu().numpy()
        ptcloud = open3d.geometry.PointCloud()
        ptcloud.points = open3d.utility.Vector3dVector(tensor)

        return ptcloud

    @classmethod
    def _get_chamfer_distancel1(cls, pred, gt):
        chamfer_distance = cls.ITEMS[1]['eval_object']
        return chamfer_distance(pred, gt).item() * 1000

    @classmethod
    def _get_chamfer_distancel2(cls, pred, gt):
        chamfer_distance = cls.ITEMS[2]['eval_object']
        return chamfer_distance(pred, gt).item() * 1000

    def __init__(self, metric_name, values):
        self._items = Metrics.items()
        self._values = [item['init_value'] for item in self._items]
        self.metric_name = metric_name

        if type(values).__name__ == 'list':
            self._values = values
        elif type(values).__name__ == 'dict':
            metric_indexes = {}
            for idx, item in enumerate(self._items):
                item_name = item['name']
                metric_indexes[item_name] = idx
            for k, v in values.items():
                if k not in metric_indexes:
                    logging.warn('Ignore Metric[Name=%s] due to disability.' % k)
                    continue
                self._values[metric_indexes[k]] = v
        else:
            raise Exception('Unsupported value type: %s' % type(values))

    def state_dict(self):
        _dict = dict()
        for i in range(len(self._items)):
            item = self._items[i]['name']
            value = self._values[i]
            _dict[item] = value

        return _dict

    def __repr__(self):
        return str(self.state_dict())

    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(self._items):
            if _item['name'] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception('Invalid metric name to compare.')

        _metric = self._items[i]
        _value = self._values[_index]
        other_value = other._values[_index]
        return _value > other_value if _metric['is_greater_better'] else _value < other_value
