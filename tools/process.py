import open3d as o3d
import numpy as np

def preprocess_point_cloud(pcd):
    """Preprocess point cloud to convert float color values to integers."""
    colors = np.asarray(pcd.colors)
    labels = np.asarray(pcd.point['label'])
    xyz = np.asarray(pcd.points)
    
    return xyz, colors, labels

# Load the PLY file
input_path = '/data/models/Point-BERT/where2act_dataset/faucet_0724/val/1788/pose_0/full_point_cloud.ply'

xyz, colors, labels = o3d.io.read_point_cloud(input_path, format='ply')
print(xyz.shape, colors.shape, labels.shape)
