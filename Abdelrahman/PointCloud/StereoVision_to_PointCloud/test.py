import open3d as o3d
import numpy as np

PATH = 'pointCloud.ply'

def visualize_point_cloud(PATH):
    pcd = o3d.io.read_point_cloud(PATH)
    print(pcd)
    pcd_np_arr = np.asarray(pcd.points)
    print(pcd_np_arr.shape)
    print(pcd_np_arr[0])
    o3d.visualization.draw_geometries([pcd])

