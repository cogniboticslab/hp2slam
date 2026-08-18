# MIT License
#
# Copyright (c) 2025 Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np
import open3d as o3d

from hp2_slam.hp2_slam_pybind import kiss_slam_pybind


class VoxelMap:
    def __init__(self, voxel_size: float):
        self.map = kiss_slam_pybind._VoxelMap(voxel_size)

    def integrate_frame(self, points: np.ndarray, pose: np.ndarray):
        vector3fvector = kiss_slam_pybind._Vector3fVector(points.astype(np.float32))
        self.map._integrate_frame(vector3fvector, pose)

    def add_points(self, points: np.ndarray):
        vector3fvector = kiss_slam_pybind._Vector3fVector(points.astype(np.float32))
        self.map._add_points(vector3fvector)

    def point_cloud(self):
        return np.asarray(self.map._point_cloud()).astype(np.float64)

    def clear(self):
        self.map._clear()

    def num_voxels(self):
        return self.map._num_voxels()

    def open3d_pcd_with_normals(self):
        points, normals = self.map._per_voxel_point_and_normal()
        # Reduce memory footprint
        pcd = o3d.t.geometry.PointCloud()
        pcd.point.positions = o3d.core.Tensor(np.asarray(points), o3d.core.Dtype.Float32)
        pcd.point.normals = o3d.core.Tensor(np.asarray(normals), o3d.core.Dtype.Float32)
        return pcd
