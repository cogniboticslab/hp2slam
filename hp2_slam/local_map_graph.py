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
from copy import deepcopy as copy

import numpy as np
import open3d as o3d

from hp2_slam.voxel_map import VoxelMap


class LocalMap:
    def __init__(self, id: np.uint64, keypose: np.ndarray):
        self.id = id
        self.keypose = keypose
        self.local_trajectory = [np.eye(4)]
        self.pcd = None

    @property
    def endpose(self):
        return self.keypose @ self.local_trajectory[-1]

    def write(self, filename):
        local_map_pcd = copy(self.pcd)
        local_map_pcd.transform(self.keypose)
        o3d.t.io.write_point_cloud(filename, local_map_pcd)


class LocalMapGraph:
    def __init__(self):
        self.graph = dict()
        local_map0 = LocalMap(id=0, keypose=np.eye(4))
        local_map0.local_trajectory.clear()
        self.graph[0] = local_map0

    def __getitem__(self, key):
        return self.graph[key]

    def local_maps(self):
        for local_map in self.graph.values():
            yield local_map

    def keyposes(self):
        for local_map in self.graph.values():
            yield local_map.keypose

    @property
    def last_id(self):
        last_id = next(reversed(self.graph))
        return last_id

    @property
    def last_local_map(self):
        return self.graph[self.last_id]

    @property
    def last_keypose(self):
        return self.last_local_map.keypose

    def erase_local_map(self, key: np.uint64):
        self.graph.pop(key)

    def erase_last_local_map(self):
        self.erase_local_map(self.last_id)

    def finalize_local_map(self, voxel_grid: VoxelMap):
        local_map = self.last_local_map
        local_map.pcd = voxel_grid.open3d_pcd_with_normals()
        proto_id = local_map.id + 1
        proto_keypose = local_map.endpose
        new_local_map = LocalMap(
            proto_id,
            np.copy(proto_keypose),
        )
        self.graph[new_local_map.id] = new_local_map
