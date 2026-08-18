# MIT License

# Copyright (c) 2025 Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta, Cyrill
# Stachniss.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np
from hybrid_icp.kiss_icp import KissICP
from hybrid_icp.voxelization import voxel_down_sample

from hp2_slam.config import KissSLAMConfig
from hp2_slam.local_map_graph import LocalMapGraph
from hp2_slam.loop_closer import LoopCloser
from hp2_slam.pose_graph_optimizer import PoseGraphOptimizer
from hp2_slam.voxel_map import VoxelMap


def transform_points(pcd, T):
    R = T[:3, :3]
    t = T[:3, -1]
    return pcd @ R.T + t


class KissSLAM:
    def __init__(self, config: KissSLAMConfig):
        self.config = config
        self.odometry = KissICP(config.kiss_icp_config())
        self.closer = LoopCloser(config.loop_closer)
        local_map_config = self.config.local_mapper
        self.local_map_voxel_size = local_map_config.voxel_size
        self.voxel_grid = VoxelMap(self.local_map_voxel_size)
        self.local_map_graph = LocalMapGraph()
        self.local_map_splitting_distance = local_map_config.splitting_distance
        self.optimizer = PoseGraphOptimizer(config.pose_graph_optimizer)
        self.optimizer.add_variable(self.local_map_graph.last_id, self.local_map_graph.last_keypose)
        self.optimizer.fix_variable(self.local_map_graph.last_id)
        self.closures = []

    def get_closures(self):
        return self.closures

    def get_keyposes(self):
        return list(self.local_map_graph.keyposes())

    def process_scan(self, frame, timestamps):
        deskewed_frame, _ = self.odometry.register_frame(frame, timestamps)
        current_pose = self.odometry.last_pose
        mapping_frame = voxel_down_sample(deskewed_frame, self.local_map_voxel_size)
        self.voxel_grid.integrate_frame(mapping_frame, current_pose)
        self.local_map_graph.last_local_map.local_trajectory.append(current_pose)
        traveled_distance = np.linalg.norm(current_pose[:3, -1])
        if traveled_distance > self.local_map_splitting_distance:
            self.generate_new_node()

    def compute_closures(self, query_id, query):
        is_good, source_id, target_id, pose_constraint = self.closer.compute(
            query_id, query, self.local_map_graph
        )
        if is_good:
            self.closures.append((source_id, target_id))
            self.optimizer.add_factor(source_id, target_id, pose_constraint, np.eye(6))
            self.optimize_pose_graph()

    def optimize_pose_graph(self):
        self.optimizer.optimize()
        estimates = self.optimizer.estimates()
        for id_, pose in estimates.items():
            self.local_map_graph[id_].keypose = np.copy(pose)

    def generate_new_node(self):
        points = self.odometry.local_map.point_cloud()
        # Reset odometry
        last_local_map = self.local_map_graph.last_local_map
        relative_motion = last_local_map.local_trajectory[-1]
        inverse_relative_motion = np.linalg.inv(relative_motion)
        transformed_local_map = transform_points(points, inverse_relative_motion)

        self.odometry.local_map.clear()
        self.odometry.local_map.add_points(transformed_local_map)
        self.odometry.last_pose = np.eye(4)

        query_id = last_local_map.id
        query_points = self.voxel_grid.point_cloud()
        self.local_map_graph.finalize_local_map(self.voxel_grid)
        self.voxel_grid.clear()
        self.voxel_grid.add_points(transformed_local_map)
        self.optimizer.add_variable(self.local_map_graph.last_id, self.local_map_graph.last_keypose)
        self.optimizer.add_factor(
            self.local_map_graph.last_id, query_id, relative_motion, np.eye(6)
        )
        self.compute_closures(query_id, query_points)

    @property
    def poses(self):
        poses = [np.eye(4)]
        for node in self.local_map_graph.local_maps():
            for rel_pose in node.local_trajectory[1:]:
                poses.append(node.keypose @ rel_pose)
        return poses

    def fine_grained_optimization(self):
        pgo = PoseGraphOptimizer(self.config.pose_graph_optimizer)
        id_ = 0
        pgo.add_variable(id_, self.local_map_graph[id_].keypose)
        pgo.fix_variable(id_)
        for node in self.local_map_graph.local_maps():
            odometry_factors = [
                np.linalg.inv(T0) @ T1
                for T0, T1 in zip(node.local_trajectory[:-1], node.local_trajectory[1:])
            ]
            for i, factor in enumerate(odometry_factors):
                pgo.add_variable(id_ + 1, node.keypose @ node.local_trajectory[i + 1])
                pgo.add_factor(id_ + 1, id_, factor, np.eye(6))
                id_ += 1
            pgo.fix_variable(id_ - 1)

        pgo.optimize()
        poses = [x for x in pgo.estimates().values()]
        return poses, pgo
