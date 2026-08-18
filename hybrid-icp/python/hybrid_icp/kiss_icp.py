# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
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

from hybrid_icp.config import KISSConfig
from hybrid_icp.mapping import get_voxel_hash_map
from hybrid_icp.preprocess import get_preprocessor
from hybrid_icp.registration import get_registration
from hybrid_icp.threshold import get_threshold_estimator
from hybrid_icp.voxelization import voxel_down_sample
import time
import open3d as o3d

class KissICP:
    def __init__(self, config: KISSConfig):
        self.last_pose = np.eye(4)
        self.last_delta = np.eye(4)
        self.config = config
        self.adaptive_threshold = get_threshold_estimator(self.config)
        self.preprocessor = get_preprocessor(self.config)
        self.registration = get_registration(self.config)
        self.local_map = get_voxel_hash_map(self.config)
        self.count = 0


    def register_frame(self, frame, timestamps):
        # Apply motion compensation
        frame_raw = frame.copy() #nam add
        # print("-----------------------preprocess: deskewing and filtering the frame--------------------------")
        frame = self.preprocessor.preprocess(frame, timestamps, self.last_delta)
        ########## NAM add ######################################################
        # import open3d as o3d

        # pcd_raw = o3d.geometry.PointCloud()
        # pcd_raw.points = o3d.utility.Vector3dVector(frame_raw)
        # pcd_raw.paint_uniform_color([1, 0, 0])  # đỏ

        # pcd_proc = o3d.geometry.PointCloud()
        # pcd_proc.points = o3d.utility.Vector3dVector(frame)
        # pcd_proc.paint_uniform_color([0, 1, 0])  # xanh

        # pcd_proc.translate((150, 0, 0))
        # o3d.visualization.draw_geometries([pcd_raw, pcd_proc],
        #                                 window_name="Red: Raw, Green: Processed")
        # print("Shape raw:", frame_raw.shape)
        # print("Shape processed:", frame.shape)
        # time.sleep(2)
        ########################################################################

        # Voxelize
        # print("-----------------------voxelization: downsample the frame--------------------------")
        source, frame_downsample = self.voxelize(frame)
        # print("Shape source:", source.shape)
        # print("Shape frame_downsample:", frame_downsample.shape)
        # time.sleep(2)

        # Get adaptive_threshold
        # print("-----------------------adaptive_threshold: get adaptive threshold--------------------------")

        sigma = self.adaptive_threshold.get_threshold()
        # print("Adaptive threshold (sigma) shape:", sigma.shape)
        # print("Adaptive threshold (sigma):", sigma)
        # time.sleep(2)
        # Compute initial_guess for ICP
        # print("-----------------------initial_guess: compute initial guess for ICP--------------------------")
        initial_guess = self.last_pose @ self.last_delta
        # print("Initial guess shape:", initial_guess.shape)
        # print("Initial guess:", initial_guess)

        # Run ICP
        # print("-----------------------registration: align points to map--------------------------")
        new_pose = self.registration.align_points_to_map(
            points=source,
            voxel_map=self.local_map,
            initial_guess=initial_guess,
            max_correspondance_distance=3 * sigma,
            kernel=sigma,
        )
        # print("New pose shape:", new_pose.shape)
        # print("New pose:", new_pose)

        # Compute the difference between the prediction and the actual estimate
        # print("-----------------------model_deviation: compute model deviation--------------------------")
        model_deviation = np.linalg.inv(initial_guess) @ new_pose
        # print("Model deviation shape:", model_deviation.shape)
        # print("Model deviation:", model_deviation)

        # Update step: threshold, local map, delta, and the last pose
        # print("-----------------------update: update threshold, local map, delta, and last pose--------------------------")
        self.adaptive_threshold.update_model_deviation(model_deviation)
        self.local_map.update(frame_downsample, new_pose)
        # visualize the local map ############################################################
        # if self.count % 100 == 0:  # visualize every 100th frame
        #     local_map_points = self.local_map.point_cloud()  # hoặc .Pointcloud(), tùy pybind

        #     pcd_map = o3d.geometry.PointCloud()
        #     pcd_map.points = o3d.utility.Vector3dVector(local_map_points)
        #     pcd_map.paint_uniform_color([0, 0.7, 1])  # màu cyan nhẹ

        #     o3d.visualization.draw_geometries([pcd_map], window_name="Current Local Map")
        ######################################################################################
        self.last_delta = np.linalg.inv(self.last_pose) @ new_pose #phép biến đổi (SE3) từ frame trước đến frame hiện tại, hay chính là ước lượng chuyển động (relative motion) giữa hai frame.
        self.last_pose = new_pose
        self.count += 1

        # Return the (deskew) input raw scan (frame) and the points used for registration (source)
        return frame, source

    def voxelize(self, iframe):
        frame_downsample = voxel_down_sample(iframe, self.config.mapping.voxel_size * 0.5)
        source = voxel_down_sample(frame_downsample, self.config.mapping.voxel_size * 1.5)
        return source, frame_downsample
