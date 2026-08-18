# # MIT License
# #
# # Copyright (c) 2025 Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta, Cyrill
# # Stachniss.
# #
# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to deal
# # in the Software without restriction, including without limitation the rights
# # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to the following conditions:
# #
# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.
import numpy as np
import open3d as o3d
from map_closures.map_closures import MapClosures

from hp2_slam.config import LoopCloserConfig
from hp2_slam.local_map_graph import LocalMapGraph
from hp2_slam.voxel_map import VoxelMap


class LoopCloser:
    def __init__(self, config: LoopCloserConfig):
        self.config = config
        self.detector = MapClosures(config.detector)
        self.local_map_voxel_size = config.detector.density_map_resolution
        self.icp_threshold = np.sqrt(3) * self.local_map_voxel_size
        self.icp_algorithm = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
        self.termination_criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
            relative_rmse=1e-4
        )
        self.overlap_threshold = config.overlap_threshold

    def compute(self, query_id, points, local_map_graph: LocalMapGraph):
        closure = self.detector.get_best_closure(query_id, points)
        is_good = False
        ref_id = -1
        pose_constraint = np.eye(4)
        if closure.number_of_inliers >= self.config.detector.inliers_threshold:
            ref_id = closure.source_id
            source = local_map_graph[ref_id].pcd
            target = local_map_graph[query_id].pcd
            print("\nKissSLAM| Closure Detected")
            is_good, pose_constraint = self.validate_closure(source, target, closure.pose)
        return is_good, ref_id, query_id, pose_constraint

    # This is the thing that takes the most time
    def validate_closure(self, source, target, initial_guess):
        registration_result = o3d.t.pipelines.registration.icp(
            source,
            target,
            self.icp_threshold,
            initial_guess,
            self.icp_algorithm,
            self.termination_criteria,
        )
        union_map = VoxelMap(self.local_map_voxel_size)
        source_pts = source.point.positions.numpy().astype(np.float64)
        target_pts = target.point.positions.numpy().astype(np.float64)
        pose = registration_result.transformation.numpy()
        union_map.integrate_frame(source_pts, pose)
        num_source_voxels = union_map.num_voxels()
        num_target_voxels = len(target_pts)
        union_map.add_points(target_pts)
        union = union_map.num_voxels()
        intersection = num_source_voxels + num_target_voxels - union
        overlap = intersection / np.min([num_source_voxels, num_target_voxels])
        closure_is_accepted = overlap > self.overlap_threshold
        print(f"KissSLAM| LocalMaps Overlap: {overlap}")
        if closure_is_accepted:
            print("KissSLAM| Closure Accepted")
        else:
            print(f"KissSLAM| Closure rejected for low overlap.")
        return closure_is_accepted, pose

# MIT License
#
# (c) 2025 Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta, Cyrill Stachniss.
# Mod by Nam's assistant: add 3D visualization only (no BEV), offset mode, PLY dump.

# import os
# import copy
# import numpy as np
# import open3d as o3d

# from map_closures.map_closures import MapClosures
# from kiss_slam.config import LoopCloserConfig
# from kiss_slam.local_map_graph import LocalMapGraph
# from kiss_slam.voxel_map import VoxelMap

# class LoopCloser:
#     """
#     Loop-closure validator + 3D visualization (Open3D only).

#     Args:
#         config: LoopCloserConfig
#         visualize: bool, bật chế độ xem 3D
#         out_dir: str | None, nếu set thì lưu PLY các submap sau ICP
#         display_mode: "overlay" | "offset"
#             - "overlay": source(init/refined) chồng lên target
#             - "offset" : dịch init sang -x, refined sang +x để không đè nhau
#         offset_m: float, khoảng dịch mỗi phía nếu display_mode="offset"
#         downsample_voxel: float | None, voxel size (m) để giảm điểm khi visualize
#         max_points: int | None, giới hạn số điểm (random) để hiển thị cho mượt
#     """
#     def __init__(self,
#                  config: LoopCloserConfig,
#                  visualize: bool = True,
#                  out_dir: str | None = None,
#                  display_mode: str = "overlay",
#                  offset_m: float = 15.0,
#                  downsample_voxel: float | None = None,
#                  max_points: int | None = 800_000):
#         self.config = config
#         self.detector = MapClosures(config.detector)
#         self.local_map_voxel_size = config.detector.density_map_resolution
#         self.icp_threshold = np.sqrt(3) * self.local_map_voxel_size
#         self.icp_algorithm = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
#         self.termination_criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
#             relative_rmse=1e-4
#         )
#         self.overlap_threshold = config.overlap_threshold

#         # Visualization options
#         self.visualize = visualize
#         self.out_dir = out_dir
#         self.display_mode = display_mode
#         self.offset_m = offset_m
#         self.downsample_voxel = downsample_voxel
#         self.max_points = max_points

#         if self.out_dir is not None:
#             os.makedirs(self.out_dir, exist_ok=True)

#     def compute(self, query_id, points, local_map_graph: LocalMapGraph):
#         closure = self.detector.get_best_closure(query_id, points)
#         is_good = False
#         ref_id = -1
#         pose_constraint = np.eye(4)

#         if closure.number_of_inliers >= self.config.detector.inliers_threshold:
#             ref_id = closure.source_id
#             source = local_map_graph[ref_id].pcd    # o3d.t.geometry.PointCloud
#             target = local_map_graph[query_id].pcd  # o3d.t.geometry.PointCloud
#             print("\nKissSLAM| Closure Detected (ref={}, query={}, inliers={})"
#                   .format(ref_id, query_id, closure.number_of_inliers))

#             is_good, pose_constraint, overlap = self.validate_closure(source, target, closure.pose)

#             if self.visualize:
#                 self._visualize_3d(ref_id, query_id, source, target,
#                                    T_init=closure.pose, T_refined=pose_constraint,
#                                    overlap=overlap)
#         return is_good, ref_id, query_id, pose_constraint

#     def validate_closure(self, source, target, initial_guess):
#         """
#         ICP refine + overlap check using VoxelMap occupancy.
#         Returns: (accepted: bool, T_refined: (4,4), overlap: float)
#         """
#         registration_result = o3d.t.pipelines.registration.icp(
#             source,
#             target,
#             self.icp_threshold,
#             initial_guess,
#             self.icp_algorithm,
#             self.termination_criteria,
#         )
#         union_map = VoxelMap(self.local_map_voxel_size)
#         source_pts = source.point.positions.numpy().astype(np.float64)
#         target_pts = target.point.positions.numpy().astype(np.float64)

#         pose = registration_result.transformation.numpy()
#         union_map.integrate_frame(source_pts, pose)
#         num_source_voxels = union_map.num_voxels()
#         num_target_voxels = len(target_pts)
#         union_map.add_points(target_pts)
#         union = union_map.num_voxels()
#         intersection = num_source_voxels + num_target_voxels - union
#         overlap = intersection / np.minimum(num_source_voxels, num_target_voxels)

#         accepted = overlap > self.overlap_threshold
#         print(f"KissSLAM| LocalMaps Overlap: {overlap:.3f} (thr={self.overlap_threshold})")
#         print("KissSLAM| Closure {}".format("Accepted" if accepted else "Rejected"))
#         return accepted, pose, overlap

#     # ------------------------- Visualization (3D only) -------------------------

#     @staticmethod
#     def _tensor_to_legacy(pcd_t) -> o3d.geometry.PointCloud:
#         """o3d.t.geometry.PointCloud -> legacy PointCloud"""
#         pts = pcd_t.point.positions.numpy()
#         return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

#     @staticmethod
#     def _random_thin(pc_legacy: o3d.geometry.PointCloud, max_points: int | None):
#         if max_points is None: return pc_legacy
#         n = np.asarray(pc_legacy.points).shape[0]
#         if n <= max_points: return pc_legacy
#         idx = np.random.choice(n, size=max_points, replace=False)
#         return pc_legacy.select_by_index(idx)

#     @staticmethod
#     def _voxel_down(pc_legacy: o3d.geometry.PointCloud, voxel: float | None):
#         if voxel is None or voxel <= 0: return pc_legacy
#         return pc_legacy.voxel_down_sample(voxel)

#     def _visualize_3d(self, ref_id: int, query_id: int,
#                       source_t, target_t,
#                       T_init: np.ndarray, T_refined: np.ndarray,
#                       overlap: float):

#         # Convert to legacy clouds
#         tgt = self._tensor_to_legacy(target_t)
#         src = self._tensor_to_legacy(source_t)

#         # Optional downsample/thin for speed
#         tgt = self._voxel_down(tgt, self.downsample_voxel)
#         src = self._voxel_down(src, self.downsample_voxel)
#         tgt = self._random_thin(tgt, self.max_points)
#         src = self._random_thin(src, self.max_points)

#         # Paint colors
#         tgt.paint_uniform_color([0.6, 0.6, 0.6])   # gray
#         src_init = copy.deepcopy(src); src_init.paint_uniform_color([1.0, 0.2, 0.2])  # red
#         src_ref  = copy.deepcopy(src); src_ref.paint_uniform_color([0.2, 0.6, 1.0])   # blue

#         # Offset transforms (if chosen)
#         T_off_I = np.eye(4)
#         T_off_L = np.eye(4); T_off_L[0, 3] = -self.offset_m
#         T_off_R = np.eye(4); T_off_R[0, 3] = +self.offset_m
#         use_offset = (self.display_mode == "offset")

#         # Apply transforms
#         src_init.transform(T_init @ (T_off_L if use_offset else T_off_I))
#         src_ref.transform (T_refined @ (T_off_R if use_offset else T_off_I))

#         # Axes
#         axes_tgt      = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
#         axes_src_init = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
#         axes_src_ref  = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
#         axes_src_init.transform(T_init    @ (T_off_L if use_offset else T_off_I))
#         axes_src_ref.transform (T_refined @ (T_off_R if use_offset else T_off_I))

#         title = f"Loop {ref_id} -> {query_id} | overlap={overlap:.3f} | mode={self.display_mode}"
#         print("KissSLAM| Visualizing 3D:", title)

#         try:
#             o3d.visualization.draw_geometries(
#                 [tgt, src_init, src_ref, axes_tgt, axes_src_init, axes_src_ref],
#                 window_name=title, width=1280, height=800
#             )
#         except Exception as e:
#             print("KissSLAM| Open3D viewer failed:", e)

#         # Optional: save PLYs for offline inspection
#         if self.out_dir is not None:
#             try:
#                 o3d.io.write_point_cloud(
#                     os.path.join(self.out_dir, f"loop_{ref_id}_{query_id}_target.ply"),
#                     tgt, write_ascii=False
#                 )
#                 o3d.io.write_point_cloud(
#                     os.path.join(self.out_dir, f"loop_{ref_id}_{query_id}_source_refined.ply"),
#                     src_ref, write_ascii=False
#                 )
#                 print("KissSLAM| Saved PLYs ->", self.out_dir)
#             except Exception as e:
#                 print("KissSLAM| Save PLY failed:", e)
