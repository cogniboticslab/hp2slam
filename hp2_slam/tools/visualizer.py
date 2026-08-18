# # MIT License
# #
# # Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# # Stachniss.


import importlib
import os
import datetime
from abc import ABC
from functools import partial
from typing import Callable, List

import numpy as np
import cv2  # pip install opencv-python
import gc  # For garbage collection

# -------- Colors / sizes --------
YELLOW = np.array([1, 0.706, 0])
GREY   = np.array([0.5, 0.5, 0.5])
RED    = np.array([128, 0, 0]) / 255.0
BLACK  = np.array([0, 0, 0]) / 255.0
BLUE   = np.array([0.4, 0.5, 0.9])
GREEN  = np.array([0.4, 0.9, 0.5])

SPHERE_SIZE_KEYPOSES = 1.0
SPHERE_SIZE_ODOMETRY = 0.5


def transform_points(pcd, T):
    R = T[:3, :3]
    t = T[:3, -1]
    return pcd @ R.T + t


class StubVisualizer(ABC):
    def __init__(self):
        pass

    def update(self, slam):
        pass


class RegistrationVisualizer(StubVisualizer):
    def __init__(self):
        try:
            self.o3d = importlib.import_module("open3d")
        except ModuleNotFoundError:
            print('open3d is not installed on your system, run "pip install open3d"')
            exit(1)

        self.block_vis = True
        self.play_crun = False
        self.reset_bounding_box = True
        self.view_mode = "topdown"

        self.local_map = self.o3d.geometry.PointCloud()
        self.closures = []
        self.key_poses = []
        self.key_frames = []
        self.global_frames = []
        self.odom_frames = []
        self.edges = []
        self.current_node = None
        self.frame_id = 0  # Frame counter for PNGs

        self.vis = self.o3d.visualization.VisualizerWithKeyCallback()
        self._register_key_callbacks()
        self._initialize_visualizer()

    def update(self, slam):
        self._update_geometries(slam)
        while self.block_vis:
            self.vis.poll_events()
            self.vis.update_renderer()
            if self.play_crun:
                break
        self.block_vis = not self.block_vis

    def _initialize_visualizer(self):
        w_name = self.__class__.__name__
        self.vis.create_window(window_name=w_name, width=1920, height=1080)
        self.vis.add_geometry(self.local_map)
        self._set_black_background(self.vis)
        ro = self.vis.get_render_option()
        ro.point_size = 1
        ro.line_width = 10
        print(
            f"{w_name} initialized. Press:\n"
            "\t[SPACE] to pause/start\n"
            "\t  [ESC] to exit\n"
            "\t    [N] to step\n"
            "\t    [C] to center the viewpoint\n"
            "\t    [W] to toggle a white background\n"
            "\t    [B] to toggle a black background\n"
            "\t    [F] to toggle camera mode (follow <-> topdown)\n"
        )

    def _register_key_callback(self, keys: List, callback: Callable):
        for key in keys:
            self.vis.register_key_callback(ord(str(key)), partial(callback))

    def _register_key_callbacks(self):
        self._register_key_callback(["Ā", "Q", "\x1b"], self._quit)
        self._register_key_callback([" "], self._start_stop)
        self._register_key_callback(["N"], self._next_frame)
        self._register_key_callback(["C"], self._center_viewpoint)
        self._register_key_callback(["B"], self._set_black_background)
        self._register_key_callback(["W"], self._set_white_background)
        self._register_key_callback(["F"], self._toggle_view_mode)

    def _set_black_background(self, vis):
        vis.get_render_option().background_color = [0.0, 0.0, 0.0]

    def _set_white_background(self, vis):
        vis.get_render_option().background_color = [1.0, 1.0, 1.0]

    def _quit(self, vis):
        print("Destroying Visualizer")
        vis.destroy_window()
        os._exit(0)

    def _next_frame(self, vis):
        self.block_vis = not self.block_vis

    def _start_stop(self, vis):
        self.play_crun = not self.play_crun

    def _center_viewpoint(self, vis):
        vis.reset_view_point(True)

    def _toggle_view_mode(self, vis):
        self.view_mode = "follow" if self.view_mode == "topdown" else "topdown"
        print(f"[View] mode -> {self.view_mode.upper()}")

    def _add_line(self, pose0, pose1, color):
        lines = [[0, 1]]
        colors = [color]
        line_set = self.o3d.geometry.LineSet()
        line_set.points = self.o3d.utility.Vector3dVector([pose0, pose1])
        line_set.lines = self.o3d.utility.Vector2iVector(lines)
        line_set.colors = self.o3d.utility.Vector3dVector(colors)
        return line_set

    def _add_frames(self, poses, size, color):
        return [self._add_frame(pose, size, color) for pose in poses]

    def _add_frame(self, pose, size, color):
        m = self.o3d.geometry.TriangleMesh.create_sphere(size)
        m.paint_uniform_color(color)
        m.compute_vertex_normals()
        m.transform(pose)
        return m

    def _follow_pose(self, pose, distance=10.0, height=5.0, zoom=0.35):
        ctr = self.vis.get_view_control()
        R = pose[:3, :3]
        t = pose[:3, 3]
        fwd = R @ np.array([1.0, 0.0, 0.0])
        up = R @ np.array([0.0, 0.0, 1.0])
        cam_pos = t - fwd * distance + up * height
        front = t - cam_pos
        n = np.linalg.norm(front)
        if n < 1e-9:
            return
        front /= n
        up_n = up / (np.linalg.norm(up) + 1e-9)
        ctr.set_lookat(t.tolist())
        ctr.set_front(front.tolist())
        ctr.set_up(up_n.tolist())
        ctr.set_zoom(zoom)

    def _topdown_view(self, center, zoom=0.5, height=50.0):
        ctr = self.vis.get_view_control()
        cam_pos = center - np.array([0.0, 0.0, height])  # Look from above
        front = center - cam_pos
        n = np.linalg.norm(front)
        if n < 1e-9:
            return
        front /= n
        up = np.array([1.0, 0.0, 0.0])
        ctr.set_lookat(center.tolist())
        ctr.set_front(front.tolist())
        ctr.set_up(up.tolist())
        ctr.set_zoom(zoom)

    def _update_geometries(self, slam):
        current_node = slam.local_map_graph.last_local_map
        local_map_in_global = transform_points(
            slam.voxel_grid.point_cloud(), current_node.keypose
        )
        self.local_map.points = self.o3d.utility.Vector3dVector(local_map_in_global)
        self.local_map.paint_uniform_color(YELLOW)
        self.vis.update_geometry(self.local_map)

        current_pose = current_node.endpose

        for frame in self.odom_frames:
            self.vis.remove_geometry(frame, reset_bounding_box=False)
        self.odom_frames = [self._add_frame(current_pose, SPHERE_SIZE_ODOMETRY, BLUE)]
        self.vis.add_geometry(self.odom_frames[0], reset_bounding_box=False)

        if self.view_mode == "follow":
            self._follow_pose(current_pose)
        else:
            self._topdown_view(current_pose[:3, 3])

        key_poses = slam.get_keyposes()
        if key_poses != self.key_poses:
            self.key_poses = key_poses

            for frame in self.key_frames:
                self.vis.remove_geometry(frame, reset_bounding_box=False)
            self.key_frames = self._add_frames(key_poses, SPHERE_SIZE_KEYPOSES, YELLOW)
            for frame in self.key_frames:
                self.vis.add_geometry(frame, reset_bounding_box=False)

            for edge in self.edges:
                self.vis.remove_geometry(edge, reset_bounding_box=False)
            self.edges.clear()

            for f0, f1 in zip(self.key_frames[:-1], self.key_frames[1:]):
                p0 = f0.get_center()
                p1 = f1.get_center()
                self.edges.append(self._add_line(p0, p1, YELLOW))

            for (idx0, idx1) in self.closures:
                if idx0 < len(self.key_frames) and idx1 < len(self.key_frames):
                    p0 = self.key_frames[idx0].get_center()
                    p1 = self.key_frames[idx1].get_center()
                    self.edges.append(self._add_line(p0, p1, RED))

            for e in self.edges:
                self.vis.add_geometry(e, reset_bounding_box=False)

        if self.reset_bounding_box:
            self.vis.reset_view_point(True)
            self.reset_bounding_box = False

        self.vis.poll_events()
        self.vis.update_renderer()

        # Save PNG image for current scan
        os.makedirs("renders", exist_ok=True)
        filename = f"renders/scan_{self.frame_id:04d}.png"
        self.vis.capture_screen_image(filename)
        self.frame_id += 1

        # Free up memory
        gc.collect()




