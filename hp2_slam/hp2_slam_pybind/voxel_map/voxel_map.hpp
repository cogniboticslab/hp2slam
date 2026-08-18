// MIT License

// Copyright (c) 2025 Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta, Cyrill
// Stachniss.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <tsl/robin_map.h>

#include <Eigen/Core>
#include <array>
#include <cstdint>
#include <vector>

using Vector3fVector = std::vector<Eigen::Vector3f>;

using Voxel = Eigen::Vector3i;
template <>
struct std::hash<Voxel> {
    std::size_t operator()(const Voxel &voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
    }
};

// Same default as Open3d
static constexpr unsigned int max_points_per_normal_computation = 20;

namespace voxel_map {

struct VoxelBlock {
    void emplace_back(const Eigen::Vector3f &point);
    inline constexpr size_t size() const { return num_points; }
    auto cbegin() const { return points.cbegin(); }
    auto cend() const { return std::next(points.cbegin(), num_points); }
    std::array<Eigen::Vector3f, max_points_per_normal_computation> points;
    size_t num_points = 0;
};

struct VoxelMap {
    explicit VoxelMap(const float voxel_size);

    inline void Clear() { map_.clear(); }
    inline bool Empty() const { return map_.empty(); }
    void IntegrateFrame(const std::vector<Eigen::Vector3f> &points, const Eigen::Matrix4f &pose);
    void AddPoints(const std::vector<Eigen::Vector3f> &points);
    Vector3fVector Pointcloud() const;

    size_t NumVoxels() const { return map_.size(); }

    std::tuple<Vector3fVector, Vector3fVector> PerVoxelPointAndNormal() const;

    float voxel_size_;
    float map_resolution_;
    tsl::robin_map<Voxel, VoxelBlock> map_;
};
}  // namespace voxel_map
