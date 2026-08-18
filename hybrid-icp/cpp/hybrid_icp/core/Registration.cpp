// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "Registration.hpp"

#include <Eigen/Eigenvalues>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdio>
#include <fstream>
#include <filesystem>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include "VoxelHashMap.hpp"
#include "VoxelUtils.hpp"

namespace fs = std::filesystem;

namespace Eigen {
using Matrix6d   = Eigen::Matrix<double, 6, 6>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
using Vector6d   = Eigen::Matrix<double, 6, 1>;
}  // namespace Eigen

namespace {

inline double square(double x) { return x * x; }

// -------------------- Hybrid correspondences structure --------------------
struct HybridCorrespondence {
    std::vector<Eigen::Vector3d> src_planar, tgt_planar, normals;
    std::vector<Eigen::Vector3d> src_non_planar, tgt_non_planar;
    size_t planar_count = 0, non_planar_count = 0;
};

// === Adaptive Planarity & Normal Estimation (PCA)
double ComputeAdaptivePlanarityThreshold(const std::vector<Eigen::Vector3d>& neighbors) {
    // threshold ~ 1/n, kẹp [0.001, 0.2]
    const double min_neighbors = 6.0;
    const double ref_neighbors = 15.0;
    const double base = 0.06;
    const double min_thr = 0.001, max_thr = 0.2;
    const double n = std::max(min_neighbors, static_cast<double>(neighbors.size()));
    const double thr = base * (ref_neighbors / n);
    return std::clamp(thr, min_thr, max_thr);
}

std::tuple<bool, Eigen::Vector3d> EstimateNormalAndPlanarity(
    const std::vector<Eigen::Vector3d>& neighbors)
{
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& pt : neighbors) mean += pt;
    mean /= static_cast<double>(neighbors.size());

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& pt : neighbors) {
        Eigen::Vector3d d = pt - mean;
        cov.noalias() += d * d.transpose();
    }
    cov /= static_cast<double>(neighbors.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    const auto& evals = eig.eigenvalues();
    const auto& evecs = eig.eigenvectors();

    const double lambda0 = evals(0);
    const double sumlam  = evals(0) + evals(1) + evals(2) + 1e-12;
    const double planarity = lambda0 / sumlam;

    const double adaptive_thr = ComputeAdaptivePlanarityThreshold(neighbors);
    const bool is_planar = planarity < adaptive_thr;
    Eigen::Vector3d normal = evecs.col(0);
    return {is_planar, normal};
}

// -------------------- Parallel TransformPoints --------------------
void TransformPoints(const Sophus::SE3d &T, std::vector<Eigen::Vector3d> &points) {
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, points.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                points[i] = T * points[i];
            }
        }
    );
}

// -------------------- Parallel Hybrid Data Association --------------------
HybridCorrespondence ComputeHybridCorrespondencesParallel(
    const std::vector<Eigen::Vector3d>& source_points,
    const hybrid_icp::VoxelHashMap& voxel_map,
    double max_correspondence_distance)
{
    struct LocalBuf {
        std::vector<Eigen::Vector3d> src_planar, tgt_planar, normals;
        std::vector<Eigen::Vector3d> src_non_planar, tgt_non_planar;
        size_t planar_count = 0, non_planar_count = 0;

        void reserve_hint(size_t n) {
            const size_t hint = std::max<size_t>(32, n / 2);
            src_planar.reserve(hint);  tgt_planar.reserve(hint); normals.reserve(hint);
            src_non_planar.reserve(hint); tgt_non_planar.reserve(hint);
        }
    };

    tbb::enumerable_thread_specific<LocalBuf> tls;
    for (auto it = tls.begin(); it != tls.end(); ++it) it->reserve_hint(source_points.size());

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, source_points.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            auto& buf = tls.local();
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const auto& pt = source_points[i];
                auto [closest, neighbors, dist] = voxel_map.GetClosestNeighborAndNeighbors(pt);
                if (dist > max_correspondence_distance) continue;

                if (neighbors.size() >= 5) {
                    auto [is_planar, normal] = EstimateNormalAndPlanarity(neighbors);
                    if (is_planar) {
                        buf.src_planar.push_back(pt);
                        buf.tgt_planar.push_back(closest);
                        buf.normals.push_back(normal);
                        buf.planar_count++;
                    } else {
                        buf.src_non_planar.push_back(pt);
                        buf.tgt_non_planar.push_back(closest);
                        buf.non_planar_count++;
                    }
                } else {
                    buf.src_non_planar.push_back(pt);
                    buf.tgt_non_planar.push_back(closest);
                    buf.non_planar_count++;
                }
            }
        }
    );

    HybridCorrespondence out;
    size_t total_planar = 0, total_nonplanar = 0;
    for (auto& buf : tls) {
        total_planar    += buf.planar_count;
        total_nonplanar += buf.non_planar_count;
    }
    out.src_planar.reserve(total_planar);
    out.tgt_planar.reserve(total_planar);
    out.normals.reserve(total_planar);
    out.src_non_planar.reserve(total_nonplanar);
    out.tgt_non_planar.reserve(total_nonplanar);

    for (auto& buf : tls) {
        out.planar_count     += buf.planar_count;
        out.non_planar_count += buf.non_planar_count;

        out.src_planar.insert(out.src_planar.end(), buf.src_planar.begin(), buf.src_planar.end());
        out.tgt_planar.insert(out.tgt_planar.end(), buf.tgt_planar.begin(), buf.tgt_planar.end());
        out.normals.insert(out.normals.end(), buf.normals.begin(), buf.normals.end());

        out.src_non_planar.insert(out.src_non_planar.end(), buf.src_non_planar.begin(), buf.src_non_planar.end());
        out.tgt_non_planar.insert(out.tgt_non_planar.end(), buf.tgt_non_planar.begin(), buf.tgt_non_planar.end());
    }
    return out;
}

// -------------------- Parallel BuildLinearSystem --------------------
std::tuple<Eigen::Matrix6d, Eigen::Vector6d> BuildHybridLinearSystemParallel(
    const HybridCorrespondence& corr, double kernel, double alpha)
{
    struct Accum {
        Eigen::Matrix<double,6,6> JTJ;
        Eigen::Matrix<double,6,1> JTr;
        Accum() { JTJ.setZero(); JTr.setZero(); }
        Accum(Accum&, tbb::split) { JTJ.setZero(); JTr.setZero(); }
        void join(const Accum& other) {
            JTJ.noalias() += other.JTJ;
            JTr.noalias() += other.JTr;
        }
    };

    const double kernel2 = kernel * kernel;

    // Point-to-plane
    Accum acc_plane = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, corr.src_planar.size()),
        Accum{},
        [&](const tbb::blocked_range<size_t>& r, Accum a)->Accum {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const Eigen::Vector3d& n = corr.normals[i];
                const double residual = (corr.src_planar[i] - corr.tgt_planar[i]).dot(n);

                Eigen::Matrix<double,1,6> J;
                J.block<1,3>(0,0) = n.transpose();
                J.block<1,3>(0,3) = (corr.src_planar[i].cross(n)).transpose();

                const double w = kernel2 / square(kernel + residual * residual);
                a.JTJ.noalias() += alpha * (J.transpose() * (w * J));
                a.JTr.noalias() += alpha * (J.transpose() * (w * residual));
            }
            return a;
        },
        [](Accum a, const Accum& b)->Accum { a.join(b); return a; }
    );

    // Point-to-point
    Accum acc_point = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, corr.src_non_planar.size()),
        Accum{},
        [&](const tbb::blocked_range<size_t>& r, Accum a)->Accum {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const Eigen::Vector3d rvec = corr.src_non_planar[i] - corr.tgt_non_planar[i];
                Eigen::Matrix<double,3,6> J;
                J.block<3,3>(0,0).setIdentity();
                J.block<3,3>(0,3) = -Sophus::SO3d::hat(corr.src_non_planar[i]);

                const double w = kernel2 / square(kernel + rvec.squaredNorm());
                a.JTJ.noalias() += (1.0 - alpha) * (J.transpose() * (w * J));
                a.JTr.noalias() += (1.0 - alpha) * (J.transpose() * (w * rvec));
            }
            return a;
        },
        [](Accum a, const Accum& b)->Accum { a.join(b); return a; }
    );

    Eigen::Matrix6d JTJ = acc_plane.JTJ + acc_point.JTJ;
    Eigen::Vector6d JTr = acc_plane.JTr + acc_point.JTr;
    return {JTJ, JTr};
}

// -------------------- Live console visualization --------------------
void VisualizeStatus(size_t planar_count, size_t non_planar_count, double alpha) {
    const int bar_width = 52;
    const std::string planar_color     = "\033[1;38;2;0;119;187m";   // blue-ish
    const std::string non_planar_color = "\033[1;38;2;238;51;119m"; // pink-ish
    const std::string alpha_color      = "\033[1;32m";               // green

    std::printf("\033[2J\033[1;1H"); // Clear terminal + move cursor to 1,1
    std::cout << "====================== Hybrid-ICP ======================\n";
    std::cout << non_planar_color << "# of non-planar points: " << non_planar_count << ", ";
    std::cout << planar_color     << "# of planar points: "     << planar_count     << "\033[0m\n";

    std::cout << "Unstructured  <-----  "
              << alpha_color << "alpha: " << std::fixed << std::setprecision(3) << alpha << "\033[0m"
              << "  ----->  Structured\n";

    const int alpha_location = std::clamp(static_cast<int>(bar_width * alpha), 0, bar_width - 1);
    std::cout << "[";
    for (int i = 0; i < bar_width; ++i) {
        if (i == alpha_location) {
            std::cout << "\033[1;32m" << "█" << "\033[0m";
        } else {
            std::cout << "-";
        }
    }
    std::cout << "]\n";
    std::cout.flush();
}

// -------------------- PLY writer helpers --------------------
void EnsureDir(const std::string& dir) {
    try {
        if (!dir.empty() && !fs::exists(dir)) fs::create_directories(dir);
    } catch (...) { /* ignore */ }
}

void WriteColoredPLY(
    const std::string& filename,
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3i>& colors)
{
    if (points.empty() || points.size() != colors.size()) return;

    std::ofstream ofs(filename, std::ios::out);
    if (!ofs) return;

    ofs << "ply\nformat ascii 1.0\n";
    ofs << "element vertex " << points.size() << "\n";
    ofs << "property float x\nproperty float y\nproperty float z\n";
    ofs << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    ofs << "end_header\n";

    for (size_t i = 0; i < points.size(); ++i) {
        const auto& p = points[i];
        const auto& c = colors[i];
        ofs << static_cast<float>(p.x()) << " "
            << static_cast<float>(p.y()) << " "
            << static_cast<float>(p.z()) << " "
            << std::clamp(c.x(), 0, 255) << " "
            << std::clamp(c.y(), 0, 255) << " "
            << std::clamp(c.z(), 0, 255) << "\n";
    }
    ofs.close();
}

void DumpHybridCorrespondencesPLY(
    const std::string& out_dir,
    int frame_id,
    double alpha,
    const HybridCorrespondence& corr,
    bool dump_targets = false)
{
    EnsureDir(out_dir);

    std::vector<Eigen::Vector3d> pts;
    std::vector<Eigen::Vector3i> cols;
    pts.reserve(corr.src_planar.size() + corr.src_non_planar.size()
                + (dump_targets ? (corr.tgt_planar.size() + corr.tgt_non_planar.size()) : 0));
    cols.reserve(pts.capacity());

    // planar -> RED
    for (const auto& p : corr.src_planar) {
        pts.push_back(p);
        cols.emplace_back(255, 0, 0);
    }
    // non-planar -> BLUE
    for (const auto& p : corr.src_non_planar) {
        pts.push_back(p);
        cols.emplace_back(0, 0, 255);
    }

    if (dump_targets) {
        for (const auto& p : corr.tgt_planar) {
            pts.push_back(p);
            cols.emplace_back(200, 100, 100);
        }
        for (const auto& p : corr.tgt_non_planar) {
            pts.push_back(p);
            cols.emplace_back(100, 100, 200);
        }
    }

    char fname[256];
    std::snprintf(fname, sizeof(fname), "frame_%06d_a%.3f.ply", frame_id, alpha);
    const std::string path = (fs::path(out_dir) / fname).string();
    WriteColoredPLY(path, pts, cols);

    std::cout << "[Dump] wrote " << path << "  ("
              << corr.planar_count << " planar | "
              << corr.non_planar_count << " non-planar; alpha=" << std::fixed << std::setprecision(3) << alpha
              << ")\n";
}

}  // namespace (anon)

// ============================ kiss_icp namespace ============================
namespace hybrid_icp {

Registration::Registration(int max_num_iteration, double convergence_criterion, int max_num_threads)
    : max_num_iterations_(max_num_iteration),
      convergence_criterion_(convergence_criterion) {
    if (max_num_threads > 0) {
        static tbb::global_control gc(tbb::global_control::max_allowed_parallelism,
                                      static_cast<size_t>(max_num_threads));
        (void)gc;
    }
}

Sophus::SE3d Registration::AlignPointsToMap(const std::vector<Eigen::Vector3d> &frame,
                                            const VoxelHashMap &voxel_map,
                                            const Sophus::SE3d &initial_guess,
                                            double max_distance,
                                            double kernel)
{
    if (voxel_map.Empty()) return initial_guess;

    std::vector<Eigen::Vector3d> source = frame;
    TransformPoints(initial_guess, source);

    Sophus::SE3d T_icp;
    // Lưu corr/alpha của iteration cuối để dump "1 file / frame"
    HybridCorrespondence last_corr;
    double last_alpha = 0.0;

    for (int j = 0; j < max_num_iterations_; ++j) {
        // Correspondence building
        auto corr = ComputeHybridCorrespondencesParallel(source, voxel_map, max_distance);

        // Alpha = tỷ lệ point-to-plane (planar) so với tổng
        const double denom = static_cast<double>(corr.planar_count + corr.non_planar_count);
        const double alpha = (denom > 0.0)
            ? static_cast<double>(corr.planar_count) / denom
            : 0.5;

        // --- Live console status (tuỳ chọn) ---
        VisualizeStatus(corr.planar_count, corr.non_planar_count, alpha);

        // Solve hybrid linear system
        auto [JTJ, JTr] = BuildHybridLinearSystemParallel(corr, kernel, alpha);
        Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);
        Sophus::SE3d delta = Sophus::SE3d::exp(dx);

        // Update source and pose
        TransformPoints(delta, source);
        T_icp = delta * T_icp;

        // Ghi nhớ để dump sau khi dừng
        last_corr  = std::move(corr);
        last_alpha = alpha;

        // Termination
        if (dx.norm() < convergence_criterion_) break;
    }

    // === Dump đúng 1 lần cho mỗi frame sau khi hội tụ ===
    // Chỉ dump khi alpha > 0.6 như yêu cầu
    if (last_alpha >0.95) {
        static int s_frame_dump_id = 0;  // counter cục bộ cho file tên đẹp
        const std::string kDumpDir = "alpha_dumps";
        const bool dump_targets = false; // đặt true nếu muốn thêm các điểm target
        DumpHybridCorrespondencesPLY(kDumpDir, s_frame_dump_id++, last_alpha, last_corr, dump_targets);

        // === TẠM DỪNG CHỜ NGƯỜI DÙNG ===
        std::cout << "[Pause] File đã được dump. Nhấn ENTER để tiếp tục...\n";
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    return T_icp * initial_guess;
}

}  // namespace hybrid_icp
