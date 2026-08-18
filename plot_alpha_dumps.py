import open3d as o3d
import glob

# Load tất cả file .ply trong alpha_dumps
files = sorted(glob.glob("alpha_dumps/*.ply"))

print(f"Tìm thấy {len(files)} file.")
for f in files[:5]:  # In thử 5 file đầu
    print(f)

# Chọn một file để visualize
pcd = o3d.io.read_point_cloud(files[0])
print(pcd)

# Visualize
o3d.visualization.draw_geometries([pcd])
