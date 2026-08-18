import os, re, glob, cv2
from natsort import natsorted  # pip install natsort

def images_to_video(
    img_dir: str,
    out_path: str = "output.mp4",
    fps: int = 30,
    pattern: str = "*.png",
    force_size=None,        # (W, H) nếu muốn ép kích thước; None = lấy theo ảnh đầu
    codec: str = "mp4v",    # "avc1" (H.264) nếu có, fallback "mp4v"
):
    # 1) Lấy danh sách ảnh & sort tự nhiên
    imgs = natsorted(glob.glob(os.path.join(img_dir, pattern)))
    if not imgs:
        raise FileNotFoundError(f"Không tìm thấy ảnh trong {img_dir}/{pattern}")

    # 2) Đọc ảnh đầu để biết kích thước
    frame0 = cv2.imread(imgs[0], cv2.IMREAD_COLOR)
    if frame0 is None:
        raise RuntimeError(f"Không đọc được ảnh: {imgs[0]}")
    h0, w0 = frame0.shape[:2]
    if force_size is None:
        W, H = w0, h0
    else:
        W, H = force_size

    # 3) Mở VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*codec)
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not vw.isOpened():
        # fallback mp4v
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
        if not vw.isOpened():
            raise RuntimeError("Không mở được VideoWriter. Thử cài: pip install opencv-python")

    # 4) Ghi từng frame
    for i, fp in enumerate(imgs, 1):
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Bỏ qua ảnh lỗi: {fp}")
            continue
        if (img.shape[1], img.shape[0]) != (W, H):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        vw.write(img)
        if i % 50 == 0:
            print(f"  -> {i}/{len(imgs)} frames")

    vw.release()
    print(f"[OK] Saved: {out_path} ({len(imgs)} frames @ {fps}fps, size={W}x{H})")

if __name__ == "__main__":
    images_to_video(
        img_dir="renders",     # thư mục chứa ảnh
        out_path="run.mp4",
        fps=30,
        pattern="frame_*.png",    # hoặc "*.jpg"
        force_size=None,          # ví dụ (1920,1080) nếu muốn ép
        codec="avc1",             # "avc1" (H.264), nếu lỗi sẽ tự fallback "mp4v"
    )
