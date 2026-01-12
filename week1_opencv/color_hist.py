import cv2
import numpy as np
from pathlib import Path
import argparse


def save_hist_image(gray: np.ndarray, out_path: str, width: int = 512, height: int = 300):
    """把灰度直方图画成一张图保存"""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.max() + 1e-6)  # 归一化到 0~1

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for x in range(width):
        idx = int(x * 256 / width)
        h = int(hist[idx] * (height - 10))
        cv2.line(canvas, (x, height - 1), (x, height - 1 - h), (0, 0, 0), 1)

    cv2.imwrite(out_path, canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", required=True)
    parser.add_argument("--out_dir", default="out_day2")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"not found: {in_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(in_path))  # BGR
    if img is None:
        raise RuntimeError("cv2.imread failed")

    # 1) BGR -> Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) 直方图均衡化
    eq = cv2.equalizeHist(gray)

    # 3) 保存对比图（原图灰度 vs 均衡化）
    compare = np.hstack([gray, eq])
    cv2.imwrite(str(out_dir / "compare_gray_eq.jpg"), compare)

    # 4) 保存直方图图像
    save_hist_image(gray, str(out_dir / "hist_gray.jpg"))
    save_hist_image(eq, str(out_dir / "hist_eq.jpg"))

    print("saved:")
    print(out_dir / "compare_gray_eq.jpg")
    print(out_dir / "hist_gray.jpg")
    print(out_dir / "hist_eq.jpg")


if __name__ == "__main__":
    main()
