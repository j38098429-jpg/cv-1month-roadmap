import cv2
import numpy as np
from pathlib import Path
import argparse


def add_salt_pepper(gray: np.ndarray, amount: float = 0.02) -> np.ndarray:
    """人为加一点椒盐噪声，方便看出滤波效果"""
    noisy = gray.copy()
    h, w = noisy.shape
    num = int(amount * h * w)
    ys = np.random.randint(0, h, num)
    xs = np.random.randint(0, w, num)
    noisy[ys, xs] = 255
    ys = np.random.randint(0, h, num)
    xs = np.random.randint(0, w, num)
    noisy[ys, xs] = 0
    return noisy


def tile(images, titles, out_path: str):
    """把多张灰度图拼成一张大图，并写上标题"""
    assert len(images) == len(titles)
    imgs = []
    for img, t in zip(images, titles):
        if img.ndim == 2:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis = img.copy()
        cv2.putText(vis, t, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        imgs.append(vis)

    # 2x3 拼图（够用）
    while len(imgs) < 6:
        imgs.append(np.zeros_like(imgs[0]))

    row1 = np.hstack(imgs[:3])
    row2 = np.hstack(imgs[3:6])
    grid = np.vstack([row1, row2])
    cv2.imwrite(out_path, grid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", required=True)
    parser.add_argument("--out_dir", default="out_day3")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--noise", action="store_true", help="add salt&pepper noise for demo")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"not found: {in_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(in_path))
    if img is None:
        raise RuntimeError("cv2.imread failed")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (args.size, args.size))

    src = add_salt_pepper(gray, 0.02) if args.noise else gray

    mean = cv2.blur(src, (5, 5))
    gauss = cv2.GaussianBlur(src, (5, 5), 0)
    median = cv2.medianBlur(src, 5)
    bilateral = cv2.bilateralFilter(src, d=9, sigmaColor=75, sigmaSpace=75)

    cv2.imwrite(str(out_dir / "src.jpg"), src)
    cv2.imwrite(str(out_dir / "mean.jpg"), mean)
    cv2.imwrite(str(out_dir / "gauss.jpg"), gauss)
    cv2.imwrite(str(out_dir / "median.jpg"), median)
    cv2.imwrite(str(out_dir / "bilateral.jpg"), bilateral)

    tile(
        [src, mean, gauss, median, bilateral],
        ["src", "mean", "gauss", "median", "bilateral"],
        str(out_dir / "compare_denoise.jpg"),
    )

    print("saved:")
    print(out_dir / "compare_denoise.jpg")


if __name__ == "__main__":
    main()
