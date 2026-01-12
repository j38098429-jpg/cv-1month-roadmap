import argparse
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


def resize_long_side(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if long_side <= 0:
        return img
    scale = long_side / max(h, w)
    if abs(scale - 1.0) < 1e-6:
        return img
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def sobel_mag(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    # Sobel x/y -> magnitude -> normalize to 0~255
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)


def make_grid(images, titles, cell_w=420) -> np.ndarray:
    # images: list of BGR/GRAY
    # convert to BGR for stacking
    bgrs = []
    for im in images:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        bgrs.append(im)

    # resize each to same width
    resized = []
    for im in bgrs:
        h, w = im.shape[:2]
        scale = cell_w / w
        nh = int(round(h * scale))
        im2 = cv2.resize(im, (cell_w, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        resized.append(im2)

    # make 2 rows (3 + 2)
    def annotate(im, text):
        out = im.copy()
        cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return out

    annotated = [annotate(im, t) for im, t in zip(resized, titles)]

    row1 = cv2.hconcat(annotated[:3])
    row2 = cv2.hconcat(annotated[3:5])

    # pad row2 to same width as row1
    h2, w2 = row2.shape[:2]
    h1, w1 = row1.shape[:2]
    if w2 < w1:
        pad = np.zeros((h2, w1 - w2, 3), dtype=row2.dtype)
        row2 = cv2.hconcat([row2, pad])

    # pad heights to match
    if row1.shape[0] != row2.shape[0]:
        # pad the smaller one
        if row1.shape[0] < row2.shape[0]:
            pad = np.zeros((row2.shape[0] - row1.shape[0], row1.shape[1], 3), dtype=row1.dtype)
            row1 = cv2.vconcat([row1, pad])
        else:
            pad = np.zeros((row1.shape[0] - row2.shape[0], row2.shape[1], 3), dtype=row2.dtype)
            row2 = cv2.vconcat([row2, pad])

    grid = cv2.vconcat([row1, row2])
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="input image path")
    ap.add_argument("--out_dir", default="out_day4", help="output directory")
    ap.add_argument("--size", type=int, default=900, help="resize long side (0 means keep)")
    ap.add_argument("--blur", type=int, default=3, help="Gaussian blur kernel for canny (odd, 0 means no blur)")
    ap.add_argument("--low1", type=int, default=60, help="canny low threshold set1")
    ap.add_argument("--high1", type=int, default=150, help="canny high threshold set1")
    ap.add_argument("--low2", type=int, default=30, help="canny low threshold set2")
    ap.add_argument("--high2", type=int, default=100, help="canny high threshold set2")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    bgr = load_bgr(in_path)
    bgr = resize_long_side(bgr, args.size)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    sob = sobel_mag(gray, ksize=3)

    g = gray
    if args.blur and args.blur > 0:
        k = args.blur if args.blur % 2 == 1 else args.blur + 1
        g = cv2.GaussianBlur(gray, (k, k), 0)

    can1 = cv2.Canny(g, args.low1, args.high1)
    can2 = cv2.Canny(g, args.low2, args.high2)

    grid = make_grid(
        images=[bgr, gray, sob, can1, can2],
        titles=[
            "src",
            "gray",
            "sobel_mag",
            f"canny {args.low1},{args.high1}",
            f"canny {args.low2},{args.high2}",
        ],
        cell_w=420,
    )

    out_path = out_dir / "compare_edges.jpg"
    cv2.imwrite(str(out_path), grid)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
