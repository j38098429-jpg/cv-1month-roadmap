import cv2
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", required=True, help="input image path")
    parser.add_argument("--out_path", default="out.jpg", help="output image path")
    parser.add_argument("--size", type=int, default=256, help="resize to size x size")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"not found: {in_path}")

    img = cv2.imread(str(in_path))  # BGR
    if img is None:
        raise RuntimeError("cv2.imread failed (check file format/path)")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (args.size, args.size))

    ok = cv2.imwrite(args.out_path, resized)
    if not ok:
        raise RuntimeError("cv2.imwrite failed")

    print(f"saved to {args.out_path}, shape={resized.shape}")

if __name__ == "__main__":
    main()
