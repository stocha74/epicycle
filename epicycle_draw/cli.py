from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable


def _require_numpy():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("numpy is required. Install with: pip install numpy") from exc
    return np


def _require_pillow():
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required. Install with: pip install pillow") from exc
    return Image, ImageDraw


def _require_cv2():
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required. Install with: pip install opencv-python") from exc
    return cv2


def _require_imageio():
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("imageio is required. Install with: pip install imageio") from exc
    return imageio


def _load_input_image(path: Path, gif_frame: int):
    np = _require_numpy()
    Image, _ = _require_pillow()

    if path.suffix.lower() == ".gif":
        with Image.open(path) as img:
            img.seek(gif_frame)
            return np.array(img.convert("RGB"))

    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def _extract_largest_contour(rgb):
    np = _require_numpy()
    cv2 = _require_cv2()

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blur))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(blur, lower, upper)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No contour detected from input image.")

    contour = max(contours, key=cv2.contourArea)
    points = contour[:, 0, :].astype(np.float64)
    if np.linalg.norm(points[0] - points[-1]) > 1.0:
        points = np.vstack([points, points[:1]])
    return points


def _resample_closed_contour(points, n_points: int):
    np = _require_numpy()

    deltas = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    total = cum_len[-1]
    if total <= 0:
        raise RuntimeError("Contour length is zero.")

    targets = np.linspace(0, total, n_points, endpoint=False)
    xs = np.interp(targets, cum_len, points[:, 0])
    ys = np.interp(targets, cum_len, points[:, 1])
    return xs + 1j * ys


def _fft_topk(signal, top_k: int):
    np = _require_numpy()

    n = signal.shape[0]
    coeffs = np.fft.fft(signal) / n
    freqs = np.fft.fftfreq(n, d=1.0 / n).astype(int)
    idx = np.argsort(np.abs(coeffs))[::-1][: min(top_k, n)]
    return coeffs[idx], freqs[idx]


def _build_epicycle_frames(coeffs, freqs, n_frames: int, width: int, height: int):
    np = _require_numpy()
    Image, ImageDraw = _require_pillow()

    ts = np.linspace(0.0, 1.0, n_frames, endpoint=False)
    recon = np.array([np.sum(coeffs * np.exp(2j * np.pi * freqs * t)) for t in ts])

    min_x, max_x = float(np.min(recon.real)), float(np.max(recon.real))
    min_y, max_y = float(np.min(recon.imag)), float(np.max(recon.imag))
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    scale = min((width * 0.8) / span_x, (height * 0.8) / span_y)

    center_src = complex((min_x + max_x) / 2, (min_y + max_y) / 2)
    center_dst = complex(width / 2, height / 2)

    def transform(z):
        zz = (z - center_src) * scale + center_dst
        return float(zz.real), float(zz.imag)

    frames = []
    trajectory = []
    order = np.argsort(np.abs(coeffs))[::-1]
    coeffs_ord = coeffs[order]
    freqs_ord = freqs[order]

    for t in ts:
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)

        pos = 0j
        for c, f in zip(coeffs_ord, freqs_ord):
            prev = pos
            pos = prev + c * np.exp(2j * np.pi * f * t)
            x0, y0 = transform(prev)
            x1, y1 = transform(pos)
            r = abs(c) * scale
            draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), outline=(190, 190, 190), width=1)
            draw.line((x0, y0, x1, y1), fill=(220, 90, 80), width=2)

        trajectory.append(transform(pos))
        if len(trajectory) > 1:
            draw.line(trajectory, fill=(60, 110, 230), width=3)
        frames.append(np.array(canvas))

    return frames


def _output_path(output: str | None) -> Path:
    if output:
        p = Path(output)
        if p.suffix:
            return p
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    return Path(f"{output or 'epicycle_output'}.{ 'mp4' if ffmpeg_ok else 'gif'}")


def _save_animation(frames: Iterable, path: Path, fps: int) -> None:
    imageio = _require_imageio()

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".gif":
        imageio.mimsave(path, list(frames), format="GIF", fps=fps)
    elif suffix == ".mp4":
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
            for frame in frames:
                writer.append_data(frame)
    else:
        raise ValueError("Output extension must be .mp4 or .gif")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fotoğraftan kontur çıkarıp FFT epicycle animasyonu üretir.")
    parser.add_argument("--input", required=True, help="Input image (.jpg/.jpeg/.png/.gif)")
    parser.add_argument("--output", help="Output (.mp4/.gif). Uzantı yoksa otomatik seçilir")
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gif_frame", type=int, default=0)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".gif"}:
        raise ValueError("Supported input formats: .jpg .jpeg .png .gif")

    output_path = _output_path(args.output)

    rgb = _load_input_image(input_path, args.gif_frame)
    contour = _extract_largest_contour(rgb)
    signal = _resample_closed_contour(contour, args.points)
    centered = signal - signal.mean()

    coeffs, freqs = _fft_topk(centered, args.top_k)
    frames = _build_epicycle_frames(coeffs, freqs, args.points, args.width, args.height)

    _save_animation(frames, output_path, args.fps)
    print(f"Saved animation: {output_path}")


if __name__ == "__main__":
    main()
