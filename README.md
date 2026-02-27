# epicycle_draw

Fotoğraf veya görselden ana konturu çıkarıp FFT tabanlı epicycle çizim animasyonu üretir.

## Kurulum

```bash
pip install -r requirements.txt
```

## CLI

```bash
python -m epicycle_draw \
  --input path/to/image.jpg \
  --output outputs/out.mp4 \
  --points 2048 \
  --top_k 300 \
  --fps 30 \
  --gif_frame 0
```

## Varsayılan davranışlar

- GIF input ise yalnızca ilk frame (`gif_frame=0`) işlenir.
- Kontur çıkarma: Gaussian blur + Canny edge.
- Kontur seçimi: en büyük dış kontur (`max(contours, key=contourArea)`).
- Yeniden örnekleme: 2048 nokta.
- Fourier seçim: genliğe göre en büyük `top_k=300` katsayı.
- Çıktı uzantısı verilmezse `ffmpeg` varsa `.mp4`, yoksa `.gif`.
- Kontur kapalı döngü kabul edilir; gerekiyorsa ilk nokta sona eklenir.
