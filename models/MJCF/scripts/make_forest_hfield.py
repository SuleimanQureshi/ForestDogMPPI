# import numpy as np
# from PIL import Image

# def smooth_noise(h, w, scale, seed=0):
#     rng = np.random.default_rng(seed)
#     n = rng.standard_normal((h, w))
#     # cheap smoothing: repeated box blur via FFT-free convolution-ish
#     for _ in range(6):
#         n = (n +
#              np.roll(n, 1, 0) + np.roll(n, -1, 0) +
#              np.roll(n, 1, 1) + np.roll(n, -1, 1)) / 5.0
#     # upsample via repeats to get “large features”
#     # (scale here is conceptual; we'll just blend multiple octaves)
#     return n

# def make_forest(h=512, w=512, seed=7):
#     rng = np.random.default_rng(seed)

#     # Base rolling terrain (low frequency)
#     low = smooth_noise(h, w, scale=64, seed=seed)
#     low = (low - low.min()) / (low.max() - low.min())

#     # Mid bumps (roots/uneven soil)
#     mid = smooth_noise(h, w, scale=16, seed=seed+1)
#     mid = (mid - mid.min()) / (mid.max() - mid.min())

#     # High small bumps (pebbles/grass clumps)
#     hi = rng.random((h, w))
#     for _ in range(3):
#         hi = (hi +
#               np.roll(hi, 1, 0) + np.roll(hi, -1, 0) +
#               np.roll(hi, 1, 1) + np.roll(hi, -1, 1)) / 5.0
#     hi = (hi - hi.min()) / (hi.max() - hi.min())

#     # Add a couple of shallow “ruts/trails”
#     y = np.linspace(-1, 1, h)[:, None]
#     x = np.linspace(-1, 1, w)[None, :]
#     rut1 = np.exp(-((y + 0.15)**2) / (2*(0.03**2))) * 0.4
#     rut2 = np.exp(-((y - 0.25)**2) / (2*(0.05**2))) * 0.25
#     ruts = rut1 + rut2

#     # Combine (tune weights)
#     height = (
#         0.55 * low +
#         0.30 * mid +
#         0.10 * hi -
#         0.20 * ruts
#     )

#     # Normalize to [0,1]
#     height = (height - height.min()) / (height.max() - height.min())

#     # Add occasional localized mounds (rocks/roots under soil)
#     for _ in range(40):
#         cx = rng.integers(0, w)
#         cy = rng.integers(0, h)
#         rad = rng.integers(6, 25)
#         yy, xx = np.ogrid[:h, :w]
#         d2 = (xx - cx)**2 + (yy - cy)**2
#         bump = np.exp(-d2 / (2*(rad**2)))
#         height += 0.08 * bump

#     # Re-normalize
#     height = (height - height.min()) / (height.max() - height.min())

#     # Convert to 8-bit grayscale
#     img = (height * 255).astype(np.uint8)
#     return img

# if __name__ == "__main__":
#     img = make_forest(512, 512, seed=7)
#     Image.fromarray(img, mode="L").save("models/MJCF/assets/terrain/forest_hfield.png")
#     print("Wrote assets/terrain/forest_hfield.png")
# -----------------------------------------------------------------------------
import numpy as np
from PIL import Image

def gaussian_blur(img, passes=8):
    for _ in range(passes):
        img = (
            img +
            np.roll(img, 1, 0) + np.roll(img, -1, 0) +
            np.roll(img, 1, 1) + np.roll(img, -1, 1)
        ) / 5.0
    return img

def make_forest(h=512, w=512, seed=5):
    rng = np.random.default_rng(seed)

    # --- Base rolling terrain (VERY smooth) ---
    base = rng.standard_normal((h, w))
    base = gaussian_blur(base, passes=20)
    base = (base - base.min()) / (base.max() - base.min())

    # --- Gentle undulations (roots / compacted soil) ---
    undulations = rng.standard_normal((h, w))
    undulations = gaussian_blur(undulations, passes=12)
    undulations = (undulations - undulations.min()) / (undulations.max() - undulations.min())

    height = 0.45 * base + 0.25 * undulations

    # --- Sparse soft bumps (rocks under soil, roots) ---
    for _ in range(25):   # FEW bumps
        cx = rng.integers(0, w)
        cy = rng.integers(0, h)
        r = rng.integers(12, 35)   # LARGE radius → gentle
        yy, xx = np.ogrid[:h, :w]
        bump = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * r * r))
        height += 0.06 * bump      # SMALL amplitude
    # # Flatten spawn zone (~1.5m radius)
    #     h, w = height.shape
    #     cx, cy = h // 2, w // 2
    #     r = 30
    #     yy, xx = np.ogrid[:h, :w]
    #     mask = (xx - cx)**2 + (yy - cy)**2 < r*r
    #     height[mask] = np.mean(height[mask])
    # --- Normalize ---
    height = (height - height.min()) / (height.max() - height.min())

    # --- Compress extremes (kills sharp cliffs) ---
    height = np.clip(height, 0.2, 0.8)
    height = (height - height.min()) / (height.max() - height.min())

    img = (height * 255).astype(np.uint8)
    return img

if __name__ == "__main__":
    img = make_forest()
    Image.fromarray(img, mode="L").save(
        "models/MJCF/go2/assets/terrain/forest_hfield.png"
    )
    print("Forest heightfield written")