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

    height = 0.4 * base + 0.25 * undulations

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