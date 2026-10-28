from pathlib import Path

import napari
import numpy as np
import tifffile
from PIL import Image

repo = Path(__file__).resolve().parent.parent
output = repo / "work" / "jpeg_experiment"
output.mkdir(parents=True, exist_ok=True)

original = tifffile.imread(repo / "testImages" / "Cell_cropped_2.tif")

if original.dtype != np.uint8 or original.shape[-1] != 3:
    raise ValueError("Expected an 8-bit RGB image.")

# Save once as JPEG, then reload its pixel data.
jpeg_path = output / "cell_once.jpg"
Image.fromarray(original).save(jpeg_path, quality=75)

with Image.open(jpeg_path) as image:
    jpeg_once = np.array(image.convert("RGB"))

# Decode and re-encode the JPEG 100 additional times.
repeated_path = output / "cell_repeated.jpg"
repeated = jpeg_once.copy()

for _ in range(100):
    Image.fromarray(repeated).save(repeated_path, quality=75)
    with Image.open(repeated_path) as image:
        repeated = np.array(image.convert("RGB"))

# Convert before subtraction to avoid unsigned-integer wraparound.
difference_once = np.abs(
    original.astype(np.int16) - jpeg_once.astype(np.int16)
)
difference_repeated = np.abs(
    original.astype(np.int16) - repeated.astype(np.int16)
)

for label, difference in [
    ("JPEG saved once", difference_once),
    ("100 additional saves", difference_repeated),
]:
    print(f"\n{label}")
    print(f"Mean absolute error: {difference.mean():.3f}")
    print(f"Maximum absolute error: {difference.max()}")

# Use one shared scale for the two difference displays.
maximum = max(
    1, int(difference_once.max()), int(difference_repeated.max())
)

viewer = napari.Viewer()
viewer.add_image(original, rgb=True, name="Original TIFF")
viewer.add_image(
    jpeg_once, rgb=True, name="JPEG — first save", visible=False
)
viewer.add_image(
    repeated, rgb=True, name="JPEG — repeated saves", visible=False
)

# Average the absolute RGB differences for a grayscale error map.
viewer.add_image(
    difference_once.mean(axis=-1),
    name="Difference — first save",
    colormap="magma",
    contrast_limits=(0, maximum),
    visible=False,
)
viewer.add_image(
    difference_repeated.mean(axis=-1),
    name="Difference — repeated saves",
    colormap="magma",
    contrast_limits=(0, maximum),
    visible=False,
)
napari.run()