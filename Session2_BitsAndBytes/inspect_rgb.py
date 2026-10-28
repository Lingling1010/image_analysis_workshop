from pathlib import Path

import napari
import numpy as np
import tifffile

repo = Path(__file__).resolve().parent.parent
image = tifffile.imread(repo / "testImages" / "Cell_cropped_2.tif")

print(f"Shape: {image.shape}")
print(f"Data type: {image.dtype}")

# Expect rows × columns × RGB.
if image.ndim != 3 or image.shape[-1] != 3:
    raise ValueError(f"Expected an RGB image; got shape {image.shape}")

rows, columns, _ = image.shape
count = max(1, round(rows * columns * 0.03))

# Select approximately 3% of pixels at regular intervals.
indices = np.linspace(0, rows * columns - 1, count, dtype=int)

print(f"\nSampling {count} of {rows * columns} pixels")
print("Row\tColumn\tR\tG\tB")
for index in indices:
    row, column = divmod(int(index), columns)
    r, g, b = image[row, column]
    print(f"{row}\t{column}\t{r}\t{g}\t{b}")

viewer = napari.Viewer()
viewer.add_image(image, rgb=True, name="RGB cell")
napari.run()