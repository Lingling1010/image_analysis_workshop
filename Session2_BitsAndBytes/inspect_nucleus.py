from pathlib import Path
import sys

import napari
import tifffile

# Locate testImages relative to this script.
repo = Path(__file__).resolve().parent.parent
filename = sys.argv[1] if len(sys.argv) > 1 else "nucleus_cropped_1.tif"
image = tifffile.imread(repo / "testImages" / filename)

print(f"File: {filename}")
print(f"Shape (rows, columns): {image.shape}")
print(f"Data type: {image.dtype}")
print(f"Pixel range: {image.min()} to {image.max()}")

print("\nFirst 10 rows and 10 columns:")
print(image[:10, :10])

print("\n10 × 10 sample, skipping two rows/columns between samples:")
print(image[:30:3, :30:3])

viewer = napari.Viewer()
viewer.add_image(image, name=filename)
napari.run()