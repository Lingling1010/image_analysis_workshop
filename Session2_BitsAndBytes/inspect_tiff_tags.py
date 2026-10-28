from pathlib import Path
import tifffile

repo = Path(__file__).resolve().parent.parent
path = repo / "testImages" / "Cell_cropped.tif"

with tifffile.TiffFile(path) as tif:
    print(f"File: {path.name}")
    print(f"Number of TIFF pages: {len(tif.pages)}")

    for index, series in enumerate(tif.series):
        print(f"\nSeries {index}")
        print(f"  Shape: {series.shape}")
        print(f"  Axes: {series.axes}")
        print(f"  Data type: {series.dtype}")

    for index, page in enumerate(tif.pages):
        print(f"\n--- Page {index}: TIFF tags ---")
        for tag in page.tags.values():
            print(f"\n{tag.name} (code {tag.code}):")
            print(tag.value)