"""Session 5: apply a manual DAPI threshold and inspect the mask.
uv run src/dapi_threshold_demo.py --threshold 700
Add --no-view to save without opening napari.
"""
import argparse
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
import tifffile

REPO=Path(__file__).resolve().parent.parent
SOURCE=REPO/'testImages'/'3Channel_FF_Corrected_1'/'3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threshold',type=float,default=700)
    parser.add_argument('--no-view',action='store_true')
    args=parser.parse_args()
    if not np.isfinite(args.threshold): parser.error('Threshold must be finite.')
    with tifffile.TiffFile(SOURCE) as tif:
        root=ET.fromstring(tif.ome_metadata)
        pixels=next(el for el in root.iter() if el.tag.endswith('}Pixels'))
        channels=[el.attrib.get('Name','') for el in pixels if el.tag.endswith('}Channel')]
        if tif.series[0].axes!='CYX': raise ValueError('Expected channels, rows, columns.')
        dapi=tif.asarray()[channels.index('DAPI')]
        sx=float(pixels.attrib['PhysicalSizeX']); sy=float(pixels.attrib['PhysicalSizeY'])
        ux=pixels.attrib.get('PhysicalSizeXUnit','µm'); uy=pixels.attrib.get('PhysicalSizeYUnit','µm')
    # One explicit comparison: strictly above threshold is foreground.
    mask=dapi>args.threshold
    output=REPO/'Pictures'/'session5_segmentation'
    output.mkdir(parents=True,exist_ok=True)
    filename=output/f'dapi_mask_threshold_{args.threshold:g}.ome.tif'
    tifffile.imwrite(filename,mask.astype(np.uint8),ome=True,
                     metadata={'axes':'YX','PhysicalSizeX':sx,'PhysicalSizeXUnit':ux,
                               'PhysicalSizeY':sy,'PhysicalSizeYUnit':uy})
    print(f'DAPI channel index: {channels.index("DAPI")}; shape: {dapi.shape}')
    print(f'Threshold: DAPI > {args.threshold:g}; foreground pixels: {mask.sum()} ({100*mask.mean():.2f}%)')
    print(f'Saved {filename}; 0=background, 1=foreground (not individual nuclei).')
    if args.no_view: return
    import napari
    viewer=napari.Viewer(title=f'DAPI manual threshold: {args.threshold:g}')
    viewer.add_image(dapi,name='Original DAPI',scale=(sy,sx),colormap='gray',
                     contrast_limits=(200,6000))
    viewer.add_labels(mask.astype(np.uint8),name=f'Mask: DAPI > {args.threshold:g}',
                      scale=(sy,sx),opacity=.35)
    print('Toggle mask visibility to inspect dim nuclei, background speckles and joined nuclei.')
    print('Changing contrast limits changes display ONLY; it does not change the threshold.')
    napari.run()


if __name__=='__main__': main()
