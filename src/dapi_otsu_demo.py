"""Session 5: Otsu threshold, connected objects, and size histogram.
Run: uv run src/dapi_otsu_demo.py
Add --no-view to generate outputs only.
"""
import argparse
from pathlib import Path
from xml.etree import ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.filters import threshold_otsu
from skimage.measure import label
from dapi_threshold_demo import SOURCE, REPO


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-view',action='store_true')
    args=parser.parse_args()
    with tifffile.TiffFile(SOURCE) as tif:
        root=ET.fromstring(tif.ome_metadata)
        pixels=next(el for el in root.iter() if el.tag.endswith('}Pixels'))
        channels=[el.attrib.get('Name','') for el in pixels if el.tag.endswith('}Channel')]
        if tif.series[0].axes!='CYX': raise ValueError('Expected CYX image.')
        image=tif.asarray()[channels.index('DAPI')]
        metadata={key:pixels.attrib[key] for key in
                  ('PhysicalSizeX','PhysicalSizeY','PhysicalSizeXUnit','PhysicalSizeYUnit')}
        for key in ('PhysicalSizeX','PhysicalSizeY'): metadata[key]=float(metadata[key])
    threshold=float(threshold_otsu(image))
    mask=image>threshold
    # Explicitly use 8-connectivity: diagonal contact joins pixels into one object.
    labels=label(mask,connectivity=2)
    sizes=np.bincount(labels.ravel())[1:]
    count=len(sizes)
    output=REPO/'Pictures'/'session5_segmentation'
    output.mkdir(parents=True,exist_ok=True)
    tifffile.imwrite(output/'dapi_otsu_labels.ome.tif',labels.astype(np.uint32),ome=True,
                     metadata={'axes':'YX',**metadata})
    np.savetxt(output/'otsu_object_sizes.csv',np.column_stack((np.arange(1,count+1),sizes)),
               delimiter=',',header='object_id,area_pixels',comments='',fmt='%d')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    # Full integer-range histogram retains the intensity range without clipping data.
    hist=np.bincount(image.ravel())
    nonzero=np.flatnonzero(hist)
    axes[0].plot(nonzero,hist[nonzero],lw=.8)
    axes[0].set_yscale('log')
    axes[0].axvline(threshold,color='red',label=f'Otsu {threshold:g}')
    axes[0].axvline(700,color='green',ls='--',label='Manual 700')
    axes[0].set(xlabel='DAPI pixel value',ylabel='Pixel count (log)',title='Intensity histogram')
    axes[0].legend()
    if count:
        bins=np.geomspace(max(.5,float(sizes.min())-.5),float(sizes.max())+.5,45)
        axes[1].hist(sizes,bins=bins)
        axes[1].set_xscale('log')
    axes[1].set(xlabel='Object area (pixels; log scale)',ylabel='Number of objects',
                title=f'Otsu: {count} connected objects, no size cutoff')
    fig.tight_layout(); fig.savefig(output/'otsu_histograms.png',dpi=160); plt.close(fig)
    lines=[f'Otsu threshold: {threshold:g}; manual threshold: 700.',
           f'8-connected objects: {count}; NO size filtering or border exclusion.',
           f'Objects of 1–10 pixels: {np.count_nonzero(sizes<=10)}.',
           'These are connected regions, not yet a validated nucleus count.']
    print('\n'.join(lines)); (output/'otsu_summary.txt').write_text('\n'.join(lines)+'\n')
    print(f'Histogram: {output / "otsu_histograms.png"}')
    if args.no_view: return
    import napari
    scale=(metadata['PhysicalSizeY'],metadata['PhysicalSizeX'])
    viewer=napari.Viewer(title=f'Otsu {threshold:g}: {count} connected regions')
    viewer.add_image(image,name='Original DAPI',scale=scale,colormap='gray',contrast_limits=(200,6000))
    viewer.add_labels((image>700).astype(np.uint8),name='Manual threshold 700 (toggle)',
                      scale=scale,opacity=.35,visible=False)
    viewer.add_labels(labels,name=f'Otsu {threshold:g}: labeled regions',scale=scale,opacity=.45)
    print('Toggle Otsu/manual layers one at a time. Check dim nuclei, small speckles and merged nuclei.')
    napari.run()


if __name__=='__main__': main()
