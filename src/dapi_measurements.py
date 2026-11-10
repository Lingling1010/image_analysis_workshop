"""Session 5: measure saved border-excluded watershed objects.
Run: uv run src/dapi_measurements.py
Optional: --min-area 100 (pixels); choose a cutoff only after inspecting object sizes.
"""
import argparse
from xml.etree import ElementTree as ET
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from dapi_threshold_demo import SOURCE, REPO


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-area',type=int,default=0)
    args=parser.parse_args()
    if args.min_area<0: parser.error('--min-area must be nonnegative')
    folder=REPO/'Pictures'/'session5_segmentation'/'watershed_distance_15'
    label_path=folder/'labels_no_border.ome.tif'
    if not label_path.exists():
        parser.error('First run: uv run src/dapi_watershed_demo.py --exclude-border --no-view')
    labels=tifffile.imread(label_path)
    with tifffile.TiffFile(SOURCE) as t:
        pixels=next(e for e in ET.fromstring(t.ome_metadata).iter() if e.tag.endswith('}Pixels'))
        names=[e.attrib.get('Name','') for e in pixels if e.tag.endswith('}Channel')]
        if t.series[0].axes!='CYX': raise ValueError('Expected CYX image.')
        image=t.asarray()[names.index('DAPI')]
        sx=float(pixels.attrib['PhysicalSizeX']); sy=float(pixels.attrib['PhysicalSizeY'])
        if pixels.attrib.get('PhysicalSizeXUnit')!='µm' or pixels.attrib.get('PhysicalSizeYUnit')!='µm':
            raise ValueError('Expected physical calibration in micrometers.')
    assert labels.shape==image.shape
    # Exclude ALL bright structures, including border objects discarded from labels.
    # Global background is an initial estimate, not a correction for uneven illumination.
    exclusion=ndi.binary_dilation(image>threshold_otsu(image),iterations=5)
    background_pixels=image[~exclusion]
    if not background_pixels.size: raise ValueError('No background pixels available.')
    background=float(np.median(background_pixels))
    rows=[]
    for region in regionprops(labels,intensity_image=image):
        if region.area<args.min_area: continue
        values=image[region.slice][region.image].astype(float)
        # Physical shape from calibrated coordinates (supports unequal X/Y spacing).
        coords=region.coords.astype(float)*np.array([sy,sx])
        covariance=np.cov(coords,rowvar=False,bias=True) if len(coords)>1 else np.zeros((2,2))
        eig=np.maximum(np.linalg.eigvalsh(covariance),0)
        minor,major=4*np.sqrt(eig)
        rows.append(dict(object_id=int(region.label),area_pixels=int(region.area),
                         area_um2=float(region.area*sx*sy),centroid_row=region.centroid[0],
                         centroid_column=region.centroid[1],major_axis_um=major,minor_axis_um=minor,
                         major_minor_ratio=major/minor if minor>0 else np.nan,
                         mean_dapi=float(values.mean()),std_dapi=float(values.std()),
                         integrated_dapi=float(values.sum()),background_estimate=background,
                         corrected_mean_dapi=float(values.mean()-background),
                         corrected_integrated_dapi=float(values.sum()-len(values)*background)))
    out=REPO/'Pictures'/'session5_segmentation'/'measurements'
    out.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0]) if rows else ['object_id','area_pixels']
    csv_path=out/f'nucleus_measurements_min_area_{args.min_area}.csv'
    with csv_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key,title in zip(axes,['area_um2','major_minor_ratio','corrected_integrated_dapi'],
                            ['Area (µm²)','Major/minor axis ratio','Background-corrected integrated DAPI']):
        data=np.array([r[key] for r in rows]); data=data[np.isfinite(data)]
        ax.hist(data,bins=min(15,max(1,len(data)//3)))
        ax.set(title=title,xlabel=title,ylabel='Object count')
    fig.suptitle(f'{len(rows)} regions; min area {args.min_area} pixels; masks require validation')
    fig.tight_layout(); fig.savefig(out/f'measurement_histograms_min_area_{args.min_area}.png',dpi=150); plt.close(fig)
    notes=[f'Measured {len(rows)} regions; size cutoff {args.min_area} pixels (0 means none).',
           f'Pixel size X={sx}, Y={sy} µm; global background median={background:g}.',
           'Corrected integrated DAPI = sum within mask - area_pixels * background.',
           'Axis ratio near 1 is roughly round; elongated objects have a higher ratio.',
           'No clipping of negative corrected values. Global background does not fix spatial gradients.',
           'Metaphase candidates need inspection: compact/bright or irregular objects are not proof of metaphase.',
           'DNA-content analysis uses corrected integrated DAPI, not mean alone; validate mask completeness, linear unsaturated acquisition and staining.',
           'Do not assign G1/G2 from brightness alone: merged objects, segmentation errors and incomplete masks can mimic DNA-content differences.']
    (out/'measurement_notes.txt').write_text('\n'.join(notes)+'\n')
    print('\n'.join(notes)); print(f'Saved {csv_path}')
    print(f'Histogram folder: {out}')
    assert all(np.isclose(r['corrected_integrated_dapi'],r['corrected_mean_dapi']*r['area_pixels']) for r in rows)


if __name__=='__main__': main()
