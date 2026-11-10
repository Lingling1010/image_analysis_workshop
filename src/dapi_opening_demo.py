"""Session 5: Otsu, erosion then dilation (two or three rounds).
Run: uv run src/dapi_opening_demo.py
Use --no-view for output files only.
"""
import argparse
from xml.etree import ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure
from skimage.filters import threshold_otsu
from skimage.measure import label
from dapi_threshold_demo import SOURCE, REPO


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-view',action='store_true')
    parser.add_argument('--rounds',type=int,default=2,choices=[2,3])
    parser.add_argument('--labels',action='store_true',help='Show individually numbered objects instead of binary masks.')
    args=parser.parse_args()
    with tifffile.TiffFile(SOURCE) as t:
        pixels=next(e for e in ET.fromstring(t.ome_metadata).iter() if e.tag.endswith('}Pixels'))
        channels=[e.attrib.get('Name','') for e in pixels if e.tag.endswith('}Channel')]
        if t.series[0].axes!='CYX': raise ValueError('Expected CYX image.')
        dapi=t.asarray()[channels.index('DAPI')]
        meta={k:pixels.attrib[k] for k in ('PhysicalSizeX','PhysicalSizeY','PhysicalSizeXUnit','PhysicalSizeYUnit')}
        for k in ('PhysicalSizeX','PhysicalSizeY'): meta[k]=float(meta[k])
    threshold=float(threshold_otsu(dapi))
    before=dapi>threshold
    # Explicit cross-shaped footprint. All iterations use zero outside the image.
    footprint=generate_binary_structure(2,1)
    eroded=binary_erosion(before,structure=footprint,iterations=args.rounds,border_value=0)
    opened=binary_dilation(eroded,structure=footprint,iterations=args.rounds,border_value=0)
    eroded_name=f'eroded_{args.rounds}_rounds'
    opened_name=f'opened_{args.rounds}_rounds'
    stages={'before':before,eroded_name:eroded,opened_name:opened}
    output=REPO/'Pictures'/'session5_segmentation'/('opening' if args.rounds==2 else 'opening_3_rounds')
    output.mkdir(parents=True,exist_ok=True)
    objects={}; sizes={}; summary=[f'Otsu threshold={threshold:g}. Morphology: cross, {args.rounds} erosions THEN {args.rounds} dilations; zero borders.',
                               'Object counting: 8-connectivity, no size filtering or border exclusion.']
    for name,mask in stages.items():
        labels=label(mask,connectivity=2)
        area=np.bincount(labels.ravel())[1:]
        objects[name]=labels; sizes[name]=area
        tifffile.imwrite(output/f'{name}_mask.ome.tif',mask.astype(np.uint8),ome=True,metadata={'axes':'YX',**meta})
        tifffile.imwrite(output/f'{name}_labels.ome.tif',labels.astype(np.uint32),ome=True,metadata={'axes':'YX',**meta})
        np.savetxt(output/f'{name}_sizes.csv',np.column_stack((np.arange(1,len(area)+1),area)),
                   delimiter=',',header='object_id,area_pixels',comments='',fmt='%d')
        summary.append(f'{name}: {len(area)} objects; {np.count_nonzero(area<=10)} objects of 1–10 pixels; foreground={mask.sum()} pixels.')
    assert np.all(~opened | before), 'Opening should not add foreground outside the original mask.'
    assert np.all(~eroded | before)
    summary.append('Counts are connected regions, not yet validated nucleus counts.')
    print('\n'.join(summary)); (output/'summary.txt').write_text('\n'.join(summary)+'\n')
    maximum=max(int(a.max()) for a in sizes.values() if len(a))
    bins=np.geomspace(.5,maximum+.5,45)
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharex=True,sharey=True)
    for ax,name in zip(axes,('before',opened_name)):
        ax.hist(sizes[name],bins=bins)
        ax.set(xscale='log',xlabel='Area (pixels; log scale)',ylabel='Object count',
               title=f'{name}: {len(sizes[name])} objects')
    fig.tight_layout(); fig.savefig(output/'size_histograms.png',dpi=160); plt.close(fig)
    print(f'Saved masks, labels, size tables and histogram to {output}')
    if args.no_view: return
    import napari
    viewer=napari.Viewer(title=f'Otsu: {args.rounds} erosions, then {args.rounds} dilations')
    scale=(meta['PhysicalSizeY'],meta['PhysicalSizeX'])
    viewer.add_image(dapi,name='Original DAPI',scale=scale,colormap='gray',contrast_limits=(200,6000))
    for name,mask in stages.items():
        viewer.add_labels(objects[name] if args.labels else mask.astype(np.uint8),
                          name=name+(' numbered objects' if args.labels else ''),
                          scale=scale,opacity=.4,visible=name==opened_name)
    print('Keep original visible; toggle just ONE mask at a time. Compare small speckles and nuclear borders.')
    if args.labels:
        print('Each connected object has an integer ID. Hover to read it; touching nuclei may share one ID.')
    napari.run()


if __name__=='__main__': main()
