"""Session 5: watershed separation of touching nuclei.
uv run src/dapi_watershed_demo.py
Try --min-distance 10 or --min-distance 25 to compare seeds.
Add --exclude-border for the following class exercise, or --no-view to save only.
"""
import argparse
from xml.etree import ElementTree as ET
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.segmentation import watershed, clear_border, relabel_sequential
from dapi_threshold_demo import SOURCE, REPO


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-distance',type=int,default=15)
    parser.add_argument('--exclude-border',action='store_true')
    parser.add_argument('--no-view',action='store_true')
    args=parser.parse_args()
    if args.min_distance<1: parser.error('--min-distance must be positive')
    with tifffile.TiffFile(SOURCE) as tif:
        pixels=next(e for e in ET.fromstring(tif.ome_metadata).iter() if e.tag.endswith('}Pixels'))
        channels=[e.attrib.get('Name','') for e in pixels if e.tag.endswith('}Channel')]
        if tif.series[0].axes!='CYX': raise ValueError('Expected CYX image.')
        dapi=tif.asarray()[channels.index('DAPI')]
        meta={k:pixels.attrib[k] for k in ('PhysicalSizeX','PhysicalSizeY','PhysicalSizeXUnit','PhysicalSizeYUnit')}
        for k in ('PhysicalSizeX','PhysicalSizeY'): meta[k]=float(meta[k])
    threshold=float(threshold_otsu(dapi))
    footprint=ndi.generate_binary_structure(2,1)
    mask=ndi.binary_erosion(dapi>threshold,structure=footprint,iterations=3)
    mask=ndi.binary_dilation(mask,structure=footprint,iterations=3)
    connected=label(mask,connectivity=2)
    # Padding treats space beyond the image as background for distance measurement.
    distance=ndi.distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]
    peaks=peak_local_max(distance,min_distance=args.min_distance,labels=connected,
                         exclude_border=False,p_norm=2)
    markers=np.zeros(mask.shape,dtype=np.int32)
    markers[tuple(peaks.T)]=np.arange(1,len(peaks)+1)
    # Ensure a small component never disappears solely for lack of a seed.
    seeded=set(connected[tuple(peaks.T)].tolist())
    extra=[]
    for obj in range(1,int(connected.max())+1):
        if obj in seeded: continue
        positions=np.argwhere(connected==obj)
        row,col=positions[np.argmax(distance[tuple(positions.T)])]
        markers[row,col]=int(markers.max())+1
        extra.append((row,col))
    all_peaks=np.vstack((peaks,np.array(extra,dtype=int).reshape(-1,2)))
    labels=watershed(-distance,markers,mask=mask,connectivity=ndi.generate_binary_structure(2,2))
    assert np.array_equal(labels>0,mask), 'Watershed should cover the entire foreground.'
    final=clear_border(labels) if args.exclude_border else labels.copy()
    final,_,_=relabel_sequential(final)
    if args.exclude_border:
        assert not any(np.any(edge) for edge in (final[0],final[-1],final[:,0],final[:,-1]))
    output=REPO/'Pictures'/'session5_segmentation'/f'watershed_distance_{args.min_distance}'
    output.mkdir(parents=True,exist_ok=True)
    filename='labels_no_border.ome.tif' if args.exclude_border else 'labels.ome.tif'
    tifffile.imwrite(output/filename,final.astype(np.uint32),ome=True,metadata={'axes':'YX',**meta})
    np.savetxt(output/'seeds.csv',all_peaks,delimiter=',',header='row,column',comments='',fmt='%d')
    print(f'Otsu={threshold:g}; three erosions then three dilations; seed spacing={args.min_distance} pixels.')
    print(f'Before watershed: {connected.max()} connected regions. After watershed: {labels.max()} regions.')
    print(f'After optional border exclusion: {final.max()} regions (exclude-border={args.exclude_border}).')
    print('Region counts require visual validation: check merged nuclei and nuclei split too many times.')
    print(f'Saved {output/filename}')
    if args.no_view: return
    import napari
    scale=(meta['PhysicalSizeY'],meta['PhysicalSizeX'])
    viewer=napari.Viewer(title=f'Watershed: minimum seed distance {args.min_distance} pixels')
    viewer.add_image(dapi,name='Original DAPI',scale=scale,colormap='gray',contrast_limits=(200,6000))
    viewer.add_image(distance.astype(np.float32),name='Distance to background (toggle)',scale=scale,visible=False)
    viewer.add_labels(connected,name='Before: connected objects',scale=scale,visible=False,opacity=.4)
    viewer.add_labels(final,name='After: watershed labels',scale=scale,opacity=.4)
    viewer.add_points(all_peaks,name='Seeds (toggle)',scale=scale,size=5,face_color='yellow',visible=False)
    print('Toggle before/after one at a time; inspect touching nuclei. Seeds are proposed centers, not confirmed nuclei.')
    napari.run()


if __name__=='__main__': main()
