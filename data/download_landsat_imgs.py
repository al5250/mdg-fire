import sys
import pdb
import time

import ee
import numpy as np
from numpy import ndarray
import pandas as pd


BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
SCALE = 30
BINS = 32
MIN_VAL = 0
MAX_VAL = 4000
NUM_FILES = 4


def parse_grid_file(grid_file: str) -> ndarray:
    df = pd.read_csv(grid_file)
    vals = df.values
    str_to_arr = lambda s: np.array([float(x) for x in s.split(',')])
    vals = np.array([[str_to_arr(x2) for x2 in x1] for x1 in vals])
    return vals


def cloudMaskL457(image):
    qa = image.select('pixel_qa')
    # If the cloud bit (5) is set and the cloud confidence (7) is high
    # or the cloud shadow bit is set (3), then it's a bad pixel.
    cloud = qa.bitwiseAnd(1 << 5).And(qa.bitwiseAnd(1 << 7)).Or(qa.bitwiseAnd(1 << 3))
    # Remove edge pixels that don't occur in all bands
    # var mask2 = image.mask().reduce(ee.Reducer.min());
    return image.updateMask(cloud.Not())# .updateMask(mask2)


def create_file(filename):
    with open(filename, 'w') as f:
        f.write(
            'rect_id,date,band,count,' + ','.join(['bin_' + str(i) for i in range(BINS)])
        )


def get_band_data(img_data, band):
    return np.array(img_data.get(band).getInfo())


def main(grid_file: str, out_file: str):

    # Initialize earth engine
    ee.Initialize()

    dataset = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR') \
            .filterDate('2014-01-01', '2020-12-31') \
    
    grid = parse_grid_file(grid_file)
    num_rects = len(grid)
    file_splits = np.linspace(0, num_rects, NUM_FILES + 1).astype('int')
    print('File Splits: ', file_splits)
    file_id = 1

    create_file(f"{file_id}_{out_file}")
    
    for rect_id in range(num_rects):

        # Start time
        start_time = time.time()

        rect = ee.Geometry.Polygon(grid[rect_id].tolist(), None, False)
        rect_imgs = dataset.filterBounds(rect).map(cloudMaskL457).select(BANDS)
        num_imgs = rect_imgs.size().getInfo()

        # Extract dates
        dates = rect_imgs.map(
            lambda x: ee.Feature(None, {'date': x.date().format('YYYY-MM-dd')})
        ).aggregate_array('date').getInfo()

        # Extract histograms
        def reduce_region(img):
            hist = img.reduceRegion(
                reducer=ee.Reducer.fixedHistogram(MIN_VAL, MAX_VAL, BINS),
                geometry=rect,
                maxPixels=1e13,
                scale=SCALE
            )
            return ee.Feature(None, {'hist': hist})
        hists = rect_imgs.map(
            lambda x: reduce_region(x)
        ).aggregate_array('hist').getInfo()

        # Log data
        for date, hist_dict in zip(dates, hists):
            for band in BANDS:
                band_data = np.array(hist_dict[band])
                
                with open(f"{file_id}_{out_file}", 'a') as f:
                    f.write('\n')
                    f.write(f"{rect_id},{date},{band},")
                    if band_data.size == 1 and band_data.item() is None:
                        num_proc = 0
                        f.write('0.0,')
                        f.write(','.join(['0.0'] * BINS))
                    else:
                        try:
                            num_proc = np.sum(band_data[:, 1])
                            f.write(str(num_proc) + ',')
                            f.write(','.join(band_data[:, 1].astype(str)))
                        except:
                            pdb.set_trace()

        # Update file id if necessary
        if rect_id >= file_splits[file_id]:
            file_id += 1
            create_file(f"{file_id}_{out_file}")
        
        # End Time
        print(f'Finished RectID={rect_id} in {time.time() - start_time:3f} seconds.')


if __name__ == '__main__':
    main(*sys.argv[1:])

