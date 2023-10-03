import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
from scipy import ndimage
import rasterio

def extraction_step7_worker(legend, map_name, legend_name, solutiona_dir, file_path, this_current_result, img_bound):
    '''
    # remove noisy white pixel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(ans_category[legend], cv2.MORPH_OPEN, kernel, iterations=1)
    ans_category[legend]=cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY)[1]
    '''
    #this_current_result = np.copy(ans_category[legend])


    if np.unique(this_current_result).shape[0] != 2:
        print('extract nothing...')
        this_current_result = np.copy(img_bound)


    if True: # print_intermediate_image == True:
        out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_v7.png'
        cv2.imwrite(out_file_path0, this_current_result)

    # convert the grayscale image to binary image
    pred_binary_raster = this_current_result.astype(float) / 255

    '''
    # print
    #print('predicted binary raster:')
    #print('shape:', pred_binary_raster.shape)
    #print('unique value(s):', np.unique(pred_binary_raster))
    #print(map_name, '/', legend_name[legend])
    print(legend_name[legend])

    # plot the raster and save it
    plt.imshow(pred_binary_raster)
    plt.show()
    '''


    # save the raster into a .tif file
    out_file_path=solutiona_dir+'intermediate7(2)/Output/'+map_name+'_'+legend_name[legend]+'_poly.tif' # output
    pred_binary_raster=pred_binary_raster.astype('uint16')
    cv2.imwrite(out_file_path, pred_binary_raster)

    # convert the image to a binary raster .tif
    raster = rasterio.open(file_path)
    transform = raster.transform
    # array     = raster.read(1)
    crs       = raster.crs 
    width     = raster.width 
    height    = raster.height 

    raster.close()

    raster = rasterio.open(out_file_path)   
    array  = raster.read(1)
    raster.close()
    with rasterio.open(out_file_path, 'w', 
                        driver    = 'GTIFF', 
                        transform = transform, 
                        dtype     = rasterio.uint8, 
                        count     = 1, 
                        compress  = 'lzw', 
                        crs       = crs, 
                        width     = width, 
                        height    = height) as dst:

        dst.write(array, indexes=1)
        dst.close()
        
    return legend, pred_binary_raster