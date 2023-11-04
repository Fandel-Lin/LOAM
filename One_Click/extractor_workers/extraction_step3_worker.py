import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
from scipy import ndimage

def extraction_step3_worker(legend, map_name, legend_name, solutiona_dir, print_intermediate_image, rgb_rb, rgb_ms, this_current_result, color_space_subset, iteration_relaxing, img_crop_black, img_crop_gray, global_solution_empty):
    # fetch current result for this legend
    #this_current_result = np.copy(ans_category[legend])

    # create a mask to only preserve current legend color in the basemap
    rgb_ms_masked = cv2.inRange(rgb_ms, color_space_subset[4 + iteration_relaxing*2], color_space_subset[5 + iteration_relaxing*2])
    rgb_rb_masked = cv2.inRange(rgb_rb, color_space_subset[4 + iteration_relaxing*2], color_space_subset[5 + iteration_relaxing*2])
    #rgb_masked = cv2.inRange(rgb_rb, color_space_subset[4 + iteration_relaxing*2], color_space_subset[5 + iteration_relaxing*2])
    rgb_masked = cv2.bitwise_or(rgb_ms_masked, rgb_rb_masked)
    rgb_masked = rgb_masked.astype('uint8')

    # remove moisy white pixels before buffer
    kernel_before_blur = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening_before_blur = cv2.morphologyEx(this_current_result, cv2.MORPH_OPEN, kernel_before_blur, iterations=1)

    # smooth the image
    blur_radius = 1.0
    threshold_blur = 0
    gaussian_buffer = ndimage.gaussian_filter(opening_before_blur, blur_radius)
    gaussian_buffer[gaussian_buffer > threshold_blur] = 255

    current_empty = 255 - this_current_result
    current_relaxing = cv2.bitwise_and(current_empty, gaussian_buffer)

    relaxing_mask = cv2.bitwise_or(rgb_masked, img_crop_black)
    relaxing_mask = cv2.bitwise_or(relaxing_mask, img_crop_gray)

    relaxing_mask = cv2.bitwise_and(relaxing_mask, global_solution_empty)
    relaxing_mask = relaxing_mask.astype('uint8')
    temp_relax = cv2.bitwise_and(current_relaxing, current_relaxing, mask=relaxing_mask)

    this_next_result = cv2.bitwise_or(this_current_result, temp_relax)

    if iteration_relaxing == 3:
        if print_intermediate_image == True:
            out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_v3.png'
            cv2.imwrite(out_file_path0, this_next_result)

    return legend, this_next_result
    