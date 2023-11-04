import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
from scipy import ndimage

def extraction_step4_worker(legend, map_name, legend_name, solutiona_dir, print_intermediate_image, rgb_rb, rgb_ms, hsv_ms, this_current_result, color_space_subset, iteration, global_solution_empty, img_crop_black_and_gray):
    # fetch current result for this legend
    #this_current_result = np.copy(ans_category[legend])

    if iteration == 0:
        updated_for_relaxing = np.ones((this_current_result.shape[0],this_current_result.shape[1]),dtype=np.uint8)*255
        # remove moisy white pixels before buffer
        kernel_before_blur = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening_before_blur = cv2.morphologyEx(this_current_result, cv2.MORPH_OPEN, kernel_before_blur, iterations=1)
    else:
        #updated_for_relaxing = np.copy(updated_region[legend])
        #opening_before_blur = np.copy(updated_for_relaxing)
        updated_for_relaxing = np.ones((this_current_result.shape[0],this_current_result.shape[1]),dtype=np.uint8)*255
        # remove moisy white pixels before buffer
        kernel_before_blur = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening_before_blur = cv2.morphologyEx(this_current_result, cv2.MORPH_OPEN, kernel_before_blur, iterations=1)

        #print(np.sum(updated_for_relaxing))
        if np.sum(updated_for_relaxing) == 0:
            if print_intermediate_image == True: # iteration == 1
                # remove noisy white pixel
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                opening = cv2.morphologyEx(this_current_result, cv2.MORPH_OPEN, kernel, iterations=1)
                this_next_result =cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY)[1]

                out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_v4v.png'
                cv2.imwrite(out_file_path0, this_next_result)
            return legend, this_current_result, updated_for_relaxing, False
        masked_update_rgb = cv2.bitwise_and(rgb_rb, rgb_rb, mask=updated_for_relaxing)

        color_dynamic_placeholder_rgb_0 = []
        color_dynamic_placeholder_rgb_1 = []
        for dimension in range(0, 3):
            masked_update_np = np.copy(masked_update_rgb[:,:,dimension]).astype(float)
            masked_update_np[masked_update_np==0] = np.nan

            color_dynamic_placeholder_rgb_0.append(min(int(np.nanquantile(masked_update_np,.05)), 254))
            color_dynamic_placeholder_rgb_1.append(max(min(int(np.nanquantile(masked_update_np,.05)), 254), min(int(np.nanquantile(masked_update_np,.95)), 254))) # prevent (255,255,255)
        

        masked_update_hsv = cv2.bitwise_and(hsv_ms, hsv_ms, mask=updated_for_relaxing)

        color_dynamic_placeholder_hsv_0 = []
        color_dynamic_placeholder_hsv_1 = []
        for dimension in range(0, 3):
            masked_update_np = np.copy(masked_update_hsv[:,:,dimension]).astype(float)
            masked_update_np[masked_update_np==0] = np.nan

            color_dynamic_placeholder_hsv_0.append(int(np.nanquantile(masked_update_np,.05)))
            color_dynamic_placeholder_hsv_1.append(int(np.nanquantile(masked_update_np,.95)))



    # smooth the image
    if iteration == 0:
        blur_radius = 5.0 # 5.0
        threshold_blur = 255*0.33 # *0.33
    else:
        blur_radius = 10.0 # 5.0
        threshold_blur = 255*0.33 # *0.33
    gaussian_buffer = ndimage.gaussian_filter(opening_before_blur, blur_radius)
    gaussian_buffer[gaussian_buffer > threshold_blur] = 255
    gaussian_buffer[gaussian_buffer <= threshold_blur] = 0

    current_empty = 255 - this_current_result
    current_relaxing = cv2.bitwise_and(current_empty, gaussian_buffer)
    #current_relaxing = cv2.bitwise_and(current_relaxing, updated_for_relaxing)  # shall not bitwise_and here <= already considered from gaussian_buffer <= opening_before_blur

    current_relaxing = cv2. bitwise_and(current_relaxing, global_solution_empty)
    #current_relaxing_arg = np.argwhere(current_relaxing == 255)

    updated_for_relaxing = np.zeros((this_current_result.shape[0],this_current_result.shape[1]),dtype=np.uint8)


    if iteration == 0:
        rgb_ms_masked = cv2.inRange(rgb_ms, color_space_subset[4 + 3*2], color_space_subset[5 + 3*2])
        rgb_rb_masked = cv2.inRange(rgb_rb, color_space_subset[4 + 3*2], color_space_subset[5 + 3*2])
        #rgb_masked_dynamic = cv2.inRange(rgb_rb, color_space_subset[4 + 3*2], color_space_subset[5 + 3*2])

        rgb_masked_dynamic = cv2.bitwise_or(rgb_ms_masked, rgb_rb_masked)
        rgb_masked_dynamic = rgb_masked_dynamic.astype('uint8')
    else:
    #else:
        rgb_dynamic_lower_box = np.array([max(0, color_dynamic_placeholder_rgb_0[0]-2), max(0, color_dynamic_placeholder_rgb_0[1]-2), max(0, color_dynamic_placeholder_rgb_0[2]-2)])
        rgb_dynamic_upper_box = np.array([min(255, color_dynamic_placeholder_rgb_1[0]+2), min(255, color_dynamic_placeholder_rgb_1[1]+2), min(255, color_dynamic_placeholder_rgb_1[2]+2)])

        # create a mask to only preserve current legend color in the basemap
        rgb_masked_dynamic = cv2.inRange(rgb_rb, rgb_dynamic_lower_box, rgb_dynamic_upper_box)
        #rgb_rb_dynamic_masked = cv2.inRange(rgb_rb, rgb_dynamic_lower_box, rgb_dynamic_upper_box)
        #rgb_ms_dynamic_masked = cv2.inRange(rgb_ms, rgb_dynamic_lower_box, rgb_dynamic_upper_box)
        #rgb_masked_dynamic = cv2.bitwise_or(rgb_rb_dynamic_masked, rgb_ms_dynamic_masked)
        rgb_masked_dynamic = rgb_masked_dynamic.astype('uint8')

        hsv_dynamic_lower_box = np.array([max(0, color_dynamic_placeholder_hsv_0[0]-2), max(0, color_dynamic_placeholder_hsv_0[1]-20), max(0, color_dynamic_placeholder_hsv_0[2]-20)])
        hsv_dynamic_upper_box = np.array([min(255, color_dynamic_placeholder_hsv_1[0]+2), min(255, color_dynamic_placeholder_hsv_1[1]+20), min(255, color_dynamic_placeholder_hsv_1[2]+20)])

        # create a mask to only preserve current legend color in the basemap
        hsv_masked_dynamic = cv2.inRange(hsv_ms, hsv_dynamic_lower_box, hsv_dynamic_upper_box)
        hsv_masked_dynamic = hsv_masked_dynamic.astype('uint8')

        rgb_masked_dynamic = cv2.bitwise_or(rgb_masked_dynamic, hsv_masked_dynamic)
        rgb_masked_dynamic = rgb_masked_dynamic.astype('uint8')



    masking_targeted_color = cv2.bitwise_and(current_relaxing, rgb_masked_dynamic)
    masking_black_and_gray = cv2.bitwise_and(current_relaxing, img_crop_black_and_gray)
    #masking_self = cv2.bitwise_and(current_relaxing, this_current_result)

    if iteration == 0:
        this_next_result = cv2.bitwise_or(this_current_result, masking_targeted_color)
        this_next_result = cv2.bitwise_or(this_next_result, masking_black_and_gray)
    else:
        this_next_result = cv2.bitwise_or(this_current_result, masking_targeted_color)
        this_next_result = cv2.bitwise_or(this_next_result, masking_black_and_gray)

    updated_for_relaxing = cv2.subtract(this_next_result, this_current_result)

    if print_intermediate_image == True:
        if iteration == 0:
            out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_v4.png'
            cv2.imwrite(out_file_path0, this_next_result)
        elif iteration == 1:
            # remove noisy white pixel
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            opening = cv2.morphologyEx(this_next_result, cv2.MORPH_OPEN, kernel, iterations=1)
            this_next_result =cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY)[1]

            out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_v4v.png'
            cv2.imwrite(out_file_path0, this_next_result)

    return legend, this_next_result, updated_for_relaxing, True