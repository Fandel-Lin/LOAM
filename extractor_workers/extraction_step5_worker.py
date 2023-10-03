import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
from scipy import ndimage

def extraction_step5_worker(legend, map_name, legend_name, solutiona_dir, print_intermediate_image, rgb_rb, hsv_ms, this_current_result, iteration, global_solution_empty, img_crop_black_and_gray, conv_kernel_set, conv_kernel_threshold, masking):
    # fetch current result for this legend
    #this_current_result = np.copy(ans_category[legend])

    if iteration == 0:
        updated_for_relaxing = np.ones((this_current_result.shape[0],this_current_result.shape[1]),dtype=np.uint8)*255
        # remove moisy white pixels before buffer
        kernel_before_blur = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening_before_blur = cv2.morphologyEx(this_current_result, cv2.MORPH_OPEN, kernel_before_blur, iterations=1)
    else:
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

                out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_v5v.png'
                cv2.imwrite(out_file_path0, this_next_result)
            return legend, this_current_result, updated_for_relaxing, False
        masked_update = cv2.bitwise_and(rgb_rb, rgb_rb, mask=updated_for_relaxing)

        color_dynamic_placeholder_0 = []
        color_dynamic_placeholder_1 = []
        for dimension in range(0, 3):
            masked_update_np = np.copy(masked_update[:,:,dimension]).astype(float)
            masked_update_np[masked_update_np==0] = np.nan

            color_dynamic_placeholder_0.append(min(int(np.nanquantile(masked_update_np,.05)), 254))
            color_dynamic_placeholder_1.append(max(min(int(np.nanquantile(masked_update_np,.05)), 254), min(int(np.nanquantile(masked_update_np,.95)), 254))) # prevent (255,255,255)
        #print(color_dynamic_placeholder_0, color_dynamic_placeholder_1)
        

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
        blur_radius = 5.0
        threshold_blur = 255*0.0
    else:
        blur_radius = 5.0
        threshold_blur = 255*0.0
    gaussian_buffer = ndimage.gaussian_filter(opening_before_blur, blur_radius)
    gaussian_buffer[gaussian_buffer > threshold_blur] = 255
    gaussian_buffer[gaussian_buffer <= threshold_blur] = 0

    current_empty = 255 - this_current_result
    current_relaxing = cv2.bitwise_and(current_empty, gaussian_buffer)
    #current_relaxing = cv2.bitwise_and(current_relaxing, updated_for_relaxing) # shall not bitwise_and here <= already considered from gaussian_buffer <= opening_before_blur

    current_relaxing = cv2. bitwise_and(current_relaxing, global_solution_empty)
    #current_relaxing_arg = np.argwhere(current_relaxing == 255)

    updated_for_relaxing = np.zeros((this_current_result.shape[0],this_current_result.shape[1]),dtype=np.uint8)



    this_current_result_scale = np.copy(this_current_result)
    this_current_result_scale[this_current_result_scale > 0] = 1
    this_current_result_scale_v0 = np.copy(this_current_result_scale)
    if iteration == 0:
        conv_masking_targeted = np.zeros((this_current_result.shape[0],this_current_result.shape[1]),dtype=np.uint8)
        for conv_set in range(0, len(conv_kernel_set)):
            conv_mask = cv2.filter2D(src=this_current_result_scale, ddepth=-1, kernel=conv_kernel_set[conv_set])
            conv_out_placeholder = np.copy(this_current_result_scale)
            conv_out_placeholder[np.logical_and(conv_mask>=conv_kernel_threshold[conv_set], this_current_result_scale_v0==0)] = 1
            conv_out_placeholder[conv_out_placeholder > 0] = 255
            conv_masking_targeted = cv2.bitwise_or(conv_masking_targeted, conv_out_placeholder)

        masking_targeted_conv = cv2.bitwise_and(current_relaxing, conv_masking_targeted)


        updated_from_filtering = np.zeros((this_current_result_scale.shape[0],this_current_result_scale.shape[1]),dtype=np.uint8)
        updated_record = np.zeros((this_current_result_scale.shape[0],this_current_result_scale.shape[1]),dtype=np.uint8)
        updated_record[0:this_current_result_scale.shape[0], 0:this_current_result_scale.shape[1]] = 3
        tabo_list = np.zeros((this_current_result_scale.shape[0],this_current_result_scale.shape[1]),dtype=np.uint8)


        for this_direction in range(0, 8+2):
            direction = this_direction
            if this_direction >= 8:
                direction = this_direction - 8

            dir_filtered = cv2.filter2D(src=this_current_result_scale, ddepth=-1, kernel=masking[direction]) # img_bw_for_filter
            #print(np.unique(dir_filtered))
            dir_filtered[dir_filtered <= 255*0.5] = 0
            dir_filtered[dir_filtered > 255*0.5] = 255 / 8.0

            #directional_filter.append(dir_filtered)
            if this_direction < 8:
                updated_from_filtering = cv2.add(updated_from_filtering, dir_filtered)

            dir_record = np.zeros((this_current_result_scale.shape[0],this_current_result_scale.shape[1]),dtype=np.uint8)
            dir_record[dir_filtered > 0.0] = 1
            updated_record = cv2.add(updated_record, dir_record)
            updated_record = cv2.subtract(updated_record, np.ones((this_current_result_scale.shape[0],this_current_result_scale.shape[1]),dtype=np.uint8))
            #print('unique value(s):', np.unique(updated_record))

            if this_direction >= 3:
                updated_tabo = np.zeros((this_current_result_scale.shape[0],this_current_result_scale.shape[1]),dtype=np.uint8)
                updated_tabo[updated_record <= 0] = 1
                tabo_list = cv2.bitwise_or(tabo_list, updated_tabo)
        
        tabo_list[tabo_list > 0] = 255
        tabo_list = tabo_list - 255

        updated_with_checking = np.copy(updated_from_filtering)
        updated_with_checking[updated_from_filtering < 255*3/8] = 0
        updated_with_checking[updated_from_filtering >= 255*3/8] = 255
        updated_with_checking[updated_from_filtering > 255*5/8] = 0
        updated_with_checking = cv2.bitwise_and(updated_with_checking, tabo_list)

        updated_from_filtering[updated_from_filtering <= 255*5/8] = 0
        updated_from_filtering[updated_from_filtering > 255*5/8] = 255


        this_next_result = cv2.bitwise_or(this_current_result, masking_targeted_conv)
        this_next_result = cv2.bitwise_or(this_next_result, updated_from_filtering)
    elif iteration > 0:
        rgb_dynamic_lower_box = np.array([color_dynamic_placeholder_0[0], color_dynamic_placeholder_0[1], color_dynamic_placeholder_0[2]])
        rgb_dynamic_upper_box = np.array([color_dynamic_placeholder_1[0], color_dynamic_placeholder_1[1], color_dynamic_placeholder_1[2]])
        #print(rgb_dynamic_lower_box, rgb_dynamic_upper_box)

        # create a mask to only preserve current legend color in the basemap
        rgb_masked_dynamic = cv2.inRange(rgb_rb, rgb_dynamic_lower_box, rgb_dynamic_upper_box)
        rgb_masked_dynamic[rgb_masked_dynamic > 0] = 1
        rgb_masked_dynamic = rgb_masked_dynamic.astype('uint8')

        hsv_dynamic_lower_box = np.array([color_dynamic_placeholder_hsv_0[0], color_dynamic_placeholder_hsv_0[1], color_dynamic_placeholder_hsv_0[2]])
        hsv_dynamic_upper_box = np.array([color_dynamic_placeholder_hsv_1[0], color_dynamic_placeholder_hsv_1[1], color_dynamic_placeholder_hsv_1[2]])
        #print(hsv_dynamic_lower_box, hsv_dynamic_upper_box)

        # create a mask to only preserve current legend color in the basemap
        hsv_masked_dynamic = cv2.inRange(hsv_ms, hsv_dynamic_lower_box, hsv_dynamic_upper_box)
        hsv_masked_dynamic[hsv_masked_dynamic > 0] = 1
        hsv_masked_dynamic = hsv_masked_dynamic.astype('uint8')

        rgb_masked_dynamic = cv2.bitwise_or(rgb_masked_dynamic, hsv_masked_dynamic)
        rgb_masked_dynamic = rgb_masked_dynamic.astype('uint8')


        #masking_targeted_color = cv2.bitwise_and(current_relaxing, rgb_masked_dynamic)
        #masking_black_and_gray = cv2.bitwise_and(current_relaxing, img_crop_black_and_gray)
        #masking_self = cv2.bitwise_and(current_relaxing, this_current_result)

        conv_masking_targeted = np.zeros((this_current_result.shape[0],this_current_result.shape[1]),dtype=np.uint8)
        for conv_set in range(0, len(conv_kernel_set)):
            conv_mask_self = cv2.filter2D(src=this_current_result_scale, ddepth=-1, kernel=conv_kernel_set[conv_set])
            conv_mask_relax = cv2.filter2D(src=rgb_masked_dynamic, ddepth=-1, kernel=conv_kernel_set[conv_set])
            #conv_mask_bg = cv2.filter2D(src=img_crop_black_and_gray, ddepth=-1, kernel=conv_kernel_set[conv_set])

            conv_out_placeholder = np.copy(this_current_result_scale)
            conv_out_placeholder[np.logical_and(conv_mask_self>=conv_kernel_threshold[conv_set], conv_mask_relax>=conv_kernel_threshold[conv_set], this_current_result_scale_v0==0)] = 1
            conv_out_placeholder[conv_out_placeholder > 0] = 255
            conv_masking_targeted = cv2.bitwise_or(conv_masking_targeted, conv_out_placeholder)


        masking_targeted_conv = cv2.bitwise_and(current_relaxing, conv_masking_targeted)


        updated_from_filtering = np.zeros((this_current_result_scale.shape[0],this_current_result_scale.shape[1]),dtype=np.uint8)
        updated_record = np.zeros((this_current_result_scale.shape[0],this_current_result_scale.shape[1]),dtype=np.uint8)
        updated_record[0:this_current_result_scale.shape[0], 0:this_current_result_scale.shape[1]] = 3
        tabo_list = np.zeros((this_current_result_scale.shape[0],this_current_result_scale.shape[1]),dtype=np.uint8)


        for this_direction in range(0, 8+2):
            direction = this_direction
            if this_direction >= 8:
                direction = this_direction - 8

            dir_filtered = cv2.filter2D(src=this_current_result_scale, ddepth=-1, kernel=masking[direction]) # img_bw_for_filter
            dir_filtered[dir_filtered <= 255*0.5] = 0
            dir_filtered[dir_filtered > 255*0.5] = 255 / 8.0

            #directional_filter.append(dir_filtered)
            if this_direction < 8:
                updated_from_filtering = cv2.add(updated_from_filtering, dir_filtered)

            dir_record = np.zeros((this_current_result_scale.shape[0],this_current_result_scale.shape[1]),dtype=np.uint8)
            dir_record[dir_filtered > 0.0] = 1
            updated_record = cv2.add(updated_record, dir_record)
            updated_record = cv2.subtract(updated_record, np.ones((this_current_result_scale.shape[0],this_current_result_scale.shape[1]),dtype=np.uint8))
            #print('unique value(s):', np.unique(updated_record))

            if this_direction >= 3:
                updated_tabo = np.zeros((this_current_result_scale.shape[0],this_current_result_scale.shape[1]),dtype=np.uint8)
                updated_tabo[updated_record <= 0] = 1
                tabo_list = cv2.bitwise_or(tabo_list, updated_tabo)
        
        tabo_list[tabo_list > 0] = 255
        tabo_list = tabo_list - 255

        updated_with_checking = np.copy(updated_from_filtering)
        updated_with_checking[updated_from_filtering < 255*3/8] = 0
        updated_with_checking[updated_from_filtering >= 255*3/8] = 255
        updated_with_checking[updated_from_filtering > 255*5/8] = 0
        updated_with_checking = cv2.bitwise_and(updated_with_checking, tabo_list)

        updated_from_filtering[updated_from_filtering <= 255*5/8] = 0
        updated_from_filtering[updated_from_filtering > 255*5/8] = 255


        this_next_result = cv2.bitwise_or(this_current_result, masking_targeted_conv)
        this_next_result = cv2.bitwise_or(this_next_result, updated_from_filtering)
    

    updated_for_relaxing = cv2.subtract(this_next_result, this_current_result)

    if print_intermediate_image == True:
        if iteration == 0:
            out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_v5.png'
            cv2.imwrite(out_file_path0, this_next_result)
        elif iteration == 1:
            # remove noisy white pixel
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            opening = cv2.morphologyEx(this_current_result, cv2.MORPH_OPEN, kernel, iterations=1)
            this_next_result =cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY)[1]

            out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_v5v.png'
            cv2.imwrite(out_file_path0, this_next_result)

    return legend, this_next_result, updated_for_relaxing, True