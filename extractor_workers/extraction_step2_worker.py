import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
from scipy import ndimage

def extraction_step2_worker(legend, map_name, legend_name, solutiona_dir, print_intermediate_image, poly_counter, img_bound_np_sum, hsv_rb, rgb_rb, hsv_ms, rgb_ms, hsv_space, color_space_subset):
    # create a mask to only preserve current legend color in the basemap
    hsv_ms_masked = cv2.inRange(hsv_ms, color_space_subset[0], color_space_subset[1])
    hsv_rb_masked = cv2.inRange(hsv_rb, color_space_subset[0], color_space_subset[1])
    hsv_masked = cv2.bitwise_or(hsv_ms_masked, hsv_rb_masked)

    rgb_ms_masked = cv2.inRange(rgb_ms, color_space_subset[2], color_space_subset[3])
    rgb_rb_masked = cv2.inRange(rgb_rb, color_space_subset[2], color_space_subset[3])
    rgb_masked = cv2.bitwise_or(rgb_ms_masked, rgb_rb_masked)
    img_masked = cv2.bitwise_or(hsv_masked, rgb_masked)


    # dilation if needed
    current_h_lower = color_space_subset[0][0]
    current_h_max = 0
    for h_space in range(current_h_lower, -1, -1):
        if hsv_space[h_space] > 0 and hsv_space[h_space] != legend:
            current_h_max = h_space
            break
    
    current_h_upper = color_space_subset[1][0]
    current_h_min = 255
    for h_space in range(current_h_upper, 255):
        if hsv_space[h_space] > 0 and hsv_space[h_space] != legend:
            current_h_min = h_space
            break
    
    '''
    color_space_subset[0][0] = current_h_lower - min(int((current_h_lower-current_h_max)/2), 2)
    color_space_subset[1][0] = current_h_upper + min(int((current_h_min-current_h_upper)/2), 2)
    color_space_subset[0][1] = color_space_subset[0][1] - 5
    color_space_subset[1][1] = color_space_subset[1][1] + 5
    color_space_subset[0][2] = color_space_subset[0][2] - 5
    color_space_subset[1][2] = color_space_subset[1][2] + 5
    #print(color_space_subset[0], color_space_subset[1])
    '''


    ### Modification 20230731
    '''
    if poly_counter <= 1:
        # dilation if needed
        color_space_subset[0][0] = color_space_subset[0][0] - 20
        color_space_subset[1][0] = color_space_subset[1][0] + 20
        color_space_subset[0][1] = color_space_subset[0][1] - 40
        color_space_subset[1][1] = color_space_subset[1][1] + 40
        color_space_subset[0][2] = color_space_subset[0][2] - 40
        color_space_subset[1][2] = color_space_subset[1][2] + 40
        #print(color_space_subset[0], color_space_subset[1])
        
        hsv_masked = cv2.inRange(hsv_rb, color_space_subset[0], color_space_subset[1])
        
        while poly_counter <= 1 and (np.sum(rgb_masked)/255 <= img_bound_np_sum/255 *0.02):
            # dilation if needed
            for rgb_set in range(2, len(color_space_subset), 2):
                color_space_subset[rgb_set] = color_space_subset[rgb_set] -4
                color_space_subset[rgb_set+1] = color_space_subset[rgb_set+1] +4
            #print(color_space_subset[2], color_space_subset[3])

            rgb_ms_masked = cv2.inRange(rgb_ms, color_space_subset[2], color_space_subset[3])
            rgb_rb_masked = cv2.inRange(rgb_rb, color_space_subset[2], color_space_subset[3])
            rgb_masked = cv2.bitwise_or(rgb_ms_masked, rgb_rb_masked)

            img_masked = cv2.bitwise_or(hsv_masked, rgb_masked)
        
        # remove noisy white pixel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
        opening = cv2.morphologyEx(img_masked, cv2.MORPH_OPEN, kernel, iterations=1)
        img_masked=cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY)[1]
            
        if (np.sum(rgb_masked)/255 <= img_bound_np_sum/255 *0.1):
            for rgb_set in range(len(color_space_subset)-2, len(color_space_subset), 2):
                color_space_subset[rgb_set] = color_space_subset[rgb_set] -15
                color_space_subset[rgb_set+1] = color_space_subset[rgb_set+1] +15
                
    elif poly_counter <= 3:
        # dilation if needed
        color_space_subset[0][0] = color_space_subset[0][0] - 6
        color_space_subset[1][0] = color_space_subset[1][0] + 6
        color_space_subset[0][1] = color_space_subset[0][1] - 20
        color_space_subset[1][1] = color_space_subset[1][1] + 20
        color_space_subset[0][2] = color_space_subset[0][2] - 20
        color_space_subset[1][2] = color_space_subset[1][2] + 20
        #print(color_space_subset[0], color_space_subset[1])
        hsv_ms_masked = cv2.inRange(hsv_ms, color_space_subset[0], color_space_subset[1])
        hsv_rb_masked = cv2.inRange(hsv_rb, color_space_subset[0], color_space_subset[1])
        hsv_masked = cv2.bitwise_or(hsv_ms_masked, hsv_rb_masked)

        rgb_ms_masked = cv2.inRange(rgb_ms, color_space_subset[2], color_space_subset[3])
        rgb_rb_masked = cv2.inRange(rgb_rb, color_space_subset[2], color_space_subset[3])
        rgb_masked = cv2.bitwise_or(rgb_ms_masked, rgb_rb_masked)

        img_masked = cv2.bitwise_and(hsv_masked, hsv_masked)
    '''
    #elif poly_counter > 3:
    if poly_counter > 0:
        #if poly_counter > 3:
        if poly_counter > 0:
            # dilation if needed
            current_h_lower = color_space_subset[0][0]
            current_h_max = 0
            for h_space in range(current_h_lower, -1, -1):
                if hsv_space[h_space] > 0 and hsv_space[h_space] != legend:
                    current_h_max = h_space
                    break
            
            current_h_upper = color_space_subset[1][0]
            current_h_min = 255
            for h_space in range(current_h_upper, 255):
                if hsv_space[h_space] > 0 and hsv_space[h_space] != legend:
                    current_h_min = h_space
                    break
            
            color_space_subset[0][0] = current_h_lower - min(int((current_h_lower-current_h_max)/4), 2)
            color_space_subset[1][0] = current_h_upper + min(int((current_h_min-current_h_upper)/4), 2)
            color_space_subset[0][1] = color_space_subset[0][1] - 5
            color_space_subset[1][1] = color_space_subset[1][1] + 5
            color_space_subset[0][2] = color_space_subset[0][2] - 5
            color_space_subset[1][2] = color_space_subset[1][2] + 5
            #print(color_space_subset[0], color_space_subset[1])

            hsv_ms_masked = cv2.inRange(hsv_ms, color_space_subset[0], color_space_subset[1])
            hsv_rb_masked = cv2.inRange(hsv_rb, color_space_subset[0], color_space_subset[1])
            hsv_masked = cv2.bitwise_or(hsv_ms_masked, hsv_rb_masked)
            
            img_masked = cv2.bitwise_or(hsv_masked, rgb_masked)
        
        hsv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        hsv_opening = cv2.morphologyEx(hsv_masked, cv2.MORPH_OPEN, hsv_kernel, iterations=1)

        '''
        dilation_step = 0
        dilation_step_threshold = 10
        while ((dilation_step<dilation_step_threshold) and (np.sum(hsv_opening)/255 <= img_bound_np_sum/255 *0.0005)) or ((dilation_step>=dilation_step_threshold) and (np.sum(hsv_masked)/255 <= img_bound_np_sum/255 *0.0005)):
            # dilation if needed
            current_h_lower = color_space_subset[0][0]
            current_h_max = 0
            for h_space in range(current_h_lower, -1, -1):
                if hsv_space[h_space] > 0 and hsv_space[h_space] != legend:
                    current_h_max = h_space
                    break
            
            current_h_upper = color_space_subset[1][0]
            current_h_min = 255
            for h_space in range(current_h_upper, 255):
                if hsv_space[h_space] > 0 and hsv_space[h_space] != legend:
                    current_h_min = h_space
                    break
            
            color_space_subset[0][0] = current_h_lower - min(int((current_h_lower-current_h_max)/4), 2)
            color_space_subset[1][0] = current_h_upper + min(int((current_h_min-current_h_upper)/4), 2)
            color_space_subset[0][1] = color_space_subset[0][1] - 5
            color_space_subset[1][1] = color_space_subset[1][1] + 5
            color_space_subset[0][2] = color_space_subset[0][2] - 5
            color_space_subset[1][2] = color_space_subset[1][2] + 5
            #print(color_space_subset[0], color_space_subset[1])

            hsv_ms_masked = cv2.inRange(hsv_ms, color_space_subset[0], color_space_subset[1])
            hsv_rb_masked = cv2.inRange(hsv_rb, color_space_subset[0], color_space_subset[1])
            hsv_masked = cv2.bitwise_or(hsv_ms_masked, hsv_rb_masked)

            img_masked = cv2.bitwise_or(hsv_masked, rgb_masked)
            
            dilation_step = dilation_step + 1
            
            hsv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            hsv_opening = cv2.morphologyEx(hsv_masked, cv2.MORPH_OPEN, hsv_kernel, iterations=1)
            
            if (dilation_step >= dilation_step_threshold*3):
                break
                
        if poly_counter < 20 and  (dilation_step >= dilation_step_threshold*3) and (np.sum(hsv_masked)/255 <= img_bound_np_sum/255 *0.0005):
            # tends to become darker
            for rgb_set in range(2, len(color_space_subset), 2):
                color_space_subset[rgb_set] = color_space_subset[rgb_set] -15
                color_space_subset[rgb_set+1] = color_space_subset[rgb_set+1] +5
            #print(color_space_subset[2], color_space_subset[3])

            rgb_ms_masked = cv2.inRange(rgb_ms, color_space_subset[2], color_space_subset[3])
            rgb_rb_masked = cv2.inRange(rgb_rb, color_space_subset[2], color_space_subset[3])
            rgb_masked = cv2.bitwise_or(rgb_ms_masked, rgb_rb_masked)

            img_masked = cv2.bitwise_or(hsv_masked, rgb_masked)

        while poly_counter > 3 and (np.sum(rgb_masked)/255 <= img_bound_np_sum/255 *0.001):
            # dilation if needed
            for rgb_set in range(2, len(color_space_subset), 2):
                color_space_subset[rgb_set] = color_space_subset[rgb_set] -1
                color_space_subset[rgb_set+1] = color_space_subset[rgb_set+1] +1
            #print(color_space_subset[2], color_space_subset[3])

            rgb_ms_masked = cv2.inRange(rgb_ms, color_space_subset[2], color_space_subset[3])
            rgb_rb_masked = cv2.inRange(rgb_rb, color_space_subset[2], color_space_subset[3])
            rgb_masked = cv2.bitwise_or(rgb_ms_masked, rgb_rb_masked)
            
            img_masked = cv2.bitwise_or(hsv_masked, rgb_masked)

        if (np.sum(rgb_masked)/255 <= img_bound_np_sum/255 *0.1):
            for rgb_set in range(len(color_space_subset)-2, len(color_space_subset), 2):
                color_space_subset[rgb_set] = color_space_subset[rgb_set] -15
                color_space_subset[rgb_set+1] = color_space_subset[rgb_set+1] +15
        '''


    if print_intermediate_image == True:
        '''
        out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_v0_hsv.png'
        cv2.imwrite(out_file_path0, hsv_masked)
        out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_v0_rgb.png'
        cv2.imwrite(out_file_path0, rgb_masked)
        '''
        out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_v2.png'
        cv2.imwrite(out_file_path0, img_masked)

        '''
        out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_v0_hsv_extended.png'
        cv2.imwrite(out_file_path0, hsv_masked_ext)
        out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_v2_extended.png'
        cv2.imwrite(out_file_path0, img_masked_ext)
        '''
    
    return legend, img_masked, color_space_subset



def extraction_step2_worker_single_round(legend, map_name, legend_name, solutiona_dir, print_intermediate_image, hsv_rb, rgb_rb, hsv_space, color_space_subset):
    # create a mask to only preserve current legend color in the basemap
    hsv_masked = cv2.inRange(hsv_rb, color_space_subset[0], color_space_subset[1])
    rgb_masked = cv2.inRange(rgb_rb, color_space_subset[2], color_space_subset[3])
    img_masked = cv2.bitwise_or(hsv_masked, rgb_masked)


    # dilation if needed
    current_h_lower = color_space_subset[0][0]
    current_h_max = 0
    for h_space in range(current_h_lower, -1, -1):
        if hsv_space[h_space] > 0 and hsv_space[h_space] != legend:
            current_h_max = h_space
            break
    
    current_h_upper = color_space_subset[1][0]
    current_h_min = 255
    for h_space in range(current_h_upper, 255):
        if hsv_space[h_space] > 0 and hsv_space[h_space] != legend:
            current_h_min = h_space
            break
    
    color_space_subset[0][0] = current_h_lower - min(int((current_h_lower-current_h_max)/2), 2)
    color_space_subset[1][0] = current_h_upper + min(int((current_h_min-current_h_upper)/2), 2)
    color_space_subset[0][1] = color_space_subset[0][1] - 5
    color_space_subset[1][1] = color_space_subset[1][1] + 5
    color_space_subset[0][2] = color_space_subset[0][2] - 5
    color_space_subset[1][2] = color_space_subset[1][2] + 5
    #print(color_space_subset[0], color_space_subset[1])
    
    hsv_masked_ext = cv2.inRange(hsv_rb, color_space_subset[0], color_space_subset[1])
    img_masked_ext = cv2.bitwise_or(hsv_masked_ext, rgb_masked)


    if print_intermediate_image == True:
        '''
        out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_v0_hsv.png'
        cv2.imwrite(out_file_path0, hsv_masked)
        out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_v0_rgb.png'
        cv2.imwrite(out_file_path0, rgb_masked)
        '''
        out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_v2.png'
        cv2.imwrite(out_file_path0, img_masked)

        '''
        out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_v0_hsv_extended.png'
        cv2.imwrite(out_file_path0, hsv_masked_ext)
        '''
        out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_v2_extended.png'
        cv2.imwrite(out_file_path0, img_masked_ext)
    
    #return legend, img_masked, color_space_subset
    return legend, img_masked_ext, color_space_subset