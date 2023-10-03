import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json

def extraction_step1_worker(legend, map_name, legend_name, solutiona_dir, print_intermediate_image, rgb_rb, rgb_ms, result_image, distances_subset, color_avg_length, color_avg_subset, poly_counter, img_bound_sum, image_shape, im):
    
    extracted_map = cv2.inRange(result_image, color_avg_subset, color_avg_subset)

    distance_threshold = 0
    while (np.sum(extracted_map)/255 <= img_bound_sum/255 *0.002):
        distances_v = np.zeros((color_avg_length, image_shape[0], image_shape[1]), dtype='uint8')
        distance_threshold = distance_threshold + 10

        for legend in range(0, poly_counter):
            distances_v[legend] = np.copy(distances_subset)
            
            extracted_map[distances_v[legend] <= distance_threshold] = 255
            #extracted_map = cv2.bitwise_and(extracted_map, img_bound)
            #ans_category[legend] = cv2.bitwise_or(ans_category[legend], extracted_map)

            print(legend_name[legend])
            #plt.imshow(extracted_map)
            #plt.show()


    extracted_map_v = np.zeros((3, extracted_map.shape[0], extracted_map.shape[1]), dtype='uint8')
    extracted_map_v = extracted_map_v.astype(float)
    extracted_avg_rgb = []
    extracted_lower_rgb = []
    extracted_upper_rgb = []

    is_null = False
    for dimension in range(0, 3):
        extracted_map_v[dimension] = np.copy(im[:,:,dimension]).astype(float)
        extracted_map_v[dimension] = cv2.bitwise_and(im[:,:,dimension], im[:,:,dimension], mask=extracted_map)
        extracted_map_v[dimension] = extracted_map_v[dimension]
        extracted_map_v[dimension][extracted_map_v[dimension] == 0] = np.nan
        if np.sum(np.isnan(extracted_map_v[dimension])) >= (im.shape[0] * im.shape[1]):
            is_null = True
            print('This situation shall not happen.')
        else:
            extracted_avg_rgb.append(int(np.nanquantile(extracted_map_v[dimension],.5)))
            extracted_lower_rgb.append(int(np.nanquantile(extracted_map_v[dimension],.2)))
            extracted_upper_rgb.append(int(np.nanquantile(extracted_map_v[dimension],.8)))
    #subtract_rgb.append(color_avg[legend] - extracted_avg_rgb)

    rgb_ms_masked = cv2.inRange(rgb_ms, np.array(extracted_lower_rgb), np.array(extracted_upper_rgb))
    rgb_rb_masked = cv2.inRange(rgb_rb, np.array(extracted_lower_rgb), np.array(extracted_upper_rgb))
    rgb_masked = cv2.bitwise_or(rgb_ms_masked, rgb_rb_masked)
    img_masked = cv2.bitwise_or(rgb_masked, rgb_masked)
    
    if print_intermediate_image == True:
        out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_v1_rgb.png'
        cv2.imwrite(out_file_path0, rgb_masked)
        out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_v2.png'
        cv2.imwrite(out_file_path0, img_masked)
    
    return legend, img_masked, (color_avg_subset - extracted_avg_rgb)
            