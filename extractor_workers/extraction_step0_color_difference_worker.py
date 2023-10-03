import numpy as np
import cv2
import os

def extraction_step0_color_difference_worker(legend, map_name, legend_name, solutiona_dir, print_intermediate_image, img_bound, rgb_rb, hsv_rb, this_color_avg, this_color_avg2, color_space_subset):
    # create a mask to only preserve current legend color in the basemap
    hsv_masked = cv2.inRange(hsv_rb, color_space_subset[0], color_space_subset[1])
    rgb_masked = cv2.inRange(rgb_rb, color_space_subset[2], color_space_subset[3])
    img_masked = cv2.bitwise_or(hsv_masked, rgb_masked)

    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
    #opening = cv2.morphologyEx(img_masked, cv2.MORPH_OPEN, kernel, iterations=1)
    #img_masked=cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY)[1]

    #rgb_rb_temp = np.copy(rgb_rb)
    #hsv_rb_temp = np.copy(hsv_rb)
    rgb_dif = np.zeros((rgb_rb.shape[0], rgb_rb.shape[1], rgb_rb.shape[2]), dtype='uint8')
    hsv_dif = np.zeros((hsv_rb.shape[0], hsv_rb.shape[1], hsv_rb.shape[2]), dtype='uint8')
    rgb_dif0 = np.zeros((rgb_rb.shape[0], rgb_rb.shape[1], rgb_rb.shape[2], 2), dtype='uint8')
    hsv_dif0 = np.zeros((hsv_rb.shape[0], hsv_rb.shape[1], hsv_rb.shape[2], 2), dtype='uint8')

    for dimension_dif in range(0, 3):
        #rgb_dif[:,:,dimension_dif] = abs(rgb_rb[:,:,dimension_dif] - this_color_avg[dimension_dif]) # range: 0 - 256
        #hsv_dif[:,:,dimension_dif] = abs(hsv_rb[:,:,dimension_dif] - this_color_avg2[dimension_dif])

        #rgb_dif[:,:,dimension_dif] = min(abs(rgb_rb[:,:,dimension_dif] - color_space_subset[2][dimension_dif]), abs(rgb_rb[:,:,dimension_dif] - color_space_subset[3][dimension_dif])) # range: 0 - 256
        #hsv_dif[:,:,dimension_dif] = min(abs(hsv_rb[:,:,dimension_dif] - color_space_subset[0][dimension_dif]), abs(hsv_rb[:,:,dimension_dif] - color_space_subset[1][dimension_dif])) # range: 0 - 256

        rgb_dif0[:,:,dimension_dif,0] = abs(rgb_rb[:,:,dimension_dif] - color_space_subset[2][dimension_dif])
        rgb_dif0[:,:,dimension_dif,1] = abs(rgb_rb[:,:,dimension_dif] - color_space_subset[3][dimension_dif])
        hsv_dif0[:,:,dimension_dif,0] = abs(hsv_rb[:,:,dimension_dif] - color_space_subset[0][dimension_dif])
        hsv_dif0[:,:,dimension_dif,1] = abs(hsv_rb[:,:,dimension_dif] - color_space_subset[1][dimension_dif])

        #print(np.unique(rgb_dif))
        #print(np.unique(hsv_dif))

    rgb_dif = np.min(rgb_dif0, axis=3)
    hsv_dif = np.min(hsv_dif0, axis=3)

    #rgb_dif = ((np.copy(rgb_dif) + 255) / 2).astype('uint8')
    #hsv_dif = ((np.copy(hsv_dif) + 255) / 2).astype('uint8')
    rgb_dif[rgb_dif > 51] = 51
    hsv_dif[hsv_dif > 51] = 51
    rgb_dif = rgb_dif*5
    hsv_dif = hsv_dif*5
    rgb_dif[rgb_dif >= 255] = 255
    hsv_dif[hsv_dif >= 255] = 255

    rgb_dif = (255 - rgb_dif).astype('uint8')
    hsv_dif = (255 - hsv_dif).astype('uint8')

    rgb_dif = cv2.bitwise_and(rgb_dif, rgb_dif, mask=img_bound)
    hsv_dif = cv2.bitwise_and(hsv_dif, hsv_dif, mask=img_bound)
    #rgb_dif = cv2.bitwise_or(rgb_dif, img_masked)
    #hsv_dif = cv2.bitwise_or(hsv_dif, img_masked)

    for dimension_dif in range(0, 3):
        rgb_dif[:,:,dimension_dif] = cv2.bitwise_or(rgb_dif[:,:,dimension_dif], img_masked)
        hsv_dif[:,:,dimension_dif] = cv2.bitwise_or(hsv_dif[:,:,dimension_dif], img_masked)

    for dimension_dif in range(0, 3):
        rgb_dif_single = np.zeros((rgb_rb.shape[0], rgb_rb.shape[1]), dtype='uint8')
        hsv_dif_single = np.zeros((hsv_rb.shape[0], hsv_rb.shape[1]), dtype='uint8')

        rgb_dif_single[:,:] = rgb_dif[:,:,dimension_dif]
        hsv_dif_single[:,:] = hsv_dif[:,:,dimension_dif]

        if print_intermediate_image == True:
            out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_c0_'+str(dimension_dif)+'.png'
            cv2.imwrite(out_file_path0, rgb_dif_single)
            out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_c1_'+str(dimension_dif)+'.png'
            cv2.imwrite(out_file_path0, hsv_dif_single)
    
    #rgb_dif_min = np.zeros((rgb_rb_temp.shape[0], rgb_rb_temp.shape[1]), dtype='uint8')
    #rgb_dif_min[:,:] = min(rgb_dif[:,:,0], rgb_dif[:,:,1], rgb_dif[:,:,2])
    rgb_dif_min = np.min(rgb_dif, axis=2).astype('uint8')
    #rgb_dif_min[rgb_dif_min < 127] = 0
    if print_intermediate_image == True:
        out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_c0_x.png'
        cv2.imwrite(out_file_path0, rgb_dif_min)
    
    #hsv_dif_min = np.zeros((hsv_rb_temp.shape[0], hsv_rb_temp.shape[1]), dtype='uint8')
    #hsv_dif_min[:,:] = min(hsv_dif[:,:,0], hsv_dif[:,:,1], hsv_dif[:,:,2])
    hsv_dif_min = np.min(hsv_dif, axis=2).astype('uint8')
    #hsv_dif_min[hsv_dif_min < 127] = 0
    if print_intermediate_image == True:
        out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_c1_x.png'
        cv2.imwrite(out_file_path0, hsv_dif_min)

    return legend, True