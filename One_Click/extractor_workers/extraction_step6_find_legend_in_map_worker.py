import numpy as np
import cv2
import os

def extraction_step6_find_legend_in_map_worker(legend, map_name, legend_name, solutiona_dir, threshold_text, this_current_image, sum_img_bound, text_pattern_probability, print_intermediate_image):
    # this_current_image

    # fetch current result for this legend
    #this_current_result = np.copy(ans_category[legend])

    #img_legend_v0 = cv2.imread(os.path.join(solutiona_dir+'intermediate7(2)', map_name, map_name+'_'+legend_name[legend]+'_poly_legend.tif'))

    '''
    img_legend_v0 = cv2.imread(os.path.join(solutiona_dir+'intermediate5', 'Extraction', map_name, map_name+'_'+legend_name[legend]+'_legend(2).tif'))
    img_legend_v1 = img_legend_v0[int(img_legend_v0.shape[0]*2.0/8.0):int(img_legend_v0.shape[0]*6.0/8.0), int(img_legend_v0.shape[1]*2.0/8.0):int(img_legend_v0.shape[1]*6.0/8.0)]

    lower_black_text = np.array([0,0,0])
    upper_black_text = np.array([80,80,80])
    mask_box_legend = cv2.inRange(img_legend_v1, lower_black_text, upper_black_text)

    mask_box_legend = cv2.medianBlur(mask_box_legend,3)
    '''

    mask_box_legend = cv2.imread(os.path.join(solutiona_dir+'intermediate5', 'Extraction', map_name, map_name+'_'+legend_name[legend]+'_legend_box.tif'))
    try:
        mask_box_legend = cv2.cvtColor(mask_box_legend, cv2.COLOR_BGR2GRAY)
    except Exception:
        update_image_space = np.zeros((threshold_text.shape[0], threshold_text.shape[1]), dtype='uint8')
        return legend, 0.0, update_image_space


    #text_pattern_probability = True
    if text_pattern_probability == True:
        mask_box_legend_check = np.copy(mask_box_legend)
        mask_box_legend_check[mask_box_legend_check > 0] = 255
        
        if (mask_box_legend_check == 0).all():
            update_image_space = np.zeros((threshold_text.shape[0], threshold_text.shape[1]), dtype='uint8')

            return legend, 0.0, update_image_space
        

    #mask_box_legend = cv2.medianBlur(mask_box_legend,3)

    #plt.imshow(mask_box_legend)
    #plt.show()

    # threshold_text = processed map with text highlighted
    res = cv2.matchTemplate(threshold_text, mask_box_legend, cv2.TM_CCOEFF_NORMED)

    #this_image_space = np.zeros((this_current_image.shape[0], this_current_image.shape[1]), dtype='uint8')
    #this_image_space[0:res.shape[0], 0:res.shape[1]] = res[0:res.shape[0], 0:res.shape[1]] * 255.0
    #this_image_space[this_image_space < 255.0*0.75] = 0
    #this_image_space[this_image_space < 0] = 0

    #this_image_space = np.zeros((res.shape[0], res.shape[1]), dtype='uint8')
    this_image_space = np.copy(res)
    this_image_space[this_image_space < 0] = 0
    this_image_space = this_image_space * 255

    #print(np.unique(res))
    #print(np.nanmax(res))

    
    #threshold = max(0.0, np.nanmax(res) - 0.15)
    threshold = max(0.0, np.nanmax(res)*0.95)
    #print(np.nanmax(res), threshold)

    #this_image_space[res >= threshold] = 255
    this_image_space[this_image_space < 255*threshold] = 0
    '''
    this_image_space[this_image_space >= 255*threshold] = 255

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 40))
    this_image_space = cv2.dilate(this_image_space, dilate_kernel, iterations=1)
    #this_image_space = cv2.medianBlur(this_image_space,3)
    '''

    #this_image_space = cv2.bitwise_and(this_image_space, this_image_space, mask = this_current_image)

    update_image_space = np.zeros((threshold_text.shape[0], threshold_text.shape[1]), dtype='uint8')
    diff_x = threshold_text.shape[0] - this_image_space.shape[0]
    diff_y = threshold_text.shape[1] - this_image_space.shape[1]
    update_image_space[int(diff_x/2):int(diff_x/2)+this_image_space.shape[0], int(diff_y/2):int(diff_y/2)+this_image_space.shape[1]] = this_image_space


    if np.sum(update_image_space) / sum_img_bound > 0.2:
        threshold = max(0.0, np.nanmax(res))
        print(np.nanmax(res), threshold)

        #this_image_space[res >= threshold] = 255
        this_image_space[this_image_space < 255*threshold] = 0
        this_image_space[this_image_space >= 255*threshold] = 255

        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 40))
        this_image_space = cv2.dilate(this_image_space, dilate_kernel, iterations=1)
        #this_image_space = cv2.medianBlur(this_image_space,3)

        #this_image_space = cv2.bitwise_and(this_image_space, this_image_space, mask = this_current_image)

        update_image_space = np.zeros((this_current_image.shape[0], this_current_image.shape[1]), dtype='uint8')
        diff_x = this_current_image.shape[0] - this_image_space.shape[0]
        diff_y = this_current_image.shape[1] - this_image_space.shape[1]
        update_image_space[int(diff_x/2):int(diff_x/2)+this_image_space.shape[0], int(diff_y/2):int(diff_y/2)+this_image_space.shape[1]] = this_image_space

    if np.sum(update_image_space) / sum_img_bound > 0.4:
        update_image_space = np.zeros((this_current_image.shape[0], this_current_image.shape[1]), dtype='uint8')
    

    if text_pattern_probability == True:
        if print_intermediate_image == True:
            #out_file_path0=solutiona_dir+'intermediate4/'+map_name+'/'+map_name+'_'+names+'_poly_t00.png'
            #cv2.imwrite(out_file_path0, res)
            this_image_space_temp = np.copy(update_image_space)
            this_image_space_temp[this_image_space_temp >= 255*threshold] = 255

            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 40))
            this_image_space_temp = cv2.dilate(this_image_space_temp, dilate_kernel, iterations=1)
            overlapping = cv2.bitwise_and(threshold_text, this_image_space_temp)
            out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_t0.png'
            cv2.imwrite(out_file_path0, overlapping)
    #global_res_probability[legend] = np.copy(res) ###
    
    return legend, threshold, update_image_space