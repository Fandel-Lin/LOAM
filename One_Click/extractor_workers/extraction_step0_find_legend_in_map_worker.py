import numpy as np
import cv2
import os

from PIL import Image, ImageEnhance

def extraction_step0_find_legend_in_map_worker(legend, map_name, names, img_legend, solutiona_dir, threshold_text, img_crop_black, sum_img_bound, text_pattern_probability, print_intermediate_image):
    this_current_image = np.copy(img_crop_black)
    #img_legend = np.copy(features)
    ### There is no groundtruth for validation data
    #print('training/'+map_name+'_'+names+'.tif')

    if print_intermediate_image == True:
        out_file_path0=solutiona_dir+'intermediate5/Extraction/'+map_name+'/'+map_name+'_'+names+'_legend.tif'
        cv2.imwrite(out_file_path0, img_legend)
        #cv2.imwrite(out_file_path0, features)

    #read the image
    im = Image.open(solutiona_dir+'intermediate5/Extraction/'+map_name+'/'+map_name+'_'+names+'_legend.tif')
    #image brightness enhancer
    enhancer = ImageEnhance.Contrast(im)
    factor = 2.5 #increase contrast
    im2 = enhancer.enhance(factor)
    if print_intermediate_image == True:
        out_file_path0=solutiona_dir+'intermediate5/Extraction/'+map_name+'/'+map_name+'_'+names+'_legend(2).tif'
        im2.save(out_file_path0)
    

    img_legend_v0 = cv2.imread(os.path.join(solutiona_dir+'intermediate5', 'Extraction', map_name, map_name+'_'+names+'_legend(2).tif'))
    img_legend_v1 = img_legend_v0[int(img_legend_v0.shape[0]*5.0/16.0):int(img_legend_v0.shape[0]*11.0/16.0), int(img_legend_v0.shape[1]*5.0/16.0):int(img_legend_v0.shape[1]*11.0/16.0)]


    lower_black_text = np.array([0,0,0])
    upper_black_text = np.array([80,80,80])
    mask_box_legend = cv2.inRange(img_legend_v1, lower_black_text, upper_black_text)

    mask_box_legend = cv2.medianBlur(mask_box_legend,3)

    if print_intermediate_image == True:
        out_file_path0=solutiona_dir+'intermediate5/Extraction/'+map_name+'/'+map_name+'_'+names+'_legend_box(0).tif'
        cv2.imwrite(out_file_path0, mask_box_legend)

    text_pattern_probability = True
    if text_pattern_probability == True:
        mask_box_legend_check = np.copy(mask_box_legend)
        mask_box_legend_check[mask_box_legend_check > 0] = 255
        
        if (mask_box_legend_check == 0).all():
            this_image_space = np.zeros((this_current_image.shape[0], this_current_image.shape[1]), dtype='uint8')
            out_file_path0=solutiona_dir+'intermediate5/Extraction/'+map_name+'/'+map_name+'_'+names+'_poly_t1.png'
            cv2.imwrite(out_file_path0, this_image_space)

            update_image_space = np.zeros((this_current_image.shape[0], this_current_image.shape[1]), dtype='uint8')
            overlapping = cv2.bitwise_and(img_crop_black, update_image_space)
            out_file_path0=solutiona_dir+'intermediate5/Extraction/'+map_name+'/'+map_name+'_'+names+'_poly_t01.png'
            cv2.imwrite(out_file_path0, overlapping)

            return legend, overlapping
    
        
    # Make sure image is binary (the one you posted was not, probably due to interpolation)
    image_binary = np.copy(mask_box_legend)
    image_binary[mask_box_legend > 0] = 255

    print(names)

    # Find top left and bottom right coords for non-background pixels
    active_pixels = np.stack(np.where(image_binary))
    top_left = np.min(active_pixels, axis=1).astype(np.int32)
    bottom_right = np.max(active_pixels, axis=1).astype(np.int32)
    print(top_left, bottom_right)

    box_diff_y = abs(bottom_right[0] - top_left[0])
    box_diff_x = abs(bottom_right[1] - top_left[1])
    #print(box_diff_x, box_diff_y)
    print(img_legend_v1.shape)


    current_x_0 = int(img_legend_v0.shape[0]*4.0/16.0)
    current_x_1 = int(img_legend_v0.shape[0]*12.0/16.0)
    current_y_0 = int(img_legend_v0.shape[1]*4.0/16.0)
    current_y_1 = int(img_legend_v0.shape[1]*12.0/16.0)
    unit_x = max(1, int(img_legend_v0.shape[0]*1.0/64.0))
    unit_y = max(1, int(img_legend_v0.shape[1]*1.0/64.0))

    abort_threshold = 50

    acc_abort = 0
    while top_left[0] <= 0 and current_x_0 > 0: # up
        current_x_0 = max(0, current_x_0 - unit_x)
        img_legend_v1 = img_legend_v0[current_x_0:current_x_1, current_y_0:current_y_1]

        lower_black_text = np.array([0,0,0])
        upper_black_text = np.array([80,80,80])
        mask_box_legend = cv2.inRange(img_legend_v1, lower_black_text, upper_black_text)

        mask_box_legend = cv2.medianBlur(mask_box_legend,3)

        # Make sure image is binary (the one you posted was not, probably due to interpolation)
        #_, image_binary = cv2.threshold(mask_box_legend, 80, 255, cv2.THRESH_BINARY)
        image_binary = np.copy(mask_box_legend)
        image_binary[mask_box_legend > 0] = 255

        # Find top left and bottom right coords for non-background pixels
        active_pixels = np.stack(np.where(image_binary))
        top_left = np.min(active_pixels, axis=1).astype(np.int32)
        bottom_right = np.max(active_pixels, axis=1).astype(np.int32)

        acc_abort = acc_abort + 1
        if acc_abort > abort_threshold:
            break
    
    acc_abort = 0
    while top_left[1] <= 0 and current_y_0 > 0: # left
        current_y_0 = max(0, current_y_0 - unit_y)
        img_legend_v1 = img_legend_v0[current_x_0:current_x_1, current_y_0:current_y_1]

        lower_black_text = np.array([0,0,0])
        upper_black_text = np.array([80,80,80])
        mask_box_legend = cv2.inRange(img_legend_v1, lower_black_text, upper_black_text)
        
        mask_box_legend = cv2.medianBlur(mask_box_legend,3)

        # Make sure image is binary (the one you posted was not, probably due to interpolation)
        #_, image_binary = cv2.threshold(mask_box_legend, 80, 255, cv2.THRESH_BINARY)
        image_binary = np.copy(mask_box_legend)
        image_binary[mask_box_legend > 0] = 255

        # Find top left and bottom right coords for non-background pixels
        active_pixels = np.stack(np.where(image_binary))
        top_left = np.min(active_pixels, axis=1).astype(np.int32)
        bottom_right = np.max(active_pixels, axis=1).astype(np.int32)

        acc_abort = acc_abort + 1
        if acc_abort > abort_threshold:
            break

    acc_abort = 0
    while bottom_right[0] >= img_legend_v1.shape[0]-1 and bottom_right[0] < img_legend_v0.shape[0]: # bottom
        current_x_1 = current_x_1 + unit_x
        img_legend_v1 = img_legend_v0[current_x_0:current_x_1, current_y_0:current_y_1]

        lower_black_text = np.array([0,0,0])
        upper_black_text = np.array([80,80,80])
        mask_box_legend = cv2.inRange(img_legend_v1, lower_black_text, upper_black_text)
        
        mask_box_legend = cv2.medianBlur(mask_box_legend,3)

        # Make sure image is binary (the one you posted was not, probably due to interpolation)
        #_, image_binary = cv2.threshold(mask_box_legend, 80, 255, cv2.THRESH_BINARY)
        image_binary = np.copy(mask_box_legend)
        image_binary[mask_box_legend > 0] = 255

        # Find top left and bottom right coords for non-background pixels
        active_pixels = np.stack(np.where(image_binary))
        top_left = np.min(active_pixels, axis=1).astype(np.int32)
        bottom_right = np.max(active_pixels, axis=1).astype(np.int32)

        acc_abort = acc_abort + 1
        if acc_abort > abort_threshold:
            break

    acc_abort = 0
    while bottom_right[1] >= img_legend_v1.shape[1]-1 and bottom_right[1] < img_legend_v0.shape[1]: # right
        current_y_1 = current_y_1 + unit_y
        img_legend_v1 = img_legend_v0[current_x_0:current_x_1, current_y_0:current_y_1]

        lower_black_text = np.array([0,0,0])
        upper_black_text = np.array([80,80,80])
        mask_box_legend = cv2.inRange(img_legend_v1, lower_black_text, upper_black_text)
        
        mask_box_legend = cv2.medianBlur(mask_box_legend,3)

        # Make sure image is binary (the one you posted was not, probably due to interpolation)
        #_, image_binary = cv2.threshold(mask_box_legend, 80, 255, cv2.THRESH_BINARY)
        image_binary = np.copy(mask_box_legend)
        image_binary[mask_box_legend > 0] = 255

        # Find top left and bottom right coords for non-background pixels
        active_pixels = np.stack(np.where(image_binary))
        top_left = np.min(active_pixels, axis=1).astype(np.int32)
        bottom_right = np.max(active_pixels, axis=1).astype(np.int32)

        acc_abort = acc_abort + 1
        if acc_abort > abort_threshold:
            break

    

    if print_intermediate_image == True:
        out_file_path0=solutiona_dir+'intermediate5/Extraction/'+map_name+'/'+map_name+'_'+names+'_legend_box.tif'
        cv2.imwrite(out_file_path0, mask_box_legend)


    box_diff_y = abs(bottom_right[0] - top_left[0])
    box_diff_x = abs(bottom_right[1] - top_left[1])

    print('---')
    print(top_left, bottom_right)
    print(box_diff_x, box_diff_y)


    
    #mask_box_legend = cv2.medianBlur(mask_box_legend,3)

    #plt.imshow(mask_box_legend)
    #plt.show()

    # threshold_text = processed map with text highlighted
    res = cv2.matchTemplate(threshold_text, mask_box_legend, cv2.TM_CCOEFF_NORMED)

    if text_pattern_probability == True:
        if print_intermediate_image == True:
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
            threshold = max(0.0, np.nanmax(res)*0.85)
            print(np.nanmax(res), threshold)

            #this_image_space[res >= threshold] = 255
            this_image_space[this_image_space < 255*threshold] = 0
            this_image_space[this_image_space >= 255*threshold] = 255

            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(box_diff_x*1.5), int(box_diff_y*1.5)))
            this_image_space = cv2.dilate(this_image_space, dilate_kernel, iterations=1)
            #this_image_space = cv2.medianBlur(this_image_space,3)

            #this_image_space = cv2.bitwise_and(this_image_space, this_image_space, mask = this_current_image)

            update_image_space = np.zeros((this_current_image.shape[0], this_current_image.shape[1]), dtype='uint8')
            diff_x = this_current_image.shape[0] - this_image_space.shape[0]
            diff_y = this_current_image.shape[1] - this_image_space.shape[1]
            update_image_space[int(diff_x/2):int(diff_x/2)+this_image_space.shape[0], int(diff_y/2):int(diff_y/2)+this_image_space.shape[1]] = this_image_space


            if np.sum(update_image_space) / sum_img_bound > 0.2:
                threshold = max(0.0, np.nanmax(res))
                print(np.nanmax(res), threshold)

                #this_image_space[res >= threshold] = 255
                this_image_space[this_image_space < 255*threshold] = 0
                this_image_space[this_image_space >= 255*threshold] = 255

                dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1,int(box_diff_x*1.5)), max(1, int(box_diff_y*1.5))))
                this_image_space = cv2.dilate(this_image_space, dilate_kernel, iterations=1)
                #this_image_space = cv2.medianBlur(this_image_space,3)

                #this_image_space = cv2.bitwise_and(this_image_space, this_image_space, mask = this_current_image)

                update_image_space = np.zeros((this_current_image.shape[0], this_current_image.shape[1]), dtype='uint8')
                diff_x = this_current_image.shape[0] - this_image_space.shape[0]
                diff_y = this_current_image.shape[1] - this_image_space.shape[1]
                update_image_space[int(diff_x/2):int(diff_x/2)+this_image_space.shape[0], int(diff_y/2):int(diff_y/2)+this_image_space.shape[1]] = this_image_space

            if np.sum(update_image_space) / sum_img_bound > 0.4:
                this_image_space = np.zeros((this_current_image.shape[0], this_current_image.shape[1]), dtype='uint8')
                out_file_path0=solutiona_dir+'intermediate5/Extraction/'+map_name+'/'+map_name+'_'+names+'_poly_t1.png'
                cv2.imwrite(out_file_path0, this_image_space)

                update_image_space = np.zeros((this_current_image.shape[0], this_current_image.shape[1]), dtype='uint8')
                overlapping = cv2.bitwise_and(img_crop_black, update_image_space)
                out_file_path0=solutiona_dir+'intermediate5/Extraction/'+map_name+'/'+map_name+'_'+names+'_poly_t01.png'
                cv2.imwrite(out_file_path0, overlapping)

                return legend, overlapping

            #out_file_path0=solutiona_dir+'intermediate5/Extraction/'+map_name+'/'+map_name+'_'+names+'_poly_t00.png'
            #cv2.imwrite(out_file_path0, res)

            out_file_path0=solutiona_dir+'intermediate5/Extraction/'+map_name+'/'+map_name+'_'+names+'_poly_t0.png'
            cv2.imwrite(out_file_path0, update_image_space)

            overlapping = cv2.bitwise_and(img_crop_black, update_image_space)
            out_file_path0=solutiona_dir+'intermediate5/Extraction/'+map_name+'/'+map_name+'_'+names+'_poly_t01.png'
            cv2.imwrite(out_file_path0, overlapping)
    
    return legend, overlapping