import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
from scipy import ndimage
from collections import Counter

def preprocessing_worker(map_id, file_name, data_dir, solutiona_dir, crop_legend):
    
    filename=file_name.replace('.json', '.tif')
    print('Working on map:', file_name)
    file_path=os.path.join(data_dir, filename)
    test_json=file_path.replace('.tif', '.json')
    
    img0 = cv2.imread(file_path)
    hsv0 = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
    rgb0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    gray0 = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)


    lower_black = np.array([0,0,0])
    upper_black = np.array([130,130,130]) #100

    blank = np.zeros((img0.shape[0],img0.shape[1],img0.shape[2]),dtype=np.uint8)
    blank[0:img0.shape[0],0:img0.shape[1],0:img0.shape[2]]=255

    # create a mask to only preserve current legend color in the basemap
    mask_box = cv2.inRange(rgb0, lower_black, upper_black)
    res_box = cv2.bitwise_and(blank,blank, mask=mask_box)

    # convert to grayscale 
    detected_gray0 = cv2.cvtColor(res_box, cv2.COLOR_BGR2GRAY)
    img_bw0 = cv2.threshold(detected_gray0, 60, 255, cv2.THRESH_BINARY)[1] # 127

    out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_blackregion.png'
    cv2.imwrite(out_file_path0, img_bw0)



    lower_black1 = np.array([100,100,100])
    upper_black1 = np.array([180,180,180])

    # create a mask to only preserve current legend color in the basemap
    mask_box1 = cv2.inRange(rgb0, lower_black1, upper_black1)
    res_box1 = cv2.bitwise_and(blank,blank, mask=mask_box1)

    # convert to grayscale 
    detected_gray1 = cv2.cvtColor(res_box1, cv2.COLOR_BGR2GRAY)
    img_bw01 = cv2.threshold(detected_gray1, 60, 255, cv2.THRESH_BINARY)[1] # 127

    out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_grayregion.png'
    cv2.imwrite(out_file_path0, img_bw01)



    blank = np.zeros((img0.shape[0],img0.shape[1],img0.shape[2]),dtype=np.uint8)
    blank[0:img0.shape[0],0:img0.shape[1],0:img0.shape[2]] = 255


    if os.path.exists(solutiona_dir+str('intermediate6/cropped_map(2)/'+file_name.replace('.json', '')+'_crop.tif')) and os.path.exists(solutiona_dir+'intermediate6/cropped_map_mask(2)/'+file_name.replace('.json', '')+'_expected_crop_region.tif'):
        img1 = np.copy(img0)

        crop_rgb2 = cv2.imread(solutiona_dir+str('intermediate6/cropped_map(2)/'+file_name.replace('.json', '')+'_crop.tif'))
        crop_rgb2 = cv2.cvtColor(crop_rgb2, cv2.COLOR_BGR2RGB)

        selected_map_for_examination = cv2.imread(solutiona_dir+'intermediate6/cropped_map_mask(2)/'+file_name.replace('.json', '')+'_expected_crop_region.tif')
        selected_map_for_examination = cv2.cvtColor(selected_map_for_examination, cv2.COLOR_BGR2GRAY)
        

        out_file_path0 = solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_expected_crop_region.tif'
        cv2.imwrite(out_file_path0, selected_map_for_examination)
        print_bgr = cv2.cvtColor(crop_rgb2, cv2.COLOR_RGB2BGR)
        out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop.tif'
        cv2.imwrite(out_file_path0, print_bgr)



        
        #crop_rgb1 = cv2.bitwise_and(rgb1,rgb1, mask=selected_map_for_examination)
        crop_rgb1 = np.copy(crop_rgb2)
        crop_img_bw0 = cv2.bitwise_and(img_bw0,img_bw0, mask=selected_map_for_examination)
        crop_img_bw01 = cv2.bitwise_and(img_bw01,img_bw01, mask=selected_map_for_examination)

    else:
        corner_area = np.copy(rgb0[int(rgb0.shape[0]*2/100):int(rgb0.shape[0]*3/100), int(rgb0.shape[1]*30/100):int(rgb0.shape[1]*70/100)])
        rgb_trimmed = np.zeros((corner_area.shape[2], corner_area.shape[0], corner_area.shape[1]), dtype='uint8')
        for dimension in range(0, 3):
            rgb_trimmed[dimension] = np.copy(corner_area[:,:,dimension])

        lower_color = np.array([250,250,250]) ### This shall be the color you want to crop off
        upper_color = np.array([256,256,256]) ### This shall be the color you want to crop off

        res_box = cv2.inRange(rgb0, lower_color, upper_color)

        # Either use threshold (less accurate, but works for rotated cases) or contour to remove black border
        '''
        lower_color = np.array([0,0,0])
        upper_color = np.array([1,1,1])

        res_box_1 = cv2.inRange(rgb0, lower_color, upper_color)
        res_box = cv2.bitwise_or(res_box, res_box_1)
        '''
        _,thresh = cv2.threshold(gray0,1,255,cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)

        res_box_1 = np.ones((img0.shape[0],img0.shape[1]),dtype=np.uint8)*255
        res_box_1[y:y+h,x:x+w] = 0
        res_box = cv2.bitwise_or(res_box, res_box_1)
        # Either use threshold or contour to remove black border


        res_box[res_box < 255] = 0
        img_bw00 = 255 - res_box

        # remove moisy white pixels before buffer
        kernel_before_blur00 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening_before_blur00 = cv2.morphologyEx(img_bw00, cv2.MORPH_OPEN, kernel_before_blur00, iterations=1)

        # smooth the image
        blur_radius = 5.0
        threshold_blur = 0
        gaussian_buffer00 = ndimage.gaussian_filter(opening_before_blur00, blur_radius)
        gaussian_buffer00[gaussian_buffer00 > threshold_blur] = 255

        # erode buffers to remove thin white lines due to crease
        kernel_erode = np.ones((50, 50), np.uint8)
        erode_buffer = cv2.erode(gaussian_buffer00, kernel_erode, cv2.BORDER_REFLECT) 

        # find connected components
        labeled00, nr_objects00 = ndimage.label(erode_buffer > threshold_blur)

        label_area = np.bincount(labeled00.flat)[1:] # 1
        arg_sort = np.argsort(label_area)

        check_generated = False
        threshold = [0.4, 0.3, 0.2, 0.1, 0.05]
        temp_max = 0
        for relaxing_threshold in range(0, len(threshold)):
            for target_arg in range(arg_sort.shape[0]-1, arg_sort.shape[0]-6, -1):
                #print(target_arg, arg_sort[target_arg])
                selected_index = arg_sort[target_arg]+1 # 1

                selected_map_for_examination = np.zeros((labeled00.shape[0],labeled00.shape[1],1),dtype=np.uint8)
                selected_map_for_examination[labeled00 == selected_index] = 255

                # dilate buffers back to it original size
                #dilate_buffer = cv2.dilate(erode_buffer, kernel_erode, iterations=1)
                selected_map_for_examination = cv2.dilate(selected_map_for_examination, kernel_erode, iterations=1)

                selected_map_for_examination_reversed = 255 - selected_map_for_examination

                # remove noisy white pixel
                kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (100,100))
                opening = cv2.morphologyEx(selected_map_for_examination_reversed, cv2.MORPH_OPEN, kernel_morph, iterations=1)
                selected_map_for_examination_reversed =cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY)[1]
                selected_map_for_examination = 255 - selected_map_for_examination_reversed
                
                # need to check
                crop_rgb0 = cv2.bitwise_and(rgb0,rgb0, mask=selected_map_for_examination)
                uniques = np.unique(crop_rgb0)
                #print(uniques.shape[0])
                if uniques.shape[0] > temp_max:
                    temp_max = uniques.shape[0]
                
                if uniques.shape[0] > 255*threshold[relaxing_threshold] or (relaxing_threshold > 0 and uniques.shape[0] == temp_max):
                    check_generated = True
                    break
            if check_generated:
                break
        
        if crop_legend == True:
            legend_mask = np.ones((img0.shape[0], img0.shape[1]), dtype='uint8') *255
            with open(test_json) as f:
                gj = json.load(f)
            for this_gj in gj['shapes']:
                names = this_gj['label']
                features = this_gj['points']
                geoms = np.array(features)
                y_min = int(np.min(geoms, axis=0)[0])
                y_max = int(np.max(geoms, axis=0)[0])
                x_min = int(np.min(geoms, axis=0)[1])
                x_max = int(np.max(geoms, axis=0)[1])
                legend_mask[x_min:x_max, y_min:y_max] = 0
            selected_map_for_examination = cv2.bitwise_and(selected_map_for_examination, legend_mask)
        
        if check_generated == False:
            selected_map_for_examination = np.ones((img0.shape[0],img0.shape[1],1),dtype=np.uint8)*255

            crop_rgb2 = np.copy(rgb0)
        else:
            # remove gray header
            header_region = np.copy(rgb0)
            header_region = header_region[0:int(rgb0.shape[0]*0.06),:]
            header_included = selected_map_for_examination[0:int(rgb0.shape[0]*0.06),:]

            lower_color = np.array([210,210,210]) ### This shall be the color you want to crop off
            upper_color = np.array([250,250,250]) ### This shall be the color you want to crop off
            header_gray = cv2.inRange(header_region, lower_color, upper_color)

            if np.sum(header_included) > 0 and np.sum(header_gray)/np.sum(header_included) > 0.85:
                print('remove gray header')
                selected_map_for_examination[0:int(rgb0.shape[0]*0.06),:] = 0
            
            blank_mask = cv2.bitwise_and(blank, blank, mask=cv2.bitwise_not(selected_map_for_examination))
            crop_rgb2 = cv2.add(crop_rgb0, blank_mask)
        #plt.imshow(crop_rgb2)
        #plt.show()
        
        out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_expected_crop_region.tif'
        cv2.imwrite(out_file_path0, selected_map_for_examination)

        print_bgr = cv2.cvtColor(crop_rgb2, cv2.COLOR_RGB2BGR)
        out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop.tif'
        cv2.imwrite(out_file_path0, print_bgr)




        '''Colored background cropping / Begin'''

        examination_counting = np.copy(selected_map_for_examination)
        if crop_legend == True:
            examination_counting = cv2.bitwise_or(examination_counting, 255-legend_mask)
            #print(np.mean(selected_map_for_examination)/255.0)
            #print(np.mean(examination_counting/255.0))
        if np.mean(examination_counting/255.0) > 0.9:
            print('Try to recognize the background colors and then remove them...')
            # try to recognize the background color and then remove them
            margin_region = np.ones((img0.shape[0],img0.shape[1],1),dtype=np.uint8)*255
            margin_region[int(img0.shape[0]*0.025):int(img0.shape[0]*0.975), int(img0.shape[1]*0.025):int(img0.shape[1]*0.975)] = 0

            if crop_legend == True:
                margin_region = cv2.bitwise_and(margin_region, legend_mask)

            selected_map_for_margin = cv2.bitwise_and(selected_map_for_examination, margin_region)
            crop_margin = cv2.bitwise_and(rgb0,rgb0, mask=selected_map_for_margin)
            #plt.imshow(crop_margin)
            #plt.show()
            
            crop_margin = crop_margin.astype(float)
            crop_margin[crop_margin == 0] = np.nan


            print('Recognizing the background colors...')
            ### background color recognization
            hsv0 = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
            hsv0_flat = hsv0.reshape(-1, hsv0.shape[-1])

            #crop_margin = np.copy(crop_margin_temp)
            crop_margin = cv2.bitwise_and(hsv0,hsv0, mask=selected_map_for_margin)
            crop_margin = crop_margin.astype(float)
            crop_margin[crop_margin == 0] = np.nan

            crop_margin = crop_margin.reshape(-1, crop_margin.shape[-1])
            #print(crop_margin.shape)
            crop_margin = crop_margin[~np.isnan(crop_margin).any(axis=1)]
            #print(crop_margin.shape)

            color_pixel_counter_0 = {}
            color_pixel_counter_1 = {}
            color_pixel_counter_2 = {}
            for this_candidate_pixel in crop_margin:
                #this_rgb = (this_candidate_pixel[0], this_candidate_pixel[1], this_candidate_pixel[2])
                
                if (this_candidate_pixel[0]) in color_pixel_counter_0:
                    color_pixel_counter_0[(this_candidate_pixel[0])] += 1
                else:
                    color_pixel_counter_0[(this_candidate_pixel[0])] = 1
                
                if (this_candidate_pixel[0], this_candidate_pixel[1]) in color_pixel_counter_1:
                    color_pixel_counter_1[(this_candidate_pixel[0], this_candidate_pixel[1])] += 1
                else:
                    color_pixel_counter_1[(this_candidate_pixel[0], this_candidate_pixel[1])] = 1

                if (this_candidate_pixel[0], this_candidate_pixel[1], this_candidate_pixel[2]) in color_pixel_counter_2:
                    color_pixel_counter_2[(this_candidate_pixel[0], this_candidate_pixel[1], this_candidate_pixel[2])] += 1
                else:
                    color_pixel_counter_2[(this_candidate_pixel[0], this_candidate_pixel[1], this_candidate_pixel[2])] = 1

            color_counter_0 = Counter(color_pixel_counter_0).most_common(10)
            color_counter_1 = Counter(color_pixel_counter_1).most_common(40)
            color_counter_2 = Counter(color_pixel_counter_2).most_common(60)

            hsv0_flat_check = np.ones((hsv0.shape[0],hsv0.shape[1]),dtype=np.uint8)*255
            hsv0_flat_check = hsv0_flat_check.flatten()

            #print(np.mean(hsv0_flat_check))
            acc_color_threshold = 0.99
            print('Removing the background colors...')

            '''
            acc_color = 0.0
            for target_color, color_count in color_counter_0:
                #print(target_color, color_count, (float(color_count)/float(crop_margin.shape[0])))
                #hsv0_flat_check[(hsv0_flat[0:2] == [int(target_color[0]), int(target_color[1])])] = 0
                hsv0_flat_check[(hsv0_flat[:,0] == int(target_color))] = 0
                acc_color += (float(color_count)/float(crop_margin.shape[0]))
                if acc_color > acc_color_threshold:
                    break
            '''
            acc_color = 0.0
            for target_color, color_count in color_counter_1:
                #print(target_color, color_count, (float(color_count)/float(crop_margin.shape[0])))
                hsv0_flat_check[(hsv0_flat[:,0] == int(target_color[0])) & (hsv0_flat[:,1] == int(target_color[1]))] = 0
                acc_color += (float(color_count)/float(crop_margin.shape[0]))
                if acc_color > acc_color_threshold:
                    break

            acc_color = 0.0
            for target_color, color_count in color_counter_2:
                #print(target_color, color_count, (float(color_count)/float(crop_margin.shape[0])))
                hsv0_flat_check[(hsv0_flat == [int(target_color[0]), int(target_color[1]), int(target_color[2])]).all(axis=1)] = 0
                acc_color += (float(color_count)/float(crop_margin.shape[0]))
                if acc_color > acc_color_threshold:
                    break

            #print(np.mean(hsv0_flat_check))
            hsv0_flat_check = hsv0_flat_check.reshape((hsv0.shape[0],hsv0.shape[1]))
            hsv0_masked = cv2.bitwise_and(hsv0, hsv0, mask=hsv0_flat_check)
            #print(np.mean(hsv0_masked))


            # remove moisy white pixels before buffer
            kernel_before_blur = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            opening_before_blur = cv2.morphologyEx(hsv0_flat_check, cv2.MORPH_OPEN, kernel_before_blur, iterations=1)

            # smooth the image
            blur_radius = 5.0
            threshold_blur = 0
            gaussian_buffer = ndimage.gaussian_filter(opening_before_blur, blur_radius)
            gaussian_buffer[gaussian_buffer > threshold_blur] = 255

            # erode buffers to remove thin white lines due to crease
            kernel_erode = np.ones((500, 500), np.uint8)
            erode_buffer = cv2.erode(gaussian_buffer, kernel_erode, cv2.BORDER_REFLECT)

            # find connected components
            labeled00, nr_objects00 = ndimage.label(erode_buffer > threshold_blur)

            label_area = np.bincount(labeled00.flat)[1:] # 1
            arg_sort = np.argsort(label_area)

            print('Identifying the greatest connected component...')
            check_generated = False
            threshold = [0.4, 0.3, 0.2, 0.1, 0.05]
            max_variety = 0
            for relaxing_threshold in range(0, len(threshold)):
                for target_arg in range(arg_sort.shape[0]-1, arg_sort.shape[0]-6, -1): # Only check the top-5 largest regions
                    #print(target_arg, arg_sort[target_arg])
                    selected_index = arg_sort[target_arg]+1 # 1

                    selected_map_for_examination = np.zeros((labeled00.shape[0],labeled00.shape[1],1),dtype=np.uint8)
                    selected_map_for_examination[labeled00 == selected_index] = 255

                    # dilate buffers back to it original size
                    #dilate_buffer = cv2.dilate(erode_buffer, kernel_erode, iterations=1)
                    selected_map_for_examination = cv2.dilate(selected_map_for_examination, kernel_erode, iterations=1)

                    selected_map_for_examination_reversed = 255 - selected_map_for_examination

                    # remove noisy white pixel
                    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (500,500))
                    opening = cv2.morphologyEx(selected_map_for_examination_reversed, cv2.MORPH_OPEN, kernel_morph, iterations=1)
                    selected_map_for_examination_reversed =cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY)[1]
                    selected_map_for_examination = 255 - selected_map_for_examination_reversed
                    
                    # need to check
                    crop_rgb0 = cv2.bitwise_and(rgb0,rgb0, mask=selected_map_for_examination)
                    uniques = np.unique(crop_rgb0)
                    if uniques.shape[0] > max_variety:
                        max_variety = uniques.shape[0]
                    
                    if uniques.shape[0] > 255*threshold[relaxing_threshold] or (relaxing_threshold > 0 and uniques.shape[0] == max_variety):
                        check_generated = True
                        break
                if check_generated:
                    break
            
            if crop_legend == True:
                legend_mask = np.ones((img0.shape[0], img0.shape[1]), dtype='uint8') *255
                with open(test_json) as f:
                    gj = json.load(f)
                for this_gj in gj['shapes']:
                    names = this_gj['label']
                    features = this_gj['points']
                    geoms = np.array(features)
                    y_min = int(np.min(geoms, axis=0)[0])
                    y_max = int(np.max(geoms, axis=0)[0])
                    x_min = int(np.min(geoms, axis=0)[1])
                    x_max = int(np.max(geoms, axis=0)[1])
                    legend_mask[x_min:x_max, y_min:y_max] = 0
                selected_map_for_examination = cv2.bitwise_and(selected_map_for_examination, legend_mask)

            if check_generated == False:
                selected_map_for_examination = np.ones((img0.shape[0],img0.shape[1],1),dtype=np.uint8)*255
                crop_rgb2 = np.copy(rgb0)
            else:
                blank_mask = cv2.bitwise_and(blank, blank, mask=cv2.bitwise_not(selected_map_for_examination))
                crop_rgb2 = cv2.add(crop_rgb0, blank_mask)

            # remove gray header
            header_region = np.copy(rgb0)
            header_region = header_region[0:int(rgb0.shape[0]*0.06), :]
            header_included = selected_map_for_examination[0:int(rgb0.shape[0]*0.06), :]

            lower_color = np.array([210,210,210]) ### This shall be the color you want to crop off
            upper_color = np.array([250,250,250]) ### This shall be the color you want to crop off
            header_gray = cv2.inRange(header_region, lower_color, upper_color)

            if np.sum(header_gray)/np.sum(header_included) > 0.85:
                print('remove gray header')
                selected_map_for_examination[0:int(rgb0.shape[0]*0.06),:] = 0

            out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_expected_crop_region.tif'
            cv2.imwrite(out_file_path0, selected_map_for_examination)

            print_bgr = cv2.cvtColor(crop_rgb2, cv2.COLOR_RGB2BGR)
            out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop.tif'
            cv2.imwrite(out_file_path0, print_bgr)


        '''Colored background cropping / End'''



        '''Inversed floodfill to fill up inner holes / Begin'''

        
        candidate_naming = file_name.replace('.json', '')+'_expected_crop_region.tif'

        candidate_filing = solutiona_dir+'intermediate7/'+candidate_naming
        img000 = cv2.imread(candidate_filing)
        gray000 = cv2.cvtColor(img000,cv2.COLOR_BGR2GRAY)

        # flood fill background to find inner holes
        holes = gray000.copy()
        cv2.floodFill(holes, None, (0, 0), 255)

        # invert holes mask, bitwise or with img fill in holes
        holes = cv2.bitwise_not(holes)
        filled_holes = cv2.bitwise_or(gray000, holes)

        #out_file_path0 = solutiona_dir+'intermediate6/cropped_map_mask(2)/'+candidate_naming.split('.')[0]+'_v2.tif'
        #cv2.imwrite(out_file_path0, filled_holes)

        #print(data_dir+'/'+candidate_naming.replace('_expected_crop_region', ''))
        img00 = cv2.imread(data_dir+'/'+candidate_naming.replace('_expected_crop_region', ''))
        img00 = cv2.cvtColor(img00,cv2.COLOR_BGR2GRAY)
        lower_color = np.array([0]) ### This shall be the color you want to crop off
        upper_color = np.array([90]) ### This shall be the color you want to crop off
        black_line = cv2.inRange(img00, lower_color, upper_color)

        blur_radius = 7.0
        threshold_blur = 255*0.0
        gaussian_buffer0 = ndimage.gaussian_filter(black_line, blur_radius)
        gaussian_buffer0[gaussian_buffer0 > threshold_blur] = 255
        gaussian_buffer0[gaussian_buffer0 <= threshold_blur] = 0

        # smooth the image
        blur_radius = 70.0
        threshold_blur = 255*0.1
        gaussian_buffer00 = ndimage.gaussian_filter(filled_holes, blur_radius)
        gaussian_buffer00[gaussian_buffer00 > threshold_blur] = 255
        gaussian_buffer00[gaussian_buffer00 <= threshold_blur] = 0

        #out_file_path0 = solutiona_dir+'intermediate6/cropped_map_mask(2)/'+candidate_naming.split('.')[0]+'_v3.tif'
        #cv2.imwrite(out_file_path0, gaussian_buffer00)

        added_region = cv2.bitwise_and(gaussian_buffer0, gaussian_buffer00)
        #out_file_path0 = solutiona_dir+'intermediate6/cropped_map_mask(2)/'+candidate_naming.split('.')[0]+'_v4.tif'
        #cv2.imwrite(out_file_path0, added_region)

        merged_region = cv2.bitwise_or(filled_holes, added_region)
        #out_file_path0 = solutiona_dir+'intermediate6/cropped_map_mask(2)/'+candidate_naming.split('.')[0]+'_v5.tif'
        #cv2.imwrite(out_file_path0, merged_region)

        # flood fill background to find inner holes
        holes = merged_region.copy()
        cv2.floodFill(holes, None, (0, 0), 255)

        # invert holes mask, bitwise or with img fill in holes
        holes = cv2.bitwise_not(holes)
        filled_holes_v2 = cv2.bitwise_or(merged_region, holes)

        #out_file_path0 = solutiona_dir+'intermediate6/cropped_map_mask(2)/'+candidate_naming.split('.')[0]+'_v6.tif'
        #cv2.imwrite(out_file_path0, filled_holes_v2)

        # find connected components
        labeled00, nr_objects00 = ndimage.label(filled_holes_v2 > threshold_blur)

        label_area = np.bincount(labeled00.flat)[1:] # 1
        arg_sort = np.argsort(label_area)

        selected_index = arg_sort[arg_sort.shape[0]-1]+1 # 1
        selected_map_for_examination = np.zeros((labeled00.shape[0],labeled00.shape[1],1),dtype=np.uint8)
        selected_map_for_examination[labeled00 == selected_index] = 255

        out_file_path0 = solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_expected_crop_region.tif'
        cv2.imwrite(out_file_path0, selected_map_for_examination)


        img0 = cv2.imread(file_path)
        rgb0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        #crop_rgb3 = cv2.bitwise_and(rgb0, rgb0, mask=selected_map_for_examination)
        blank = np.zeros((img0.shape[0],img0.shape[1],img0.shape[2]),dtype=np.uint8)
        blank[0:img0.shape[0],0:img0.shape[1],0:img0.shape[2]] = 255

        blank_mask = cv2.bitwise_and(blank, blank, mask=cv2.bitwise_not(selected_map_for_examination))
        crop_rgb2 = cv2.add(rgb0, blank_mask)
        print_bgr = cv2.cvtColor(crop_rgb2, cv2.COLOR_RGB2BGR)
        out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop.tif'
        cv2.imwrite(out_file_path0, print_bgr)


        '''Inversed floodfill to fill up inner holes / End'''




        #img_bw0_arg = np.argwhere(img_bw0 == 255)
        #print(img_bw0_arg.shape)

        img1 = np.copy(img0)
        rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        #print(rgb1.shape)
        '''
        crop_rgb1 = rgb1
        crop_img_bw0 = img_bw0
        crop_img_bw01 = img_bw01
        '''
        #print(rgb1.shape)
        #print(selected_map_for_examination.shape)

        
        #crop_rgb1 = cv2.bitwise_and(rgb1,rgb1, mask=selected_map_for_examination)
        crop_rgb1 = np.copy(crop_rgb2)
        crop_img_bw0 = cv2.bitwise_and(img_bw0,img_bw0, mask=selected_map_for_examination)
        crop_img_bw01 = cv2.bitwise_and(img_bw01,img_bw01, mask=selected_map_for_examination)
        
        


        #print(crop_rgb1.shape)
        #plt.imshow(crop_rgb1)

        print_bgr = cv2.cvtColor(crop_rgb1, cv2.COLOR_RGB2BGR)
        out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop.png'
        cv2.imwrite(out_file_path0, print_bgr)
        #break

        img_bw0_arg = np.argwhere(crop_img_bw0 == 255)
        #print(img_bw0_arg.shape)

    
    #plt.imshow(crop_img_bw01)
    #plt.show()
    out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop_grayregion.png'
    cv2.imwrite(out_file_path0, crop_img_bw01)

    out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop_blackregion.png'
    cv2.imwrite(out_file_path0, crop_img_bw0)




    trans_hsv1 = cv2.cvtColor(crop_rgb1, cv2.COLOR_RGB2HSV)

    print_bgr = cv2.cvtColor(crop_rgb1, cv2.COLOR_RGB2BGR)
    out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_remove_black.png'
    cv2.imwrite(out_file_path0, print_bgr)


    # without mean shift (for rgb)
    trans_rgb0 = cv2.cvtColor(trans_hsv1, cv2.COLOR_HSV2RGB)
    trans_hsv0 = cv2.cvtColor(trans_rgb0, cv2.COLOR_RGB2HSV)
    #crop_rgb1 = np.copy(trans_rgb0)
    #trans_hsv1 = np.copy(trans_hsv0)
    
    # mean shift (for rgb and hsv)
    spatialRadius = 10
    colorRadius = 25
    maxPyramidLevel = 2

    img = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_remove_black.png')
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img_rgb_part = cv2.bitwise_and(img_rgb, img_rgb, mask=selected_map_for_examination)
    

    result = cv2.pyrMeanShiftFiltering(img_rgb, spatialRadius, colorRadius, maxPyramidLevel)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    #crop_rgb1 = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    #trans_hsv1 = cv2.cvtColor(crop_rgb1, cv2.COLOR_RGB2HSV)
    out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_remove_black_mean_shift.png'
    cv2.imwrite(out_file_path0, result)


    '''Text removal from boundaries / Begin'''
    # black_pixel_for_text
    # black_text_for_text
    # black_line_for_text // black_line_for_text = black_pixel_for_text - black_text_for_text
    img_rb = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_remove_black.png')
    img_bound = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_expected_crop_region.tif')

    img_background_v0 = np.copy(img_rb)
    img_background_v0 = cv2.cvtColor(img_background_v0, cv2.COLOR_RGB2GRAY)

    lower_black_text = np.array([0])
    upper_black_text = np.array([70])
    mask_box_text0 = cv2.inRange(img_background_v0, lower_black_text, upper_black_text)
    res_box_text1 = cv2.bitwise_and(img_bound, img_bound, mask=mask_box_text0)

    threshold_text = cv2.medianBlur(res_box_text1,3)
    threshold_text_strict = cv2.medianBlur(res_box_text1,5)

    threshold_text_strict = ndimage.gaussian_filter(threshold_text_strict, sigma=20)
    '''
    out_file_path0=solutiona_dir+'intermediate6/0_poly_t1g.png'
    cv2.imwrite(out_file_path0, threshold_text_strict)
    '''
    threshold_text_strict[threshold_text_strict > 255.0*0.05] = 255
    threshold_text_strict[threshold_text_strict <= 255.0*0.05] = 0

    threshold_text0 = cv2.bitwise_and(threshold_text, threshold_text_strict)
    out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_text_detection.png'
    cv2.imwrite(out_file_path0, threshold_text0)

    threshold_line = cv2.subtract(threshold_text, threshold_text0)
    kernel_dilate = np.ones((5, 5), np.uint8)
    threshold_line = cv2.dilate(threshold_line, kernel_dilate, iterations=1)
    out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_boundary_detection.png'
    cv2.imwrite(out_file_path0, threshold_line)

    '''Text removal from boundaries / End'''
    
    return selected_map_for_examination