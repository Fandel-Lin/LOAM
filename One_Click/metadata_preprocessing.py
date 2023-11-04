
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from statistics import median
import csv
import math
import scipy
from scipy import ndimage
from scipy.signal import convolve2d
from scipy.stats import beta
import os
from os.path import exists
import cv2
from PIL import Image
import json
import math
from datetime import datetime
from skimage.morphology import skeletonize

import multiprocessing
#print(multiprocessing.cpu_count())
PROCESSES = 8


split_multiprocessing = True # Always set to True !!
for_each_loop_global = PROCESSES

map_cropping = True # Set to True if map cropping is needed
map_preprocessing = True  # Set to True if map preprocessing is needed

generate_boundary_extraction = True # Set to True if boundary extraction is needed (viewed as one of the preprocessing step)
generate_boundary_groundtruth = True # Set to True for one time

crop_legend = True # Always set to True (remove legend area from basemap)
preprocessing_recoloring = True # Always set to True under current setting
printing_auxiliary_information = True # Always set to True under current setting

simple_preprocessing = True # Set to True to support fast testing during development (always set to True during development)

smoothing_map = False # Always set to False under current setting
generate_text_pattern_probability = False # Always set to False under current setting
postprocessing_floodfill = False # Set to False under current setting

print_intermediate_image = True # create directory and output intermediate images 
# output for cropped map => 'intermediate6/cropped_map_mask'
# output for polygon extraction => 'intermediate7(2)'

data_dir='Data/validation' # path to input maps
data_boundary_dir='Data/validation_groundtruth' # path to groundtruth maps (only used for generating the groundtruth of boundary)
solutiona_dir='Solution_1102/'
targeted_map_list='targeted_map.csv'


import extractor_workers.preprocessing_worker as preprocessing_worker
import extractor_workers.cropping_worker as cropping_worker

import extractor_workers.extraction_step0_color_difference_worker as extraction_step0_color_difference_worker
import extractor_workers.extraction_step0_find_legend_in_map_worker as extraction_step0_find_legend_in_map_worker
import extractor_workers.extraction_step1_worker as extraction_step1_worker
import extractor_workers.extraction_step2_worker as extraction_step2_worker
import extractor_workers.extraction_step3_worker as extraction_step3_worker
import extractor_workers.extraction_step4_worker as extraction_step4_worker
import extractor_workers.extraction_step5_worker as extraction_step5_worker
import extractor_workers.extraction_step6_pre_update_worker as extraction_step6_pre_update_worker
import extractor_workers.extraction_step6_specify_overlap_legend_worker as extraction_step6_specify_overlap_legend_worker
import extractor_workers.extraction_step6_find_legend_in_map_worker as extraction_step6_find_legend_in_map_worker
import extractor_workers.extraction_step6_compare_against_competitor_worker as extraction_step6_compare_against_competitor_worker
import extractor_workers.extraction_step7_worker as extraction_step7_worker
import extractor_workers.extraction_step8_postprocessing_worker as extraction_step8_postprocessing_worker



def multiprocessing_setting():
    global PROCESSES

    multiprocessing.set_start_method('spawn', True)
    if PROCESSES > multiprocessing.cpu_count():
        PROCESSES = (int)(multiprocessing.cpu_count()/2)



def setting_summary():
    print('========================================== Polygon Extraction Setting ==========================================')
    print('*Solution directory => "' + solutiona_dir + '"')

    print('')
    print('*Intput for polygon extraction => "' + data_dir + '"')
    print('*Output for polygon extraction => "' + solutiona_dir + 'intermediate7(2)/Output"')

    if print_intermediate_image == True:
        print(' - Output for intermediate basemap => "' + solutiona_dir + 'intermediate7"')
        print(' - Output for intermediate extraction => "' + solutiona_dir + 'intermediate7(2)/(Map_Name)"')
        print(' - Output for cropped map => "' + solutiona_dir + 'intermediate6/cropped_map_mask" (Supporting point, line, polygon)')
    if generate_boundary_groundtruth == True:
        print(' - Output for boundary groundtruth => "' + solutiona_dir + 'intermediate5/Groundtruth"')
    if generate_boundary_extraction == True:
        print(' - Output for boundary extraction => "' + solutiona_dir + 'intermediate5/Extraction"')

    print('')
    print('*Adjustable (Set to "True" is highly recommended due to time complexity)')
    print(' - Multiprocessing => ' + str(split_multiprocessing))
    print(' - Simple preprocessing => ' + str(simple_preprocessing))
    print('')
    print('*Adjustable (Set to "True" for only one time to process the basemap)')
    print(' - Crop basemap => ' + str(map_cropping) + '\t\t\t\t (set to "True" only if one needs to crop those maps that do not have polygon features)')
    print(' - Preprocess basemap => ' + str(map_preprocessing) + '\t\t\t (set to "True" to crop those maps that have polygon features at the same time)')
    print(' - Generate boundary groundtruth => ' + str(generate_boundary_groundtruth) + '\t (set to "True" to generate boundary groundtruth)')
    #print('')
    #print('*Experimental (Set to "True" for experimental functionalities)')
    print(' - Extract boundaries as polygons => ' + str(generate_boundary_extraction) + '\t (set to "True" to extract boundaries/ black as a polygon key)')
    print('')
    print('*Currently fixed')
    print(' - Crop legend => ' + str(crop_legend) + ' (shall be "True")')
    print(' - Input DTM-smoothed basemap => ' + str(smoothing_map) + ' (shall be "False")')
    print(' - Generate text pattern probability => ' + str(generate_text_pattern_probability) + ' (shall be "False")')

    print('')
    if split_multiprocessing == True:
        print('*Multiprocessing with '+str(PROCESSES)+' processes...')

    print('================================================================================================================')

    if not os.path.exists(solutiona_dir):
        os.makedirs(solutiona_dir)




def specify_polygon():
    print('')
    print('=== Specify maps with polygon features ===')
    global candidate_file_name_for_polygon
    global poly_legend_counter

    candidate_file_name_for_polygon = []
    poly_legend_counter = []
    #poly_legend_counter_v2 = []
    for file_name in os.listdir(data_dir):
        if '.json' in file_name:
            filename=file_name.replace('.json', '.tif')
            #print('Working on map:', file_name)
            file_path=os.path.join(data_dir, filename)
            test_json=file_path.replace('.tif', '.json')
            
            poly_counter = 0
            legend_counter = 0
            poly_name_list = []

            with open(test_json) as f:
                gj = json.load(f)
            for this_gj in gj['shapes']:
                #print(this_gj)
                names = this_gj['label']
                features = this_gj['points']
                
                if '_poly' not in names and '_pt' not in names and '_line' not in names:
                    print(names)
                if '_poly' not in names:
                    continue
                if names not in poly_name_list:
                    poly_name_list.append(names)
                legend_counter = legend_counter + 1
                
            if legend_counter > 0:
                poly_counter = poly_counter + 1
                poly_legend_counter.append(len(poly_name_list))
                #poly_legend_counter.append(legend_counter)
                #poly_legend_counter_v2.append(len(poly_name_list))
            
            if poly_counter > 0:
                candidate_file_name_for_polygon.append(file_name)
    #print(len(candidate_file_name_for_polygon))
    #print(candidate_file_name_for_polygon)
    #print(poly_legend_counter)
    #print(poly_legend_counter_v2)



    file_target_map = open(targeted_map_list, 'r')
    data_target_map = list(csv.reader(file_target_map, delimiter=','))
    file_target_map.close()
    #print(data_target_map)



    candidate_file_name_for_polygon_temp = candidate_file_name_for_polygon.copy()
    poly_legend_counter_temp = poly_legend_counter.copy()

    candidate_file_name_for_polygon = []
    poly_legend_counter = []

    for file_counter in range(0, len(candidate_file_name_for_polygon_temp)):
        file_name = candidate_file_name_for_polygon_temp[file_counter]
        #print(file_name.split('.')[0])
        if any(file_name.split('.')[0] in target_map for target_map in data_target_map):
            candidate_file_name_for_polygon.append(file_name)
            poly_legend_counter.append(poly_legend_counter_temp[file_counter])

    print('Total number of maps to extract: ', len(candidate_file_name_for_polygon))
    print('Title of maps: ', candidate_file_name_for_polygon)
    print('Number of keys: ', poly_legend_counter)

    

def worker_preprocessing():
    print('')
    print('=== Specify maps with polygon features ===')
    runningtime_start_global = datetime.now()


    #data_dir='validation'
    if not os.path.exists(solutiona_dir+str('intermediate7/')):
        os.makedirs(solutiona_dir+str('intermediate7/'))


    # import preprocessing_worker
    if map_preprocessing == True:
        if split_multiprocessing:
            with multiprocessing.Pool(int(PROCESSES/2)) as pool:
                callback = pool.starmap_async(preprocessing_worker.preprocessing_worker, [(this_map,candidate_file_name_for_polygon[this_map], data_dir, solutiona_dir, crop_legend, ) for this_map in range(0, len(candidate_file_name_for_polygon))])
                multiprocessing_results = callback.get()
                
                for this_map in multiprocessing_results:
                    this_crop_map = this_map
                    # plt.imshow(this_crop_map)
                    # plt.show()
        else:
            for this_map in range(0, len(candidate_file_name_for_polygon)):
                preprocessing_worker.preprocessing_worker(this_map)
        print('time check... worker_preprocessing: ', datetime.now()-runningtime_start_global)
    else:
        print('Preprocessing already done...')
    






## def supporting further usage

def generate_mask(given_size):
    sliding_window_size = given_size
    initial_mask = np.zeros((sliding_window_size*2+3,sliding_window_size*2+3), dtype=int)
    center_j = sliding_window_size+1
    center_i = sliding_window_size+1
    initial_mask[center_j-1:center_j+2, center_i-1:center_i+2] = -1

    initial_mask[center_j-3][center_i-1] = 2
    initial_mask[center_j-3][center_i+1] = 2
    initial_mask[center_j-2][center_i] = 2

    initial_mask[center_j-1][center_i+3] = 4
    initial_mask[center_j+1][center_i+3] = 4
    initial_mask[center_j][center_i+2] = 4

    initial_mask[center_j+3][center_i-1] = 6
    initial_mask[center_j+3][center_i+1] = 6
    initial_mask[center_j+2][center_i] = 6

    initial_mask[center_j-1][center_i-3] = 8
    initial_mask[center_j+1][center_i-3] = 8
    initial_mask[center_j][center_i-2] = 8


    for j in range(0, sliding_window_size-1):
        for i in range(center_i-(sliding_window_size-2-j), center_i+(sliding_window_size-2-j)+1):
            initial_mask[j][i] = 2

    for j in range(sliding_window_size+3+1, sliding_window_size*2+3):
        for i in range(center_i-(j-(sliding_window_size+3+1)), center_i+(j-(sliding_window_size+3+1))+1):
            initial_mask[i][j] = 4

    for j in range(sliding_window_size+3+1, sliding_window_size*2+3):
        for i in range(center_i-(j-(sliding_window_size+3+1)), center_i+(j-(sliding_window_size+3+1))+1):
            initial_mask[j][i] = 6

    for j in range(0, sliding_window_size-1):
        for i in range(center_i-(sliding_window_size-2-j), center_i+(sliding_window_size-2-j)+1):
            initial_mask[i][j] = 8

    initial_mask_arg = np.argwhere(initial_mask == 0)
    for i, j in initial_mask_arg:
        if i<=sliding_window_size and j<=sliding_window_size:
            initial_mask[i][j] = 1
        elif i<=sliding_window_size and j>=sliding_window_size:
            initial_mask[i][j] = 3
        elif i>=sliding_window_size and j>sliding_window_size:
            initial_mask[i][j] = 5
        elif i>=sliding_window_size and j<=sliding_window_size:
            initial_mask[i][j] = 7
    #print(initial_mask)


    masking = []
    for direction in range(0, 8):
        masking.append(np.zeros((sliding_window_size*2+3,sliding_window_size*2+3), dtype=bool))
    masking = np.array(masking)

    for direction in range(0, 8):
        initial_mask_arg = np.argwhere(initial_mask == (direction+1))
        for i, j in initial_mask_arg:
            masking[direction][i][j] = True
    #print(masking.shape)

    return masking

def kernel_assign_white(img, i, j):
    img[max(i-1, 0)][max(j-1, 0)] = 255
    img[max(i-1, 0)][j] = 255
    img[max(i-1, 0)][min(j+1, 0)] = 255
    img[i][max(j-1, 0)] = 255
    img[i][j] = 255
    img[i][min(j+1, 0)] = 255
    img[min(i+1, 0)][max(j-1, 0)] = 255
    img[min(i+1, 0)][j] = 255
    img[min(i+1, 0)][min(j+1, 0)] = 255

def center_assign_white(img, i, j):
    img[i][j] = 255










def worker_boundary_extraction():
    print('')
    print('=== Extract boundaries from the maps ===')
    runningtime_start_global = datetime.now()


    if not os.path.exists(solutiona_dir+str('intermediate5/Extraction/')):
        os.makedirs(solutiona_dir+str('intermediate5/Extraction/'))
        
    if generate_boundary_extraction == True:
        for target_file_q in range(0, len(candidate_file_name_for_polygon), 1):
        #for target_file_q in range(len(candidate_file_name_for_polygon)-1, 0, -1):
        #for target_file_q in range(4, 5, 1):
            file_name = candidate_file_name_for_polygon[target_file_q]
            
            # get the .tif files
            if '.json' in file_name:
                runningtime_start=datetime.now()


                filename=file_name.replace('.json', '.tif')
                print('Working on map:', file_name)
                file_path=os.path.join(data_dir, filename)
                test_json=file_path.replace('.tif', '.json')
                file_name_json = test_json.replace('.json', '.json')
                
                #print(test_json)
                img000 = cv2.imread(file_path)
                #hsv0 = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
                #rgb0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                img_bound = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_expected_crop_region.tif')
                img_bound = cv2.cvtColor(img_bound, cv2.COLOR_BGR2GRAY)

                img_crop_gray = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop_grayregion.png')
                img_crop_black = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop_blackregion.png')
                img_crop_gray = cv2.cvtColor(img_crop_gray, cv2.COLOR_BGR2GRAY)
                img_crop_black = cv2.cvtColor(img_crop_black, cv2.COLOR_BGR2GRAY)

                img_rb = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_remove_black.png')
                img_ms = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_remove_black_mean_shift.png')






                this_current_image = np.copy(img_rb)
                overall_overlapping = np.zeros((this_current_image.shape[0], this_current_image.shape[1]), dtype='uint8')

                # black_pixel_for_text
                # black_text_for_text
                # black_line_for_text // black_line_for_text = black_pixel_for_text - black_text_for_text
                text_pattern_probability = True
                img_backgroun_v0 = np.copy(img_rb)
                img_backgroun_v0 = cv2.cvtColor(img_backgroun_v0, cv2.COLOR_RGB2GRAY)

                lower_black_text = np.array([0])
                upper_black_text = np.array([65])
                mask_box_text0 = cv2.inRange(img_backgroun_v0, lower_black_text, upper_black_text)
                res_box_text1 = cv2.bitwise_and(img_bound, img_bound, mask=mask_box_text0)
                black_pixel_for_text = np.copy(res_box_text1)

                threshold_text = cv2.medianBlur(res_box_text1,3)





                with open(file_name_json) as f:
                    gj = json.load(f)
                json_height = gj['imageHeight']
                json_width = gj['imageWidth']
                rescale_factor_0 = 1.0
                rescale_factor_1 = 1.0



                ## Non-white background
                non_white_background = False
                if np.sum(img_bound) / 255 >= (img_bound.shape[0]*img_bound.shape[1]) * 0.99 or np.unique(img_bound).shape[0] == 1:
                    lower_white = np.array([250,250,250])
                    upper_white = np.array([256,256,256])
                    mask_white_img000 = cv2.inRange(img000, lower_white, upper_white)
                    lower_white = np.array([0,0,0])
                    upper_white = np.array([130,130,130])
                    mask_white_img000_2 = cv2.inRange(img000, lower_white, upper_white)
                    mask_white_img000 = cv2.bitwise_or(mask_white_img000, mask_white_img000_2)

                    corner_avg_white = np.sum(mask_white_img000[int(mask_white_img000.shape[0]*98/100): int(mask_white_img000.shape[0]*99/100), int(mask_white_img000.shape[1]*98/100): int(mask_white_img000.shape[1]*99/100)])/255.0
                    corner_area = (int(mask_white_img000.shape[0]*99/100) - int(mask_white_img000.shape[0]*98/100)) * (int(mask_white_img000.shape[1]*99/100) - int(mask_white_img000.shape[1]*98/100))

                    if corner_avg_white / corner_area < 0.66:
                        non_white_background = True
                        print('non_white_background')



                

                ### Legend is always not considered
                if True:
                    for this_gj in gj['shapes']:
                        #print(this_gj)
                        names = this_gj['label']
                        features = this_gj['points']

                        geoms = np.array(features)
                        y_min = int(np.min(geoms, axis=0)[0]*rescale_factor_1)
                        y_max = int(np.max(geoms, axis=0)[0]*rescale_factor_1)
                        x_min = int(np.min(geoms, axis=0)[1]*rescale_factor_0)
                        x_max = int(np.max(geoms, axis=0)[1]*rescale_factor_0)

                        legend_mask = np.ones((img_rb.shape[0], img_rb.shape[1]), dtype='uint8') *255
                        legend_mask[x_min:x_max, y_min:y_max] = 0
                        img_bound = cv2.bitwise_and(img_bound, legend_mask)
                    img_rb = cv2.bitwise_and(img_rb, img_rb, mask=img_bound)
                    img_ms = cv2.bitwise_and(img_ms, img_ms, mask=img_bound)
                    img_crop_gray = cv2.bitwise_and(img_crop_gray, img_crop_gray, mask=img_bound)
                    img_crop_black = cv2.bitwise_and(img_crop_black, img_crop_black, mask=img_bound)
                hsv_rb = cv2.cvtColor(img_rb, cv2.COLOR_BGR2HSV)
                rgb_rb = cv2.cvtColor(img_rb, cv2.COLOR_BGR2RGB)
                hsv_ms = cv2.cvtColor(img_ms, cv2.COLOR_BGR2HSV)
                rgb_ms = cv2.cvtColor(img_ms, cv2.COLOR_BGR2RGB)


                
                poly_counter = 0
                color_space = []
                color_avg = []
                color_avg2 = []
                map_name = file_name.replace('.json', '')
                legend_name = []
                legend_name_check = []
                extracted_legend_name = []


                hsv_space = np.zeros((255), dtype='uint8') # only for h space
                rgb_space = np.zeros((255,255,3), dtype='uint8')


                if not os.path.exists(solutiona_dir+'intermediate5/Extraction/'+map_name):
                    os.makedirs(solutiona_dir+'intermediate5/Extraction/'+map_name)



                temp_legend_name = []
                temp_legend_feature = []
                for this_gj in gj['shapes']:
                    #if '_poly' not in names:
                        #continue
                    #print(this_gj)
                    names = this_gj['label']
                    features = this_gj['points']
                    
                    if '_poly' not in names:
                        continue
                    if names in legend_name_check:
                        continue

                    ### Read json source for the legend
                    geoms = np.array(features)
                    y_min = int(np.min(geoms, axis=0)[0]*rescale_factor_1)
                    y_max = int(np.max(geoms, axis=0)[0]*rescale_factor_1)
                    x_min = int(np.min(geoms, axis=0)[1]*rescale_factor_0)
                    x_max = int(np.max(geoms, axis=0)[1]*rescale_factor_0)

                    img_legend = np.zeros((x_max-x_min, y_max-y_min, 3), dtype='uint8')
                    img_legend = np.copy(img000[x_min:x_max, y_min:y_max, :])


                    legend_name_check.append(names)
                    legend_name.append(names.replace('_poly',''))

                    temp_legend_name.append(names.replace('_poly',''))
                    temp_legend_feature.append(img_legend)

                    poly_counter = poly_counter+1


                    
                    
                
                if split_multiprocessing == True:
                    with multiprocessing.Pool(int(PROCESSES*2)) as pool:
                        callback = pool.starmap_async(extraction_step0_find_legend_in_map_worker.extraction_step0_find_legend_in_map_worker, [(this_poly, map_name, temp_legend_name[this_poly], temp_legend_feature[this_poly], solutiona_dir, threshold_text, img_crop_black, np.sum(img_bound), text_pattern_probability, print_intermediate_image, ) for this_poly in range(0, poly_counter)])
                        multiprocessing_results = callback.get()

                        for legend, overlapping in multiprocessing_results:
                            overall_overlapping = cv2.bitwise_or(overall_overlapping, overlapping)
                    

                out_file_path0=solutiona_dir+'intermediate5/Extraction/'+map_name+'_overall_text.png'
                cv2.imwrite(out_file_path0, overall_overlapping)

                inversed_overall = cv2.bitwise_and(img_crop_black, 255-overall_overlapping)
                out_file_path0=solutiona_dir+'intermediate5/Extraction/'+map_name+'_overall_boundary.png'
                cv2.imwrite(out_file_path0, inversed_overall)
                

                print('time checkpoint _vt0:', datetime.now()-runningtime_start)
    else:
        print('No need to extract boundaries (extraction already done)...')


    # 21m 50.6s





    if generate_boundary_extraction == True:
        #data_dir='validation'
        if not os.path.exists(solutiona_dir+str('intermediate5/')):
            os.makedirs(solutiona_dir+str('intermediate5/'))
        if not os.path.exists(solutiona_dir+str('intermediate5/Extraction(2)/')):
            os.makedirs(solutiona_dir+str('intermediate5/Extraction(2)/'))
        

        for target_file_q in range(0, len(candidate_file_name_for_polygon), 1):
        #for target_file_q in range(len(candidate_file_name_for_polygon)-1, 0, -1):
        #for target_file_q in range(4, 5, 1):
            file_name = candidate_file_name_for_polygon[target_file_q]
            running_time_v = []
            
            
            # get the .tif files
            if '.json' in file_name:
                runningtime_start=datetime.now()


                filename=file_name.replace('.json', '.tif')
                print('Working on map:', file_name)
                file_path=os.path.join(data_dir, filename)
                test_json=file_path.replace('.tif', '.json')
                file_name_json = test_json.replace('.json', '.json')
                
                #print(test_json)
                img000 = cv2.imread(file_path)
                #hsv0 = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
                #rgb0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                img_bound = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_expected_crop_region.tif')
                img_bound = cv2.cvtColor(img_bound, cv2.COLOR_BGR2GRAY)

                img_crop_gray = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop_grayregion.png')
                img_crop_black = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop_blackregion.png')
                img_crop_gray = cv2.cvtColor(img_crop_gray, cv2.COLOR_BGR2GRAY)
                img_crop_black = cv2.cvtColor(img_crop_black, cv2.COLOR_BGR2GRAY)

                img_rb = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_remove_black.png')
                img_ms = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_remove_black_mean_shift.png')


                with open(file_name_json) as f:
                    gj = json.load(f)
                json_height = gj['imageHeight']
                json_width = gj['imageWidth']
                rescale_factor_0 = 1.0
                rescale_factor_1 = 1.0



                ## Non-white background
                non_white_background = False
                if np.sum(img_bound) / 255 >= (img_bound.shape[0]*img_bound.shape[1]) * 0.99 or np.unique(img_bound).shape[0] == 1:
                    lower_white = np.array([250,250,250])
                    upper_white = np.array([256,256,256])
                    mask_white_img000 = cv2.inRange(img000, lower_white, upper_white)
                    lower_white = np.array([0,0,0])
                    upper_white = np.array([130,130,130])
                    mask_white_img000_2 = cv2.inRange(img000, lower_white, upper_white)
                    mask_white_img000 = cv2.bitwise_or(mask_white_img000, mask_white_img000_2)

                    corner_avg_white = np.sum(mask_white_img000[int(mask_white_img000.shape[0]*98/100): int(mask_white_img000.shape[0]*99/100), int(mask_white_img000.shape[1]*98/100): int(mask_white_img000.shape[1]*99/100)])/255.0
                    corner_area = (int(mask_white_img000.shape[0]*99/100) - int(mask_white_img000.shape[0]*98/100)) * (int(mask_white_img000.shape[1]*99/100) - int(mask_white_img000.shape[1]*98/100))

                    if corner_avg_white / corner_area < 0.66:
                        non_white_background = True
                        print('non_white_background')



                

                ### Legend is always not considered
                if True:
                    for this_gj in gj['shapes']:
                        #print(this_gj)
                        names = this_gj['label']
                        features = this_gj['points']

                        geoms = np.array(features)
                        y_min = int(np.min(geoms, axis=0)[0]*rescale_factor_1)
                        y_max = int(np.max(geoms, axis=0)[0]*rescale_factor_1)
                        x_min = int(np.min(geoms, axis=0)[1]*rescale_factor_0)
                        x_max = int(np.max(geoms, axis=0)[1]*rescale_factor_0)

                        legend_mask = np.ones((img_rb.shape[0], img_rb.shape[1]), dtype='uint8') *255
                        legend_mask[x_min:x_max, y_min:y_max] = 0
                        img_bound = cv2.bitwise_and(img_bound, legend_mask)
                    img_rb = cv2.bitwise_and(img_rb, img_rb, mask=img_bound)
                    img_ms = cv2.bitwise_and(img_ms, img_ms, mask=img_bound)
                    img_crop_gray = cv2.bitwise_and(img_crop_gray, img_crop_gray, mask=img_bound)
                    img_crop_black = cv2.bitwise_and(img_crop_black, img_crop_black, mask=img_bound)
                hsv_rb = cv2.cvtColor(img_rb, cv2.COLOR_BGR2HSV)
                rgb_rb = cv2.cvtColor(img_rb, cv2.COLOR_BGR2RGB)
                hsv_ms = cv2.cvtColor(img_ms, cv2.COLOR_BGR2HSV)
                rgb_ms = cv2.cvtColor(img_ms, cv2.COLOR_BGR2RGB)

                laplacian = cv2.Laplacian(hsv_rb,cv2.CV_64F)
                if print_intermediate_image == True:
                    out_file_path0=solutiona_dir+'intermediate5/Extraction(2)/'+file_name.replace('.json', '')+'_hsv_rb.png'
                    cv2.imwrite(out_file_path0, laplacian)
                laplacian = cv2.Laplacian(rgb_rb,cv2.CV_64F)
                if print_intermediate_image == True:
                    out_file_path0=solutiona_dir+'intermediate5/Extraction(2)/'+file_name.replace('.json', '')+'_rgb_rb.png'
                    cv2.imwrite(out_file_path0, laplacian)
                laplacian = cv2.Laplacian(hsv_ms,cv2.CV_64F)
                if print_intermediate_image == True:
                    out_file_path0=solutiona_dir+'intermediate5/Extraction(2)/'+file_name.replace('.json', '')+'_hsv_ms.png'
                    cv2.imwrite(out_file_path0, laplacian)
                laplacian = cv2.Laplacian(rgb_ms,cv2.CV_64F)
                if print_intermediate_image == True:
                    out_file_path0=solutiona_dir+'intermediate5/Extraction(2)/'+file_name.replace('.json', '')+'_rgb_ms.png'
                    cv2.imwrite(out_file_path0, laplacian)

                for candidate_space in range(0,3):
                    img = np.copy(hsv_ms[:,:,candidate_space])
                    laplacian = cv2.Laplacian(img,cv2.CV_64F)
                    if print_intermediate_image == True:
                        out_file_path0=solutiona_dir+'intermediate5/Extraction(2)/'+file_name.replace('.json', '')+'_hsv_ms_'+str(candidate_space)+'.png'
                        cv2.imwrite(out_file_path0, laplacian)

    else:
        print('No need to extract boundaries...')

    # 11m 16.5s








    if generate_boundary_extraction == True:
        #data_dir='validation'
        if not os.path.exists(solutiona_dir+str('intermediate5/')):
            os.makedirs(solutiona_dir+str('intermediate5/'))
        if not os.path.exists(solutiona_dir+str('intermediate5/Extraction(3)/')):
            os.makedirs(solutiona_dir+str('intermediate5/Extraction(3)/'))
        

        for target_file_q in range(0, len(candidate_file_name_for_polygon), 1):
        #for target_file_q in range(len(candidate_file_name_for_polygon)-1, 0, -1):
        #for target_file_q in range(4, 5, 1):
            file_name = candidate_file_name_for_polygon[target_file_q]
            running_time_v = []
            
            
            # get the .tif files
            if '.json' in file_name:
                runningtime_start=datetime.now()


                filename=file_name.replace('.json', '.tif')
                print('Working on map:', file_name)

                img_cand_1 = cv2.imread(solutiona_dir+'intermediate5/Extraction(2)/'+file_name.replace('.json', '')+'_hsv_ms_1.png')
                img_cand_1 = cv2.cvtColor(img_cand_1, cv2.COLOR_BGR2GRAY)

                img_cand_2 = cv2.imread(solutiona_dir+'intermediate5/Extraction(2)/'+file_name.replace('.json', '')+'_hsv_ms_2.png')
                img_cand_2 = cv2.cvtColor(img_cand_2, cv2.COLOR_BGR2GRAY)

                dilate_kernel = np.ones((3,3), np.uint8)
                img_cand_3 = cv2.dilate(img_cand_1, dilate_kernel, iterations=1)

                if print_intermediate_image == True:
                    out_file_path0=solutiona_dir+'intermediate5/Extraction(3)/'+file_name.replace('.json', '')+'_hsv_ms_1_dilation.png'
                    cv2.imwrite(out_file_path0, img_cand_3)

                img_cand_comb = cv2.add(img_cand_3, img_cand_2)

                #erode_kernel = np.ones((1,1), np.uint8)
                #img_cand_comb = cv2.erode(img_cand_comb, erode_kernel, iterations=1)
                if print_intermediate_image == True:
                    out_file_path0=solutiona_dir+'intermediate5/Extraction(3)/'+file_name.replace('.json', '')+'_hsv_ms_1_combined.png'
                    cv2.imwrite(out_file_path0, img_cand_comb)

                #img_cand_comb[img_cand_comb > 127] = 255
                #img_cand_comb[img_cand_comb <= 127] = 0

                #erode_kernel = np.ones((2,2), np.uint8)
                #img_cand_comb = cv2.dilate(img_cand_comb, erode_kernel, iterations=1)
                #img_cand_comb = cv2.erode(img_cand_comb, erode_kernel, iterations=1)

                img_cand_comb[img_cand_comb > 255*0.05] = 255
                img_cand_comb[img_cand_comb <= 255*0.05] = 0

                img_cand_5 = cv2.imread(solutiona_dir+'intermediate5/Extraction/'+file_name.replace('.json', '')+'_overall_boundary.png')
                img_cand_5 = cv2.cvtColor(img_cand_5, cv2.COLOR_BGR2GRAY)

                #blur_radius = 3
                #img_cand_5_blur = ndimage.gaussian_filter(img_cand_5, blur_radius)
                #img_cand_5_blur[img_cand_5_blur > 0.25] = 255
                #img_cand_5_blur[img_cand_5_blur <= 0.25] = 0

                #erode_kernel = np.ones((5,5), np.uint8)
                #img_cand_5_blur = cv2.dilate(img_cand_5, erode_kernel, iterations=1)
                img_cand_5_blur = np.copy(img_cand_5)

                if print_intermediate_image == True:
                    out_file_path0=solutiona_dir+'intermediate5/Extraction(3)/'+file_name.replace('.json', '')+'_boundary_blur.png'
                    cv2.imwrite(out_file_path0, img_cand_5_blur)
                
                #print(img_cand_comb.shape)
                #print(img_cand_5_blur.shape)
                img_cand_6 = cv2.bitwise_and(img_cand_comb, img_cand_5_blur)

                if print_intermediate_image == True:
                    out_file_path0=solutiona_dir+'intermediate5/Extraction(3)/'+file_name.replace('.json', '')+'_overall_boundary_candidate.png'
                    cv2.imwrite(out_file_path0, img_cand_6)
                
                

                
        print('time check... worker_boundary_extraction: ', datetime.now()-runningtime_start_global)
    else:
        print('No need to extract boundaries (extraction already done)...')

    # 1m 44.2s





def worker_auxiliary_info():
    print('')
    print('=== Extract auxiliary information from the maps ===')
    runningtime_start_global = datetime.now()


    if printing_auxiliary_information == True:
        # auxiliary information needs to be printed:
        ### number of legends in one map
        ### range of H space for each legend
        ### difference in (1) H space (2) RGB-min (3) RGB-dist to the nearest 10 other legends for each legend

        #data_dir='validation'
        if not os.path.exists(solutiona_dir+str('intermediate9/')):
            os.makedirs(solutiona_dir+str('intermediate9/'))

        for target_file_q in range(0, len(candidate_file_name_for_polygon), 1):
        #for target_file_q in range(len(candidate_file_name_for_polygon)-1, 0, -1):
        #for target_file_q in range(4, 5, 1):
            file_name = candidate_file_name_for_polygon[target_file_q]
            running_time_v = []
            
            
            # get the .tif files
            if '.json' in file_name:
                runningtime_start=datetime.now()


                filename=file_name.replace('.json', '.tif')
                print('Working on map:', file_name)
                file_path=os.path.join(data_dir, filename)
                test_json=file_path.replace('.tif', '.json')
                file_name_json = test_json.replace('.json', '.json')
                
                #print(test_json)
                img000 = cv2.imread(file_path)
                #hsv0 = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
                #rgb0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                img_bound = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_expected_crop_region.tif')
                img_bound = cv2.cvtColor(img_bound, cv2.COLOR_BGR2GRAY)

                img_crop_gray = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop_grayregion.png')
                img_crop_black = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop_blackregion.png')
                img_crop_gray = cv2.cvtColor(img_crop_gray, cv2.COLOR_BGR2GRAY)
                img_crop_black = cv2.cvtColor(img_crop_black, cv2.COLOR_BGR2GRAY)

                #img_boundary = cv2.imread(solutiona_dir+'intermediate5/Extraction/'+file_name.replace('.json', '')+'_overall_boundary.png')
                img_boundary = cv2.imread(solutiona_dir+'intermediate5/Extraction(3)/'+file_name.replace('.json', '')+'_overall_boundary_candidate.png')
                img_boundary = cv2.cvtColor(img_boundary, cv2.COLOR_BGR2GRAY)

                img_rb = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_remove_black.png')
                img_ms = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_remove_black_mean_shift.png')


                with open(file_name_json) as f:
                    gj = json.load(f)
                json_height = gj['imageHeight']
                json_width = gj['imageWidth']
                rescale_factor_0 = 1.0
                rescale_factor_1 = 1.0



                ## Non-white background
                non_white_background = False
                if np.sum(img_bound) / 255 >= (img_bound.shape[0]*img_bound.shape[1]) * 0.99 or np.unique(img_bound).shape[0] == 1:
                    lower_white = np.array([250,250,250])
                    upper_white = np.array([256,256,256])
                    mask_white_img000 = cv2.inRange(img000, lower_white, upper_white)
                    lower_white = np.array([0,0,0])
                    upper_white = np.array([130,130,130])
                    mask_white_img000_2 = cv2.inRange(img000, lower_white, upper_white)
                    mask_white_img000 = cv2.bitwise_or(mask_white_img000, mask_white_img000_2)

                    corner_avg_white = np.sum(mask_white_img000[int(mask_white_img000.shape[0]*98/100): int(mask_white_img000.shape[0]*99/100), int(mask_white_img000.shape[1]*98/100): int(mask_white_img000.shape[1]*99/100)])/255.0
                    corner_area = (int(mask_white_img000.shape[0]*99/100) - int(mask_white_img000.shape[0]*98/100)) * (int(mask_white_img000.shape[1]*99/100) - int(mask_white_img000.shape[1]*98/100))

                    if corner_avg_white / corner_area < 0.66:
                        non_white_background = True
                        print('non_white_background')



                

                ### Legend is always not considered
                if True:
                    for this_gj in gj['shapes']:
                        #print(this_gj)
                        names = this_gj['label']
                        features = this_gj['points']

                        geoms = np.array(features)
                        y_min = int(np.min(geoms, axis=0)[0]*rescale_factor_1)
                        y_max = int(np.max(geoms, axis=0)[0]*rescale_factor_1)
                        x_min = int(np.min(geoms, axis=0)[1]*rescale_factor_0)
                        x_max = int(np.max(geoms, axis=0)[1]*rescale_factor_0)

                        legend_mask = np.ones((img_rb.shape[0], img_rb.shape[1]), dtype='uint8') *255
                        legend_mask[x_min:x_max, y_min:y_max] = 0
                        img_bound = cv2.bitwise_and(img_bound, legend_mask)
                    img_rb = cv2.bitwise_and(img_rb, img_rb, mask=img_bound)
                    img_ms = cv2.bitwise_and(img_ms, img_ms, mask=img_bound)
                    img_crop_gray = cv2.bitwise_and(img_crop_gray, img_crop_gray, mask=img_bound)
                    img_crop_black = cv2.bitwise_and(img_crop_black, img_crop_black, mask=img_bound)
                hsv_rb = cv2.cvtColor(img_rb, cv2.COLOR_BGR2HSV)
                rgb_rb = cv2.cvtColor(img_rb, cv2.COLOR_BGR2RGB)
                hsv_ms = cv2.cvtColor(img_ms, cv2.COLOR_BGR2HSV)
                rgb_ms = cv2.cvtColor(img_ms, cv2.COLOR_BGR2RGB)



                
                poly_counter = 0
                color_space = []
                color_avg = []
                color_avg2 = []
                color_dif = []
                color_key_variety = []
                map_name = file_name.replace('.json', '')
                legend_name = []
                legend_name_check = []
                extracted_legend_name = []


                hsv_space = np.zeros((255), dtype='uint8') # only for h space
                rgb_space = np.zeros((255,255,3), dtype='uint8')


                if not os.path.exists(solutiona_dir+'intermediate7(2)/'+map_name):
                    os.makedirs(solutiona_dir+'intermediate7(2)/'+map_name)



                for this_gj in gj['shapes']:
                    #if '_poly' not in names:
                        #continue
                    #print(this_gj)
                    names = this_gj['label']
                    features = this_gj['points']
                    
                    if '_poly' not in names:
                        continue
                    if names in legend_name_check:
                        continue


                    legend_name_check.append(names)
                    legend_name.append(names.replace('_poly',''))

                    poly_counter = poly_counter+1


                    ### There is no groundtruth for validation data
                    #print('training/'+map_name+'_'+names+'.tif')


                    ### Read json source for the legend
                    geoms = np.array(features)
                    y_min = int(np.min(geoms, axis=0)[0]*rescale_factor_1)
                    y_max = int(np.max(geoms, axis=0)[0]*rescale_factor_1)
                    x_min = int(np.min(geoms, axis=0)[1]*rescale_factor_0)
                    x_max = int(np.max(geoms, axis=0)[1]*rescale_factor_0)

                    img_legend = np.zeros((x_max-x_min, y_max-y_min, 3), dtype='uint8')
                    img_legend = np.copy(img000[x_min:x_max, y_min:y_max, :])
                    
                    if print_intermediate_image == True:
                        out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+names+'_legend.tif'
                        cv2.imwrite(out_file_path0, img_legend)
                    
                    
                    img_legend = cv2.cvtColor(img_legend, cv2.COLOR_BGR2RGB)
                    img_legend = img_legend[int(img_legend.shape[0]/8):int(img_legend.shape[0]*7/8), int(img_legend.shape[1]/8):int(img_legend.shape[1]*7/8), :]
                    hsv_legend = cv2.cvtColor(img_legend, cv2.COLOR_RGB2HSV)
                    black_threshold = 30 #130
                    white_threshold = 250 #245

                    lower_black_rgb_trimmed0 = np.array([0,0,0])
                    upper_black_rgb_trimmed0 = np.array([130,130,130])
                    mask_test_img_legend = cv2.inRange(img_legend, lower_black_rgb_trimmed0, upper_black_rgb_trimmed0)
                    if np.sum(mask_test_img_legend == 255) > np.sum(img_legend > 0) * 0.25:
                        black_threshold = 30
                    
                    rgb_trimmed = np.zeros((img_legend.shape[2], img_legend.shape[0], img_legend.shape[1]), dtype='uint8')
                    hsv_trimmed = np.zeros((img_legend.shape[2], img_legend.shape[0], img_legend.shape[1]), dtype='uint8')
                    rgb_trimmed = rgb_trimmed.astype(float)
                    hsv_trimmed = hsv_trimmed.astype(float)
                    for dimension in range(0, 3):
                        rgb_trimmed[dimension] = np.copy(img_legend[:,:,dimension]).astype(float)
                        hsv_trimmed[dimension] = np.copy(hsv_legend[:,:,dimension]).astype(float)

                    rgb_trimmed_temp = np.copy(rgb_trimmed)
                    rgb_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
                    hsv_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
                    rgb_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
                    hsv_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
                    rgb_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
                    hsv_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan

                    rgb_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
                    hsv_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
                    rgb_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
                    hsv_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
                    rgb_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
                    hsv_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan



                    if np.sum(np.isnan(hsv_trimmed)) >= (hsv_trimmed.shape[0]*hsv_trimmed.shape[1]*hsv_trimmed.shape[2]):
                        color_space_holder = []
                        rgb_lower_box = np.array((0,0,0), dtype='uint8')
                        rgb_upper_box = np.array((0,0,255), dtype='uint8')
                        color_space_holder.append(rgb_lower_box)
                        color_space_holder.append(rgb_upper_box)
                        rgb_lower_box = np.array((245,245,245), dtype='uint8')
                        rgb_upper_box = np.array((255,255,255), dtype='uint8')
                        color_space_holder.append(rgb_lower_box)
                        color_space_holder.append(rgb_upper_box)
                        rgb_lower_box = np.array((245,245,245), dtype='uint8')
                        rgb_upper_box = np.array((255,255,255), dtype='uint8')
                        color_space_holder.append(rgb_lower_box)
                        color_space_holder.append(rgb_upper_box)
                        rgb_lower_box = np.array((245,245,245), dtype='uint8')
                        rgb_upper_box = np.array((255,255,255), dtype='uint8')
                        color_space_holder.append(rgb_lower_box)
                        color_space_holder.append(rgb_upper_box)
                        rgb_lower_box = np.array((245,245,245), dtype='uint8')
                        rgb_upper_box = np.array((255,255,255), dtype='uint8')
                        color_space_holder.append(rgb_lower_box)
                        color_space_holder.append(rgb_upper_box)
                        rgb_lower_box = np.array((245,245,245), dtype='uint8')
                        rgb_upper_box = np.array((255,255,255), dtype='uint8')
                        color_space_holder.append(rgb_lower_box)
                        color_space_holder.append(rgb_upper_box)

                        color_avg_holder = np.array((0,0,0), dtype='uint8')
                        color_avg_holder2 = np.array((0,0,0), dtype='uint8')
                    else:
                        color_space_holder = []
                        hsv_lower_box = np.array([int(np.nanquantile(hsv_trimmed[0],.2)),int(np.nanquantile(hsv_trimmed[1],.1)),int(np.nanquantile(hsv_trimmed[2],.1))]) #.2
                        hsv_upper_box = np.array([int(np.nanquantile(hsv_trimmed[0],.8)),int(np.nanquantile(hsv_trimmed[1],.9)),int(np.nanquantile(hsv_trimmed[2],.9))]) #.8
                        color_space_holder.append(hsv_lower_box)
                        color_space_holder.append(hsv_upper_box)
                        hsv_space[int(np.nanquantile(hsv_trimmed[0],.2)): int(np.nanquantile(hsv_trimmed[0],.8))] += poly_counter
                        rgb_lower_box = np.array([int(np.nanquantile(rgb_trimmed[0],.2)),int(np.nanquantile(rgb_trimmed[1],.2)),int(np.nanquantile(rgb_trimmed[2],.2))])
                        rgb_upper_box = np.array([int(np.nanquantile(rgb_trimmed[0],.8)),int(np.nanquantile(rgb_trimmed[1],.8)),int(np.nanquantile(rgb_trimmed[2],.8))])
                        color_space_holder.append(rgb_lower_box)
                        color_space_holder.append(rgb_upper_box)
                        rgb_space[int(np.nanquantile(rgb_trimmed[0],.3)): int(np.nanquantile(rgb_trimmed[0],.7)), int(np.nanquantile(rgb_trimmed[1],.3)): int(np.nanquantile(rgb_trimmed[1],.7)), int(np.nanquantile(rgb_trimmed[2],.3)): int(np.nanquantile(rgb_trimmed[2],.7))] = poly_counter
                        rgb_lower_box = np.array([int(np.nanquantile(rgb_trimmed[0],.1)),int(np.nanquantile(rgb_trimmed[1],.1)),int(np.nanquantile(rgb_trimmed[2],.1))])
                        rgb_upper_box = np.array([int(np.nanquantile(rgb_trimmed[0],.9)),int(np.nanquantile(rgb_trimmed[1],.9)),int(np.nanquantile(rgb_trimmed[2],.9))])
                        color_space_holder.append(rgb_lower_box)
                        color_space_holder.append(rgb_upper_box)
                        rgb_lower_box = np.array([int(np.nanquantile(rgb_trimmed[0],.05)),int(np.nanquantile(rgb_trimmed[1],.05)),int(np.nanquantile(rgb_trimmed[2],.05))])
                        rgb_upper_box = np.array([int(np.nanquantile(rgb_trimmed[0],.95)),int(np.nanquantile(rgb_trimmed[1],.95)),int(np.nanquantile(rgb_trimmed[2],.95))])
                        color_space_holder.append(rgb_lower_box)
                        color_space_holder.append(rgb_upper_box)
                        rgb_lower_box = np.array([int(np.nanquantile(rgb_trimmed[0],.03)),int(np.nanquantile(rgb_trimmed[1],.03)),int(np.nanquantile(rgb_trimmed[2],.03))])
                        rgb_upper_box = np.array([int(np.nanquantile(rgb_trimmed[0],.97)),int(np.nanquantile(rgb_trimmed[1],.97)),int(np.nanquantile(rgb_trimmed[2],.97))])
                        color_space_holder.append(rgb_lower_box)
                        color_space_holder.append(rgb_upper_box)
                        rgb_lower_box = np.array([int(np.nanquantile(rgb_trimmed[0],.02)),int(np.nanquantile(rgb_trimmed[1],.02)),int(np.nanquantile(rgb_trimmed[2],.02))])
                        rgb_upper_box = np.array([int(np.nanquantile(rgb_trimmed[0],.98)),int(np.nanquantile(rgb_trimmed[1],.98)),int(np.nanquantile(rgb_trimmed[2],.98))])
                        color_space_holder.append(rgb_lower_box)
                        color_space_holder.append(rgb_upper_box)
                        rgb_lower_box = np.array([int(np.nanquantile(rgb_trimmed[0],.01)),int(np.nanquantile(rgb_trimmed[1],.01)),int(np.nanquantile(rgb_trimmed[2],.01))])
                        rgb_upper_box = np.array([int(np.nanquantile(rgb_trimmed[0],.99)),int(np.nanquantile(rgb_trimmed[1],.99)),int(np.nanquantile(rgb_trimmed[2],.99))])
                        color_space_holder.append(rgb_lower_box)
                        color_space_holder.append(rgb_upper_box)

                        color_avg_holder = np.array([int(np.nanquantile(rgb_trimmed[0],.5)),int(np.nanquantile(rgb_trimmed[1],.5)),int(np.nanquantile(rgb_trimmed[2],.5))])
                        color_avg_holder2 = np.array([int(np.nanquantile(hsv_trimmed[0],.5)),int(np.nanquantile(hsv_trimmed[1],.5)),int(np.nanquantile(hsv_trimmed[2],.5))])

                    color_space.append(color_space_holder)
                    color_avg.append(color_avg_holder)
                    color_avg2.append(color_avg_holder2)

                    try:
                        color_dif_in_h_space = int(np.nanquantile(hsv_trimmed[0],.8)) - int(np.nanquantile(hsv_trimmed[0],.2))
                    except:
                        color_dif_in_h_space = -1
                    color_dif.append(color_dif_in_h_space)

                    color_key_variety_counting = max(0, np.unique(hsv_trimmed[0]).shape[0]-1)
                    color_key_variety.append(color_key_variety_counting)

                print('time checkpoint _v0:', datetime.now()-runningtime_start)
                running_time_v.append(datetime.now()-runningtime_start)

                ans_category = np.zeros((poly_counter+1, img_rb.shape[0], img_rb.shape[1]), dtype='uint8')

                blank = np.ones((img_rb.shape[0],img_rb.shape[1]),dtype=np.uint8)*255
                ans_category[poly_counter] = np.zeros((img_rb.shape[0],img_rb.shape[1]),dtype=np.uint8)

                #print(legend_name)
                #print(color_space)


            
            def extraction_step0_auxiliary_info(legend, map_name, legend_name, solutiona_dir, print_intermediate_image, all_color_avg, all_color_avg2, color_dif_value, color_key_variety_value, this_poly_value, poly_counter_value, subregion_ratio_value, grayregion_ratio_value, blackregion_ratio_value, color_variety_value, legend_color_range_set):
                rgb_dif = np.zeros((poly_counter_value-1, 3), dtype='uint8')
                hsv_dif = np.zeros((poly_counter_value-1, 3), dtype='uint8')
                #rgb_dif = []
                #hsv_dif = []

                counting_dif = 0
                for counter_legend in range(0, poly_counter_value):
                    if counter_legend == this_poly_value:
                        continue
                    rgb_dif[counting_dif, :] = abs(all_color_avg[counter_legend, :] - all_color_avg[this_poly_value, :])
                    hsv_dif[counting_dif, :] = abs(all_color_avg2[counter_legend, :] - all_color_avg2[this_poly_value, :])
                    counting_dif += 1
                
                #print(rgb_dif)

                #if print_intermediate_image == True:
                    #out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+legend_name[legend]+'_poly_c0_x.png'
                #print(hsv_dif_h_space)

                #print(np.sort(rgb_dif[:, 0]))
                #print(np.sort(rgb_dif[:, 1]))
                #print(np.sort(rgb_dif[:, 2]))

                #print('name')
                concat_name = str(map_name)+'_'+str(legend_name)+'_poly'
                #print(concat_name)

                #print('H space')
                output_values = np.ones((10), dtype='uint8')*180
                if counting_dif > 0:
                    output_values[0: min(10, counting_dif)] = np.sort(hsv_dif[:, 0])[0: min(10,counting_dif)]
                #print(output_values)

                #print('RGB-min')
                output_values2 = np.ones((10), dtype='uint8')*180
                if counting_dif > 0:
                    output_values2[0: min(10, counting_dif)] = np.sort(np.min(rgb_dif, axis=1))[0: min(10,counting_dif)]
                #print(output_values2)

                #print('RGB-dist')
                rgb_dif_distances = np.sqrt(np.sum(rgb_dif**2,axis=1))
                rgb_dif_distances_int = (np.rint(rgb_dif_distances)).astype(int)
                output_values3 = np.ones((10), dtype='uint8')*180
                if counting_dif > 0:
                    output_values3[0: min(10, counting_dif)] = np.sort(rgb_dif_distances_int)[0: min(10,counting_dif)]
                #print(output_values3)

                #print('strict range in h space in this key')
                #print((color_dif_value+1))

                #print('number of distinct colors in this key')
                #print(color_key_variety_value)

                #print('===')
                #print('number of keys (legends) in map (/100.0)')
                #print((poly_counter_value/100.0))

                #print('ratio of subregion')
                #print(subregion_ratio_value)

                #print('ratio of grayregion')
                #print(grayregion_ratio_value)
                
                #print('ratio of blackregion')
                #print(blackregion_ratio_value)

                #print('number of distinct colors (h space) in subregion (/180.0)')
                #print(color_variety_value)

                #print('color range (r/g/b space, h space) of all keys (/255.0 or /180.0)')
                #print(legend_color_range_set)

                if print_intermediate_image == True:
                    if os.path.isfile(solutiona_dir+'intermediate9/'+'auxiliary_info.csv') == False:
                        with open(solutiona_dir+'intermediate9/'+'auxiliary_info.csv','w') as fd:
                            fd.write('Map_name,Key_name')
                            for looping in range(0, 10):
                                fd.write(',Nearest_H_dist('+str(looping)+')')
                            for looping in range(0, 10):
                                fd.write(',Nearest_RGB_min('+str(looping)+')')
                            for looping in range(0, 10):
                                fd.write(',Nearest_RGB_dist('+str(looping)+')')
                            fd.write(',strict_range_H,number_of_H_in_key')
                            fd.write(',number_of_keys,ratio_subregion,ratio_grayregion,ratio_blackregion,number_of_H_in_map,color_range_R_across_keys,color_range_G_across_keys,color_range_B_across_keys,color_range_H_across_keys')
                            fd.write('\n')
                            fd.close()
                    with open(solutiona_dir+'intermediate9/'+'auxiliary_info.csv','a') as fd:
                        fd.write(map_name+','+concat_name)
                        for looping in range(0, 10):
                            fd.write(','+str(output_values[looping]))
                        for looping in range(0, 10):
                            fd.write(','+str(output_values2[looping]))
                        for looping in range(0, 10):
                            fd.write(','+str(output_values3[looping]))
                        fd.write(','+str(color_dif_value+1)+','+str(color_key_variety_value))

                        fd.write(','+str(poly_counter_value/100.0)+','+str(subregion_ratio_value)+','+str(grayregion_ratio_value)+','+str(blackregion_ratio_value)+','+str(color_variety_value))
                        for looping in range(0, 4):
                            fd.write(','+str(legend_color_range_set[looping]))
                        fd.write('\n')
                        fd.close()

                return
            
            

            img_bound00 = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_expected_crop_region.tif')
            img_bound00 = cv2.cvtColor(img_bound00, cv2.COLOR_BGR2GRAY)
            img_bound00[img_bound00 > 0] = 1
            subregion_ratio = (np.sum(img_bound00)) / (img_bound00.shape[0]*img_bound00.shape[1])

            img_crop_gray00 = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop_grayregion.png')
            img_crop_black00 = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop_blackregion.png')
            img_crop_gray00 = cv2.cvtColor(img_crop_gray00, cv2.COLOR_BGR2GRAY)
            img_crop_black00 = cv2.cvtColor(img_crop_black00, cv2.COLOR_BGR2GRAY)
            img_crop_gray00[img_crop_gray00 > 0] = 1
            img_crop_black00[img_crop_black00 > 0] = 1
            grayregion_ratio = (np.sum(img_crop_gray00)) / (np.sum(img_bound00))
            blackregion_ratio = (np.sum(img_crop_black00)) / (np.sum(img_bound00))

            color_variety = np.unique(hsv_rb[:,:,0]).shape[0] / 180.0

            all_color_avg_np = np.array(color_avg)
            all_color_avg_np2 = np.array(color_avg2)

            legend_color_range_rgb_r_space = np.max(all_color_avg_np2, axis=0)[0] - np.min(all_color_avg_np2, axis=0)[0]
            legend_color_range_rgb_g_space = np.max(all_color_avg_np2, axis=0)[1] - np.min(all_color_avg_np2, axis=0)[1]
            legend_color_range_rgb_b_space = np.max(all_color_avg_np2, axis=0)[2] - np.min(all_color_avg_np2, axis=0)[2]
            legend_color_range_hsv_h_space = np.max(all_color_avg_np2, axis=0)[0] - np.min(all_color_avg_np2, axis=0)[0]
            legend_color_range = [legend_color_range_rgb_r_space / 255.0, legend_color_range_rgb_g_space / 255.0, legend_color_range_rgb_b_space / 255.0, legend_color_range_hsv_h_space / 180.0]

            if os.path.isfile(solutiona_dir+'intermediate9/'+'auxiliary_info.csv') == False:
                with open(solutiona_dir+'intermediate9/'+'auxiliary_info.csv','w') as fd:
                    fd.write('Map_name,Key_name')
                    for looping in range(0, 10):
                        fd.write(',Nearest_H_dist('+str(looping)+')')
                    for looping in range(0, 10):
                        fd.write(',Nearest_RGB_min('+str(looping)+')')
                    for looping in range(0, 10):
                        fd.write(',Nearest_RGB_dist('+str(looping)+')')
                    fd.write(',strict_range_H,number_of_H_in_key')
                    fd.write(',number_of_keys,ratio_subregion,ratio_grayregion,ratio_blackregion,number_of_H_in_map,color_range_R_across_keys,color_range_G_across_keys,color_range_B_across_keys,color_range_H_across_keys')
                    fd.write('\n')
                    fd.close()

            print('time checkpoint _v1:', datetime.now()-runningtime_start)
            running_time_v.append(datetime.now()-runningtime_start)

            
            for this_poly in range(0, poly_counter):
                extraction_step0_auxiliary_info(this_poly, map_name, legend_name[this_poly], solutiona_dir, print_intermediate_image, all_color_avg_np, all_color_avg_np2, color_dif[this_poly], color_key_variety[this_poly], this_poly, poly_counter, subregion_ratio, grayregion_ratio, blackregion_ratio, color_variety, legend_color_range)
            
            print('time checkpoint _v2:', datetime.now()-runningtime_start)
            running_time_v.append(datetime.now()-runningtime_start)



        print('time check... worker_auxiliary_info: ', datetime.now()-runningtime_start_global)
    
    else:
        print('No need to extract auxiliary information (extraction already done)...')

    # 7m 52.0s





def worker_recoloring():
    print('')
    print('=== Apply recoloring (color-set matching) for the maps ===')
    runningtime_start_global = datetime.now()


    if preprocessing_recoloring == True:
        print('Applying recoloring...')

        if not os.path.exists(solutiona_dir+str('intermediate8/')):
            os.makedirs(solutiona_dir+str('intermediate8/'))
        
        for target_file_q in range(0, len(candidate_file_name_for_polygon), 1):
            file_name = candidate_file_name_for_polygon[target_file_q]
            running_time_v = []
            
            
            # get the .tif files
            if '.json' in file_name:
                runningtime_start=datetime.now()


                filename=file_name.replace('.json', '.tif')
                print('Working on map:', file_name)
                file_path=os.path.join(data_dir, filename)
                test_json=file_path.replace('.tif', '.json')
                file_name_json = test_json.replace('.json', '.json')
                
                #print(test_json)
                img000 = cv2.imread(file_path)
                #hsv0 = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
                #rgb0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                img_bound = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_expected_crop_region.tif')
                img_bound = cv2.cvtColor(img_bound, cv2.COLOR_BGR2GRAY)

                img_crop_gray = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop_grayregion.png')
                img_crop_black = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop_blackregion.png')
                img_crop_gray = cv2.cvtColor(img_crop_gray, cv2.COLOR_BGR2GRAY)
                img_crop_black = cv2.cvtColor(img_crop_black, cv2.COLOR_BGR2GRAY)


                img_rb = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_remove_black.png')
                img_ms = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_remove_black_mean_shift.png')


                with open(file_name_json) as f:
                    gj = json.load(f)
                json_height = gj['imageHeight']
                json_width = gj['imageWidth']
                rescale_factor_0 = 1.0
                rescale_factor_1 = 1.0



                ## Non-white background
                non_white_background = False
                if np.sum(img_bound) / 255 >= (img_bound.shape[0]*img_bound.shape[1]) * 0.99 or np.unique(img_bound).shape[0] == 1:
                    lower_white = np.array([250,250,250])
                    upper_white = np.array([256,256,256])
                    mask_white_img000 = cv2.inRange(img000, lower_white, upper_white)
                    lower_white = np.array([0,0,0])
                    upper_white = np.array([130,130,130])
                    mask_white_img000_2 = cv2.inRange(img000, lower_white, upper_white)
                    mask_white_img000 = cv2.bitwise_or(mask_white_img000, mask_white_img000_2)

                    corner_avg_white = np.sum(mask_white_img000[int(mask_white_img000.shape[0]*98/100): int(mask_white_img000.shape[0]*99/100), int(mask_white_img000.shape[1]*98/100): int(mask_white_img000.shape[1]*99/100)])/255.0
                    corner_area = (int(mask_white_img000.shape[0]*99/100) - int(mask_white_img000.shape[0]*98/100)) * (int(mask_white_img000.shape[1]*99/100) - int(mask_white_img000.shape[1]*98/100))

                    if corner_avg_white / corner_area < 0.66:
                        non_white_background = True
                        print('non_white_background')



                

                ### Legend is always not considered
                if True:
                    for this_gj in gj['shapes']:
                        #print(this_gj)
                        names = this_gj['label']
                        features = this_gj['points']

                        geoms = np.array(features)
                        y_min = int(np.min(geoms, axis=0)[0]*rescale_factor_1)
                        y_max = int(np.max(geoms, axis=0)[0]*rescale_factor_1)
                        x_min = int(np.min(geoms, axis=0)[1]*rescale_factor_0)
                        x_max = int(np.max(geoms, axis=0)[1]*rescale_factor_0)

                        legend_mask = np.ones((img_rb.shape[0], img_rb.shape[1]), dtype='uint8') *255
                        legend_mask[x_min:x_max, y_min:y_max] = 0
                        img_bound = cv2.bitwise_and(img_bound, legend_mask)
                    img_rb = cv2.bitwise_and(img_rb, img_rb, mask=img_bound)
                    img_ms = cv2.bitwise_and(img_ms, img_ms, mask=img_bound)
                    img_crop_gray = cv2.bitwise_and(img_crop_gray, img_crop_gray, mask=img_bound)
                    img_crop_black = cv2.bitwise_and(img_crop_black, img_crop_black, mask=img_bound)
                hsv_rb = cv2.cvtColor(img_rb, cv2.COLOR_BGR2HSV)
                rgb_rb = cv2.cvtColor(img_rb, cv2.COLOR_BGR2RGB)
                hsv_ms = cv2.cvtColor(img_ms, cv2.COLOR_BGR2HSV)
                rgb_ms = cv2.cvtColor(img_ms, cv2.COLOR_BGR2RGB)



                
                poly_counter = 0
                #color_space = []
                color_avg = []
                color_avg2 = []
                color_set_avg = []
                map_name = file_name.replace('.json', '')
                legend_name = []
                legend_name_check = []
                extracted_legend_name = []


                hsv_space = np.zeros((255), dtype='uint8') # only for h space
                rgb_space = np.zeros((255,255,3), dtype='uint8')


                if not os.path.exists(solutiona_dir+'intermediate7(2)/'+map_name):
                    os.makedirs(solutiona_dir+'intermediate7(2)/'+map_name)



                for this_gj in gj['shapes']:
                    #if '_poly' not in names:
                        #continue
                    #print(this_gj)
                    names = this_gj['label']
                    features = this_gj['points']
                    
                    if '_poly' not in names:
                        continue
                    if names in legend_name_check:
                        continue


                    legend_name_check.append(names)
                    legend_name.append(names.replace('_poly',''))

                    poly_counter = poly_counter+1


                    ### There is no groundtruth for validation data
                    #print('training/'+map_name+'_'+names+'.tif')


                    ### Read json source for the legend
                    geoms = np.array(features)
                    y_min = int(np.min(geoms, axis=0)[0]*rescale_factor_1)
                    y_max = int(np.max(geoms, axis=0)[0]*rescale_factor_1)
                    x_min = int(np.min(geoms, axis=0)[1]*rescale_factor_0)
                    x_max = int(np.max(geoms, axis=0)[1]*rescale_factor_0)

                    img_legend = np.zeros((x_max-x_min, y_max-y_min, 3), dtype='uint8')
                    img_legend = np.copy(img000[x_min:x_max, y_min:y_max, :])
                    
                    if print_intermediate_image == True:
                        out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+names+'_legend.tif'
                        cv2.imwrite(out_file_path0, img_legend)
                    
                    
                    img_legend = cv2.cvtColor(img_legend, cv2.COLOR_BGR2RGB)
                    img_legend = img_legend[int(img_legend.shape[0]/8):int(img_legend.shape[0]*7/8), int(img_legend.shape[1]/8):int(img_legend.shape[1]*7/8), :]
                    hsv_legend = cv2.cvtColor(img_legend, cv2.COLOR_RGB2HSV)
                    black_threshold = 30 #130
                    white_threshold = 250 #245

                    lower_black_rgb_trimmed0 = np.array([0,0,0])
                    upper_black_rgb_trimmed0 = np.array([130,130,130])
                    mask_test_img_legend = cv2.inRange(img_legend, lower_black_rgb_trimmed0, upper_black_rgb_trimmed0)
                    if np.sum(mask_test_img_legend == 255) > np.sum(img_legend > 0) * 0.25:
                        black_threshold = 30
                    
                    rgb_trimmed = np.zeros((img_legend.shape[2], img_legend.shape[0], img_legend.shape[1]), dtype='uint8')
                    hsv_trimmed = np.zeros((img_legend.shape[2], img_legend.shape[0], img_legend.shape[1]), dtype='uint8')
                    rgb_trimmed = rgb_trimmed.astype(float)
                    hsv_trimmed = hsv_trimmed.astype(float)
                    for dimension in range(0, 3):
                        rgb_trimmed[dimension] = np.copy(img_legend[:,:,dimension]).astype(float)
                        hsv_trimmed[dimension] = np.copy(hsv_legend[:,:,dimension]).astype(float)

                    rgb_trimmed_temp = np.copy(rgb_trimmed)
                    rgb_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
                    hsv_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
                    rgb_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
                    hsv_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
                    rgb_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
                    hsv_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan

                    rgb_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
                    hsv_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
                    rgb_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
                    hsv_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
                    rgb_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
                    hsv_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan

                    #color_preset = [.2, .3, .35, .4, .45, .5, .55, .6, .65, .7, .8]
                    color_preset = [.3, .4, .45, .5, .55, .6, .7]

                    if np.sum(np.isnan(hsv_trimmed)) >= (hsv_trimmed.shape[0]*hsv_trimmed.shape[1]*hsv_trimmed.shape[2]):
                        color_space_holder2 = []
                        for color_preseted in color_preset:
                            color_avg_holder = np.array([int(np.nanquantile(rgb_trimmed_temp[0], color_preseted)),int(np.nanquantile(rgb_trimmed_temp[1], color_preseted)),int(np.nanquantile(rgb_trimmed_temp[2], color_preseted))])
                            color_space_holder2.append(color_avg_holder)

                        color_avg_holder = np.array((0,0,0), dtype='uint8')
                        color_avg_holder2 = np.array((0,0,0), dtype='uint8')
                    else:
                        color_space_holder2 = []
                        for color_preseted in color_preset:
                            color_avg_holder = np.array([int(np.nanquantile(rgb_trimmed[0], color_preseted)),int(np.nanquantile(rgb_trimmed[1], color_preseted)),int(np.nanquantile(rgb_trimmed[2], color_preseted))])
                            color_space_holder2.append(color_avg_holder)
                        
                        color_avg_holder = np.array([int(np.nanquantile(rgb_trimmed[0],.5)),int(np.nanquantile(rgb_trimmed[1],.5)),int(np.nanquantile(rgb_trimmed[2],.5))])
                        color_avg_holder2 = np.array([int(np.nanquantile(hsv_trimmed[0],.5)),int(np.nanquantile(hsv_trimmed[1],.5)),int(np.nanquantile(hsv_trimmed[2],.5))])

                    #color_avg.append(color_avg_holder)
                    color_set_avg.append(color_space_holder2)
                    color_avg2.append(color_avg_holder2)

                print('time checkpoint _v0:', datetime.now()-runningtime_start)
                running_time_v.append(datetime.now()-runningtime_start)

                ans_category = np.zeros((poly_counter+1, img_rb.shape[0], img_rb.shape[1]), dtype='uint8')

                blank = np.ones((img_rb.shape[0],img_rb.shape[1]),dtype=np.uint8)*255
                ans_category[poly_counter] = np.zeros((img_rb.shape[0],img_rb.shape[1]),dtype=np.uint8)

                #print(legend_name)
                #print(color_space)


                mapping_color_to_color_set = []
                for color_set_id in range(0, len(color_set_avg)):
                    for color_id in range(0, len(color_set_avg[color_set_id])):
                        color_avg.append(color_set_avg[color_set_id][color_id])
                        mapping_color_to_color_set.append(color_set_id)


                
                

                total_color_set = len(color_set_avg)

                # add contour/ background color
                color_avg.append(np.array([0,0,0])) # black (contour, text)
                color_avg2.append(np.array([0,0,0]))
                mapping_color_to_color_set.append(total_color_set)
                total_color_set += 1

                color_avg.append(np.array([255,255,255])) # white (background)
                color_avg2.append(np.array([255,255,255]))
                mapping_color_to_color_set.append(total_color_set)
                total_color_set += 1

                check_ocean = False
                ### check if an ocean-like color is already included
                np_color_avg = np.array(color_avg)
                ocean_cand = []
                ocean_cand.append([218, 240, 254]) # ocean
                np_ocean_cand = np.array(ocean_cand)
                #print(np.isin(np_ocean_cand, np_color_avg).all(-1).any(-1))
                #print((abs(np.subtract(np_ocean_cand, np_color_avg))))
                #print(np_color_avg.shape[0])

                check_ocean = ((abs(np.subtract(np_ocean_cand, np_color_avg))) < (max(5, 15-int(np_color_avg.shape[0]/10)))).all(-1).any(-1)
                #print(((abs(np.subtract(np_ocean_cand, np_color_avg))) < (max(5, 10-int(np_color_avg.shape[0]/10)))).all(-1).any(-1))
                if check_ocean == False:
                    color_avg.append(np.array([218,240,254])) # blue (ocean)
                    color_avg2.append(np.array([218,240,254]))
                    mapping_color_to_color_set.append(total_color_set)
                    total_color_set += 1
                    print('Add ocean color...')
                else:
                    print('Ocean-like color is already included...')

                print('total # of colors:', len(color_avg))
                print('total # of keys:', total_color_set)
                #print(color_avg)
                #print(mapping_color_to_color_set)


                temp_mapping = np.array(mapping_color_to_color_set)
                mapping_color_set_to_color_prob = np.empty(shape=(total_color_set, len(color_avg)))
                mapping_color_set_to_color_prob.fill(0.0)

                for set_id in range(0, total_color_set):
                    targeted_index = np.where(temp_mapping == set_id)
                    #print(targeted_index[0])
                    #print(targeted_index[0].shape[0])

                    for targeted_id in targeted_index[0]:
                        mapping_color_set_to_color_prob[set_id][targeted_id] = 1/targeted_index[0].shape[0]

                #print(mapping_color_set_to_color_prob.shape)
                #print(mapping_color_set_to_color_prob)

                mapping_color_set_to_color_prob_tp = np.copy(mapping_color_set_to_color_prob)
                mapping_color_set_to_color_prob_tp = np.transpose(mapping_color_set_to_color_prob_tp)


                # If there is color shift for all legends, but this section is never used for the final solution.
                minimal_grid_size = 1000
                distance_kernel = np.ones((5,5)) / 25.0
                distance_kernel = distance_kernel[:, :, None]

                smoothing_map_experimental = True
                if smoothing_map_experimental == True:

                    img_bound_argwhere = np.argwhere(img_bound)
                    (ystart, xstart), (ystop, xstop) = img_bound_argwhere.min(0), img_bound_argwhere.max(0) + 1

                    gridize_processing = True
                    y_shape = ystop-ystart
                    x_shape = xstop-xstart
                    if gridize_processing == True and y_shape > minimal_grid_size and x_shape > minimal_grid_size:
                        repaste_image = np.zeros((img_bound.shape[0], img_bound.shape[1], 3), dtype='uint8')
                        repaste_index = np.zeros((img_bound.shape[0], img_bound.shape[1], 1), dtype='uint8')
                        grid_counting = math.ceil(y_shape/minimal_grid_size) * math.ceil(x_shape/minimal_grid_size)
                        grid_completed = 0

                        for r in range(0, math.ceil(y_shape/minimal_grid_size)):
                            for c in range(0, math.ceil(x_shape/minimal_grid_size)):
                                r_0 = ystart + r*minimal_grid_size
                                r_1 = ystart + min(r*minimal_grid_size+minimal_grid_size, y_shape)
                                c_0 = xstart + c*minimal_grid_size
                                c_1 = xstart + min(c*minimal_grid_size+minimal_grid_size, x_shape)
                                #print(r, c, r_0, r_1, c_0, c_1)

                                ### only process a small part of the whole subregion
                                #im = np.copy(rgb_rb[ystart:ystop, xstart:xstop, :])
                                im = np.copy(rgb_rb[r_0:r_1, c_0:c_1, :])
                                image = im.reshape(im.shape[0],im.shape[1],1,3)

                                # Create color container 
                                colors_container = np.ones(shape=[image.shape[0],image.shape[1],len(color_avg),3])
                                for i,color in enumerate(color_avg):
                                    colors_container[:,:,i,:] = color
                                colors_container2 = np.ones(shape=[image.shape[0],image.shape[1],len(color_avg2),3])
                                for i,color in enumerate(color_avg2):
                                    colors_container2[:,:,i,:] = color
                                
                                rgb_weight = np.ones(shape=[image.shape[0],image.shape[1],1,3])
                                rgb_weight[:,:,:,0] = 1 # 2
                                rgb_weight[:,:,:,1] = 1 # 4
                                rgb_weight[:,:,:,2] = 1 # 3

                                background_correction_direct_rgb = np.ones(shape=[image.shape[0],image.shape[1],1,3])
                                background_correction_direct_rgb[:,:,:,0] = 1.0
                                background_correction_direct_rgb[:,:,:,1] = 1.0
                                background_correction_direct_rgb[:,:,:,2] = 1.0

                                image_deviation = np.zeros(shape=[image.shape[0],image.shape[1],1,3])
                                image_deviation[:,:,:,0] = image[:,:,:,0] - image[:,:,:,1]
                                image_deviation[:,:,:,1] = image[:,:,:,0] - image[:,:,:,2]
                                image_deviation[:,:,:,2] = image[:,:,:,1] - image[:,:,:,2]

                                legend_deviation = np.zeros(shape=[image.shape[0],image.shape[1],len(color_avg),3])
                                legend_deviation[:,:,:,0] = colors_container[:,:,:,0] - colors_container[:,:,:,1]
                                legend_deviation[:,:,:,1] = colors_container[:,:,:,0] - colors_container[:,:,:,2]
                                legend_deviation[:,:,:,2] = colors_container[:,:,:,1] - colors_container[:,:,:,2]
                                
                                background_correction_deviated_rgb = np.ones(shape=[image.shape[0],image.shape[1],1,3])
                                background_correction_deviated_rgb[:,:,:,:] = 0.5 + 0.5*(1.0-abs(image_deviation[:,:,:,:])/255.0)


                                def closest(image,color_container):
                                    shape = image.shape[:2]
                                    total_shape = shape[0]*shape[1]

                                    # calculate distances
                                    distances_0 = np.sqrt(np.sum(rgb_weight*((color_container*background_correction_direct_rgb-image)**2),axis=3))
                                    distances_1 = np.sqrt(np.sum(((legend_deviation*background_correction_deviated_rgb-image_deviation)**2),axis=3))
                                    distances = distances_0*0.95 + distances_1*0.05

                                    # in the 1st version, the distance is the distance to the color of each key
                                    # in the 2nd version, the distance is the distance to the color under the color set of each key

                                    #print(distances.shape) # shape: (1500, 1500, # of colors)
                                    #print(mapping_color_set_to_color_prob_tp.shape) # shape: (# of colors, # of keys)

                                    multiplied_distances = np.dot(distances, mapping_color_set_to_color_prob_tp)

                                    #print(multiplied_distances.shape) # shape: (1500, 1500, # of keys)
                                    
                                    conv_distances = scipy.ndimage.convolve(multiplied_distances, distance_kernel)


                                    min_index_map = np.argmin(conv_distances, axis=2)
                                    min_index = min_index_map.reshape(-1)
                                    natural_index = np.arange(total_shape)

                                    reshaped_container = colors_container2.reshape(-1,len(color_avg2),3) # only use one color to re-color the map

                                    color_view = reshaped_container[natural_index, min_index].reshape(shape[0], shape[1], 3)
                                    return color_view, min_index_map
                                
                                
                                result_image, min_index_map = closest(image, colors_container)
                                result_image = result_image.astype(np.uint8)
                                min_index_map = min_index_map.astype(np.uint8)

                                grid_completed += 1
                                print('processing _v0 >>> _v1 (finding the closest color)... (grid completed '+str(grid_completed)+'/'+str(grid_counting)+')... :', datetime.now()-runningtime_start)

                                #plt.imshow(result_image)
                                #plt.show()

                                #Image.fromarray(result_image.astype(np.uint8)).show()


                                #subtract_rgb = []

                                #repaste_image[ystart:ystop, xstart:xstop, :] = np.copy(result_image[:, :, :])
                                repaste_image[r_0:r_1, c_0:c_1, :] = np.copy(result_image[:, :, :])
                                repaste_index[r_0:r_1, c_0:c_1, 0] = np.copy(min_index_map[:, :])
                    else:
                        im = np.copy(rgb_rb[ystart:ystop, xstart:xstop, :])
                        image = im.reshape(im.shape[0],im.shape[1],1,3)

                        # Create color container 
                        colors_container = np.ones(shape=[image.shape[0],image.shape[1],len(color_avg),3])
                        for i,color in enumerate(color_avg):
                            colors_container[:,:,i,:] = color
                        colors_container2 = np.ones(shape=[image.shape[0],image.shape[1],len(color_avg2),3])
                        for i,color in enumerate(color_avg2):
                            colors_container2[:,:,i,:] = color
                        
                        rgb_weight = np.ones(shape=[image.shape[0],image.shape[1],1,3])
                        rgb_weight[:,:,:,0] = 1 # 2
                        rgb_weight[:,:,:,1] = 1 # 4
                        rgb_weight[:,:,:,2] = 1 # 3

                        background_correction_direct_rgb = np.ones(shape=[image.shape[0],image.shape[1],1,3])
                        background_correction_direct_rgb[:,:,:,0] = 1.0
                        background_correction_direct_rgb[:,:,:,1] = 1.0
                        background_correction_direct_rgb[:,:,:,2] = 1.0

                        image_deviation = np.zeros(shape=[image.shape[0],image.shape[1],1,3])
                        image_deviation[:,:,:,0] = image[:,:,:,0] - image[:,:,:,1]
                        image_deviation[:,:,:,1] = image[:,:,:,0] - image[:,:,:,2]
                        image_deviation[:,:,:,2] = image[:,:,:,1] - image[:,:,:,2]

                        legend_deviation = np.zeros(shape=[image.shape[0],image.shape[1],len(color_avg),3])
                        legend_deviation[:,:,:,0] = colors_container[:,:,:,0] - colors_container[:,:,:,1]
                        legend_deviation[:,:,:,1] = colors_container[:,:,:,0] - colors_container[:,:,:,2]
                        legend_deviation[:,:,:,2] = colors_container[:,:,:,1] - colors_container[:,:,:,2]
                        
                        background_correction_deviated_rgb = np.ones(shape=[image.shape[0],image.shape[1],1,3])
                        background_correction_deviated_rgb[:,:,:,:] = 0.5 + 0.5*(1.0-abs(image_deviation[:,:,:,:])/255.0)

                        print('processing _v0 >>> _v1 (finding the closest color)... :', datetime.now()-runningtime_start)


                        def closest(image,color_container):
                            shape = image.shape[:2]
                            total_shape = shape[0]*shape[1]

                            # calculate distances
                            distances_0 = np.sqrt(np.sum(rgb_weight*((color_container*background_correction_direct_rgb-image)**2),axis=3))
                            distances_1 = np.sqrt(np.sum(((legend_deviation*background_correction_deviated_rgb-image_deviation)**2),axis=3))
                            distances = distances_0*0.95 + distances_1*0.05

                            # in the 1st version, the distance is the distance to the color of each key
                            # in the 2nd version, the distance is the distance to the color under the color set of each key

                            #print(distances.shape) # shape: (1000, 1000, # of colors)
                            #print(mapping_color_set_to_color_prob_tp.shape) # shape: (# of colors, # of keys)

                            multiplied_distances = np.dot(distances, mapping_color_set_to_color_prob_tp)

                            conv_distances = scipy.ndimage.convolve(multiplied_distances, distance_kernel)

                            min_index_map = np.argmin(conv_distances, axis=2)
                            min_index = min_index_map.reshape(-1)
                            natural_index = np.arange(total_shape)

                            reshaped_container = colors_container2.reshape(-1,len(color_avg2),3) # only use one color to re-color the map

                            color_view = reshaped_container[natural_index, min_index].reshape(shape[0], shape[1], 3)
                            return color_view, min_index_map
                        
                        result_image, min_index_map = closest(image, colors_container)
                        result_image = result_image.astype(np.uint8)
                        min_index_map = min_index_map.astype(np.uint8)

                        #plt.imshow(result_image)
                        #plt.show()

                        #Image.fromarray(result_image.astype(np.uint8)).show()


                        #subtract_rgb = []

                        repaste_image = np.zeros((img_bound.shape[0], img_bound.shape[1], 3), dtype='uint8')
                        repaste_image[ystart:ystop, xstart:xstop, :] = np.copy(result_image[:, :, :])

                        repaste_index = np.zeros((img_bound.shape[0], img_bound.shape[1], 1), dtype='uint8')
                        repaste_index[ystart:ystop, xstart:xstop, 0] = np.copy(min_index_map[:, :])

                            
                    
                    # print('processing _v1 >>> _v2 (legend '+str(legend+1)+'/'+str(poly_counter)+')... :', datetime.now()-runningtime_start)


                    #result_image0 = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                    #out_file_path0=solutiona_dir+'intermediate8/'+map_name+'_polygon_recoloring_attempt_0.png'
                    #cv2.imwrite(out_file_path0, result_image0)

                    repaste_image = cv2.cvtColor(repaste_image, cv2.COLOR_RGB2BGR)
                    out_file_path0=solutiona_dir+'intermediate8/'+map_name+'_polygon_recoloring_attempt_1.png'
                    cv2.imwrite(out_file_path0, repaste_image)

                    out_file_path0=solutiona_dir+'intermediate8/'+map_name+'_polygon_recoloring_index_1.png'
                    cv2.imwrite(out_file_path0, repaste_index)

                    repaste_image = cv2.cvtColor(repaste_image, cv2.COLOR_BGR2RGB)
                    repaste_image_v2 = np.copy(repaste_image)
                    if check_ocean == False:
                        # if we manually add the ocean color
                        ocean_mask = cv2.inRange(repaste_image, np.array([218, 240, 254]), np.array([218, 240, 254]))
                        repaste_image_v2 = cv2.bitwise_and(repaste_image, repaste_image, mask=(255-ocean_mask))
                        #repaste_image_v2[targeted_image == np.array([218, 240, 254])] = np.array([0,0,0])
                    
                    repaste_image_v2 = cv2.cvtColor(repaste_image_v2, cv2.COLOR_RGB2BGR)
                    out_file_path0=solutiona_dir+'intermediate8/'+map_name+'_polygon_recoloring_attempt_2.png'
                    cv2.imwrite(out_file_path0, repaste_image_v2)

                    
                    #targeted_image = cv2.imread(out_file_path0)
                    #color_thief = ColorThief(out_file_path0)
                    
                    # get the dominant color
                    #dominant_color = color_thief.get_color(quality=1)
                    #palette = color_thief.get_palette(color_count=10)

                    running_time_v.append(datetime.now()-runningtime_start)

                    #print(dominant_color)
                    #print(palette)
                    if os.path.isfile(solutiona_dir+'intermediate8/'+'running_time_record_v3.csv') == False:
                        with open(solutiona_dir+'intermediate8/'+'running_time_record_v3.csv','w') as fd:
                            fd.write('File,checkpoint_0,checkpoint_1\n')
                            fd.close()
                    with open(solutiona_dir+'intermediate8/'+'running_time_record_v3.csv','a') as fd:
                        fd.write(map_name+',')
                        for rtc in range(0, len(running_time_v)):
                            fd.write(str(running_time_v[rtc])+',')
                        fd.write('\n')
                        fd.close()



    else:
        print('Not applying recoloring...')

    # 603m 18.7s


    if preprocessing_recoloring == True:
        print('Applying recoloring...')

        #data_dir='validation'
        if not os.path.exists(solutiona_dir+str('intermediate8(2)/')):
            os.makedirs(solutiona_dir+str('intermediate8(2)/'))
        
        for target_file_q in range(0, len(candidate_file_name_for_polygon), 1):
        #for target_file_q in range(len(candidate_file_name_for_polygon)-1, 0, -1):
        #for target_file_q in range(4, 5, 1):
            file_name = candidate_file_name_for_polygon[target_file_q]
            running_time_v = []
            
            
            # get the .tif files
            if '.json' in file_name:
                runningtime_start=datetime.now()


                filename=file_name.replace('.json', '.tif')
                print('Working on map:', file_name)
                file_path=os.path.join(data_dir, filename)
                test_json=file_path.replace('.tif', '.json')
                file_name_json = test_json.replace('.json', '.json')
                
                #print(test_json)
                img000 = cv2.imread(file_path)
                #hsv0 = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
                #rgb0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                img_bound = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_expected_crop_region.tif')
                img_bound = cv2.cvtColor(img_bound, cv2.COLOR_BGR2GRAY)

                img_crop_gray = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop_grayregion.png')
                img_crop_black = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop_blackregion.png')
                img_crop_gray = cv2.cvtColor(img_crop_gray, cv2.COLOR_BGR2GRAY)
                img_crop_black = cv2.cvtColor(img_crop_black, cv2.COLOR_BGR2GRAY)


                img_rb = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_remove_black.png')
                img_ms = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_remove_black_mean_shift.png')


                with open(file_name_json) as f:
                    gj = json.load(f)
                json_height = gj['imageHeight']
                json_width = gj['imageWidth']
                rescale_factor_0 = 1.0
                rescale_factor_1 = 1.0



                ## Non-white background
                non_white_background = False
                if np.sum(img_bound) / 255 >= (img_bound.shape[0]*img_bound.shape[1]) * 0.99 or np.unique(img_bound).shape[0] == 1:
                    lower_white = np.array([250,250,250])
                    upper_white = np.array([256,256,256])
                    mask_white_img000 = cv2.inRange(img000, lower_white, upper_white)
                    lower_white = np.array([0,0,0])
                    upper_white = np.array([130,130,130])
                    mask_white_img000_2 = cv2.inRange(img000, lower_white, upper_white)
                    mask_white_img000 = cv2.bitwise_or(mask_white_img000, mask_white_img000_2)

                    corner_avg_white = np.sum(mask_white_img000[int(mask_white_img000.shape[0]*98/100): int(mask_white_img000.shape[0]*99/100), int(mask_white_img000.shape[1]*98/100): int(mask_white_img000.shape[1]*99/100)])/255.0
                    corner_area = (int(mask_white_img000.shape[0]*99/100) - int(mask_white_img000.shape[0]*98/100)) * (int(mask_white_img000.shape[1]*99/100) - int(mask_white_img000.shape[1]*98/100))

                    if corner_avg_white / corner_area < 0.66:
                        non_white_background = True
                        print('non_white_background')



                

                ### Legend is always not considered
                if True:
                    for this_gj in gj['shapes']:
                        #print(this_gj)
                        names = this_gj['label']
                        features = this_gj['points']

                        geoms = np.array(features)
                        y_min = int(np.min(geoms, axis=0)[0]*rescale_factor_1)
                        y_max = int(np.max(geoms, axis=0)[0]*rescale_factor_1)
                        x_min = int(np.min(geoms, axis=0)[1]*rescale_factor_0)
                        x_max = int(np.max(geoms, axis=0)[1]*rescale_factor_0)

                        legend_mask = np.ones((img_rb.shape[0], img_rb.shape[1]), dtype='uint8') *255
                        legend_mask[x_min:x_max, y_min:y_max] = 0
                        img_bound = cv2.bitwise_and(img_bound, legend_mask)
                    img_rb = cv2.bitwise_and(img_rb, img_rb, mask=img_bound)
                    img_ms = cv2.bitwise_and(img_ms, img_ms, mask=img_bound)
                    img_crop_gray = cv2.bitwise_and(img_crop_gray, img_crop_gray, mask=img_bound)
                    img_crop_black = cv2.bitwise_and(img_crop_black, img_crop_black, mask=img_bound)
                hsv_rb = cv2.cvtColor(img_rb, cv2.COLOR_BGR2HSV)
                rgb_rb = cv2.cvtColor(img_rb, cv2.COLOR_BGR2RGB)
                hsv_ms = cv2.cvtColor(img_ms, cv2.COLOR_BGR2HSV)
                rgb_ms = cv2.cvtColor(img_ms, cv2.COLOR_BGR2RGB)



                
                poly_counter = 0
                #color_space = []
                color_avg = []
                color_avg2 = []
                color_set_avg = []
                map_name = file_name.replace('.json', '')
                legend_name = []
                legend_name_check = []
                extracted_legend_name = []


                hsv_space = np.zeros((255), dtype='uint8') # only for h space
                rgb_space = np.zeros((255,255,3), dtype='uint8')


                if not os.path.exists(solutiona_dir+str('intermediate8(2)/')+str(map_name)+'/'):
                    os.makedirs(solutiona_dir+str('intermediate8(2)/')+str(map_name)+'/')


                for this_gj in gj['shapes']:
                    #if '_poly' not in names:
                        #continue
                    #print(this_gj)
                    names = this_gj['label']
                    features = this_gj['points']
                    
                    if '_poly' not in names:
                        continue
                    if names in legend_name_check:
                        continue


                    legend_name_check.append(names)
                    legend_name.append(names.replace('_poly',''))

                    poly_counter = poly_counter+1



                #print('time checkpoint _v0:', datetime.now()-runningtime_start)
                #running_time_v.append(datetime.now()-runningtime_start)

                ans_category = np.zeros((poly_counter+1, img_rb.shape[0], img_rb.shape[1]), dtype='uint8')

                blank = np.ones((img_rb.shape[0],img_rb.shape[1]),dtype=np.uint8)*255
                ans_category[poly_counter] = np.zeros((img_rb.shape[0],img_rb.shape[1]),dtype=np.uint8)

                #print(legend_name)
                #print(color_space)

                bgr_image = cv2.imread(solutiona_dir+'intermediate8/'+map_name+'_polygon_recoloring_index_1.png')
                #print(bgr_image[:,:,0].shape)

                temp_rgb_recoloring = bgr_image[:,:,0]
                temp_rgb_recoloring = temp_rgb_recoloring.flatten()
                #temp_rgb_recoloring2 = np.zeros(shape=(temp_rgb_recoloring.shape[0], 3),dtype=np.uint8)
                #print(temp_rgb_recoloring.shape)
                #print(temp_rgb_recoloring2.shape)

                for color_index in range(0, poly_counter):
                    #temp_rgb_recoloring2[temp_rgb_recoloring == color_index] = color_avg2[color_index]

                    temp_rgb_recoloring3 = np.zeros(shape=(temp_rgb_recoloring.shape[0], 1),dtype=np.uint8)
                    temp_rgb_recoloring3[temp_rgb_recoloring == color_index] = 255

                    temp_rgb_recoloring3 = np.reshape(temp_rgb_recoloring3, (-1, bgr_image.shape[1], 1))
                    temp_rgb_recoloring3 = cv2.bitwise_and(temp_rgb_recoloring3, img_bound)
                    out_file_path0 = os.path.join(solutiona_dir+'intermediate8(2)', map_name, map_name+'_'+legend_name[color_index]+'_poly_rc_v0.png')
                    cv2.imwrite(out_file_path0, temp_rgb_recoloring3)


                #temp_rgb_recoloring2 = np.reshape(temp_rgb_recoloring2, (-1, bgr_image.shape[1], 3))
                
                #print(temp_rgb_recoloring.shape)
                #temp_rgb_recoloring2 = cv2.cvtColor(temp_rgb_recoloring2, cv2.COLOR_RGB2BGR)


        print('time check... worker_recoloring: ', datetime.now()-runningtime_start_global)
    
    else:
        print('No need to apply recoloring (extraction already done)...')
    # 12m 57.8s







def worker_main_component():
    print('')
    print('=== Apply general metadata preprocessing for polygon extraction from the maps ===')
    runningtime_start_global = datetime.now()


    #data_dir='validation'
    if not os.path.exists(solutiona_dir+str('intermediate7(2)/')):
        os.makedirs(solutiona_dir+str('intermediate7(2)/'))
    if not os.path.exists(solutiona_dir+str('intermediate7(2)/Output/')):
        os.makedirs(solutiona_dir+str('intermediate7(2)/Output/'))

    for target_file_q in range(0, len(candidate_file_name_for_polygon), 1):
        file_name = candidate_file_name_for_polygon[target_file_q]
        running_time_v = []
        
        # get the .tif files
        if '.json' in file_name:
            runningtime_start=datetime.now()


            filename=file_name.replace('.json', '.tif')
            print('Working on map:', file_name)
            file_path=os.path.join(data_dir, filename)
            test_json=file_path.replace('.tif', '.json')
            file_name_json = test_json.replace('.json', '.json')
            
            #print(test_json)
            img000 = cv2.imread(file_path)
            #hsv0 = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
            #rgb0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            img_bound = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_expected_crop_region.tif')
            img_bound = cv2.cvtColor(img_bound, cv2.COLOR_BGR2GRAY)

            img_crop_gray = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop_grayregion.png')
            img_crop_black = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop_blackregion.png')
            img_crop_gray = cv2.cvtColor(img_crop_gray, cv2.COLOR_BGR2GRAY)
            img_crop_black = cv2.cvtColor(img_crop_black, cv2.COLOR_BGR2GRAY)

            #img_boundary = cv2.imread(solutiona_dir+'intermediate5/Extraction/'+file_name.replace('.json', '')+'_overall_boundary.png')
            img_boundary = cv2.imread(solutiona_dir+'intermediate5/Extraction(3)/'+file_name.replace('.json', '')+'_overall_boundary_candidate.png')
            img_boundary = cv2.cvtColor(img_boundary, cv2.COLOR_BGR2GRAY)

            img_rb = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_remove_black.png')
            img_ms = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_remove_black_mean_shift.png')


            with open(file_name_json) as f:
                gj = json.load(f)
            json_height = gj['imageHeight']
            json_width = gj['imageWidth']
            rescale_factor_0 = 1.0
            rescale_factor_1 = 1.0



            ## Non-white background
            non_white_background = False
            if np.sum(img_bound) / 255 >= (img_bound.shape[0]*img_bound.shape[1]) * 0.99 or np.unique(img_bound).shape[0] == 1:
                lower_white = np.array([250,250,250])
                upper_white = np.array([256,256,256])
                mask_white_img000 = cv2.inRange(img000, lower_white, upper_white)
                lower_white = np.array([0,0,0])
                upper_white = np.array([130,130,130])
                mask_white_img000_2 = cv2.inRange(img000, lower_white, upper_white)
                mask_white_img000 = cv2.bitwise_or(mask_white_img000, mask_white_img000_2)

                corner_avg_white = np.sum(mask_white_img000[int(mask_white_img000.shape[0]*98/100): int(mask_white_img000.shape[0]*99/100), int(mask_white_img000.shape[1]*98/100): int(mask_white_img000.shape[1]*99/100)])/255.0
                corner_area = (int(mask_white_img000.shape[0]*99/100) - int(mask_white_img000.shape[0]*98/100)) * (int(mask_white_img000.shape[1]*99/100) - int(mask_white_img000.shape[1]*98/100))

                if corner_avg_white / corner_area < 0.66:
                    non_white_background = True
                    print('non_white_background')



            

            ### Legend is always not considered
            if True:
                for this_gj in gj['shapes']:
                    #print(this_gj)
                    names = this_gj['label']
                    features = this_gj['points']

                    geoms = np.array(features)
                    y_min = int(np.min(geoms, axis=0)[0]*rescale_factor_1)
                    y_max = int(np.max(geoms, axis=0)[0]*rescale_factor_1)
                    x_min = int(np.min(geoms, axis=0)[1]*rescale_factor_0)
                    x_max = int(np.max(geoms, axis=0)[1]*rescale_factor_0)

                    legend_mask = np.ones((img_rb.shape[0], img_rb.shape[1]), dtype='uint8') *255
                    legend_mask[x_min:x_max, y_min:y_max] = 0
                    img_bound = cv2.bitwise_and(img_bound, legend_mask)
                img_rb = cv2.bitwise_and(img_rb, img_rb, mask=img_bound)
                img_ms = cv2.bitwise_and(img_ms, img_ms, mask=img_bound)
                img_crop_gray = cv2.bitwise_and(img_crop_gray, img_crop_gray, mask=img_bound)
                img_crop_black = cv2.bitwise_and(img_crop_black, img_crop_black, mask=img_bound)
            hsv_rb = cv2.cvtColor(img_rb, cv2.COLOR_BGR2HSV)
            rgb_rb = cv2.cvtColor(img_rb, cv2.COLOR_BGR2RGB)
            hsv_ms = cv2.cvtColor(img_ms, cv2.COLOR_BGR2HSV)
            rgb_ms = cv2.cvtColor(img_ms, cv2.COLOR_BGR2RGB)



            
            poly_counter = 0
            color_space = []
            color_avg = []
            color_avg2 = []
            map_name = file_name.replace('.json', '')
            legend_name = []
            legend_name_check = []
            extracted_legend_name = []


            hsv_space = np.zeros((255), dtype='uint8') # only for h space
            rgb_space = np.zeros((255,255,3), dtype='uint8')


            if not os.path.exists(solutiona_dir+'intermediate7(2)/'+map_name):
                os.makedirs(solutiona_dir+'intermediate7(2)/'+map_name)



            for this_gj in gj['shapes']:
                #if '_poly' not in names:
                    #continue
                #print(this_gj)
                names = this_gj['label']
                features = this_gj['points']
                
                if '_poly' not in names:
                    continue
                if names in legend_name_check:
                    continue


                legend_name_check.append(names)
                legend_name.append(names.replace('_poly',''))

                poly_counter = poly_counter+1


                ### There is no groundtruth for validation data
                #print('training/'+map_name+'_'+names+'.tif')


                ### Read json source for the legend
                geoms = np.array(features)
                y_min = int(np.min(geoms, axis=0)[0]*rescale_factor_1)
                y_max = int(np.max(geoms, axis=0)[0]*rescale_factor_1)
                x_min = int(np.min(geoms, axis=0)[1]*rescale_factor_0)
                x_max = int(np.max(geoms, axis=0)[1]*rescale_factor_0)

                img_legend = np.zeros((x_max-x_min, y_max-y_min, 3), dtype='uint8')
                img_legend = np.copy(img000[x_min:x_max, y_min:y_max, :])
                
                if print_intermediate_image == True:
                    out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_'+names+'_legend.tif'
                    cv2.imwrite(out_file_path0, img_legend)
                
                
                img_legend = cv2.cvtColor(img_legend, cv2.COLOR_BGR2RGB)
                img_legend = img_legend[int(img_legend.shape[0]/8):int(img_legend.shape[0]*7/8), int(img_legend.shape[1]/8):int(img_legend.shape[1]*7/8), :]
                hsv_legend = cv2.cvtColor(img_legend, cv2.COLOR_RGB2HSV)
                black_threshold = 30 #130
                white_threshold = 250 #245

                lower_black_rgb_trimmed0 = np.array([0,0,0])
                upper_black_rgb_trimmed0 = np.array([130,130,130])
                mask_test_img_legend = cv2.inRange(img_legend, lower_black_rgb_trimmed0, upper_black_rgb_trimmed0)
                if np.sum(mask_test_img_legend == 255) > np.sum(img_legend > 0) * 0.25:
                    black_threshold = 30
                
                rgb_trimmed = np.zeros((img_legend.shape[2], img_legend.shape[0], img_legend.shape[1]), dtype='uint8')
                hsv_trimmed = np.zeros((img_legend.shape[2], img_legend.shape[0], img_legend.shape[1]), dtype='uint8')
                rgb_trimmed = rgb_trimmed.astype(float)
                hsv_trimmed = hsv_trimmed.astype(float)
                for dimension in range(0, 3):
                    rgb_trimmed[dimension] = np.copy(img_legend[:,:,dimension]).astype(float)
                    hsv_trimmed[dimension] = np.copy(hsv_legend[:,:,dimension]).astype(float)

                rgb_trimmed_temp = np.copy(rgb_trimmed)
                rgb_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
                hsv_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
                rgb_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
                hsv_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
                rgb_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
                hsv_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan

                rgb_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
                hsv_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
                rgb_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
                hsv_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
                rgb_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
                hsv_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan



                if np.sum(np.isnan(hsv_trimmed)) >= (hsv_trimmed.shape[0]*hsv_trimmed.shape[1]*hsv_trimmed.shape[2]):
                    color_space_holder = []
                    rgb_lower_box = np.array((0,0,0), dtype='uint8')
                    rgb_upper_box = np.array((0,0,255), dtype='uint8')
                    color_space_holder.append(rgb_lower_box)
                    color_space_holder.append(rgb_upper_box)
                    rgb_lower_box = np.array((245,245,245), dtype='uint8')
                    rgb_upper_box = np.array((255,255,255), dtype='uint8')
                    color_space_holder.append(rgb_lower_box)
                    color_space_holder.append(rgb_upper_box)
                    rgb_lower_box = np.array((245,245,245), dtype='uint8')
                    rgb_upper_box = np.array((255,255,255), dtype='uint8')
                    color_space_holder.append(rgb_lower_box)
                    color_space_holder.append(rgb_upper_box)
                    rgb_lower_box = np.array((245,245,245), dtype='uint8')
                    rgb_upper_box = np.array((255,255,255), dtype='uint8')
                    color_space_holder.append(rgb_lower_box)
                    color_space_holder.append(rgb_upper_box)
                    rgb_lower_box = np.array((245,245,245), dtype='uint8')
                    rgb_upper_box = np.array((255,255,255), dtype='uint8')
                    color_space_holder.append(rgb_lower_box)
                    color_space_holder.append(rgb_upper_box)
                    rgb_lower_box = np.array((245,245,245), dtype='uint8')
                    rgb_upper_box = np.array((255,255,255), dtype='uint8')
                    color_space_holder.append(rgb_lower_box)
                    color_space_holder.append(rgb_upper_box)

                    color_avg_holder = np.array((0,0,0), dtype='uint8')
                    color_avg_holder2 = np.array((0,0,0), dtype='uint8')
                else:
                    color_space_holder = []
                    hsv_lower_box = np.array([int(np.nanquantile(hsv_trimmed[0],.2)),int(np.nanquantile(hsv_trimmed[1],.1)),int(np.nanquantile(hsv_trimmed[2],.1))]) #.2
                    hsv_upper_box = np.array([int(np.nanquantile(hsv_trimmed[0],.8)),int(np.nanquantile(hsv_trimmed[1],.9)),int(np.nanquantile(hsv_trimmed[2],.9))]) #.8
                    color_space_holder.append(hsv_lower_box)
                    color_space_holder.append(hsv_upper_box)
                    hsv_space[int(np.nanquantile(hsv_trimmed[0],.2)): int(np.nanquantile(hsv_trimmed[0],.8))] += poly_counter
                    rgb_lower_box = np.array([int(np.nanquantile(rgb_trimmed[0],.2)),int(np.nanquantile(rgb_trimmed[1],.2)),int(np.nanquantile(rgb_trimmed[2],.2))])
                    rgb_upper_box = np.array([int(np.nanquantile(rgb_trimmed[0],.8)),int(np.nanquantile(rgb_trimmed[1],.8)),int(np.nanquantile(rgb_trimmed[2],.8))])
                    color_space_holder.append(rgb_lower_box)
                    color_space_holder.append(rgb_upper_box)
                    rgb_space[int(np.nanquantile(rgb_trimmed[0],.3)): int(np.nanquantile(rgb_trimmed[0],.7)), int(np.nanquantile(rgb_trimmed[1],.3)): int(np.nanquantile(rgb_trimmed[1],.7)), int(np.nanquantile(rgb_trimmed[2],.3)): int(np.nanquantile(rgb_trimmed[2],.7))] = poly_counter
                    rgb_lower_box = np.array([int(np.nanquantile(rgb_trimmed[0],.1)),int(np.nanquantile(rgb_trimmed[1],.1)),int(np.nanquantile(rgb_trimmed[2],.1))])
                    rgb_upper_box = np.array([int(np.nanquantile(rgb_trimmed[0],.9)),int(np.nanquantile(rgb_trimmed[1],.9)),int(np.nanquantile(rgb_trimmed[2],.9))])
                    color_space_holder.append(rgb_lower_box)
                    color_space_holder.append(rgb_upper_box)
                    rgb_lower_box = np.array([int(np.nanquantile(rgb_trimmed[0],.05)),int(np.nanquantile(rgb_trimmed[1],.05)),int(np.nanquantile(rgb_trimmed[2],.05))])
                    rgb_upper_box = np.array([int(np.nanquantile(rgb_trimmed[0],.95)),int(np.nanquantile(rgb_trimmed[1],.95)),int(np.nanquantile(rgb_trimmed[2],.95))])
                    color_space_holder.append(rgb_lower_box)
                    color_space_holder.append(rgb_upper_box)
                    rgb_lower_box = np.array([int(np.nanquantile(rgb_trimmed[0],.03)),int(np.nanquantile(rgb_trimmed[1],.03)),int(np.nanquantile(rgb_trimmed[2],.03))])
                    rgb_upper_box = np.array([int(np.nanquantile(rgb_trimmed[0],.97)),int(np.nanquantile(rgb_trimmed[1],.97)),int(np.nanquantile(rgb_trimmed[2],.97))])
                    color_space_holder.append(rgb_lower_box)
                    color_space_holder.append(rgb_upper_box)
                    rgb_lower_box = np.array([int(np.nanquantile(rgb_trimmed[0],.02)),int(np.nanquantile(rgb_trimmed[1],.02)),int(np.nanquantile(rgb_trimmed[2],.02))])
                    rgb_upper_box = np.array([int(np.nanquantile(rgb_trimmed[0],.98)),int(np.nanquantile(rgb_trimmed[1],.98)),int(np.nanquantile(rgb_trimmed[2],.98))])
                    color_space_holder.append(rgb_lower_box)
                    color_space_holder.append(rgb_upper_box)
                    rgb_lower_box = np.array([int(np.nanquantile(rgb_trimmed[0],.01)),int(np.nanquantile(rgb_trimmed[1],.01)),int(np.nanquantile(rgb_trimmed[2],.01))])
                    rgb_upper_box = np.array([int(np.nanquantile(rgb_trimmed[0],.99)),int(np.nanquantile(rgb_trimmed[1],.99)),int(np.nanquantile(rgb_trimmed[2],.99))])
                    color_space_holder.append(rgb_lower_box)
                    color_space_holder.append(rgb_upper_box)

                    color_avg_holder = np.array([int(np.nanquantile(rgb_trimmed[0],.5)),int(np.nanquantile(rgb_trimmed[1],.5)),int(np.nanquantile(rgb_trimmed[2],.5))])
                    color_avg_holder2 = np.array([int(np.nanquantile(hsv_trimmed[0],.5)),int(np.nanquantile(hsv_trimmed[1],.5)),int(np.nanquantile(hsv_trimmed[2],.5))])

                color_space.append(color_space_holder)
                color_avg.append(color_avg_holder)
                color_avg2.append(color_avg_holder2)

            print('time checkpoint _v0:', datetime.now()-runningtime_start)
            running_time_v.append(datetime.now()-runningtime_start)

            ans_category = np.zeros((poly_counter+1, img_rb.shape[0], img_rb.shape[1]), dtype='uint8')

            blank = np.ones((img_rb.shape[0],img_rb.shape[1]),dtype=np.uint8)*255
            ans_category[poly_counter] = np.zeros((img_rb.shape[0],img_rb.shape[1]),dtype=np.uint8)






            # Some preprocessing for basemap to support further text detection and polygon separation
            color_dif_counter = 0
            if split_multiprocessing == True:
                with multiprocessing.Pool(int(PROCESSES)) as pool:
                    callback = pool.starmap_async(extraction_step0_color_difference_worker.extraction_step0_color_difference_worker, [(this_poly, map_name, legend_name, solutiona_dir, print_intermediate_image, img_bound, rgb_rb, hsv_rb, color_avg[this_poly], color_avg2[this_poly], color_space[this_poly], ) for this_poly in range(0, poly_counter)])
                    multiprocessing_results = callback.get()

                    for legend, rec in multiprocessing_results:
                        if rec == True:
                            color_dif_counter = color_dif_counter + 1
            print('time checkpoint _v1:', datetime.now()-runningtime_start)
            running_time_v.append(datetime.now()-runningtime_start)

            


            # If there is color shift for all legends, but this section is never used for the final solution. (smoothing_map always set to False)
            if smoothing_map == True:
                color_avg.append(np.array([0,0,0]))
                #color_avg.append(np.array([255,255,255]))

                img_bound_argwhere = np.argwhere(img_bound)
                (ystart, xstart), (ystop, xstop) = img_bound_argwhere.min(0), img_bound_argwhere.max(0) + 1 
                im = np.copy(rgb_rb[ystart:ystop, xstart:xstop, :])
                image = im.reshape(im.shape[0],im.shape[1],1,3)

                # Create color container 
                colors_container = np.ones(shape=[image.shape[0],image.shape[1],len(color_avg),3])
                for i,color in enumerate(color_avg):
                    colors_container[:,:,i,:] = color
                
                rgb_weight = np.ones(shape=[image.shape[0],image.shape[1],1,3])
                rgb_weight[:,:,:,0] = 1 # 2
                rgb_weight[:,:,:,1] = 1 # 4
                rgb_weight[:,:,:,2] = 1 # 3

                background_correction_direct_rgb = np.ones(shape=[image.shape[0],image.shape[1],1,3])
                background_correction_direct_rgb[:,:,:,0] = 0.9
                background_correction_direct_rgb[:,:,:,1] = 0.9
                background_correction_direct_rgb[:,:,:,2] = 0.9

                image_deviation = np.zeros(shape=[image.shape[0],image.shape[1],1,3])
                image_deviation[:,:,:,0] = image[:,:,:,0] - image[:,:,:,1]
                image_deviation[:,:,:,1] = image[:,:,:,0] - image[:,:,:,2]
                image_deviation[:,:,:,2] = image[:,:,:,1] - image[:,:,:,2]

                legend_deviation = np.zeros(shape=[image.shape[0],image.shape[1],len(color_avg),3])
                legend_deviation[:,:,:,0] = colors_container[:,:,:,0] - colors_container[:,:,:,1]
                legend_deviation[:,:,:,1] = colors_container[:,:,:,0] - colors_container[:,:,:,2]
                legend_deviation[:,:,:,2] = colors_container[:,:,:,1] - colors_container[:,:,:,2]
                
                background_correction_deviated_rgb = np.ones(shape=[image.shape[0],image.shape[1],1,3])
                background_correction_deviated_rgb[:,:,:,:] = 0.5 + 0.5*(1.0-abs(image_deviation[:,:,:,:])/255.0)

                print('processing _v0 >>> _v1 (finding the closest color)... :', datetime.now()-runningtime_start)


                def closest(image,color_container):
                    shape = image.shape[:2]
                    total_shape = shape[0]*shape[1]

                    # calculate distances
                    distances_0 = np.sqrt(np.sum(rgb_weight*((color_container*background_correction_direct_rgb-image)**2),axis=3))
                    distances_1 = np.sqrt(np.sum(((legend_deviation*background_correction_deviated_rgb-image_deviation)**2),axis=3))
                    distances = distances_0*0.9 + distances_1*0.1

                    min_index = np.argmin(distances,axis=2).reshape(-1)
                    natural_index = np.arange(total_shape)

                    reshaped_container = colors_container.reshape(-1,len(color_avg),3)

                    color_view = reshaped_container[natural_index,min_index].reshape(shape[0],shape[1],3)
                    return color_view, distances
                
                result_image, distances = closest(image,colors_container)
                result_image = result_image.astype(np.uint8)

                #plt.imshow(result_image)
                #plt.show()

                #Image.fromarray(result_image.astype(np.uint8)).show()


                #subtract_rgb = []

                
                # multiprocessing_step1
                with multiprocessing.Pool(int(PROCESSES)) as pool:
                    callback = pool.starmap_async(extraction_step1_worker.extraction_step1_worker, [(this_poly, map_name, legend_name, solutiona_dir, print_intermediate_image, rgb_rb, rgb_ms, result_image, distances[:,:,this_poly], len(color_avg), color_avg[this_poly], poly_counter, np.sum(img_bound), image.shape, im, ) for this_poly in range(0, poly_counter)])
                    multiprocessing_results = callback.get()

                    for legend, img_masked, this_subtract_rgb in multiprocessing_results:
                        # add masked result into private ans_category
                        ans_category[legend] = np.copy(img_masked)
                        # add mophological result into global ans_category
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                        img_masked_morphology = cv2.morphologyEx(img_masked, cv2.MORPH_OPEN, kernel, iterations=1)
                        img_masked_morphology[img_masked_morphology > 0] = legend+1
                        ans_category[poly_counter] = cv2.add(ans_category[poly_counter], img_masked_morphology)
                        
                        for space in range(2, len(color_space[legend]), 2):
                            color_space[legend][space] = color_space[legend][space] - this_subtract_rgb
                            color_space[legend][space+1] = color_space[legend][space+1] - this_subtract_rgb + 1
                        
                
                print('processing _v1 >>> _v2 (legend '+str(legend+1)+'/'+str(poly_counter)+')... :', datetime.now()-runningtime_start)


                result_image0 = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                out_file_path0=solutiona_dir+'intermediate7(2)/'+map_name+'/'+map_name+'_nearest.png'
                cv2.imwrite(out_file_path0, result_image0)



            else:

                if split_multiprocessing == True:
                    with multiprocessing.Pool(int(PROCESSES*2)) as pool:
                        callback = pool.starmap_async(extraction_step2_worker.extraction_step2_worker, [(this_poly, map_name, legend_name, solutiona_dir, print_intermediate_image, poly_counter, np.sum(img_bound), hsv_rb, rgb_rb, hsv_ms, rgb_ms, hsv_space, color_space[this_poly], ) for this_poly in range(0, poly_counter)])
                        multiprocessing_results = callback.get()

                        for legend, img_masked, this_updated_color_space in multiprocessing_results:
                            # add masked result into private ans_category
                            ans_category[legend] = np.copy(img_masked)
                            # add mophological result into global ans_category
                            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                            img_masked_morphology = cv2.morphologyEx(img_masked, cv2.MORPH_OPEN, kernel, iterations=1)
                            img_masked_morphology[img_masked_morphology > 0] = legend+1
                            ans_category[poly_counter] = cv2.add(ans_category[poly_counter], img_masked_morphology)
                            
                            color_space[legend] = np.copy(this_updated_color_space)

                print('processing _v0 >>> _v2 (legend '+str(legend+1)+'/'+str(poly_counter)+')... :', datetime.now()-runningtime_start)
            print('time checkpoint _v2:', datetime.now()-runningtime_start)
            running_time_v.append(datetime.now()-runningtime_start)



                
                
            if split_multiprocessing == True and poly_counter > 150: # split legends into multiple parts, multiprocessing for part of legends at a time => recommended if 'more than [around 150] legends in a map'
                for_each_loop = for_each_loop_global
                looping_times = math.ceil(poly_counter/for_each_loop)
            else: # = direct multiprocessing for all legends at a time => recommended if runnable
                looping_times = 1
                for_each_loop = poly_counter
            
            for looping in range(0, looping_times):
                range_min = 0 + for_each_loop*looping
                range_max = min(for_each_loop + for_each_loop*looping, poly_counter)
                print('looping... (round: '+str(looping+1)+'/'+str(looping_times)+')... (legend: '+str(range_min)+'-'+str(range_max)+' /'+str(poly_counter)+')')


                for iteration_relaxing in range(0, 4):
                    global_solution = np.copy(ans_category[poly_counter])
                    global_solution[global_solution > 0] = 0
                    global_solution_empty = 255 - global_solution
                    global_solution_empty = cv2.bitwise_and(global_solution_empty, img_bound)
                    ans_category[poly_counter] = np.zeros((img_rb.shape[0],img_rb.shape[1]),dtype=np.uint8)

                    if split_multiprocessing == True:
                        with multiprocessing.Pool(int(PROCESSES*2)) as pool:
                            callback = pool.starmap_async(extraction_step3_worker.extraction_step3_worker, [(this_poly, map_name, legend_name, solutiona_dir, print_intermediate_image, rgb_rb, rgb_ms, ans_category[this_poly], color_space[this_poly], iteration_relaxing, img_crop_black, img_crop_gray, global_solution_empty, ) for this_poly in range(range_min, range_max)])
                            multiprocessing_results = callback.get()

                            for legend, this_next_result in multiprocessing_results:
                                # add masked result into private ans_category
                                ans_category[legend] = np.copy(this_next_result)
                                # add mophological result into global ans_category
                                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                                img_masked_morphology = cv2.morphologyEx(this_next_result, cv2.MORPH_OPEN, kernel, iterations=1)
                                img_masked_morphology[img_masked_morphology > 0] = legend+1
                                ans_category[poly_counter] = cv2.add(ans_category[poly_counter], img_masked_morphology)

                    print('processing _v2 >>> _v3 (iteration '+str(iteration_relaxing+1)+'/4)... (legend '+str(legend+1)+'/'+str(poly_counter)+')... :', datetime.now()-runningtime_start)
                print('time checkpoint _v3:', datetime.now()-runningtime_start)
                running_time_v.append(datetime.now()-runningtime_start)

                img_crop_black_and_gray = cv2.bitwise_or(img_crop_black, img_crop_gray)




                # keep record of updated region
                #updated_region = np.zeros(poly_counter)
                updated_region = []
                updated_for_relaxing = np.ones((ans_category[poly_counter].shape[0],ans_category[poly_counter].shape[1]),dtype=np.uint8)*255

                # fill ip white pixel (remove noisy black pixel)
                if poly_counter > 150: # MaybeEncodingError # Reason: MemoryError()
                    for iteration in range(0, 2):
                        global_solution = np.copy(ans_category[poly_counter])
                        global_solution_temp = np.copy(ans_category[poly_counter])
                        global_solution_temp[global_solution_temp > 0] = 0
                        global_solution_empty = 255 - global_solution_temp
                        global_solution_empty = cv2.bitwise_and(global_solution_empty, img_bound)
                        ans_category[poly_counter] = np.zeros((img_rb.shape[0],img_rb.shape[1]),dtype=np.uint8)

                        updating_counter_0 = 0
                        updating_counter_1 = 0

                        #updated_region = []
                        next_updated_region = []

                        if split_multiprocessing == True:
                            with multiprocessing.Pool(int(PROCESSES/2)) as pool:
                                if iteration == 0:
                                    callback = pool.starmap_async(extraction_step4_worker.extraction_step4_worker, [(this_poly, map_name, legend_name, solutiona_dir, print_intermediate_image, rgb_rb, rgb_ms, None, ans_category[this_poly], color_space[this_poly], iteration, global_solution_empty, img_crop_black_and_gray, ) for this_poly in range(range_min, range_max)])
                                else:
                                    callback = pool.starmap_async(extraction_step4_worker.extraction_step4_worker, [(this_poly, map_name, legend_name, solutiona_dir, print_intermediate_image, rgb_rb, None, hsv_ms, ans_category[this_poly], color_space[this_poly], iteration, global_solution_empty, img_crop_black_and_gray, ) for this_poly in range(range_min, range_max)])
                                multiprocessing_results = callback.get()
                                
                                for legend, this_next_result, updated_for_relaxing, polygon_updated in multiprocessing_results:
                                    if iteration == 0:
                                        # add masked result into private ans_category
                                        ans_category[legend] = np.copy(this_next_result)
                                        # add mophological result into global ans_category
                                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                                        img_masked_morphology = cv2.morphologyEx(this_next_result, cv2.MORPH_OPEN, kernel, iterations=1)
                                        img_masked_morphology[img_masked_morphology > 0] = legend+1
                                        ans_category[poly_counter] = cv2.add(ans_category[poly_counter], img_masked_morphology)

                                        next_updated_region.append(np.copy(updated_for_relaxing))
                                    else:
                                        if polygon_updated == True:
                                            updating_counter_1 = updating_counter_1 + 1
                                        updating_counter_0 = updating_counter_0 + 1
                                    
                        print('processing _v3 >>> _v4 (iteration '+str(iteration+1)+'/2)... (legend '+str(legend+1)+'/'+str(poly_counter)+')... :', datetime.now()-runningtime_start)
                        updated_region = np.array(np.copy(next_updated_region))

                        if iteration == 1:
                            print(' - dynamic update ('+str(updating_counter_1)+' / '+str(updating_counter_0)+')')
                    ans_category_temp = np.copy(ans_category)
                else:
                    for iteration in range(0, 2):
                        global_solution = np.copy(ans_category[poly_counter])
                        global_solution_temp = np.copy(ans_category[poly_counter])
                        global_solution_temp[global_solution_temp > 0] = 0
                        global_solution_empty = 255 - global_solution_temp
                        global_solution_empty = cv2.bitwise_and(global_solution_empty, img_bound)
                        ans_category[poly_counter] = np.zeros((img_rb.shape[0],img_rb.shape[1]),dtype=np.uint8)

                        updating_counter_0 = 0
                        updating_counter_1 = 0

                        #updated_region = []
                        next_updated_region = []

                        if split_multiprocessing == True:
                            with multiprocessing.Pool(int(PROCESSES)) as pool:
                                if iteration == 0:
                                    callback = pool.starmap_async(extraction_step4_worker.extraction_step4_worker, [(this_poly, map_name, legend_name, solutiona_dir, print_intermediate_image, rgb_rb, rgb_ms, None, ans_category[this_poly], color_space[this_poly], iteration, global_solution_empty, img_crop_black_and_gray, ) for this_poly in range(range_min, range_max)])
                                else:
                                    callback = pool.starmap_async(extraction_step4_worker.extraction_step4_worker, [(this_poly, map_name, legend_name, solutiona_dir, print_intermediate_image, rgb_rb, None, hsv_ms, ans_category[this_poly], color_space[this_poly], iteration, global_solution_empty, img_crop_black_and_gray, ) for this_poly in range(range_min, range_max)])
                                multiprocessing_results = callback.get()
                                
                                for legend, this_next_result, updated_for_relaxing, polygon_updated in multiprocessing_results:
                                    if iteration == 0:
                                        # add masked result into private ans_category
                                        ans_category[legend] = np.copy(this_next_result)
                                        # add mophological result into global ans_category
                                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                                        img_masked_morphology = cv2.morphologyEx(this_next_result, cv2.MORPH_OPEN, kernel, iterations=1)
                                        img_masked_morphology[img_masked_morphology > 0] = legend+1
                                        ans_category[poly_counter] = cv2.add(ans_category[poly_counter], img_masked_morphology)

                                        next_updated_region.append(np.copy(updated_for_relaxing))
                                    else:
                                        if polygon_updated == True:
                                            updating_counter_1 = updating_counter_1 + 1
                                        updating_counter_0 = updating_counter_0 + 1
                                    
                        print('processing _v3 >>> _v4 (iteration '+str(iteration+1)+'/2)... (legend '+str(legend+1)+'/'+str(poly_counter)+')... :', datetime.now()-runningtime_start)
                        updated_region = np.array(np.copy(next_updated_region))
                        
                        if iteration == 1:
                            print(' - dynamic update ('+str(updating_counter_1)+' / '+str(updating_counter_0)+')')
                    ans_category_temp = np.copy(ans_category)

                print('time checkpoint _v4:', datetime.now()-runningtime_start)
                running_time_v.append(datetime.now()-runningtime_start)





                conv_kernel_set = []
                conv_kernel_threshold0 = [1.0, 0.75, 0.75, 0.5, 0.5, 0.5, 0.5]#, 0.5, 0.5, 0.5]
                conv_kernel_threshold = []

                conv_kernel_0 = np.ones((3,3),dtype=np.uint8)
                conv_kernel_0[1,1] = 0
                conv_kernel_1 = np.ones((5,5),dtype=np.uint8)
                conv_kernel_1[2,2] = 0
                conv_kernel_2 = np.ones((7,7),dtype=np.uint8)
                conv_kernel_2[2:5,2:5] = 0
                conv_kernel_3 = np.ones((9,9),dtype=np.uint8)
                conv_kernel_3[3:6,3:6] = 0
                conv_kernel_4 = np.ones((11,11),dtype=np.uint8)
                conv_kernel_4[3:8,3:8] = 0
                conv_kernel_5 = np.ones((13,13),dtype=np.uint8)
                conv_kernel_5[4:9,4:9] = 0
                conv_kernel_6 = np.ones((15,15),dtype=np.uint8)
                conv_kernel_6[4:11,4:11] = 0

                conv_kernel_set.append(conv_kernel_0)
                conv_kernel_set.append(conv_kernel_1)
                conv_kernel_set.append(conv_kernel_2)
                conv_kernel_set.append(conv_kernel_3)
                conv_kernel_set.append(conv_kernel_4)
                conv_kernel_set.append(conv_kernel_5)
                conv_kernel_set.append(conv_kernel_6)

                for conv_set in range(0, len(conv_kernel_set)):
                    conv_kernel_threshold.append(np.sum(conv_kernel_set[conv_set])*conv_kernel_threshold0[conv_set])



                boundingRange = 3
                masking0 = generate_mask(boundingRange)
                masking = np.copy(masking0)
                masking = masking.astype(float)

                for direction in range(0, 8):
                    #print((masking[direction]==1.0).sum())
                    region_sum = (masking[direction]==1.0).sum()
                    for i, j in np.argwhere(masking[direction]==1.0):
                        masking[direction][i][j] = (1.0/region_sum)


                # keep record of updated region
                #updated_region = np.zeros(poly_counter)
                updated_region = []
                updated_for_relaxing = np.ones((ans_category[poly_counter].shape[0],ans_category[poly_counter].shape[1]),dtype=np.uint8)*255

                # fill ip white pixel (remove noisy black pixel)
                for iteration in range(0, 1):
                    global_solution = np.copy(ans_category[poly_counter])
                    global_solution_temp = np.copy(ans_category[poly_counter])
                    global_solution_temp[global_solution_temp > 0] = 0
                    global_solution_empty = 255 - global_solution_temp
                    global_solution_empty = cv2.bitwise_and(global_solution_empty, img_bound)
                    ans_category[poly_counter] = np.zeros((img_rb.shape[0],img_rb.shape[1]),dtype=np.uint8)

                    updating_counter_0 = 0
                    updating_counter_1 = 0

                    #updated_region = []
                    next_updated_region = []


                    if split_multiprocessing == True:
                        with multiprocessing.Pool(int(PROCESSES)) as pool:
                            if iteration == 0:
                                callback = pool.starmap_async(extraction_step5_worker.extraction_step5_worker, [(this_poly, map_name, legend_name, solutiona_dir, print_intermediate_image, rgb_rb, None, ans_category[this_poly], iteration, global_solution_empty, None, conv_kernel_set, conv_kernel_threshold, masking, ) for this_poly in range(range_min, range_max)]) # img_crop_black_and_gray
                            else:
                                callback = pool.starmap_async(extraction_step5_worker.extraction_step5_worker, [(this_poly, map_name, legend_name, solutiona_dir, print_intermediate_image, rgb_rb, hsv_ms, ans_category[this_poly], iteration, global_solution_empty, None, conv_kernel_set, conv_kernel_threshold, masking, ) for this_poly in range(range_min, range_max)])
                            multiprocessing_results = callback.get()

                            for legend, this_next_result, updated_for_relaxing, polygon_updated in multiprocessing_results:
                                if iteration == 0:
                                    # add masked result into private ans_category
                                    ans_category[legend] = np.copy(this_next_result)
                                    # add mophological result into global ans_category
                                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                                    img_masked_morphology = cv2.morphologyEx(this_next_result, cv2.MORPH_OPEN, kernel, iterations=1)
                                    img_masked_morphology[img_masked_morphology > 0] = legend+1
                                    ans_category[poly_counter] = cv2.add(ans_category[poly_counter], img_masked_morphology)

                                    next_updated_region.append(np.copy(updated_for_relaxing))
                                else:
                                    if polygon_updated == True:
                                        updating_counter_1 = updating_counter_1 + 1
                                    updating_counter_0 = updating_counter_0 + 1
                                
                    print('processing _v4 >>> _v5 (iteration '+str(iteration+1)+'/1)... (legend '+str(legend+1)+'/'+str(poly_counter)+')... :', datetime.now()-runningtime_start)
                    updated_region = np.array(np.copy(next_updated_region))
                    
                    if iteration == 1:
                        print(' - dynamic update ('+str(updating_counter_1)+' / '+str(updating_counter_0)+')')
                #ans_category_updated = []
                #ans_category_temp = np.copy(ans_category)
                print('time checkpoint _v5:', datetime.now()-runningtime_start)
                running_time_v.append(datetime.now()-runningtime_start)
                
                

                

                    
                if poly_counter >= 5 and poly_counter <= 150:
                    print('proceed to text detection...')

                    img_backgroun_v0 = np.copy(img_rb)

                    lower_black_text = np.array([0,0,0])
                    upper_black_text = np.array([70,70,70])
                    mask_box_text0 = cv2.inRange(img_backgroun_v0, lower_black_text, upper_black_text)
                    res_box_text1 = cv2.bitwise_and(img_bound, img_bound, mask=mask_box_text0)
                    threshold_text = cv2.medianBlur(res_box_text1,3)

                    global_hsv_space = np.zeros((3, 400), dtype='uint8')
                    local_hsv_space = np.zeros((poly_counter, 3, 400), dtype='uint8')
                    #hsv_color_space = np.zeros((poly_counter, 2, 3), dtype='uint8')
                    hsv_color_space = []

                    global_hsv_space = np.zeros((3, 400), dtype='uint8')
                    local_hsv_space = np.zeros((poly_counter, 3, 400), dtype='uint8')
                    #hsv_color_space = np.zeros((poly_counter, 2, 3), dtype='uint8')
                    hsv_color_space = []

                    for legend in range(range_min, range_max): ###
                        color_space_holder = []
                        color_space_holder.append(color_space[legend][0])
                        color_space_holder.append(color_space[legend][1])

                        this_hsv_color_space = np.copy(color_space_holder)
                        #hsv_color_space[legend] = np.copy(color_space_holder)
                        hsv_color_space.append(color_space_holder)
                        #print(legend_name[legend], color_space_holder, hsv_color_space[legend][1], this_hsv_color_space)

                        global_hsv_space[0][max(this_hsv_color_space[0][0]-1, 0): 1+this_hsv_color_space[1][0]+1] += 1 # h space
                        global_hsv_space[1][max(this_hsv_color_space[0][1]-15, 0): 1+this_hsv_color_space[1][1]+15] += 1 # s space
                        global_hsv_space[2][max(this_hsv_color_space[0][2]-15, 0): 1+this_hsv_color_space[1][2]+15] += 1 # v space
                        local_hsv_space[legend][0][max(this_hsv_color_space[0][0]-1, 0): 1+this_hsv_color_space[1][0]+1] = 1 # h space
                        local_hsv_space[legend][1][max(this_hsv_color_space[0][1]-15, 0): 1+this_hsv_color_space[1][1]+15] = 1 # s space
                        local_hsv_space[legend][2][max(this_hsv_color_space[0][2]-15, 0): 1+this_hsv_color_space[1][2]+15] = 1 # v space

                    #print('legend loaded...')

                    print('time checkpoint _text_v0:', datetime.now()-runningtime_start)
                    running_time_v.append(datetime.now()-runningtime_start)

                    
                    if split_multiprocessing == True:
                        with multiprocessing.Pool(int(PROCESSES)) as pool:
                            callback = pool.starmap_async(extraction_step6_pre_update_worker.extraction_step6_pre_update_worker, [(this_poly, ans_category[this_poly], ) for this_poly in range(range_min, range_max)])
                            multiprocessing_results = callback.get()

                            for legend, return_image, unique_counter in multiprocessing_results:
                                if unique_counter != 2:
                                    print('extract nothing...')
                                    return_image = np.copy(img_bound)
                                ans_category[legend] = np.copy(return_image)
                    #print('v6 updated...')




                    comparison_needed = []
                    comparison_target = np.empty(poly_counter, dtype=object)
                    if split_multiprocessing == True:
                        with multiprocessing.Pool(int(PROCESSES)) as pool:
                            callback = pool.starmap_async(extraction_step6_specify_overlap_legend_worker.extraction_step6_specify_overlap_legend_worker, [(this_poly, legend_name, hsv_color_space[this_poly], local_hsv_space, global_hsv_space[0], range_min, range_max, ) for this_poly in range(range_min, range_max)])
                            multiprocessing_results = callback.get()

                            for legend, candidate_similar_legend_1, candidate_similar_legend_2 in multiprocessing_results:
                                similar_legend = []

                                for counter_legend in candidate_similar_legend_1:
                                    if np.mean(ans_category[legend]) > 0 and np.mean(ans_category[counter_legend]) > 0:
                                        ans_overlap = cv2.bitwise_and(ans_category[legend], ans_category[counter_legend])
                                        if (np.mean(ans_overlap) / np.mean(ans_category[legend])) > 0.66 and (np.mean(ans_overlap) / np.mean(ans_category[counter_legend])) > 0.66:
                                            # if there are few overlaps in v6 extracted answer, than we don't need text detection
                                            #print('we need to compare them')
                                            #print('overlapping issue with large area: '+legend_name[legend]+' <-> '+legend_name[counter_legend])
                                            similar_legend.append(counter_legend)

                                for counter_legend in candidate_similar_legend_2:
                                    ans_overlap = cv2.bitwise_and(ans_category[legend], ans_category[counter_legend])
                                    if np.mean(ans_category[legend]) > 0 and np.mean(ans_category[counter_legend]) > 0:
                                        if (np.mean(ans_overlap) / np.mean(ans_category[legend])) > 0.2 and (np.mean(ans_overlap) / np.mean(ans_category[counter_legend])) > 0.2:
                                            # if there are few overlaps in v6 extracted answer, than we don't need text detection
                                            #print('we need to compare them')
                                            #print('overlapping issue with similar color: '+legend_name[legend]+' <-> '+legend_name[counter_legend])
                                            similar_legend.append(counter_legend)

                                comparison_target[legend] = np.copy(similar_legend)
                                if len(similar_legend) > 0:
                                    comparison_needed.append(legend)
                    #print(comparison_target)
                    print(comparison_needed)
                    print('time checkpoint _text_v1:', datetime.now()-runningtime_start)
                    running_time_v.append(datetime.now()-runningtime_start)


                    global_res_probability = np.empty(poly_counter, dtype=object)
                    global_confidence = np.empty(poly_counter, dtype=object)


                    # multiprocessing
                    list_for_multiprocessing = []
                    for list_id in range(0, len(comparison_needed)):
                        legend = comparison_needed[list_id]
                        list_for_multiprocessing.append(legend)

                    if split_multiprocessing == True:
                        with multiprocessing.Pool(int(PROCESSES)) as pool:
                            callback = pool.starmap_async(extraction_step6_find_legend_in_map_worker.extraction_step6_find_legend_in_map_worker, [(this_poly, map_name, legend_name, solutiona_dir, threshold_text, None, np.sum(img_bound), True, print_intermediate_image, ) for this_poly in list_for_multiprocessing])
                            multiprocessing_results = callback.get()

                            for legend, threshold, update_image_space in multiprocessing_results:
                                # legend, res, confidence_placeholder = this_poly.get()
                                # global_res_probability[legend] = np.copy(res)
                                # global_confidence[legend] = np.copy(confidence_placeholder)

                                #legend, res = this_poly.get()
                                global_res_probability[legend] = np.copy(update_image_space)
                                loc_arg = np.argwhere(update_image_space >= 255*threshold)
                                global_confidence[legend] = np.copy(loc_arg)
                    print('time checkpoint _text_v2:', datetime.now()-runningtime_start)
                    running_time_v.append(datetime.now()-runningtime_start)



                    


                    # multiprocessing
                    temp_ans_category = np.copy(ans_category)
                    list_for_multiprocessing = []
                    for list_id in range(0, len(comparison_needed)):
                        legend = comparison_needed[list_id]
                        list_for_multiprocessing.append(legend)
                    print(list_for_multiprocessing)


                    if split_multiprocessing == True:
                        def generate_cluster_sequential(legend, input_image, blur_radius_initial, blur_radius_step):
                            # smooth the image (to remove small objects)
                            blur_radius = blur_radius_initial
                            threshold_blur = 0
                            imgf = ndimage.gaussian_filter(input_image, blur_radius)

                            # find connected components
                            labeled, nr_objects = ndimage.label(imgf > threshold_blur)

                            while nr_objects > 100:
                                # smooth the image (to remove small objects)
                                blur_radius = blur_radius + blur_radius_step
                                threshold_blur = 0
                                imgf = ndimage.gaussian_filter(input_image, blur_radius)

                                # find connected components
                                labeled, nr_objects = ndimage.label(imgf > threshold_blur)

                            return labeled, nr_objects


                        for legend in list_for_multiprocessing:
                            updated_region = np.copy(ans_category[legend])
                            updated_region_rollback = np.copy(updated_region)

                            try:
                                with multiprocessing.Pool(int(PROCESSES)) as pool:
                                    callback = pool.starmap_async(extraction_step6_compare_against_competitor_worker.update_based_on_text, [(legend, counter_legend, ans_category[legend], ans_category[counter_legend], global_confidence[legend], global_confidence[counter_legend], global_res_probability[legend], global_res_probability[counter_legend], img_boundary, ) for counter_legend in comparison_target[legend]]) # img_crop_black
                                    multiprocessing_results = callback.get()

                                    for this_legend, this_counter_legend, img_ans_v1 in multiprocessing_results:
                                        ban_region = cv2.subtract(ans_category[legend], img_ans_v1)
                                        updated_region = cv2.subtract(updated_region, ban_region)
                            
                            except OSError:
                                print('OSError... Run Sequential Processing Instead...')

                                # Sequential processing
                                updated_region = np.copy(updated_region_rollback)

                                # Sequential processing
                                for counter_legend in comparison_target[legend]:
                                    img_ans_v0 = np.copy(ans_category[legend])
                                    #save_region_temp = cv2.subtract(ans_category_this_legend, ans_category[counter_legend])
                                    temp_competitor = 255 - ans_category[counter_legend]
                                    save_region_temp = cv2.bitwise_and(ans_category[legend], temp_competitor)
                                    img_ans_v0 = cv2.subtract(img_ans_v0, save_region_temp)

                                    labeled, nr_objects = generate_cluster_sequential(legend, img_ans_v0, 15.0, 5.0)

                                    depot_checked_polygon = np.zeros((labeled.shape[0],labeled.shape[1]),dtype=np.uint8)

                                    depot_got = 0
                                    distance_threshold = 300

                                    for object_traverse in range(1, nr_objects):
                                        cluster_object = np.argwhere(labeled == object_traverse)
                                        if cluster_object.shape[0] == 0:
                                            continue

                                        center_x = np.mean(cluster_object, axis=0)[0]
                                        center_y = np.mean(cluster_object, axis=0)[1]

                                        belong_to_other_group = False
                                        belong_to_this_group = False

                                        # The text located 'in' this polygon => include
                                        in_polygon_arg = np.logical_and(labeled[0:global_res_probability[legend].shape[0], 0:global_res_probability[legend].shape[1]] == object_traverse, global_res_probability[legend] >= 0.5)
                                        in_polygon_bool = (True in in_polygon_arg)
                                        if in_polygon_bool == True:
                                            belong_to_other_group = False
                                            belong_to_this_group = True

                                        # The text located 'nearby' this polygon => include (especially for small polygons, where legends are labeled outside)
                                        if belong_to_other_group == False:
                                            center_placeholder = np.array([[center_x, center_y]])
                                            center_placeholder = np.repeat(center_placeholder, len(global_confidence[legend]), axis=0)
                                            distances_2_depot = np.sqrt(np.sum((global_confidence[legend]-center_placeholder)**2,axis=1))
                                            min_distance_to_self = np.min(distances_2_depot)
                                            
                                            if min_distance_to_self < distance_threshold:
                                                belong_to_other_group = False
                                                belong_to_this_group = True

                                        # The counter-text located 'in' this polygon => exclude
                                        if belong_to_other_group == False:
                                            center_placeholder = np.array([[center_x, center_y]])
                                            center_placeholder = np.repeat(center_placeholder, len(global_confidence[counter_legend]), axis=0)
                                            distances_2_depot = np.sqrt(np.sum((global_confidence[counter_legend]-center_placeholder)**2,axis=1))
                                            min_distance_to_counter = np.min(distances_2_depot)
                                            
                                            in_polygon_arg = np.logical_and(labeled[0:global_res_probability[counter_legend].shape[0], 0:global_res_probability[counter_legend].shape[1]] == object_traverse, global_res_probability[counter_legend] >= 0.75)
                                            in_polygon_bool = (True in in_polygon_arg)
                                            if in_polygon_bool == True:
                                                belong_to_other_group = True
                                                belong_to_this_group = False
                                            

                                        if belong_to_other_group == False and belong_to_this_group == False:
                                            #if min_distance_to_self/confidence_to_self < (min_distance_to_counter/confidence_to_counter)*1.33:
                                            if min_distance_to_self < (min_distance_to_counter)*2.0:
                                                belong_to_this_group = True


                                        if belong_to_this_group == True and belong_to_other_group == False :
                                            # create a mask to only preserve current legend color in the basemap
                                            depot_checked_polygon_0 = np.zeros((labeled.shape[0],labeled.shape[1]),dtype=np.uint8)
                                            depot_checked_polygon_0[np.logical_and(labeled==object_traverse, ans_category[legend]>0)] = 255
                                            depot_checked_polygon_0 = cv2.bitwise_and(depot_checked_polygon_0, ans_category[legend])
                                            depot_checked_polygon = cv2.bitwise_or(depot_checked_polygon, depot_checked_polygon_0)
                                            depot_got = depot_got+1


                                    #print('updated against - '+legend_name[counter_legend])
                                    #plt.imshow(save_region_temp)
                                    #plt.show()
                                    #plt.imshow(depot_checked_polygon)
                                    #plt.show()
                                    img_ans_v1 = cv2.bitwise_and(ans_category[legend], depot_checked_polygon)
                                    img_ans_v1 = cv2.bitwise_or(img_ans_v1, save_region_temp)
                                    #plt.imshow(img_ans_v1)
                                    #plt.show()

                                    # return legend, counter_legend, img_ans_v1
                                    ban_region = cv2.subtract(ans_category[legend], img_ans_v1)
                                    updated_region = cv2.subtract(updated_region, ban_region)

                                    

                            # remove noisy white pixel
                            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                            opening = cv2.morphologyEx(updated_region, cv2.MORPH_OPEN, kernel, iterations=1)
                            updated_region=cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY)[1]
                            
                            if poly_counter <= 30:
                                if np.unique(updated_region).shape[0] != 2 or (np.sum(updated_region)/np.unique(updated_region)[1]) / (np.sum(ans_category[legend])/np.unique(ans_category[legend])[1]) < 0.0005:
                                    print(legend_name[legend]+' rollback...')
                                    updated_region = np.copy(ans_category[legend])
                            else:
                                if np.unique(updated_region).shape[0] != 2 or (np.sum(updated_region)/np.unique(updated_region)[1]) / (np.sum(ans_category[legend])/np.unique(ans_category[legend])[1]) < 0.0001:
                                    print(legend_name[legend]+' rollback...')
                                    updated_region = np.copy(ans_category[legend])


                            if print_intermediate_image == True:
                                out_file_path000=os.path.join(solutiona_dir+'intermediate7(2)', map_name, map_name+'_'+legend_name[legend]+'_poly_v6.png')
                                cv2.imwrite(out_file_path000, updated_region)
                            
                            temp_ans_category[legend] = np.copy(updated_region) ### updated v6 to v7 (3/3)

                            print('processing _v2(t) >>> _v3(t) (selected legend '+str(legend)+'/'+str(list_for_multiprocessing)+')... :', datetime.now()-runningtime_start)
                            
                        print('time checkpoint _text_v3:', datetime.now()-runningtime_start)
                        running_time_v.append(datetime.now()-runningtime_start)


                    print('text detection finished...')
                    ans_category = np.copy(temp_ans_category)

                else:
                    print('no text detection needed...')
                    running_time_v.append(datetime.now()-runningtime_start)
                    running_time_v.append(datetime.now()-runningtime_start)
                    running_time_v.append(datetime.now()-runningtime_start)
                    running_time_v.append(datetime.now()-runningtime_start)





                # multiprocessing_step7
                finisher_counter = 0
                if split_multiprocessing == True:
                    with multiprocessing.Pool(PROCESSES) as pool:
                        callback = pool.starmap_async(extraction_step7_worker.extraction_step7_worker, [(this_poly, map_name, legend_name, solutiona_dir, file_path, ans_category[this_poly], img_bound, ) for this_poly in range(range_min, range_max)])
                        multiprocessing_results = callback.get()

                        for legend, pred_binary_raster in multiprocessing_results:
                            #legend, pred_binary_raster = this_poly.get()
                            # doing nothing
                            finisher_counter = finisher_counter + 1
                '''
                else:
                    for this_poly in range(range_min, range_max):
                        legend, pred_binary_raster = extraction_step6, (this_poly)
                '''

                print('time checkpoint _v7:', datetime.now()-runningtime_start)
                running_time_v.append(datetime.now()-runningtime_start)



            if os.path.isfile(solutiona_dir+'intermediate7(2)/'+'running_time_record_v3.csv') == False:
                with open(solutiona_dir+'intermediate7(2)/'+'running_time_record_v3.csv','w') as fd:
                    fd.write('File,checkpoint_0,checkpoint_1,checkpoint_2,checkpoint_3,checkpoint_4,checkpoint_5,checkpoint_t0,checkpoint_t1,checkpoint_t2,checkpoint_t3,checkpoint_7,\n')
                    fd.close()
            with open(solutiona_dir+'intermediate7(2)/'+'running_time_record_v3.csv','a') as fd:
                fd.write(map_name+',')
                for rtc in range(0, len(running_time_v)):
                    fd.write(str(running_time_v[rtc])+',')
                fd.write('\n')
                fd.close()




    print('time check... worker_main_component: ', datetime.now()-runningtime_start_global)
    # 589m 21.9s













def run():
    multiprocessing_setting()
    setting_summary()
    specify_polygon()
    worker_preprocessing()
    worker_boundary_extraction()
    worker_auxiliary_info()
    worker_recoloring()
    worker_main_component()



def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')



def metadata_preprocessing(
        input_data_dir = 'Data/validation',
        input_data_boundary_dir = 'Data/validation_groundtruth',
        input_solutiona_dir = 'Solution_1102/',
        input_targeted_map_list = 'targeted_map.csv',
        input_map_preprocessing = True,
        input_generate_boundary_extraction = True,
        input_printing_auxiliary_information = True,
        input_preprocessing_recoloring = True
):
    global data_dir
    global data_boundary_dir
    global solutiona_dir
    global targeted_map_list
    global map_preprocessing
    global generate_boundary_extraction
    global printing_auxiliary_information
    global preprocessing_recoloring


    data_dir = input_data_dir
    data_boundary_dir = input_data_boundary_dir
    solutiona_dir = input_solutiona_dir
    targeted_map_list = input_targeted_map_list
    map_preprocessing = input_map_preprocessing
    generate_boundary_extraction = input_generate_boundary_extraction
    printing_auxiliary_information = input_printing_auxiliary_information
    preprocessing_recoloring = input_preprocessing_recoloring

    print(map_preprocessing, generate_boundary_extraction, printing_auxiliary_information, preprocessing_recoloring)

    run()




def main():
    global data_dir
    global data_boundary_dir
    global solutiona_dir
    global targeted_map_list
    global map_preprocessing
    global generate_boundary_extraction
    global printing_auxiliary_information
    global preprocessing_recoloring

    data_dir = args.data_dir
    data_boundary_dir = args.data_boundary_dir
    solutiona_dir = 'Solution_' + args.solutiona_dir + '/'
    targeted_map_list = args.targeted_map_list
    map_preprocessing = str_to_bool(args.map_preprocessing)
    generate_boundary_extraction = str_to_bool(args.generate_boundary_extraction)
    printing_auxiliary_information = str_to_bool(args.printing_auxiliary_information)
    preprocessing_recoloring = str_to_bool(args.preprocessing_recoloring)

    run()




import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Data/validation')
    parser.add_argument('--data_boundary_dir', type=str, default='Data/validation_groundtruth')
    parser.add_argument('--solutiona_dir', type=str, default='1102')
    parser.add_argument('--targeted_map_list', type=str, default='targeted_map.csv')
    parser.add_argument('--map_preprocessing', type=str, default='True')
    parser.add_argument('--generate_boundary_extraction', type=str, default='True')
    parser.add_argument('--printing_auxiliary_information', type=str, default='True')
    parser.add_argument('--preprocessing_recoloring', type=str, default='True')

    args = parser.parse_args()
    
    #print(f"Processing output for: {args.result_name}")
    main()

