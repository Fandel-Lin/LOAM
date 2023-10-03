import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
from scipy import ndimage
import math

def generate_cluster(legend, input_image, blur_radius_initial, blur_radius_step, img_boundary, ans_category_this_legend_shape_0, ans_category_this_legend_shape_1): # ans_category_this_legend # img_crop_black
    # smooth the image (to remove small objects)
    blur_radius = blur_radius_initial
    threshold_blur = 0
    imgf = ndimage.gaussian_filter(input_image, blur_radius)

    boundary_attempting = True
    temp_imgf = np.copy(imgf)

    if boundary_attempting == True:
        # separate candidate polygons based on boundaries
        imgf = cv2.bitwise_and(imgf, imgf, mask=(255-img_boundary))

        # find connected components
        labeled, nr_objects = ndimage.label(imgf > threshold_blur)

        achievable = 0
        while nr_objects > 100 and boundary_attempting == True:
            # smooth the image (to remove small objects)
            blur_radius = blur_radius + blur_radius_step
            threshold_blur = 0
            imgf = ndimage.gaussian_filter(input_image, blur_radius)

            # separate candidate polygons based on boundaries
            imgf = cv2.bitwise_and(imgf, imgf, mask=(255-img_boundary))

            # find connected components
            labeled, nr_objects = ndimage.label(imgf > threshold_blur)

            achievable = achievable + 1
            if achievable > 5:
                boundary_attempting = False
                break

        if boundary_attempting == True:
            for object_traverse in range(1, nr_objects):
                background = np.zeros((labeled.shape[0], labeled.shape[1]), np.uint8)
                background[labeled == object_traverse] = 255

                dilate_kernel = np.ones((10,10), np.uint8)
                dilate_candidate = cv2.dilate(background, dilate_kernel, iterations=1)
                dilate_candidate = cv2.bitwise_and(dilate_candidate, img_boundary)

                labeled[np.logical_and(labeled==0, dilate_candidate>0)] = object_traverse
    if boundary_attempting == False:
        imgf = np.copy(temp_imgf)
        blur_radius = blur_radius_initial

        # find connected components
        labeled, nr_objects = ndimage.label(imgf > threshold_blur)

        while nr_objects > 100:
            # smooth the image (to remove small objects)
            blur_radius = blur_radius + blur_radius_step
            threshold_blur = 0
            imgf = ndimage.gaussian_filter(input_image, blur_radius)

            # find connected components
            labeled, nr_objects = ndimage.label(imgf > threshold_blur)




    current_nr_objects = nr_objects
    for object_traverse in range(1, nr_objects):
        # if this object is too large (span from large x-y), split into pieces based on grid
        cluster_object_arg = np.argwhere(labeled == object_traverse)
        #print(cluster_object_arg.min(0), cluster_object_arg.max(0))

        (y_min, x_min), (y_max, x_max) = cluster_object_arg.min(0), cluster_object_arg.max(0) + 1 
        if (y_max-y_min) > 3000 and (x_max-x_min) > 3000:
            #print(y_min,y_max, x_min,x_max)
            this_labeled_object = np.copy(labeled)
            this_labeled_object[this_labeled_object > object_traverse] = 0
            this_labeled_object[this_labeled_object < object_traverse] = 0
            this_labeled_object[this_labeled_object == object_traverse] = current_nr_objects

            # construct a 1000*1000 grid
            min_w = math.floor(y_min/500)
            max_w = math.floor(y_max/500)
            min_h = math.floor(x_min/500)
            max_h = math.floor(x_max/500)

            #print(min_w, max_w, min_h, max_h)
            #this_grid = np.zeros((ans_category_this_legend.shape[0], ans_category_this_legend.shape[1]), dtype='uint8')
            this_grid = np.zeros((ans_category_this_legend_shape_0, ans_category_this_legend_shape_1), dtype='uint8')
            for grid_w in range(min_w, max_w+1):
                for grid_h in range(min_h, max_h+1):
                    this_adding = (grid_w-min_w)*(max_h+1-min_h) + grid_h + 1
                    this_grid[grid_w*500:(grid_w+1)*500, grid_h*500:(grid_h+1)*500] = this_adding

            #updated_labeled = np.copy(labeled)
            grid_count = (max_w-min_w)*(max_h-min_h)
            for local_grid in range(0, grid_count):
                #print(current_nr_objects+local_grid)
                labeled[np.logical_and(labeled==object_traverse, this_grid==local_grid)] = (current_nr_objects+local_grid)

            current_nr_objects = current_nr_objects+grid_count
    nr_objects = current_nr_objects


    return labeled, nr_objects


def update_based_on_text(legend, counter_legend, ans_category_this_legend, ans_category_counter_legend, global_confidence_this_legend, global_confidence_counter_legend, global_res_probability_this_legend, global_res_probability_counter_legend, img_boundary):
    img_ans_v0 = np.copy(ans_category_this_legend)
    #save_region_temp = cv2.subtract(ans_category_this_legend, ans_category[counter_legend])
    temp_competitor = 255 - ans_category_counter_legend
    save_region_temp = cv2.bitwise_and(ans_category_this_legend, temp_competitor)
    img_ans_v0 = cv2.subtract(img_ans_v0, save_region_temp)

    labeled, nr_objects = generate_cluster(legend, img_ans_v0, 15.0, 5.0, img_boundary, ans_category_this_legend.shape[0], ans_category_this_legend.shape[1])

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
        in_polygon_arg = np.logical_and(labeled[0:global_res_probability_this_legend.shape[0], 0:global_res_probability_this_legend.shape[1]] == object_traverse, global_res_probability_this_legend >= 0.5)
        in_polygon_bool = (True in in_polygon_arg)
        if in_polygon_bool == True:
            belong_to_other_group = False
            belong_to_this_group = True

        # The text located 'nearby' this polygon => include (especially for small polygons, where legends are labeled outside)
        if belong_to_other_group == False:
            center_placeholder = np.array([[center_x, center_y]])
            center_placeholder = np.repeat(center_placeholder, len(global_confidence_this_legend), axis=0)
            distances_2_depot = np.sqrt(np.sum((global_confidence_this_legend-center_placeholder)**2,axis=1))
            min_distance_to_self = np.min(distances_2_depot)
            
            if min_distance_to_self < distance_threshold:
                belong_to_other_group = False
                belong_to_this_group = True

        # The counter-text located 'in' this polygon => exclude
        if belong_to_other_group == False:
            center_placeholder = np.array([[center_x, center_y]])
            center_placeholder = np.repeat(center_placeholder, len(global_confidence_counter_legend), axis=0)
            distances_2_depot = np.sqrt(np.sum((global_confidence_counter_legend-center_placeholder)**2,axis=1))
            min_distance_to_counter = np.min(distances_2_depot)
            
            in_polygon_arg = np.logical_and(labeled[0:global_res_probability_counter_legend.shape[0], 0:global_res_probability_counter_legend.shape[1]] == object_traverse, global_res_probability_counter_legend >= 0.75)
            in_polygon_bool = (True in in_polygon_arg)
            if in_polygon_bool == True:
                belong_to_other_group = True
                belong_to_this_group = False
            

        if belong_to_other_group == False and belong_to_this_group == False:
            #if min_distance_to_self/confidence_to_self < (min_distance_to_counter/confidence_to_counter)*1.33:
            if min_distance_to_self < (min_distance_to_counter)*2.5:
                belong_to_this_group = True


        if belong_to_this_group == True and belong_to_other_group == False :
            # create a mask to only preserve current legend color in the basemap
            depot_checked_polygon_0 = np.zeros((labeled.shape[0],labeled.shape[1]),dtype=np.uint8)
            depot_checked_polygon_0[np.logical_and(labeled==object_traverse, ans_category_this_legend>0)] = 255
            depot_checked_polygon_0 = cv2.bitwise_and(depot_checked_polygon_0, ans_category_this_legend)
            depot_checked_polygon = cv2.bitwise_or(depot_checked_polygon, depot_checked_polygon_0)
            depot_got = depot_got+1


    #print('updated against - '+legend_name[counter_legend])
    #plt.imshow(save_region_temp)
    #plt.show()
    #plt.imshow(depot_checked_polygon)
    #plt.show()
    img_ans_v1 = cv2.bitwise_and(ans_category_this_legend, depot_checked_polygon)
    img_ans_v1 = cv2.bitwise_or(img_ans_v1, save_region_temp)
    #plt.imshow(img_ans_v1)
    #plt.show()

    return legend, counter_legend, img_ans_v1













def generate_cluster_linux(legend, input_image, blur_radius_initial, blur_radius_step, ans_category):
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


    current_nr_objects = nr_objects
    for object_traverse in range(1, nr_objects):
        # if this object is too large (span from large x-y), split into pieces based on grid
        cluster_object_arg = np.argwhere(labeled == object_traverse)
        #print(cluster_object_arg.min(0), cluster_object_arg.max(0))

        (y_min, x_min), (y_max, x_max) = cluster_object_arg.min(0), cluster_object_arg.max(0) + 1 
        if (y_max-y_min) > 3000 and (x_max-x_min) > 3000:
            #print(y_min,y_max, x_min,x_max)
            this_labeled_object = np.copy(labeled)
            this_labeled_object[this_labeled_object > object_traverse] = 0
            this_labeled_object[this_labeled_object < object_traverse] = 0
            this_labeled_object[this_labeled_object == object_traverse] = current_nr_objects

            # construct a 1000*1000 grid
            min_w = math.floor(y_min/1000)
            max_w = math.floor(y_max/1000)
            min_h = math.floor(x_min/1000)
            max_h = math.floor(x_max/1000)

            #print(min_w, max_w, min_h, max_h)
            this_grid = np.zeros((ans_category[legend].shape[0], ans_category[legend].shape[1]), dtype='uint8')
            for grid_w in range(min_w, max_w+1):
                for grid_h in range(min_h, max_h+1):
                    this_adding = (grid_w-min_w)*(max_h+1-min_h) + grid_h + 1
                    this_grid[grid_w*1000:(grid_w+1)*1000, grid_h*1000:(grid_h+1)*1000] = this_adding

            #updated_labeled = np.copy(labeled)
            grid_count = (max_w-min_w)*(max_h-min_h)
            for local_grid in range(0, grid_count):
                #print(current_nr_objects+local_grid)
                labeled[np.logical_and(labeled==object_traverse, this_grid==local_grid)] = (current_nr_objects+local_grid)

            current_nr_objects = current_nr_objects+grid_count
    nr_objects = current_nr_objects

    return labeled, nr_objects


def update_based_on_text_linux(legend, counter_legend, ans_category, global_confidence, global_res_probability):
    img_ans_v0 = np.copy(ans_category[legend])
    #save_region_temp = cv2.subtract(ans_category[legend], ans_category[counter_legend])
    temp_competitor = 255 - ans_category[counter_legend]
    save_region_temp = cv2.bitwise_and(ans_category[legend], temp_competitor)
    img_ans_v0 = cv2.subtract(img_ans_v0, save_region_temp)

    labeled, nr_objects = generate_cluster_linux(legend, img_ans_v0, 15.0, 5.0, ans_category)

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

    return img_ans_v1


def compare_against_competitor_linux(legend, map_name, legend_name, solutiona_dir, print_intermediate_image, poly_counter, ans_category, comparison_target_subset, global_confidence, global_res_probability):
    updated_region = np.copy(ans_category[legend])

    for counter_list_id in range(0, len(comparison_target_subset)):
        counter_legend = comparison_target_subset[counter_list_id]
        #print(legend_name[legend]+' <-> '+legend_name[counter_legend])
        img_ans_v1 = update_based_on_text_linux(legend, counter_legend, ans_category, global_confidence, global_res_probability)
        ban_region = cv2.subtract(ans_category[legend], img_ans_v1)
        updated_region = cv2.subtract(updated_region, ban_region)
    
    
    # remove noisy white pixel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(updated_region, cv2.MORPH_OPEN, kernel, iterations=1)
    updated_region=cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY)[1]
    
    if poly_counter <= 40:
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

    return legend, updated_region