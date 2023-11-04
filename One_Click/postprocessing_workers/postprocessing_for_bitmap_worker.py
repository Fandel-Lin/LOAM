import numpy as np
from matplotlib import pyplot as plt
import cv2
import random
import os
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
import math
import json
from datetime import datetime
from scipy import sparse
import pyvips
import shutil

def postprocessing_for_bitmap_worker_multiple_image(map_id, legend_id, this_map_name, this_legend_name, data_dir0, data_dir1, data_dir2, data_dir3, data_dir4, target_dir_img, target_dir_mask, target_dir_img_small, target_dir_mask_small, crop_size=1024):
    source_path_0 = os.path.join(data_dir0, this_map_name+'/'+this_map_name+'_'+this_legend_name+"_v7.png") # input segmentation (input polygon candidate)
    target_path_0 = os.path.join(target_dir_img, str(this_map_name+'_'+this_legend_name+".png"))
    source_path_1 = os.path.join(data_dir1, this_map_name+'_'+this_legend_name+".tif") # groundtruth
    target_path_1 = os.path.join(target_dir_mask, str(this_map_name+'_'+this_legend_name+"_mask.png"))

    source_path_2 = os.path.join(data_dir0, this_map_name+'/'+this_map_name+'_'+this_legend_name+"_v4v.png") # dynamic-threshold polygon candidate
    target_path_2 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_0.png"))
    source_path_3 = os.path.join(data_dir0, this_map_name+'/'+this_map_name+'_'+this_legend_name+'_c0_x.png') # rgb color difference
    target_path_3 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_1.png"))
    source_path_4 = os.path.join(data_dir0, this_map_name+'/'+this_map_name+'_'+this_legend_name+'_c1_x.png') # hsv color difference
    target_path_4 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_2.png"))
    source_path_5 = os.path.join(data_dir2, this_map_name+'_crop_blackregion.png') # black basemap
    target_path_5 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_3.png"))
    source_path_6 = os.path.join(data_dir3, 'Extraction(3)', str(this_map_name+'_overall_boundary_candidate.png')) # identified boundary
    target_path_6 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_4.png"))
    source_path_7 = os.path.join(data_dir4, this_map_name+'/'+this_map_name+'_'+this_legend_name+"_rc_v0.png") # recoloring polygon
    target_path_7 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_5.png"))

    #source_path_8 = os.path.join(data_dir3, 'Groundtruth', str(this_map_name+'_boundary_identified.png')) # identified boundary
    #target_path_8 = os.path.join(target_dir_mask, 'sup', str(this_map_name+'_'+this_legend_name+"_mask_sup.png"))

    #source_path_ext = os.path.join(data_dir2, this_map_name+'_expected_crop_region.png')
    shutil.copyfile(source_path_0, target_path_0)
    shutil.copyfile(source_path_1, target_path_1)
    #shutil.copyfile(source_path_8, target_path_8)


    figure_info = []
    figure_info.append([source_path_0])
    figure_info.append([source_path_1])
    figure_info.append([source_path_2])
    figure_info.append([source_path_3])
    figure_info.append([source_path_4])
    figure_info.append([source_path_5])
    figure_info.append([source_path_6])
    figure_info.append([source_path_7])


    for this_img in range(0, len(figure_info)):
        img = cv2.imread(figure_info[this_img][0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        empty_grid = np.zeros((crop_size, crop_size), dtype='uint8').astype(float)


        for r in range(0,math.ceil(img.shape[0]/crop_size)):
            for c in range(0,math.ceil(img.shape[1]/crop_size)):
                if this_img == 0:
                    this_output_file_name = os.path.join(target_dir_img_small, str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+".png"))
                elif this_img == 1:
                    this_output_file_name = os.path.join(target_dir_mask_small, str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+"_mask.png"))
                elif this_img == 2:
                    this_output_file_name = os.path.join(target_dir_img_small, 'sup', str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+"_sup_0.png"))
                elif this_img == 3:
                    this_output_file_name = os.path.join(target_dir_img_small, 'sup', str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+"_sup_1.png"))
                elif this_img == 4:
                    this_output_file_name = os.path.join(target_dir_img_small, 'sup', str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+"_sup_2.png"))
                elif this_img == 5:
                    this_output_file_name = os.path.join(target_dir_img_small, 'sup', str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+"_sup_3.png"))
                elif this_img == 6:
                    this_output_file_name = os.path.join(target_dir_img_small, 'sup', str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+"_sup_4.png"))
                elif this_img == 7:
                    this_output_file_name = os.path.join(target_dir_img_small, 'sup', str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+"_sup_5.png"))
                else:
                    this_output_file_name = os.path.join(target_dir_img_small, 'sup', str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+"_others.png"))
                
                if (min(r*crop_size+crop_size, img.shape[0]) - r*crop_size <= 0) or (min(c*crop_size+crop_size, img.shape[1]) - c*crop_size <= 0):
                    continue
                
                r_0 = r*crop_size
                r_1 = min(r*crop_size+crop_size, img.shape[0])
                c_0 = c*crop_size
                c_1 = min(c*crop_size+crop_size, img.shape[1])

                print(r, c, r_0, r_1, c_0, c_1)
                if True:
                    if r_1-r_0 < crop_size or c_1-c_0 < crop_size:
                        if r_1-r_0 < crop_size:
                            img_concat_temp = np.concatenate([img[r_0:r_1, c_0:c_1], empty_grid[0:crop_size-(r_1-r_0), 0:(c_1-c_0)]], axis=0)
                            print(img[r_0:r_1, c_0:c_1].shape, empty_grid[0:crop_size-(r_1-r_0), 0:(c_1-c_0)].shape, img_concat_temp.shape)
                        else:
                            img_concat_temp = np.copy(img[r_0:r_1, c_0:c_1]).astype(float)
                            print(img[r_0:r_1, c_0:c_1].shape, img_concat_temp.shape)
                        if c_1-c_0 < crop_size:
                            img_concat = np.concatenate([img_concat_temp[:, 0:(c_1-c_0)], empty_grid[:, 0:crop_size-(c_1-c_0)]], axis=1)
                            print(img_concat_temp[:, :(c_1-c_0)].shape, empty_grid[:, 0:crop_size-(c_1-c_0)].shape, img_concat.shape)
                        else:
                            img_concat = np.copy(img_concat_temp).astype(float)
                            print(img_concat_temp.shape, img_concat.shape)
                    else:
                        img_concat = np.copy(img[r_0:r_1, c_0:c_1]).astype(float)
                
                
                if this_img == 1 or this_img == 8:
                    img_concat[img_concat > 0] = 1
                    #img_concat[img_concat > 0] = 255
                elif this_img == 0 or this_img == 2 or this_img == 5: # or this_img == 6 or this_img == 7
                    img_concat[img_concat > 0] = 255

                # print(np.unique(img_concat), np.unique(img_concat1))
                
                if this_img == 1 or this_img == 8:
                    cv2.imwrite(this_output_file_name, img_concat)

                    img_concat[img_concat > 0] = 255
                    if this_img == 1:
                        this_output_file_name = os.path.join(target_dir_mask_small, str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+"_mask_vis.png"))
                        cv2.imwrite(this_output_file_name, img_concat)
                else:
                    cv2.imwrite(this_output_file_name, img_concat)
    
    return map_id, legend_id, (math.ceil(img.shape[0]/crop_size) * math.ceil(img.shape[1]/crop_size))
