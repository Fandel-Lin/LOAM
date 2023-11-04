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

def preprocessing_for_unet_worker_single_numpy(map_id, legend_id, this_map_name, this_legend_name, data_dir0, data_dir1, data_dir2, target_dir_img, target_dir_mask, target_dir_img_small, target_dir_mask_small):
    source_path_0 = os.path.join(data_dir0, this_map_name+'/'+this_map_name+'_'+this_legend_name+"_v7.png")
    target_path_0 = os.path.join(target_dir_img, str(this_map_name+'_'+this_legend_name+".png"))
    source_path_1 = os.path.join(data_dir1, this_map_name+'_'+this_legend_name+".tif")
    target_path_1 = os.path.join(target_dir_mask, str(this_map_name+'_'+this_legend_name+"_mask.png"))

    source_path_2 = os.path.join(data_dir0, this_map_name+'/'+this_map_name+'_'+this_legend_name+"_v2.png")
    target_path_2 = os.path.join(target_dir_img, str(this_map_name+'_'+this_legend_name+"_sup_1.png"))
    source_path_3 = os.path.join(data_dir2, this_map_name+'_crop_blackregion.png')
    target_path_3 = os.path.join(target_dir_img, str(this_map_name+'_'+this_legend_name+"_sup_2.png"))
    source_path_4 = os.path.join(data_dir2, this_map_name+'_crop.png')
    target_path_4 = os.path.join(target_dir_img, str(this_map_name+'_'+this_legend_name+"_sup_3.png"))
    source_path_5 = os.path.join(data_dir2, this_map_name+'_crop.png')
    target_path_5 = os.path.join(target_dir_img, str(this_map_name+'_'+this_legend_name+"_sup_4.png"))
    #source_path_ext = os.path.join(data_dir2, this_map_name+'_expected_crop_region.png')

    
    shutil.copyfile(source_path_0, target_path_0)
    shutil.copyfile(source_path_1, target_path_1)
    '''
    shutil.copyfile(source_path_2, target_path_2)
    shutil.copyfile(source_path_3, target_path_3)
    shutil.copyfile(source_path_4, target_path_4)
    shutil.copyfile(source_path_5, target_path_5)
    '''

    figure_info = []
    figure_info.append([source_path_0])
    figure_info.append([source_path_1])
    figure_info.append([source_path_2])
    figure_info.append([source_path_3])
    figure_info.append([source_path_4])
    figure_info.append([source_path_5])

    img_set = []
    empty_grid0 = np.zeros((1024, 1024), dtype='uint8')#.astype(float)
    empty_grid1 = np.zeros((1024, 1024, 3), dtype='uint8')#.astype(float)

    for this_img in range(0, len(figure_info)):
        img = cv2.imread(figure_info[this_img][0])
        if this_img != 5:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #empty_grid = np.zeros((1024, 1024), dtype='uint8').astype(float)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #empty_grid = np.zeros((1024, 1024, 3), dtype='uint8').astype(float)
        
        if this_img != 5 and np.unique(img).shape[0] > 2:
            print('1??????', np.unique(img), figure_info[this_img][0])
        
        img_set.append(img)
    img = cv2.imread(figure_info[0][0])
        

    for r in range(0,math.ceil(img.shape[0]/1024)):
        for c in range(0,math.ceil(img.shape[1]/1024)):
            #this_output_file_name_0 = os.path.join(target_dir_img_small, str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+".npy"))
            this_output_file_name = os.path.join(target_dir_img_small, str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+".txt"))
            this_output_file_name_mask = os.path.join(target_dir_mask_small, str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+"_mask.png"))
            '''
            if this_img == 0:
                this_output_file_name = os.path.join(target_dir_img_small, str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+".png"))
            elif this_img == 1:
                this_output_file_name = os.path.join(target_dir_mask_small, str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+"_mask.png"))
            elif this_img == 2:
                this_output_file_name = os.path.join(target_dir_img_small, str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+"_sup_1.png"))
            elif this_img == 3:
                this_output_file_name = os.path.join(target_dir_img_small, str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+"_sup_2.png"))
            elif this_img == 4:
                this_output_file_name = os.path.join(target_dir_img_small, str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+"_sup_3.png"))
            elif this_img == 5:
                this_output_file_name = os.path.join(target_dir_img_small, str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+"_sup_4.png"))  
            else:
                this_output_file_name = os.path.join(target_dir_img_small, str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+"_others.png"))
            '''

            img_combined = np.zeros((1024, 1024, 7), dtype='uint8').astype(float)
            img_mask = np.zeros((1024, 1024), dtype='uint8').astype(float)

            
            if (min(r*1024+1024, img.shape[0]) - r*1024 <= 0) or (min(c*1024+1024, img.shape[1]) - c*1024 <= 0):
                continue
            
            r_0 = r*1024
            r_1 = min(r*1024+1024, img.shape[0])
            c_0 = c*1024
            c_1 = min(c*1024+1024, img.shape[1])

            print(r, c, r_0, r_1, c_0, c_1)
            

            for this_img in range(0, len(figure_info)):
                # img = img_set[this_img]

                if this_img != 5:
                    if r_1-r_0 < 1024 or c_1-c_0 < 1024:
                        if r_1-r_0 < 1024:
                            img_concat_temp = np.concatenate([img_set[this_img][r_0:r_1, c_0:c_1], empty_grid0[0:1024-(r_1-r_0), 0:(c_1-c_0)]], axis=0)
                            print(img_set[this_img][r_0:r_1, c_0:c_1].shape, empty_grid0[0:1024-(r_1-r_0), 0:(c_1-c_0)].shape, img_concat_temp.shape)
                        else:
                            img_concat_temp = np.copy(img_set[this_img][r_0:r_1, c_0:c_1]).astype(float)
                            print(img_set[this_img][r_0:r_1, c_0:c_1].shape, img_concat_temp.shape)
                        if c_1-c_0 < 1024:
                            img_concat = np.concatenate([img_concat_temp[:, 0:(c_1-c_0)], empty_grid0[:, 0:1024-(c_1-c_0)]], axis=1)
                            print(img_concat_temp[:, :(c_1-c_0)].shape, empty_grid0[:, 0:1024-(c_1-c_0)].shape, img_concat.shape)
                        else:
                            img_concat = np.copy(img_concat_temp).astype(float)
                            print(img_concat_temp.shape, img_concat.shape)
                    else:
                        img_concat = np.copy(img_set[this_img][r_0:r_1, c_0:c_1]).astype(float)
                    
                    if this_img == 0:
                        img_concat[img_concat > 0] = 255
                        img_combined[:, :, 0] = np.copy(img_concat)
                    elif this_img == 1:
                        img_concat[img_concat > 0] = 1
                        img_mask = np.copy(img_concat)
                    elif this_img == 2:
                        img_concat[img_concat > 0] = 255
                        img_combined[:, :, 1] = np.copy(img_concat)
                    elif this_img == 3:
                        img_concat[img_concat > 0] = 255
                        img_combined[:, :, 2] = np.copy(img_concat)
                    elif this_img == 4:
                        img_combined[:, :, 3] = np.copy(img_concat)
                    
                    if this_img != 5 and np.unique(img_concat).shape[0] > 2:
                        print('2??????', np.unique(img_concat), this_output_file_name)
                else:
                    if r_1-r_0 < 1024 or c_1-c_0 < 1024:
                        if r_1-r_0 < 1024:
                            img_concat_temp = np.concatenate([img_set[this_img][r_0:r_1, c_0:c_1, :], empty_grid1[0:1024-(r_1-r_0), 0:(c_1-c_0), :]], axis=0)
                            print(img_set[this_img][r_0:r_1, c_0:c_1, :].shape, empty_grid1[0:1024-(r_1-r_0), 0:(c_1-c_0), :].shape, img_concat_temp.shape)
                        else:
                            img_concat_temp = np.copy(img_set[this_img][r_0:r_1, c_0:c_1, :]).astype(float)
                            print(img_set[this_img][r_0:r_1, c_0:c_1, :].shape, img_concat_temp.shape)
                        if c_1-c_0 < 1024:
                            img_concat = np.concatenate([img_concat_temp[:, 0:(c_1-c_0), :], empty_grid1[:, 0:1024-(c_1-c_0), :]], axis=1)
                            print(img_concat_temp[:, :(c_1-c_0), :].shape, empty_grid1[:, 0:1024-(c_1-c_0), :].shape, img_concat.shape)
                        else:
                            img_concat = np.copy(img_concat_temp).astype(float)
                            print(img_concat_temp.shape, img_concat.shape)
                    else:
                        img_concat = np.copy(img_set[this_img][r_0:r_1, c_0:c_1, :]).astype(float)

                    img_concat0 = img_concat.astype(dtype='uint8')
                    img_concat = cv2.cvtColor(img_concat0, cv2.COLOR_RGB2BGR)
                    img_combined[:, :, 4:7] = np.copy(img_concat[:, :, :])
            
            #cv2.imwrite(this_output_file_name, img_concat)
            cv2.imwrite(this_output_file_name_mask, img_mask)
            with open(this_output_file_name, 'wb') as fff:
                np.save(fff, img_combined, allow_pickle=True)
            #img_combined_reshape = img_combined.reshape(img_combined.shape[0], -1)
            #np.savetxt(this_output_file_name, img_combined_reshape, fmt='%i', delimiter=',')
    
    return map_id, legend_id, (math.ceil(img.shape[0]/1024) * math.ceil(img.shape[1]/1024))



def preprocessing_for_unet_worker_multiple_image(map_id, legend_id, this_map_name, this_legend_name, data_dir0, data_dir1, data_dir2, data_dir3, target_dir_img, target_dir_mask, target_dir_img_small, target_dir_mask_small):
    source_path_0 = os.path.join(data_dir0, this_map_name+'/'+this_map_name+'_'+this_legend_name+"_v7.png") # input segmentation (input polygon candidate)
    target_path_0 = os.path.join(target_dir_img, str(this_map_name+'_'+this_legend_name+".png"))
    source_path_1 = os.path.join(data_dir1, this_map_name+'_'+this_legend_name+".tif") # groundtruth
    target_path_1 = os.path.join(target_dir_mask, str(this_map_name+'_'+this_legend_name+"_mask.png"))

    '''
    source_path_2 = os.path.join(data_dir0, this_map_name+'/'+this_map_name+'_'+this_legend_name+"_v2.png") # preliminary polygon candidate
    target_path_2 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_1.png"))
    source_path_3 = os.path.join(data_dir2, this_map_name+'_crop_blackregion.png') # black lines
    target_path_3 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_2.png"))
    source_path_4 = os.path.join(data_dir2, this_map_name+'_crop.png') # cropped map (read into binary)
    target_path_4 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_3.png"))
    source_path_5 = os.path.join(data_dir2, this_map_name+'_crop.png') # cropped map (read into color)
    target_path_5 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_4.png"))
    '''

    '''
    source_path_2 = os.path.join(data_dir0, this_map_name+'/'+this_map_name+'_'+this_legend_name+"_v4v.png") # dynamic-threshold polygon candidate
    target_path_2 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_0.png"))
    source_path_3 = os.path.join(data_dir0, this_map_name+'/'+this_map_name+'_'+this_legend_name+"_v5v.png") # dynamic-threshold polygon candidate
    target_path_3 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_1.png"))
    source_path_4 = os.path.join(data_dir2, this_map_name+'_crop_blackregion.png') # black lines
    target_path_4 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_2.png"))
    source_path_5 = os.path.join(data_dir0, this_map_name+'/'+this_map_name+'_'+this_legend_name+'_c0_x.png') # rgb color difference
    target_path_5 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_3.png"))
    source_path_6 = os.path.join(data_dir0, this_map_name+'/'+this_map_name+'_'+this_legend_name+'_c1_x.png') # hsv color difference
    target_path_6 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_4.png"))
    #source_path_7 = os.path.join(data_dir2, this_map_name+'_crop.png') # cropped color map
    source_path_7 = os.path.join(data_dir0, this_map_name+'/'+this_map_name+'_'+this_legend_name+'_c1_0.png') # hsv color difference
    target_path_7 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_5.png"))
    '''

    source_path_2 = os.path.join(data_dir0, this_map_name+'/'+this_map_name+'_'+this_legend_name+"_v4v.png") # dynamic-threshold polygon candidate
    target_path_2 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_0.png"))
    source_path_3 = os.path.join(data_dir0, this_map_name+'/'+this_map_name+'_'+this_legend_name+'_c0_x.png') # rgb color difference
    target_path_3 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_1.png"))
    source_path_4 = os.path.join(data_dir0, this_map_name+'/'+this_map_name+'_'+this_legend_name+'_c1_x.png') # hsv color difference
    target_path_4 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_2.png"))
    source_path_5 = os.path.join(data_dir2, this_map_name+'_crop_blackregion.png') # black basemap
    target_path_5 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_3.png"))
    source_path_6 = os.path.join(data_dir3, 'Extraction', str(this_map_name+'_BoundaryAsPolygon_v0.png')) # gray basemap
    target_path_6 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_4.png"))
    source_path_7 = os.path.join(data_dir3, 'Extraction', str(this_map_name+'_BoundaryAsPolygon_v2_1.png')) # identified boundary
    target_path_7 = os.path.join(target_dir_img, 'sup', str(this_map_name+'_'+this_legend_name+"_sup_5.png"))

    source_path_8 = os.path.join(data_dir3, 'Groundtruth', str(this_map_name+'_boundary_identified.png')) # identified boundary
    target_path_8 = os.path.join(target_dir_mask, 'sup', str(this_map_name+'_'+this_legend_name+"_mask_sup.png"))


    # groundtruth boundary (output)



    #source_path_ext = os.path.join(data_dir2, this_map_name+'_expected_crop_region.png')


    shutil.copyfile(source_path_0, target_path_0)
    shutil.copyfile(source_path_1, target_path_1)
    shutil.copyfile(source_path_8, target_path_8)
    '''
    shutil.copyfile(source_path_2, target_path_2)
    shutil.copyfile(source_path_3, target_path_3)
    shutil.copyfile(source_path_4, target_path_4)
    shutil.copyfile(source_path_5, target_path_5)
    shutil.copyfile(source_path_6, target_path_6)
    '''

    figure_info = []
    figure_info.append([source_path_0])
    figure_info.append([source_path_1])
    figure_info.append([source_path_2])
    figure_info.append([source_path_3])
    figure_info.append([source_path_4])
    figure_info.append([source_path_5])
    figure_info.append([source_path_6])
    figure_info.append([source_path_7])
    figure_info.append([source_path_8])


    for this_img in range(0, len(figure_info)):
        img = cv2.imread(figure_info[this_img][0])
        '''
        if this_img == 5 or this_img == 6 or this_img == 7:
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #empty_grid = np.zeros((1024, 1024, 3), dtype='uint8').astype(float)
        else:
        '''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        empty_grid = np.zeros((1024, 1024), dtype='uint8').astype(float)
        
        #if this_img == 4 or this_img == 5:
            #img_ext = cv2.imread(source_path_ext)
            #img = cv2.bitwise_and(img, img, mask=img_ext)
        
        '''
        if (this_img != 5 and this_img != 6 and this_img != 7) and np.unique(img).shape[0] > 2:
            print('1??????', np.unique(img), figure_info[this_img][0])
        '''

        for r in range(0,math.ceil(img.shape[0]/1024)):
            for c in range(0,math.ceil(img.shape[1]/1024)):
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
                elif this_img == 8:
                    this_output_file_name = os.path.join(target_dir_mask_small, 'sup', str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+"_mask_sup.png"))
                else:
                    this_output_file_name = os.path.join(target_dir_img_small, 'sup', str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+"_others.png"))
                
                if (min(r*1024+1024, img.shape[0]) - r*1024 <= 0) or (min(c*1024+1024, img.shape[1]) - c*1024 <= 0):
                    continue
                
                r_0 = r*1024
                r_1 = min(r*1024+1024, img.shape[0])
                c_0 = c*1024
                c_1 = min(c*1024+1024, img.shape[1])

                print(r, c, r_0, r_1, c_0, c_1)
                '''
                if this_img == 5 or this_img == 6 or this_img == 7:
                    if r_1-r_0 < 1024 or c_1-c_0 < 1024:
                        if r_1-r_0 < 1024:
                            img_concat_temp = np.concatenate([img[r_0:r_1, c_0:c_1, :], empty_grid[0:1024-(r_1-r_0), 0:(c_1-c_0), :]], axis=0)
                            print(img[r_0:r_1, c_0:c_1, :].shape, empty_grid[0:1024-(r_1-r_0), 0:(c_1-c_0), :].shape, img_concat_temp.shape)
                        else:
                            img_concat_temp = np.copy(img[r_0:r_1, c_0:c_1, :]).astype(float)
                            print(img[r_0:r_1, c_0:c_1, :].shape, img_concat_temp.shape)
                        if c_1-c_0 < 1024:
                            img_concat = np.concatenate([img_concat_temp[:, 0:(c_1-c_0), :], empty_grid[:, 0:1024-(c_1-c_0), :]], axis=1)
                            print(img_concat_temp[:, :(c_1-c_0), :].shape, empty_grid[:, 0:1024-(c_1-c_0), :].shape, img_concat.shape)
                        else:
                            img_concat = np.copy(img_concat_temp).astype(float)
                            print(img_concat_temp.shape, img_concat.shape)
                    else:
                        img_concat = np.copy(img[r_0:r_1, c_0:c_1, :]).astype(float)
                else:
                '''
                if True:
                    if r_1-r_0 < 1024 or c_1-c_0 < 1024:
                        if r_1-r_0 < 1024:
                            img_concat_temp = np.concatenate([img[r_0:r_1, c_0:c_1], empty_grid[0:1024-(r_1-r_0), 0:(c_1-c_0)]], axis=0)
                            print(img[r_0:r_1, c_0:c_1].shape, empty_grid[0:1024-(r_1-r_0), 0:(c_1-c_0)].shape, img_concat_temp.shape)
                        else:
                            img_concat_temp = np.copy(img[r_0:r_1, c_0:c_1]).astype(float)
                            print(img[r_0:r_1, c_0:c_1].shape, img_concat_temp.shape)
                        if c_1-c_0 < 1024:
                            img_concat = np.concatenate([img_concat_temp[:, 0:(c_1-c_0)], empty_grid[:, 0:1024-(c_1-c_0)]], axis=1)
                            print(img_concat_temp[:, :(c_1-c_0)].shape, empty_grid[:, 0:1024-(c_1-c_0)].shape, img_concat.shape)
                        else:
                            img_concat = np.copy(img_concat_temp).astype(float)
                            print(img_concat_temp.shape, img_concat.shape)
                    else:
                        img_concat = np.copy(img[r_0:r_1, c_0:c_1]).astype(float)
                
                
                if this_img == 1 or this_img == 8:
                    img_concat[img_concat > 0] = 1
                    #img_concat[img_concat > 0] = 255
                elif this_img == 0 or this_img == 2 or this_img == 5 or this_img == 6 or this_img == 7:
                    img_concat[img_concat > 0] = 255
                '''
                elif (this_img == 5 or this_img == 6 or this_img == 7):
                    img_concat0 = img_concat.astype(dtype='uint8')
                    img_concat = cv2.cvtColor(img_concat0, cv2.COLOR_RGB2BGR)
                elif this_img != 3:
                    img_concat[img_concat > 0] = 255
                '''

                # print(np.unique(img_concat), np.unique(img_concat1))

                '''
                if (this_img != 5 and this_img != 6 and this_img != 7) and np.unique(img_concat).shape[0] > 2:
                    print('2??????', np.unique(img_concat), this_output_file_name)
                '''
                
                if this_img == 1 or this_img == 8:
                    cv2.imwrite(this_output_file_name, img_concat)

                    img_concat[img_concat > 0] = 255
                    if this_img == 1:
                        this_output_file_name = os.path.join(target_dir_mask_small, str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+"_mask_vis.png"))
                        cv2.imwrite(this_output_file_name, img_concat)
                    elif this_img == 8:
                        this_output_file_name = os.path.join(target_dir_mask_small, 'sup', str(this_map_name+'_'+this_legend_name+"_"+str(r)+"_"+str(c)+"_mask_vis.png"))
                        cv2.imwrite(this_output_file_name, img_concat)
                else:
                    cv2.imwrite(this_output_file_name, img_concat)
    
    return map_id, legend_id, (math.ceil(img.shape[0]/1024) * math.ceil(img.shape[1]/1024))
