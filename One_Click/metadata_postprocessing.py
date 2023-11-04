
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

import postprocessing_workers.postprocessing_for_bitmap_worker as postprocessing_for_bitmap_worker


import multiprocessing
#print(multiprocessing.cpu_count())
PROCESSES = 8


data_dir='Data/testing' # set path to the targeted dataset
solution_dir='Solution_1102' # set path to metadata-preprocessing output
data_dir_groundtruth='Data/testing_groundtruth' # set path to the groundtruth of the targeted dataset

target_dir_img = 'LOAM\data\cma\imgs'
target_dir_mask = 'LOAM\data\cma\masks'

target_dir_img_small = 'LOAM\data\cma_small\imgs'
target_dir_mask_small = 'LOAM\data\cma_small\masks'




def multiprocessing_setting():
    global PROCESSES

    multiprocessing.set_start_method('spawn', True)
    if PROCESSES > multiprocessing.cpu_count():
        PROCESSES = (int)(multiprocessing.cpu_count()/2)


def dir_setting():
    if not os.path.exists(os.path.join('LOAM', 'data')):
        os.makedirs(os.path.join('LOAM', 'data'))
    if not os.path.exists(os.path.join('LOAM', 'data', 'cma')):
        os.makedirs(os.path.join('LOAM', 'data', 'cma'))
    if not os.path.exists(os.path.join('LOAM', 'data', 'cma_small')):
        os.makedirs(os.path.join('LOAM', 'data', 'cma_small'))

    if not os.path.exists(target_dir_img):
        os.makedirs(target_dir_img)
        os.makedirs(os.path.join(target_dir_img, 'sup'))
    if not os.path.exists(target_dir_mask):
        os.makedirs(target_dir_mask)
        os.makedirs(os.path.join(target_dir_mask, 'sup'))
    if not os.path.exists(target_dir_img_small):
        os.makedirs(target_dir_img_small)
        os.makedirs(os.path.join(target_dir_img_small, 'sup'))
    if not os.path.exists(target_dir_mask_small):
        os.makedirs(target_dir_mask_small)
        os.makedirs(os.path.join(target_dir_mask_small, 'sup'))

    shutil.copyfile('targeted_map.csv', 'LOAM/targeted_map.csv')
    shutil.copyfile(solution_dir+'/intermediate9/auxiliary_info.csv', 'LOAM/data/auxiliary_info.csv')

    print('Set up directories in "LOAM/data"')


def file_summary():
    global candidate_map_name_for_polygon
    global candidate_legend_name_for_polygon


    #data_dir='Data/validation'
    candidate_map_name_for_polygon = []
    candidate_legend_name_for_polygon = []
    total_poly_counter = 0
    for file_name in os.listdir(data_dir):
        if '.json' in file_name:
            filename=file_name.replace('.json', '.tif')
            map_name = file_name.replace('.json', '')
            #print('Working on map:', file_name)
            file_path=os.path.join(data_dir, filename)
            test_json=file_path.replace('.tif', '.json')
            
            legend_name_placeholder = []
            poly_counter = 0

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
                poly_counter = poly_counter+1
                legend_name_placeholder.append(names)
            
            if poly_counter > 0:
                candidate_map_name_for_polygon.append(map_name)
                candidate_legend_name_for_polygon.append(legend_name_placeholder)

                total_poly_counter = total_poly_counter + poly_counter
                print(poly_counter, '\t', filename)
    print(total_poly_counter)



def worker_postprocessing(crop_size):
    #data_dir = 'Data/validation'
    #data_dir0 = 'validation_extraction'
    data_dir0 = solution_dir + str('/intermediate7(2)')
    data_dir1 = data_dir_groundtruth

    data_dir2 = solution_dir + str('/intermediate7')
    data_dir3 = solution_dir + str('/intermediate5')
    data_dir4 = solution_dir + str('/intermediate8(2)')


    info_set = []

    ### For polygon extraction
    for map_id in range(0, len(candidate_map_name_for_polygon)):
        runningtime_start=datetime.now()
        grid_counter = 0


        target_map = candidate_map_name_for_polygon[map_id]+'.tif'
        file_map = os.path.join(data_dir, target_map)
        file_json = os.path.join(data_dir, target_map.replace('.tif','.json'))
        
        legend_for_multiprocessing = []
        for legend_id in range(0, len(candidate_legend_name_for_polygon[map_id])):
            target_legend = candidate_map_name_for_polygon[map_id]+'_'+candidate_legend_name_for_polygon[map_id][legend_id]+".tif" # groundtruth
            target_legend_v1 = candidate_map_name_for_polygon[map_id]+'/'+candidate_map_name_for_polygon[map_id]+'_'+candidate_legend_name_for_polygon[map_id][legend_id]+"_v7.png" # extraction
            #target_legend_v2 = candidate_map_name_for_polygon[map_id]+'/'+candidate_map_name_for_polygon[map_id]+'_'+candidate_legend_name_for_polygon[map_id][legend_id]+"_v2.png"
                        
            file_extraction = os.path.join(data_dir0, target_legend_v1)
            file_groundtruth = os.path.join(data_dir1, target_legend)
            
            if os.path.isfile(file_groundtruth) == False:
                print('not provided... ', file_groundtruth)
                continue
            
            if os.path.isfile(file_extraction) == False:
                print('not extracted... ', file_extraction)
                continue

            legend_for_multiprocessing.append(legend_id)
        print(map_id, len(legend_for_multiprocessing))
        
        
        with multiprocessing.Pool(PROCESSES) as pool:
            callback = pool.starmap_async(postprocessing_for_bitmap_worker.postprocessing_for_bitmap_worker_multiple_image, [(map_id, this_legend_id, candidate_map_name_for_polygon[map_id], candidate_legend_name_for_polygon[map_id][this_legend_id], data_dir0, data_dir1, data_dir2, data_dir3, data_dir4, target_dir_img, target_dir_mask, target_dir_img_small, target_dir_mask_small, crop_size, ) for this_legend_id in legend_for_multiprocessing])
            multiprocessing_results = callback.get()

            for return_map_id, return_legend_id, number_of_grids in multiprocessing_results:
                grid_counter = grid_counter + number_of_grids

        runningtime_end = datetime.now()-runningtime_start

        if os.path.isfile('LOAM/data/'+'running_time_record_v1.csv') == False:
            with open('LOAM/data/'+'running_time_record_v1.csv','w') as fd:
                fd.write('Map_Id,Map_Name,Legend_Count,RunningTime\n')
                fd.close()
        if os.path.isfile('LOAM/data/'+'generated_grids_record_v1.csv') == False:
            with open('LOAM/data/'+'generated_grids_record_v1.csv','w') as fd:
                fd.write('Map_Id,Map_Name,Legend_Count,GeneratedGrids\n')
                fd.close()

        with open('LOAM/data/'+'running_time_record_v1.csv','a') as fd:
            fd.write(str(map_id)+','+candidate_map_name_for_polygon[map_id]+','+str(len(legend_for_multiprocessing))+','+str(runningtime_end)+'\n')
            fd.close()
        with open('LOAM/data/'+'generated_grids_record_v1.csv','a') as fd:
            fd.write(str(map_id)+','+candidate_map_name_for_polygon[map_id]+','+str(len(legend_for_multiprocessing))+','+str(grid_counter)+'\n')
            fd.close()

    # 59m 11.4s




def run(crop_size):
    multiprocessing_setting()
    dir_setting()
    file_summary()
    worker_postprocessing(crop_size)


def metadata_postprocessing(
        input_data_dir = 'Data/validation',
        input_solution_dir = 'Solution_1102',
        input_data_dir_groundtruth = 'Data/validation_groundtruth',
        crop_size = 256
):
    global data_dir
    global solution_dir
    global data_dir_groundtruth

    data_dir = input_data_dir
    solution_dir = input_solution_dir
    data_dir_groundtruth = input_data_dir_groundtruth

    run(crop_size)



def main():
    global data_dir
    global solution_dir
    global data_dir_groundtruth

    data_dir = args.data_dir
    solution_dir = args.solution_dir
    data_dir_groundtruth = args.data_dir_groundtruth
    crop_size = int(args.crop_size)

    run(crop_size)


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Data/validation')
    parser.add_argument('--solution_dir', type=str, default='Solution_1102')
    parser.add_argument('--data_dir_groundtruth', type=str, default='Data/validation_groundtruth')
    parser.add_argument('--crop_size', type=str, default='256')

    args = parser.parse_args()
    
    #print(f"Processing output for: {args.result_name}")
    main()



