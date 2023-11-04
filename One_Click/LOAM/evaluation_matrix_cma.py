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


# ==================   NEW MEMORY OPTIMIZED CODE  ================== 

"""
Changelog:
1. Uses sparse arrays
2. min_valid_range can be both float; fraction of the actual diagonal distance or, int; distance in pixels.
    By default it is set to 10 pixels now.
3. Uses Pyvips to load the images faster.
4. Minor bug fixes for edge cases in pt and line scoring.

"""

def overlap_distance_calculate(mat_true, mat_pred, mat_overlap, min_valid_range=10):
    """
    mat_true, mat_pred: 2d matrices, with 0s and 1s only
    min_valid_range: the maximum distance in % if floating point or pixel if integer, of the largest size of the image (diagonal)
        between a predicted pixel vs. a true one that will be considered
        as valid to include in the scoring.
    calculate_distance: when True this will not only calculate overlapping pixels
        but also the distances between nearesttrue and predicted pixels
    """
    
    lowest_dist_pairs=[]
    points_done_pred=set()
    points_done_true=set()

    mat_overlap=mat_overlap.tocoo()
    for x_true, y_true in tqdm(zip(mat_overlap.row, mat_overlap.col)):
        lowest_dist_pairs.append((((x_true, y_true), (x_true, y_true)), 0.0)) 
        points_done_true.add((x_true, y_true))
        points_done_pred.add((x_true, y_true))
    print('len(lowest_dist_pairs) by overlapping only:', len(lowest_dist_pairs))
    
    diagonal_length=math.sqrt(math.pow(mat_true.shape[0], 2)+ math.pow(mat_true.shape[1], 2))
    if type(min_valid_range)!=int:
        min_valid_range=int((min_valid_range*diagonal_length)/100) # in pixels
    print('min_valid_range (in pixels):', min_valid_range)
    
    # create the distance kernel
    dist_kernel=np.zeros((min_valid_range*2+1 , min_valid_range*2+1))
    for i in range(min_valid_range*2+1):
        for j in range(min_valid_range*2+1):
            dist_kernel[i][j]=math.pow(i-min_valid_range, 2)+math.pow(j-min_valid_range, 2)    
        
    def nearest_pixels(x_true, y_true):
        result=[]
        # find all the points in pred withing min_valid_range rectangle
        mat_pred_inrange=mat_pred[
         max(x_true-min_valid_range, 0): min(x_true+min_valid_range, mat_true.shape[0]),
            max(y_true-min_valid_range, 0): min(y_true+min_valid_range, mat_true.shape[1])
        ].tocoo()
        for x_pred_shift, y_pred_shift in zip(mat_pred_inrange.row, mat_pred_inrange.col):
            y_pred=max(y_true-min_valid_range, 0)+y_pred_shift
            x_pred=max(x_true-min_valid_range, 0)+x_pred_shift
            if (x_pred, y_pred) in points_done_pred:
                continue
            # get eucledean distances
            dist_square=dist_kernel[x_pred_shift][y_pred_shift]
            # dist_square=math.pow(x_true-x_pred, 2)+math.pow(y_true-y_pred, 2)
            result.append((((x_true, y_true), (x_pred, y_pred)), dist_square))
        return result
    
    mat_true=mat_true.tocoo()
    candidates=[(x_true, y_true) for x_true, y_true in tqdm(zip(mat_true.row, mat_true.col)) if (x_true, y_true) not in points_done_true]
    distances=[nearest_pixels(x_true, y_true) for x_true, y_true in tqdm(candidates)]
    distances = [item for sublist in distances for item in sublist]

    # sort based on distances
    distances=sorted(distances, key=lambda x: x[1])

    # find the lowest distance pairs
    for ((x_true, y_true), (x_pred, y_pred)), distance in tqdm(distances):
        if ((x_true, y_true) in points_done_true) or ((x_pred, y_pred) in points_done_pred):
            # do not consider a taken point again
            continue
        # normalize all distances by diving by the diagonal length  
        lowest_dist_pairs.append((((x_true, y_true), (x_pred, y_pred)), math.sqrt(float(distance))/diagonal_length)) 
        points_done_true.add((x_true, y_true))
        points_done_pred.add((x_pred, y_pred))
    
    return lowest_dist_pairs


def detect_easy_pixels(map_image, binary_raster, legend_coor, plot=True, set_false_as='hard', color_range=4, baseline_raster=None):
    """
    map_image: the image array for the map image
    binary_raster: 2D array of any channel (out of 3 present) from the true binary raster image 
    legend_coor: coordinate for the legend feature, from the legend json file
    plot: plots different rasters
    set_false_as: when set to 'hard' the pixels that are not within the true polygon area will be considered hard
    """
    
    plt.rcParams["figure.figsize"] = (15,22)
        
    # detect pixels based on color of legend
    if legend_coor is not None:
        #print('running baseline...')
        pred_by_color=match_by_color(map_image, legend_coor, color_range=color_range, plot=plot)
        if plot:
            print('prediction based on color of legend:')
            plt.imshow(pred_by_color)
            plt.show()   
    
    binary_raster=sparse.csr_matrix(binary_raster) 
    pred_by_color=sparse.csr_matrix(pred_by_color) 
    pred_by_color=binary_raster.multiply(pred_by_color) # keep only the part within the true polygon
    if plot:
        print('color predictions within in the polygon range:')
        plt.imshow(pred_by_color)
        plt.show()
    
    if set_false_as=='hard':
        # the pixels that are not within the true polygon should are deemed hard pixels
        final_easy_pixels=pred_by_color
    else: # this will not work, as it is not assuming csr matrix!
        # the outside pixels will be deemed easy!
        final_easy_pixels=(1-binary_raster)|pred_by_color
    
    if plot:
        print('final easy pixels (merged):')
        plt.imshow(final_easy_pixels)
        plt.show()

    return final_easy_pixels

def match_by_color(img, legend_coor, color_range=4, plot=False):
    """
    img: the image array for the map image
    legend_coor: coordinate for the legend feature, from the legend json file
    """
    start=datetime.now()
    # get the legend coors and the predominant color
    (x_min, y_min), (x_max, y_max) = legend_coor            
    legend_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
    if plot:
        print('legend feature:')
        plt.imshow(legend_img)
        plt.show()
    # take the median of the colors to find the predominant color
    r=int(np.median(legend_img[:,:,0]))
    g=int(np.median(legend_img[:,:,1]))
    b=int(np.median(legend_img[:,:,2]))
    sought_color=[r, g, b]
    # capture the variations of legend color due to scanning errors
    lower = np.array(sought_color)-color_range
    lower[lower<0] = 0
    lower=tuple(lower.tolist())
    upper = np.array(sought_color)+color_range
    upper[upper>255] = 255
    upper=tuple(upper.tolist())
    #print('matching the color:', sought_color, 'with color range:', color_range, ', lower:', lower, 'upper:', upper)
    #print('time check 6:', datetime.now()-start)
    # create a mask to only preserve current legend color in the basemap
    start=datetime.now()
    pred_by_color = cv2.inRange(img, lower, upper)/255
    #print('time check 7:', datetime.now()-start)
    return pred_by_color

def feature_f_score(map_image_path, predicted_raster_path, true_raster_path, legend_json_path=None, min_valid_range=.1,
                      difficult_weight=.7, set_false_as='hard', plot=True, color_range=4):
    
    """
    map_image_path: path to the the actual map image
    predicted_raster_path: path to the the predicted binary raster image 
    true_raster_path: path to the the true binary raster image 
    legend_json_path: (only used for polygons) path to the json containing the coordinates for the corresponding legend (polygon) feature
    min_valid_range: (only used for points and lines) the maximum distance in % if floating point or pixel if integer, of the largest size of the image (diagonal)
        between a predicted pixel vs. a true one that will be considered
        as valid to include in the scoring.
    difficult_weight: (only used for polygons) float within [0, 1], weight for the difficlut pixels in the scores (only for polygins)
    set_false_as: (only used for polygons) when set to 'hard' the pixels that are not within the true polygon area will be considered hard
    """
    
    start=datetime.now()
    true_raster=pyvips.Image.new_from_file(true_raster_path, access="sequential").numpy()
    #true_raster=cv2.imread(true_raster_path)
    
    if len(true_raster.shape)==3:
        true_raster=true_raster[:,:,0]
    elif len(true_raster.shape)==2:
        true_raster=true_raster
    else:
        print('true_raster shape is not 3 or 2!!!')
        raise ValueError
        
    predicted_raster=pyvips.Image.new_from_file(predicted_raster_path, access="sequential").numpy()
    #predicted_raster=cv2.imread(predicted_raster_path)
    
    if len(predicted_raster.shape)==3:
        predicted_raster=predicted_raster[:,:,0]
    elif len(predicted_raster.shape)==2:
        predicted_raster=predicted_raster
    else:
        print('predicted_raster shape is not 3 or 2!!!')
        raise ValueError
    
    unique_values=np.unique(predicted_raster)
    for item in unique_values:
        if int(item) not in [0, 1, 255]:
            print('value in predicted raster:', int(item), 'not in permissible values:', [0, 1, 255])
            raise ValueError
            
    if len(unique_values)==1:
        return 0.0, 0.0, 0.0
    
    predicted_raster[predicted_raster==255] = 1
    #print('time check 0:', datetime.now()-start)
    
    extention=os.path.basename(true_raster_path).split('.')[-1]
    
    legend_feature=os.path.basename(true_raster_path).replace(os.path.basename(map_image_path).replace('.'+extention, '')+'_', '').replace('.'+extention, '')
    feature_type=legend_feature.split('_')[-1]
    #print('feature type:', feature_type)
    
    start=datetime.now()
    if plot or feature_type =='poly':
        img = pyvips.Image.new_from_file(map_image_path, access="sequential").numpy()
        #img=cv2.imread(map_image_path)
    #print('time check 1:', datetime.now()-start)
    
    # plot: overlay the true and predicted values on the map image
    if plot:
        im_copy=img.copy()
        for center in np.argwhere(predicted_raster==1):
            cv2.circle(im_copy, (center[1], center[0]), 1, (0,255,0), -1) # green
        print('Predicted raster overlayed on map image:')
        plt.rcParams["figure.figsize"] = (15,22)
        plt.imshow(im_copy)
        plt.show()
        im_copy=img.copy()
        for center in np.argwhere(true_raster==1):
            cv2.circle(im_copy, (center[1], center[0]), 1, (255,0,0), -1) # red
        print('True raster overlayed on map image:')
        plt.imshow(im_copy)
        plt.show()
    
    start=datetime.now()
    legend_coor=None
    #print('looking for legend_feature in the json:', legend_feature)
    if legend_json_path is not None:
        legends=json.loads(open(legend_json_path, 'r').read())
        for shape in legends['shapes']:
            if legend_feature ==shape['label']:
                legend_coor=shape['points']
        #print('legend_coor:', legend_coor)
    #print('time check 2:', datetime.now()-start)
    
    mat_true, mat_pred=true_raster, predicted_raster
    
    mat_pred=sparse.csr_matrix(mat_pred) 
    mat_true=sparse.csr_matrix(mat_true) 
    overlap=mat_true.multiply(mat_pred)
    
    if feature_type in ['line', 'pt']: # for point and lines
        lowest_dist_pairs=overlap_distance_calculate(mat_true, mat_pred, overlap,
                                                     min_valid_range=min_valid_range)
        print('len(lowest_dist_pairs):', len(lowest_dist_pairs))
        sum_of_similarities=sum([1-item[1] for item in lowest_dist_pairs])
        print('sum_of_similarities:', sum_of_similarities)        
        num_mat_pred=mat_pred.getnnz()
        num_mat_true=mat_true.getnnz()
        print('num all pixel pred:', num_mat_pred)
        print('num all pixel true:', num_mat_true)
        precision=sum_of_similarities/num_mat_pred if num_mat_pred!=0 else 0.0
        recall=sum_of_similarities/num_mat_true if num_mat_true!=0 else 0.0
        
    else: # for polygon
        
        start=datetime.now()
        #print('time check 3:', datetime.now()-start)

        if difficult_weight is not None:
            
            start=datetime.now()
            easy_pixels=detect_easy_pixels(img, true_raster, legend_coor=legend_coor, set_false_as=set_false_as, plot=plot,
                                                    color_range=color_range)
            #print('time check 4:', datetime.now()-start)
            
            easy_pixels=sparse.csr_matrix(easy_pixels) 
            easy_overlap=overlap.multiply(easy_pixels)
            num_overlap_easy=easy_overlap.getnnz()
            # print('num_overlap_easy:', num_overlap_easy)
            num_overlap_difficult=(overlap-easy_overlap).getnnz()
            # print('num_overlap_difficult:', num_overlap_difficult)
            points_from_overlap=(num_overlap_difficult*difficult_weight)+(num_overlap_easy*(1-difficult_weight))
            # print('points_from_overlap:', points_from_overlap)
            
            pred_easy=mat_pred.multiply(easy_pixels)
            num_mat_pred_easy=easy_pixels.getnnz()
            # print('num_mat_pred_easy:', num_mat_pred_easy)
            num_mat_pred_difficult=(mat_pred-pred_easy).getnnz()
            # print('num_mat_pred_difficult:', num_mat_pred_difficult)
            total_pred=(num_mat_pred_difficult*difficult_weight)+(num_mat_pred_easy*(1-difficult_weight))
            # print('total prediction points contended:', total_pred)
            precision=points_from_overlap/total_pred if total_pred!=0 else 0.0
            
            true_easy=mat_true.multiply(easy_pixels)
            num_mat_true_easy=true_easy.getnnz()
            # print('num_mat_true_easy:', num_mat_true_easy)
            num_mat_true_difficult= (mat_true-true_easy).getnnz()
            # print('num_mat_true_difficult:', num_mat_true_difficult)
            total_true=(num_mat_true_difficult*difficult_weight)+(num_mat_true_easy*(1-difficult_weight))
            # print('total true points to be had:', total_true)
            recall=points_from_overlap/total_true if total_true!=0 else 0.0
            #print('time check 5:', datetime.now()-start)
        
        else:
            num_overlap=overlap.getnnz()
            print('num_overlap:', num_overlap)
            num_mat_pred=mat_pred.getnnz()
            print('num_mat_pred:', num_mat_pred)
            num_mat_true=mat_true.getnnz()
            print('num_mat_true:', num_mat_true)

            precision=num_overlap/num_mat_pred if num_mat_pred!=0 else 0.0
            recall=num_overlap/num_mat_true if num_mat_true!=0 else 0.0
        
    
    # calculate f-score
    f_score=(2 * precision * recall) / (precision + recall) if precision+recall!=0 else 0.0

    return precision, recall, f_score


"""
Default settings for types of feature:

# point
precision, recall, f_score=feature_f_score(map_image_path, predicted_raster_path, true_raster_path,
legend_json_path=None, min_valid_range=10, difficult_weight=None, set_false_as=None, plot=False)

# line
precision, recall, f_score=feature_f_score(map_image_path, predicted_raster_path, true_raster_path, 
legend_json_path=None, min_valid_range=10, difficult_weight=None, set_false_as=None, plot=False)

# polygon
precision, recall, f_score=feature_f_score(map_image_path, predicted_raster_path, true_raster_path,
legend_json_path=legend_json_path, min_valid_range=None, difficult_weight=.7, color_range=4, set_false_as='hard', plot=False)
"""

