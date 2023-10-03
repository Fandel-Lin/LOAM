import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
from scipy import ndimage
from collections import Counter

def extraction_step8_postprocessing_worker(legend_id, map_name, legend_full_name, solutiona_dir, floodfill_candidate):
    
    polygon_candidate_covered = cv2.imread(solutiona_dir+'intermediate7(2)/'+map_name+'/'+legend_full_name+'_v7.png')
    polygon_candidate_covered = cv2.cvtColor(polygon_candidate_covered, cv2.COLOR_BGR2GRAY)

    # flood fill background to find inner holes
    holes = polygon_candidate_covered.copy()
    cv2.floodFill(holes, None, (0, 0), 255)

    # invert holes mask, bitwise or with img fill in holes
    holes = cv2.bitwise_not(holes)
    valid_holes = cv2.bitwise_and(holes, floodfill_candidate)
    filled_holes = cv2.bitwise_or(polygon_candidate_covered, valid_holes)

    out_file_path0=solutiona_dir+'intermediate7(3)/'+map_name+'/'+legend_full_name+'_v8.png'
    cv2.imwrite(out_file_path0, filled_holes)

    #polygon_covered_v2 = cv2.bitwise_or(polygon_covered_v2, filled_holes)

    return legend_id, filled_holes