import numpy as np
import cv2

def extraction_step6_pre_update_worker(legend, this_current_result):
    ### v6
    # remove noisy white pixel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(this_current_result, cv2.MORPH_OPEN, kernel, iterations=1)
    return_image =cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY)[1]

    return legend, return_image, np.unique(this_current_result).shape[0]
