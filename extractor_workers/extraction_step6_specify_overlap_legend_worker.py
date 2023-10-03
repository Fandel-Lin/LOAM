import numpy as np
import cv2

def extraction_step6_specify_overlap_legend_worker(legend, legend_name, hsv_color_space_subset, local_hsv_space, global_hsv_space_subset, range_min, range_max):
    #has_similar_legend = False
    #overlapping_issue = False
    #similar_legend = []
    candidate_similar_legend_1 = []
    candidate_similar_legend_2 = []
    

    for counter_legend in range(range_min, range_max):
        if counter_legend == legend:
            continue

        # First, check huge simple overlapping
        ## a loose restriction for color space
        if np.sum(local_hsv_space[counter_legend][0][max(hsv_color_space_subset[0][0]-10, 0): min(1+hsv_color_space_subset[1][0]+10, 255)]) > 0:
            if np.sum(local_hsv_space[counter_legend][1][max(hsv_color_space_subset[0][1]-25, 0): min(1+hsv_color_space_subset[1][1]+25, 255)]) > 0:
                if np.sum(local_hsv_space[counter_legend][2][max(hsv_color_space_subset[0][2]-25, 0): min(1+hsv_color_space_subset[1][2]+25, 255)]) > 0:
                    candidate_similar_legend_1.append(counter_legend)
                    continue
                    '''
                    if np.mean(ans_category[legend]) > 0 and np.mean(ans_category[counter_legend]) > 0:
                        ans_overlap = cv2.bitwise_and(ans_category[legend], ans_category[counter_legend])

                        if (np.mean(ans_overlap) / np.mean(ans_category[legend])) > 0.66 and (np.mean(ans_overlap) / np.mean(ans_category[counter_legend])) > 0.66:
                            # if there are few overlaps in v6 extracted answer, than we don't need text detection
                            #print('we need to compare them')
                            #print('overlapping issue with large area: '+legend_name[legend]+' <-> '+legend_name[counter_legend])
                            similar_legend.append(counter_legend)
                            overlapping_issue = True
                            continue
                    '''


        # Second, check color overlapping
        ## no need to proceed if there is no overlap in h space for this legend
        if np.mean(global_hsv_space_subset[max(hsv_color_space_subset[0][0]-1, 0): min(1+hsv_color_space_subset[1][0]+1, 255)]) > 1:

            # only compare legends with the same first character
            if legend_name[legend][0] not in legend_name[counter_legend][0]:
                continue

            combined_hsv_space = local_hsv_space[legend] + local_hsv_space[counter_legend]

            if np.max(combined_hsv_space[0]) > 1 and np.max(combined_hsv_space[1]) > 1 and np.max(combined_hsv_space[2]) > 1:
                # if there are overlaps in all hsv spaces, than we probably need to proceed to text detection

                candidate_similar_legend_2.append(counter_legend)
                '''
                ans_overlap = cv2.bitwise_and(ans_category[legend], ans_category[counter_legend])
                if np.mean(ans_category[legend]) > 0 and np.mean(ans_category[counter_legend]) > 0:
                    if (np.mean(ans_overlap) / np.mean(ans_category[legend])) > 0.2 and (np.mean(ans_overlap) / np.mean(ans_category[counter_legend])) > 0.2:
                        # if there are few overlaps in v6 extracted answer, than we don't need text detection
                        #print('we need to compare them')
                        #print('overlapping issue with similar color: '+legend_name[legend]+' <-> '+legend_name[counter_legend])
                        similar_legend.append(counter_legend)
                        has_similar_legend = True
                '''
    return legend, candidate_similar_legend_1, candidate_similar_legend_2



def extraction_step6_specify_overlap_legend_worker_linux(legend, legend_name, ans_category, hsv_color_space_subset, local_hsv_space, global_hsv_space_subset, range_min, range_max):
    has_similar_legend = False
    overlapping_issue = False
    similar_legend = []
    

    for counter_legend in range(range_min, range_max):
        if counter_legend == legend:
            continue

        # First, check huge simple overlapping
        ## a loose restriction for color space
        if np.sum(local_hsv_space[counter_legend][0][max(hsv_color_space_subset[0][0]-10, 0): 1+hsv_color_space_subset[1][0]+10]) > 0:
            if np.sum(local_hsv_space[counter_legend][1][max(hsv_color_space_subset[0][1]-25, 0): 1+hsv_color_space_subset[1][1]+25]) > 0:
                if np.sum(local_hsv_space[counter_legend][2][max(hsv_color_space_subset[0][2]-25, 0): 1+hsv_color_space_subset[1][2]+25]) > 0:
                    if np.mean(ans_category[legend]) > 0 and np.mean(ans_category[counter_legend]) > 0:
                        ans_overlap = cv2.bitwise_and(ans_category[legend], ans_category[counter_legend])

                        if (np.mean(ans_overlap) / np.mean(ans_category[legend])) > 0.66 and (np.mean(ans_overlap) / np.mean(ans_category[counter_legend])) > 0.66:
                            # if there are few overlaps in v6 extracted answer, than we don't need text detection
                            #print('we need to compare them')
                            #print('overlapping issue with large area: '+legend_name[legend]+' <-> '+legend_name[counter_legend])
                            similar_legend.append(counter_legend)
                            overlapping_issue = True
                            continue


        # Second, check color overlapping
        ## no need to proceed if there is no overlap in h space for this legend
        if np.mean(global_hsv_space_subset[max(hsv_color_space_subset[0][0]-1, 0): 1+hsv_color_space_subset[1][0]+1]) > 1:

            # only compare legends with the same first character
            if legend_name[legend][0] not in legend_name[counter_legend][0]:
                continue

            combined_hsv_space = local_hsv_space[legend] + local_hsv_space[counter_legend]

            if np.max(combined_hsv_space[0]) > 1 and np.max(combined_hsv_space[1]) > 1 and np.max(combined_hsv_space[2]) > 1:
                # if there are overlaps in all hsv spaces, than we probably need to proceed to text detection

                ans_overlap = cv2.bitwise_and(ans_category[legend], ans_category[counter_legend])
                if np.mean(ans_category[legend]) > 0 and np.mean(ans_category[counter_legend]) > 0:
                    if (np.mean(ans_overlap) / np.mean(ans_category[legend])) > 0.2 and (np.mean(ans_overlap) / np.mean(ans_category[counter_legend])) > 0.2:
                        # if there are few overlaps in v6 extracted answer, than we don't need text detection
                        #print('we need to compare them')
                        #print('overlapping issue with similar color: '+legend_name[legend]+' <-> '+legend_name[counter_legend])
                        similar_legend.append(counter_legend)
                        has_similar_legend = True
    return legend, similar_legend

