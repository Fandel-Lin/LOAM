from evaluation_matrix_cma import feature_f_score

def validation_evaluation_worker(info_id, info_subset):
    # info_subset = info_set[info_id]
    file_map = info_subset[0]
    file_json = info_subset[1]
    file_extraction00 = info_subset[2]
    file_groundtruth = info_subset[3]
    
    map_name = info_subset[4]
    legend_name = info_subset[5]

    try:
        precision_0, recall_0, f_score_0=feature_f_score(file_map, file_extraction00, file_groundtruth,
            legend_json_path=file_json, min_valid_range=None, difficult_weight=.7, color_range=4, set_false_as='hard', plot=False)
    except:
        precision_0 = 0.0
        recall_0 = 0.0
        f_score_0 = 0.0
    

    return map_name, legend_name, precision_0, recall_0, f_score_0
