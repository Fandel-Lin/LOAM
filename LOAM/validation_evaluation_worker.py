from evaluation_matrix_cma import feature_f_score

def validation_evaluation_worker(info_id, info_subset):
    # info_subset = info_set[info_id]
    file_map = info_subset[0]
    file_json = info_subset[1]
    file_extraction00 = info_subset[2]
    file_extraction01 = info_subset[3]
    file_groundtruth = info_subset[4]
    
    map_name = info_subset[5]
    legend_name = info_subset[6]

    try:
        #precision_0, recall_0, f_score_0=feature_f_score(file_map, file_extraction00, file_groundtruth,
            #legend_json_path=file_json, min_valid_range=None, difficult_weight=None, color_range=4, set_false_as='hard', plot=False)
        precision_0, recall_0, f_score_0=feature_f_score(file_map, file_extraction00, file_groundtruth,
            legend_json_path=file_json, min_valid_range=None, difficult_weight=.7, color_range=4, set_false_as='hard', plot=False)
    except:
        precision_0 = 0.0
        recall_0 = 0.0
        f_score_0 = 0.0
    
    try:
        #precision_1, recall_1, f_score_1=feature_f_score(file_map, file_extraction01, file_groundtruth,
            #legend_json_path=file_json, min_valid_range=None, difficult_weight=None, color_range=4, set_false_as='hard', plot=False)
        precision_1, recall_1, f_score_1=feature_f_score(file_map, file_extraction01, file_groundtruth,
            legend_json_path=file_json, min_valid_range=None, difficult_weight=.7, color_range=4, set_false_as='hard', plot=False)
    except:
        precision_1 = 0.0
        recall_1 = 0.0
        f_score_1 = 0.0

    return map_name, legend_name, precision_0, recall_0, f_score_0, precision_1, recall_1, f_score_1
    '''
    file_extraction = info_subset[6]
    precision_1, recall_1, f_score_1=feature_f_score(file_map, file_extraction, file_groundtruth,
        legend_json_path=file_json, min_valid_range=None, difficult_weight=.7, color_range=4, set_false_as='hard', plot=False)
    
    file_extraction = info_subset[7]
    precision_2, recall_2, f_score_2=feature_f_score(file_map, file_extraction, file_groundtruth,
        legend_json_path=file_json, min_valid_range=None, difficult_weight=.7, color_range=4, set_false_as='hard', plot=False)

    file_extraction = info_subset[8]
    precision_3, recall_3, f_score_3=feature_f_score(file_map, file_extraction, file_groundtruth,
        legend_json_path=file_json, min_valid_range=None, difficult_weight=.7, color_range=4, set_false_as='hard', plot=False)

    file_extraction = info_subset[9]
    precision_4, recall_4, f_score_4=feature_f_score(file_map, file_extraction, file_groundtruth,
        legend_json_path=file_json, min_valid_range=None, difficult_weight=.7, color_range=4, set_false_as='hard', plot=False)

    file_extraction = info_subset[10]
    precision_5, recall_5, f_score_5=feature_f_score(file_map, file_extraction, file_groundtruth,
        legend_json_path=file_json, min_valid_range=None, difficult_weight=.7, color_range=4, set_false_as='hard', plot=False)

    file_extraction = info_subset[11]
    precision_6, recall_6, f_score_6=feature_f_score(file_map, file_extraction, file_groundtruth,
        legend_json_path=file_json, min_valid_range=None, difficult_weight=.7, color_range=4, set_false_as='hard', plot=False)

    
    
    #print(map_name, legend_name, precision, recall, f_score)
    return map_name, legend_name, precision_0, recall_0, f_score_0, precision_1, recall_1, f_score_1, precision_2, recall_2, f_score_2, precision_3, recall_3, f_score_3, precision_4, recall_4, f_score_4, precision_5, recall_5, f_score_5, precision_6, recall_6, f_score_6
    '''