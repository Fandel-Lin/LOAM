
import os
import metadata_preprocessing
import metadata_postprocessing


cwd_flag = False

def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def main():
    global data_dir
    global data_groundtruth_dir
    global solutiona_dir
    global targeted_map_list
    global map_preprocessing
    global generate_boundary_extraction
    global printing_auxiliary_information
    global preprocessing_recoloring

    global cwd_flag

    data_dir = args.data_dir
    data_groundtruth_dir = args.data_groundtruth_dir
    solutiona_dir = 'Solution_' + args.solutiona_dir + '/'
    targeted_map_list = args.targeted_map_list
    map_preprocessing = str_to_bool(args.map_preprocessing)
    generate_boundary_extraction = str_to_bool(args.generate_boundary_extraction)
    printing_auxiliary_information = str_to_bool(args.printing_auxiliary_information)
    preprocessing_recoloring = str_to_bool(args.preprocessing_recoloring)
    model_inference = str_to_bool(args.model_inference)

    if model_inference == False:
        metadata_preprocessing.metadata_preprocessing(
            input_data_dir = data_dir,
            input_data_boundary_dir = data_groundtruth_dir,
            input_solutiona_dir = solutiona_dir,
            input_targeted_map_list = targeted_map_list,
            input_map_preprocessing = map_preprocessing,
            input_generate_boundary_extraction = generate_boundary_extraction,
            input_printing_auxiliary_information = printing_auxiliary_information,
            input_preprocessing_recoloring = preprocessing_recoloring
        )

        metadata_postprocessing.metadata_postprocessing(
            input_data_dir = data_dir,
            input_solution_dir = 'Solution_' + args.solutiona_dir,
            input_data_dir_groundtruth = data_groundtruth_dir,
            crop_size = 256
        )
    else:
        '''
        if cwd_flag == False:
            cwd_flag = True

            original_cwd = os.getcwd()
            print(os.getcwd())

            os.chdir(original_cwd + '\LOAM')
            print(os.getcwd())
        '''
            
        import loam_inference

        loam_inference.loam_inference(
            input_filtering_new_dataset = True,
            input_filtering_threshold = 0.33,
            input_k_fold_testing = 1,
            input_crop_size = 256,
            input_separate_validating_set = False,
            input_reading_predefined_testing = True,
            input_training_needed = False,
            input_targeted_map_file = 'targeted_map.csv',
            input_map_source_dir = data_dir,
            input_groundtruth_dir = data_groundtruth_dir
        )




import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Data/validation')
    parser.add_argument('--data_groundtruth_dir', type=str, default='Data/validation_groundtruth')
    parser.add_argument('--solutiona_dir', type=str, default='1102')
    parser.add_argument('--targeted_map_list', type=str, default='targeted_map.csv')
    parser.add_argument('--map_preprocessing', type=str, default='True')
    parser.add_argument('--generate_boundary_extraction', type=str, default='True')
    parser.add_argument('--printing_auxiliary_information', type=str, default='True')
    parser.add_argument('--preprocessing_recoloring', type=str, default='True')
    parser.add_argument('--model_inference', type=str, default='False')

    args = parser.parse_args()
    
    #print(f"Processing output for: {args.result_name}")
    main()
