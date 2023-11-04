
import torch
#print(torch.cuda.is_available())
#print(torch.cuda.get_device_name(0))

''' For training '''
#import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from loam.evaluate import evaluate
from loam.loam_model import LOAM
from loam.utils.data_loading import BasicDataset, CarvanaDataset
from loam.utils.dice_score import dice_loss

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import numpy as np
import time

import math

import csv
import random

from datetime import datetime


''' For predicting '''
#import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from loam.utils.data_loading import BasicDataset
from loam.loam_model import LOAM
from loam.utils.utils import plot_img_and_mask



''' For performance evaluating '''
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

# Code moved to validation_evaluation_worker.py
import validation_evaluation_worker

import multiprocessing
PROCESSES = 10


from loss import losses


crop_size = 256

filtering_new_dataset = True # Set to [True] if one wants to filter a new dataset
filtering_threshold = 0.33 # An image must have more than [0.33] labeled pixels to be filtered as a valid image candidate for training

k_fold_testing = 1 # can set to any values...
separate_validating_set = False
reading_predefined_testing = True
training_needed = False

targeted_map_file = 'targeted_map'
training_map_list = 'targeted_map_validation.csv'


dir_source = Path('data/cma/imgs/')

dir_img_0 = Path('data/cma_small/imgs/')
dir_mask_0 = Path('data/cma_small/masks/')
dir_checkpoint = Path('checkpoints/')

dir_img = Path('data/cma_small/imgs(2)/')
dir_mask = Path('data/cma_small/masks(2)/')


dir_img_testing = Path('data/cma_small/imgs/')
dir_mask_testing = Path('data/cma_small/masks/')




def multiprocessing_setting():
    global PROCESSES

    multiprocessing.set_start_method('spawn', True)
    if PROCESSES > multiprocessing.cpu_count():
        PROCESSES = (int)(multiprocessing.cpu_count()/2)


auxiliary_dict_indexed = {}
def setup_auxiliary():
    global auxiliary_dict_indexed

    auxiliary_info_source = Path('data/auxiliary_info.csv')

    auxiliary_info = np.genfromtxt(auxiliary_info_source, delimiter=',', dtype=None, encoding='utf8')
    print(auxiliary_info.shape)
    print(auxiliary_info)

    #auxiliary_dict_indexed = {}
    for info_index in range(1, auxiliary_info.shape[0]):
        auxiliary_dict_indexed.update({auxiliary_info[info_index, 1] : [torch.as_tensor(auxiliary_info[info_index, 2:34].astype(float)).float().contiguous(), torch.as_tensor(auxiliary_info[info_index, 34:].astype(float)).float().contiguous()]})


def setup_directory():
    if not os.path.exists(dir_img):
        os.makedirs(dir_img)
    if not os.path.exists(os.path.join(dir_img, 'sup')):
        os.makedirs(os.path.join(dir_img, 'sup'))
    if not os.path.exists(dir_mask):
        os.makedirs(dir_mask)
    if not os.path.exists('output'):
        os.makedirs('output')
    if not os.path.exists('predict'):
        os.makedirs('predict')


def filter_training():
    ### only filter the training dataset

    if filtering_new_dataset == True:
        file_target_map = open(training_map_list, 'r')
        data_target_map = list(csv.reader(file_target_map, delimiter=','))
        file_target_map.close()

        print(len(data_target_map))
        print(data_target_map)



        if os.path.isfile('data/cma_small/polygon_area_record.csv') == False:
            with open('data/cma_small/polygon_area_record.csv','w') as fd:
                fd.write('Key_Name,Area\n')
                fd.close()

        counter_0 = 0
        counter_1 = 0
        runningtime_start = datetime.now()

        targeted_image_list = ([all_image for all_image in os.listdir(dir_img_0) if any(targeted_image[0] in all_image for targeted_image in data_target_map)])
        #for filtering_training_set in os.listdir(dir_img_0):
        for filtering_training_set in targeted_image_list:
            filtering_training_filename = os.path.join(dir_img_0, filtering_training_set)
            if '_sup_' in filtering_training_filename:
                continue
            ext = os.path.splitext(filtering_training_filename)[1]
            if ext != '.png':
                continue
            filtering_training_filename2 = os.path.join(dir_mask_0, filtering_training_set.split('.')[0]+'_mask.png')
            #print(filtering_training_filename2)

            img = cv2.imread(filtering_training_filename2)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            counter_0 = counter_0 + 1

            
            if np.unique(img).shape[0] == 2:
                if (img > 1).any():
                    img = img / 255.0
                this_area = np.mean(img)
                if this_area * 255.0 > 255.0 * filtering_threshold:
                    filtered_training_filename = os.path.join(dir_img, filtering_training_set)
                    filtered_training_filename2 = os.path.join(dir_mask, filtering_training_set.split('.')[0]+'_mask.png')
                    shutil.copyfile(filtering_training_filename, filtered_training_filename)
                    shutil.copyfile(filtering_training_filename2, filtered_training_filename2)

                    shutil.copyfile(os.path.join(dir_img_0, 'sup', filtering_training_set.split('.')[0]+'_sup_0.png'), os.path.join(dir_img, 'sup', filtering_training_set.split('.')[0]+'_sup_0.png'))
                    shutil.copyfile(os.path.join(dir_img_0, 'sup', filtering_training_set.split('.')[0]+'_sup_1.png'), os.path.join(dir_img, 'sup', filtering_training_set.split('.')[0]+'_sup_1.png'))
                    shutil.copyfile(os.path.join(dir_img_0, 'sup', filtering_training_set.split('.')[0]+'_sup_2.png'), os.path.join(dir_img, 'sup', filtering_training_set.split('.')[0]+'_sup_2.png'))
                    shutil.copyfile(os.path.join(dir_img_0, 'sup', filtering_training_set.split('.')[0]+'_sup_3.png'), os.path.join(dir_img, 'sup', filtering_training_set.split('.')[0]+'_sup_3.png'))
                    shutil.copyfile(os.path.join(dir_img_0, 'sup', filtering_training_set.split('.')[0]+'_sup_4.png'), os.path.join(dir_img, 'sup', filtering_training_set.split('.')[0]+'_sup_4.png'))
                    shutil.copyfile(os.path.join(dir_img_0, 'sup', filtering_training_set.split('.')[0]+'_sup_5.png'), os.path.join(dir_img, 'sup', filtering_training_set.split('.')[0]+'_sup_5.png'))

                    counter_1 = counter_1 +1

                with open('data/cma_small/polygon_area_record.csv','a') as fd:
                    fd.write(str(filtering_training_set.split('.')[0])+','+str(this_area)+'\n')
                    fd.close()
            else:
                if (img > 1).any():
                    this_area = 1.0
                else:
                    this_area = 0.0
                with open('data/cma_small/polygon_area_record.csv','a') as fd:
                    fd.write(str(filtering_training_set.split('.')[0])+','+str(this_area)+'\n')
                    fd.close()
            
            '''
            filtered_training_filename = os.path.join(dir_img, filtering_training_set)
            filtered_training_filename2 = os.path.join(dir_mask, filtering_training_set.split('.')[0]+'_mask.png')
            shutil.copyfile(filtering_training_filename, filtered_training_filename)
            shutil.copyfile(filtering_training_filename2, filtered_training_filename2)

            shutil.copyfile(os.path.join(dir_img_0, 'sup', filtering_training_set.split('.')[0]+'_sup_0.png'), os.path.join(dir_img, 'sup', filtering_training_set.split('.')[0]+'_sup_0.png'))
            shutil.copyfile(os.path.join(dir_img_0, 'sup', filtering_training_set.split('.')[0]+'_sup_1.png'), os.path.join(dir_img, 'sup', filtering_training_set.split('.')[0]+'_sup_1.png'))
            shutil.copyfile(os.path.join(dir_img_0, 'sup', filtering_training_set.split('.')[0]+'_sup_2.png'), os.path.join(dir_img, 'sup', filtering_training_set.split('.')[0]+'_sup_2.png'))
            shutil.copyfile(os.path.join(dir_img_0, 'sup', filtering_training_set.split('.')[0]+'_sup_3.png'), os.path.join(dir_img, 'sup', filtering_training_set.split('.')[0]+'_sup_3.png'))
            shutil.copyfile(os.path.join(dir_img_0, 'sup', filtering_training_set.split('.')[0]+'_sup_4.png'), os.path.join(dir_img, 'sup', filtering_training_set.split('.')[0]+'_sup_4.png'))
            shutil.copyfile(os.path.join(dir_img_0, 'sup', filtering_training_set.split('.')[0]+'_sup_5.png'), os.path.join(dir_img, 'sup', filtering_training_set.split('.')[0]+'_sup_5.png'))

            counter_1 = counter_1 +1
            '''

            if counter_0 % 5000 == 0:
                #print('filtering training dataset: (', str(counter_0), ' / ', str(len(os.listdir(dir_img_0))-1), ')... ', datetime.now()-runningtime_start)
                print('filtering training dataset: (', str(counter_0), ' / ', str(len(targeted_image_list)), ')... ', datetime.now()-runningtime_start)
            

        print(str(counter_1) + ' / ' + str(counter_0))
    else:
        print('training dataset is already filtered...')




training_map = np.empty((1), dtype=object)
validating_map = np.empty((1), dtype=object)
testing_map = np.empty((1), dtype=object)

def identify_dataset():
    global training_map
    global validating_map
    global testing_map


    if reading_predefined_testing == False:
        target_map_list = []
        '''
        with open('E:/Research/LOAM/targeted_map.csv','r') as target_map_file:
            data_iter = csv.reader(target_map_file, delimiter = ',', quotechar = '"')
            target_map_list = [data for data in data_iter]
        '''

        
        temp_test_map_name = ''
        for testing_input in os.listdir(dir_img):
            testing_name = os.fsdecode(testing_input)
            if '_sup_' in testing_name:
                continue
            ext = os.path.splitext(testing_input)[1]
            if ext != '.png':
                continue

            this_map_name = '_'.join(os.path.splitext(testing_input)[0].split('_')[:-4])
            if this_map_name != temp_test_map_name:
                # further check whether all sub-strings start with a capital letter
                this_map_name_split = os.path.splitext(this_map_name)[0].split('_')
                this_map_name_check = this_map_name_split[0]
                for sub_string in this_map_name_split[1:]:
                    if sub_string[0].isupper():
                        this_map_name_check = this_map_name_check + '_' + sub_string
                this_map_name = this_map_name_check

                if this_map_name != temp_test_map_name:
                    target_map_list.append(this_map_name)
                    print(this_map_name)
            temp_test_map_name = this_map_name
        target_map_list = np.asarray(target_map_list).flatten()


        print('')
        print(target_map_list.shape[0])
        print(target_map_list)
    else:
        print('Since we have the testing dataset, k-fold validation is no longer used...')



    if reading_predefined_testing == False:
        folded_count = target_map_list.shape[0] / k_fold_testing
        np.random.shuffle(target_map_list)

        batch_map = []
        training_map = np.empty((k_fold_testing), dtype=object)
        validating_map = np.empty((k_fold_testing), dtype=object)
        testing_map = np.empty((k_fold_testing), dtype=object)

        print('****** Performing '+str(k_fold_testing)+'-Fold Testing ('+str(k_fold_testing)+' Batches for '+str(target_map_list.shape[0])+' Maps) ******')
        for kb in range(0, k_fold_testing):
            #print(int(folded_count*k), int(folded_count*(k+1)))
            batch_map.append(target_map_list[int(folded_count*kb): int(folded_count*(kb+1))])
            print(batch_map[kb])

        with open('output/batch_map.csv', 'w', newline="") as fd:
            writer = csv.writer(fd)
            writer.writerows(batch_map)

        for k in range(0, k_fold_testing):
            candidate_batch = list(range(0, k_fold_testing))
            
            this_testing_map = k
            candidate_batch.remove(this_testing_map)

            if separate_validating_set == True:
                this_validating_map = k+1
                if this_validating_map >= k_fold_testing:
                    this_validating_map = this_validating_map - k_fold_testing
                candidate_batch.remove(this_validating_map)

            this_training_map_batch = []
            for extended_training_map_batch in candidate_batch:
                this_training_map_batch.extend(batch_map[extended_training_map_batch])

            training_map[k] = np.asarray(this_training_map_batch)
            if separate_validating_set == True:
                validating_map[k] = np.asarray(batch_map[this_validating_map])
            testing_map[k] = np.asarray(batch_map[this_testing_map])    

        for k in range(0, k_fold_testing):
            print('')
            print('================== Maps for Training (' + str(training_map[k].shape[0]) + 'Maps) ('+str(k)+' Fold) ==================')
            print(training_map[k])
            if separate_validating_set == True:
                print('================== Maps for Validating (' + str(validating_map[k].shape[0]) + 'Maps) ('+str(k)+' Fold) ==================')
                print(validating_map[k])
            print('================== Maps for Testing (' + str(testing_map[k].shape[0]) + 'Maps) ('+str(k)+' Fold) ==================')
            print(testing_map[k])
    else:
        print('Since we have the testing dataset, k-fold validation is no longer used...')

    


    if reading_predefined_testing == True:
        file_target_map = open(training_map_list, 'r')
        data_target_map_0 = list(csv.reader(file_target_map, delimiter=','))
        file_target_map.close()

        file_target_map = open(targeted_map_file, 'r')
        data_target_map_1 = list(csv.reader(file_target_map, delimiter=','))
        file_target_map.close()

        training_map = np.empty((k_fold_testing), dtype=object)
        validating_map = np.empty((k_fold_testing), dtype=object)
        testing_map = np.empty((k_fold_testing), dtype=object)

        #for k in range(0, k_fold_testing):
        for k in range(0, 1):
            this_training_map_batch = []
            for targeted_map in data_target_map_0:
                this_training_map_batch.extend(targeted_map)
            training_map[k] = np.asarray(this_training_map_batch)

        folded_count = len(data_target_map_1) / k_fold_testing
        for k in range(0, k_fold_testing):
            this_testing_map_batch = []
            for targeted_map in range(int(k*folded_count), int((k+1)*folded_count)):
                this_testing_map_batch.extend(data_target_map_1[targeted_map])
            testing_map[k] = np.asarray(this_testing_map_batch)

        k = 0
        print('================== Maps for Training (' + str(training_map[k].shape[0]) + 'Maps) ('+str(k)+' Fold) ==================')
        print(training_map[k])
        print('')
        for k in range(0, k_fold_testing):
            print('================== Maps for Testing (' + str(testing_map[k].shape[0]) + 'Maps) ('+str(k)+' Fold) ==================')
            print(testing_map[k])
    else:
        print('Using the k-fold validation for testing...')




def identify_dataset_v2():
    global training_map
    global validating_map
    global testing_map

    if reading_predefined_testing == True:
        file_target_map = open(targeted_map_file, 'r')
        data_target_map_0 = list(csv.reader(file_target_map, delimiter=','))
        file_target_map.close()

        file_target_map = open(targeted_map_file, 'r')
        data_target_map_1 = list(csv.reader(file_target_map, delimiter=','))
        file_target_map.close()

        training_map = np.empty((k_fold_testing), dtype=object)
        validating_map = np.empty((k_fold_testing), dtype=object)
        testing_map = np.empty((k_fold_testing), dtype=object)

        #for k in range(0, k_fold_testing):
        for k in range(0, 1):
            this_training_map_batch = []
            for targeted_map in data_target_map_0:
                this_training_map_batch.extend(targeted_map)
            training_map[k] = np.asarray(this_training_map_batch)

        folded_count = len(data_target_map_1) / k_fold_testing
        for k in range(0, k_fold_testing):
            this_testing_map_batch = []
            for targeted_map in range(int(k*folded_count), int((k+1)*folded_count)):
                this_testing_map_batch.extend(data_target_map_1[targeted_map])
            testing_map[k] = np.asarray(this_testing_map_batch)

        k = 0
        '''
        print('================== Maps for Training (' + str(training_map[k].shape[0]) + 'Maps) ('+str(k)+' Fold) ==================')
        print(training_map[k])
        print('')
        '''
        for k in range(0, k_fold_testing):
            print('================== Maps for Testing (' + str(testing_map[k].shape[0]) + 'Maps) ('+str(k)+' Fold) ==================')
            print(testing_map[k])
    else:
        print('Using the k-fold validation for testing...')







def train_model(
        model,
        device,
        epochs: int = 10,
        batch_size: int = 1,
        learning_rate: float = 1e-5, # 1e-5
        val_percent: float = 0.3, # 0.25
        save_checkpoint: bool = True,
        img_scale: float = 1.0,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999, # 0.995
        gradient_clipping: float = 1.0,
        pre_defined_val: bool = False,
        this_training_map: np = None,
        this_validating_map: np = None,
        auxiliary_dict: dict = None,
):
    print('check dictionary integrity:', len(auxiliary_dict))

    if pre_defined_val == False:
        # 1. Create dataset
        try:
            dataset = CarvanaDataset(dir_img, dir_mask, this_training_map, auxiliary_dict, img_scale)
        except (AssertionError, RuntimeError, IndexError):
            dataset = BasicDataset(dir_img, dir_mask, this_training_map, auxiliary_dict, img_scale)

        # 2. Split into train / validation partitions
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    elif pre_defined_val == True:
        try:
            train_set = CarvanaDataset(dir_img, dir_mask, this_training_map, auxiliary_dict, img_scale)
            val_set = CarvanaDataset(dir_img, dir_mask, this_validating_map, auxiliary_dict, img_scale)
        except (AssertionError, RuntimeError, IndexError):
            train_set = BasicDataset(dir_img, dir_mask, this_training_map, auxiliary_dict, img_scale)
            val_set = BasicDataset(dir_img, dir_mask, this_validating_map, auxiliary_dict, img_scale)
        dataset = train_set + val_set # merge two sub-datasets
        n_train = len(train_set)
        n_val = len(val_set)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    #criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    criterion = losses.FocalLoss()
    global_step = 0

    # Set up early stopping
    last_loss = 0
    patience = 2
    trigger_times = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks, auxiliary_info_1, auxiliary_info_2 = batch['image'], batch['mask'], batch['auxiliary_1'], batch['auxiliary_2']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                auxiliary_info_1 = auxiliary_info_1.to(device=device, dtype=torch.float32)
                auxiliary_info_2 = auxiliary_info_2.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images, auxiliary_info_1, auxiliary_info_2)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not torch.isinf(value).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not torch.isinf(value.grad).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
        
        # Check for early stopping
        current_loss = evaluate(model, val_loader, device, amp) # range=[0, 1], to be maximized
        print('Current Loss: ', current_loss)
        if current_loss < last_loss:
            trigger_times += 1
            print('Trigger Times: ', trigger_times)
            if trigger_times > patience:
                print('Early Stopping...')
                break
        else:
            trigger_times = 0
            print('Trigger Times: ', trigger_times)

        last_loss = current_loss




def combine_img(img_path, sup_0, sup_1, sup_2, sup_3, sup_4, sup_5, scale_factor):
    img = Image.open(img_path)
    img_sup_0 = Image.open(sup_0)
    img_sup_1 = Image.open(sup_1)
    img_sup_2 = Image.open(sup_2)
    img_sup_3 = Image.open(sup_3)
    img_sup_4 = Image.open(sup_4)
    img_sup_5 = Image.open(sup_5)

    img = BasicDataset.preprocess(None, img, scale_factor, is_mask=False)
    img_sup_0 = BasicDataset.preprocess(None, img_sup_0, scale_factor, is_mask=False)
    img_sup_1 = BasicDataset.preprocess(None, img_sup_1, scale_factor, is_mask=False)
    img_sup_2 = BasicDataset.preprocess(None, img_sup_2, scale_factor, is_mask=False)
    img_sup_3 = BasicDataset.preprocess(None, img_sup_3, scale_factor, is_mask=False)
    img_sup_4 = BasicDataset.preprocess(None, img_sup_4, scale_factor, is_mask=False)
    img_sup_5 = BasicDataset.preprocess(None, img_sup_5, scale_factor, is_mask=False)
    
    img_combined = np.zeros((7, img.shape[1], img.shape[2]), dtype=float)
    img_combined[0] = img
    img_combined[1] = img_sup_0
    img_combined[2] = img_sup_1
    img_combined[3] = img_sup_2
    img_combined[4] = img_sup_3
    img_combined[5] = img_sup_4
    img_combined[6] = img_sup_5
    
    img_combined = img_combined.astype(float)

    return img_combined


def predict_img(net,
                full_img,
                targeted_auxiliary_info,
                device,
                scale_factor=1.0,
                out_threshold=0.5,
):
    net.eval()
    #img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    #img = torch.from_numpy(full_img)
    img = torch.as_tensor(full_img.copy()).float().contiguous()
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    #img = img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

    auxiliary_info_1 = targeted_auxiliary_info[0]
    auxiliary_info_1 = auxiliary_info_1.unsqueeze(0)
    auxiliary_info_1 = auxiliary_info_1.to(device=device, dtype=torch.float32)

    auxiliary_info_2 = targeted_auxiliary_info[1]
    auxiliary_info_2 = auxiliary_info_2.unsqueeze(0)
    auxiliary_info_2 = auxiliary_info_2.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img, auxiliary_info_1, auxiliary_info_2).cpu()
        #output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        output = F.interpolate(output, (full_img.shape[2], full_img.shape[1]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args['output'] or list(map(_generate_name, args['input']))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)




def print_model_summary():
    from torchinfo import summary
    from loam.loam_model import LOAM

    temp_model = LOAM(n_channels=7, n_classes=2, bilinear=False)
    summary(temp_model, input_size=[(1,7,crop_size,crop_size), (1,32), (1,9)])

    # print(LOAM(n_channels=4, n_classes=2, bilinear=False))





def model_training():
    if training_needed == True:
        #for k in range(0, k_fold_testing):
        k = 0

        print('')
        print('================== Perform '+str(k_fold_testing)+'-Fold Testing (' + str(k) + ' Fold) ==================')

        runningtime_start_global = datetime.now()
        #dir_checkpoint = Path('checkpoints/fold_'+str(k)+'/')
        dir_checkpoint = Path('checkpoints/fold_0/')

        ''' Setup training arguments '''
        args = {
            "epochs": 20,
            "batch-size": 1,
            "learning-rate": 1e-5,
            "load": False, # load model from a .pth file
            "scale": 1.0, # downscaling factor of the images
            "validation": 20, # percentage (0-100)
            "amp": True, # mixed precision
            "bilinear": False, # bilinear upsampling
            "classes": 2 # number of classes
        }

        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')

        # Change here to adapt to your data
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
        model = LOAM(n_channels=7, n_classes=args['classes'], bilinear=args['bilinear'])
        model = model.to(memory_format=torch.channels_last)

        logging.info(f'Network:\n'
                        f'\t{model.n_channels} input channels\n'
                        f'\t{model.n_classes} output channels (classes)\n'
                        f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')


        ''' Perform training '''
        if args['load']:
            state_dict = torch.load(args['load'], map_location=device)
            del state_dict['mask_values']
            model.load_state_dict(state_dict)
            logging.info(f'Model loaded from {str(args["load"])}')

        model.to(device=device)
        try:
            train_model(
                model=model,
                epochs=args['epochs'],
                batch_size=args['batch-size'],
                learning_rate=args['learning-rate'],
                device=device,
                img_scale=args['scale'],
                val_percent=args['validation'] / 100,
                amp=args['amp'],
                pre_defined_val=separate_validating_set,
                this_training_map=training_map[k],
                this_validating_map=validating_map[k],
                auxiliary_dict = auxiliary_dict_indexed
            )
        except torch.cuda.OutOfMemoryError:
            logging.error('Detected OutOfMemoryError! '
                            'Enabling checkpointing to reduce memory usage, but this slows down training. '
                            'Consider enabling AMP (--amp) for fast and memory efficient training')
            torch.cuda.empty_cache()
            model.use_checkpointing()
            train_model(
                model=model,
                epochs=args['epochs'],
                batch_size=args['batch-size'],
                learning_rate=args['learning-rate'],
                device=device,
                img_scale=args['scale'],
                val_percent=args['validation'] / 100,
                amp=args['amp'],
                pre_defined_val=separate_validating_set,
                this_training_map=training_map[k],
                this_validating_map=validating_map[k],
                auxiliary_dict = auxiliary_dict_indexed
            )

        print('time_checkpoint (model training): ', datetime.now()-runningtime_start_global)
    else:
        print('training is already done...')

    
    


def model_testing():
    targeted_epoch_found = False
    targeted_epoch = -1
    dir_checkpoint = Path('checkpoints/fold_0/')

    if os.path.isdir('checkpoints/fold_0/') == False:
        dir_checkpoint = Path('checkpoints/')

    for epoch_id in range(20, 0, -1):
        if os.path.isfile(os.path.join(dir_checkpoint, 'checkpoint_epoch'+str(epoch_id)+'.pth')):
            targeted_epoch_found = True
            targeted_epoch = epoch_id
            break

    if targeted_epoch_found == False:
        print('No epoch for a successfully trained model is found...')
    else:
        epoch_id_mod = epoch_id
        if os.path.isfile(os.path.join(dir_checkpoint, 'checkpoint_epoch'+str(epoch_id_mod)+'.pth')):
            targeted_epoch = epoch_id_mod
        print('Selecting epoch '+str(targeted_epoch)+' for testing...')




    runningtime_start_global = datetime.now()

    for k in range(0, k_fold_testing):

        ''' Setup predicting arguments '''
        dir_pred_testing = Path('predict/fold_'+str(k)+'/cma_small/predict/')
        dir_pred_testing1 = Path('predict/fold_'+str(k)+'/cma/predict/')
        #dir_pred_testing = Path('predict/fold_0/cma_small/predict/')
        #dir_pred_testing1 = Path('predict/fold_0/cma/predict/')

        if not os.path.exists(dir_pred_testing):
            os.makedirs(dir_pred_testing)
        if not os.path.exists(dir_pred_testing1):
            os.makedirs(dir_pred_testing1)

        args = {
            "model": os.path.join(dir_checkpoint, 'checkpoint_epoch'+str(targeted_epoch)+'.pth'),
            "input": dir_img_testing, # Filenames of input images
            "output": dir_pred_testing, # Filenames of output images
            "output_merged": dir_pred_testing1, # Filenames of output images (merged)
            "viz": False,
            "no-save": False,
            "mask-threshold": 0.1, # Minimum probability value to consider a mask pixel white
            "scale": 1.0, # Scale factor for the input images
            "amp": True, # mixed precision
            "bilinear": False, # bilinear upsampling
            "classes": 2 # number of classes
        }

        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

        net = LOAM(n_channels=7, n_classes=args['classes'], bilinear=args['bilinear'])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Loading model {args["model"]}')
        logging.info(f'Using device {device}')

        net.to(device=device)
        state_dict = torch.load(args['model'], map_location=device)
        mask_values = state_dict.pop('mask_values', [0, 1])
        net.load_state_dict(state_dict)

        logging.info('Model loaded!')


        ''' Check predicting information '''
        #testing_key = [os.path.splitext(file)[0] for file in os.listdir(args['input']) if os.path.isfile(os.path.join(args['input'], file)) and not file.startswith('.') and (('_'.join(os.path.splitext(file)[0].split('_')[:-4])) in testing_map[k])]
        testing_key = [os.path.splitext(file)[0] for file in os.listdir(args['input']) if os.path.isfile(os.path.join(args['input'], file)) and not file.startswith('.') and any(testing_map_name in file for testing_map_name in testing_map[k])]
        
        testing_key_count = len(testing_key)
        print(str(testing_key_count)+'images to be predicted... ')


        ''' Perform predicting '''
        candidate_to_merge = []

        predict_counter = 0
        runningtime_start = datetime.now()

        with tqdm(total=testing_key_count, desc=f'Prediction - Fold {(k+1)}/{k_fold_testing}', unit='img') as pbar0:
            for testing_input in os.listdir(args['input']):
                testing_name = os.fsdecode(testing_input)
                if '_sup_' in testing_name:
                    continue
                ext = os.path.splitext(testing_input)[1]
                if ext != '.png':
                    continue

                #this_map_name = '_'.join(os.path.splitext(testing_input)[0].split('_')[:-4])
                #if this_map_name in testing_map[k]:
                if any(testing_map_name in testing_input for testing_map_name in testing_map[k]):
                    # print('get testing map...')
                    testing_input_filename = os.path.join(args['input'], testing_input)
                    testing_output_filename = os.path.join(args['output'], testing_name.split('.')[0]+'_predict.png')

                    # logging.info(f'Predicting image {testing_input_filename} ...')
                    img = Image.open(testing_input_filename)
                    img_path = testing_input_filename

                    img_sup_0 = os.path.join(args['input'], 'sup', testing_input.split('.')[0]+'_sup_0.png')
                    img_sup_1 = os.path.join(args['input'], 'sup', testing_input.split('.')[0]+'_sup_1.png')
                    img_sup_2 = os.path.join(args['input'], 'sup', testing_input.split('.')[0]+'_sup_2.png')
                    img_sup_3 = os.path.join(args['input'], 'sup', testing_input.split('.')[0]+'_sup_3.png')
                    img_sup_4 = os.path.join(args['input'], 'sup', testing_input.split('.')[0]+'_sup_4.png')
                    img_sup_5 = os.path.join(args['input'], 'sup', testing_input.split('.')[0]+'_sup_5.png')

                    image_key_name = str('_'.join((testing_input.split('.')[0]).split('_')[:-2]))
                    #print(image_key_name)
                    targeted_auxi = auxiliary_dict_indexed[image_key_name]

                    mask = predict_img(net=net,
                                        full_img=combine_img(img_path, img_sup_0, img_sup_1, img_sup_2, img_sup_3, img_sup_4, img_sup_5, args['scale']),
                                        targeted_auxiliary_info = targeted_auxi,
                                        scale_factor=args['scale'],
                                        out_threshold=args['mask-threshold'],
                                        device=device)
                    
                    if not args['no-save']:
                        out_filename = testing_output_filename
                        result = mask_to_image(mask, mask_values)
                        result.save(out_filename)
                        # logging.info(f'Mask saved to {out_filename}')

                        if 'poly_0_0.png' in testing_name:
                            this_map_name_index = -4
                            this_map_name = ''
                            this_legend_name = ''
                            for index_attemp in range(-4, -testing_input.count('_')-1, -1):
                                this_map_name_candidate = '_'.join(os.path.splitext(testing_input)[0].split('_')[:index_attemp])
                                if this_map_name_candidate in testing_map[k]:
                                    this_map_name = this_map_name_candidate
                                    this_legend_name = '_'.join(os.path.splitext(testing_input)[0].split('_')[index_attemp:-2])
                                    break
                            #if this_map_name != temp_name_set[0] or this_legend_name != temp_name_set[1]:
                                #print(this_map_name, this_legend_name)
                                #temp_name_set = [this_map_name, this_legend_name]
                                #name_set_counting += 1

                            map_name = this_map_name
                            label_name = this_legend_name
                            candidate_to_merge.append([map_name, label_name])
                    
                            #map_name = '_'.join(testing_name.split('_')[:-4])
                            #label_name = testing_name.split('_')[-4]
                            #candidate_to_merge.append([map_name, label_name])

                    if args['viz']:
                        # logging.info(f'Visualizing results for image {testing_name}, close to continue...')
                        plot_img_and_mask(img, mask)
                    
                    predict_counter = predict_counter + 1
                    if predict_counter % 2500 == 0:
                        print('Making predictions... ('+str(predict_counter)+' / '+str(testing_key_count)+')... ', datetime.now()-runningtime_start)
                    pbar0.update(1)


        ''' Merge back to complete image '''
        print(str(len(candidate_to_merge))+' images to be merged... ')

        with tqdm(total=len(candidate_to_merge), desc=f'Merge - Fold {(k+1)}/{k_fold_testing}', unit='img') as pbar0:
            for map_name, label_name in candidate_to_merge:
                source_name = map_name + '_' + label_name + '.png'
                source_filename = os.path.join(dir_source, source_name)

                img = cv2.imread(source_filename)
                # original_shape = img.shape
                # print(source_filename, original_shape[0:2])
                empty_grid = np.zeros((img.shape[0], img.shape[1]), dtype='uint8').astype(float)
                empty_flag = True

                for r in range(0,math.ceil(img.shape[0]/crop_size)):
                    for c in range(0,math.ceil(img.shape[1]/crop_size)):
                        this_block_source = os.path.join(args['output'], str(source_name.split('.')[0]+"_"+str(r)+"_"+str(c)+"_predict.png"))
                        #print(this_block_source)
                        already_predicted = os.path.isfile(this_block_source)

                        if already_predicted == True:
                            block_img = cv2.imread(this_block_source)
                            block_img = cv2.cvtColor(block_img, cv2.COLOR_BGR2GRAY)

                            r_0 = r*crop_size
                            r_1 = min(r*crop_size+crop_size, img.shape[0])
                            c_0 = c*crop_size
                            c_1 = min(c*crop_size+crop_size, img.shape[1])
                            
                            empty_grid[r_0:r_1, c_0:c_1] = block_img[r_0%crop_size:(r_1-r_0), c_0%crop_size:(c_1-c_0)]
                        else:
                            empty_flag = False
                            break
                    if empty_flag == False:
                        break
                
                if empty_flag == True:
                    cv2.imwrite(os.path.join(args['output_merged'], str(source_name.split('.')[0]+"_predict.png")), empty_grid)
                    #logging.info(f'Merging predicted image {source_name} ...')
                    pbar0.update(1)
                else:
                    continue


        ''' Conduct performance evaluation '''
        print(str(len(os.listdir(args['output_merged'])))+' images to be evaluated with groundtruth... ')

        info_set = []
        for prediction_merged in os.listdir(args['output_merged']):
            prediction_name = os.fsdecode(prediction_merged)
            prediction_filename = os.path.join(args['output_merged'], prediction_name)

            map_name = '_'.join(prediction_name.split('_')[:-3])
            label_name = prediction_name.split('_')[-3]

            map_filename = os.path.join(map_source_dir, map_name+str('.tif')) # Path to the input dataset
            label_filename = map_filename.replace('.tif','.json')
            groundtruth_filename = os.path.join(groundtruth_dir, map_name+'_'+label_name+str('_poly.tif')) # Path to the corresponding groundtruth dataset
            if os.path.isfile(groundtruth_filename) == False:
                print('no groundturht provided... ', groundtruth_filename)
                continue

            prediction_source_filename = os.path.join('data/cma/imgs', map_name+'_'+label_name+str('.png')) # Path to the output

            info_set_placeholder = []
            info_set_placeholder.append(map_filename)
            info_set_placeholder.append(label_filename)
            info_set_placeholder.append(prediction_filename)
            info_set_placeholder.append(groundtruth_filename)
            info_set_placeholder.append(map_name)
            info_set_placeholder.append(label_name)

            info_set_placeholder.append(prediction_source_filename)
            # print(info_set_placeholder)
            info_set.append(info_set_placeholder)


        #runningtime_start=datetime.now()
        performance_block = []

        if os.path.isfile('output/performance_folded.csv') == False:
            with open('output/performance_folded.csv','w') as fd:
                fd.write('Fold,Map_Name,Key_Name,Precision,Recall,F1_Score\n')
                fd.close()

        with tqdm(total=len(info_set), desc=f'Evaluation - Fold {(k+1)}/{k_fold_testing}', unit='img') as pbar0:
            for batch in range(0, math.ceil(len(info_set)/PROCESSES)):
                batch_range = [PROCESSES*batch, min(PROCESSES*(batch+1), len(info_set))]
                #print(batch_range)

                with multiprocessing.Pool(PROCESSES) as pool:
                    #multiprocessing_results = [pool.apply_async(validation_evaluation_worker, (info_id,info_set[info_id],)) for info_id in range(0, len(info_set))]
                    callback = pool.starmap_async(validation_evaluation_worker.validation_evaluation_worker, [(info_id, info_set[info_id], ) for info_id in range(batch_range[0], batch_range[1])]) # len(info_set)
                    multiprocessing_results  = callback.get()
                        
                    for returned_info in multiprocessing_results:
                        #map_name, legend_name, precision, recall, f_score = returned_info
                        try:
                            returned = returned_info
                            map_name = returned[0]
                            legend_name = returned[1]
                            precision_0 = returned[2]
                            recall_0 = returned[3]
                            f_score_0 = returned[4]
                            print(map_name, legend_name, precision_0, recall_0, f_score_0)
                            performance_block.append(returned)

                            with open('output/performance_folded.csv','a') as fd:
                                fd.write(str(k)+','+map_name+','+legend_name+','+str(precision_0)+','+str(recall_0)+','+str(f_score_0)+','+'\n')
                                fd.close()
                                pbar0.update(1)
                        except:
                            with open('output/performance_folded.csv','a') as fd:
                                fd.write(str(k)+','+'error,error\n')
                                fd.close()
                                pbar0.update(1)
                #if batch%5 == 0:
                    #print('time_checkpoint('+str(batch)+'): ', datetime.now()-runningtime_start)
                #print(performance_block)
                performance_block = []

        
        print('time_checkpoint (model inferring, fold- '+str(k)+'): ', datetime.now()-runningtime_start_global)





def run():
    multiprocessing_setting()
    setup_auxiliary()
    setup_directory()
    filter_training()
    identify_dataset()
    print_model_summary()

    model_training()
    model_testing()


def run_testing():
    multiprocessing_setting()
    setup_auxiliary()
    setup_directory()
    identify_dataset_v2()
    print_model_summary()

    model_testing()





map_source_dir = ''
groundtruth_dir = ''

def loam_inference(
        input_filtering_new_dataset = True,
        input_filtering_threshold = 0.33,
        input_k_fold_testing = 1,
        input_crop_size = 256,
        input_separate_validating_set = False,
        input_reading_predefined_testing = True,
        input_training_needed = False,
        input_targeted_map_file = 'targeted_map.csv',
        input_map_source_dir = 'H:/Research/LOAM/Data/testing',
        input_groundtruth_dir = 'H:/Research/LOAM/Data/testing_groundtruth'
):
    global filtering_new_dataset
    global filtering_threshold
    global k_fold_testing
    global crop_size
    global separate_validating_set
    global reading_predefined_testing
    global training_needed
    global targeted_map_file
    global map_source_dir
    global groundtruth_dir

    filtering_new_dataset = input_filtering_new_dataset
    filtering_threshold = input_filtering_threshold
    k_fold_testing = input_k_fold_testing
    crop_size = input_crop_size
    separate_validating_set = input_separate_validating_set
    reading_predefined_testing = input_reading_predefined_testing
    training_needed = input_training_needed
    targeted_map_file = input_targeted_map_file
    map_source_dir = input_map_source_dir
    groundtruth_dir = input_groundtruth_dir

    run_testing()


def main():    
    global filtering_new_dataset
    global filtering_threshold
    global k_fold_testing
    global crop_size
    global separate_validating_set
    global reading_predefined_testing
    global training_needed
    global targeted_map_file
    global training_map_list
    global map_source_dir
    global groundtruth_dir

    filtering_new_dataset = str_to_bool(args.filtering_new_dataset)
    filtering_threshold = float(args.filtering_threshold)
    k_fold_testing = int(args.k_fold_testing)
    crop_size = int(args.crop_size)
    separate_validating_set = str_to_bool(args.separate_validating_set)
    reading_predefined_testing = str_to_bool(args.reading_predefined_testing)
    training_needed = str_to_bool(args.training_needed)
    targeted_map_file = args.targeted_map_file
    training_map_list = args.training_map_list
    map_source_dir = args.map_source_dir
    groundtruth_dir = args.groundtruth_dir

    run_testing()



def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')




import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filtering_new_dataset', type=str, default='True')
    parser.add_argument('--filtering_threshold', type=str, default='0.33')
    parser.add_argument('--k_fold_testing', type=str, default='1')
    parser.add_argument('--crop_size', type=str, default='256')
    parser.add_argument('--separate_validating_set', type=str, default='False')
    parser.add_argument('--reading_predefined_testing', type=str, default='True')
    parser.add_argument('--training_needed', type=str, default='False')
    parser.add_argument('--targeted_map_file', type=str, default='targeted_map.csv')
    parser.add_argument('--training_map_list', type=str, default='targeted_map_validation.csv')
    parser.add_argument('--map_source_dir', type=str, default='H:/Research/LOAM/Data/testing')
    parser.add_argument('--groundtruth_dir', type=str, default='H:/Research/LOAM/Data/testing_groundtruth')


    args = parser.parse_args()
    
    #print(f"Processing output for: {args.result_name}")
    main()
