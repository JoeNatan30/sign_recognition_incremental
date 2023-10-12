import gc
import ast
import tqdm
import time
import h5py
import glob
import torch
import pandas as pd
import numpy as np
from collections import Counter
import torch.utils.data as torch_data
from torch.utils.data import Dataset
import logging
import random

from augmentations import augmentations

import cv2

def get_data_from_h5(path):
    hf = h5py.File(path, 'r')
    return hf

####################################################################
# Function that helps to see keypoints in an image
####################################################################
def prepare_keypoints_image(keypoints,tag):
    # this vaariable is feeded in draw points process and used in the draw joint lines proceess
    part_line = {}

    # DRAW POINTS
    img = np.zeros((256, 256, 3), np.uint8)

    for n, coords in enumerate(keypoints):

        cor_x = int(coords[0] * 256)
        cor_y = int(coords[1] * 256)

        cv2.circle(img, (cor_x, cor_y), 1, (0, 0, 255), -1)
        part_line[n] = (cor_x, cor_y)

    cv2.imwrite(f'foo_{tag}.jpg', img)

##########################################################
# Process used to normalize the pose
##########################################################
def normalize_pose(data, body_dict):

    sequence_size = data.shape[0]

    last_starting_point, last_ending_point = None, None

    for sequence_index in range(sequence_size):

        # Prevent from even starting the analysis if some necessary elements are not present
        if (data[sequence_index][body_dict['pose_left_shoulder']][0] == 0.0 or data[sequence_index][body_dict['pose_right_shoulder']][0] == 0.0):
            if last_starting_point:
                starting_point, ending_point = last_starting_point, last_ending_point
            else:
                continue
    
        else:

            # NOTE:
            #
            # While in the paper, it is written that the head metric is calculated by halving the shoulder distance,
            # this is meant for the distance between the very ends of one's shoulder, as literature studying body
            # metrics and ratios generally states. The Vision Pose Estimation API, however, seems to be predicting
            # rather the center of one's shoulder. Based on our experiments and manual reviews of the data, employing
            # this as just the plain shoulder distance seems to be more corresponding to the desired metric.
            #
            # Please, review this if using other third-party pose estimation libraries.

            if data[sequence_index][body_dict['pose_left_shoulder']][0] != 0 and data[sequence_index][body_dict['pose_right_shoulder']][0] != 0:
                
                left_shoulder = data[sequence_index][body_dict['pose_left_shoulder']]
                right_shoulder = data[sequence_index][body_dict['pose_right_shoulder']]

                shoulder_distance = ((((left_shoulder[0] - right_shoulder[0]) ** 2) + (
                                       (left_shoulder[1] - right_shoulder[1]) ** 2)) ** 0.5)

                mid_distance = (0.5,0.5)#(left_shoulder - right_shoulder)/2
                head_metric = shoulder_distance/2
            '''
            # use it if you have the neck keypoint
            else:
                neck = (data["neck_X"][sequence_index], data["neck_Y"][sequence_index])
                nose = (data["nose_X"][sequence_index], data["nose_Y"][sequence_index])
                neck_nose_distance = ((((neck[0] - nose[0]) ** 2) + ((neck[1] - nose[1]) ** 2)) ** 0.5)
                head_metric = neck_nose_distance
            '''
            # Set the starting and ending point of the normalization bounding box
            starting_point = [mid_distance[0] - 3 * head_metric, data[sequence_index][body_dict['pose_right_eye']][1] - (head_metric / 2)]
            ending_point = [mid_distance[0] + 3 * head_metric, mid_distance[1] + 4.5 * head_metric]

            last_starting_point, last_ending_point = starting_point, ending_point

        # Normalize individual landmarks and save the results
        for pos, kp in enumerate(data[sequence_index]):
            
            # Prevent from trying to normalize incorrectly captured points
            if data[sequence_index][pos][0] == 0:
                continue

            normalized_x = (data[sequence_index][pos][0] - starting_point[0]) / (ending_point[0] -
                                                                                    starting_point[0])
            normalized_y = (data[sequence_index][pos][1] - ending_point[1]) / (starting_point[1] -
                                                                                    ending_point[1])

            data[sequence_index][pos][0] = normalized_x
            data[sequence_index][pos][1] = 1 - normalized_y
            
    return data
################################################
# Function that normalize the hands (but also the face)
################################################
def normalize_hand(data, body_section_dict):
    """
    Normalizes the skeletal data for a given sequence of frames with signer's hand pose data. The normalization follows
    the definition from our paper.
    :param data: Dictionary containing key-value pairs with joint identifiers and corresponding lists (sequences) of
                that particular joints coordinates
    :return: Dictionary with normalized skeletal data (following the same schema as input data)
    """

    sequence_size = data.shape[0]
    
    # Treat each element of the sequence (analyzed frame) individually
    for sequence_index in range(sequence_size):

        # Retrieve all of the X and Y values of the current frame
        landmarks = data[sequence_index]
        x_values = landmarks[:, 0]
        y_values = landmarks[:, 1]

        # Prevent from even starting the analysis if some necessary elements are not present
        #if not landmarks_x_values or not landmarks_y_values:
        #    continue

        # Calculate the deltas
        width, height = np.ptp(x_values), np.ptp(y_values)
        if width > height:
            delta_x = 0.1 * width
            delta_y = delta_x + ((width - height) / 2)
        else:
            delta_y = 0.1 * height
            delta_x = delta_y + ((height - width) / 2)

        # Set the starting and ending point of the normalization bounding box
        starting_point = np.array([np.min(x_values) - delta_x, np.min(y_values) - delta_y])
        ending_point = np.array([np.max(x_values) + delta_x, np.max(y_values) + delta_y])

        for pos, kp in enumerate(data[sequence_index]):

            # Prevent from trying to normalize incorrectly captured points or in case of zero bounding box dimensions
            if (data[sequence_index][pos][0] == 0 or (ending_point[0] - starting_point[0]) == 0 or (starting_point[1] - ending_point[1]) == 0):
                continue

            normalized_x = (data[sequence_index][pos][0] - starting_point[0]) / (ending_point[0] - starting_point[0])
            normalized_y = (data[sequence_index][pos][1] - starting_point[1]) / (ending_point[1] - starting_point[1])

            data[sequence_index][pos][0] = normalized_x
            data[sequence_index][pos][1] = normalized_y

        '''
        # Normalize individual landmarks and save the results
        for pos, kp in enumerate(data[sequence_index]):

            # Prevent from trying to normalize incorrectly captured points
            if data[sequence_index][pos][0] == 0 or (ending_point[0] - starting_point[0]) == 0 or (
                    starting_point[1] - ending_point[1]) == 0:
                continue

            normalized_x = (data[sequence_index][pos][0] - starting_point[0]) / (ending_point[0] -
                                                                                    starting_point[0])
            normalized_y = (data[sequence_index][pos][1] - starting_point[1]) / (ending_point[1] -
                                                                                    starting_point[1])

            data[sequence_index][pos][0] = normalized_x
            data[sequence_index][pos][1] = normalized_y
        '''
    return data

###################################################################################
# This function normalize the body and the hands separately
# body_section has the general body part name (ex: pose, face, leftHand, rightHand)
# body_part has the specific body part name (ex: pose_left_shoulder, face_right_mouth_down, etc)
###################################################################################
def normalize_pose_hands_function(data, body_section, body_part):

    pose = [pos for pos, body in enumerate(body_section) if body == 'pose' or body == 'face']
    face = [pos for pos, body in enumerate(body_section) if body == 'face']
    leftHand = [pos for pos, body in enumerate(body_section) if body == 'leftHand']
    rightHand = [pos for pos, body in enumerate(body_section) if body == 'rightHand']

    body_section_dict = {body:pos for pos, body in enumerate(body_part)}

    assert len(pose) > 0 and len(leftHand) > 0 and len(rightHand) > 0 #and len(face) > 0

    #prepare_keypoints_image(data[2][0][leftHand+rightHand+pose,:],"before")

    for index_video in range(len(data)):
        data[index_video][:,pose,:] = normalize_pose(data[index_video][:,pose,:], body_section_dict)
        #data[index_video][:,face,:] = normalize_hand(data[index_video][:,face,:], body_section_dict)
        data[index_video][:,leftHand,:] = normalize_hand(data[index_video][:,leftHand,:], body_section_dict)
        data[index_video][:,rightHand,:] = normalize_hand(data[index_video][:,rightHand,:], body_section_dict)

    #prepare_keypoints_image(data[2][0][leftHand+rightHand+pose,:],"after")

    kp_bp_index = {'pose':pose,
                   'left_hand':leftHand,
                   'rigth_hand':rightHand}

    return data, kp_bp_index, body_section_dict

def limitIntancesPerClass(videos, labels, num_labels, video_names, limit_type="NC", instance_inc=20, increment_count='22-43-64'):
        
    dict_class = {_num_label:[] for _num_label in set(num_labels)}

    range_values = increment_count.split('-')
    range_values = [int(_i) for _i in range_values]
    range_pos = list(range(len(range_values)))[::-1]

    _init = 0
    pos = -1

    for _pos, _num_label in enumerate(num_labels):
        dict_class[_num_label].append(_pos)

    for _num_label in set(num_labels):

        if limit_type == "NC": 
            maximun = instance_inc
            dict_class[_num_label] = dict_class[_num_label][:maximun]

        elif limit_type == "NIC":
            for range_val, _pos in zip(range_values, range_pos):
                _end = range_val
                if _init <= _num_label < _end:
                    pos = _pos
                    break
                _init = range_val

            minimun = (pos)*instance_inc
            maximun = (pos+1)*instance_inc
            dict_class[_num_label] = dict_class[_num_label][minimun:maximun]

        elif limit_type == "exemplar":
            for range_val, _pos in zip(range_values, range_pos):
                _end = range_val
                if _init <= _num_label < _end:
                    pos = _pos
                    break
                _init = range_val
            
            minimun = 0 if pos==0 else instance_inc + 2 * (pos-1)
            maximun = instance_inc + 2 * pos
            dict_class[_num_label] = dict_class[_num_label][minimun:maximun]
 
        elif limit_type == "total":
            dict_class[_num_label] = dict_class[_num_label]

    ind_list = [ind for values in dict_class.values() for ind in values]

    for pos in range(len(labels)-1, -1, -1):
        if pos not in ind_list:
            labels.pop(pos)
            video_names.pop(pos)
            num_labels.pop(pos)
            videos.pop(pos)

    return videos, labels, num_labels, video_names

def get_dataset_from_hdf5(path,keypoints_model,words,landmarks_ref,keypoints_number,
                          threshold_frecuency_labels=10,
                          list_labels_banned=[],
                          dict_labels_dataset=None,
                          inv_dict_labels_dataset=None,
                          limit_type="NC",
                          instance_inc=20,
                          increment_count='22-43-64'):

    print('path                       :',path)
    print('keypoints_model            :',keypoints_model)
    print('landmarks_ref              :',landmarks_ref)
    print('threshold_frecuency_labels :',threshold_frecuency_labels)
    print('list_labels_banned         :',list_labels_banned)
    
    # Prepare the data to process the dataset

    index_array_column = None #'mp_indexInArray', 'wp_indexInArray','op_indexInArray'

    print('Use keypoint model : ',keypoints_model) 
    if keypoints_model == 'openpose':
        index_array_column  = 'op_indexInArray'
    if keypoints_model == 'mediapipe':
        index_array_column  = 'mp_indexInArray'
    if keypoints_model == 'wholepose':
        index_array_column  = 'wp_indexInArray'
    print('use column for index keypoint :',index_array_column)


    assert not index_array_column is None

    # all the data from landmarks_ref
    df_keypoints = pd.read_csv(landmarks_ref, skiprows=1)
    
    if keypoints_number == 29:
        df_keypoints = df_keypoints[(df_keypoints['Selected 29']=='x' )& (df_keypoints['Key']!='wrist')]
    elif keypoints_number == 71:
        df_keypoints = df_keypoints[(df_keypoints['Selected 71']=='x' )& (df_keypoints['Key']!='wrist')]
    else:
        df_keypoints = df_keypoints[(df_keypoints['Selected 54']=='x')]
    
    logging.info(" using keypoints_number: "+str(keypoints_number))

    idx_keypoints = sorted(df_keypoints[index_array_column].astype(int).values)
    name_keypoints = df_keypoints['Key'].values
    section_keypoints = (df_keypoints['Section']+'_'+df_keypoints['Key']).values

    print('section_keypoints : ',len(section_keypoints),' -- uniques: ',len(set(section_keypoints)))
    print('name_keypoints    : ',len(name_keypoints),' -- uniques: ',len(set(name_keypoints)))
    print('idx_keypoints     : ',len(idx_keypoints),' -- uniques: ',len(set(idx_keypoints)))
    print('')
    print('section_keypoints used:')
    print(section_keypoints)

    # process the dataset (start)

    print('Reading dataset .. ')
    data = get_data_from_h5(path)
    
    group = data['LSA64']
    _data = group['data']
    _label = group['label']
    _length = group['length']
    _class_number = group['class_number']
    _data_video_name = group['video_name']
    _shape = group['shape']

    print('Total size dataset : ',len(data.keys()))
    #print('Keys in dataset:', data.keys())

    labels_dataset = [value.decode('utf-8') for value in _label]
    
    video_dataset  = [np.transpose(np.array(value).reshape(length, _shape[0], _shape[1]), (0,2,1)) for value, length, lab in zip(_data, _length, labels_dataset) if lab in words]
    num_labels_dataset = [int(value) for value, lab in zip(_class_number, labels_dataset) if lab in words]
    video_name_dataset = [value.decode('utf-8') for value, lab in zip(_data_video_name, labels_dataset) if lab in words]
    labels_dataset = [value for value in labels_dataset if value in words]
    assert len(video_dataset) == len(labels_dataset) == len(num_labels_dataset) == len(video_name_dataset)

    false_seq_dataset = []
    percentage_dataset = []
    max_consec_dataset = []

    video_dataset, labels_dataset, num_labels_dataset, video_name_dataset = limitIntancesPerClass(video_dataset, 
                                                                                                  labels_dataset,
                                                                                                  num_labels_dataset,
                                                                                                  video_name_dataset,
                                                                                                  limit_type,
                                                                                                  instance_inc,
                                                                                                  increment_count)



    del data
    gc.collect()
    
    if dict_labels_dataset is None:

        dict_labels_dataset = {_class: _label for _class, _label in zip(labels_dataset, num_labels_dataset)}
        inv_dict_labels_dataset = {_label: _class for _class, _label in zip(labels_dataset, num_labels_dataset)}

    print('sorted(set(labels_dataset))  : ',sorted(set(labels_dataset)))
    print('dict_labels_dataset      :',dict_labels_dataset)
    print('inv_dict_labels_dataset  :',inv_dict_labels_dataset)
    encoded_dataset = num_labels_dataset
    print('encoded_dataset:',len(encoded_dataset))
    print('labe_dataset:',len(labels_dataset))
    print('data_dataset:',len(video_dataset))

    print('label encoding completed!')

    print('total unique labels : ',len(set(labels_dataset)))
    

    return video_dataset, video_name_dataset, labels_dataset, encoded_dataset, dict_labels_dataset, inv_dict_labels_dataset, df_keypoints['Section'], section_keypoints

class LSP_Dataset(Dataset):
    """Advanced object representation of the HPOES dataset for loading hand joints landmarks utilizing the Torch's
    built-in Dataset properties"""

    data: [np.ndarray]  # type: ignore
    labels: [np.ndarray]  # type: ignore

    def __init__(self, dataset_filename: str,keypoints_model:str, words=None, transform=None, have_aumentation=True,
                 augmentations_prob=0.5, normalize=False,landmarks_ref= 'Mapeo landmarks librerias.csv',
                dict_labels_dataset=None,inv_dict_labels_dataset=None, keypoints_number = 54,
                limit_type="NC", instance_inc=20, increment_count='22-43-64'):
        """
        Initiates the HPOESDataset with the pre-loaded data from the h5 file.

        :param dataset_filename: Path to the h5 file
        :param transform: Any data transformation to be applied (default: None)
        """
        print("*"*20)
        print("*"*20)
        print("*"*20)
        print('Use keypoint model : ',keypoints_model) 
        logging.info('Use keypoint model : '+str(keypoints_model))

        self.list_labels_banned = []

        if  'AEC' in  dataset_filename:
            self.list_labels_banned += []

        if  'PUCP' in  dataset_filename:
            self.list_labels_banned += []
            self.list_labels_banned += []

        if  'WLASL' in  dataset_filename:
            self.list_labels_banned += []

        print('self.list_labels_banned',self.list_labels_banned)
        logging.info('self.list_labels_banned '+str(self.list_labels_banned))

        video_dataset, video_name_dataset, labels_dataset, encoded_dataset, dict_labels_dataset, inv_dict_labels_dataset, body_section, body_part = get_dataset_from_hdf5(path=dataset_filename,
                                                                                                                                       keypoints_model=keypoints_model,
                                                                                                                                       words=words,
                                                                                                                                       landmarks_ref=landmarks_ref,
                                                                                                                                       keypoints_number = keypoints_number,
                                                                                                                                       threshold_frecuency_labels =0,
                                                                                                                                       list_labels_banned =self.list_labels_banned,
                                                                                                                                       dict_labels_dataset=dict_labels_dataset,
                                                                                                                                       inv_dict_labels_dataset=inv_dict_labels_dataset,
                                                                                                                                       limit_type=limit_type, 
                                                                                                                                       instance_inc=instance_inc,
                                                                                                                                       increment_count=increment_count)
        print("Normalizing data...")
        # HAND AND POSE NORMALIZATION
        video_dataset, keypoint_body_part_index, body_section_dict = normalize_pose_hands_function(video_dataset, body_section, body_part)

        self.data = video_dataset
        self.video_name = video_name_dataset
        #self.false_seq = false_seq_dataset
        #self.percentage = percentage_dataset
        #self.max_consec = max_consec_dataset
        self.labels = encoded_dataset
        self.label_freq = Counter(self.labels)
        #self.targets = list(encoded_dataset)
        self.text_labels = list(labels_dataset)
        self.transform = transform
        self.dict_labels_dataset = dict_labels_dataset
        self.inv_dict_labels_dataset = inv_dict_labels_dataset
        
        self.have_aumentation = have_aumentation
        #print(keypoint_body_part_index, body_section_dict)
        self.augmentation = augmentations.augmentation(keypoint_body_part_index, body_section_dict)
        self.augmentations_prob = augmentations_prob
        self.normalize = normalize


    def __getitem__(self, idx):
        """
        Allocates, potentially transforms and returns the item at the desired index.

        :param idx: Index of the item
        :return: Tuple containing both the depth map and the label
        """
        depth_map = torch.from_numpy(np.copy(self.data[idx]))


        # Apply potential augmentations
        if self.have_aumentation and random.random() < self.augmentations_prob:

            selected_aug = random.randrange(4)

            if selected_aug == 0:
                depth_map = self.augmentation.augment_rotate(depth_map, angle_range=(-13, 13))

            if selected_aug == 1:
                depth_map = self.augmentation.augment_shear(depth_map, "perspective", squeeze_ratio=(0, 0.1))

            if selected_aug == 2:
                depth_map = self.augmentation.augment_shear(depth_map, "squeeze", squeeze_ratio=(0, 0.15))

            if selected_aug == 3:
                depth_map = self.augmentation.augment_arm_joint_rotate(depth_map, 0.3, angle_range=(-4, 4))


        video_name = self.video_name[idx]
        #false_seq = self.false_seq
        #percentage_group = self.percentage
        #max_consec = self.max_consec
   
        label = torch.tensor([self.labels[idx]], dtype=torch.int64)

        depth_map = depth_map - 0.5
        if self.transform:
            depth_map = self.transform(depth_map)
        return depth_map, label, video_name #, false_seq, percentage_group, max_consec

    def __len__(self):
        return len(self.labels)

