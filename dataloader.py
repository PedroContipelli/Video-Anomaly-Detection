import os
import random
from torch.utils.data import Dataset, DataLoader
import configuration as cfg
import parameters as params
import pickle, json
from utils.array_util import interpolate, interpolate_1D
from utils.data_util import build_groundtruth
import joblib
import numpy as np
from torch import nn
import torch
import time
import h5py as h5
import argparse


class DataGenerator(Dataset):
    def __init__(self, data_split, features_folder, args):
        self.data_file = cfg.train_split_file if data_split == 'train' else cfg.test_split_file
        self.data_split = data_split
        self.features_folder = features_folder
        assert self.data_split in ['train', 'test']
        self.args = args
        self.data_percentage = params.train_percent
        self.inputs = self.get_inputs()
        self.samples = self.build_samples()
        len_data = int(len(self.samples) * self.data_percentage)
        self.samples = self.samples[0:len_data]
        self.anomaly_classes = json.load(open(cfg.classes_json, 'r'))['classes']
  
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_id, label = self.samples[index]
        features_file = os.path.join(self.features_folder, video_id + '_x264.sav')
        #features_file = os.path.join(self.features_folder, video_id + '_x264.h5')
        if not os.path.exists(features_file):
            return None, None

        try:
            features = joblib.load(features_file, mmap_mode="r")
            #features = h5.File(features_file, 'r')['data']
            if self.args.interpolate_features > 0:
                features = interpolate(features, self.args.bags_per_video)
            features = np.array(features).squeeze()
        except:
            features = []

        if len(features) == 0:
            return None, None

        if self.data_split == 'test':
            localization_groundtruth = build_groundtruth(video_id, label)
            localization_groundtruth = interpolate_1D(localization_groundtruth, len(features))
            label = self.anomaly_classes[label]
            return features, label, localization_groundtruth

        label = self.anomaly_classes[label]
        return features, label, []

    def get_inputs(self):
        inputs = {'anomaly': [], 'normal': []}
        videos = open(self.data_file, 'r')
        for video in videos.readlines():
            video_id = '_'.join(video.rstrip().split('/')[1].split('_')[:-1])
            if 'Normal' in video_id:
                inputs['normal'].append((video_id, 'Normal'))
            else:
                inputs['anomaly'].append((video_id, video.split('/')[0]))
        return inputs

    def build_samples(self):
        anomaly_inputs = self.inputs['anomaly']
        normal_inputs = self.inputs['normal']
        samples = anomaly_inputs + normal_inputs
        return samples


def filter_none(batch):
    features, labels, localization = [], [], []
    for item in batch:
        if item[0] is not None and item[1] is not None and item[2] is not None:
            features.append(item[0])
            labels.append(item[1])
            localization.append(item[2])
    return np.array(features), np.array(labels), np.array(localization)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train Anomaly Classifier model')

    parser.add_argument('--interpolate_features', type=int, default=params.interpolate_features,
                        help='Flag to interpolate features.')
    parser.add_argument('--bags_per_video', type=int, default=params.bags_per_video,
                        help='Flag to interpolate features.')

    args = parser.parse_args()
    data_generator = DataGenerator('train', cfg.c3d_features_folder, args)
    dataloader = DataLoader(data_generator, batch_size=params.batch_size, shuffle=True, num_workers=4, drop_last=True, collate_fn=filter_none)
    for i, (features, label, _) in enumerate(dataloader):
        print(i, features.shape, label)
        break
    data_generator = DataGenerator('test', cfg.c3d_features_folder, args)
    dataloader = DataLoader(data_generator, batch_size=int(params.batch_size), shuffle=True, num_workers=4, drop_last=True, collate_fn=filter_none)
    for i, (features, label, groundtruth) in enumerate(dataloader):
        print(i, features.shape, label, groundtruth)
        break
