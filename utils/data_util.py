import configuration as cfg
from skvideo.io import FFmpegReader 
#import cv2
import os
import numpy as np
from utils.array_util import interpolate_1D
import parameters as params
import pickle


def build_groundtruth(video_id, label):
    frame_counts = pickle.load(open(cfg.frame_counts, 'rb'))
    test_annotations = [line.rstrip().split('  ') for line in open(cfg.test_annotations_file, 'r').readlines()]
    video_ids = ['_'.join(annotation[0].split('_')[:-1]) for annotation in test_annotations]
    assert video_id in video_ids
    # video_path = os.path.join(cfg.videos_folder, label, video_id + '_x264.mp4')
    # assert os.path.exists(video_path)
    num_frames = frame_counts[video_id]
    ground_truth = np.zeros(num_frames)
    for annotation in test_annotations:
        if video_id in annotation[0]:
            start_1 = int(annotation[2])
            end_1 = int(annotation[3])
            start_2 = int(annotation[4])
            end_2 = int(annotation[5])
            ground_truth[start_1:end_1] = 1
            ground_truth[start_2:end_2] = 1
    return ground_truth


def count_frames():
    frame_counts = {}
    test_annotations = [line.rstrip().split('  ') for line in open(cfg.test_annotations_file, 'r').readlines()]
    video_ids = ['_'.join(annotation[0].split('_')[:-1]) for annotation in test_annotations]
    for video_id in video_ids:
        print(video_id)
        label = 'Normal' if 'Normal' in video_id else video_id[:-3]
        video_path = os.path.join(cfg.videos_folder, label, video_id + '_x264.mp4')
        assert os.path.exists(video_path)
        vid_reader = FFmpegReader(video_path)
        (num_frames, _, _, _) = vid_reader.getShape()
        frame_counts[video_id] = num_frames
    pickle.dump(frame_counts, open('frame_counts.pkl', 'wb'))


if __name__ == '__main__':
    count_frames()
