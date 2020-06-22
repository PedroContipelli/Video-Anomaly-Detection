classes_json = 'anomaly_classes.json'
frame_counts = 'frame_counts.pkl'

path = '/home/crcvreu.student9/UCF_Crime'

c3d_features_folder = path + '/features/C3D/rgb'
i3d_features_folder = path + '/features/I3D/rgb/'
r2p1d_features_folder = path + '/features/R2P1D/rgb/'

train_split_file = path + '/Anomaly_Train.txt'
test_split_file = path + '/Anomaly_Test.txt'
test_annotations_file = path + '/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'

saved_models_dir = './results/saved_models'
tf_logs_dir = './results/logs'
