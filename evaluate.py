import torch
from torch.autograd import Variable
from torch import nn
from sklearn.metrics import roc_curve, auc
import numpy as np
from dataloader import DataGenerator, DataLoader, filter_none
import configuration as cfg
from loss import CustomLoss
from model import load_model
import math


def evaluate(data_loader, model, criterion, use_cuda):

    model.eval()
    if use_cuda:
        model.cuda()

    pdist = nn.PairwiseDistance(p=2, keepdim=True)

    auc_scores = []
    for i, (features, labels, localization_scores) in enumerate(data_loader):
        assert len(features) == len(labels)
        num_samples = len(features)
        for j in range(num_samples):
            feature, label, localization_score = features[j], labels[j], localization_scores[j]
            if use_cuda:
                feature = Variable(torch.from_numpy(feature)).cuda()
            else:
                feature = Variable(torch.from_numpy(feature))

            predicted_feature, predicted_score = model(feature)

            loss = criterion(feature, predicted_feature, label, predicted_score)

            d = pdist(feature[1:], predicted_feature[:-1]).squeeze()
            d = d.cpu().data.numpy()

            predicted_score = predicted_score.cpu().data.numpy()

            predicted_score = predicted_score.flatten()
            #predicted_score = (predicted_score - min(predicted_score)) / (max(predicted_score) - min(predicted_score))
            #localization_score = (localization_score - min(localization_score)) / (max(localization_score) - min(localization_score))
            #d = (d - min(d)) / (max(d) - min(d))

            localization_groundtruth = np.array(localization_score).flatten()
            localization_predictions = np.array(predicted_score).flatten()
            false_positive_rate, true_positive_rate, thresholds = roc_curve(localization_groundtruth, localization_predictions)
            auc_score = auc(false_positive_rate, true_positive_rate)
            print(auc_score)
            if not math.isnan(auc_score):
                auc_scores.append(auc_score)
    print(np.mean(auc_scores))


if __name__ == '__main__':
    criterion = CustomLoss()
    model_save_path = '/home/c3-0/praveen/Research/Anomaly_Detection/Future_Prediction/results/saved_models/run1_C3D_05-02-20_1158/model_11_0.7862.pth'
    model = load_model(4096, model_save_path)
    features_folder = cfg.c3d_features_folder
    test_data_generator = DataGenerator('test', features_folder)
    test_dataloader = DataLoader(test_data_generator, batch_size=1, shuffle=False, num_workers=1, drop_last=True, collate_fn=filter_none, pin_memory=False)
    evaluate(test_dataloader, model, criterion, True)
