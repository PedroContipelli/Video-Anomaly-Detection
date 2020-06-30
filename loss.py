import torch
import torch.nn as nn


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, features, predicted_features, labels, predicted_scores):
        loss = 0
        for i in range(len(features)):
            pdist = nn.PairwiseDistance(p=2, keepdim=True)
            predicted_score = predicted_scores[i].squeeze()
            d = pdist(features[i][1:], predicted_features[i][:-1]).squeeze()
            max_index = torch.argmax(d)
            
            if labels[i] == 13: # Not Anomalous
                loss += d[max_index] + predicted_score[max_index]
            else: # Anomalous
                loss += 1 - predicted_score[max_index]
        return loss

