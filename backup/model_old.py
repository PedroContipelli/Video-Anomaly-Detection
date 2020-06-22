import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.linalg as sla


class FuturePredictionModel(nn.Module):
    def __init__(self, input_dim):
        super(FuturePredictionModel, self).__init__()
        self.encoder_1 = nn.Linear(input_dim, 1024)
        self.encoder_2 = nn.Linear(1024, 256)
        self.encoder_3 = nn.Linear(256, 64)
        self.decoder_1 = nn.Linear(64, 256)
        self.decoder_2 = nn.Linear(256, 1024)
        self.decoder_3 = nn.Linear(1024, input_dim)
        self.fc_1 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.4)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, original_features):
        (batch_size, bags_per_video, num_features) = original_features.shape
        original_features = original_features.view(-1, num_features)
        encoder_1 = self.activation(self.encoder_1(original_features))
        encoder_2 = self.activation(self.encoder_2(encoder_1))
        encoder_3 = self.activation(self.encoder_3(encoder_2))
        decoder_1 = self.activation(self.decoder_1(encoder_3))
        decoder_2 = self.activation(self.decoder_2(decoder_1))
        reconstructed_features = self.activation(self.decoder_3(decoder_2))
        classification_scores = self.sigmoid(self.fc_1(encoder_3))
        reconstructed_features = reconstructed_features.view(batch_size, bags_per_video, -1)
        classification_scores =  classification_scores.view(batch_size, bags_per_video, -1)
        return reconstructed_features, classification_scores


def load_model(input_dim, save_path):
    model = FuturePredictionModel(input_dim=input_dim)
    model.load_state_dict(torch.load(save_path)['state_dict'])
    return model


if __name__ == '__main__':
    features = Variable(torch.randn(8, 32, 4096).float()).cuda()
    model = FuturePredictionModel(input_dim=4096).cuda()
    reconstructed_features, classification_output = model(features)
    print(features.shape, reconstructed_features.shape, classification_output.shape)
