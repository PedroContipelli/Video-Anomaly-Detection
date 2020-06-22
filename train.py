from dataloader import *
from model import build_model 
from torch.nn import CrossEntropyLoss
from torch.nn.init import xavier_normal_, xavier_uniform_, kaiming_normal_, kaiming_uniform_
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score
import torch
import os
from torch import nn
import numpy as np
from torch.autograd import Variable
import configuration as cfg
from loss import CustomLoss
from torchcontrib.optim import SWA

anomaly_classes = json.load(open(cfg.classes_json, 'r'))['classes']
anomaly_classes = [k for k in sorted(anomaly_classes, key=anomaly_classes.get)]


def weights_init_kaiming_normal(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        kaiming_normal_(m.weight.data)
        m.bias.data.fill_(1.0)


def weights_init_kaiming_uniform(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        kaiming_uniform_(m.weight.data)
        m.bias.data.fill_(1.0)


def weights_init_xavier_normal(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        xavier_normal_(m.weight.data)
        m.bias.data.fill_(1.0)


def weights_init_xavier_uniform(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        xavier_uniform_(m.weight.data)
        m.bias.data.fill_(1.0)


def train_epoch(run_id, epoch, data_loader, model, optimizer, criterion, writer, use_cuda, args):
    print('train at epoch {}'.format(epoch))

    losses = []

    model.train()
    if use_cuda:
        model.cuda()

    for i, (features, labels, _) in enumerate(data_loader):
        assert len(features) == len(labels)
        num_samples = len(features)
        loss = 0

        if args.interpolate_features > 0:
            features = np.array(features, dtype='f')
            labels = np.array(labels, dtype='long')

            if use_cuda:
                features = Variable(torch.from_numpy(features)).cuda()
            else:
                features = Variable(torch.from_numpy(features))

            optimizer.zero_grad()

#           predicted_features, predicted_scores = model(features)
            predicted_features, predicted_scores, _ = model(features)

            loss = criterion(features, predicted_features, labels, predicted_scores)

        else:

            for j in range(num_samples):
                feature, label = features[j], labels[j]
                if use_cuda:
                    feature = Variable(torch.from_numpy(np.array([feature]))).cuda()
                else:
                    feature = Variable(torch.from_numpy(np.array([feature])))

                optimizer.zero_grad()

#               predicted_feature, predicted_score = model(feature)
                predicted_feature, predicted_score, _ = model(feature)

                loss += criterion(feature, predicted_feature, [label], predicted_score)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if i % 10 == 0:
            print("Training Epoch ", epoch, " Batch ", i, "- Loss : ", loss.item())
        del loss

    if 0 < args.swa_start < epoch:
        optimizer.update_swa()
        if epoch % params.swa_update_interval == 0:
            optimizer.swap_swa_sgd()

    print('Training Epoch: %d, Loss: %.4f' % (epoch, np.mean(losses)))

    writer.add_scalar('Training Loss', np.mean(losses), epoch)
    return model


def val_epoch(epoch, data_loader, model, criterion, writer, use_cuda, args):
    print('validation at epoch {}'.format(epoch))

    model.eval()
    if use_cuda:
        model.cuda()

    losses = []
    localization_groundtruth, localization_predictions = [], []
    for i, (features, labels, localization_scores) in enumerate(data_loader):
        assert len(features) == len(labels)
        num_samples = len(features)
        loss = 0

        if args.interpolate_features > 0:
            features = np.array(features, dtype='f')
            labels = np.array(labels, dtype='long')

            if use_cuda:
                features = Variable(torch.from_numpy(features)).cuda()
            else:
                features = Variable(torch.from_numpy(features))

#           predicted_features, predicted_scores = model(features)
            predicted_features, predicted_scores, _ = model(features)

            loss = criterion(features, predicted_features, labels, predicted_scores)

            localization_predictions.extend(predicted_scores.cpu().data.numpy())
            localization_groundtruth.extend(localization_scores)

        else:

            for j in range(num_samples):
                feature, label, localization_score = features[j], labels[j], localization_scores[j]
                if use_cuda:
                    feature = Variable(torch.from_numpy(np.array([feature]))).cuda()
                else:
                    feature = Variable(torch.from_numpy(np.array([feature])))

#               predicted_feature, predicted_score = model(feature)
                predicted_feature, predicted_score, _ = model(feature)

                loss += criterion(feature, predicted_feature, [label], predicted_score)

                localization_predictions.extend(predicted_score.cpu().data.numpy().flatten())
                localization_groundtruth.extend(localization_score)

        losses.append(loss.item())

        if i % 10 == 0:
            print("Validation Epoch ", epoch, " Batch ", i, "- Loss : ", loss.item())
        del loss

    localization_groundtruth = np.array(localization_groundtruth).flatten()
    localization_predictions = np.array(localization_predictions).flatten()
    false_positive_rate, true_positive_rate, thresholds = roc_curve(localization_groundtruth, localization_predictions)
    auc_score = auc(false_positive_rate, true_positive_rate)

    print('Validation Epoch: %d, AUC Score: %.4f' % (epoch, auc_score))

    writer.add_scalar('Validation Loss', np.mean(losses), epoch)
    writer.add_scalar('Validation AUC Score', auc_score, epoch)

    return auc_score


def train_model(run_id, save_dir, use_cuda, args, writer):

    print("Run ID : " + args.run_id)
    print("Features used : ", args.features)

    print("Parameters used : ")
    print("batch_size: " + str(args.batch_size))
    print("lr: " + str(args.learning_rate))

    if args.features == 'c3d':
        features_folder = cfg.c3d_features_folder
    if args.features == 'i3d':
        features_folder = cfg.i3d_features_folder
    if args.features == 'r2p1d':
        features_folder = cfg.r2p1d_features_folder

    train_data_generator = DataGenerator('train', features_folder, args)
    train_dataloader = DataLoader(train_data_generator, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, collate_fn=filter_none, pin_memory=False)

    test_data_generator = DataGenerator('test', features_folder, args)
    test_dataloader = DataLoader(test_data_generator, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=True, collate_fn=filter_none, pin_memory=False)

    print("Number of training samples : " + str(len(train_data_generator)))
    steps_per_epoch = len(train_data_generator) / args.batch_size
    print("Steps per epoch: " + str(steps_per_epoch))

    print("Number of validation samples : " + str(len(test_data_generator)))
    steps_per_epoch = len(test_data_generator) / args.batch_size
    print("Steps per epoch: " + str(steps_per_epoch))

#   model = FuturePredictionModel(input_dim=args.input_dim)
    model = build_model(args.input_dim, 8)                     # make nheads an argument later

    if args.model_init == 'kaiming_normal':
        model.apply(weights_init_kaiming_normal)
    elif args.model_init == 'kaiming_uniform':
        model.apply(weights_init_kaiming_uniform)
    elif args.model_init == 'xavier_normal':
        model.apply(weights_init_xavier_normal)
    elif args.model_init == 'xavier_uniform':
        model.apply(weights_init_xavier_uniform)

    if args.optimizer == 'ADAM':
        print("Using ADAM optimizer")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.swa_start > 0:
        optimizer = SWA(optimizer)

    criterion = CustomLoss()

    max_auc_score = 0
    # loop for each epoch
    for epoch in range(args.num_epochs):
        model = train_epoch(run_id, epoch, train_dataloader, model, optimizer, criterion, writer, use_cuda, args)
        auc_score = val_epoch(epoch, test_dataloader, model, criterion, writer, use_cuda, args)

        if auc_score > max_auc_score:
            save_file_path = os.path.join(save_dir, 'model_{}_{:.4f}.pth'.format(epoch, auc_score))
            states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
            max_auc_score = auc_score

        train_data_generator = DataGenerator('train', features_folder, args)
        train_dataloader = DataLoader(train_data_generator, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, collate_fn=filter_none, pin_memory=False)

        test_data_generator = DataGenerator('test', features_folder, args)
        test_dataloader = DataLoader(test_data_generator, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True, collate_fn=filter_none, pin_memory=False)


