from __future__ import print_function
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import data_loader
import Models
import time
from torch.utils import model_zoo
import numpy as np
import pandas as pd
import os
from shutil import rmtree, copytree, copyfile
import random
from sklearn.metrics import roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=12, help='batch size')
parser.add_argument('--model', type=str, default='resnet101', help='coco.data file path')
parser.add_argument('--gpu', type=str, default='0', help='coco.data file path')
parser.add_argument('--num_class', type=int, default=2, help='coco.data file path')
parser.add_argument('--random_seed', type=int, default=1, help='coco.data file path')
parser.add_argument('--split_train_ratio', type=float, default=0.8, help='coco.data file path')
parser.add_argument('--task_name', type=str, default='ThyNet', help='coco.data file path')
parser.add_argument('--path', type=str, default='data/', help='coco.data file path')
parser.add_argument('--auto_split', type=str, default='1', help='coco.data file path')
arg = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
# Training settings
batch_size = arg.batch_size
epochs = 400
lr = 0.01
momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10
l2_decay = 5e-4
random_seed = int(arg.random_seed)
split_train_ratio = arg.split_train_ratio
path = arg.path
source_name = "train"
target_name = "val"

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}


def split_data():
    for name in [source_name, target_name]:
        if os.path.exists(os.path.join(path, name)):
            rmtree(os.path.join(path, name))
            os.makedirs(os.path.join(path, name))
        else:
            os.makedirs(os.path.join(path, name))

    tmp = os.listdir(path)
    tmp = [i for i in tmp if i not in [source_name, target_name]]
    for properties in tmp:
        files = os.listdir(os.path.join(path, properties))
        random.seed(random_seed)
        random.shuffle(files)
        for file in files[: int(len(files) * split_train_ratio)]:
            if not os.path.exists(os.path.join(path, 'train', properties)):
                os.makedirs(os.path.join(path, 'train', properties))
            copyfile(os.path.join(path, properties, file),
                     os.path.join(path, 'train', properties, file)
                     )
        for file in files[int(len(files) * split_train_ratio):]:
            if not os.path.exists(os.path.join(path, 'val', properties)):
                os.makedirs(os.path.join(path, 'val', properties))
            copyfile(os.path.join(path, properties, file),
                     os.path.join(path, 'val', properties, file)
                     )
    print('complete data split')

if arg.auto_split == '1':
    split_data()
else:
    pass

num_classes = len(os.listdir(os.path.join(path, source_name)))
source_loader = data_loader.load_training(path, source_name, batch_size, kwargs)
target_test_loader, names, label = data_loader.load_testing(path, target_name, batch_size, kwargs)

len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
len_source_loader = len(source_loader)


def draw_auc(fpr, tpr, auc_value, filepath):
    fig = plt.figure()
    plt.title("auc:{}".format(round(auc_value, 4)))
    plt.xlabel('FPR')
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    ax1 = fig.add_subplot(111)
    ax1.plot(fpr, tpr)
    ax1.legend()
    ax1.set_ylabel('TPR')
    plt.savefig(filepath.replace('.jpg', '.eps'), format='eps', dpi=600, bbox_inches='tight')
    plt.savefig(filepath)


def youden(tpr, fpr, thresholds):
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_idx, optimal_threshold


def save_dict(model):
    dict = model.module.state_dict() if type(
        model) is nn.parallel.DistributedDataParallel else model.state_dict()
    if not os.path.exists('model/{}'.format(arg.task_name)):
        os.makedirs('model/{}'.format(arg.task_name))
    torch.save(dict, 'model/{}/{}.pth'.format(arg.task_name, arg.model))


def save_csv(df, specific, sensitivity):
    if not os.path.exists('csv'):
        os.makedirs('csv')
    csv_path = 'csv/{}.csv'.format(arg.task_name)
    pd.DataFrame(
        columns=['Specific:{:.4f} sensitivity:{:.4f}'.format(float(specific), float(sensitivity))]).to_csv(
        csv_path, index=False)
    df.to_csv(csv_path, index=False, mode='a')


def save_roc_auc(fpr, tpr, auc_value, filename):
    if not os.path.exists('pic'):
        os.makedirs('pic')
    draw_auc(fpr, tpr, auc_value, os.path.join('pic', '{}.jpg'.format(filename)))


def train(epoch, model):
    LEARNING_RATE = max(lr * (0.1 ** (epoch // 100)), 1e-5)
    optimizer = torch.optim.SGD([
        {'params': model.parameters()}
    ], lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)

    model.train()
    for data, label in source_loader:
        data = data.float().cuda()
        label = label.long().cuda()
        pred = model(data)
        optimizer.zero_grad()
        loss = F.nll_loss(F.log_softmax(pred, dim=1), label)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} loss :{}, learning_rate:{}'.format(epoch, loss.item(), LEARNING_RATE))


def test(model, best_auc):
    model.eval()
    test_loss = 0
    correct = 0
    possbilitys = None
    for data, target in target_test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()

        s_output = model(data)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim=1), target, size_average=False).item()  # sum up batch loss
        pred = s_output.data.max(1)[1]  # get the index of the max log-probability
        possbility = F.softmax(s_output).cpu().data.numpy()
        if possbilitys is None:
            possbilitys = possbility
        else:
            possbilitys = np.append(possbilitys, possbility, axis=0)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    label_onehot = np.eye(num_classes)[np.array(label).astype(np.int32).tolist()]
    fpr, tpr, thresholds = roc_curve(label_onehot.ravel(), possbilitys.ravel())
    index, optimal_threshold = youden(fpr, tpr, thresholds)
    auc_value = auc(fpr, tpr)
    test_loss /= len_target_dataset
    print('Specific:{} sensitivity:{} Auc:{}'.format(1 - fpr[index], tpr[index], auc_value))
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        target_name, test_loss, correct, len_target_dataset,
        100. * correct / len_target_dataset))
    return best_auc


if __name__ == '__main__':
    if arg.model == 'resnet101':
        model = Models.Resnet101(num_classes=num_classes)
    if arg.model == 'resnext101':
        model = Models.Resnext101(num_classes=num_classes)
    if arg.model == 'densenet201':
        model = Models.Densnet201(num_classes=num_classes)

    correct = 0
    print(model)
    model = torch.nn.DataParallel(model, device_ids=list(range(len(arg.gpu.split(',')))))
    model.cuda()
    best_auc = 0
    for epoch in range(1, epochs + 1):
        train(epoch, model)
        with torch.no_grad():
            best_auc = test(model, best_auc)
        dict = model.module.state_dict() if type(
            model) is nn.parallel.DistributedDataParallel else model.state_dict()
        if not os.path.exists(os.path.join('model', arg.task_name)):
            os.makedirs(os.path.join('model', arg.task_name))
        torch.save(dict, os.path.join('model', arg.task_name, arg.model + '_' + str(epoch) + '.pth'))