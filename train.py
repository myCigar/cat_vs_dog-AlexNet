import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from datasets import CatDogDataset
from utils import save_model

from net import AlexNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=160, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--load', type=bool, default=False, help='load file model')
    # parser.add_argument('--data_path', type=str, default='/home/ccit16/jh_test/dataset/猫狗数据集/', help='dataset path')
    parser.add_argument('--data_path', type=str, default='F:/dataset/猫狗数据集/kaggle/train', help='dataset path')
    parser.add_argument('--test', type=bool, default=True, help='calc acc if the value is set True')
    args = parser.parse_args()
    print(args)
    return args


def eval(alexnet, test_loader, test_dataset, device):
    print('train:{} imgs'.format(len(test_dataset)))
    true_count = 0
    for idx, (imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        pre_labels = alexnet(imgs)
        pre_labels = pre_labels.squeeze(0)
        pre_item_label = -1
        if pre_labels[0].item() > pre_labels[1].item():
            pre_item_label = 0
        else:
            pre_item_label = 1

        if pre_item_label == labels.item():
            true_count += 1
            print('[{}/{}] true'.format(idx+1, len(test_dataset)))
        else:
            print('[{}/{}] false'.format(idx+1, len(test_dataset)))

    acc = true_count / len(test_dataset)
    return acc


def main():
    args = get_args()

    pre_transforms = transforms.Compose([
        transforms.Resize((227, 227)),  # 要求PIL image
        transforms.ToTensor(),  # 所以ToTensor放中间
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 要求Tensor img
    ])

    # get datasets
    train_dataset = CatDogDataset(args, 'train', pre_transforms)
    test_dataset = CatDogDataset(args, 'test', pre_transforms)

    # length
    print('train:{} imgs'.format(len(train_dataset)))

    # generate DataLoader
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, 1, shuffle=False)

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    alexnet = AlexNet()
    base_epoch = 0
    if args.load:
        model_path = './checkpoints/99_loss_0.523277.pth'
        alexnet.load_state_dict(torch.load(model_path)['alexnet'])
        base_epoch = torch.load(model_path)['epoch']

    alexnet.to(device)

    # loss and optim function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(alexnet.parameters(),
                    lr=args.lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(args.epochs):
        alexnet.train()
        epoch += base_epoch
        epoch_loss = 0
        for idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            pre_labels = alexnet(imgs)

            optimizer.zero_grad()
            loss = criterion(pre_labels, labels.long())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print('[{}/{}][{}/{}] loss:{:.4f}'
                  .format(epoch+1, args.epochs, idx+1, int(len(train_dataset) / args.batch_size), loss.item()))

        # save model
        aver_loss = epoch_loss * args.batch_size / len(train_dataset)
        state = {
            'epoch': epoch,
            'alexnet': alexnet.state_dict()
        }
        if args.test:
            acc = eval(alexnet, test_loader, test_dataset, device)
            save_model(state, './checkpoints', '{}_{:.6f}_{:.3f}.pth'.format(epoch, aver_loss, acc))


if __name__ == '__main__':
    main()
