import os
import argparse

import torch
import torchvision.transforms as transforms

from net import AlexNet
from torch.utils.data import DataLoader
from datasets import TestDataset

from torchvision.datasets import

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training')
    parser.add_argument('--data_path', type=str, default='F:/dataset/猫狗数据集/kaggle/test', help='dataset path')
    parser.add_argument('--model_path', type=str, default='./checkpoints/135_0.321503_0.943.pth', help='model path')
    args = parser.parse_args()
    print(args)
    return args


def eval():
    args = get_args()

    pre_transforms = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = TestDataset(args, pre_transforms)
    print(len(test_dataset))
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model and to GPU
    alexnet = AlexNet()
    alexnet.load_state_dict(torch.load(args.model_path)['alexnet'])
    alexnet.to(device)

    alexnet.eval()
    file = open('sampleSubmission.csv', 'w+')
    file.writelines('id,label\n')
    for idx, (imgs) in enumerate(test_loader):
        imgs = imgs.to(device)

        pre_labels = alexnet(imgs)
        pre_labels = pre_labels.squeeze(0)
        pre_item_label = -1
        if pre_labels[0].item() > pre_labels[1].item():
            pre_item_label = 1 - pre_labels[0].item()
        else:
            pre_item_label = pre_labels[1].item()

        file.writelines(str(idx+1) + ',' + str(pre_item_label) + '\n')
        print('[{}/{}] {} {:.6f}'.format(idx+1, len(test_dataset), 'cat' if pre_item_label < 0.5 else 'dog', pre_item_label))


if __name__ == '__main__':
    eval()

