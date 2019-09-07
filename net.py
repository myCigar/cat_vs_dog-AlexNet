import torch.nn as nn
import torch.nn.functional as F


# 局部响应归一化
class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

# conv
# out_size = (in_size - kernel_size + 2 * padding) / stride
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        # conv
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)

        # LRN
        self.LRN = LRN(local_size=5, alpha=0.0001, beta=0.75)

        # FC
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2)

        # Dropout
        self.Dropout = nn.Dropout()



    def forward(self, x):
        # conv1 -> relu -> maxpool1
        # conv1: [n, 3, 227, 227] --> [n, 96, 55, 55]
        # maxpool1: [n, 96, 55, 55] --> [n, 96, 27, 27]
        x = F.relu(self.conv1(x))
        x = self.LRN(x)
        x = F.max_pool2d(x, (3, 3), 2)

        # conv2 -> relu -> maxpool2
        # conv2: [n, 96, 27, 27] --> [n, 256, 27, 27]
        # maxpool2: [n, 256, 27, 27] --> [n, 256, 13, 13]
        x = F.relu(self.conv2(x))
        x = self.LRN(x)
        x = F.max_pool2d(x, (3, 3), 2)

        # conv3 -> relu -> conv4 -> relu
        # oonv3: [n, 256, 13, 13] --> [n, 384, 13, 13]
        # conv4: [n, 384, 13, 13] --> [n, 384, 13, 13]
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # conv5 -> relu -> maxpool3
        # conv5: [n. 384, 13, 13] --> [n, 256, 13, 13]
        # maxpool3: [n, 256, 13, 13] --> [n, 256, 6, 6]
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, (3, 3), 2)

        # need view first for conv --> FC
        x = x.view(x.size()[0], -1)

        # fc1 -> fc2 -> fc3 -> softmax
        # fc1: 256*6*6 --> 4096
        # fc2: 4096 --> 4096
        # fc3: 4096 --> 2
        x = F.relu(self.fc1(x))
        x = self.Dropout(x)
        x = F.relu(self.fc2(x))
        x = self.Dropout(x)
        x = self.fc3(x)
        x = F.softmax(x)
        return x