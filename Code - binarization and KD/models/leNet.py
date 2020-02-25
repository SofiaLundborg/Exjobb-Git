from binaryUtils import *


class LeNet(nn.Module):
    def __init__(self, net_type='full_precision'):
        super(LeNet, self).__init__()

        self.net_type = net_type
        self.input_size = [32]

        self.superSimpleNetFeatures = nn.Sequential(
            myConv2d(3, 6, self.input_size, kernel_size=5, stride=1, padding=0, net_type=self.net_type, bias=True),
            myMaxPool2d(2, 2, input_size=self.input_size),
            myConv2d(6, 16, self.input_size, kernel_size=5, stride=1, padding=0, net_type=self.net_type, bias=True),
            myMaxPool2d(2, 2, input_size=self.input_size)
        )

        self.superSimpleNetClassifier = nn.Sequential(
            nn.Linear(16 * self.input_size[0] * self.input_size[0], 120),
            nn.ReLU(inplace=False),
            nn.Linear(120, 84),
            nn.ReLU(inplace=False),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.superSimpleNetFeatures(x)
        x = x.view(-1, 16 * self.input_size[0] * self.input_size[0])
        x = self.superSimpleNetClassifier(x)
        return x


