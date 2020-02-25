from binaryUtils import *


class NinNet(nn.Module):
    def __init__(self, net_type='full_precision'):
        super(NinNet, self).__init__()

        self.net_type = net_type
        self.input_size = [32]

        self.randomNet = nn.Sequential(
            myConv2d(3, 192, self.input_size, kernel_size=5, stride=1, padding=2, net_type='full_precision'),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            myConv2d(192, 160, self.input_size, kernel_size=1, stride=1, padding=0, net_type=self.net_type),
            myConv2d(160, 96, self.input_size, kernel_size=1, stride=1, padding=0),
            myMaxPool2d(kernel_size=3, stride=2, padding=1, input_size=self.input_size,),

            myConv2d(96, 192, self.input_size, kernel_size=5, stride=1, padding=2),
            myConv2d(192, 192, self.input_size, kernel_size=1, stride=1, padding=0),
            myConv2d(192, 192, self.input_size, kernel_size=1, stride=1, padding=0),
            myAvgPool2d(kernel_size=3, stride=2, padding=1, input_size=self.input_size),

            myConv2d(192, 192, self.input_size, kernel_size=3, stride=1, padding=1),
            myConv2d(192, 192, self.input_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
            myConv2d(192, 10, self.input_size, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            myAvgPool2d(kernel_size=8, stride=1, padding=0, input_size=self.input_size),
        )

    def forward(self, x):
        x = self.randomNet(x)
        x = x.view(x.size(0), 10)
        return x


