from torch import nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
    )


class Residual_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = conv_block(in_channels, out_channels)
        self.conv2 = conv_block(out_channels, out_channels)
        self.conv3 = conv_block(out_channels, out_channels)

    def forward(self, x):
        res = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x + res


def trunk_branch(in_channels, out_channels):
    return nn.Sequential(
        Residual_block(in_channels, out_channels),
        Residual_block(out_channels, out_channels)
    )


def encoder_block(in_channels, out_channels):
    return nn.Sequential(
        conv_block(in_channels, out_channels),
        Residual_block(out_channels, out_channels),
        conv_block(out_channels, out_channels)
    )


class RUnet(nn.Module):
    def __init__(self, in_channels=3, in_features=64, out_channels=1):
        super(RUnet, self).__init__()
        self.enc1 = encoder_block(in_channels, in_features)
        self.down_sample1 = nn.MaxPool2d(2, 2)
        self.trunk_branch1 = trunk_branch(in_features, in_features)

        self.enc2 = encoder_block(in_features, 2 * in_features)
        self.trunk_branch2 = trunk_branch(2 * in_features, 2 * in_features)
        self.down_sample2 = nn.MaxPool2d(2, 2)

        self.enc3 = encoder_block(2 * in_features, 4 * in_features)
        self.trunk_branch3 = trunk_branch(4 * in_features, 4 * in_features)
        self.down_sample3 = nn.MaxPool2d(2, 2)

        self.enc4 = encoder_block(4 * in_features, 8 * in_features)
        self.trunk_branch4 = trunk_branch(8 * in_features, 8 * in_features)
        self.down_sample4 = nn.MaxPool2d(2, 2)

        self.enc5 = encoder_block(8 * in_features, 16 * in_features)

        self.up_sample1 = nn.ConvTranspose2d(16 * in_features, 8 * in_features, kernel_size=2, stride=2)
        self.decod1 = encoder_block(8 * in_features, 8 * in_features)

        self.up_sample2 = nn.ConvTranspose2d(8 * in_features, 4 * in_features, kernel_size=2, stride=2)
        self.decod2 = encoder_block(4 * in_features, 4 * in_features)

        self.up_sample3 = nn.ConvTranspose2d(4 * in_features, 2 * in_features, kernel_size=2, stride=2)
        self.decod3 = encoder_block(2 * in_features, 2 * in_features)

        self.up_sample4 = nn.ConvTranspose2d(2 * in_features, in_features, kernel_size=2, stride=2)
        self.decod4 = nn.Sequential(conv_block(in_features, in_features),
                                    Residual_block(in_features, in_features),
                                    conv_block(in_features, out_channels))

    def forward(self, input):
        input = self.enc1(input)
        trunk1 = self.trunk_branch1(input)
        input = self.down_sample1(input)

        input = self.enc2(input)
        trunk2 = self.trunk_branch2(input)
        input = self.down_sample2(input)

        input = self.enc3(input)
        trunk3 = self.trunk_branch3(input)
        input = self.down_sample3(input)

        input = self.enc4(input)
        trunk4 = self.trunk_branch4(input)
        input = self.down_sample4(input)

        input = self.enc5(input)

        input = self.up_sample1(input)
        input *= trunk4
        input += trunk4
        input = self.decod1(input)

        input = self.up_sample2(input)
        input *= trunk3
        input += trunk3
        input = self.decod2(input)

        input = self.up_sample3(input)
        input *= trunk2
        input += trunk2
        input = self.decod3(input)

        input = self.up_sample4(input)
        input *= trunk1
        input += trunk1
        input = self.decod4(input)
        sigmoid = nn.Sigmoid()
        return sigmoid(input)
