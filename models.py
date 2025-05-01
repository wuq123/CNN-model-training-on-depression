import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, input_channels):
        super(CNN, self).__init__()
        # 三个Conv-BatchNorm-ReLU-Dropout块
        self.conv_blocks = nn.Sequential(
            # 第一个块
            nn.Conv2d(input_channels, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            # 第二个块
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            # 第三个块
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )
        # 权重初始化
        nn.init.kaiming_normal_(self.conv_blocks[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_blocks[4].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_blocks[8].weight, mode='fan_out', nonlinearity='relu')

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接部分，添加更多层和Dropout
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CAFM(nn.Module):
    """
        Channel Attention Feature Modulation (CAFM) module

        Args:
            channels (int): Number of input channels
            reduction (int, optional): Reduction ratio for intermediate channels. Default: 16
        """

    def __init__(self, channels, reduction=16):
        super(CAFM, self).__init__()
        self.channels = channels
        self.reduction = reduction

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

        # Modulation parameters
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, _, _ = x.size()

        # Channel attention
        y = self.gap(x).view(b, c)
        y = self.attention(y).view(b, c, 1, 1)

        # Feature modulation
        out = x * y.expand_as(x)
        out = self.gamma * out + self.beta + x  # Residual connection

        return out


# input = tensor(8,29,80,2)
class CNNWithCAFM(nn.Module):
    def __init__(self, input_channels):
        super(CNNWithCAFM, self).__init__()

        # 第一个Conv-BatchNorm-ReLU-Dropout块
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )

        # 在第一个卷积块后加入CAFM
        self.cafm1 = CAFM(128)

        # 第二个Conv-BatchNorm-ReLU-Dropout块
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )

        # 在第二个卷积块后加入CAFM
        self.cafm2 = CAFM(128)

        # 第三个Conv-BatchNorm-ReLU-Dropout块
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )

        # 权重初始化
        nn.init.kaiming_normal_(self.conv_block1[0].weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv_block2[0].weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv_block3[0].weight, mode='fan_out', nonlinearity='leaky_relu')

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接部分
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.cafm1(x)  # 应用第一个CAFM

        x = self.conv_block2(x)
        x = self.cafm2(x)  # 应用第二个CAFM

        x = self.conv_block3(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class GRUModel(nn.Module):
    """
    特征通道为29的GRU模型

    参数:
        input_size (int): 输入特征维度, 这里设置为29
        hidden_size (int): 隐藏层特征维度
        num_layers (int): GRU层数, 默认为1
        bidirectional (bool): 是否使用双向GRU, 默认为False
        dropout (float): 非最后一层GRU的dropout比例, 默认为0
        num_classes (int): 输出类别数, 默认为1(用于回归或二分类)
    """

    def __init__(self, input_size=29, hidden_size=64, num_layers=1,
                 bidirectional=False, dropout=0.0, num_classes=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # GRU层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # 输入输出使用(batch, seq, feature)格式
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0  # 只有多层时才有dropout
        )

        # 全连接输出层
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """初始化GRU和全连接层的权重"""
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # x.shape = (batch_size, 29, 80)
        # 需要的x为形状: (batch_size, sequence_length, input_size=29)
        x = x.permute(0, 2, 1)  # 交换维度1和2

        # 初始化隐藏状态
        h0 = torch.zeros(
            self.num_layers * self.num_directions,
            x.size(0),
            self.hidden_size
        ).to(x.device)

        # GRU前向传播
        out, _ = self.gru(x, h0)  # out形状: (batch_size, seq_length, hidden_size * num_directions)

        # 取最后一个时间步的输出
        out = out[:, -1, :]  # 形状: (batch_size, hidden_size * num_directions)

        # 全连接层
        out = self.fc(out)  # 形状: (batch_size, num_classes)

        return out


class CNNWithCAFM256(nn.Module):
    def __init__(self, input_channels):
        super(CNNWithCAFM256, self).__init__()

        # 第一个Conv-BatchNorm-ReLU-Dropout块，扩大卷积核大小
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=(5, 5), padding=(2, 2)),  # 卷积核大小扩大到5x5
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )

        # 在第一个卷积块后加入CAFM
        self.cafm1 = CAFM(128)

        # 第二个Conv-BatchNorm-ReLU-Dropout块，扩大卷积核大小
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(5, 5), padding=(2, 2)),  # 卷积核大小扩大到5x5
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )

        # 在第二个卷积块后加入CAFM
        self.cafm2 = CAFM(128)

        # 第三个Conv-BatchNorm-ReLU-Dropout块，扩大卷积核大小
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(5, 5), padding=(2, 2)),  # 卷积核大小扩大到5x5
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )

        # 权重初始化
        nn.init.kaiming_normal_(self.conv_block1[0].weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv_block2[0].weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv_block3[0].weight, mode='fan_out', nonlinearity='leaky_relu')

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接部分
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.cafm1(x)  # 应用第一个CAFM

        x = self.conv_block2(x)
        x = self.cafm2(x)  # 应用第二个CAFM

        x = self.conv_block3(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class CNNWithCAFMdropto(nn.Module):
    def __init__(self, input_channels):
        super(CNNWithCAFMdropto, self).__init__()

        # 第一个Conv-BatchNorm-ReLU-Dropout块，扩大卷积核大小
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=(5, 5), padding=(2, 2)),  # 卷积核大小扩大到5x5
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )

        # 在第一个卷积块后加入CAFM
        self.cafm1 = CAFM(128)

        # 第二个Conv-BatchNorm-ReLU-Dropout块，扩大卷积核大小
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(5, 5), padding=(2, 2)),  # 卷积核大小扩大到5x5
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )

        # 在第二个卷积块后加入CAFM
        self.cafm2 = CAFM(128)

        # 第三个Conv-BatchNorm-ReLU-Dropout块，扩大卷积核大小
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(5, 5), padding=(2, 2)),  # 卷积核大小扩大到5x5
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )

        # 权重初始化
        nn.init.kaiming_normal_(self.conv_block1[0].weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv_block2[0].weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv_block3[0].weight, mode='fan_out', nonlinearity='leaky_relu')

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接部分
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.cafm1(x)  # 应用第一个CAFM

        x = self.conv_block2(x)
        x = self.cafm2(x)  # 应用第二个CAFM

        x = self.conv_block3(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class CNNWithCAFMMoreBlock(nn.Module):
    def __init__(self, input_channels):
        super(CNNWithCAFMMoreBlock, self).__init__()

        # 第一个Conv-BatchNorm-ReLU-Dropout块
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )

        # 在第一个卷积块后加入CAFM
        self.cafm1 = CAFM(128)

        # 第二个Conv-BatchNorm-ReLU-Dropout块
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )

        # 在第二个卷积块后加入CAFM
        self.cafm2 = CAFM(128)

        # 第三个Conv-BatchNorm-ReLU-Dropout块
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )
        self.cafm3 = CAFM(64)

        # 新增的第四个Conv-BatchNorm-ReLU-Dropout块
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )

        # 在第四个卷积块后加入CAFM
        self.cafm4 = CAFM(64)

        # 新增的第五个Conv-BatchNorm-ReLU-Dropout块
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )

        # 在第五个卷积块后加入CAFM
        self.cafm5 = CAFM(32)

        # 新增的第六个Conv-BatchNorm-ReLU-Dropout块
        self.conv_block6 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )

        # 权重初始化
        nn.init.kaiming_normal_(self.conv_block1[0].weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv_block2[0].weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv_block3[0].weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv_block4[0].weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv_block5[0].weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv_block6[0].weight, mode='fan_out', nonlinearity='leaky_relu')

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接部分
        self.classifier = nn.Sequential(
            nn.Linear(16, 128),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.cafm1(x)  # 应用第一个CAFM

        x = self.conv_block2(x)
        x = self.cafm2(x)  # 应用第二个CAFM

        x = self.conv_block3(x)
        x = self.cafm3(x)  # 应用第五个CAFM

        x = self.conv_block4(x)
        x = self.cafm4(x)  # 应用第三个CAFM

        x = self.conv_block5(x)
        x = self.cafm5(x)  # 应用第四个CAFM

        x = self.conv_block6(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
