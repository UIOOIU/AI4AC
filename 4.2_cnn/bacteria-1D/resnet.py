"""一维 ResNet 网络结构定义（用于光谱分类）。

包含残差块 ResidualBlock、主网络 ResNet（Conv1d 版本）以及激活函数工厂 add_activation。
供细菌-1D 光谱分类训练脚本构建分类模型使用。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ResidualBlock(nn.Module):
    """一维残差块：主路为两层「Conv1d+BN」，加捷径（shortcut）后过 ReLU。"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Layers
        # 卷积核为 5、padding=2 保持长度不变，仅 conv1 用 stride 控制降采样
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5,
            stride=stride, padding=2, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5,
            stride=1, padding=2, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)

        # 当输入输出维度不一致（stride != 1 或通道数不同）时，
        # 用 1x1 卷积 + BN 对齐维度，保证捷径分支可与主路相加
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False),
                nn.BatchNorm1d(out_channels))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 主路第一层：Conv-BN-ReLU
        out = self.bn2(self.conv2(out))        # 主路第二层：Conv-BN（暂不加激活）
        out += self.shortcut(x)                # 加入捷径分支
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """一维 ResNet：对光谱信号先卷积提取特征，再经多个残差块堆叠编码，最后由全连接层映射为各类别得分。"""
    def __init__(self, hidden_sizes, num_blocks, input_dim=1000,
        in_channels=64, n_classes=30):
        super(ResNet, self).__init__()
        # 每个阶段的残差块数量需与阶段数（hidden_sizes 长度）一一对应
        assert len(num_blocks) == len(hidden_sizes)
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.n_classes = n_classes
        
        # 输入层：把一维光谱当作单通道信号，卷积核 5、长度不变
        self.conv1 = nn.Conv1d(1, self.in_channels, kernel_size=5, stride=1,
            padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        
        # Flexible number of residual encoding layers
        layers = []
        # 第一阶段步长为 1，其后各阶段步长为 2，逐级将序列长度减半
        strides = [1] + [2] * (len(hidden_sizes) - 1)
        for idx, hidden_size in enumerate(hidden_sizes):
            layers.append(self._make_layer(hidden_size, num_blocks[idx],
                stride=strides[idx]))
        self.encoder = nn.Sequential(*layers)

        # 自动推断展平后的编码维度，作为全连接分类器的输入
        self.z_dim = self._get_encoding_size()
        self.linear = nn.Linear(self.z_dim, self.n_classes)


    def encode(self, x):
        """卷积特征提取：输入 (N,1,L) -> 输出展平编码 z (N, z_dim)。"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.encoder(x)
        z = x.view(x.size(0), -1)  # 将每个样本的通道与长度维展平
        return z

    def forward(self, x):
        """前向传播：先编码，再经全连接层输出类别得分。"""
        z = self.encode(x)
        return self.linear(z)


    def _make_layer(self, out_channels, num_blocks, stride=1):
        """构建一个阶段：顺序堆叠 num_blocks 个残差块，仅第一个块用 stride 降采样。"""
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(ResidualBlock(self.in_channels, out_channels,
                stride=stride))
            self.in_channels = out_channels  # 每个阶段结束后通道数更新为 out_channels
        return nn.Sequential(*blocks)

    def _get_encoding_size(self):
        """
        Returns the dimension of the encoded input.
        """
        # 用随机张量跑一遍网络，以输出宽度作为编码维度
        temp = Variable(torch.rand(1, 1, self.input_dim))
        z = self.encode(temp)
        z_dim = z.data.size(1)
        return z_dim


def add_activation(activation='relu'):
    """
    Adds specified activation layer, choices include:
    - 'relu'
    - 'elu' (alpha)
    - 'selu'
    - 'leaky relu' (negative_slope)
    - 'sigmoid'
    - 'tanh'
    - 'softplus' (beta, threshold)
    """
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'elu':
        return nn.ELU(alpha=1.0)
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'leaky relu':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    # SOFTPLUS DOESN'T WORK with automatic differentiation in pytorch
    elif activation == 'softplus':
        return nn.Softplus(beta=1, threshold=20)