"""训练/评估工具函数。

run_epoch 执行一个 epoch 的训练或验证，返回准确率与平均损失；
get_predictions 在 DataLoader 上收集模型的预测类别或类别概率。
供细菌-1D 光谱分类训练脚本调用。
"""
from torch.autograd import Variable
from torch import nn
import torch
import numpy as np


def run_epoch(epoch, model, dataloader, cuda, training=False, optimizer=None):
    """运行一个 epoch：training=True 时反向传播更新参数；返回 (准确率%, 平均损失)。"""
    # 切换训练/评估模式（影响 BatchNorm、Dropout 等层的行为）
    if training:
        model.train()
    else:
        model.eval()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if cuda: inputs, targets = inputs.cuda(), targets.cuda()  # 若指定 cuda 则在 GPU 上计算
        inputs, targets = Variable(inputs), Variable(targets.long())  # 交叉熵损失要求整数类别标签
        outputs = model(inputs)
        # Compute the loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        if training:
            optimizer.zero_grad()  # 清空上一步的梯度
            loss.backward()        # 反向传播计算梯度
            optimizer.step()       # 按梯度更新参数

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)  # 取得分最高的类别作为预测
        total += targets.size(0)
        if cuda:
            correct += predicted.eq(targets.data).cpu().sum().item()
        else:
            correct += predicted.eq(targets.data).sum().item()
    acc = 100 * correct / total    # 准确率（百分比）
    avg_loss = total_loss / total  # 平均到每个样本的损失
    return acc, avg_loss


def get_predictions(model, dataloader, cuda, get_probs=False):
    """在整个 DataLoader 上收集预测结果：get_probs=True 时返回 softmax 类别概率矩阵，否则返回预测类别编号数组。"""
    preds = []
    model.eval()  # 评估模式，不做参数更新
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if cuda: inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets.long())
        outputs = model(inputs)
        if get_probs:
            probs = torch.nn.functional.softmax(outputs, dim=1)  # 将得分转为各类别概率
            if cuda: probs = probs.data.cpu().numpy()
            else: probs = probs.data.numpy()
            preds.append(probs)
        else:
            _, predicted = torch.max(outputs.data, 1)  # 取得分最高的类别编号
            if cuda: predicted = predicted.cpu()
            preds += list(predicted.numpy().ravel())
    if get_probs:
        return np.vstack(preds)  # 各批次的概率矩阵堆叠为 (N, 类别数)
    else:
        return np.array(preds)