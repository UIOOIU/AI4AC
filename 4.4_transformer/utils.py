"""细菌光谱分类实验的可视化工具集。

包含：plot_spectra（绘制一条或多条带纵向偏移的光谱曲线）与
plot_cm（把混淆矩阵绘制成带百分比标注的热力图）。
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
plt.rcParams['font.sans-serif'] = ['SimSun'] # 用来正常显示中文标签


def plot_spectra(x, ys, title, labels=None, dy=0.1):
    """
    x: x轴坐标
    ys: y轴坐标，支持一条光谱数据(1D数组)与多条数据(2D数组)
    title: 图的标题
    labels：每条数据的标签
    dy: 多条数光谱数据偏移高度，主要目的是不使数据的谱线重合
    """
    if isinstance(ys, np.ndarray) and len(ys.shape) == 1:
        ys = np.array([ys])  # 使ys为2D数组

    plt.figure(figsize=(10, 6))
    if x is None:  # 如果没传x轴坐标，则使用最后一个维度上的数组序号作为x轴坐标
        x = np.array(list(range(ys.shape[-1])))

    for i, y in enumerate(ys):
        if labels is not None and len(labels) > 0:  # 如果有传labels则作图时使用label
            plt.plot(x, y + dy * i, label=labels[i])
        else:
            plt.plot(x, y + dy * i)
    plt.xlabel('位移(cm$^{-1}$)')
    plt.ylabel('强度(a.u.)')
    plt.title(title)
    if labels is not None and len(labels) > 0:  # 如果有传labels则作图时显示出label
        plt.legend()
    plt.show()


def plot_cm(cm, label):
    """
    cm: 混淆矩阵
    label: x与y轴标签
    """
    sb.set_context("talk", rc={"font":"Helvetica", "font.size":12})  # 设置绘图的上下文风格（字体、字号等）
    plt.figure(figsize=(15, 12))
    cm = 100 * cm / cm.sum(axis=1)[:,np.newaxis]  # 按行归一化为百分比：每格除以该真实类别样本总数
    ax = sb.heatmap(cm, annot=True, cmap='YlGnBu', fmt='0.0f',
                    xticklabels=label, yticklabels=label)  # cm方阵(所有分类标签)热力图
    ax.xaxis.tick_top()  # 把x轴刻度标签移到顶部
    plt.xticks(rotation=90)  # x轴标签旋转90度，避免标签文字互相重叠
    plt.show()
