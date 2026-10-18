# AI4AC —— AI for Applied Chemistry 教学代码库

人工智能在化学（光谱分析）中的应用：从 Python 基础、机器学习入门，到深度学习（ANN / CNN / RNN / Transformer）与化学信息学工具的渐进式教学项目。全部内容以 Jupyter Notebook 组织，配有中文注释与导读（第 1~3 章面向零基础读者逐行注释，第 4~5 章为主要注释）。

## 环境准备

1. 安装 Miniconda（步骤详见 [readme.txt](readme.txt) 与 [1_python环境安装.pptx](1_python环境安装.pptx)）。
2. 安装基础依赖：

   ```bat
   conda install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

3. 深度学习章节（4.x）需要 PyTorch，工具章节（5_tools）按笔记本内首格的安装说明安装 RDKit、DeepChem、ASE、DScribe、PyCaret 等。
4. 在项目根目录用以下脚本启动（会先结束已有的 Jupyter 进程再启动）：

   | 脚本 | 作用 |
   |---|---|
   | [startup.bat](startup.bat) | conda 环境 + Jupyter Notebook |
   | [startup-lab.bat](startup-lab.bat) | conda 环境 + Jupyter Lab |
   | [startup-none.bat](startup-none.bat) | 使用本机指定虚拟环境 + Jupyter Notebook（硬编码了作者机器的 venv 路径，他人使用需自行修改） |

## 目录结构

| 目录 | 内容 |
|---|---|
| [1_basic](1_basic/) | 四本入门笔记本：Python 编程基础、NumPy、pandas、Matplotlib（含官方文档中文讲解） |
| [2.1_regression](2.1_regression/) | 线性回归入门：简单/多元/多项式回归，以及 NumPy 手写梯度下降实现 |
| [2.2_classification](2.2_classification/) | 鸢尾花分类：逻辑回归、SVM、决策树、随机森林、KNN，含混淆矩阵与评估指标 |
| [2.3_clustering](2.3_clustering/) | 鸢尾花聚类（无监督）：KMeans、层次聚类与树状图 |
| [2.4_redunction](2.4_redunction/) | PCA 降维：iris-pca（入门）；other-pca（进阶：人脸特征脸、去噪、高维可视化、t-SNE） |
| [3_data_process](3_data_process/) | 数据处理实战：`iris/DS&ML.ipynb`（数据清洗 + 完整 ML 流程）；`raman/`（拉曼光谱预处理：平滑去噪、基线校正、峰检测、批量处理） |
| [4.1_ann](4.1_ann/) | MNIST 手写数字全连接神经网络（ANN），含训练/验证/过拟合讨论 |
| [4.2_cnn](4.2_cnn/bacteria-1D/) | 一维 CNN 细菌拉曼光谱识别与抗生素分组：传统机器学习对比（ML.ipynb）、CNN 训练与微调笔记本，及配套训练代码（config / datasets / resnet / training / utils.py） |
| [4.3_rnn](4.3_rnn/) | LSTM 原理与手写实现（从 LSTMCell 到序列建模） |
| [4.4_transformer](4.4_transformer/) | Transformer：`nn.Transformer.ipynb`（复制任务）、`ViT1D.py`（一维 Vision Transformer）及其在 MNIST 与细菌光谱上的应用 |
| [5_tools](5_tools/) | 化学/科学计算工具：RDKit（分子结构、描述符、分子指纹与相似度）、SMILES→logP（PyCaret 建模）、DeepChem（分子数据集与建模）、ASE（原子模拟：表面吸附、结构弛豫、分子动力学）、DScribe（Coulomb Matrix 等结构描述符 + 聚类可视化） |
| [data](data/) 及各章 `data/` | 数据集（MNIST、拉曼光谱、分子结构文件等） |
| [tmp](tmp/) | 个人实验草稿区，不属于教学主线 |
| `readme.txt` / `1_python环境安装.pptx` | Windows 环境安装教程 |

## 建议学习路径

```
1_basic（Python/NumPy/pandas/Matplotlib）
   → 2.1 回归 → 2.2 分类 → 2.3 聚类 → 2.4 降维
   → 3 数据处理实战（鸢尾花 / 拉曼光谱）
   → 4.1 ANN → 4.2 CNN → 4.3 RNN → 4.4 Transformer
   → 5_tools（化学信息学与科学计算工具箱）
```

## 使用说明

- 笔记本按单元格顺序自上而下运行（Shift+Enter）；数据集路径均相对项目根目录或所在章节目录。
- `.ipynb_checkpoints/` 是 Jupyter 自动备份目录，可忽略；`__pycache__/` 为 Python 编译缓存。
