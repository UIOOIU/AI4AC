#!/usr/bin/python
# -*- coding: UTF-8 -*-
# create date: 2024/3/01
# __author__: 'Alex Lu'
"""ViT1D：面向一维光谱数据的 Vision Transformer 实现。

模块内包含 Patch 切分与线性嵌入（PatchEmbed）、多头自注意力（Attention）、
LayerScale、随机深度（DropPath）、前馈网络（MLPBlock）、Transformer 编码器块、
分类 token、可学习位置嵌入，以及最终的一维 ViT 分类模型（ViT1D）。
"""
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from functools import partial
import torch
from torch import nn, einsum
from torch.nn import functional
from typing import Callable, Optional, Tuple, Union


class PatchEmbed(nn.Module):
    """
    1D Spectral data to Patch Embedding
    """

    def __init__(self, spectra_size: int, patch_size: int, embed_dim: int, in_channels: int = 1,
                 norm_layer: Callable[..., nn.Module] = None, dynamic_pad: bool = False):
        """初始化Patch嵌入：计算patch数量，并构建“切分patch + 线性映射”的嵌入层与可选归一化层。"""
        super(PatchEmbed, self).__init__()
        if dynamic_pad:
            # 动态填充模式：patch数向上取整，长度不足的部分前向时在光谱末尾补0
            import math
            self.num_patches = math.ceil(spectra_size / patch_size)
        else:
            # 非填充模式：要求光谱长度能被patch长度整除，否则无法均匀切分
            assert spectra_size % patch_size == 0, \
                "spectra_size {spectra_size} should be divisible by patch_size {patch_size}."
            self.num_patches = spectra_size // patch_size
        self.spectra_size = spectra_size
        self.patch_size = patch_size
        self.dynamic_pad = dynamic_pad

        # [B, c, np] -> [B, num_patches, patch_size*c]
        self.patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            nn.Linear(patch_size*in_channels, embed_dim),
        )
        # Alex: patch_embedding也可以使用Conv1d来实现，因为原模型使用conv2d实现
        # self.patch_embedding = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

        # patch嵌入后的可选LayerNorm；不传norm_layer时用Identity（不做任何处理）
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """前向：校验光谱长度 -> （动态填充时）末尾补0 -> 切patch并线性映射 -> 归一化。"""
        b, c, l = x.shape
        # 1D data
        assert l == self.spectra_size, \
            f"Input spectral_size {l} doesn't match model spectral_size {self.spectra_size}."
        if self.dynamic_pad:
            # 计算需要补多少个0，使总长度恰好是patch_size的整数倍
            pad_l = (self.patch_size - self.spectra_size % self.patch_size) % self.patch_size
            # pad_l = self.num_patches * self.patch_size - self.spectra_size
            x = functional.pad(x, (0, pad_l))  # 只在序列末尾（右侧）补0

        x = self.patch_embedding(x)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """多头自注意力模块（Multi-Head Self-Attention）。"""

    def __init__(self, dim: int, num_heads: int = 8,
                 qkv_bias: bool = False, qk_norm: bool = False, attn_drop: float = 0., proj_drop: float = 0.,
                 norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
                 ) -> None:
        """初始化QKV投影、可选的Q/K归一化、注意力dropout和输出投影。"""
        super().__init__()
        # 总维度必须能被头数整除，才能均匀拆成多个注意力头
        assert dim % num_heads == 0, 'dim should be divisible by num_heads.'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 每个注意力头分到的维度
        self.scale = self.head_dim ** -0.5  # 缩放因子1/sqrt(head_dim)，防止点积过大把softmax推向饱和

        # 用一个线性层一次性算出Q、K、V三份（输出维度为3*dim）
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 可选对Q、K分别做LayerNorm（较新ViT变体的稳定训练技巧）
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)

        # 多头结果拼接后的输出投影W_o，外加一层dropout
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """前向：QKV投影 -> 拆多头 -> QK^T缩放打分(+mask) -> softmax -> 对V加权 -> 合头并输出投影。"""
        # b n h d <=> batch (num_tokens+num_patches) num_heads head_dim,  num_heads * head_dim = channel = embed_dim
        b, n, hd = x.shape

        # 一次线性变换得到Q/K/V，再沿最后一维等分成三份
        qkv = self.qkv(x).chunk(3, dim=-1)
        # 把最后一维拆成(头数, 每头维度)并将头维度提前：b n (h d) -> b h n d
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        q, k = self.q_norm(q), self.k_norm(k)

        # attn = q @ k.transpose(-2, -1) * self.scale
        # 各头独立计算两两token之间的注意力打分：Q与K做点积后乘缩放因子
        attn = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = attn + mask if mask is not None else attn  # 叠加mask（如padding mask / 因果mask）
        attn = attn.softmax(dim=-1)  # 沿被关注维度softmax，把打分归一化为和为1的权重
        attn = self.attn_drop(attn)

        # attn = attn @ v
        # attn = attn.transpose(1, 2).reshape(b, n, hd)
        # 用注意力权重对V加权求和，再把多个头拼回一个维度：b h n d -> b n (h d)
        attn = einsum('b h i j, b h j d -> b h i d', attn, v)
        attn = rearrange(attn, 'b h n d -> b n (h d)')

        x = self.proj(attn)  # 拼接后的多头结果再过一次线性投影
        return x


class LayerScale(nn.Module):
    """层级缩放（Layer Scale）：用可学习的gamma向量逐通道缩放子层输出，帮助深层ViT稳定训练。"""

    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False) -> None:
        """创建长度为dim的可学习缩放向量gamma，初始为很小的init_values。"""
        super(LayerScale, self).__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """逐通道乘以gamma；inplace=True时直接在原张量上修改以节省显存。"""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    # 推理阶段，或丢弃概率为0时，原样返回
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob  # 每条残差路径被保留的概率
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # 按样本生成0/1随机掩码（同一batch内不同样本各自决定整条路径是否保留）
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        # 对保留下来的路径除以保留概率做放大，保证训练/推理时输出的期望值一致
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        """保存路径丢弃概率与是否按保留概率缩放的开关，真正的随机逻辑在drop_path函数中。"""
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """训练时按样本随机丢弃整条残差路径（随机深度），推理时原样返回。"""
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        """打印模块时额外展示丢弃概率，方便查看网络配置。"""
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class MLPBlock(nn.Module):
    """ MLP Block used in Transformer Encoder Block
    """

    def __init__(
            self, in_features: int, hidden_features: int = None,
            out_features: int = None, act_layer: Callable[..., nn.Module] = nn.GELU,
            bias: Union[bool, Tuple] = True, dropout: Union[float, Tuple] = 0.
    ):
        """构建Transformer中的前馈网络（FFN）：两层线性变换，中间夹GELU激活与dropout。"""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # bias/dropout既可以传单个值（两层共用），也可以传分别配置两层的元组
        biases = (bias, bias) if not isinstance(bias, tuple) else bias
        dropouts = (dropout, dropout) if not isinstance(dropout, tuple) else dropout

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=biases[0]),  # 先升维到隐藏层维度
            act_layer(),
            nn.Dropout(dropouts[0]) if dropouts[0] > 0.0 else nn.Identity(),
            nn.Linear(hidden_features, out_features, bias=biases[1]),  # 再降维回输出维度
            nn.Dropout(dropouts[1]) if dropouts[1] > 0.0 else nn.Identity(),
        )

    def forward(self, x):
        """直接顺序通过两层全连接网络。"""
        x = self.net(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """Transformer编码器块：Pre-LN结构，包含“多头注意力残差子层”和“MLP残差子层”两部分。"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path_ratio: float = 0.,
            act_layer: Callable[..., nn.Module] = nn.GELU,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            mlp_layer: Callable[..., nn.Module] = MLPBlock,
    ) -> None:
        """组装两个子层各自所需的LayerNorm、注意力/MLP、LayerScale与DropPath。"""
        super().__init__()
        # ---- 第一个子层：LayerNorm -> 多头自注意力 -> LayerScale -> DropPath ----
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        # init_values为空时LayerScale退化为Identity；drop_path率为0时同理
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        # ---- 第二个子层：LayerNorm -> MLP前馈网络 -> LayerScale -> DropPath ----
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),  # FFN隐藏层宽度默认是嵌入维度的4倍
            act_layer=act_layer,
            dropout=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """两个子层都采用残差连接（Pre-LN：先归一化再进子层，输出加回输入）。"""
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))  # 注意力残差子层
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))  # 前馈网络残差子层
        return x


class ClsToken(nn.Module):
    """分类token（CLS）：一个与具体输入无关的可学习向量，拼在每个样本patch序列的最前面。"""

    def __init__(self, embed_dim):
        """用均值0、标准差0.02的正态分布初始化一个(1,1,embed_dim)的分类token参数。"""
        super(ClsToken, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim).normal_(std=0.02))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        # Alex: 上述实现初始结果不同，但随着训练会最终相同   [-0.04, 0.04]  [-0.06, 0.06]

    def forward(self, x):
        """把同一个分类token复制batch份，拼接到patch序列的最前面（后续取它的输出做分类）。"""
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # 沿batch维复制b份
        x = torch.cat((cls_tokens, x), dim=1)  # 在token维度(序列维)上拼接
        return x


class PosEmbed(nn.Module):
    """可学习的位置嵌入：长度为“patch数+特殊token数”，与token序列逐元素相加以注入位置信息。"""

    def __init__(self, num_patches, embed_dim, num_tokens):
        """初始化位置嵌入参数，序列长度要容纳全部patch以及前面拼接的cls等特殊token。"""
        super(PosEmbed, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + num_tokens, embed_dim).normal_(std=0.02))

    def forward(self, x):
        """位置嵌入与token表示逐元素相加（依赖广播覆盖整个batch）。"""
        x = x + self.pos_embed
        return x


class ViT1D(nn.Module):
    def __init__(self, spectra_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 drop_ratio=0.1, attn_drop_ratio=0.1, drop_path_ratio=0.1,
                 norm_layer: Callable[..., nn.Module] = None,
                 act_layer: Callable[..., nn.Module] = None):
        """ 1D ViT
        Args:
            spectra_size (int): spectra data size
            patch_size (int, tuple): patch size
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int | list): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): active layer
        """
        super(ViT1D, self).__init__()
        self.num_classes = num_classes
        # assert embed_dim == patch_size * 1, "embed_dim = patch_size * in_channels"
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = PatchEmbed(spectra_size=spectra_size, patch_size=patch_size,
                                      embed_dim=embed_dim, dynamic_pad=True)
        # Alex: spectra_size -> spectra_size + (self.patch_size - self.spectra_size % self.patch_size) % self.patch_size
        # Alex: spectra_size -> self.patch_embed.num_patches * self.patch_size

        self.num_patches = self.patch_embed.num_patches
        # token嵌入流水线：Patch嵌入 -> 拼接分类token -> 加位置嵌入 -> dropout
        self.embedding = nn.Sequential(
            self.patch_embed,
            ClsToken(embed_dim=embed_dim),
            PosEmbed(num_patches=self.num_patches, embed_dim=embed_dim, num_tokens=self.num_tokens),
            nn.Dropout(drop_ratio) if drop_ratio > 0.0 else nn.Identity()
        )

        # 随机深度丢弃率从0线性增长到drop_path_ratio：越靠后的块丢弃概率越大
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule

        # Alex@20240401 num_heads --> try to test multiple num_heads
        # 兼容num_heads传int/tuple/list三种形式：层数多于头数配置时，用最后一个头数补齐
        if isinstance(num_heads, int):
            num_heads = [num_heads]
        if isinstance(num_heads, tuple):
            num_heads = list(num_heads)
        if isinstance(num_heads, list):
            if len(num_heads) < depth:
                n_h = num_heads[-1]
                l_h = len(num_heads)
                for i in range(l_h, depth):
                    num_heads.append(n_h)

        # self.transformer_blocks = nn.Sequential(*[
        #     TransformerEncoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                             proj_drop=drop_ratio, attn_drop=attn_drop_ratio,
        #                             drop_path_ratio=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
        #     for i in range(depth)
        # ])
        # 顺序堆叠depth个Transformer编码器块，每块可用不同的头数和递增的drop_path率
        self.transformer_blocks = nn.Sequential(*[
            TransformerEncoderBlock(dim=embed_dim, num_heads=num_heads[i], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                    proj_drop=drop_ratio, attn_drop=attn_drop_ratio,
                                    drop_path_ratio=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        # Alex@20240401 num_heads --> try to test multiple num_heads

        # 所有编码器块之后的最终LayerNorm
        self.norm = norm_layer(self.num_features)

        # Classifier head(s)
        # 分类头：把cls token的特征线性映射到类别数；num_classes<=0时退化为Identity（只做特征提取）
        self.mlp_head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """提特征：token嵌入 -> 全部Transformer编码器块 -> 最终归一化，取cls token对应的特征向量。"""
        # [b, 1, l] -> [b, num_token+num_patches, embed_dim]
        x = self.embedding(x)
        x = self.transformer_blocks(x)
        x = self.norm(x)  # [b, l, c]
        # x[:, 0] -> cls_token
        return x[:, 0]  # 分类只使用序列第一个位置（cls token）的输出

    def forward(self, x):
        """完整前向：可选把2D图像展平成一维 -> 提特征 -> 分类头输出各类别的logits。"""
        if len(x.shape) == 4:
            # 兼容2D图像输入[b,c,h,w]：把高宽拉平成一条一维序列再当作光谱处理
            x = rearrange(x, 'b c h w -> b c (h w)', )
        assert len(x.shape) == 3 and x.shape[1] == 1, "x.shape {x.shape} should be [batch_size, 1, spectra_size]"
        x = self.forward_features(x)
        x = self.mlp_head(x)
        return x
