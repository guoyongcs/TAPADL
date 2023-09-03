

import torch
from torch import nn
import math

from functools import partial
from timm.models.layers import DropPath, drop_path
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.vision_transformer import _cfg
from einops import rearrange


from timm.models.registry import register_model
from torch.nn import functional as F
from torch.nn import Parameter
from cnn_backbone import _create_hybrid_backbone, HybridEmbed


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)



class MatchedDropout(nn.Module):
    """ definition of mode
        0: no mask, i.e., no dropout
        1: dropout
        2: use previous mask
    """
    def __init__(self, drop_p, inplace=False):
        super().__init__()
        self.masker = nn.Dropout(p=drop_p, inplace=inplace)
        self.mode = 0
        self.pre_mask = None

    def forward(self, input):
        self.masker.training = True
        if self.mode == 0:
            output = input
        elif self.mode == 1:
            mask = self.masker(torch.ones_like(input))
            self.pre_mask = mask.clone()
            output = input * mask
        elif self.mode == 2:
            assert self.pre_mask is not None
            mask = self.pre_mask
            if mask.size(0) != input.size(0):
                new_shape = (2,) + (-1,) * input.ndim
                mask = mask.expand(new_shape)
                mask = mask.reshape(input.shape)
            output = input * mask
        return output


class MatchedDropPath(nn.Module):
    """ definition of mode
        0: no mask, i.e., no droppath
        1: droppath
        2: use previous mask
    """

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        self.mode = 0
        self.pre_mask = None

    def forward(self, input):
        if self.mode == 0:
            output = input
        elif self.mode == 1:
            mask = drop_path(torch.ones_like(input), self.drop_prob, True, self.scale_by_keep)
            self.pre_mask = mask.clone()
            output = input * mask
        elif self.mode == 2:
            assert self.pre_mask is not None
            mask = self.pre_mask
            if mask.size(0) != input.size(0):
                new_shape = (2,) + (-1,) * input.ndim
                mask = mask.expand(new_shape)
                mask = mask.reshape(input.shape)
            output = input * mask
        return output


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        if in_features == 768:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)
        else:
            self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
            self.bn1 = nn.BatchNorm2d(hidden_features)
            self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
            self.bn2 = nn.BatchNorm2d(hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
            self.bn3 = nn.BatchNorm2d(out_features)
            self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.in_features == 768:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
        else:
            B,N,C = x.shape
            x = x.reshape(B, int(N**0.5), int(N**0.5), C).permute(0,3,1,2)
            x = self.bn1(self.fc1(x))
            x = self.act(x)
            x = self.drop(x)
            x = self.act(self.bn2(self.dwconv(x)))
            x = self.bn3(self.fc2(x))
            x = self.drop(x)
            x = x.permute(0,2,3,1).reshape(B, -1, C)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., use_mask=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_mask = use_mask

        if use_mask:
            self.att_mask = nn.Parameter(torch.Tensor(self.num_heads, 196, 196))
            trunc_normal_(self.att_mask, std=.02)

        self.vis_attn = None
        self.distraction_loss = None
        self.negattn_loss = None
        self.distraction_loss_type = 'None'
        self.clean_attention_noisy_feature = False
        self.noisy_attention_clean_feature = False

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.use_mask:
            attn = attn * torch.sigmoid(self.att_mask[:,:N,:N]).expand(B, -1, -1, -1)

        attn = attn.softmax(dim=-1)
        # compute distraction loss
        if self.distraction_loss_type == 'l2' and B > 1:
            # self.distraction_loss = -(torch.log(attn[B//2:]) * attn[B//2:]).sum(dim=-1)
            # self.distraction_loss = self.distraction_loss.mean()
            self.distraction_loss = F.mse_loss(attn[0:B//2], attn[B//2:], reduction='sum') / (B//2)
            self.negattn_loss = F.cosine_similarity(attn[0:B//2], attn[B//2:], -1, 1e-6).sum() / (self.num_heads*N*B//2)
        elif self.distraction_loss_type == 'kl' and B > 1:
            layer_attn_0 = attn[0:B//2]
            layer_attn_1 = attn[B//2:]
            self.distraction_loss = F.kl_div(F.log_softmax(layer_attn_1, dim=-1), F.softmax(layer_attn_0, dim=-1), size_average=False) / (B//2)
            self.negattn_loss = F.cosine_similarity(attn[0:B//2], attn[B//2:], -1, 1e-6).sum() / (self.num_heads*N*B//2)
            # self.negattn_loss = F.kl_div(F.log_softmax(layer_attn_1, dim=-1), F.softmax((1-layer_attn_0), dim=-1), size_average=False) / (B//2)
        elif self.distraction_loss_type == 'cosine' and B > 1:
            layer_attn_0 = attn[0:B//2]
            layer_attn_1 = attn[B//2:]
            self.distraction_loss = F.cosine_similarity(layer_attn_0, layer_attn_1, -1, 1e-6).sum()
            self.distraction_loss = self.distraction_loss / (self.num_heads*N*B//2)
            self.negattn_loss = self.distraction_loss

        if self.clean_attention_noisy_feature:
            attn[B//2:] = attn[0:B//2]
        if self.noisy_attention_clean_feature:
            v[B//2:] = v[0:B//2]

        self.vis_attn = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_mask=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, use_mask=use_mask)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, base_dim, depth, heads, mlp_ratio,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None, use_mask=False, masked_block=None):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        if use_mask==True:
            assert masked_block is not None
            self.blocks = nn.ModuleList()
            for i in range(depth):
                if i < masked_block:
                    self.blocks.append(Block(
                        dim=embed_dim,
                        num_heads=heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path_prob[i],
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        use_mask=use_mask
                    ))
                else:
                    self.blocks.append(Block(
                        dim=embed_dim,
                        num_heads=heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path_prob[i],
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        use_mask=False
                    ))
        else:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_prob[i],
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    use_mask=use_mask
                )
                for i in range(depth)])


    def forward(self, x, mask_matrix=None, mask_layer_index=None):
        B,C,H,W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        # x = x.permute(0,2,3,1).reshape(B, H * W, C)
        for i in range(self.depth):
            if mask_layer_index is not None:
                if (i+1) == mask_layer_index:
                    mask_matrix = rearrange(mask_matrix, 'b c h w -> b (h w) c')
                    x = x * mask_matrix
            x = self.blocks[i](x)
        # x = x.reshape(B, H, W, C).permute(0,3,1,2)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class conv_head_pooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride,
                 padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()

        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride + 1,
                              padding=stride // 2, stride=stride,
                              padding_mode=padding_mode, groups=in_feature)

    def forward(self, x):

        x = self.conv(x)

        return x


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size,
                 stride, padding):
        super(conv_embedding, self).__init__()

        self.out_channels = out_channels
        self.patch_size = patch_size

        if patch_size==4:
            final_ks = 1
        else:
            final_ks = 4
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(7, 7), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(32, out_channels, kernel_size=(final_ks, final_ks), stride=(final_ks, final_ks))
        )
        # self.upsampler = nn.Upsample((224,224), mode='bicubic', align_corners=False)

    def forward(self, x):
        _, _, H, W = x.shape
        # if self.patch_size==16 and H<224:
        #     x = self.upsampler(x)
        x = self.proj(x)
        return x


class PoolingTransformer(nn.Module):
    def __init__(self, image_size, patch_size, stride, base_dims, depth, heads,
                 mlp_ratio, num_classes=1000, in_chans=3,
                 attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0, use_mask=False, masked_block=None, mask_batch_size=512, mask_num_patches=196, mask_layer_list=[], rescale_pow=0, combine_feature=False, mask_mode='split', num_splits=1, inf_split=False, fuse_conv=False, fuse_ksize=3, controller_hid=100, force_uniform=False, residual=False, att_mode='split', inside_residual=False, nhead_ratio=1, backbone=None):
        super(PoolingTransformer, self).__init__()

        total_block = sum(depth)
        padding = 0
        block_idx = 0

        width = math.floor(
            (image_size / stride))

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes

        self.patch_size = patch_size

        if backbone is not None:
            self.patch_embed = backbone
        else:
            self.patch_embed = conv_embedding(in_chans, base_dims[0] * heads[0],
                                              patch_size, stride, padding)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])

        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]

            if stage == 0:
                self.transformers.append(
                    Transformer(base_dims[stage], depth[stage], heads[stage],
                                mlp_ratio,
                                drop_rate, attn_drop_rate, drop_path_prob, use_mask=use_mask, masked_block=masked_block)
                )
            else:
                self.transformers.append(
                    Transformer(base_dims[stage], depth[stage], heads[stage],
                                mlp_ratio,
                                drop_rate, attn_drop_rate, drop_path_prob)
                )
            if stage < len(heads) - 1:
                self.pools.append(
                    conv_head_pooling(base_dims[stage] * heads[stage],
                                      base_dims[stage + 1] * heads[stage + 1],
                                      stride=2
                                      )
                )

        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        if num_classes > 0:
            self.head = nn.Linear(base_dims[-1] * heads[-1], num_classes)
        else:
            self.head = nn.Identity()


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for stage in range(len(self.pools)):
            x = self.transformers[stage](x)
            x = self.pools[stage](x)
        x = self.transformers[-1](x)
        cls_features = self.norm(torch.flatten(self.gap(x), 1))

        return cls_features

    def forward(self, x, x1=None):
        if x1 is not None:
            x = torch.cat((x, x1), 0)
            cls_features = self.forward_features(x)
            output = self.head(cls_features)
            return output[0:output.size(0)//2], output[output.size(0)//2:]
        else:
            cls_features = self.forward_features(x)
            output = self.head(cls_features)
            return output


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, student_width=1.0, drop_rate=0):
        super(BasicBlock, self).__init__()
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(drop_rate, inplace=False)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, student_width=student_width)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.block_index = 0

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.drop_rate > 0:
            out = self.dropout(out)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0):
        super(Bottleneck, self).__init__()
        self.name = "resnet-bottleneck"
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(drop_rate, inplace=False)
        self.conv1 = conv1x1(inplanes, planes)
        # nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        # nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * 4)
        # nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.block_index = 0

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.drop_rate > 0:
            out = self.dropout(out)

        out += residual
        out = self.relu(out)
        return out


class ResNet_IMAGENET(nn.Module):

    def __init__(self, depth, num_classes=1000, student_width=1.0, drop_rate=0):
        self.inplanes = 64
        super(ResNet_IMAGENET, self).__init__()
        self.num_classes = num_classes
        if depth < 50:
            block = BasicBlock
        else:
            block = Bottleneck

        if depth == 18:
            layers = [2, 2, 2, 2]
        elif depth == 34:
            layers = [3, 4, 6, 3]
        elif depth == 50:
            layers = [3, 4, 6, 3]
        elif depth == 101:
            layers = [3, 4, 23, 3]
        elif depth == 152:
            layers = [3, 8, 36, 3]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], student_width=student_width, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, student_width=student_width,
                                       drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, student_width=student_width,
                                       drop_rate=drop_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, student_width=student_width,
                                       drop_rate=drop_rate)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, student_width=1.0, drop_rate=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate=drop_rate))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, drop_rate=drop_rate))

        return nn.Sequential(*layers)

    def forward(self, x, mask_matrix=None, mask_layer_index=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


@register_model
def resnet_cifar(pretrained, **kwargs):
    model = ResNet_IMAGENET(
        50, num_classes=10
    )
    model.default_cfg = _cfg()
    return model


@register_model
def rvt_tiny(pretrained, **kwargs):
    # _ = kwargs.pop('num_noisy_tokens')
    # _ = kwargs.pop('noise_level')
    # _ = kwargs.pop('temperature')
    # _ = kwargs.pop('tokenaug_type')
    # _ = kwargs.pop('noise_which_layer')
    _ = kwargs.pop('num_scales')
    _ = kwargs.pop('pretrained_cfg')
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=16,
        base_dims=[32, 32],
        depth=[10, 2],
        heads=[6, 12],
        mlp_ratio=4,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def rvt_tiny_plus(pretrained, **kwargs):
    # _ = kwargs.pop('num_noisy_tokens')
    # _ = kwargs.pop('noise_level')
    # _ = kwargs.pop('temperature')
    # _ = kwargs.pop('tokenaug_type')
    # _ = kwargs.pop('noise_which_layer')
    _ = kwargs.pop('num_scales')
    _ = kwargs.pop('pretrained_cfg')
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=16,
        base_dims=[32, 32],
        depth=[10, 2],
        heads=[6, 12],
        mlp_ratio=4,
        use_mask=True,
        masked_block=10,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def rvt_small(pretrained, **kwargs):
    # _ = kwargs.pop('num_noisy_tokens')
    # _ = kwargs.pop('noise_level')
    # _ = kwargs.pop('temperature')
    # _ = kwargs.pop('tokenaug_type')
    # _ = kwargs.pop('noise_which_layer')
    _ = kwargs.pop('num_scales')
    _ = kwargs.pop('pretrained_cfg')
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=16,
        base_dims=[64],
        depth=[12],
        heads=[6],
        mlp_ratio=4,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def rvt_small_plus(pretrained, **kwargs):
    # _ = kwargs.pop('num_noisy_tokens')
    # _ = kwargs.pop('noise_level')
    # _ = kwargs.pop('temperature')
    # _ = kwargs.pop('tokenaug_type')
    # _ = kwargs.pop('noise_which_layer')
    _ = kwargs.pop('num_scales')
    _ = kwargs.pop('pretrained_cfg')
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=16,
        base_dims=[64],
        depth=[12],
        heads=[6],
        mlp_ratio=4,
        use_mask=True,
        masked_block=5,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def rvt_base(pretrained, **kwargs):
    # _ = kwargs.pop('num_noisy_tokens')
    # _ = kwargs.pop('noise_level')
    # _ = kwargs.pop('temperature')
    # _ = kwargs.pop('tokenaug_type')
    # _ = kwargs.pop('noise_which_layer')
    _ = kwargs.pop('num_scales')
    _ = kwargs.pop('pretrained_cfg')
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=16,
        base_dims=[64],
        depth=[12],
        heads=[12],
        mlp_ratio=4,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def rvt_base_plus(pretrained, **kwargs):
    # _ = kwargs.pop('num_noisy_tokens')
    # _ = kwargs.pop('noise_level')
    # _ = kwargs.pop('temperature')
    # _ = kwargs.pop('tokenaug_type')
    # _ = kwargs.pop('noise_which_layer')
    _ = kwargs.pop('num_scales')
    _ = kwargs.pop('pretrained_cfg')
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=16,
        base_dims=[64],
        depth=[12],
        heads=[12],
        mlp_ratio=4,
        use_mask=True,
        masked_block=5,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


class MaskRobustPreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, student_width=1.0, drop_rate=0):
        super(MaskRobustPreActBlock, self).__init__()
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(drop_rate, inplace=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion * planes, stride=stride)
            )

        self.n_paths = 2
        self.n_channels = [planes, 1]
        self.register_buffer("path_beta", torch.ones(self.n_paths, dtype=torch.long, requires_grad=False))
        self.alpha = Parameter(torch.ones(self.n_paths))

    def sample_beta(self, active_paths, active_indexes):
        self.conv2.sample_beta(active_indexes[0])
        self.path_beta.fill_(0)
        self.path_beta.index_fill_(0, torch.tensor(active_paths).type_as(self.path_beta), 1)

    def reset_beta(self):
        self.conv2.reset_beta()
        self.path_beta.fill_(1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        if self.drop_rate > 0:
            out = self.dropout(out)

        if self.path_beta.sum().item() < self.n_paths:
            out = out * self.path_beta[0]
            shortcut = shortcut * self.path_beta[1]

        out += shortcut
        return out


class Robust_PreActResNet_CIFAR(nn.Module):
    def __init__(self, depth, num_classes=10, student_width=1.0, drop_rate=0):
        super(Robust_PreActResNet_CIFAR, self).__init__()
        self.in_planes = 64
        self.student_width = student_width
        self.drop_rate = drop_rate
        self.num_classes = num_classes

        assert depth in [18, 34, 50], 'invalid model depth'
        if depth == 18:
            num_blocks = [2, 2, 2, 2]
            block = MaskRobustPreActBlock
        elif depth == 34:
            num_blocks = [3, 4, 6, 3]
            block = MaskRobustPreActBlock

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, student_width=student_width, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, student_width=student_width, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, student_width=student_width, drop_rate=drop_rate)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, student_width=student_width, drop_rate=drop_rate)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.norm = False

    def _make_layer(self, block, planes, num_blocks, stride, student_width=1.0, drop_rate=0):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, student_width=student_width, drop_rate=drop_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _set_norm(self):
        self.norm = True

    def _normalize(self, x):
        if self.num_classes == 10:
            CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
            CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
        elif self.num_classes == 100:
            CIFAR_MEAN = [0.50705882, 0.48666667, 0.44078431]
            CIFAR_STD = [0.26745098, 0.25568627, 0.27607843]
        mu = torch.tensor(CIFAR_MEAN).view(3, 1, 1).cuda()
        std = torch.tensor(CIFAR_STD).view(3, 1, 1).cuda()
        return (x - mu)/std

    def forward(self, x, mask_matrix=None, mask_layer_index=None):
        if self.norm:
            x = self._normalize(x)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


@register_model
def preresnet18(pretrained, **kwargs):
    model = Robust_PreActResNet_CIFAR(
        depth=18, num_classes=kwargs['num_classes']
    )
    model.default_cfg = _cfg()
    return model


@register_model
def rvt_tiny_plus_hybrid(pretrained, **kwargs):
    model_args = dict(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False)
    backbone = _create_hybrid_backbone(pretrained=False, pretrained_strict=False, **model_args)
    cnn = HybridEmbed(backbone, patch_size=1, embed_dim=192)

    model = PoolingTransformer(
        image_size=224,
        # patch_size=16,
        stride=16,
        base_dims=[32, 32],
        depth=[7, 2],
        heads=[6, 12],
        mlp_ratio=4,
        use_mask=True,
        masked_block=10,
        backbone=cnn,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model


