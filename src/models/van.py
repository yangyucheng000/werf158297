#import torch
import mindspore
#import torch.nn as nn
import mindspore.nn as nn
#import torch.nn.functional as F
import mindspore.ops as F
from functools import partial
import mindspore.numpy as np
#import numpy as np
#from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#from timm.models.registry import register_model
#from timm.models.vision_transformer import _cfg
import math
from mindspore import Tensor, context
from mindspore.common.initializer import Normal,TruncatedNormal
from mindspore.common import initializer as weight_init
from src.models.layers.drop_path import DropPath2D as DropPath
from src.models.layers.identity import Identity
import os
if os.getenv("device_target") == "Ascend" and int(os.getenv("RANK_SIZE")) > 1:
    BatchNorm2d = nn.SyncBatchNorm
else:
    BatchNorm2d = nn.BatchNorm2d

class DWConv(nn.Cell):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3,has_bias=True,group=dim)

    def construct(self, x):
        x = self.dwconv(x)
        return x
        
class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1,has_bias=True)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1,has_bias=True)
        self.drop = nn.Dropout(1-drop)
        #self.apply(self._init_weights)

    def construct(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class LKA(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, group=dim,has_bias=True)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, group=dim, dilation=3,has_bias=True)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def construct(self, x):
        u = x     
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Cell):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def construct(self, x):
        shorcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Cell):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = nn.Dropout(1-drop_path) if drop_path > 0. else Identity()

        self.norm2 = BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2    
        #self.weight = Parameter(Tensor(np.ones((1, 2)), mindspore.float32), name="w", requires_grad=True)        
        #self.layer_scale_1 = nn.Parameter(
            #layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_1 = mindspore.Parameter(
            layer_scale_init_value * Tensor(np.ones((dim)), mindspore.float32), requires_grad=True)
        #self.layer_scale_2 = nn.Parameter(
            #layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = mindspore.Parameter(
            layer_scale_init_value * Tensor(np.ones((dim)), mindspore.float32), requires_grad=True)

        #self.apply(self._init_weights)

    def construct(self, x):
		
		
        #x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        expand_dims = F.ExpandDims()
        layer_scale_1 = expand_dims(expand_dims(self.layer_scale_1, -1),-1)
        x = x + self.drop_path(layer_scale_1 * self.attn(self.norm1(x)))
        #expand_dims replace unsqueeze
        layer_scale_2 = expand_dims(expand_dims(self.layer_scale_2, -1),-1)
        x = x + self.drop_path(layer_scale_2  * self.mlp(self.norm2(x)))
        return x

class OverlapPatchEmbed(nn.Cell):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        #patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,has_bias=True)
        self.norm = BatchNorm2d(embed_dim)

        #self.apply(self._init_weights)

    def construct(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W


class VAN_c(nn.Cell):
    def __init__(self, img_size=224, in_chans=3,patch_size=7, stride=4,embed_dims=64,mlp_ratios=4, drop_rate=0., drop_path_rate=0.,drop_path_i=0., norm_layer=nn.LayerNorm,depth=3, flag=True):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(img_size=img_size,patch_size=patch_size,stride=stride,in_chans=in_chans,embed_dim=embed_dims)
        self.block= nn.CellList([Block(dim=embed_dims, mlp_ratio=mlp_ratios, drop=drop_rate, drop_path=drop_path_rate*j+drop_path_i) for j in range(depth)])
        self.norm= norm_layer([embed_dims])
        self.embed_dims=embed_dims
        self.flag=flag
    
    def construct(self, x):
        B = x.shape[0]
        x, H, W = self.patch_embed(x)
        for blk in self.block:
            x = blk(x)
        input_perm = (0, 2, 1)
        transpose = F.Transpose()
        x = x.reshape(B,self.embed_dims, -1)
        x = transpose(x, input_perm)
        x = self.norm(x)
        if self.flag==True:
            input_perm1 = (0,3,1,2)
            x = x.reshape(B, H, W, -1)
            x = transpose(x, input_perm1)
        return x

class VAN(nn.Cell):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,depths=[3, 4, 6, 3],num_stages=4):
        super().__init__()
        dpr=drop_path_rate/(sum(depths)-1)
        self.layer1=VAN_c(img_size=img_size, in_chans=in_chans,patch_size=7, stride=4,embed_dims=embed_dims[0],mlp_ratios=mlp_ratios[0], drop_rate=0., drop_path_rate=dpr,drop_path_i=0., norm_layer=norm_layer,depth=depths[0], flag=True)
        self.layer2=VAN_c(img_size=img_size//4, in_chans=embed_dims[0],patch_size=3, stride=2,embed_dims=embed_dims[1],mlp_ratios=mlp_ratios[1], drop_rate=0., drop_path_rate=dpr,drop_path_i=dpr*depths[0], norm_layer=norm_layer,depth=depths[1], flag=True)
        self.layer3=VAN_c(img_size=img_size//8, in_chans=embed_dims[1],patch_size=3, stride=2,embed_dims=embed_dims[2],mlp_ratios=mlp_ratios[2], drop_rate=0., drop_path_rate=dpr,drop_path_i=dpr*(depths[0]+depths[1]), norm_layer=norm_layer,depth=depths[2], flag=True)
        self.layer4=VAN_c(img_size=img_size//16, in_chans=embed_dims[2],patch_size=3, stride=2,embed_dims=embed_dims[3],mlp_ratios=mlp_ratios[3], drop_rate=0., drop_path_rate=dpr,drop_path_i=dpr*(depths[0]+depths[1]+depths[2]), norm_layer=norm_layer,depth=depths[3], flag=False)    
        self.head = nn.Dense(embed_dims[3], num_classes) 
        self.init_weights()
    def init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.HeNormal(mode='fan_out', nonlinearity='relu'),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))

    def construct(self, x):
        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x=x.mean(axis=1)
        x = self.head(x)
        return x

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict



def _cfg(url='', **kwargs):
    return {
        #'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }
    

def van_tiny(pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 3, 5, 2],
        **kwargs)
    model.default_cfg = _cfg()
    '''if pretrained:
        model = load_model_weights(model, "van_tiny", kwargs)'''
    return model



def van_small(pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[2, 2, 4, 2],
        **kwargs)
    model.default_cfg = _cfg()
    '''if pretrained:
        model = load_model_weights(model, "van_small", kwargs)'''
    return model


def van_base(pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=nn.LayerNorm, depths=[3, 3, 12, 3],
        **kwargs)
    model.default_cfg = _cfg()
    '''if pretrained:
        model = load_model_weights(model, "van_base", kwargs)'''
    return model


def van_large(pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 5, 27, 3],
        **kwargs)
    model.default_cfg = _cfg()
   
    return model
