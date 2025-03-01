import torch.nn as nn
from networks.utils.utils import DropPath, Mlp
from networks.utils.Attention import SD_attn


class Windowattn_block(nn.Module):
    def __init__(self, dim, window_size, num_heads=1, mlp_ratio=4., 
                qkv_bias=True, drop=0., attn_drop=0., drop_path=0., 
                act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                attn_type="windowattn", pre_norm=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.pre_norm = pre_norm
        self.attn_type = attn_type
        if "save_attn" in kwargs:
            self.save_attn = kwargs['save_attn']
        else:
            self.save_attn = False

        self.norm = norm_layer(dim)
        # self.GAU1 = Flash_attn(dim, window_size=self.window_size, uv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, expansion_factor=2, attn_type='lin')
        if attn_type == "windowattn":
            if "shift_size" not in kwargs:
                shift_size = [0, 0, 0]
            else:
                shift_size = kwargs["shift_size"]
            if "dilated_size" in kwargs:
                dilated_size = kwargs["dilated_size"]
            else:
                dilated_size = [1, 1, 1]
            self.attn = SD_attn(
                dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=drop, shift_size=shift_size, dilated_size=dilated_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    

    

    def forward(self, x):
        shortcut = x
        # partition windows

        if self.pre_norm:
            if self.attn_type == "windowattn":
                x, save_attn = self.attn(self.norm(x))
            else:
                x = self.attn(self.norm(x))
            
            x = shortcut + self.drop_path(x)

        else:
            if self.attn_type == "windowattn":
                x, save_attn = self.attn(x)
            else:
                x = self.attn(x)
            
            x = self.norm(shortcut + self.drop_path(x))


        if self.pre_norm:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        
        if self.attn_type == "windowattn" and self.save_attn:
            return x, save_attn
        else:
            return x

