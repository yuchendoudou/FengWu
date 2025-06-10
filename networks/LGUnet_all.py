import torch.nn as nn
import torch
import copy
from functools import partial
from networks.utils.Blocks import Windowattn_block
from einops import rearrange
from timm.models.layers import trunc_normal_
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from networks.utils.utils import Mlp
from torchvision import utils as vutils



class PatchEmbed(nn.Module):
    def __init__(self, img_size=[32, 64], patch_size=[2,2], stride=[2,2], in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patches_resolution = [(img_size[0] - patch_size[0]) // stride[0] + 1, (img_size[1] - patch_size[1]) // stride[1] + 1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops




class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x).view(B, H//2, W//2, 4*C)
        x = self.reduction(x)

        return x


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        x = self.expand(x)
        B, H, W, C = x.shape

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x= self.norm(x)

        return x

        return x




class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, 
                 use_checkpoint=False, pre_norm=True, attn_type="windowattn"):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Windowattn_block(
                        dim=dim,
                        window_size=window_size,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                        pre_norm=pre_norm,
                        shift_size=[0,0] if i%2==0 else [i//2 for i in window_size],
                        attn_type=attn_type
                    )
            if use_checkpoint:
                block = checkpoint_wrapper(block, offload_to_cpu=True)
            self.blocks.append(block)


        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim//2, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for blk in self.blocks:

            x = blk(x)
        return x


class BasicLayer_up(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, 
                 use_checkpoint=False, pre_norm=True, attn_type="windowattn"):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Windowattn_block(
                    dim=dim,
                    window_size=window_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    pre_norm=pre_norm,
                    shift_size=[0,0] if i%2==0 else [i//2 for i in window_size],
                    attn_type=attn_type
                )
            if use_checkpoint:
                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class Transformer_Encoder(nn.Module):
    def __init__(self, img_size=224, patch_size=2,  stride=[2, 2], in_chans=3, 
                 embed_dim=96, depths=[2, 2, 2], num_heads=[3, 6, 12],
                 window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pre_norm=True, attn_type="windowattn", **kwargs):
        super().__init__()


        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer > 0) else None,
                               use_checkpoint=use_checkpoint,
                               pre_norm=pre_norm,
                               attn_type=attn_type
                               )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)     #B, L, C
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []
        x = x.view(B, self.patches_resolution[0], self.patches_resolution[1], -1)
        for layer in self.layers:
            x = layer(x)
            x_downsample.append(x)

        x = self.norm(x)  # B H, W C
  
        return x, x_downsample
    
class Transformer_Decoder(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pre_norm=True, final_upsample="expand_first", attn_type="windowattn", **kwargs):
        super().__init__()


        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample


        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),
            int(embed_dim*2**(self.num_layers-1-i_layer)))
            layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                            depth=depths[(self.num_layers-1-i_layer)],
                            num_heads=num_heads[(self.num_layers-1-i_layer)],
                            window_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                            norm_layer=norm_layer,
                            upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                            use_checkpoint=use_checkpoint,
                            pre_norm=pre_norm,
                            attn_type=attn_type
                            )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm_up= norm_layer(self.embed_dim)      
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            x = torch.cat([x,x_downsample[len(x_downsample)-1-inx]],-1)
            x = self.concat_back_dim[inx](x)
            x = layer_up(x)

        x = self.norm_up(x)  # B L C
  
        return x





class Layer(nn.Module):
    def __init__(self, dim, depth, window_size, 
                num_heads=1, mlp_ratio=4., qkv_bias=True, 
                drop=0., attn_drop=0., drop_path=0., 
                norm_layer=nn.LayerNorm, layer_type="convnet_block",
                use_checkpoint=False, pre_norm=True, attn_type="windowattn") -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.save_attn = False
        self.layer_type = layer_type


        self.blocks = nn.ModuleList()
        for i in range(depth):
            if layer_type == "window_block":
                block = Windowattn_block(
                        dim=dim,
                        window_size=window_size,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                        pre_norm=pre_norm,
                        save_attn=self.save_attn,
                        attn_type=attn_type
                    )
            elif layer_type == "swin_block":
                block = Windowattn_block(
                    dim=dim,
                    window_size=window_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    pre_norm=pre_norm,
                    shift_size=[0,0] if i%2==0 else [i//2 for i in window_size],
                    attn_type=attn_type
                )
            if use_checkpoint:
                block = checkpoint_wrapper(block, offload_to_cpu=True)
            self.blocks.append(block)



    def forward(self, x):
        for i, blk in enumerate(self.blocks):
            if self.layer_type == "window_block" and self.save_attn:
                x, save_attn = blk(x)

                for j in range(save_attn[0].shape[1]):
                    vutils.save_image(save_attn[0][j].detach().cpu().reshape(90, 180)/save_attn[0][j].detach().cpu().reshape(90, 180).max(), f"./attn_img/attn_{i}_{j}.png")

            else:
                x = blk(x)
    
        return x

class Enc_net(nn.Module):
    def __init__(self, img_size=[32, 64], patch_size=[2,2], stride=[2, 2], inchans_list=[6, 13, 13, 13, 13, 13, 4, 13, 13, 13, 13, 13], embed_dim=96, lgnet_embed=768, depths=[2,2,2],
                 num_heads=[3,6,12], window_size=[6, 12], mlp_ratio=4, qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                 patch_norm=False,use_checkpoint=False, channel_num=37, inp_length=1, use_mlp=False, pre_norm=True, attn_type="windowattn"):
        super().__init__()
        self.inchans_list = inchans_list
        self.inp_length = inp_length
        
        self.enc_list = nn.ModuleList()
        for i in range(len(self.inchans_list)//inp_length):
            if inp_length == 1:
                inchans = self.inchans_list[i]
            elif inp_length == 2:
                inchans = self.inchans_list[i]+self.inchans_list[len(self.inchans_list)//2+i]
            else:
                raise ValueError
            self.enc_list.append(Transformer_Encoder(img_size=img_size, patch_size=patch_size, stride=stride, in_chans=inchans,
                                                     embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size,
                                                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                     drop_path_rate=drop_path_rate, norm_layer=norm_layer, ape=True, patch_norm=patch_norm,
                                                     use_checkpoint=use_checkpoint, pre_norm=pre_norm, attn_type=attn_type))

        self.proj = nn.Linear(embed_dim*len(self.inchans_list)//inp_length*2**(len(depths)-1), lgnet_embed)
    
    def forward(self, x):
        data_split = torch.split(x, self.inchans_list, dim=1)
        last_data_list = []
        outdata_list = []
        for i in range(len(self.inchans_list)//self.inp_length):
            if self.inp_length == 1:
                data = data_split[i]
            elif self.inp_length == 2:
                data = torch.cat((data_split[i], data_split[len(self.inchans_list)//2+i]), dim=1)
            data, data_downsample_list = self.enc_list[i](data)
            last_data_list.append(data)
            outdata_list.append(data_downsample_list)
        
        out_data = self.proj(torch.cat(last_data_list, dim=-1))
        return out_data, outdata_list

class Dec_net(nn.Module):
    def __init__(self, img_size=[32, 64], patch_size=[2,2], stride=[2,2], outchans_list=[8, 26, 26, 26, 26, 26], channel_num=37, embed_dim=96, lgnet_embed=768, depths=[2,2,2],
                 num_heads=[3,6,12], window_size=[6, 12], mlp_ratio=4, qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                 patch_norm=False,use_checkpoint=False, use_mlp=False, pre_norm=True, attn_type="windowattn"):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.outchans_list = outchans_list
        # print(self.outchans_list)
        self.dec_list = nn.ModuleList()
        self.final_proj_list = nn.ModuleList()
        for i in range(len(self.outchans_list)):

            self.dec_list.append(Transformer_Decoder(img_size=img_size, patch_size=patch_size, in_chans=self.outchans_list[i],
                                                     embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size,
                                                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                     drop_path_rate=drop_path_rate, norm_layer=norm_layer, ape=False, patch_norm=patch_norm,
                                                     use_checkpoint=use_checkpoint, pre_norm=pre_norm, attn_type=attn_type))

            self.final_proj_list.append(nn.ConvTranspose2d(in_channels=embed_dim, out_channels=self.outchans_list[i],
                                                           kernel_size=patch_size, stride=stride))

        self.proj = nn.Linear(lgnet_embed, embed_dim*2**(len(self.depths)-1)*len(self.outchans_list))

    
    def forward(self, x, downsample_list):
        data_proj = self.proj(x)
        data_split = torch.split(data_proj, self.embed_dim*2**(len(self.depths)-1), dim=-1)
        out_data_mean_list = []
        out_data_std_list = []
        for i in range(len(self.outchans_list)):
            out_data = self.dec_list[i](data_split[i], downsample_list[i]).permute(0,3,1,2)
            out_data = self.final_proj_list[i](out_data)

            out_data_mean_list.append(out_data[:, :out_data.shape[1]//2])
            out_data_std_list.append(out_data[:, out_data.shape[1]//2:])
        

        out_data_mean = torch.cat(out_data_mean_list, dim=1)
        out_data_std = torch.cat(out_data_std_list, dim=1)

        out_data = torch.cat((out_data_mean, out_data_std), dim=1)
        # out_data = torch.cat(out_data_list, dim=1)
        return out_data


class LG_net(nn.Module):

    def __init__(self, img_size=[32, 64], patch_size=(1, 1, 1), in_chans=3, out_chans=20,
                 embed_dim=768, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=(2, 4, 8), mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), patch_norm=False,
                 use_checkpoint=False, pre_norm=True, attn_type="windowattn", 
                 use_globalattn=True):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.img_size = img_size
        self.patch_size = patch_size

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layers = Layer(dim=embed_dim,
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=[img_size[-2]//patch_size[-2], img_size[-1]//patch_size[-1]] if i_layer==0 and use_globalattn else window_size,
                                # window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                layer_type="window_block" if i_layer==0 and use_globalattn else "swin_block",
                                # layer_type="swin_block",
                                use_checkpoint=use_checkpoint,
                                pre_norm=pre_norm,
                                attn_type=attn_type
                                )
            self.layers.append(layers)



        self.pos_embed = nn.Parameter(torch.zeros(1, img_size[0]//patch_size[-2]*(img_size[1]//patch_size[-1]), embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x):

        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        T = 1

        x = x + self.pos_embed
        x = self.pos_drop(x)
        if len(self.window_size) == 3:
            x = x.view(B, T, H, W, -1)
        elif len(self.window_size) == 2:
            x = x.view(B, H, W, -1)

        for layer in self.layers:
            x= layer(x)

        return x



class LGUnet_all(nn.Module):
    def __init__(self, img_size=[32, 64], patch_size=(1,1,1), stride=[2,2], in_chans=20, out_chans=20, 
                 enc_depths=[2,2], enc_heads=[3,6], lg_depths=[], lg_heads=[], inchans_list=[20], 
                 outchans_list=[20], enc_dim=96, embed_dim=768, window_size=[4,8], Weather_T=16, 
                 drop_rate=0., attn_drop_rate=0., drop_path=0., use_checkpoint=False, channel_num=37, 
                 inp_length=1, use_mlp=False, pre_norm=True, attn_type="windowattn", use_globalattn=True) -> None:
        super().__init__()
        self.enc = Enc_net(img_size=img_size, patch_size=patch_size, stride=stride, inchans_list=inchans_list, embed_dim=enc_dim, lgnet_embed=embed_dim, depths=enc_depths,
                            num_heads=enc_heads, window_size=window_size, mlp_ratio=4, qkv_bias=True, drop_rate=drop_rate,
                            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path, use_checkpoint=use_checkpoint, 
                            channel_num=channel_num, inp_length=inp_length, use_mlp=use_mlp, pre_norm=pre_norm, attn_type=attn_type)
        lg_patch_size = [stride[i] * 2** (len(enc_depths)-1) for i in range(len(stride))]
        self.net = LG_net(img_size=img_size, patch_size=lg_patch_size, \
                                        embed_dim=embed_dim, depths=lg_depths, num_heads=lg_heads, \
                                        window_size=window_size, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, 
                                        drop_path_rate=drop_path, use_checkpoint=use_checkpoint, pre_norm=pre_norm, 
                                        attn_type=attn_type, use_globalattn=use_globalattn)
        self.dec = Dec_net(img_size=img_size, patch_size=patch_size, stride=stride, outchans_list=outchans_list, 
                            channel_num=channel_num*2, embed_dim=enc_dim, lgnet_embed=embed_dim, depths=enc_depths,
                            num_heads=enc_heads, window_size=window_size, mlp_ratio=4, qkv_bias=True, drop_rate=drop_rate,
                            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path, use_checkpoint=use_checkpoint, 
                            use_mlp=use_mlp, pre_norm=pre_norm, attn_type=attn_type) 

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, data, **kwargs):
        data, enc_list = self.enc(data)
        out = self.net(data)
        out = self.dec(out, enc_list)

        return out

