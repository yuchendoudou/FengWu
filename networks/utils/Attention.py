from re import X
import torch.nn as nn
import torch
from networks.utils.positional_encodings import rope2
from networks.utils.utils import window_partition, window_reverse


class SD_attn(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., shift_size=[0, 0, 0], dilated_size=[1,1,1]) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = torch.tensor(head_dim ** -0.5)
        
        self.dilated_size = dilated_size[-len(window_size):]
        self.window_size = window_size
        self.shift_size = shift_size
        self.total_window_size = [window_size[i] * dilated_size[i] for i in range(len(window_size))]
        


        self.rope_quad = rope2(self.window_size, head_dim)
       
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.position_enc = rope2(window_size, head_dim)


    def create_mask(self, x):

  
        _, H, W, _ = x.shape
        img_mask = torch.zeros((1, H, W, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], 0),
                    slice(0, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.total_window_size)  
        mask_windows = mask_windows.reshape(-1, *self.total_window_size, 1)
        B_ = mask_windows.shape[0]

        mask_windows = window_partition(mask_windows, self.dilated_size).reshape(B_, -1, 
                                    self.dilated_size[0]*self.dilated_size[1], 1).permute(
                                    0, 2, 1, 3).reshape(B_*self.dilated_size[0]*self.dilated_size[1], -1)

            
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -torch.inf).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    
    def forward(self, x):

        T=1

        _, H, W, C = x.shape

        if (self.shift_size[-1] == 0) or (self.total_window_size[-1] == W):
            mask = None
        else:
            mask = self.create_mask(x).to(x)

        if self.shift_size[-1] > 0:
            if len(self.window_size) == 3:
                shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            elif len(self.window_size) == 2:
                shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x=x
            mask = None



        x_windows = window_partition(shifted_x, self.total_window_size)  
        x_windows = x_windows.reshape(-1, *self.total_window_size, C)
        B = x_windows.shape[0]
        x_windows = window_partition(x_windows, self.dilated_size).reshape(B, -1, 
                                    self.dilated_size[0]*self.dilated_size[1], C).permute(
                                    0, 2, 1, 3).reshape(B*self.dilated_size[0]*self.dilated_size[1], -1, C)
        B_, N, C = x_windows.shape


        qkv = self.qkv(x_windows).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0) 

        q = self.position_enc(q.reshape(-1, *self.window_size, C // self.num_heads)).reshape(B_, self.num_heads, -1, C // self.num_heads)
        k = self.position_enc(k.reshape(-1, *self.window_size, C // self.num_heads)).reshape(B_, self.num_heads, -1, C // self.num_heads)


        q = q * self.scale.to(q)
        attn = (q @ k.transpose(-2, -1))


        if mask is not None:
            nW = mask.shape[0] 

            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)


        save_attn = attn.mean(dim=-3)

        attn_windows = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        if len(self.window_size) == 3:
            attn_windows = attn_windows.reshape(B, -1, N, C).permute(0, 2, 1, 3).reshape(
                                            -1, self.dilated_size[0]*self.dilated_size[1]*self.dilated_size[2], C)
            attn_windows = window_reverse(attn_windows, self.dilated_size, *self.total_window_size)
        elif len(self.window_size) == 2:
            attn_windows = attn_windows.reshape(B, -1, N, C).permute(0, 2, 1, 3).reshape(
                                            -1, self.dilated_size[0]*self.dilated_size[1], C)
            attn_windows = window_reverse(attn_windows, self.dilated_size, 1, *self.total_window_size)
            
        shifted_x = window_reverse(attn_windows, self.total_window_size, T, H, W)

        if self.shift_size[0] > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, save_attn

