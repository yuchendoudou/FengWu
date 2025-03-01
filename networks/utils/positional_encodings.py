import torch
import torch.nn as nn




class rope2(nn.Module):
    def __init__(self, shape, dim, origin_shape=[0,0]) -> None:
        super().__init__()
        
        coords_0 = torch.arange(shape[0])
        coords_1 = torch.arange(shape[1])
        
        if origin_shape[0] > 0:
            coords_0 = coords_0 / (shape[0] - 1) * (origin_shape[0] - 1)
            coords_1 = coords_1 / (shape[1] - 1) * (origin_shape[1] - 1)
        coords = torch.stack(torch.meshgrid([coords_0, coords_1], indexing="ij")).reshape(2, -1)

        half_size = dim // 2
        self.dim1_size = half_size // 2
        self.dim2_size = half_size - half_size // 2
        freq_seq1 = torch.arange(0, self.dim1_size) / self.dim1_size
        freq_seq2 = torch.arange(0, self.dim2_size) / self.dim2_size
        inv_freq1 = 10000 ** -freq_seq1
        inv_freq2 = 10000 ** -freq_seq2

        sinusoid1 = coords[0].unsqueeze(-1) * inv_freq1    
        sinusoid2 = coords[1].unsqueeze(-1) * inv_freq2     

        self.sin1 = torch.sin(sinusoid1).reshape(*shape, sinusoid1.shape[-1])
        self.cos1 = torch.cos(sinusoid1).reshape(*shape, sinusoid1.shape[-1])
        self.sin2 = torch.sin(sinusoid2).reshape(*shape, sinusoid2.shape[-1])
        self.cos2 = torch.cos(sinusoid2).reshape(*shape, sinusoid2.shape[-1])


    def forward(self, x):

        self.sin1 = self.sin1.to(x)
        self.cos1 = self.cos1.to(x)
        self.sin2 = self.sin2.to(x)
        self.cos2 = self.cos2.to(x)

        x11, x21, x12, x22 = x.split([self.dim1_size, self.dim2_size, \
                                        self.dim1_size, self.dim2_size], dim=-1)
        
        res = torch.cat([x11 * self.cos1 - x12 * self.sin1, x21 * self.cos2 - x22 * self.sin2, \
                        x12 * self.cos1 + x11 * self.sin1, x22 * self.cos2 + x21 * self.sin2], dim=-1)

        return res

#3D旋转式位置编码

class rope3(nn.Module):
    def __init__(self, shape, dim) -> None:
        super().__init__()
        
        coords_0 = torch.arange(shape[0])
        coords_1 = torch.arange(shape[1])
        coords_2 = torch.arange(shape[2])
        coords = torch.stack(torch.meshgrid([coords_0, coords_1, coords_2], indexing="ij")).reshape(3, -1)

        half_size = dim // 2
        self.dim1_2_size = half_size // 3
        self.dim3_size = half_size - half_size // 3 * 2
        freq_seq1_2 = torch.arange(0, self.dim1_2_size) / self.dim1_2_size
        freq_seq3 = torch.arange(0, self.dim3_size) / self.dim3_size
        inv_freq1_2 = 10000 ** -freq_seq1_2
        inv_freq3 = 10000 ** -freq_seq3

        sinusoid1 = coords[0].unsqueeze(-1) * inv_freq1_2    
        sinusoid2 = coords[1].unsqueeze(-1) * inv_freq1_2    
        sinusoid3 = coords[2].unsqueeze(-1) * inv_freq3    

        self.sin1 = torch.sin(sinusoid1).reshape(*shape, sinusoid1.shape[-1])
        self.cos1 = torch.cos(sinusoid1).reshape(*shape, sinusoid1.shape[-1])
        self.sin2 = torch.sin(sinusoid2).reshape(*shape, sinusoid2.shape[-1])
        self.cos2 = torch.cos(sinusoid2).reshape(*shape, sinusoid2.shape[-1])
        self.sin3 = torch.sin(sinusoid3).reshape(*shape, sinusoid3.shape[-1])
        self.cos3 = torch.cos(sinusoid3).reshape(*shape, sinusoid3.shape[-1])


    def forward(self, x):

        self.sin1 = self.sin1.to(x)
        self.cos1 = self.cos1.to(x)
        self.sin2 = self.sin2.to(x)
        self.cos2 = self.cos2.to(x)
        self.sin3 = self.sin3.to(x)
        self.cos3 = self.cos3.to(x)

        x11, x21, x31, x12, x22, x32 = x.split([self.dim1_2_size, self.dim1_2_size, self.dim3_size, \
                                            self.dim1_2_size, self.dim1_2_size, self.dim3_size], dim=-1)
        
        res = torch.cat([x11 * self.cos1 - x12 * self.sin1, x21 * self.cos2 - x22 * self.sin2, x31 * self.cos3 - x32 * self.sin3, \
                        x12 * self.cos1 + x11 * self.sin1, x22 * self.cos2 + x21 * self.sin2, x32 * self.cos3 + x31 * self.sin3], dim=-1)

        return res

