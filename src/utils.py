import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 
from einops import rearrange 

class SwinEmbedding(nn.Module): 

    def __init__(self, patch_size=4, C=96): 
        super().__init__() 
        self.linear_embedding = nn.Conv2d(in_channels=3, out_channels=C, kernel_size=patch_size, stride=patch_size) 
        self.layer_norm = nn.LayerNorm(C) 
        self.relu = nn.ReLU() 

    def forward(self, x):
        x = self.linear_embedding(x) 
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.relu(self.layer_norm(x))
        return x 
    

class PatchMerging(nn.Module):

    def __init__(self, C):
        super().__init__() 
        self.linear = nn.Linear(in_features=4*C, out_features=2*C) 
        self.layer_norm = nn.LayerNorm(2*C) 

    def forward(self, x):
        height = width = int(math.sqrt(x.shape[1])/2) 
        x = rearrange(x, 'b (h s1 w s2) c -> b (h w) (s2 s1 c)', s1 = 2, s2 = 2, h = height, w = width) 
        x = self.layer_norm(self.linear(x))
        return x 
    
class RelativeEmbedding(nn.Module):

    def __init__(self, window_size=7):
        super().__init__() 
        B = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        x = torch.arange(1, window_size + 1, 1 / window_size) 
        x = (x[None, :] - x[:, None]).int() 
        y = torch.cat([torch.arange(1, window_size + 1)] * window_size)
        y = (y[None, :] - y[:, None])

        self.embeddings = nn.Parameter(B[x[:, :], y[:, :]], requires_grad=False) 

    def forward(self, x):
        return x + self.embeddings 
    
class SwinEncoderBlock(nn.Module):
    
    def __init__(self, embed_dim, num_heads, window_size, mask):
        super().__init__() 
        self.layer_norm = nn.LayerNorm(embed_dim) 
        self.dropout = nn.Dropout(0.1) 
        self.WMSA = ShiftedWindowMSA(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, mask=mask) 

        self.MLP1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4), 
            nn.GELU(), 
            nn.Linear(embed_dim*4, embed_dim)
        )

    def forward(self, x):
        height, widht = x.shape[1:3] 
        res1 = self.dropout(self.WMSA(self.layer_norm(x)) + x) 
        x = self.layer_norm(res1) 
        x = self.MLP1(x) 
        return self.dropout(x + res1)
    
class AlternatingEncoderBlock(nn.Module): 

    def __init__(self, embed_dim, num_heads, window_size=7):
        super().__init__() 
        self.WSA = SwinEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, mask=False) 
        self.SWSA = SwinEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, mask=True) 

    def forward(self, x):
        return self.SWSA(self.WSA(x))
    
class ShiftedWindowMSA(nn.Module):

    def __init__(self, embed_dim, num_heads, window_size=7, mask=False): 
        super().__init__() 
        self.embed_dim = embed_dim 
        self.num_heads = num_heads 
        self.window_size = window_size 
        self.mask = mask 

        self.proj1 = nn.Linear(in_features=embed_dim, out_features=embed_dim*3) 
        self.proj2 = nn.Linear(in_features=embed_dim, out_features=embed_dim) 
        self.embedding = RelativeEmbedding()

    def forward(self, x):

        height = width = int(math.sqrt(x.shape[1]))

        x = self.proj1(x) 

        x = rearrange(x, 'b (h w) (c K) -> b h w c K', K = 3, h=height, w=width)

        if self.mask: 
            x = torch.roll(x, (-self.window_size//2, -self.window_size//2), dims=(1, 2)) 

        x = rearrange(x, 'b (h m1) (w m2) (H E) K -> b H h w (m1 m2) E K', H = self.num_heads, m1 = self.window_size, m2 = self.window_size) 

        Q, K, V = x.chunk(3, dim=6) 
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)

        att_scores = (Q @ K.transpose(4, 5) / math.sqrt(self.embed_dim / self.num_heads)) 

        if self.mask: 
            row_mask = torch.zeros((self.window_size**2, self.window_size**2))
            row_mask[-self.window_size * (self.window_size//2):, 0:-self.window_size * (self.window_size//2)] = float('-inf')
            row_mask[0:-self.window_size * (self.window_size//2), -self.window_size * (self.window_size//2):] = float('-inf') 
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_size, w2=self.window_size) 
            att_scores[:, :, -1, :] += row_mask 
            att_scores[:, :, :, -1] += column_mask

        att = F.softmax(att_scores, dim=-1) @ V 

        x = rearrange(att, 'b H h w (m1 m2) E -> b (h m1) (w m2) (H E)', m1 = self.window_size, m2 = self.window_size) 

        if self.mask: 
            x = torch.roll(x, (self.window_size//2, self.window_size//2), (1, 2)) 

        # Rearrange back to original shape and project to final output 
        x = rearrange(x, 'b h w c -> b (h w) c')

        return self.proj2(x)

