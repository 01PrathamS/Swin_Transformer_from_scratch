import torch 
from torch import nn 
import math 
from einops import rearrange 
from torch.nn import functional as F 
from utils import RelativeEmbeddings


class ShiftedWindowMSA(nn.Module):

    def __init__(self, embed_dim, num_heads, window_size=7, mask=False):
        super(ShiftedWindowMSA, self).__init__() 
        self.embed_dim = embed_dim 
        self.num_labels = num_heads 
        self.window_size = window_size 
        self.mask = mask 

        self.proj1 = nn.Linear(in_features=embed_dim, out_features=embed_dim*3) 
        self.proj2 = nn.Linear(in_features=embed_dim, out_features=embed_dim) 
        self.embeddings  = RelativeEmbeddings() 

    def forward(self, x):

        height = width = int(math.sqrt(x.shape[1]))

        x = self.proj1(x) 

        x = rearrange(x, 'b (h w) (d k) -> b h w c K', K = 3, h = height, w = width) 

        # shifted window masking

        if self.mask: 
            x = torch.roll(x, (-self.window_size//2, -self.window_size//2), dims=(1,2))

        x = rearrange(x, 'b (h m1) (w m2) (H E) K -> b H h w (m1 m2) E K', H=self.num_heads, m1=self.window_size, m2=self.window_size) 

        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)

        att_scores = (Q @ K.transpose(4, 5) / math.sqrt(self.embed_dim / self.num_heads))


        if self.mask: 
            row_mask = torch.zeros((self.window_size**2, self.window_size**2))
            row_mask[-self.window_size * (self.window_size // 2):, 0:-self.window_size * (self.window_size//2)] = float('-inf') 
            row_mask[0:-self.window_size * (self.window_size//2), -self.window_size * (self.window_size//2):] = float('-inf') 
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_size, w2=self.window_size).cuda() 
            att_scores[:, :, -1, :] += row_mask 
            att_scores[:, :, :, -1] += column_mask 

        att = F.softmax(att_scores, dim=-1) @ V 

        x = rearrange(att, 'b H h w (m1 m2) E -> b (h m1) (w m2) (H E)', m1=self.window_size, m2=self.window_size) 

        if self.mask: 
            x = torch.roll(x, (self.window_size//2, self.window_size//2), (1, 2)) 

        x = rearrange(x, 'b h w c -> b (h w) c')
        return self.proj2(x)