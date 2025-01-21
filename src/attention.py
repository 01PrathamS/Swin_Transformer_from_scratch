import torch 
from torch import nn 
import math 
from einops import rearrange 
from torch.nn import functional as F 
#from utils import RelativeEmbeddings


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
