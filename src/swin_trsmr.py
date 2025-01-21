import torch 
from torch import nn 
from torch.nn import functional as F 

#from utils import AlternatingEncoderBlock, PatchMerging, SwinEmbedding 

class SwinTransformer(nn.Module):
    
    def __init__(self):
        super().__init__() 
        self.Embedding = SwinEmbedding() 
        self.PatchMerge1 = PatchMerging(96)
        self.PatchMerge2 = PatchMerging(192) 
        self.PatchMerge3 = PatchMerging(384) 

        self.Stage1 = AlternatingEncoderBlock(96, 3) 
        self.Stage2 = AlternatingEncoderBlock(192, 3) 
        self.Stage3_1 = AlternatingEncoderBlock(384, 3) 
        self.Stage3_2 = AlternatingEncoderBlock(384, 12) 
        self.Stage3_3 = AlternatingEncoderBlock(384, 12) 
        self.stage4 = AlternatingEncoderBlock(784, 24)

    def forward(self, x):
        x = self.Embedding(x) 
        x = self.PatchMerge1(self.Stage1(x)) 
        x = self.PatchMerge2(self.Stage2(x))    
        x = self.Stage3_1(x) 
        x = self.Stage3_2(x) 
        x = self.Stage3_3(x) 
        x = self.PatchMerge3(x) 
        x = self.Stage4(x) 
        return x 
    
def main(): 
    x = torch.randn((1, 3, 224, 224)) 
    model = SwinTransformer() 
    print(model(x).shape)

if __name__ == "__main__":
    main()
