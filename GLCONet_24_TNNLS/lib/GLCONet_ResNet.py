
import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.GLCONet_module import Local, Global, Global_1, FI_1,FI_2, GL_FI, BasicConv2d

'''
backbone: resnet50
'''

class Network(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=64):
        super(Network, self).__init__()
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)

        self.dePixelShuffle = torch.nn.PixelShuffle(2)

        self.reduce  = nn.Sequential(
            BasicConv2d(channels*2, channels, kernel_size=1),
            BasicConv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.up = nn.Sequential(
            BasicConv2d(channels//4, channels, kernel_size=1),
            BasicConv2d(channels, channels, kernel_size=3, padding=1)
        )

        self.Global5 = Global(2048, channels)
        self.Global4 = Global(1024+channels, channels)
        self.Global3 = Global(512+channels, channels)
        self.Global2 = Global(256+channels, channels)

        self.Global6 = Global_1(2048 + channels, channels)
        self.GL_FI   = GL_FI(channels)

        self.Local5 = Local(2048, channels)
        self.Local4 = Local(1024+channels, channels)
        self.Local3 = Local(512+channels, channels)
        self.Local2 = Local(256+channels, channels)

        self.FI_1 = FI_1(channels,channels)
        self.FI_2 = FI_2(channels,channels)
    def forward(self, x):
        image = x

        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats

        g4 = self.Global5(x4)
        g4_up = self.up(self.dePixelShuffle(g4))

        g3 = self.Global4(torch.cat((x3,g4_up),1))
        g3_up = self.up(self.dePixelShuffle(g3))

        g2 = self.Global3(torch.cat((x2,g3_up),1))
        g2_up = self.up(self.dePixelShuffle(g2))

        g1 = self.Global2(torch.cat((x1,g2_up),1))


        l4 = self.Local5(x4)
        l4_up = self.up(self.dePixelShuffle(l4))

        l3 = self.Local4(torch.cat((x3,l4_up),1))
        l3_up = self.up(self.dePixelShuffle(l3))

        l2 = self.Local3(torch.cat((x2,l3_up),1))
        l2_up = self.up(self.dePixelShuffle(l2))
        l1 = self.Local2(torch.cat((x1,l2_up),1))


        p1  = self.Global6(torch.cat((x4,self.GL_FI(g4,l4)),1))

        x4 = self.FI_1(g4,l4,p1)
        x3 = self.FI_2(g3,l3,x4,p1)
        x2 = self.FI_2(g2,l2,x3,x4)
        x1 = self.FI_2(g1,l1,x2,x3)

        f4 = F.interpolate(x4, size=image.size()[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(x3, size=image.size()[2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(x2, size=image.size()[2:], mode='bilinear', align_corners=True)
        f1 = F.interpolate(x1, size=image.size()[2:], mode='bilinear', align_corners=True)

        f5 = F.interpolate(p1, size=image.size()[2:], mode='bilinear', align_corners=True)

        return f5, f4, f3, f2, f1

