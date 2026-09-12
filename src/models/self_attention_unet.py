import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import Down, Up, OutConv, DoubleConv

class SelfAttentionBlock(nn.Module):
    """ Lightweight Spatial Self-Attention """
    def __init__(self, in_channels):
        super(SelfAttentionBlock, self).__init__()
        # Reduce channels for Q, K, V to keep it lightweight
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # N = W * H
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1) # B x N x C/8
        proj_key = self.key_conv(x).view(batch_size, -1, width * height) # B x C/8 x N
        
        # Energy: B x N x N
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1) # B x N x N
        
        proj_value = self.value_conv(x).view(batch_size, -1, width * height) # B x C x N
        
        # Output: B x C x N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        out = self.gamma * out + x
        return out

class SelfAttentionUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super(SelfAttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Insert Self-Attention at the deepest representation (bottleneck)
        self.attention = SelfAttentionBlock(1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Apply self-attention at bottleneck
        x5_att = self.attention(x5)
        
        x = self.up1(x5_att, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)
