import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic building blocks
class PointConv(nn.Module):
    """
    A lightweight pointwise (1×1) convolution followed by BatchNorm and ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(PointConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DownSample(nn.Module):
    """
    Downsamples the input by a factor of 2 using 2×2 average pooling.
    """
    def __init__(self):
        super(DownSample, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        return self.pool(x)

class UpSample(nn.Module):
    """
    Upsamples the input by a factor of 2 using bilinear interpolation.
    """
    def __init__(self):
        super(UpSample, self).__init__()
    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

class UpProj(nn.Module):
    """
    Upsampling + projection block.
    Upsamples using UpSample then projects with a PointConv.
    """
    def __init__(self, in_channels, out_channels):
        super(UpProj, self).__init__()
        self.up = UpSample()
        self.proj = PointConv(in_channels, out_channels)
    def forward(self, x):
        return self.proj(self.up(x))

# Novel modules for feature extraction and fusion
class AdaptiveKernelFusionBlock(nn.Module):
    """
    Adaptive Kernel Fusion Block (AKFB)
    """
    def __init__(self, channels):
        super(AdaptiveKernelFusionBlock, self).__init__()
        self.dw_local = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1,
                                  groups=channels, bias=False)
        self.dw_dilated = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=2,
                                    dilation=2, groups=channels, bias=False)
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.fuse_conv = PointConv(channels, channels)
    def forward(self, x):
        local_feat = self.dw_local(x)
        dilated_feat = self.dw_dilated(x)
        descriptor = F.adaptive_avg_pool2d(x, 1)
        alpha = torch.sigmoid(self.gate_conv(descriptor))
        fused = alpha * local_feat + (1 - alpha) * dilated_feat
        out = self.fuse_conv(fused)
        return out

class ResidualGlobalFusionBlock(nn.Module):
    """
    Residual Global Fusion Block (RGFB)
    """
    def __init__(self, channels):
        super(ResidualGlobalFusionBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.tanh = nn.Tanh()
    def forward(self, x):
        g = self.pool(x)
        g = self.conv(g)
        g = self.tanh(g)
        return x + x * g

class LightSkipGate(nn.Module):
    """
    Lightweight Skip Fusion Gate
    """
    def __init__(self, dec_channels, enc_channels):
        super(LightSkipGate, self).__init__()
        self.enc_proj = nn.Identity()
        if dec_channels != enc_channels:
            self.enc_proj = nn.Conv2d(enc_channels, dec_channels, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(dec_channels * 2, dec_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(dec_channels, momentum=0.1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, decoder, encoder):
        encoder = self.enc_proj(encoder)
        fusion = torch.cat([decoder, encoder], dim=1)
        gate = self.sigmoid(self.bn(self.conv(fusion)))
        return decoder + gate * encoder

# MainPipeline Model (Encoder–Bottleneck–Decoder)
class MainPipeline(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, c_list=[5,10,15,25,35,60]):
        super(MainPipeline, self).__init__()
        
        # ---------------- Encoder ----------------
        self.enc0_conv = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_list[0], momentum=0.1),
            nn.ReLU(inplace=True)
        )
        self.enc0_akfb = AdaptiveKernelFusionBlock(c_list[0])
        
        self.down1 = DownSample()
        self.enc1_akfb = AdaptiveKernelFusionBlock(c_list[0])
        self.enc1_proj = PointConv(c_list[0], c_list[1])
        
        self.down2 = DownSample()
        self.enc2_akfb = AdaptiveKernelFusionBlock(c_list[1])
        self.enc2_proj = PointConv(c_list[1], c_list[2])
        
        self.down3 = DownSample()
        self.enc3_akfb = AdaptiveKernelFusionBlock(c_list[2])
        self.enc3_proj = PointConv(c_list[2], c_list[3])
        
        self.down4 = DownSample()
        self.enc4_akfb = AdaptiveKernelFusionBlock(c_list[3])
        self.enc4_proj = PointConv(c_list[3], c_list[4])
        
        self.down5 = DownSample()
        self.enc5_akfb = AdaptiveKernelFusionBlock(c_list[4])
        self.enc5_proj = PointConv(c_list[4], c_list[5])
        
        # ---------------- Bottleneck ----------------
        self.bottleneck_rgfb = ResidualGlobalFusionBlock(c_list[5])
        
        # ---------------- Decoder ----------------
        self.up4 = UpProj(c_list[5], c_list[4])
        self.skip4 = LightSkipGate(c_list[4], c_list[4])
        self.dec4_akfb = AdaptiveKernelFusionBlock(c_list[4])
        
        self.up3 = UpProj(c_list[4], c_list[3])
        self.skip3 = LightSkipGate(c_list[3], c_list[3])
        self.dec3_akfb = AdaptiveKernelFusionBlock(c_list[3])
        
        self.up2 = UpProj(c_list[3], c_list[2])
        self.skip2 = LightSkipGate(c_list[2], c_list[2])
        self.dec2_akfb = AdaptiveKernelFusionBlock(c_list[2])
        
        self.up1 = UpProj(c_list[2], c_list[1])
        self.skip1 = LightSkipGate(c_list[1], c_list[1])
        self.dec1_akfb = AdaptiveKernelFusionBlock(c_list[1])
        
        self.up0 = UpProj(c_list[1], c_list[0])
        self.skip0 = LightSkipGate(c_list[0], c_list[0])
        self.dec0_akfb = AdaptiveKernelFusionBlock(c_list[0])
        
        # ---------------- Final Head ----------------
        self.head = nn.Sequential(
            nn.Conv2d(c_list[0], num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):

        # Encoder 
        e0 = self.enc0_conv(x)           
        e0 = self.enc0_akfb(e0)
        
        e1_in = self.down1(e0)           
        e1_in = self.enc1_akfb(e1_in)
        e1 = self.enc1_proj(e1_in)       
        
        e2_in = self.down2(e1)          
        e2_in = self.enc2_akfb(e2_in)
        e2 = self.enc2_proj(e2_in)       
        
        e3_in = self.down3(e2)           
        e3_in = self.enc3_akfb(e3_in)
        e3 = self.enc3_proj(e3_in)       
        
        e4_in = self.down4(e3)           
        e4_in = self.enc4_akfb(e4_in)
        e4 = self.enc4_proj(e4_in)       
        
        e5_in = self.down5(e4)           
        e5_in = self.enc5_akfb(e5_in)
        e5 = self.enc5_proj(e5_in)       
        
        # Bottleneck
        b = self.bottleneck_rgfb(e5)     
        
        # Decoder
        d4 = self.up4(b)                 
        d4 = self.skip4(d4, e4)
        d4 = self.dec4_akfb(d4)
        
        d3 = self.up3(d4)                
        d3 = self.skip3(d3, e3)
        d3 = self.dec3_akfb(d3)
        
        d2 = self.up2(d3)                
        d2 = self.skip2(d2, e2)
        d2 = self.dec2_akfb(d2)

        d1 = self.up1(d2)                
        d1 = self.skip1(d1, e1)
        d1 = self.dec1_akfb(d1)

        d0 = self.up0(d1)                
        d0 = self.skip0(d0, e0)
        d0 = self.dec0_akfb(d0)
        
        out = self.head(d0)              
        return out
