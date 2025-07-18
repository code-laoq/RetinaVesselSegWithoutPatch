import torch
from torch import nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from torchinfo import summary
from torchvision import models
# ----------------------------
# DropBlock2D 实现
# ----------------------------
class DropBlock2D(nn.Module):
    """
    DropBlock2D with zero fill and scaling to maintain expected activation.

    Args:
        block_size (int): Size of each dropped block.
        drop_prob (float): Probability of dropping any given block.
    """
    def __init__(self, block_size=3, drop_prob=0.2):
        super(DropBlock2D, self).__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        # No-op during evaluation or zero drop probability
        if not self.training or self.drop_prob <= 0.0:
            return x

        B, C, H, W = x.shape
        block_size = min(self.block_size, H, W)
        # Fallback to standard Dropout2d if block_size == 1
        if block_size == 1:
            return F.dropout2d(x, p=self.drop_prob, training=True)

        # Compute valid starting locations
        n_h = H - block_size + 1
        n_w = W - block_size + 1

        # Randomly sample blocks to drop
        noise = torch.rand(B, C, n_h, n_w, device=x.device, dtype=x.dtype)
        block_mask = (noise < self.drop_prob).float()

        # Expand the small block mask to full size
        block_mask = F.pad(block_mask,
                           (0, block_size - 1, 0, block_size - 1))
        block_mask = F.max_pool2d(block_mask,
                                  kernel_size=block_size,
                                  stride=1)

        # Invert mask: 1 = keep, 0 = drop
        mask = 1 - block_mask

        # Scale mask so that E[mask] = 1, preserving activation magnitude
        mask = mask * (1.0 / (1.0 - self.drop_prob))

        # Apply mask
        return x * mask


class UpSample(nn.Module):
    def __init__(self,ch_in, ch_out,scale_factor=2,mode=None):
        super(UpSample,self).__init__()
        # 上采样方式，
        if mode == 'bilinear':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
                nn.Conv2d(ch_in, ch_out, kernel_size=1)  # 因为双线性插值不会自动调整通道数，所以这里手动调整
            )
        elif mode == 'pixelshuffle':
            self.up = nn.Sequential(
                nn.Conv2d(ch_in, scale_factor * ch_out * 2 , kernel_size=1),  #PixelShuffle需要输入特征图的通道数是输出通道数的r^2倍，也就是拿通道换分辨率的提示
                nn.PixelShuffle(scale_factor)  # 上采样
            )
        elif mode == 'tran_conv':
            self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
        else:
            pass

    def forward(self,x):
        return self.up(x)

# ----------------------------
# DoubleConvBlock（含正则化）
# ----------------------------
class DoubleConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, ch_mid=None,
                 regularization_type='none',
                 dropout_p=0.5, dropblock_block_size=3, dropblock_drop_prob=0.2):
        super(DoubleConvBlock, self).__init__()
        if ch_mid is None:
            ch_mid = ch_out

        # 构建模块列表
        layers = [
            nn.ReflectionPad2d(1),   #padding=(kernel_size - 1) // 2
            nn.Conv2d(ch_in, ch_mid, kernel_size=3, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(ch_mid),
            nn.ReLU(inplace=True)
        ]
        # 插入正则化层
        if regularization_type == 'spatial_dropout':
            layers.append(nn.Dropout2d(p=dropout_p))
        elif regularization_type == 'dropblock':
            layers.append(DropBlock2D(block_size=dropblock_block_size, drop_prob=dropblock_drop_prob))
        # 添加第二个卷积
        layers.extend([
            nn.ReflectionPad2d(1),   #padding=(kernel_size - 1) // 2
            nn.Conv2d(ch_mid, ch_out, kernel_size=3, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self,ch_in, ch_out,
            regularization_type='none',
            dropout_p=0.5, dropblock_block_size=3, dropblock_drop_prob=0.2
    ):
        super(ResidualBlock,self).__init__()
        # 构建模块列表
        layers = [
            nn.ReflectionPad2d(1),   #padding=(kernel_size - 1) // 2
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        ]
        # 插入正则化层
        if regularization_type == 'spatial_dropout':
            layers.append(nn.Dropout2d(p=dropout_p))
        elif regularization_type == 'dropblock':
            layers.append(DropBlock2D(block_size=dropblock_block_size, drop_prob=dropblock_drop_prob))
        # 添加第二个卷积
        layers.extend([
            nn.ReflectionPad2d(1),   #padding=(kernel_size - 1) // 2
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_out),
        ])
        self.conv_block = nn.Sequential(*layers)
        self.conv_1x1=nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 1, 1),
            nn.BatchNorm2d(ch_out)
        )
        self.activation=nn.ReLU(inplace=True)

    def forward(self, x):
        origin=self.conv_1x1(x)
        output=origin+self.conv_block(x)
        return self.activation(output)

class EncoderBlock(nn.Module):
    def __init__(self, ch_in, ch_out,
                 regularization_type='none',
                 dropout_p=0.5, dropblock_block_size=3, dropblock_drop_prob=0.2):
        super(EncoderBlock, self).__init__()
        self.conv = DoubleConvBlock(ch_in, ch_out, regularization_type=regularization_type,
            dropout_p=dropout_p, dropblock_block_size=dropblock_block_size, dropblock_drop_prob=dropblock_drop_prob)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.conv(x)
        down = self.pool(feat)
        return feat, down

class ResEncoderBlock(nn.Module):
    def __init__(self, ch_in, ch_out,
                 regularization_type='none',
                 dropout_p=0.5, dropblock_block_size=3, dropblock_drop_prob=0.2):
        super(ResEncoderBlock, self).__init__()
        self.conv = ResidualBlock(ch_in, ch_out, regularization_type=regularization_type,
            dropout_p=dropout_p, dropblock_block_size=dropblock_block_size, dropblock_drop_prob=dropblock_drop_prob)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.conv(x)
        down = self.pool(feat)
        return feat, down

class DecoderBlock(nn.Module):
    def __init__(self, ch_in, ch_mid, ch_out , mode):
        super().__init__()
        # 上采样
        self.up = UpSample(ch_in, ch_mid, scale_factor=2, mode = mode)
        # 特征融合:上一层输出channel=ch_mid，对应编码器的特征图channel=ch_in
        self.conv = DoubleConvBlock(ch_in+ch_mid, ch_out, ch_mid)

    def forward(self, decoder, encoder):
        #上采样
        decoder = self.up(decoder)           #→ [ch_mid, H*2, W*2]
        # 拼接
        decoder = torch.cat([decoder, encoder], dim=1)      #→ [ch_in+ch_mid, H*2, W*2]
        # 特征融合，拼接后通道数为ch_in+ch_mid
        decoder = self.conv(decoder)
        return decoder

class Bottleneck(nn.Module):
    def __init__(self, ch_in, ch_mid, ch_out):
        super(Bottleneck,self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch_in, ch_mid// 4, 1,bias=False), nn.BatchNorm2d(ch_mid// 4), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),   #padding=(kernel_size - 1) // 2
            nn.Conv2d(ch_mid// 4, ch_mid// 4, 3, padding=0,bias=False), nn.BatchNorm2d(ch_mid// 4), nn.ReLU(inplace=True),
            nn.Conv2d(ch_mid// 4, ch_out, 1,bias=False), nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True),
        )
    def forward(self,x):
        res = x+self.bottleneck(x)
        return res

class UNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=2,upsample_mode='pixelshuffle'):
        super(UNet,self).__init__()
        # 编码器部分
        self.enc1 = EncoderBlock(img_ch, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512, regularization_type='spatial_dropout')
        # 中心块
        self.center = Bottleneck(512,1024,512)
        # 解码器部分
        self.dec4 = DecoderBlock(512, 256, 256, mode=upsample_mode)
        self.dec3 = DecoderBlock(256, 128, 128, mode=upsample_mode)
        self.dec2 = DecoderBlock(128, 64, 64, mode=upsample_mode)
        self.dec1 = DecoderBlock(64, 32, 32, mode=upsample_mode)
        # 输出层
        self.final = nn.Conv2d(32, output_ch, kernel_size=1 , bias=True)

    def forward(self, x):
        # 编码器
        enc1_feat, enc1_down = self.enc1(x)
        enc2_feat, enc2_down = self.enc2(enc1_down)
        enc3_feat, enc3_down = self.enc3(enc2_down)
        enc4_feat, enc4_down = self.enc4(enc3_down)
        # 中心块
        center = cp.checkpoint(self.center, enc4_down)
        # 解码器
        dec4 = cp.checkpoint(self.dec4, center, enc4_feat)
        dec3 = cp.checkpoint(self.dec3, dec4, enc3_feat)
        dec2 = self.dec2(dec3, enc2_feat)
        dec1 = self.dec1(dec2, enc1_feat)
        # 最终输出
        final = self.final(dec1)
        return final

class ResUNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=2,upsample_mode='pixelshuffle'):
        super(ResUNet,self).__init__()
        # 编码器部分
        self.enc1 = ResEncoderBlock(img_ch, 64)
        self.enc2 = ResEncoderBlock(64, 128)
        self.enc3 = ResEncoderBlock(128, 256)
        self.enc4 = ResEncoderBlock(256, 512, regularization_type='spatial_dropout')
        # 中心块
        self.center = Bottleneck(512,1024,512)
        # 解码器部分
        self.dec4 = DecoderBlock(512, 256, 256, mode=upsample_mode)
        self.dec3 = DecoderBlock(256, 128, 128, mode=upsample_mode)
        self.dec2 = DecoderBlock(128, 64, 64, mode=upsample_mode)
        self.dec1 = DecoderBlock(64, 32, 32, mode=upsample_mode)
        # 输出层
        self.final = nn.Conv2d(32, output_ch, kernel_size=1 , bias=True)

    def forward(self, x):
        # 编码器
        enc1_feat, enc1_down = self.enc1(x)
        enc2_feat, enc2_down = self.enc2(enc1_down)
        enc3_feat, enc3_down = self.enc3(enc2_down)
        enc4_feat, enc4_down = self.enc4(enc3_down)
        # 中心块
        center = self.center(enc4_down)
        # 解码器
        dec4 = self.dec4(center, enc4_feat)
        dec3 = self.dec3(dec4, enc3_feat)
        dec2 = self.dec2(dec3, enc2_feat)
        dec1 = self.dec1(dec2, enc1_feat)
        # 最终输出
        final = self.final(dec1)
        return final

if __name__ == '__main__':
    # center=torch.randn(1,512,8,8)
    # enc4_feat = torch.randn(1, 512, 16, 16)
    # dec = DecoderBlock(512, 256, 256)
    # y = dec(center, enc4_feat)
    # print(y.size())

    x = torch.randn(1, 1, 128, 128)
    unet = UNet(1,2)

    resunet = ResUNet(1,2)

    summary(unet, input_size=(1, 1, 128, 128))
    # summary(resunet, input_size=(1, 1, 128, 128))
    # output = net(x)
    # print(output.size())