import numpy as np
import torch
import torch.nn as nn
from torch.fft import fft2, fftshift, ifft2, ifftshift
import torch.nn.functional as F
from option import getOption
import pdb

opt = getOption()



class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.selu = nn.SELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.selu(x)

        return x


class DWConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, dilation, padding):
        super(DWConv, self).__init__()
        self.out_channel = out_channel
        self.DWConv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, padding=padding, groups=in_channel,
                                dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.selu = nn.SELU()

    def forward(self, x):
        x = self.DWConv(x)
        out = self.selu(self.bn(x))

        return out


class DWSConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, padding, kernels_per_layer):
        super(DWSConv, self).__init__()
        self.out_channel = out_channel
        self.DWConv = nn.Conv2d(in_channel, in_channel * kernels_per_layer, kernel_size=kernel, padding=padding,
                                groups=in_channel, bias=False)
        self.bn = nn.BatchNorm2d(in_channel * kernels_per_layer)
        self.selu = nn.SELU()
        self.PWConv = nn.Conv2d(in_channel * kernels_per_layer, out_channel, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.DWConv(x)
        x = self.selu(self.bn(x))
        out = self.PWConv(x)
        out = self.selu(self.bn2(out))

        return out
class Frequency_Edge_Module(nn.Module):
    def __init__(self, radius, channel):
        super(Frequency_Edge_Module, self).__init__()
        self.radius = radius
        self.UAM = UnionAttentionModule(channel, only_channel_tracing=True)

        # DWS + DWConv
        self.DWSConv = DWSConv(channel, channel, kernel=3, padding=1, kernels_per_layer=1)
        self.DWConv1 = nn.Sequential(
            DWConv(channel, channel, kernel=1, padding=0, dilation=1),
            BasicConv2d(channel, channel // 4, 1),
        )
        self.DWConv2 = nn.Sequential(
            DWConv(channel, channel, kernel=3, padding=1, dilation=1),
            BasicConv2d(channel, channel // 4, 1),
        )
        self.DWConv3 = nn.Sequential(
            DWConv(channel, channel, kernel=3, padding=3, dilation=3),
            BasicConv2d(channel, channel // 4, 1),
        )
        self.DWConv4 = nn.Sequential(
            DWConv(channel, channel, kernel=3, padding=5, dilation=5),
            BasicConv2d(channel, channel // 4, 1),
        )
        self.conv = BasicConv2d(channel, 1, 1)

    def distance(self, i, j, imageSize, r):
        dis = np.sqrt((i - imageSize / 2) ** 2 + (j - imageSize / 2) ** 2)
        if dis < r:
            return 1.0
        else:
            return 0

    def mask_radial(self, img, r):
        batch, channels, rows, cols = img.shape
        mask = torch.zeros((rows, cols), dtype=torch.float32)
        for i in range(rows):
            for j in range(cols):
                mask[i, j] = self.distance(i, j, imageSize=rows, r=r)
        return mask

    def forward(self, x):
        """
        Input:
            The first encoder block representation: (B, C, H, W)
        Returns:
            Edge refined representation: X + edge (B, C, H, W)
        """
        x_fft = fft2(x, dim=(-2, -1))
        x_fft = fftshift(x_fft)

        # Mask -> low, high separate
        mask = self.mask_radial(img=x, r=self.radius).cuda()
        high_frequency = x_fft * (1 - mask)
        x_fft = ifftshift(high_frequency)
        x_fft = ifft2(x_fft, dim=(-2, -1))
        x_H = torch.abs(x_fft)

        x_H, _ = self.UAM.Channel_Tracer(x_H)
        edge_maks = self.DWSConv(x_H)
        skip = edge_maks.clone()

        edge_maks = torch.cat([self.DWConv1(edge_maks), self.DWConv2(edge_maks),
                               self.DWConv3(edge_maks), self.DWConv4(edge_maks)], dim=1) + skip
        edge = torch.relu(self.conv(edge_maks))

        x = x + edge  # Feature + Masked Edge information

        return x, edge


class RFB_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_Block, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)

        x = self.relu(x_cat + self.conv_res(x))
        return x


class GlobalAvgPool(nn.Module):
    def __init__(self, flatten=False):
        super(GlobalAvgPool, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class UnionAttentionModule(nn.Module):
    def __init__(self, n_channels, only_channel_tracing=False):
        super(UnionAttentionModule, self).__init__()
        self.GAP = GlobalAvgPool()
        self.confidence_ratio = opt.gamma
        self.bn = nn.BatchNorm2d(n_channels)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(n_channels),
            nn.Dropout3d(self.confidence_ratio)
        )
        self.channel_q = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.channel_k = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.channel_v = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)

        self.fc = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1,
                            padding=0, bias=False)

        if only_channel_tracing == False:
            self.spatial_q = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, stride=1,
                                       padding=0, bias=False)
            self.spatial_k = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, stride=1,
                                       padding=0, bias=False)
            self.spatial_v = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, stride=1,
                                       padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def masking(self, x, mask):
        # pdb.set_trace()
        mask = mask.squeeze(3).squeeze(2)
        threshold = torch.quantile(mask, self.confidence_ratio, dim=-1, keepdim=True)
        mask[mask <= threshold] = 0.0
        mask = mask.unsqueeze(2).unsqueeze(3)
        mask = mask.expand(-1, x.shape[1], x.shape[2], x.shape[3]).contiguous()
        masked_x = x * mask

        return masked_x

    def Channel_Tracer(self, x):
        # pdb.set_trace()
        avg_pool = self.GAP(x)
        x_norm = self.norm(avg_pool)

        q = self.channel_q(x_norm).squeeze(-1)
        k = self.channel_k(x_norm).squeeze(-1)
        v = self.channel_v(x_norm).squeeze(-1)

        # softmax(Q*K^T)
        QK_T = torch.matmul(q, k.transpose(1, 2))
        alpha = F.softmax(QK_T, dim=-1)

        # a*v
        att = torch.matmul(alpha, v).unsqueeze(-1)
        att = self.fc(att)
        att = self.sigmoid(att)

        output = (x * att) + x
        alpha_mask = att.clone()

        return output, alpha_mask

    def forward(self, x):
        X_c, alpha_mask = self.Channel_Tracer(x)
        X_c = self.bn(X_c)
        x_drop = self.masking(X_c, alpha_mask)
        # pdb.set_trace()
        q = self.spatial_q(x_drop).squeeze(1) # squeeze channel
        k = self.spatial_k(x_drop).squeeze(1)
        v = self.spatial_v(x_drop).squeeze(1)

        # softmax(Q*K^T)
        QK_T = torch.matmul(q, k.transpose(1, 2))
        alpha = F.softmax(QK_T, dim=-1)

        output = torch.matmul(alpha, v).unsqueeze(1) + v.unsqueeze(1)

        return output



class aggregation_edge(nn.Module):
    def __init__(self, channel):
        super(aggregation_edge, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel[2], channel[1], 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel[2], channel[0], 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel[1], channel[0], 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel[2], channel[2], 3, padding=1)
        self.conv_upsample5 = BasicConv2d(channel[2] + channel[1], channel[2] + channel[1], 3, padding=1)

        self.conv_concat2 = BasicConv2d((channel[2] + channel[1]), (channel[2] + channel[1]), 3, padding=1)
        self.conv_concat3 = BasicConv2d((channel[0] + channel[1] + channel[2]),
                                        (channel[0] + channel[1] + channel[2]), 3, padding=1)

        self.UAM = UnionAttentionModule(channel[0] + channel[1] + channel[2])

    def forward(self, e4, e3, e2):
        e4_1 = e4
        e3_1 = self.conv_upsample1(self.upsample(e4)) * e3
        e2_1 = self.conv_upsample2(self.upsample(e4)) * self.conv_upsample3(e3) * e2

        e3_2 = torch.cat((e3_1, self.conv_upsample4(self.upsample(e4_1))), 1)
        e3_2 = self.conv_concat2(e3_2)

        e2_2 = torch.cat((e2_1, self.conv_upsample5(e3_2)), 1)
        x = self.conv_concat3(e2_2)

        output = self.UAM(x)

        return output

class aggregation(nn.Module):
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel[2], channel[1], 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel[2], channel[0], 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel[1], channel[0], 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel[2], channel[2], 3, padding=1)
        self.conv_upsample5 = BasicConv2d(channel[2] + channel[1], channel[2] + channel[1], 3, padding=1)

        self.conv_concat2 = BasicConv2d((channel[2] + channel[1]), (channel[2] + channel[1]), 3, padding=1)
        self.conv_concat3 = BasicConv2d((channel[0] + channel[1] + channel[2]),
                                        (channel[0] + channel[1] + channel[2]), 3, padding=1)

        self.UAM = UnionAttentionModule(channel[0] + channel[1] + channel[2])

    def forward(self, e4, e3, e2):
        e4_1 = e4
        e3_1 = self.conv_upsample1(self.upsample(e4)) * e3
        e2_1 = self.conv_upsample2(self.upsample(self.upsample(e4))) \
               * self.conv_upsample3(self.upsample(e3)) * e2

        e3_2 = torch.cat((e3_1, self.conv_upsample4(self.upsample(e4_1))), 1)
        e3_2 = self.conv_concat2(e3_2)

        e2_2 = torch.cat((e2_1, self.conv_upsample5(self.upsample(e3_2))), 1)
        x = self.conv_concat3(e2_2)

        output = self.UAM(x)

        return output

class GRA(nn.Module):
    def __init__(self, channel, subchannel):
        super(GRA, self).__init__()
        self.group = channel//subchannel
        self.conv = nn.Sequential(
            nn.Conv2d(channel + self.group, channel, 3, padding=1), nn.ReLU(True),
        )
        self.score = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x, y):
        # pdb.set_trace()
        if self.group == 1:
            x_cat = torch.cat((x, y), 1)
        elif self.group == 2:
            xs = torch.chunk(x, 2, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y), 1)
        elif self.group == 4:
            xs = torch.chunk(x, 4, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y), 1)
        elif self.group == 8:
            xs = torch.chunk(x, 8, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y), 1)
        elif self.group == 16:
            xs = torch.chunk(x, 16, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y), 1)
        elif self.group == 32:
            xs = torch.chunk(x, 32, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y,
            xs[16], y, xs[17], y, xs[18], y, xs[19], y, xs[20], y, xs[21], y, xs[22], y, xs[23], y,
            xs[24], y, xs[25], y, xs[26], y, xs[27], y, xs[28], y, xs[29], y, xs[30], y, xs[31], y), 1)
        elif self.group == 64:
            xs = torch.chunk(x, 64, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y,
            xs[16], y, xs[17], y, xs[18], y, xs[19], y, xs[20], y, xs[21], y, xs[22], y, xs[23], y,
            xs[24], y, xs[25], y, xs[26], y, xs[27], y, xs[28], y, xs[29], y, xs[30], y, xs[31], y,
            xs[32], y, xs[33], y, xs[34], y, xs[35], y, xs[36], y, xs[37], y, xs[38], y, xs[39], y,
            xs[40], y, xs[41], y, xs[42], y, xs[43], y, xs[44], y, xs[45], y, xs[46], y, xs[47], y,
            xs[48], y, xs[49], y, xs[50], y, xs[51], y, xs[52], y, xs[53], y, xs[54], y, xs[55], y,
            xs[56], y, xs[57], y, xs[58], y, xs[59], y, xs[60], y, xs[61], y, xs[62], y, xs[63], y), 1)
        else:
            raise Exception("Invalid Channel")

        x = x + self.conv(x_cat)
        y = y + self.score(x)

        return x, y

class ObjectAttention(nn.Module):
    def __init__(self, channel, kernel_size, shallow = False):
        super(ObjectAttention, self).__init__()
        self.channel = channel
        self.DWSConv = DWSConv(channel, channel // 2, kernel=kernel_size, padding=1, kernels_per_layer=1)
        self.DWConv1 = nn.Sequential(
            DWConv(channel // 2, channel // 2, kernel=1, padding=0, dilation=1),
            BasicConv2d(channel // 2, channel // 8, 1),
        )
        self.DWConv2 = nn.Sequential(
            DWConv(channel // 2, channel // 2, kernel=3, padding=1, dilation=1),
            BasicConv2d(channel // 2, channel // 8, 1),
        )
        self.DWConv3 = nn.Sequential(
            DWConv(channel // 2, channel // 2, kernel=3, padding=3, dilation=3),
            BasicConv2d(channel // 2, channel // 8, 1),
        )
        self.DWConv4 = nn.Sequential(
            DWConv(channel // 2, channel // 2, kernel=3, padding=5, dilation=5),
            BasicConv2d(channel // 2, channel // 8, 1),
        )
        self.conv1 = BasicConv2d(channel // 2, 1, 1)
        if shallow == True:
            self.gra1 = GRA(channel, channel)
            self.gra2 = GRA(channel, 32)
            self.gra3 = GRA(channel, 8)
        else:
            self.gra1 = GRA(channel, channel)
            self.gra2 = GRA(channel, 8)
            self.gra3 = GRA(channel, 2)
        

    def forward(self, decoder_map, encoder_map, edge_map):
        """
        Args:
            decoder_map: decoder representation (B, 1, H, W).
            encoder_map: encoder block output (B, C, H, W).
        Returns:
            decoder representation: (B, 1, H, W)
        """
        mask_bg = -1 * torch.sigmoid(decoder_map) + 1  # Sigmoid & Reverse
        # mask_ob = torch.sigmoid(decoder_map)  # object attention
        x1, y1 = self.gra1(encoder_map, mask_bg)
        x2, y2 = self.gra2(x1, y1)
        x3, y4 = self.gra3(x2, y2)

        if edge_map.shape[2]!=decoder_map.shape[2]:
            edge_map = F.interpolate(edge_map, size=(decoder_map.shape[2], decoder_map.shape[3]))
        x4, y4 = self.gra1(x3, edge_map)
        x5, y5 = self.gra2(x4, y4)
        x6, y6 = self.gra3(x5, y5)



        # x = mask_bg.expand(-1, self.channel, -1, -1).mul(encoder_map)
        # edge = mask_bg.clone()
        # edge[edge > opt.denoise] = 0
        # x = x + (edge * encoder_map)

        # x = x + (edge_map * encoder_map)

        # x = self.DWSConv(x)
        # skip = x.clone()
        # x = torch.cat([self.DWConv1(x), self.DWConv2(x), self.DWConv3(x), self.DWConv4(x)], dim=1) + skip
        # x = torch.relu(self.conv1(x))

        return y6 + decoder_map