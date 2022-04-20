import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Res2net import res2net50_v1b_26w_4s
from modules.block_modules import RFB_Block, aggregation, aggregation_edge, ObjectAttention
from modules.edge_modules import DexiNed

class PEN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.edge_model = DexiNed()
        self.resnet = res2net50_v1b_26w_4s(pretrained = True)
        self.channels = [64, 256, 512, 1024, 2048]

        # Receptive Field Blocks
        channels = [int(arg_c) for arg_c in opt.RFB_aggregated_channel]
        self.rfb0 = RFB_Block(self.channels[0], channels[0])
        self.rfb1 = RFB_Block(self.channels[1], channels[1])
        self.rfb2 = RFB_Block(self.channels[2], channels[2])
        self.rfb3 = RFB_Block(self.channels[2], channels[0])    
        self.rfb4 = RFB_Block(self.channels[3], channels[1])
        self.rfb5 = RFB_Block(self.channels[4], channels[2])
        
        # Multi-level aggregation
        self.agg = aggregation(channels)
        self.agg_e = aggregation_edge(channels)
        # Object Attention
        # self.FEM = Frequency_Edge_Module(radius=opt.frequency_radius, channel=self.channels[0])
        self.ObjectAttention2 = ObjectAttention(channel=self.channels[2], kernel_size=3)
        self.ObjectAttention1 = ObjectAttention(channel=self.channels[1], kernel_size=3)

    def forward(self, inputs):
        B, C, H, W = inputs.size()

        x = self.resnet.conv1(inputs)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x0 = self.resnet.maxpool(x)      # bs, 64, 88, 88
        x1 = self.resnet.layer1(x0)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)      # bs, 512, 44, 44
        x3 = x2.clone()                  # bs, 512, 44, 44
        x4 = self.resnet.layer3(x3)      # bs, 1024, 22, 22
        x5 = self.resnet.layer4(x4)      # bs, 2048, 11, 11

        #edge
        # x0_rfb = self.rfb0(x0)  # 32 
        # x1_rfb = self.rfb1(x1)  # 64
        # x2_rfb = self.rfb2(x2)  # 128

        # edge = self.agg_e(x2_rfb, x1_rfb, x0_rfb)
        # edge = F.interpolate(edge, scale_factor=4, mode='bilinear')
        edges = self.edge_model(inputs)
        edges = [torch.sigmoid(x) for x in edges]
        edge = edges[-1]


        #consealed map
        x3_rfb = self.rfb3(x3)  # 32
        x4_rfb = self.rfb4(x4)  # 64
        x5_rfb = self.rfb5(x5)  # 128

        D_0 = self.agg(x5_rfb, x4_rfb, x3_rfb)                           # D_0 (44,44)
        ds_map0 = F.interpolate(D_0, scale_factor=8, mode='bilinear')

        D_1 = self.ObjectAttention2(D_0, x3, edge)                             # D_1 (44,44)
        ds_map1 = F.interpolate(D_1, scale_factor=8, mode='bilinear')    
                                                                         
        ds_map = F.interpolate(D_1, scale_factor=2, mode='bilinear')     # ds_map (88,88)

        D_2 = self.ObjectAttention1(ds_map, x1, edge)
        ds_map2 = F.interpolate(D_2, scale_factor=4, mode='bilinear')     

        final_map = (ds_map2 + ds_map1 + ds_map0) / 3


        return torch.sigmoid(final_map), edges, \
               (torch.sigmoid(ds_map0), torch.sigmoid(ds_map1), torch.sigmoid(ds_map2))