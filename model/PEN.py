import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
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
        self.rfba = RFB_Block(self.channels[0], channels[0])
        self.rfbb = RFB_Block(self.channels[1], channels[1])
        self.rfbc = RFB_Block(self.channels[2], channels[2])
        self.rfb3 = RFB_Block(self.channels[2], channels[0])    
        self.rfb4 = RFB_Block(self.channels[3], channels[1])
        self.rfb5 = RFB_Block(self.channels[4], channels[2])
        
        # Multi-level aggregation
        self.agg = aggregation(channels)
        self.agg_e = aggregation_edge(channels)
        # Object Attention
        # self.FEM = Frequency_Edge_Module(radius=opt.frequency_radius, channel=self.channels[0])
        self.ObjectAttention5 = ObjectAttention(channel=32, kernel_size=3,shallow=False)
        self.ObjectAttention4 = ObjectAttention(channel=64, kernel_size=3,shallow=False)
        self.ObjectAttention3 = ObjectAttention(channel=128, kernel_size=3,shallow=False)
        self.ObjectAttention2 = ObjectAttention(channel=32, kernel_size=3)
        self.ObjectAttention1 = ObjectAttention(channel=64, kernel_size=3)
        self.ObjectAttention0 = ObjectAttention(channel=128, kernel_size=3)

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

        temp_0 = x0.clone().detach()
        temp_1 = x1.clone().detach()
        temp_2 = x2.clone().detach()
        x0_rfb = self.rfb0(temp_0)  # 32 
        x1_rfb = self.rfb1(temp_1)  # 64
        x2_rfb = self.rfb2(temp_2)  # 128

        xa_rfb = self.rfba(temp_0)  # 32 
        xb_rfb = self.rfbb(temp_1)  # 64
        xc_rfb = self.rfbc(temp_2)  # 128


        edge = self.agg_e(x2_rfb, x1_rfb, x0_rfb)



        # edges = self.edge_model(inputs)
        # # pdb.set_trace()
        # edges = [torch.sigmoid(x) for x in edges]
        # edge = edges[-1]


        #consealed map
        x3_rfb = self.rfb3(x3)  # 32
        x4_rfb = self.rfb4(x4)  # 64
        x5_rfb = self.rfb5(x5)  # 128

        D_0 = self.agg(x5_rfb, x4_rfb, x3_rfb)                           # D_0 (44,44)
        ds_map0 = F.interpolate(D_0, scale_factor=8, mode='bilinear')

        # t_map0 = F.interpolate(D_0, scale_factor=0.25, mode='bilinear')                                               # t_map0 (44,44)
        # D_1 = self.ObjectAttention0(t_map0, x5_rfb, edge)               
        # ds_map1 = F.interpolate(D_1, scale_factor=32, mode='bilinear')    

        # t_map1 = F.interpolate(D_1, scale_factor=2, mode='bilinear')
        # D_2 = self.ObjectAttention1(t_map1, x4_rfb, edge)
        # ds_map2 = F.interpolate(D_2, scale_factor=16, mode='bilinear')    
        
        # t_map2 = F.interpolate(D_2, scale_factor=2, mode='bilinear')
        # D_3 = self.ObjectAttention2(t_map2, x3_rfb, edge)
        # ds_map3 = F.interpolate(D_3, scale_factor=8, mode='bilinear')   

        t_map3 = D_0                                     
        D_4 = self.ObjectAttention3(t_map3, xc_rfb, edge)              
        ds_map4 = F.interpolate(D_4, scale_factor=8, mode='bilinear')    

        t_map4 = F.interpolate(D_4, scale_factor=2, mode='bilinear')
        D_5 = self.ObjectAttention4(t_map4, xb_rfb, edge)
        ds_map5 = F.interpolate(D_5, scale_factor=4, mode='bilinear')
        
        t_map5 = D_5
        D_6 = self.ObjectAttention5(t_map5, xa_rfb, edge)
        ds_map6 = F.interpolate(D_6, scale_factor=4, mode='bilinear')   

        # final_map = (ds_map4 + ds_map3 + ds_map2 + ds_map1 + ds_map0) / 4
        
        edge = F.interpolate(edge, scale_factor=4, mode='bilinear')

        return torch.sigmoid(ds_map6), torch.sigmoid(edge), \
               (torch.sigmoid(ds_map0), \
                #    torch.sigmoid(ds_map1), torch.sigmoid(ds_map2), torch.sigmoid(ds_map3),\
                   torch.sigmoid(ds_map4),torch.sigmoid(ds_map5),torch.sigmoid(ds_map6))