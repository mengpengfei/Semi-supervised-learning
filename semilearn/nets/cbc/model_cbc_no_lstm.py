#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from collections import OrderedDict
import math
from semilearn.nets.cbc.BackBoneNet import CBCNet_BACKBONE
from semilearn.cbc_config import cbcconfig as config
import torch.amp as amp
import torch.utils.checkpoint as cp

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class IndicatorLayer(nn.Module):
    def __init__(self,inchannel, reduction=16):
        super(IndicatorLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(inchannel // reduction, 2, bias=False),
            nn.Sigmoid()
        )
    def forward(self, xin):
        b, c, _, _ = xin.size()
        y = self.avg_pool(xin).view(b, c)
        y = self.fc(y).view(b, 2, 1)
        return y[:,0,:],y[:,1,:]

class basic_res_block(nn.Module):
    def __init__(self, nIn=256, nOut=256, stride=1):
        super( basic_res_block, self ).__init__()
        m = OrderedDict()
        m['bn1']=nn.BatchNorm2d(nIn)
        m['LeakyReLU1']=nn.LeakyReLU(inplace=True)
        m['conv1'] = nn.Conv2d(nIn,int(nIn/4),1,padding=0,stride=stride,bias=False)

        m['bn2']=nn.BatchNorm2d(int(nIn/4))
        m['LeakyReLU2']=nn.LeakyReLU(inplace=True)
        m['conv2'] = nn.Conv2d(int(nIn/4),int(nIn/4),3,padding=1,stride=stride,bias=True)  #same卷积padding=dilation*(k-1)/2

        m['bn3']=nn.BatchNorm2d(int(nIn/4))
        m['LeakyReLU3']=nn.LeakyReLU(inplace=True)
        m['conv3'] = nn.Conv2d(int(nIn/4),nOut,1,padding=0,stride=stride,bias=False)
        # m['bn4']=nn.BatchNorm2d(nOut)
        # self.bn4=nn.BatchNorm2d(nIn)
        self.group1 = nn.Sequential( m )

        # self.gap=nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        residual = x
        out = self.group1( x ) + residual
        # ga=self.gap(out)
        return out

class basic_res_block1(nn.Module):

    def __init__(self):
        super( basic_res_block1, self ).__init__()
        m1 = OrderedDict()
        m1["conv1"]=nn.Conv2d(256,128,1,padding=0,stride=1,bias=False)
        m1["bn1"]=nn.BatchNorm2d(128)
        m1["LeakyReLU1"]=nn.LeakyReLU(inplace=True)

        m1["conv2"]=nn.Conv2d(128,128,3,padding=1,stride=2,bias=True) #same卷积padding=dilation*(k-1)/2
        m1["bn2"]=nn.BatchNorm2d(128)
        m1["LeakyReLU2"]=nn.LeakyReLU(inplace=True)

        m1["conv3"]=nn.Conv2d(128,512,1,padding=0,stride=1,bias=False)
        # m1["bn3"]=nn.BatchNorm2d(512)

        self.group11 = nn.Sequential( m1 )

        self.conv=nn.Sequential(
            nn.MaxPool2d(3,padding=1,stride=2),
            nn.Conv2d(256,512,1,padding=0,stride=1,bias=False)
        )

    def forward(self, x):
        residual = x
        res=self.conv(residual)
        group=self.group11(x)
        out=group+res
        # ga=self.gap(out)
        return out

class CBCNet_head(nn.Module):
    def __init__(self,timestamp,nclass):
        super(CBCNet_head,self).__init__()
        self.timestamp = timestamp
        if math.ceil(config.input_w/8.0)!= self.timestamp:
            self.indicatorLayer1=IndicatorLayer(128)
            kernel_size=(math.ceil(config.input_h/8),math.ceil(config.input_w/8/timestamp))
            stride = (1,math.ceil(config.input_w/8/timestamp))
            timestamp=math.ceil(config.input_w/8.0/kernel_size[1])
            pooling = (0, math.floor((kernel_size[1]*timestamp-math.ceil(config.input_w/8)+1)/2))

            self.indicator1=nn.Sequential(
                nn.MaxPool2d(kernel_size=kernel_size,stride=stride
                             ,padding=pooling)
            )
            self.linear00=nn.Linear(math.ceil(config.input_h/8.0),1)
            self.linear0=nn.Linear(math.ceil(config.input_w/8.0),timestamp)
            self.linear=nn.Linear(128,nclass)
        else:
            self.linear=nn.Linear(128*math.ceil(config.input_h/8),nclass)

        self._init_params()
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x10):
        b1,c1,h1,w1=x10.size()
        if w1 != self.timestamp:
            ind0,ind1=self.indicatorLayer1(x10)
            #w b c
            # x10_1=self.indicator(x10)
            x10_2=self.indicator1(x10)
            b0,c0,h0,w0=x10_2.size()
            # x10_2=(x10_2).squeeze(2)

            # b,c,h,w=x10.size()
            x100=x10.permute(0,1,3,2)
            x100=self.linear00(x100)
            b100,c100,w100,h100=x100.size() #h100=1
            x10_3 = x100.reshape([b100,c100*h100,w100]) #b c w
            x10_3=self.linear0(x10_3) #b c w1
            ind00=ind0.unsqueeze(1).expand_as(x10_3)
            conv = x10_3*ind00 # w,b,c*h
            conv=conv.permute(2, 0, 1)
            conv1=x10_2.reshape([b0,c0*h0,w0])
            ind11=ind1.unsqueeze(1).expand_as(conv1)
            conv1=conv1*ind11
            # conv1=conv1.permute(0, 2, 1)
            # conv1=conv1.permute(0, 2, 1)
            conv1=conv1.permute(2, 0, 1)
            conv=conv+conv1  #W,B,512
        else:
            # x100=x10.permute(0,1,3,2)
            # x100=self.linear00(x100)
            # b100,c100,w100,h100=x100.size() #h100=1
            # x10_3 = x100.reshape([b100,c100*h100,w100]) #b c w
            # x10_3=self.linear0(x10_3) #b c w1
            # conv=x10_3.permute(2, 0, 1)
            x10_3 = x10.reshape([b1,c1*h1,w1]) #b c w
            # x10_3=self.linear0(x10_3) #b c w1
            conv=x10_3.permute(2, 0, 1) #w b c*h
            # conv=x10_3.permute(0, 2, 1)
            # conv=self.linear(conv)
        output = self.linear(conv)
        return output

class CBCNet(nn.Module):
    def __init__(self,num_classes=None,pretrained=True,pretrained_path=None,timestamp=config.timestamp,nclass=config.nclass,dropout_ratio=config.dropout_ratio):
        super(CBCNet,self).__init__()
        self.timestamp=timestamp

        self.features=CBCNet_BACKBONE(pretrained=pretrained)

        # self.dropout =

        self.cbcnet_head=nn.Sequential(
            nn.Dropout(p=dropout_ratio),
            CBCNet_head(config.timestamp,nclass)
        )
    def forward(self, x):
        if config.fp16:
            with amp.autocast(config.auto_cast_device, dtype=torch.float16, cache_enabled=False):
                if config.check_points:
                    # x9=self.features(x)
                    x9 = cp.checkpoint(self.features, x, use_reentrant=False)
                    # x10=self.dropout(x9)
                    output = cp.checkpoint(self.cbcnet_head,x9, use_reentrant=False)
                else:
                    x9=self.features(x)
                    output=self.cbcnet_head(x9)
            result_dict = {'logits':output.permute(1,0,2)[:,-1,:], 'feat':x9}
            return result_dict
            # return output.permute(1,0,2)
        else:
            if config.check_points:
                x9 = cp.checkpoint(self.features, x, use_reentrant=False)
                output = cp.checkpoint(self.cbcnet_head,x9, use_reentrant=False)
            else:
                x9=self.features(x)
                output=self.cbcnet_head(x9) #w,b,cnum
            result_dict = {'logits':output.permute(1,0,2)[:,-1,:], 'feat':x9}
            return result_dict
            # return output.permute(1,0,2)

    def group_matcher(self, coarse=False, prefix=''):
        if coarse:
            matcher = dict(
                stem=r'^{}features\.stg1'.format(prefix),
                blocks=r'^{}features\.stg2|^{}features\.stg3|^{}features\.stg4|^{}features\.stg5|^{}features\.stg6|^{}features\.stg7|^{}features\.stg8|^{}features\.stg9'.format(
                    prefix, prefix, prefix, prefix, prefix, prefix, prefix, prefix),
                head=r'^{}cbcnet_head'.format(prefix)
            )
        else:
            matcher = dict(
                stem=r'^{}features\.stg1'.format(prefix),
                blocks=r'^{}features\.stg2_1|^{}features\.stg2_2|^{}features\.stg3|^{}features\.stg4|^{}features\.stg4_1|^{}features\.stg5|^{}features\.stg6|^{}features\.stg7|^{}features\.stg8|^{}features\.stg9'.format(
                    prefix, prefix, prefix, prefix, prefix, prefix, prefix, prefix, prefix, prefix),
                head=r'^{}cbcnet_head'.format(prefix)
            )
        return matcher

    def no_weight_decay(self):
        nwd = []
        for n, _ in self.named_parameters():
            if 'bn' in n or 'bias' in n:
                nwd.append(n)
        return nwd

if __name__ == '__main__':
    # from torchvision import models
    # m=models.mobilenetv3.mobilenet_v3_large(pretrained=True)
    # print(m)
    input=torch.rand([6,3,32,32])
    b, c, h, w = input.size()
    net=CBCNet(timestamp=math.ceil(w/8.0),nclass=37,dropout_ratio=0)
    #
    # print(net)
    # net.eval()

    from torchvision.models.resnet import resnet18
    # from torchvision.models.mobilenet import MobileNetV3,mobilenet_v3_small
    # # # from torchvision.models.vgg import mobilenet_v3_large
    # # from torchvision.models.segmentation import lraspp_mobilenet_v3_large
    # from thop import profile
    # mobilenetv3=MobileNetV3(pretrained=True)
    # print(mobilenetv3)
    # mobilenetv3.features[4].block[1]=nn.Sequential(
    #     nn.Conv2d(72, 72, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=72, bias=False),
    #     nn.BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
    #     nn.ReLU(inplace=True)
    # )
    # mobilenetv3.features[13].block[1]=nn.Sequential(
    #     nn.Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False),
    #     nn.BatchNorm2d(672, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
    #     nn.Hardswish()
    # )
    # print(mobilenetv3)
    # net=resnet18(num_classes=1000)
    print(net.cbcnet_head.parameters())
    net.eval();
    out=net(input)
    print(out)
    # # net=lraspp_mobilenet_v3_large(pretrained=False, progress=True, num_classes=2)
    # # net.eval()
    # out=net(input)
    # print(out.size())
    # flops, params = profile(net, inputs=(input,))
    # print('FLOPs = ' + str(flops/1000**3) + 'G')
    # print('Params = ' + str(params/1000**2) + 'M')
    # # torch.save(net.state_dict(),'./abcd.pth')
    # # torch.onnx.export(net, input,'./model.onnx',export_params=True,verbose=False)
    # # output=net.forward(input)
    # # print(output.size())
