from collections import OrderedDict

import torch
import torch.nn as nn
import os
import torchvision.models as models

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

class CBCNet_BACKBONE(nn.Module):
    def __init__(self,pretrained=True):
        super(CBCNet_BACKBONE,self).__init__()
        self.stg1=nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3,32,3,padding=1,stride=1),#same卷积padding=dilation*(k-1)/2
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32,64,3,padding=2,stride=2,dilation=2),
            # nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1,stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            nn.Conv2d(64,64,1,padding=0,stride=1,bias=False),
            # StemBlock(),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        # stg2
        # stg2_1
        self.stg2_1=nn.Sequential(
            nn.Conv2d(64,64,3,padding=1,stride=1,bias=True),#same卷积padding=dilation*(k-1)/2
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64,128,3,padding=2,stride=2,bias=True,dilation=2),
            # nn.Conv2d(64,128,3,padding=1,stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128,256,1,padding=0,stride=1,bias=False),
            # nn.BatchNorm2d(256)
        )
        # stg2_2
        self.stg2_2=nn.Sequential(
            # nn.Conv2d(64,128,3,padding=1,stride=2),
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(3,padding=1,stride=2),
            nn.Conv2d(64,256,1,padding=0,stride=1,bias=False),
            # nn.BatchNorm2d(256)
            # nn.LeakyReLU(inplace=True)
        )

        # stg3
        self.stg3=nn.Sequential(
            basic_res_block(256,256,1),
            basic_res_block(256,256,1),
            basic_res_block(256,256,1)
        )
        #stg4
        self.stg4=basic_res_block(256,256,1)

        self.stg4_1=nn.Sequential(
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True))

        # stg5
        self.stg5=basic_res_block1()

        #stg6
        self.stg6=basic_res_block(512,512,1)

        #stg7
        self.stg7=basic_res_block(512,512,1)

        #stg8
        self.stg8=nn.Sequential(
            basic_res_block(512,512,1),
            basic_res_block(512,512,1),
            basic_res_block(512,512,1)
        )

        self.conv1x1=nn.Conv2d(512,128,1,1)

        # self.gap=nn.AdaptiveAvgPool2d(1)
        self.stg9=nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True))
        # self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.fc=nn.Linear(128,nclass)
        self._init_params(pretrained)

    def _init_params(self, pretrained):
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
        if pretrained:
            cur_path = os.path.abspath(os.path.dirname(__file__))
            pretrain_weights=os.path.join(cur_path,"cbcnet_pretrain230315.pth")
            if not os.path.exists(pretrain_weights):
                pretrain_weights=os.path.join(cur_path,"model","cbcnet_pretrain230315.pth")
            pretrained_dict = torch.load(pretrain_weights,
                                         map_location={'cuda:0': 'cpu'})
            print('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            # pretrained_dict = {k.replace('last_layer',
            #                              'aux_head').replace('model.', ''): v
            #                    for k, v in pretrained_dict.items()}
            #print(set(model_dict) - set(pretrained_dict))
            #print(set(pretrained_dict) - set(model_dict))
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def group_matcher(self, coarse=False, prefix=''):
        if coarse:
            matcher = dict(
                stem=r'^{}stg1'.format(prefix),
                blocks=r'^{}stg2|^{}stg3|^{}stg4|^{}stg5|^{}stg6|^{}stg7|^{}stg8|^{}stg9'.format(
                    prefix, prefix, prefix, prefix, prefix, prefix, prefix, prefix),
            )
        else:
            matcher = dict(
                stem=r'^{}stg1'.format(prefix),
                blocks=r'^{}stg2_1|^{}stg2_2|^{}stg3|^{}stg4|^{}stg4_1|^{}stg5|^{}stg6|^{}stg7|^{}stg8|^{}stg9'.format(
                    prefix, prefix, prefix, prefix, prefix, prefix, prefix, prefix, prefix, prefix),
            )
        return matcher

    def no_weight_decay(self):
        nwd = []
        for n, _ in self.named_parameters():
            if 'bn' in n or 'bias' in n:
                nwd.append(n)
        return nwd

    def forward(self, x):
        x1=self.stg1(x)
        x2_1=self.stg2_1(x1)
        x2_2=self.stg2_2(x1)
        x2=x2_1+x2_2
        x3=self.stg3(x2)
        x4=self.stg4(x3)
        x4_1=self.stg4_1(x4)
        x5=self.stg5(x4_1)
        x6=self.stg6(x5)
        x7=self.stg7(x6)
        x8=self.stg8(x7)
        x8=self.conv1x1(x8)
        # x8=self.gap(x8)*x8
        x9=self.stg9(x8)
        # x10 = self.avg_pool(x9)
        # x11 = torch.flatten(x10, 1)
        # x11 = self.fc(x11)
        return x9

class AlexNet_DG(nn.Module):
    def __init__(self, pretrained=True):
        super(AlexNet_DG,self).__init__()
        alexnet = models.alexnet(weights=None)
        # alexnet = models.alexnet(pretrained=False)
        alexnet.features[1]=nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        alexnet.features[4]=nn.Sequential(
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        alexnet.features[7]=nn.Sequential(
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        alexnet.features[9]=nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        alexnet.features[11]=nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        del alexnet.features[12]
        del alexnet.features[2]
        self.features=alexnet.features
        # self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.fc=nn.Linear(256,nclass)
        self._init_params(pretrained)
    def _init_params(self, pretrained):
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
        if pretrained:
            cur_path = os.path.abspath(os.path.dirname(__file__))
            pretrain_weights=os.path.join(cur_path,"alexnet_pretrain230317.pth")
            if not os.path.exists(pretrain_weights):
                pretrain_weights=os.path.join(cur_path,"model","alexnet_pretrain230317.pth")
            pretrained_dict = torch.load(pretrain_weights,
                                         map_location={'cuda:0': 'cpu'})
            print('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            # pretrained_dict = {k.replace('last_layer',
            #                              'aux_head').replace('model.', ''): v
            #                    for k, v in pretrained_dict.items()}
            #print(set(model_dict) - set(pretrained_dict))
            #print(set(pretrained_dict) - set(model_dict))
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def forward(self,x):
        x = self.features(x)
        # x = self.avg_pool(x)
        # x=torch.flatten(x, start_dim=1)
        # x=self.fc(x)
        return x
if __name__=='__main__':
    input = torch.ones((1,3,60,60))
    alexnet_dg=AlexNet_DG(pretrained=False)
    print(alexnet_dg(input).shape)
    # import os
    # cur_path = os.path.abspath(os.path.dirname(__file__))
    # pretrain_weights=os.path.join(cur_path,"cbcnet_pretrain230315.pth")
    # print(os.path.exists(pretrain_weights))
