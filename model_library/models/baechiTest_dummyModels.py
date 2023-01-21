import torch
import torch.nn as nn
from typing import Any

import torch.nn.functional as F

class _concatenateLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *x):
        return torch.cat(x, 1)
    
class _squeezeLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.squeeze()

class _addLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2

class _avgPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

class _avgPool53(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=5, stride=3)

class _maxPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.max_pool2d(x, kernel_size=3, stride=2)

class _adaptiveAvgPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1))


class _flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.flatten(x, 1)


class ParallelTwoLayer(nn.Module):

    def __init__(self, factor, repetable =0) -> None:
        super().__init__() # python 3 syntax
        
        self.factor = factor
        self.linear1N = int(512*self.factor)
        self.linear2N = int(2048*self.factor)
        self.linear3N = int(512*self.factor)
        self.linear4N = 2*self.linear3N
        self.linear5N = int(512*self.factor)


        self.squeeze = _squeezeLayer()
        self.fc1 = nn.Linear(self.linear1N, self.linear2N)
        self.fc2a1 = nn.Linear(self.linear2N, self.linear3N)
        self.fc2a2 = nn.Linear(self.linear3N, self.linear3N)
        self.fc2b1 = nn.Linear(self.linear2N, self.linear3N)
        self.fc2b2 = nn.Linear(self.linear3N, self.linear3N)
        self.concatenate = _concatenateLayer()
        self.fc3 = nn.Linear(self.linear4N, self.linear5N)
        #self.add1 = _addLayer()
        self.fc4 = nn.Linear(self.linear5N, self.linear5N)

        if repetable:
            torch.nn.init.constant_(self.fc1.weight, 1/512); torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.constant_(self.fc2a1.weight, 1/512); torch.nn.init.zeros_(self.fc2a1.bias)
            torch.nn.init.constant_(self.fc2a2.weight, 1/512); torch.nn.init.zeros_(self.fc2a2.bias)
            torch.nn.init.constant_(self.fc2b1.weight, 1/512); torch.nn.init.zeros_(self.fc2b1.bias)
            torch.nn.init.constant_(self.fc2b2.weight, 1/512); torch.nn.init.zeros_(self.fc2b2.bias)
            torch.nn.init.constant_(self.fc3.weight, 1/512); torch.nn.init.zeros_(self.fc3.bias)
            torch.nn.init.constant_(self.fc4.weight, 1/512); torch.nn.init.zeros_(self.fc4.bias)
        

    def forward(self, x):
        x = self.squeeze(x)
        x = self.fc1(x)
        xa1 = self.fc2a1(x)
        xa2 = self.fc2a2(xa1)
        xb1 = self.fc2b1(x)
        xb2 = self.fc2b2(xb1)
        y = self.concatenate(xa2,xb2)
        y = self.fc3(y)
        #y = self.add1(y,xb2)
        y = self.fc4(y)
        return y


class ParallelThreeLayer(nn.Module):

    def __init__(self, factor, repetable =0) -> None:
        super().__init__() # python 3 syntax

        self.parallels = 2
        
        self.factor = factor
        self.linear1N = int(512*self.factor)
        self.linear2N = int(2048*self.factor)
        self.linear3N = int(512*self.factor)
        #self.linear3N = int(1024*self.factor)
        self.linear4N = self.parallels*self.linear3N
        self.linear5N = int(512*self.factor)


        self.squeeze = _squeezeLayer()
        self.fc1 = nn.Linear(self.linear1N, self.linear2N)
        self.fc2a1 = nn.Linear(self.linear2N, self.linear3N)
        self.fc2a2 = nn.Linear(self.linear3N, self.linear3N)
        self.fc2b1 = nn.Linear(self.linear2N, self.linear3N)
        self.fc2b2 = nn.Linear(self.linear3N, self.linear3N)
        if self.parallels==3:
            self.fc2c1 = nn.Linear(self.linear2N, self.linear3N)
            self.fc2c2 = nn.Linear(self.linear3N, self.linear3N)
        self.concatenate = _concatenateLayer()
        self.fc3 = nn.Linear(self.linear4N, self.linear5N)
        #self.add1 = _addLayer()
        self.fc4 = nn.Linear(self.linear5N, self.linear5N)

        if repetable:
            torch.nn.init.constant_(self.fc1.weight, 1/512); torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.constant_(self.fc2a1.weight, 1/512); torch.nn.init.zeros_(self.fc2a1.bias)
            torch.nn.init.constant_(self.fc2a2.weight, 1/512); torch.nn.init.zeros_(self.fc2a2.bias)
            torch.nn.init.constant_(self.fc2b1.weight, 1/512); torch.nn.init.zeros_(self.fc2b1.bias)
            torch.nn.init.constant_(self.fc2b2.weight, 1/512); torch.nn.init.zeros_(self.fc2b2.bias)
            if self.parallels==3:
                torch.nn.init.constant_(self.fc2c1.weight, 1/512); torch.nn.init.zeros_(self.fc2c1.bias)
                torch.nn.init.constant_(self.fc2c2.weight, 1/512); torch.nn.init.zeros_(self.fc2c2.bias)
            torch.nn.init.constant_(self.fc3.weight, 1/512); torch.nn.init.zeros_(self.fc3.bias)
            torch.nn.init.constant_(self.fc4.weight, 1/512); torch.nn.init.zeros_(self.fc4.bias)
        

    def forward(self, x):
        x = self.squeeze(x)
        x = self.fc1(x)
        xa1 = self.fc2a1(x)
        xa2 = self.fc2a2(xa1)
        xb1 = self.fc2b1(x)
        xb2 = self.fc2b2(xb1)
        if self.parallels==3:
            xc1 = self.fc2c1(x)
            xc2 = self.fc2c2(xc1)
            y = self.concatenate(xa2,xb2,xc2)
        else:
            y = self.concatenate(xa2,xb2)
        y = self.fc3(y)
        #y = self.add1(y,xb2)
        y = self.fc4(y)
        return y

class ParallelThreeLayerOld(nn.Module):

    def __init__(self, factor: int = 1) -> None:
        super().__init__() # python 3 syntax
        
        self.factor = factor
        self.linear1N = 512*self.factor
        self.linear2N = 2048*self.factor
        self.linear3N = 1024*self.factor
        self.linear4N = 3*self.linear3N
        self.linear5N = 512*self.factor


        self.squeeze = _squeezeLayer()
        
        self.fc1   = nn.Linear(self.linear1N, self.linear2N)
        torch.nn.init.constant_(self.fc1.weight, 1/512)
        torch.nn.init.zeros_(self.fc1.bias)
        
        self.fc2a1 = nn.Linear(self.linear2N, self.linear3N)
        torch.nn.init.constant_(self.fc2a1.weight, 1/512)
        torch.nn.init.zeros_(self.fc2a1.bias)
        
        self.fc2a2 = nn.Linear(self.linear3N, self.linear3N)
        torch.nn.init.constant_(self.fc2a2.weight, 1/512)
        torch.nn.init.zeros_(self.fc2a2.bias)
        
        self.fc2b1 = nn.Linear(self.linear2N, self.linear3N)
        torch.nn.init.constant_(self.fc2b1.weight, 1/512)
        torch.nn.init.zeros_(self.fc2b1.bias)
        
        self.fc2b2 = nn.Linear(self.linear3N, self.linear3N)
        torch.nn.init.constant_(self.fc2b2.weight, 1/512)
        torch.nn.init.zeros_(self.fc2b2.bias)
        
        self.fc2c1 = nn.Linear(self.linear2N, self.linear3N)
        torch.nn.init.constant_(self.fc2c1.weight, 1/512)
        torch.nn.init.zeros_(self.fc2c1.bias)
        
        self.fc2c2 = nn.Linear(self.linear3N, self.linear3N)
        torch.nn.init.constant_(self.fc2c2.weight, 1/512)
        torch.nn.init.zeros_(self.fc2c2.bias)
        
        self.concatenate = _concatenateLayer()
        
        self.fc3   = nn.Linear(self.linear4N, self.linear5N)
        torch.nn.init.constant_(self.fc3.weight, 1/512)
        torch.nn.init.zeros_(self.fc3.bias)
        
        
        self.fc4   = nn.Linear(self.linear5N, self.linear5N)
        torch.nn.init.constant_(self.fc4.weight, 1/512)
        torch.nn.init.zeros_(self.fc4.bias)
        

    def forward(self, x):
        x = self.squeeze(x)
        x = self.fc1(x)
        xa1 = self.fc2a1(x)
        xa2 = self.fc2a2(xa1)
        xb1 = self.fc2b1(x)
        xb2 = self.fc2b2(xb1)
        xc1 = self.fc2c1(x)
        xc2 = self.fc2c2(xc1)
        y = self.concatenate(xa2,xb2,xc2)
        y = self.fc3(y)
        y1 = self.fc4(y)
        return y1


class TallParallelModel(nn.Module):

    def __init__(self, factor, repetable =0) -> None:
        super().__init__() # python 3 syntax
        
        self.factor = factor
        self.linear1N = int(512*self.factor)
        self.linear2N = int(2048*self.factor)
        self.linear3N = int(512*self.factor)
        self.linear4N = 2*self.linear3N
        self.linear5N = int(512*self.factor)

        self.squeeze = _squeezeLayer()
        self.fc1 = nn.Linear(self.linear1N, self.linear2N)
        self.fc2a = nn.Linear(self.linear2N, self.linear3N)
        self.fc2b = nn.Linear(self.linear2N, self.linear3N)
        self.concatenate = _concatenateLayer()
        self.fc3 = nn.Linear(self.linear4N, self.linear5N)
        self.fc4 = nn.Linear(self.linear5N, self.linear5N)
        self.fc5 = nn.Linear(self.linear5N, self.linear5N)
        self.fc6 = nn.Linear(self.linear5N, self.linear5N)
        self.fc7 = nn.Linear(self.linear5N, self.linear5N)
        self.fc8 = nn.Linear(self.linear5N, self.linear5N)
        self.fc9 = nn.Linear(self.linear5N, self.linear5N)
        self.fc10 = nn.Linear(self.linear5N, self.linear5N)
        self.fc11 = nn.Linear(self.linear5N, self.linear5N)
        self.fc12 = nn.Linear(self.linear5N, self.linear5N)
        self.fc13 = nn.Linear(self.linear5N, self.linear5N)
        self.fc14 = nn.Linear(self.linear5N, self.linear5N)
        self.fc15 = nn.Linear(self.linear5N, self.linear5N)
        self.fc16 = nn.Linear(self.linear5N, self.linear5N)
        self.fc17 = nn.Linear(self.linear5N, self.linear5N)
        self.fc18 = nn.Linear(self.linear5N, self.linear5N)
        self.fc19 = nn.Linear(self.linear5N, self.linear5N)
        self.fc20 = nn.Linear(self.linear5N, self.linear5N)
        self.fc21 = nn.Linear(self.linear5N, self.linear5N)
        self.fc22 = nn.Linear(self.linear5N, self.linear5N)
        self.fc23 = nn.Linear(self.linear5N, self.linear5N)
        self.fc24 = nn.Linear(self.linear5N, self.linear5N)
        self.fc25 = nn.Linear(self.linear5N, self.linear5N)
        self.fc26 = nn.Linear(self.linear5N, self.linear5N)
        self.fc27 = nn.Linear(self.linear5N, self.linear5N)
        self.fc28 = nn.Linear(self.linear5N, self.linear5N)
        self.fc29 = nn.Linear(self.linear5N, self.linear5N)

        if repetable:
            torch.nn.init.constant_(self.fc1.weight, 1/512); torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.constant_(self.fc2a.weight, 1/512); torch.nn.init.zeros_(self.fc2a.bias)
            torch.nn.init.constant_(self.fc2b.weight, 1/512); torch.nn.init.zeros_(self.fc2b.bias)
            torch.nn.init.constant_(self.fc3.weight, 1/512); torch.nn.init.zeros_(self.fc3.bias)
            torch.nn.init.constant_(self.fc4.weight, 1/512); torch.nn.init.zeros_(self.fc4.bias)
            torch.nn.init.constant_(self.fc5.weight, 1/512); torch.nn.init.zeros_(self.fc5.bias)
            torch.nn.init.constant_(self.fc6.weight, 1/512); torch.nn.init.zeros_(self.fc6.bias)
            torch.nn.init.constant_(self.fc7.weight, 1/512); torch.nn.init.zeros_(self.fc7.bias)
            torch.nn.init.constant_(self.fc8.weight, 1/512); torch.nn.init.zeros_(self.fc8.bias)
            torch.nn.init.constant_(self.fc9.weight, 1/512); torch.nn.init.zeros_(self.fc9.bias)
            torch.nn.init.constant_(self.fc10.weight, 1/512); torch.nn.init.zeros_(self.fc10.bias)
            torch.nn.init.constant_(self.fc11.weight, 1/512); torch.nn.init.zeros_(self.fc11.bias)
            torch.nn.init.constant_(self.fc12.weight, 1/512); torch.nn.init.zeros_(self.fc12.bias)
            torch.nn.init.constant_(self.fc13.weight, 1/512); torch.nn.init.zeros_(self.fc13.bias)
            torch.nn.init.constant_(self.fc14.weight, 1/512); torch.nn.init.zeros_(self.fc14.bias)
            torch.nn.init.constant_(self.fc15.weight, 1/512); torch.nn.init.zeros_(self.fc15.bias)
            torch.nn.init.constant_(self.fc16.weight, 1/512); torch.nn.init.zeros_(self.fc16.bias)
            torch.nn.init.constant_(self.fc17.weight, 1/512); torch.nn.init.zeros_(self.fc17.bias)
            torch.nn.init.constant_(self.fc18.weight, 1/512); torch.nn.init.zeros_(self.fc18.bias)
            torch.nn.init.constant_(self.fc19.weight, 1/512); torch.nn.init.zeros_(self.fc19.bias)
            torch.nn.init.constant_(self.fc20.weight, 1/512); torch.nn.init.zeros_(self.fc20.bias)
            torch.nn.init.constant_(self.fc21.weight, 1/512); torch.nn.init.zeros_(self.fc21.bias)
            torch.nn.init.constant_(self.fc22.weight, 1/512); torch.nn.init.zeros_(self.fc22.bias)
            torch.nn.init.constant_(self.fc23.weight, 1/512); torch.nn.init.zeros_(self.fc23.bias)
            torch.nn.init.constant_(self.fc24.weight, 1/512); torch.nn.init.zeros_(self.fc24.bias)
            torch.nn.init.constant_(self.fc25.weight, 1/512); torch.nn.init.zeros_(self.fc25.bias)
            torch.nn.init.constant_(self.fc26.weight, 1/512); torch.nn.init.zeros_(self.fc26.bias)
            torch.nn.init.constant_(self.fc27.weight, 1/512); torch.nn.init.zeros_(self.fc27.bias)
            torch.nn.init.constant_(self.fc28.weight, 1/512); torch.nn.init.zeros_(self.fc28.bias)
            torch.nn.init.constant_(self.fc29.weight, 1/512); torch.nn.init.zeros_(self.fc29.bias)

          

    def forward(self, x):
        x = self.squeeze(x)
        x = self.fc1(x)
        xb = self.fc2b(x)
        xa = self.fc2a(x)
        y = self.concatenate(xa,xb)
        y = self.fc3(y)
        y = self.fc4(y)
        y = self.fc5(y)
        y = self.fc6(y)
        y = self.fc7(y)
        y = self.fc8(y)
        y = self.fc9(y)
        y = self.fc10(y)
        y = self.fc11(y)
        y = self.fc12(y)
        y = self.fc13(y)
        y = self.fc14(y)
        y = self.fc15(y)
        y = self.fc16(y)
        y = self.fc17(y)
        y = self.fc18(y)
        y = self.fc19(y)
        y = self.fc20(y)
        y = self.fc21(y)
        y = self.fc22(y)
        y = self.fc23(y)
        y = self.fc24(y)
        y = self.fc25(y)
        y = self.fc26(y)
        y = self.fc27(y)
        y = self.fc28(y)
        y = self.fc29(y)
        
        return y

    #    self.linear1N = int(512*self.factor)
    #     self.linear2N = int(7*512*self.factor)
    #     self.linear3N = int(6*512*self.factor)
    #     self.linear4N = int(8*512*self.factor)
    #     self.linear5N = int(512*self.factor)

class OneLayer(nn.Module):

    def __init__(self, factor, repetable =0) -> None:
        super().__init__() # python 3 syntax
        
        self.factor = factor
        self.linear1N = int(512*self.factor)
 

        self.squeeze = _squeezeLayer()
        self.fc1 = nn.Linear(self.linear1N, self.linear1N)
        self.fc2 = nn.Linear(self.linear1N, self.linear1N)

        #if repetable:
        #    torch.nn.init.constant_(self.fc1.weight, 1/512); torch.nn.init.zeros_(self.fc1.bias)
        #    torch.nn.init.constant_(self.fc2.weight, 1/512); torch.nn.init.zeros_(self.fc2.bias)
        #    torch.nn.init.constant_(self.fc3.weight, 1/512); torch.nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ShortLinearModel(nn.Module):

    def __init__(self, factor, repetable =0) -> None:
        super().__init__() # python 3 syntax
        self.n1 = 7
        self.n2 = 2
        self.n3 = 8 
        
        self.factor = factor
        self.linear1N = int(512*self.factor)
        self.linear2N = int(self.n1*512*self.factor)
        self.linear3N = int(self.n2*512*self.factor)
        self.linear4N = int(self.n3*512*self.factor)
        self.linear5N = int(512*self.factor)

        self.squeeze = _squeezeLayer()
        self.fc1 = nn.Linear(self.linear1N, self.linear2N)
        self.fc2 = nn.Linear(self.linear2N, self.linear3N)
        self.fc3 = nn.Linear(self.linear3N, self.linear4N)
        self.fc4 = nn.Linear(self.linear4N, self.linear5N)
        #self.fc5 = nn.Linear(self.linear4N, self.linear4N)
        #self.fc6 = nn.Linear(self.linear4N, self.linear4N)


        #if repetable:
        #    torch.nn.init.constant_(self.fc1.weight, 1/512); torch.nn.init.zeros_(self.fc1.bias)
        #    torch.nn.init.constant_(self.fc2.weight, 1/512); torch.nn.init.zeros_(self.fc2.bias)
        #    torch.nn.init.constant_(self.fc3.weight, 1/512); torch.nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x) 
        x = self.fc4(x)
        # x = self.fc5(x) 
        # x = self.fc6(x) 
        return x

class LinearModel(nn.Module):

    def __init__(self, factor, repetable =0) -> None:
        super().__init__() # python 3 syntax
        self.n1 = 2
        self.n2 = 9
        self.n3 = 3 
        
        self.factor = factor
        self.linear1N = int(512*self.factor)
        self.linear2N = int(self.n1*512*self.factor)
        self.linear3N = int(self.n2*512*self.factor)
        self.linear4N = int(self.n3*512*self.factor)
        self.linear5N = int(512*self.factor)

        self.squeeze = _squeezeLayer()
        self.fc1 = nn.Linear(self.linear1N, self.linear2N)
        self.fc2 = nn.Linear(self.linear2N, self.linear3N)
        self.fc3 = nn.Linear(self.linear3N, self.linear4N)
        self.fc4 = nn.Linear(self.linear4N, self.linear5N)
        self.fc5 = nn.Linear(self.linear5N, self.linear5N)
        self.fc6 = nn.Linear(self.linear5N, self.linear5N)
        self.fc7 = nn.Linear(self.linear5N, self.linear5N)
        self.fc8 = nn.Linear(self.linear5N, self.linear5N)
        self.fc9 = nn.Linear(self.linear5N, self.linear5N)
        #self.fc6 = nn.Linear(self.linear4N, self.linear4N)


        #if repetable:
        #    torch.nn.init.constant_(self.fc1.weight, 1/512); torch.nn.init.zeros_(self.fc1.bias)
        #    torch.nn.init.constant_(self.fc2.weight, 1/512); torch.nn.init.zeros_(self.fc2.bias)
        #    torch.nn.init.constant_(self.fc3.weight, 1/512); torch.nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x) 
        x = self.fc4(x)
        x = self.fc5(x) 
        x = self.fc6(x) 
        x = self.fc7(x) 
        x = self.fc8(x) 
        x = self.fc9(x) 
        return x

class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        repetable=1,
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

        if repetable:
            #p,q,r,s=self.conv.weight.size()
            #w = torch.FloatTensor([[[[(j-k-l+m)/10 for j in range(s)] for k in range(r)] for l in range(q)] for m in range(p)])
            #self.conv.weight = torch.nn.parameter.Parameter(w)
            torch.nn.init.constant_(self.conv.weight, 1/10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
        #return F.relu(x, inplace=True)
        #return F.relu(x, inplace=False)

class InceptionE3(nn.Module):

    def __init__(self, in_channels: int, repetable=0) -> None:
        super().__init__() # python 3 syntax
        self.branch1x1 = BasicConv2d(in_channels, 320, repetable, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, repetable, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, repetable, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384,repetable, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, repetable, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, repetable, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, repetable, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, repetable, kernel_size=(3, 1), padding=(1, 0))

        self.avg_pool_2d = _avgPool()
        # start /defined by chirag/
        self.branch_pool = BasicConv2d(in_channels, 192,repetable, kernel_size=1)
        
        self.concatenateFinal5 = _concatenateLayer()
        self.concatenateFinal6 = _concatenateLayer()
        self.concatenateFinal7 = _concatenateLayer()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.squeeze1 = _squeezeLayer()
        self.squeeze2 = _squeezeLayer()
        # end /defined by chirag/
    def forward(self, x):
        x = self.squeeze1(x)
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)

        branch3x3_2a_out = self.branch3x3_2a(branch3x3)
        branch3x3_2b_out = self.branch3x3_2b(branch3x3)

        # start /modified by chirag/
        #print(branch3x3_2a_out.size())
        branch3x3 = self.concatenateFinal5(branch3x3_2a_out,branch3x3_2b_out)
        # end /modified by chirag/

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl_3a_out = self.branch3x3dbl_3a(branch3x3dbl)
        branch3x3dbl_3b_out = self.branch3x3dbl_3b(branch3x3dbl)

        # start /modified by chirag/
        branch3x3dbl = self.concatenateFinal6(branch3x3dbl_3a_out, branch3x3dbl_3b_out)
        # end /modified by chirag/

        branch_pool_out = self.avg_pool_2d(x)
        branch_pool_out = self.branch_pool(branch_pool_out)

        #outputs = self._forward(x)
        # start /modified by chirag/
        outputs = self.concatenateFinal7(branch1x1, branch3x3, branch3x3dbl, branch_pool_out)
        outputs = self.avgpool(outputs)
        outputs = self.squeeze2(outputs)
        #print(outputs.size())
        return outputs
        # end /modified by chirag/

class InceptionE2(nn.Module):

    def __init__(self, in_channels: int, repetable=0) -> None:
        super().__init__() # python 3 syntax
        self.branch1x1 = BasicConv2d(in_channels, 320,repetable, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384,repetable, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, repetable,kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, repetable,kernel_size=(3, 1), padding=(1, 0))

        self.avg_pool_2d = _avgPool()
        # start /defined by chirag/
        self.branch_pool = BasicConv2d(in_channels, 192, repetable, kernel_size=1)
        
        self.concatenateFinal5 = _concatenateLayer()

        self.concatenateFinal7 = _concatenateLayer()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.squeeze1 = _squeezeLayer()
        self.squeeze2 = _squeezeLayer()
        # end /defined by chirag/
    def forward(self, x):
        x = self.squeeze1(x)
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)

        branch3x3_2a_out = self.branch3x3_2a(branch3x3)
        branch3x3_2b_out = self.branch3x3_2b(branch3x3)

        # start /modified by chirag/
        #print(branch3x3_2a_out.size())
        branch3x3 = self.concatenateFinal5(branch3x3_2a_out,branch3x3_2b_out)
        # end /modified by chirag/


        branch_pool_out = self.avg_pool_2d(x)
        branch_pool_out = self.branch_pool(branch_pool_out)

        #outputs = self._forward(x)
        # start /modified by chirag/
        outputs = self.concatenateFinal7(branch1x1, branch3x3, branch_pool_out)
        outputs = self.avgpool(outputs)
        outputs = self.squeeze2(outputs)
        #print(outputs.size())
        return outputs
        # end /modified by chirag/

class ShortInceptionE(nn.Module):

    def __init__(self, in_channels: int, repetable=0) -> None:
        super().__init__() # python 3 syntax
        self.squeeze1 = _squeezeLayer()

        self.branch11 = nn.Conv2d(in_channels, 320,bias=False, kernel_size=1)
        self.branch12 = nn.Conv2d(320, 640, kernel_size=3,bias=False, stride=3)

        self.branch21 = nn.Conv2d(in_channels, 320,bias=False, kernel_size=1)
        self.branch22 = nn.Conv2d(320, 320,bias=False, kernel_size=3, stride=3)

        self.concatenate = _concatenateLayer()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.squeeze2 = _squeezeLayer()

        self.fc1 = nn.Linear(640+320, 512)
        self.relu1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU(inplace=True)

        if repetable:
            torch.nn.init.constant_(self.fc1.weight, 1/960); torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.constant_(self.fc2.weight, 1/512); torch.nn.init.zeros_(self.fc2.bias)
            torch.nn.init.constant_(self.branch11.weight, 1/10)
            torch.nn.init.constant_(self.branch12.weight, 1/10)
            torch.nn.init.constant_(self.branch21.weight, 1/10)
            torch.nn.init.constant_(self.branch22.weight, 1/10)

    def forward(self, x):
        x = self.squeeze1(x)
        #print(x.size());print()

        branch11_out = self.branch11(x)
        #print(branch11_out.size())
        branch12_out = self.branch12(branch11_out)
        #print(branch12_out.size());print()
        

        branch21_out = self.branch21(x)
        branch22_out = self.branch22(branch21_out)

        con_out = self.concatenate(branch12_out, branch22_out)
        #print(con_out.size());print()
        pool_out = self.avgpool(con_out)
        #print(pool_out.size());print()

        outputs = self.squeeze2(pool_out)
        #print(outputs.size())

        outputs = self.fc1(outputs)
        outputs = self.relu1(outputs)
        #print(outputs.size())

        outputs = self.fc2(outputs)
        outputs = self.relu2(outputs)
        #print(outputs.size())

        return outputs


class InceptionA(nn.Module):

    def __init__(
        self,
        in_channels: int,
        pool_features: int,
        repetable:int
    ) -> None:
        super(InceptionA, self).__init__()
        
        conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64,repetable, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48,repetable, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, repetable, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64,repetable, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96,repetable, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96,repetable, kernel_size=3, padding=1)

        self.avgpool = _avgPool()
        self.branch_pool = conv_block(in_channels, pool_features,repetable, kernel_size=1)
        # start /defined by chirag/
        self.concatenateFinal = _concatenateLayer()
        # end /defined by chirag/

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.avgpool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        # print("-"*10)
        # print("inside inceptionA:")
        # print(branch1x1[0][0])
        # print(branch5x5[0][0])
        # print(branch3x3dbl[0][0])
        # print(branch_pool[0][0])
        # print("-"*10)
        return outputs

    def forward(self,x):
        outputs = self._forward(x)
        # start /modified by chirag/
        outputs = self.concatenateFinal(*outputs)
        return outputs
        # end /modified by chirag/
class InceptionB(nn.Module):

    def __init__(
        self,
        in_channels: int
    ) -> None:
        super(InceptionB, self).__init__()
        conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)
        # start /defined by chirag/
        self.maxpool = _maxPool()
        self.concatenateFinal2 = _concatenateLayer()
        # end /defined by chirag/

    def _forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.maxpool(x)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        # start /modified by chirag/
        outputs = self.concatenateFinal2(*outputs)
        return outputs
        # end /modified by chirag/


class InceptionC(nn.Module):

    def __init__(
        self,
        in_channels: int,
        channels_7x7: int
    ) -> None:
        super(InceptionC, self).__init__()

        conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        
        # start /defined by chirag/
        self.avgpool = _avgPool()
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)
        
        self.concatenateFinal3 = _concatenateLayer()
        # end /defined by chirag/

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = self.avgpool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        # start /modified by chirag/
        outputs = self.concatenateFinal3(*outputs)
        return outputs
        # end /modified by chirag/


class InceptionD(nn.Module):

    def __init__(
        self,
        in_channels: int
    ) -> None:
        super(InceptionD, self).__init__()

        conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)
        # start /defined by chirag/
        self.maxpool = _maxPool()
        self.concatenateFinal4 = _concatenateLayer()
        # end /defined by chirag/

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = self.maxpool(x)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        # start /modified by chirag/
        outputs = self.concatenateFinal4(*outputs)
        return outputs
        # end /modified by chirag/


class InceptionE(nn.Module):

    def __init__(
        self,
        in_channels: int
    ) -> None:
        super(InceptionE, self).__init__()
        conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.avgpool = _avgPool()
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)
        # start /defined by chirag/
        self.concatenateFinal5 = _concatenateLayer()
        self.concatenateFinal6 = _concatenateLayer()
        self.concatenateFinal7 = _concatenateLayer()
        # end /defined by chirag/

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        # start /modified by chirag/
        branch3x3 = self.concatenateFinal5(*branch3x3)
        # end /modified by chirag/


        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        # start /modified by chirag/
        branch3x3dbl = self.concatenateFinal6(*branch3x3dbl)
        # end /modified by chirag/

        branch_pool = self.avgpool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        # start /modified by chirag/
        outputs = self.concatenateFinal7(*outputs)
        return outputs
        # end /modified by chirag/

#xyz
class Inception3(nn.Module):

    def __init__(
        self,
        num_classes: int = 1000,
        repetable: int = 1,
    ) -> None:
        super(Inception3, self).__init__()
        conv_block = BasicConv2d
        inception_a = InceptionA
        inception_b = InceptionB
        inception_c = InceptionC

        self.Conv2d_1a_3x3 = conv_block(3, 32, repetable, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, repetable, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, repetable, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, repetable, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192,repetable, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(192, pool_features=32, repetable=repetable)
        self.Mixed_5c = inception_a(256, pool_features=64, repetable=repetable)
        self.Mixed_5d = inception_a(288, pool_features=64, repetable=repetable)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.dropout = nn.Dropout()
        self.flatten = _flatten()
        self.fc = nn.Linear(768, num_classes)
        #elf.addLayer = _addLayer()

        if repetable:
            torch.nn.init.constant_(self.fc.weight, 1/288); torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # N x 3 x 299 x 299
        #--$print("0");print(x[0][0]);print("#"*20)
        x = self.Conv2d_1a_3x3(x)
        #print("1");print(x[0][0]);print("#"*20)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        #print("2");print(x[0][0]);print("#"*20)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        #print("3");print(x[0][0]);print("#"*20)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        #print("4");print(x[0][0]);print("#"*20)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        #--$print("5a");print(x[0][0]);print("#"*20)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        #--$print("5b");print(x[0][0]);print("#"*20)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        #--$print("5c");print(x[0][0]);print("#"*20)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        ######-print("5d");print(x[0][0]);print("#"*20)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        #print("6a");print(x[0][0]);print("#"*20)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        #print("6b");print(x[0][0]);print("#"*20)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        #print("6c");print(x[0][0]);print("#"*20)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        #print("6d");print(x[0][0]);print("#"*20)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        #print("6e");print(x[0][0]);print("#"*20)
        # N x 768 x 17 x 17
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 768 x 1 x 1
        x = self.flatten(x)
        # N x 768
        x = self.fc(x)
        # N x 1000 (num_classes)
        #--$print("final");print(x[0][0]);print("#"*20)
        return x
        #return x

#####################################################################

#xyz
class Inception3_test(nn.Module):

    def __init__(
        self,
        num_classes: int = 1000,
        repetable: int = 1,
    ) -> None:
        super(Inception3, self).__init__()
        conv_block = BasicConv2d
        inception_a = InceptionA
        inception_b = InceptionB
        inception_c = InceptionC

        self.Conv2d_1a_3x3 = conv_block(3, 32, repetable, kernel_size=3, stride=2)
        #~self.Conv2d_2a_3x3 = conv_block(32, 32, repetable, kernel_size=3)
        #~self.Conv2d_2b_3x3 = conv_block(32, 64, repetable, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        #~self.Conv2d_3b_1x1 = conv_block(64, 80, repetable, kernel_size=1)
        #~self.Conv2d_4a_3x3 = conv_block(80, 192,repetable, kernel_size=3)
        #~self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        #~self.Mixed_5b = inception_a(192, pool_features=32, repetable=repetable)
        #~self.Mixed_5c = inception_a(256, pool_features=64, repetable=repetable)
        #~self.Mixed_5d = inception_a(288, pool_features=64, repetable=repetable)
        #~self.Mixed_6a = inception_b(288)
        #~self.Mixed_6b = inception_c(768, channels_7x7=128)
        #~self.Mixed_6c = inception_c(768, channels_7x7=160)
        #~self.Mixed_6d = inception_c(768, channels_7x7=160)
        #~self.Mixed_6e = inception_c(768, channels_7x7=192)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.dropout = nn.Dropout()
        self.flatten = _flatten()
        self.fc = nn.Linear(32, num_classes)
        #elf.addLayer = _addLayer()

        if repetable:
            torch.nn.init.constant_(self.fc.weight, 1/288); torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # N x 3 x 299 x 299
        #--$print("0");print(x[0][0]);print("#"*20)
        x = self.Conv2d_1a_3x3(x)
        #print("1");print(x[0][0]);print("#"*20)
        # N x 32 x 149 x 149
        #~x = self.Conv2d_2a_3x3(x)
        #print("2");print(x[0][0]);print("#"*20)
        # N x 32 x 147 x 147
        #~x = self.Conv2d_2b_3x3(x)
        #print("3");print(x[0][0]);print("#"*20)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        #print("4");print(x[0][0]);print("#"*20)
        # N x 64 x 73 x 73
        #~x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        #~x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        #~x = self.maxpool2(x)
        #--$print("5a");print(x[0][0]);print("#"*20)
        # N x 192 x 35 x 35
        #~x = self.Mixed_5b(x)
        #--$print("5b");print(x[0][0]);print("#"*20)
        # N x 256 x 35 x 35
        #~x = self.Mixed_5c(x)
        #--$print("5c");print(x[0][0]);print("#"*20)
        # N x 288 x 35 x 35
        #~x = self.Mixed_5d(x)
        ######-print("5d");print(x[0][0]);print("#"*20)
        # N x 288 x 35 x 35
        #~x = self.Mixed_6a(x)
        #print("6a");print(x[0][0]);print("#"*20)
        # N x 768 x 17 x 17
        #~x = self.Mixed_6b(x)
        #print("6b");print(x[0][0]);print("#"*20)
        # N x 768 x 17 x 17
        #~x = self.Mixed_6c(x)
        #print("6c");print(x[0][0]);print("#"*20)
        # N x 768 x 17 x 17
        #~x = self.Mixed_6d(x)
        #print("6d");print(x[0][0]);print("#"*20)
        # N x 768 x 17 x 17
        #~x = self.Mixed_6e(x)
        #print("6e");print(x[0][0]);print("#"*20)
        # N x 768 x 17 x 17
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 768 x 1 x 1
        x = self.flatten(x)
        # N x 768
        x = self.fc(x)
        # N x 1000 (num_classes)
        #--$print("final");print(x[0][0]);print("#"*20)
        return x
        #return x

#####################################################################


def parallelThreeLayer(factor, repetable=0) -> ParallelThreeLayer:
    model = ParallelThreeLayer(factor, repetable)
    return model

def parallelTwoLayer(factor, repetable=0) -> ParallelTwoLayer:
    model = ParallelTwoLayer(factor, repetable)
    return model

def parallelThreeLayerOld(factor) -> ParallelThreeLayerOld:
    model = ParallelThreeLayerOld(factor)
    return model

def tallParallelModel(factor, repetable=0) -> TallParallelModel:
    model = TallParallelModel(factor, repetable)
    return model

def shortLinearModel(factor, repetable=0) -> ShortLinearModel:
    model = ShortLinearModel(factor, repetable)
    return model

def linearModel(factor, repetable=0) -> LinearModel:
    model = LinearModel(factor, repetable)
    return model

def inceptionE3(in_channels, repetable=0) -> InceptionE3:
    model = InceptionE3(in_channels, repetable)
    return model

def inceptionE2(in_channels, repetable=0) -> InceptionE2:
    model = InceptionE2(in_channels, repetable)
    return model

def shortInceptionE(in_channels, repetable=0) -> ShortInceptionE:
    model = ShortInceptionE(in_channels, repetable)
    return model

def oneLayer(factor, repetable=0) -> OneLayer:
    model = OneLayer(factor, repetable)
    return model

def inception3(num_classes=1000, repetable=0) -> Inception3:
    model = Inception3(num_classes, repetable)
    return model