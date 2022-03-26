import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stirde=stride, padding=1, bias=False)
    
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)
    
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        
        self.conv2 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)     # stride = stride
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)   # stirde = 1
        out = self.bn2(out)
        
        if self.downsample is not None:     # stride = 2 일 때
            identity = self.downsample(x)   # 
            
        out += identity
        out = self.relu(out)
        
        return out
    
class BottleNeck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None) -> None:
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(inplanes, planes*self.expansion)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)     # stride =1
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)   # stride = stride
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)   # stride = 1
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 zero_init_residual=False) -> None:
        super().__init__() 
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stirde=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stirde=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stirde=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stirde=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias, 0)
                
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes !=planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion,
                        stride),
                nn.BatchNorm2d(planes * block.expansion)
            )          
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    
    


