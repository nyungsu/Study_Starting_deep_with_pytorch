import torch.nn as nn 
import torch.nn.functional as F

VGG_type = {
        'VGG11' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512,512,512, 'M',512,512,512,'M'],
        'VGG19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512,512,512,512, 'M',512,512,512,512,'M'],
        'custom' : [64, 64 ,64, 'M', 128, 128, 128, 'M']
    }

class VGGNet(nn.Module):
    def __init__(self,feature,num_classes=10, init_weights=True) -> None:
        super().__init__()
        self.feature = self.make_layers(VGG_type[feature])  # Convolution
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(),
            
            nn.Linear(4096, num_classes)
        )# FC layer
        
        if init_weights:
            self._initiallize_weights()
            
    def forward(self, x):
        x = self.feature(x)         # Convolution
        x = self.avgpool(x)         # avgpool
        x = self.view(x.size(0), -1)# Fc layer
        x = self.classifier(x)
        
        return x
    
    def _initiallize_weights(self):
        for m in self.modules(): # feature 에 모듈들을 하나씩 m에
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else :
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v     # 다음 channel을 위해 channel 수 바꿔주는 거 중요
     
        return nn.Sequential(*layers)

    '''
    # 'VGG11' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M']가
    # make_layers 함수에 들어가면
    # 아래와 같은 레이어가 생성된다.
    이 쉐애끼들 for문으로 레이어 쌓을 생각하다니 똑똑한데 ?
    
    1) 64
        conv2d = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        nn.ReLU(inplace=True)
    2) 'M'
        nn.MaxPool2d(kernel_size=2, stirde=2)
    3) 128
        conv2d = nn.Conv2d(64,128, kernel_size=3, padding=1)
        nn.ReLU(inplace=True)
    4) 'M'
        nn.MaxPool2d(kernel_size=2, stirde=2)
    5) 256
        conv2d = nn.Conv2d(128,256, kernel_size=3, padding=1)
        nn.ReLU(inplace=True)
    6) 256
        conv2d = nn.Conv2d(256,256, kernel_size=3, padding=1)
        nn.ReLU(inplace=True)
    7) 'M'
        nn.MaxPool2d(kernel_size=2, stirde=2)
    8) 512
        conv2d = nn.Conv2d(256,512, kernel_size=3, padding=1)
        nn.ReLU(inplace=True)
    9) 512
        conv2d = nn.Conv2d(512,512, kernel_size=3, padding=1)
        nn.ReLU(inplace=True)
    10) 'M'
        nn.MaxPool2d(kernel_size=2, stirde=2)
        
        '''
        

    
model = VGGNet('VGG16',num_classes=10,init_weights=True)
print(model)