import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

# cuda or cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'     
torch.manual_seed(777)
if torch == 'cuda':
    torch.cuda.manual_seed_all(777)

# 하이퍼 파라미터 
BATCH_SIZE = 100
LEARNING_RATE = 0.01
TRAINING_EPOCHS = 15                                        

# train, test 데이터 불러와서 dataloader에 연결    
train_data = dsets.MNIST(root='data/',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)

test_data = dsets.MNIST(root='data/',
                        train=False,
                        transform=transforms.ToTensor(),
                        download=True)

loader_train = DataLoader(dataset=train_data,
                          shuffle=True,
                          batch_size=BATCH_SIZE,
                          drop_last=True)

loader_test = DataLoader(dataset=test_data,
                         shuffle=True,
                         batch_size=BATCH_SIZE,
                         drop_last=True)                     

x_test_origin = test_data.data
y_test_origin = test_data.test_labels

# 모델 구현
class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(7*7*64,10, bias=True)
    
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        
        return out

# 모델 정의
model = CNN()

# optimizer 정의
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)        
total_batch = len(loader_train)

def train(epoch):
    for epoch in range(TRAINING_EPOCHS+1):
        avg_cost = 0
        for batch_idx, sample in enumerate(loader_train):
            x_train, y_train = sample     # x_train = (100,1,28,28)
                                          # y_train = (100)
            hypothesis = model(x_train)
            
            cost = F.cross_entropy(hypothesis, y_train)
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            avg_cost += cost/total_batch
            if batch_idx % 100 ==0:
                print(f'epoch : {epoch}/{TRAINING_EPOCHS}')
                print(f'batch idx : {batch_idx+1}/{total_batch}')
                print(f'cost : {cost:.2f}')
def test():
    with torch.no_grad():
        model.eval()
        
        x_test = x_test_origin
        y_test = y_test_origin
        
        prediction = model(x_test)
        correct_prediction = torch.argmax(prediction, 1)==y_test
        accuracy = correct_prediction.float().mean()
        
        print(f'accuracy : {accuracy*100}')
        
        
train(1)

