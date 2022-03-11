import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.x_data = torch.FloatTensor([[1,2],
                                         [2,3],
                                         [3,1],
                                         [4,3],
                                         [5,3],
                                         [6,2]])
        self.y_data = torch.FloatTensor([[0],
                                         [0],
                                         [0],
                                         [1],
                                         [1],
                                         [1]])
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self,idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x,y
    
class BinaryClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        return self.sigmoid(self.linear(x))
    
dataset = CustomDataset()
dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=2)

model = BinaryClassifier()
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epoch = 1000

for epoch in range(nb_epoch+1):
    for batch_idx, sample in enumerate(dataloader):
        x_train, y_train = sample
        
        hypothesis = model(x_train)
        
        # binary_cross_entropy는 input과 target 두 개의 인자가 있다.
        # input : 모델에서 나온 값으로 0과 1로 "바뀔" 값
        # target : 0 과 1
        # 순서 중요함
        cost = F.binary_cross_entropy(hypothesis,y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        
        if epoch %200 == 0:
            # 0~1 사이 값을 가진 (6,1) hypothesis
           # 0.5 보다 크면 True, 작으면 False 
            prediction = hypothesis >= torch.FloatTensor([0.5])
            
            # True,Flase 등의 bool에 .float()를 먹이면 1, 0 으로 변함
            # y_train 과 같은지 비교해서 같으면 True 아니면 False
            correct_prediction = prediction.float() == y_train
            
            # 현재 correct_prediction은 bool인데,
            # .sum() : 1,0으로 인식해서 합을 tensor로 내준다
            # .item() : tensor2int pytorch, only one element
            accuracy = correct_prediction.sum().item()/len(correct_prediction)
            
            print(f'epoch : {epoch}/{nb_epoch}')
            print(f'batch idx : {batch_idx+1}/{len(dataloader)}')
            print(f'cost : {cost:.2f}')
            print(f'accurcy : {accuracy*100}')