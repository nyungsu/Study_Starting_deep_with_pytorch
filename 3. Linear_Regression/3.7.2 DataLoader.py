import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader





class Dataset(torch.utils.data.Dataset):
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
    
dataset = Dataset()
dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=2)

# 총 6개의 데이터 셑
print(list(dataloader))
print()

# batch size=2 라서 3팀으로 쪼개짐
dataset1, dataset2, dataset3  = dataloader
print(dataset1)
print()

# 쪼개진 데이터 셋에서도 x_trian, y_train으로 나눠짐
x_train, y_train = dataset1
print(x_train)
print()

# 3팀으로 쪼개지고 팀당 2개씩 쪼개지니까
# 처음부터 6개로 쪼갤 수는 없나 확인해봤는데
# not enough values to unpack 이라는 오류가 뜸

# 처음부터 튜플로 받으면 언팩킹이 됨
(x_train1, y_train) ,(x_train2, y_train2), (x_train3, y_train3) = dataloader
print(x_train1)

