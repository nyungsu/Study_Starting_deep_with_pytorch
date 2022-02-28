import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1,2],
          [2,3],
          [3,1],
          [4,3],
          [5,3],
          [6,2]]

y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# nn.sequential은 여러 함수들을 연결해주는 역할을 함
model = nn.Sequential(
    nn.Linear(2,1),
    nn.Sigmoid()    
)

optimizer = optim.SGD(model.parameters(), lr=1)

nb_epoch = 1000

for epoch in range(nb_epoch+1):
    hypothesis = model(x_train)
    
    # binary_cross_entropy는 input과 target 두 개의 인자가 있다.
    # input : 모델에서 나온 값으로 0과 1로 "바뀔" 값
    # target : 0 과 1
    # 순서 중요함
    cost = F.binary_cross_entropy(input=hypothesis, target=y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch %100 ==0:
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
        
        # print(f'epoch : {epoch}/{nb_epoch}')
        # print(f'cost : {cost:.2f}')
        # print(f'accuracy : {accuracy*100}')
        # print()
        

print(model(x_train))
print(list(model.parameters()))
