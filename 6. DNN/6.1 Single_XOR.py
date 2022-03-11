'''
XOR을 모델로 구현해도 일정 수치 밑으로 cost를 줄이지 못 함
'''


import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
    
x_train = torch.FloatTensor([[0,0],
                       [0,1],
                       [1,0],
                       [1,1]]).to(device)

y_train = torch.FloatTensor([[0],
                       [1],
                       [1],
                       [0]]).to(device)

model = nn.Sequential(
    nn.Linear(2,1),
    nn.Sigmoid()
).to(device)

optimizer = optim.SGD(model.parameters(),lr = 1)

nb_epochs = 1000

for epoch in range(nb_epochs+1):
    hypothesis = model(x_train)
    
    cost = F.binary_cross_entropy(hypothesis,y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    
    if epoch % 100 == 0:
        print(f'cost : {cost:.2f}')
        
with torch.no_grad():
    hypothesis = model(x_train)
    prediction = (hypothesis > 0.5).float()
    accuracy = (prediction == y_train).float().mean()
    
    print(f'모델의 출력 값 : {hypothesis}')
    print(f'실제값 : {y_train}')
    print(f'정확도 : {accuracy}')