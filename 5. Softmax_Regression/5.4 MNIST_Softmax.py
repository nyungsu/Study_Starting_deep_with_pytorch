import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import random

USE_CUDA = torch.cuda.is_available()
# GPU 사용 가능하면 True, d아니면 False 리턴

device = torch.device("cuda" if USE_CUDA else "cpu")

print('다음 기기로 학습합니다. : ', device)

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
    
nb_epochs = 15
batch_size = 100

mnist_train = dsets.MNIST(root='data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          # 데이터를 파이토치 tensor로 변경
                          download=True)

mnist_test  = dsets.MNIST(root='data/',
                          train=False,
                          transform=transforms.ToTensor(),
                          # 데이터를 파이토치 tensor로 변경
                          download=True)

dataloader = DataLoader(dataset=mnist_train,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True) # 마지막 배치를 버릴 것
'''
drop_last의 이해
10개의 데이터, batch_size=3 일 때 마지막 1개를 어떻게 할지
'''

model = nn.Linear(784,10,bias=True).to(device)
# input : mnist(28x28) 펴서 784개
# output : 0~9까지 10개 softmax

optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(nb_epochs+1):
    avg_cost = 0
    total_batch = len(dataloader)
    
    for batch_idx, sample in enumerate(dataloader):
        x_train, y_train = sample
        
        x_train = x_train.view(-1,28*28). to(device)
        y_train = y_train.to(device)
        
        hypothesis = model(x_train)
        cost = F.cross_entropy(hypothesis,y_train).to(device)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        avg_cost += cost/total_batch
    
    print(f'epcoh : {epoch}/{nb_epochs}')
    print(f'cost : {avg_cost:.2f}')

print('leaning finish')
        
# ----------------------------------------------------

with torch.no_grad():
    #torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
    
    x_test = mnist_test.test_data.view(-1,28*28).float().to(device)
    y_test = mnist_test.test_labels.to(device)
    
    prediction = model(x_test)
    correct_prediciton = torch.argmax(prediction,1) == y_test
    accuracy = correct_prediciton.float().mean()
    print(f'accracy : {accuracy.item()}')
    
    # MNIST 테스트 데이터 중에 무작위로 하나 뽑아서 예측
    r = random.randint(0,len(mnist_test)-1)
    x_single_data = mnist_test.test_data[r:r+1].view(-1,28*28).float().to(device)
    y_single_data = mnist_test.test_labels[r:r+1].to(device)
    
    print(f'label : {y_single_data.item()}')
    single_prediction = model(x_single_data)
    print(f'prediction : {torch.argmax(single_prediction,1).item()}')
    
    plt.imshow(mnist_test.test_data[r:r+1].view(28,28),cmap='Greys',interpolation='nearest')
    plt.show()