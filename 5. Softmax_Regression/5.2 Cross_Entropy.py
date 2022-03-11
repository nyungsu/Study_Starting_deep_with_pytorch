import torch
import torch.nn.functional as F 

torch.manual_seed(1)


z = torch.rand(3,5, requires_grad=True)

hypothesis = F.softmax(z,dim=1)        # softmax 차원 니가 아는 차원
print(hypothesis)

y = torch.randint(5,(3,)).long()
print(y)

y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1,y.unsqueeze(1),1)
# zeors_like 함수로 크기는 같고 0으로 채워진 y_one_hot
# unsqueeze(1) : (3,) -> (3,1) 차원 확장 
# tensor [0,2,1] -> tensor [[0],
#                           [2],
#                           [1]]
# scatter 뒤에 _ -> inplace = True
# scatter 첫번째 dim, 두번째가 가르키는 idx에 1을 뿌려라

print(y_one_hot)
'''
다 똑같은 거임
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()


cost = (y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()

cost = (y_one_hot * -F.log_softmax(z, dim=1)).sum(dim=1).mean()

cost = F.nll_loss(F.log_softmax(z, dim=1),y)

'''
cost = F.cross_entropy(z,y)
print(cost)