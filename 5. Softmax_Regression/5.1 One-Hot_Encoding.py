import torch
import torch.nn.functional as F 

z = torch.rand(3,5, requires_grad=True)

hypothesis =F.softmax(z, dim=1)
print(hypothesis)

y = torch.tensor([0,2,1])   # (3,)

y.unsqueeze_(1)             # unsqueeze(여기 적히는 idx에 차원 추가)
                            # (3,1)

one_hot = torch.zeros_like(hypothesis)
one_hot.scatter_(1,y,1)
# scatter 뒤에 _ -> inplace = True
# scatter 첫번째 dim, 두번째가 가르키는 idx에 1을 뿌려라

print(one_hot)