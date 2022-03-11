import torch
import torch.nn as nn 

inputs = torch.Tensor(1,1,28,28)
print(inputs.shape)             # (1,1,28,28)
'''
torch.Tensor는 tensor 자료구조의 클래스
이것의 경우 데이터를 입력 해주지 않아도 빈 tensor가 저장 됨
(batch_size, chanel, w, d)

torch.tensor는 어떤 data를 tensor로 copy 해주는 함수
데이터를 입력해주지 않으면 복사할 데이터가 없으므로 에러가 난다

torch.size(mini-batch size, channel size, img size, img size)
'''



conv1 = nn.Conv2d(1,5,5)        # input chanel : 1
                                # output chanel : 5
                                # kenel size : 5
                                
pool = nn.MaxPool2d(2)          # 2x2 사이즈로 맥스 풀링


out1 = conv1(inputs)
print(out1.shape)               # (1, 5, 24, 24)

out2 = pool(out1)
print(out2.shape)               # (1, 5, 12, 12)

'''
size 계산법 = (input size - kenel size -2*padding)
              -----------------------------------  + 1
                            stride
'''
