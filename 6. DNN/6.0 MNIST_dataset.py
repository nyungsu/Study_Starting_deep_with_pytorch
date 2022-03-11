import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

mnist_train = dsets.MNIST(root='data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

x_train_orig = mnist_train.data
y_train_orig = mnist_train.train_labels

x_test_orig = mnist_test.data
y_test_orig = mnist_test.test_labels

print(f'x_train shape : {x_train_orig.shape}')  # [60000,28,28]
print(f'y_train shape : {y_train_orig.shape}')  # [60000]

print(f'x_test shape : {x_test_orig.shape}')    # [10000,28,28]
print(f'y_test shape : {y_test_orig.shape}')    # [10000]

dataloader = DataLoader(dataset=mnist_train,
                        batch_size=100,
                        shuffle=True,
                        drop_last=True)

# 배치 크기가 100이기 때문에
# for x,y in dataloader: 로 받으면
# x : [100,1,28,28] : batch size 갯수 만큼의 1x28x28의 텐서
# y : [100] : batch size 갯수 만큼의 레이블

# for x,y in dataloader:
#     print(x.shape)
#     print('----------------------')
#     print(x[0])
#     print('----------------------')
#     print(y.shape)
#     print('----------------------')
#     print(y)
#     break

x_train_show = x_train_orig[0].numpy()
print(f'label : {y_train_orig[0]}')
plt.imshow(x_train_show.reshape(28,28),cmap='gray')
plt.show()

