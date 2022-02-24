import torch

a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)

c = 2*a**2 + 3*b**2

d = torch.tensor([4.0], requires_grad=True)

e = c+d

e.backward()

print(a.grad)       #8
print(b.grad)       #18

print(c.grad)       #none



