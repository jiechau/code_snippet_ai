


import sys
print('python: ' + str(sys.version)) # will print out python version

import torch
print('pytorch: ' + str(torch.__version__))
import torchvision
print('torchvision: ' + str(torchvision.__version__))

print('cuda: ' + str(torch.cuda.is_available()))

t = torch.randn(3, 3)
#t = torch.rand(3, 3)
if torch.cuda.is_available():
    t = t.to('cuda')
print(t.is_cuda)
print(t.device)


