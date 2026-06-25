
#
# https://pytorch.org/get-started/locally/#anaconda

# conda create -n py39pt210; conda activate py39pt210
# # all ok
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm 

# Define model 
class build_model(nn.Module):
    def __init__(self):
        super(build_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Load data
train_loader = torch.utils.data.DataLoader(
        #datasets.MNIST('C:/share/tmp/data', train=True, download=True, 
        datasets.MNIST('/tmp/tmp/data', train=True, download=True, # ok default to 'C:/tmp/data'
                 transform=transforms.Compose([
                   transforms.ToTensor(), 
                   transforms.Normalize((0.1307,), (0.3081,))
                  ])),
  batch_size=64, shuffle=True)

print('train_loader', len(train_loader))

# Initialize model and optimizer
#model = build_model()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('***device', device)
# Move model to device 
model = build_model().to(device)

# check cuDNN
if torch.cuda.is_available():
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True
print('**cuDNN', torch.backends.cudnn.enabled)
print('**deterministic', torch.backends.cudnn.deterministic)
print('**benchmark', torch.backends.cudnn.benchmark)

'''
from thop import profile
from thop import clever_format
# Define dummy input with the same shape as your actual input
dummy_input = torch.randn(1, 1, 28, 28, device=device)  # Adjust the shape as per your input size
# Use thop to profile the model
macs, params = profile(model, inputs=(dummy_input,))
flops = clever_format(macs, "%.3f")
print(f"FLOPS: {flops}")
import sys
sys.exit()
'''

# training

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train_step(_model, _data, _target):
  optimizer.zero_grad()
  output = _model(_data)
  _loss = F.nll_loss(output, _target)
  _loss.backward()
  optimizer.step()
  return _loss



# Measure time
iterations = 100
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
starter.record()


import time
start = time.time()

# Train model
for epoch in range(2):
  enu_train_loader = enumerate(train_loader)
  for batch_idx in tqdm(range(len(train_loader)), desc ="Step"):
  #for batch_idx, (data, target) in enumerate(train_loader):
    _, (data, target) = next(enu_train_loader)

    # Move tensors to device
    if torch.cuda.is_available():
      data, target = data.to(device), target.to(device)

    # call training fun
    loss = train_step(model, data, target)

  print('Epoch:', epoch, ', Loss:', loss.item())

end = time.time()
print(end - start)


ender.record()
torch.cuda.synchronize()
elapsed_time = starter.elapsed_time(ender) / 1000
print(elapsed_time)


from thop import profile
from thop import clever_format
# Define dummy input with the same shape as your actual input
dummy_input = torch.randn(1, 1, 28, 28, device=device)  # Adjust the shape as per your input size
# Use thop to profile the model
#macs, params = profile(model, inputs=(dummy_input,))
macs, params = profile(model, inputs=(data,))
flops = clever_format(macs, "%.3f")
print(f"FLOPS: {flops}")
print(f"Average FLOPs/s: {macs / elapsed_time * 2 / 1e9:.3f} G")
import sys
sys.exit()


# for consistence. move to cpu format
model.cpu()
# Save model
#torch.save(model.state_dict(), 'C:/share/tmp/pt_model.pt')
torch.save(model.state_dict(), '/tmp/tmp/pt_model.pt') # ok default to 'C:/tmp/pt_model.pt'
