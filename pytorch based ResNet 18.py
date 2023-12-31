#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available else 'cpu'


# In[2]:


batch_size = 100
learning_rate = 0.01
epochs = 20


# In[3]:


import ssl

# SSL 인증서 검증 비활성화
ssl._create_default_https_context = ssl._create_unverified_context

#download 
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_set = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, drop_last = True)

test_set = torchvision.datasets.CIFAR10(root = '/data', train = False, download = True, transform = transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False, drop_last = True)


# In[31]:


class myResnet(nn.Module):
    def __init__(self):
        super(myResnet, self).__init__()

        self.block0 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        self.blockA = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        
        self.blockB = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        self.blockC = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.blockD = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.blockE = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.blockF = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        self.blockG = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.MaxPool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        
        self.fc = nn.Linear(512, 10)
        
        #skip connections
        self.conv1X1_a = nn.Conv2d(16, 32, kernel_size = 1, stride = 1)
        self.conv1X1_b = nn.Conv2d(32, 64, kernel_size = 1, stride = 1)
        self.conv1X1_c = nn.Conv2d(64, 128, kernel_size = 1, stride = 1)
        self.conv1X1_d = nn.Conv2d(128, 256, kernel_size = 1, stride = 1)
        self.conv1X1_e = nn.Conv2d(256, 512, kernel_size = 1, stride = 1)
        
    
    def forward(self, x):
        #layer 1
        out = self.blockA(x)
        
        #layer2
        skip = self.conv1X1_a(out)
        out = self.blockB(out)
        out = out + skip
        out = self.MaxPool(out)
        
        #layer3
        skip = self.conv1X1_b(out)
        out = self.blockC(out)
        out = out + skip
        out = self.MaxPool(out)

        #layer 4
        skip = self.conv1X1_c(out)
        out = self.blockD(out)
        out = out + skip
        out = self.MaxPool(out)

        #layer 5
        skip = self.conv1X1_d(out)
        out = self.blockE(out)
        out = out + skip

        #layer 6
        skip = self.conv1X1_e(out)
        out = self.blockF(out)
        out = out + skip
        
        #fc layer
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


# In[32]:


model = myResnet()
model.to(device)


# In[33]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


# In[34]:


#learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)


# In[35]:


#train & evaluate
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        #move data to device
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

      
        running_loss += loss.item()
        if i % 100 == 99:  
            print('[%3d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
        
    scheduler.step()


# In[36]:


correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy : %d %%' % (100 * correct / total))


# In[ ]:




